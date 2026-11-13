"""Nonlinear regression (curve fitting): model library, fitter, model comparison.

Method
------
Parameters are estimated by (optionally weighted) least squares with
:func:`scipy.optimize.least_squares` (trust-region-reflective, plus MINPACK
Levenberg-Marquardt when there are no bounds) using analytic Jacobians: closed-form for
the built-in models, forward-mode automatic differentiation for user-defined equations.
Starting values of the built-in models are found automatically by a separable
least-squares grid search (the parameters that enter linearly are solved exactly for a
grid of the nonlinear ones; Golub & Pereyra 1973). If the first fit fails or stops at a
bound, a few restarts from perturbed starting values are tried.

Inference
---------
* Standard errors: square roots of the diagonal of s^2 (J'WJ)^-1 with s^2 = SS / (N - P)
  (Bates & Watts 1988, ch. 2; Seber & Wild 1989, ch. 5).
* Asymptotic CI: estimate +/- t(1 - alpha/2, N - P) * SE.
* Profile-likelihood CI (``ci_method="profile"``): the set of values of one parameter for
  which the residual SS, minimised over all other parameters, stays below
  SS_best * (1 + F(1 - alpha; 1, N - P) / (N - P)) (Bates & Watts 1988, sec. 6.1;
  Venzon & Moolgavkar 1988). Asymmetric, and exact for models linear in the parameters.
* Derived quantities that are a monotone function of one parameter (EC50 = 10^LogEC50,
  half-life = ln 2 / K) get their CI by transforming that parameter's CI; others (span)
  use the delta method.
* Confidence band of the curve: yhat +/- t sqrt(g' Cov g) with g the gradient of the model
  with respect to the parameters (delta method); the prediction band adds s^2.
* Convergence is judged by the relative-offset criterion of Bates & Watts (1981).
* Model comparison: extra-sum-of-squares F test for nested models and AICc
  (least-squares form, K = number of fitted parameters + 1).

References
----------
* Bates DM, Watts DG (1988) *Nonlinear Regression Analysis and Its Applications*, Wiley.
* Bates DM, Watts DG (1981) Technometrics 23:179-183 (relative offset convergence criterion).
* Seber GAF, Wild CJ (1989) *Nonlinear Regression*, Wiley.
* Golub GH, Pereyra V (1973) SIAM J Numer Anal 10:413-432 (separable least squares).
* Branch MA, Coleman TF, Li Y (1999) SIAM J Sci Comput 21:1-23 (trust-region-reflective).
* More JJ (1978) in *Numerical Analysis*, Lecture Notes in Mathematics 630:105-116 (LM).
* Venzon DJ, Moolgavkar SH (1988) Appl Stat 37:87-94 (profile-likelihood intervals).
* Hill AV (1910) J Physiol 40:iv-vii; DeLean A, Munson PJ, Rodbard D (1978) Am J Physiol
  235:E97-E102 (four-parameter logistic dose-response curve).
* Michaelis L, Menten ML (1913) Biochem Z 49:333-369.
* Hurvich CM, Tsai CL (1989) Biometrika 76:297-307; Burnham KP, Anderson DR (2002) *Model
  Selection and Multimodel Inference*, 2nd ed., Springer (AICc, Akaike weights).
* Wald A, Wolfowitz J (1940) Ann Math Stat 11:147-162 (runs test).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import optimize, stats
from scipy.special import expit

from ..data.tables import TableError
from ._utils import StatsError, check_alpha, ci_phrase, fmt, p_phrase, sig_phrase
from .equation import Equation, parse_equation
from .factorial import _unit_scale
from .registry import ParamSpec, register
from .regression import (
    CI_LEVEL_PARAM,
    GRID_POINTS,
    check_level,
    guarded,
    information_criteria,
    is_constant,
    lack_of_fit,
    lack_of_fit_assumption,
    mean_by_x,
    residual_normality,
    runs_assumption,
    runs_test,
    xy_series_points,
)
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

__all__ = [
    "MODELS",
    "MODEL_KEYS",
    "CurveModel",
    "compare_datasets",
    "compare_fits",
    "fit_curve",
    "get_model",
    "parse_assignments",
]

_LN10 = math.log(10.0)
_LN2 = math.log(2.0)

WEIGHTINGS = ("none", "1/y", "1/y^2", "1/sd^2")
CI_METHODS = ("asymptotic", "profile")

# Optimiser settings. Tolerances are close to machine precision because scientists compare
# results between programs to many digits; fits are small so the extra iterations are cheap.
_FTOL = 1e-14
_XTOL = 1e-14
_GTOL = 1e-14
_MAX_NFEV = 3000
#: Function-evaluation budget of each restart (kept small so hopeless data fail quickly).
_MAX_NFEV_RESTART = 300
#: Number of restarts from perturbed starting values when the first fit is unsatisfactory.
N_RESTARTS = 6
#: Bates-Watts relative offset below which a fit counts as converged.
OFFSET_TOL = 1e-4
#: Weighted SS below this fraction of sum(w y^2) is treated as a numerically perfect fit.
NEAR_PERFECT_RATIO = 1e-20
#: AICc differences smaller than this are rounding noise: the two fits are the same.
AICC_TIE = 1e-6

_CITE_BW = "Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications, Wiley"
_CITE_OFFSET = "Bates DM, Watts DG (1981) Technometrics 23:179-183"
_CITE_4PL = "DeLean A, Munson PJ, Rodbard D (1978) Am J Physiol 235:E97-E102"
_CITE_AICC = "Hurvich CM, Tsai CL (1989) Biometrika 76:297-307"
_CITE_BA = ("Burnham KP, Anderson DR (2002) Model Selection and Multimodel Inference, 2nd ed., "
            "Springer")
_CITE_PROFILE = "Venzon DJ, Moolgavkar SH (1988) Appl Stat 37:87-94"
_CITE_RUNS = "Wald A, Wolfowitz J (1940) Ann Math Stat 11:147-162"


# ============================================================== model library
@dataclass(frozen=True)
class Derived:
    """A quantity computed from fitted parameters (EC50, half-life, span...)."""

    name: str
    sources: tuple[str, ...]
    value: Callable[[dict[str, float]], float]
    gradient: Callable[[dict[str, float]], dict[str, float]]
    #: True when the quantity is a monotone function of ONE parameter (CI by transformation).
    monotone: bool = False
    #: True when the source must be positive for the quantity to make sense (ln 2 / K).
    needs_positive: bool = False
    note: str = ""


@dataclass(frozen=True)
class CurveModel:
    """One entry of the model library."""

    key: str
    label: str
    equation: str
    parameters: tuple[str, ...]
    descriptions: dict[str, str]
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray, np.ndarray], np.ndarray]
    guess: Callable[[np.ndarray, np.ndarray, dict[str, float]], dict[str, float]]
    summary: str = ""
    default_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    x_is_log: bool = False
    derived: tuple[Derived, ...] = ()
    family: str = "general"
    #: Parameters that are fixed by the model definition (HillSlope of the 3-parameter curves).
    default_fixed: dict[str, float] = field(default_factory=dict)
    #: Parameters that should come out positive for the fit to be interpretable.
    positive: tuple[str, ...] = ()
    #: Transformation applied to the X data before ``func`` (log10 for linear-X dose-response).
    x_transform: Callable[[np.ndarray], np.ndarray] | None = None
    #: Key of a model this one is a constrained special case of, with the constrained names.
    nested_in: tuple[str, ...] = ()
    #: Remarks shown with every fit of this model (how a typed equation was read).
    notes: tuple[str, ...] = ()

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)


def _col(*columns: Any) -> np.ndarray:
    return np.column_stack([np.asarray(c, dtype=float) for c in columns])


def _separable_guess(
    y: np.ndarray, candidates: Sequence[tuple[Any, np.ndarray]]
) -> tuple[Any, np.ndarray] | None:
    """Pick the candidate (tag, basis) whose linear least-squares fit to y has the smallest SS.

    Returns (tag, linear coefficients) or ``None`` when nothing could be evaluated.
    """
    best: tuple[float, Any, np.ndarray] | None = None
    for tag, basis in candidates:
        if not np.all(np.isfinite(basis)):
            continue
        try:
            coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - lstsq (SVD) rarely fails
            continue
        resid = y - basis @ coef
        ss = float(resid @ resid)
        if math.isfinite(ss) and (best is None or ss < best[0]):
            best = (ss, tag, coef)
    if best is None:
        return None
    return best[1], best[2]


def _x_span(x: np.ndarray) -> float:
    span = float(np.ptp(x))
    if span > 0:
        return span
    return max(abs(float(x[0])), 1.0)


# ---- sigmoidal dose-response (Hill) --------------------------------------------
def _hill_s(lx: np.ndarray, logec50: float, hill: float) -> np.ndarray:
    """Fraction of the span reached: 1 / (1 + 10^((LogEC50 - lx) * Hill)), safe at lx = -inf."""
    with np.errstate(all="ignore"):
        u = (logec50 - lx) * hill
        u = np.where(np.isnan(u), 0.0, u)  # inf * 0: (EC50 / 0)^0 = 1
        return expit(-u * _LN10)


def _hill_func(lx: np.ndarray, p: np.ndarray) -> np.ndarray:
    bottom, top, logec50, hill = p
    return bottom + (top - bottom) * _hill_s(lx, logec50, hill)


def _hill_jac(lx: np.ndarray, p: np.ndarray) -> np.ndarray:
    bottom, top, logec50, hill = p
    s = _hill_s(lx, logec50, hill)
    span = top - bottom
    with np.errstate(all="ignore"):
        ds = -_LN10 * s * (1.0 - s)  # d s / d u, with u = (LogEC50 - lx) * Hill
        d_hill = np.where(ds == 0.0, 0.0, span * ds * (logec50 - lx))
    return _col(1.0 - s, s, span * ds * hill, d_hill)


def _log10_dose(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), -np.inf)


def _guess_hill(lx: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
    finite = np.isfinite(lx)
    lxf = lx[finite] if np.any(finite) else np.zeros(1)
    lo, hi = float(lxf.min()), float(lxf.max())
    span = hi - lo if hi > lo else 1.0
    b_fix, t_fix = fixed.get("Bottom"), fixed.get("Top")
    h_fix, c_fix = fixed.get("HillSlope"), fixed.get("LogEC50")
    magnitudes = (0.4, 0.7, 1.0, 1.5, 2.5, 4.0)
    if h_fix is not None:
        hills: tuple[float, ...] = (float(h_fix),)
    elif b_fix is None and t_fix is None:
        hills = magnitudes  # the mirror solution is recovered by swapping Top and Bottom
    else:
        hills = magnitudes + tuple(-m for m in magnitudes)
    centres = (float(c_fix),) if c_fix is not None else tuple(
        np.linspace(lo - 0.25 * span, hi + 0.25 * span, 31))
    candidates = []
    target = y.astype(float)
    for h in hills:
        for c in centres:
            s = _hill_s(lx, c, h)
            cols = []
            offset = np.zeros_like(y, dtype=float)
            if b_fix is None:
                cols.append(1.0 - s)
            else:
                offset = offset + b_fix * (1.0 - s)
            if t_fix is None:
                cols.append(s)
            else:
                offset = offset + t_fix * s
            if cols:
                candidates.append(((c, h, offset), np.column_stack(cols)))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    fallback = {"Bottom": ymin, "Top": ymax, "LogEC50": 0.5 * (lo + hi), "HillSlope": 1.0}
    if not candidates:
        return fallback
    best: tuple[float, float, float, np.ndarray] | None = None
    for (c, h, offset), basis in candidates:
        try:
            coef, *_ = np.linalg.lstsq(basis, target - offset, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        resid = target - offset - basis @ coef
        ss = float(resid @ resid)
        if math.isfinite(ss) and (best is None or ss < best[0] * (1.0 - 1e-12)):
            best = (ss, c, h, coef)
    if best is None:  # pragma: no cover - defensive
        return fallback
    _, c, h, coef = best
    values = list(coef)
    bottom = float(values.pop(0)) if b_fix is None else float(b_fix)
    top = float(values.pop(0)) if t_fix is None else float(t_fix)
    if b_fix is None and t_fix is None and h_fix is None and top < bottom:
        bottom, top, h = top, bottom, -h
    # keep plateau guesses within a sensible distance of the data
    pad = 2.0 * (ymax - ymin) if ymax > ymin else 1.0
    if b_fix is None:
        bottom = float(np.clip(bottom, ymin - pad, ymax + pad))
    if t_fix is None:
        top = float(np.clip(top, ymin - pad, ymax + pad))
    return {"Bottom": bottom, "Top": top, "LogEC50": float(c), "HillSlope": float(h)}


def _ec50_value(d: dict[str, float]) -> float:
    return 10.0 ** d["LogEC50"]


_HILL_DERIVED = (
    Derived(
        name="EC50", sources=("LogEC50",), value=_ec50_value,
        gradient=lambda d: {"LogEC50": _LN10 * 10.0 ** d["LogEC50"]}, monotone=True,
        note="10^LogEC50, in the units of your X values (before taking the logarithm); "
             "CI obtained by transforming the LogEC50 interval, so it is asymmetric",
    ),
    Derived(
        name="Span", sources=("Top", "Bottom"), value=lambda d: d["Top"] - d["Bottom"],
        gradient=lambda d: {"Top": 1.0, "Bottom": -1.0},
        note="Top - Bottom; SE and CI by the delta method",
    ),
)

_HILL_DESCRIPTIONS = {
    "Bottom": "The plateau of the curve at one end, in the units of Y (the lower plateau when "
              "Top > Bottom).",
    "Top": "The plateau of the curve at the other end, in the units of Y.",
    "LogEC50": "log10 of the concentration that gives a response halfway between Bottom and Top.",
    "HillSlope": "Steepness of the curve. 1 is the standard slope of simple one-site binding; "
                 "larger magnitudes are steeper. Positive = response rises with dose, negative "
                 "= response falls with dose (inhibition).",
}


# ---- hyperbolas -----------------------------------------------------------------
def _hyperbola_func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return p[0] * x / (p[1] + x)


def _hyperbola_jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        denom = p[1] + x
        return _col(x / denom, -p[0] * x / denom**2)


def _half_grid(x: np.ndarray, n: int = 40) -> np.ndarray:
    pos = np.abs(x[x != 0])
    if pos.size == 0:
        return np.array([1.0])
    lo, hi = float(pos.min()), float(pos.max())
    return np.geomspace(lo / 10.0, hi * 10.0, n)


def _guess_hyperbola(names: tuple[str, str]) -> Callable[..., dict[str, float]]:
    def guess(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
        with np.errstate(all="ignore"):
            candidates = [(k, (x / (k + x))[:, None]) for k in _half_grid(x)]
        found = _separable_guess(y, candidates)
        if found is None:
            return {names[0]: float(np.max(y)), names[1]: float(np.median(x))}
        k, coef = found
        return {names[0]: float(coef[0]), names[1]: float(k)}
    return guess


def _total_binding_func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return p[0] * x / (p[1] + x) + p[2] * x + p[3]


def _total_binding_jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        denom = p[1] + x
        return _col(x / denom, -p[0] * x / denom**2, x, np.ones_like(x))


def _guess_total_binding(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
    with np.errstate(all="ignore"):
        candidates = [(k, _col(x / (k + x), x, np.ones_like(x))) for k in _half_grid(x)]
    found = _separable_guess(y, candidates)
    if found is None:
        return {"Bmax": float(np.max(y)), "Kd": float(np.median(x)), "NS": 0.0, "Background": 0.0}
    k, coef = found
    return {"Bmax": float(coef[0]), "Kd": float(k), "NS": float(coef[1]),
            "Background": float(coef[2])}


# ---- exponentials ---------------------------------------------------------------
def _decay_func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    y0, plateau, k = p
    with np.errstate(all="ignore"):
        return (y0 - plateau) * np.exp(-k * x) + plateau


def _decay_jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    y0, plateau, k = p
    with np.errstate(all="ignore"):
        e = np.exp(-k * x)
        return _col(e, 1.0 - e, -(y0 - plateau) * x * e)


def _guess_exponential(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
    span = _x_span(x)
    candidates = []
    with np.errstate(all="ignore"):
        for k in np.geomspace(0.01 / span, 100.0 / span, 50):
            e = np.exp(-k * x)
            candidates.append((k, _col(e, 1.0 - e)))
    found = _separable_guess(y, candidates)
    if found is None:
        return {"Y0": float(y[np.argmin(x)]), "Plateau": float(y[np.argmax(x)]), "K": 1.0 / span}
    k, coef = found
    return {"Y0": float(coef[0]), "Plateau": float(coef[1]), "K": float(k)}


def _growth_func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return p[0] * np.exp(p[1] * x)


def _growth_jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        e = np.exp(p[1] * x)
        return _col(e, p[0] * x * e)


def _guess_growth(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
    span = _x_span(x)
    sign = 1.0 if np.all(y > 0) else -1.0 if np.all(y < 0) else 0.0
    if sign != 0.0 and not is_constant(x):
        slope, intercept = np.polyfit(x, np.log(sign * y), 1)
        return {"Y0": float(sign * math.exp(min(intercept, 700.0))), "K": float(slope)}
    xmax = float(np.max(x))
    candidates = []
    with np.errstate(all="ignore"):
        for k in np.geomspace(0.01 / span, 20.0 / span, 30):
            for sgn in (1.0, -1.0):
                candidates.append(((sgn * k), np.exp(sgn * k * (x - xmax))[:, None]))
    found = _separable_guess(y, candidates)
    if found is None:  # pragma: no cover - defensive
        return {"Y0": float(y[np.argmin(x)]), "K": 1.0 / span}
    k, coef = found
    return {"Y0": float(coef[0] * math.exp(max(min(-k * xmax, 700.0), -700.0))), "K": float(k)}


def _rate_derived(span_sign: float) -> tuple[Derived, ...]:
    return (
        Derived(
            name="HalfLife", sources=("K",), value=lambda d: _LN2 / d["K"],
            gradient=lambda d: {"K": -_LN2 / d["K"] ** 2}, monotone=True, needs_positive=True,
            note="ln(2) / K, in the units of X; CI obtained by transforming the CI of K",
        ),
        Derived(
            name="Tau", sources=("K",), value=lambda d: 1.0 / d["K"],
            gradient=lambda d: {"K": -1.0 / d["K"] ** 2}, monotone=True, needs_positive=True,
            note="Time constant 1 / K, in the units of X; CI obtained by transforming the CI of K",
        ),
        Derived(
            name="Span", sources=("Y0", "Plateau"),
            value=lambda d: span_sign * (d["Y0"] - d["Plateau"]),
            gradient=lambda d: {"Y0": span_sign, "Plateau": -span_sign},
            note=("Y0 - Plateau" if span_sign > 0 else "Plateau - Y0")
                 + "; SE and CI by the delta method",
        ),
    )


# ---- gaussian and polynomials ---------------------------------------------------
def _gauss_func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    amp, mean, sd = p
    with np.errstate(all="ignore"):
        return amp * np.exp(-0.5 * ((x - mean) / sd) ** 2)


def _gauss_jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    amp, mean, sd = p
    with np.errstate(all="ignore"):
        z = (x - mean) / sd
        e = np.exp(-0.5 * z * z)
        return _col(e, amp * e * z / sd, amp * e * z * z / sd)


def _guess_gauss(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
    span = _x_span(x)
    lo, hi = float(np.min(x)), float(np.max(x))
    candidates = []
    with np.errstate(all="ignore"):
        for mean in np.linspace(lo, hi, 25):
            for sd in (span / 1.5, span / 3.0, span / 6.0, span / 12.0, span / 24.0):
                candidates.append(((mean, sd), np.exp(-0.5 * ((x - mean) / sd) ** 2)[:, None]))
    found = _separable_guess(y, candidates)
    if found is None:  # pragma: no cover - defensive
        return {"Amplitude": float(np.max(y)), "Mean": 0.5 * (lo + hi), "SD": span / 4.0}
    (mean, sd), coef = found
    return {"Amplitude": float(coef[0]), "Mean": float(mean), "SD": float(sd)}


def _poly_model(degree: int) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray],
                                      Callable[..., dict[str, float]]]:
    names = ("Intercept", "Slope") if degree == 1 else tuple(f"B{i}" for i in range(degree + 1))

    def func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for power in range(degree, -1, -1):
            out = out * x + p[power]
        return out

    def jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.column_stack([x**power for power in range(degree + 1)])

    def guess(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
        basis = np.column_stack([x**power for power in range(degree + 1)])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        return {name: float(c) for name, c in zip(names, coef)}

    return func, jac, guess


def _build_models() -> dict[str, CurveModel]:
    models: dict[str, CurveModel] = {}

    def add(model: CurveModel) -> None:
        models[model.key] = model

    hill_logx = CurveModel(
        key="dose_response_4pl_logx",
        label="Dose-response, four parameters (X is log10 of dose)",
        equation="Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) * HillSlope))",
        parameters=("Bottom", "Top", "LogEC50", "HillSlope"),
        descriptions=_HILL_DESCRIPTIONS,
        func=_hill_func, jac=_hill_jac, guess=_guess_hill,
        summary="Sigmoidal (S-shaped) dose-response curve with a variable slope. Enter X as "
                "log10(concentration). Works for stimulation (EC50) and inhibition (IC50).",
        x_is_log=True, derived=_HILL_DERIVED, family="dose_response",
    )
    add(hill_logx)
    hill_linear = replace(
        hill_logx,
        key="dose_response_4pl",
        label="Dose-response, four parameters (X is dose)",
        equation="Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)",
        summary="Sigmoidal dose-response curve with a variable slope. Enter X as the "
                "concentration itself (not its logarithm). The fit is done on log10(EC50) "
                "internally, which keeps EC50 positive and gives a properly asymmetric CI.",
        x_is_log=False, x_transform=_log10_dose,
        descriptions={**_HILL_DESCRIPTIONS,
                      "LogEC50": "log10 of EC50 (the parameter actually fitted)."},
    )
    add(hill_linear)
    add(replace(
        hill_logx,
        key="dose_response_3pl_logx",
        label="Dose-response, three parameters (X is log10 of dose, standard slope)",
        equation="Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) * HillSlope)), "
                 "HillSlope fixed at +1 or -1",
        summary="Sigmoidal dose-response curve with the standard slope (Hill slope fixed at +1 "
                "for stimulation or -1 for inhibition). Use it when there are too few points to "
                "determine the slope.",
        default_fixed={"HillSlope": 1.0}, nested_in=("dose_response_4pl_logx", "HillSlope"),
    ))
    add(replace(
        hill_linear,
        key="dose_response_3pl",
        label="Dose-response, three parameters (X is dose, standard slope)",
        equation="Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope), "
                 "HillSlope fixed at +1 or -1",
        summary="Dose-response curve with the standard slope (Hill slope fixed at +1 or -1); "
                "X is the concentration itself.",
        default_fixed={"HillSlope": 1.0}, nested_in=("dose_response_4pl", "HillSlope"),
    ))
    add(CurveModel(
        key="michaelis_menten",
        label="Michaelis-Menten enzyme kinetics",
        equation="Y = Vmax * X / (Km + X)",
        parameters=("Vmax", "Km"),
        descriptions={
            "Vmax": "Maximum enzyme velocity at saturating substrate, in the units of Y.",
            "Km": "Substrate concentration that gives half of Vmax, in the units of X.",
        },
        func=_hyperbola_func, jac=_hyperbola_jac, guess=_guess_hyperbola(("Vmax", "Km")),
        summary="Enzyme velocity as a function of substrate concentration.",
        family="hyperbola", positive=("Vmax", "Km"),
        nested_in=("one_site_binding_total", "NS", "Background"),
    ))
    add(CurveModel(
        key="one_phase_decay",
        label="One-phase exponential decay",
        equation="Y = (Y0 - Plateau) * exp(-K * X) + Plateau",
        parameters=("Y0", "Plateau", "K"),
        descriptions={
            "Y0": "Y value when X is zero, in the units of Y.",
            "Plateau": "Y value at very large X, in the units of Y.",
            "K": "Rate constant, in reciprocal X units (per minute if X is in minutes).",
        },
        func=_decay_func, jac=_decay_jac, guess=_guess_exponential,
        summary="Y falls from Y0 towards a plateau at a rate proportional to its distance from "
                "the plateau (dissociation, drug elimination, radioactive decay).",
        derived=_rate_derived(1.0), family="exponential", positive=("K",),
    ))
    add(CurveModel(
        key="one_phase_association",
        label="One-phase exponential association",
        equation="Y = Y0 + (Plateau - Y0) * (1 - exp(-K * X))",
        parameters=("Y0", "Plateau", "K"),
        descriptions={
            "Y0": "Y value when X is zero, in the units of Y.",
            "Plateau": "Y value at very large X, in the units of Y.",
            "K": "Rate constant, in reciprocal X units.",
        },
        func=_decay_func, jac=_decay_jac, guess=_guess_exponential,
        summary="Y rises from Y0 and levels off at a plateau (association binding, product "
                "formation, growth to a ceiling).",
        derived=_rate_derived(-1.0), family="exponential", positive=("K",),
    ))
    add(CurveModel(
        key="one_site_binding_specific",
        label="One site: specific binding",
        equation="Y = Bmax * X / (Kd + X)",
        parameters=("Bmax", "Kd"),
        descriptions={
            "Bmax": "Maximum specific binding, in the units of Y.",
            "Kd": "Equilibrium dissociation constant: the ligand concentration that occupies "
                  "half the sites, in the units of X.",
        },
        func=_hyperbola_func, jac=_hyperbola_jac, guess=_guess_hyperbola(("Bmax", "Kd")),
        summary="Saturation binding to a single class of sites, after non-specific binding has "
                "been subtracted.",
        family="hyperbola", positive=("Bmax", "Kd"),
        nested_in=("one_site_binding_total", "NS", "Background"),
    ))
    add(CurveModel(
        key="one_site_binding_total",
        label="One site: total binding",
        equation="Y = Bmax * X / (Kd + X) + NS * X + Background",
        parameters=("Bmax", "Kd", "NS", "Background"),
        descriptions={
            "Bmax": "Maximum specific binding, in the units of Y.",
            "Kd": "Equilibrium dissociation constant, in the units of X.",
            "NS": "Slope of non-specific binding (Y units per X unit).",
            "Background": "Signal with no ligand added, in the units of Y.",
        },
        func=_total_binding_func, jac=_total_binding_jac, guess=_guess_total_binding,
        summary="Saturation binding measured as total binding: specific binding plus a linear "
                "non-specific component and a constant background.",
        family="hyperbola", positive=("Bmax", "Kd"),
    ))
    add(CurveModel(
        key="exponential_growth",
        label="Exponential growth",
        equation="Y = Y0 * exp(K * X)",
        parameters=("Y0", "K"),
        descriptions={
            "Y0": "Y value when X is zero, in the units of Y.",
            "K": "Growth rate constant, in reciprocal X units.",
        },
        func=_growth_func, jac=_growth_jac, guess=_guess_growth,
        summary="Y multiplies by a constant factor in every unit of X (unrestricted growth).",
        derived=(
            Derived(
                name="DoublingTime", sources=("K",), value=lambda d: _LN2 / d["K"],
                gradient=lambda d: {"K": -_LN2 / d["K"] ** 2}, monotone=True, needs_positive=True,
                note="ln(2) / K, in the units of X; CI obtained by transforming the CI of K",
            ),
            Derived(
                name="Tau", sources=("K",), value=lambda d: 1.0 / d["K"],
                gradient=lambda d: {"K": -1.0 / d["K"] ** 2}, monotone=True, needs_positive=True,
                note="Time constant 1 / K, in the units of X; CI obtained by transforming the CI of K",
            ),
        ),
        family="exponential", positive=("K",),
    ))
    add(CurveModel(
        key="gaussian",
        label="Gaussian (bell-shaped curve)",
        equation="Y = Amplitude * exp(-0.5 * ((X - Mean) / SD)^2)",
        parameters=("Amplitude", "Mean", "SD"),
        descriptions={
            "Amplitude": "Height of the peak, in the units of Y.",
            "Mean": "X value at the centre of the peak.",
            "SD": "Width of the peak (standard deviation), in the units of X.",
        },
        func=_gauss_func, jac=_gauss_jac, guess=_guess_gauss,
        summary="A symmetric bell-shaped peak.",
        default_bounds={"SD": (0.0, None)},
        derived=(
            Derived(
                name="FWHM", sources=("SD",),
                value=lambda d: 2.0 * math.sqrt(2.0 * _LN2) * d["SD"],
                gradient=lambda d: {"SD": 2.0 * math.sqrt(2.0 * _LN2)}, monotone=True,
                note="Full width at half maximum = 2.3548 * SD; CI obtained by transforming "
                     "the CI of SD",
            ),
        ),
        family="peak", positive=("SD",),
    ))
    lin_func, lin_jac, lin_guess = _poly_model(1)
    add(CurveModel(
        key="linear",
        label="Straight line",
        equation="Y = Intercept + Slope * X",
        parameters=("Intercept", "Slope"),
        descriptions={
            "Intercept": "Y value when X is zero.",
            "Slope": "Change in Y for a one-unit increase in X.",
        },
        func=lin_func, jac=lin_jac, guess=lin_guess,
        summary="A straight line; mainly useful as a simpler model to compare a curve against.",
        family="polynomial", nested_in=("quadratic", "B2"),
    ))
    quad_func, quad_jac, quad_guess = _poly_model(2)
    add(CurveModel(
        key="quadratic",
        label="Quadratic (second-order polynomial)",
        equation="Y = B0 + B1 * X + B2 * X^2",
        parameters=("B0", "B1", "B2"),
        descriptions={
            "B0": "Y value when X is zero.",
            "B1": "Linear coefficient.",
            "B2": "Quadratic coefficient (curvature).",
        },
        func=quad_func, jac=quad_jac, guess=quad_guess,
        summary="A parabola; a descriptive curve with no mechanistic meaning.",
        family="polynomial",
    ))
    return models


#: The built-in model library (key -> :class:`CurveModel`). ``user_defined`` is built on demand.
MODELS: dict[str, CurveModel] = _build_models()
#: Keys accepted by :func:`fit_curve` (built-ins plus ``user_defined``).
MODEL_KEYS: tuple[str, ...] = (*MODELS, "user_defined")


def _user_model(equation: str | Equation) -> CurveModel:
    eq = equation if isinstance(equation, Equation) else parse_equation(equation)
    if eq.n_parameters == 0:
        raise StatsError(
            "The equation has no parameters to fit. Use names for the quantities to estimate, "
            "for example 'Y = A * exp(-K * X)'."
        )
    names = tuple(eq.parameters)

    def func(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return eq.func(x, *p)

    def jac(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return eq.jacobian(x, *p)

    def guess(x: np.ndarray, y: np.ndarray, fixed: dict[str, float]) -> dict[str, float]:
        return dict.fromkeys(names, 1.0)

    return CurveModel(
        key="user_defined", label="User-defined equation", equation=eq.text, parameters=names,
        descriptions={name: ("Parameter of your equation (fitted; a lone 'e' is not Euler's number: "
                             "write exp(1) for that)." if name == "e" else "Parameter of your equation.")
                      for name in names},
        func=func, jac=jac, guess=guess, summary="An equation you typed.", family="user",
        notes=tuple(eq.notes),
    )


def get_model(model: str | CurveModel, equation: str | Equation | None = None) -> CurveModel:
    """Look up a built-in model by key, or build a ``user_defined`` model from equation text."""
    if isinstance(model, CurveModel):
        return model
    key = str(model).strip()
    if key == "user_defined":
        if equation is None or (isinstance(equation, str) and not equation.strip()):
            raise StatsError(
                "The 'user_defined' model needs an equation, for example "
                "'Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)'."
            )
        return _user_model(equation)
    if key not in MODELS:
        raise StatsError(f"Unknown model '{model}'. Available models: {', '.join(MODEL_KEYS)}.")
    return MODELS[key]


# ============================================================ option handling
def parse_assignments(text: Any) -> dict[str, float]:
    """Leniently parse ``"Top=100, Bottom=0; EC50: 10"`` into ``{"Top": 100.0, ...}``.

    Accepts a dict (returned as floats), ``None`` or empty text (empty dict). Items are
    separated by commas, semicolons or new lines; name and value by ``=`` or ``:``.
    """
    if text is None:
        return {}
    if isinstance(text, dict):
        out = {}
        for key, value in text.items():
            try:
                out[str(key).strip()] = float(value)
            except (TypeError, ValueError):
                raise StatsError(f"The value given for '{key}' is not a number.") from None
        return out
    if not isinstance(text, str):
        raise StatsError("Parameter values must be given as text such as 'Top=100, Bottom=0'.")
    out = {}
    for part in re.split(r"[,;\n]+", text):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"([^\s=:]+)\s*[=:]\s*(\S+)", part)
        if not match:
            raise StatsError(
                f"Could not understand '{part}'. Write parameter values like 'Top=100, Bottom=0'."
            )
        try:
            value = float(match.group(2))
        except ValueError:
            raise StatsError(
                f"The value '{match.group(2)}' given for '{match.group(1)}' is not a number."
            ) from None
        out[match.group(1)] = value
    return out


def _canonical(model: CurveModel, name: str) -> tuple[str, bool]:
    """Map a user-typed parameter name to (internal name, value is on the EC50 scale)."""
    lookup = {p.lower(): p for p in model.parameters}
    key = str(name).strip().lower()
    if model.family == "dose_response":
        aliases = {"logic50": ("LogEC50", False), "ec50": ("LogEC50", True),
                   "ic50": ("LogEC50", True), "hill": ("HillSlope", False),
                   "slope": ("HillSlope", False)}
        if key in aliases:
            return aliases[key]
    if key in lookup:
        return lookup[key], False
    valid = list(model.parameters)
    if model.family == "dose_response":
        valid.append("EC50")
    raise StatsError(
        f"'{name}' is not a parameter of this model. Parameters: {', '.join(valid)}."
    )


def _to_internal(value: float, on_ec50_scale: bool, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise StatsError(f"The value given for '{name}' must be a finite number.")
    if on_ec50_scale:
        if value <= 0:
            raise StatsError(f"{name} must be greater than zero (it is a concentration).")
        return math.log10(value)
    return value


def _resolve_values(model: CurveModel, values: Any, what: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in parse_assignments(values).items():
        internal, ec50_scale = _canonical(model, name)
        if internal in out:
            raise StatsError(f"{what}: '{internal}' was given twice (as '{name}').")
        out[internal] = _to_internal(value, ec50_scale, name)
    return out


def _resolve_bounds(
    model: CurveModel, bounds: Any
) -> dict[str, tuple[float | None, float | None]]:
    out: dict[str, tuple[float | None, float | None]] = dict(model.default_bounds)
    if not bounds:
        return out
    if not isinstance(bounds, dict):
        raise StatsError("bounds must be a dict such as {'HillSlope': (0, None)}.")
    for name, pair in bounds.items():
        internal, ec50_scale = _canonical(model, name)
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            raise StatsError(
                f"The bounds for '{name}' must be a (lower, upper) pair; use None for no limit."
            ) from None
        try:
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
        except (TypeError, ValueError):
            raise StatsError(f"The bounds for '{name}' must be numbers or None.") from None
        lo = None if lo is None or lo == -math.inf else lo
        hi = None if hi is None or hi == math.inf else hi
        if ec50_scale:
            lo = None if lo is None or lo <= 0 else math.log10(lo)
            if hi is not None:
                hi = _to_internal(hi, True, name)
        if lo is not None and hi is not None and not lo < hi:
            raise StatsError(f"The lower bound for '{name}' must be below the upper bound.")
        out[internal] = (lo, hi)
    return out


# ================================================================ fit engine
@dataclass
class _Candidate:
    theta: np.ndarray
    ss: float
    status: int
    message: str
    nfev: int
    method: str
    offset: float | None = None
    converged: bool = False
    active: np.ndarray | None = None


@dataclass
class _Problem:
    """A least-squares problem in terms of the FREE parameter vector."""

    residual: Callable[[np.ndarray], np.ndarray]   # weighted residuals (model - data)
    jacobian: Callable[[np.ndarray], np.ndarray]   # d residual / d theta
    lower: np.ndarray
    upper: np.ndarray
    n: int
    exact_ss: float                                # residual SS at or below which the fit is exact
    #: Power of two close to the size of the weighted data. The optimiser works on
    #: residuals divided by it (exact), so that its stopping rules (some of which are
    #: absolute thresholds on the gradient) do not depend on the unit of Y.
    unit: float = 1.0

    @property
    def bounded(self) -> bool:
        return bool(np.any(np.isfinite(self.lower)) or np.any(np.isfinite(self.upper)))


def _safe_residual(problem: _Problem) -> Callable[[np.ndarray], np.ndarray]:
    """Residuals for the optimiser: in units of ``problem.unit`` (data of order 1), with a
    large finite penalty where the model is undefined."""
    def fun(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            r = problem.residual(theta) / problem.unit
        return np.where(np.isfinite(r), r, 1e150)
    return fun


def _safe_jacobian(problem: _Problem) -> Callable[[np.ndarray], np.ndarray]:
    def jac(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            j = problem.jacobian(theta) / problem.unit
        return np.where(np.isfinite(j), j, 0.0)
    return jac


def _active_mask(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Which parameters sit on a bound. "On the bound" is judged relative to the size of
    the parameter and of the bound (never against an absolute distance), so a Km of 1e-9
    with a lower bound of 0 is not mistaken for a parameter stuck at 0."""
    with np.errstate(invalid="ignore"):
        tol_lo = 1e-9 * np.maximum(np.abs(theta), np.where(np.isfinite(lower), np.abs(lower), 0.0))
        tol_hi = 1e-9 * np.maximum(np.abs(theta), np.where(np.isfinite(upper), np.abs(upper), 0.0))
        at_lower = np.isfinite(lower) & (theta - lower <= tol_lo)
        at_upper = np.isfinite(upper) & (upper - theta <= tol_hi)
    return at_lower | at_upper


def _tangent_basis(j: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of ``j`` (rank-revealing, column-scaled SVD)."""
    norms = np.linalg.norm(j, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    u, sv, _ = np.linalg.svd(j / norms, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0:
        return u[:, :0]
    keep = sv > sv[0] * max(j.shape) * np.finfo(float).eps * 10
    return u[:, keep]


def _relative_offset(problem: _Problem, theta: np.ndarray, active: np.ndarray) -> float | None:
    """Bates & Watts (1981) relative offset: size of the residual component in the tangent
    plane of the model relative to the component orthogonal to it. Zero at an exact minimum.
    Returns ``None`` when it cannot be evaluated (as many parameters as points, or
    undefined model values)."""
    with np.errstate(all="ignore"):
        r = problem.residual(theta)
        j = problem.jacobian(theta)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(j))):
        return None
    j = j[:, ~active]
    n = r.size
    if j.shape[1] == 0:
        return 0.0
    try:
        q = _tangent_basis(j)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
    p = q.shape[1]
    if p == 0:
        return 0.0
    if n <= p:
        return None
    tangent = float(np.sum((q.T @ r) ** 2))
    orth = float(r @ r) - tangent
    if orth <= 0:
        return 0.0
    return math.sqrt(tangent / p) / math.sqrt(orth / (n - p))


def _attempt(problem: _Problem, start: np.ndarray, method: str,
             max_nfev: int = _MAX_NFEV) -> _Candidate | None:
    fun = _safe_residual(problem)
    jac = _safe_jacobian(problem)
    with np.errstate(all="ignore"):
        r0 = problem.residual(start)
    if not np.all(np.isfinite(r0)):
        return None
    kwargs: dict[str, Any] = {}
    if method != "lm":
        kwargs["bounds"] = (problem.lower, problem.upper)
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            res = optimize.least_squares(
                fun, start, jac=jac, method=method, x_scale="jac", ftol=_FTOL, xtol=_XTOL,
                gtol=_GTOL, max_nfev=max_nfev, **kwargs,
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
        return None
    theta = np.asarray(res.x, dtype=float)
    with np.errstate(all="ignore"):
        r = problem.residual(theta)
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(r))):
        return None
    ss = float(r @ r)
    cand = _Candidate(theta=theta, ss=ss, status=int(res.status), message=str(res.message),
                      nfev=int(res.nfev), method=method)
    return _judge(problem, cand)


def _judge(problem: _Problem, cand: _Candidate) -> _Candidate:
    cand.active = _active_mask(cand.theta, problem.lower, problem.upper)
    cand.offset = _relative_offset(problem, cand.theta, cand.active)
    near_perfect = cand.ss <= problem.exact_ss
    if near_perfect:
        cand.converged = True
    elif cand.offset is None:
        cand.converged = False
    else:
        cand.converged = cand.offset < OFFSET_TOL
    return cand


def _tangent_norm(r: np.ndarray, j: np.ndarray, norms: np.ndarray) -> float:
    """Length of the projection of the residual vector on the tangent plane of the model."""
    return float(np.linalg.norm(_tangent_basis(j).T @ r))


def _polish(problem: _Problem, cand: _Candidate) -> _Candidate:
    """A few undamped Gauss-Newton steps to squeeze out the last digits.

    Near the minimum the SS is flat to rounding error long before the parameters stop
    improving, so a step is accepted when it shrinks the tangent-plane component of the
    residuals (which is zero at the exact least-squares solution) without increasing the SS
    beyond rounding. Skipped when a bound is active.
    """
    if cand.active is not None and np.any(cand.active):
        return cand
    theta, ss = cand.theta.copy(), cand.ss
    if theta.size == 0 or theta.size >= problem.n:
        return cand
    try:
        with np.errstate(all="ignore"):
            r = problem.residual(theta)
            j = problem.jacobian(theta)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(j))):
            return cand
        norms = np.linalg.norm(j, axis=0)
        if np.any(norms == 0):
            return cand
        tangent = _tangent_norm(r, j, norms)
        for _ in range(12):
            step, *_ = np.linalg.lstsq(j / norms, -r, rcond=None)
            trial = theta + step / norms
            if np.any(trial <= problem.lower) or np.any(trial >= problem.upper):
                break
            with np.errstate(all="ignore"):
                r_new = problem.residual(trial)
                j_new = problem.jacobian(trial)
            if not (np.all(np.isfinite(r_new)) and np.all(np.isfinite(j_new))):
                break
            ss_new = float(r_new @ r_new)
            norms_new = np.linalg.norm(j_new, axis=0)
            if np.any(norms_new == 0) or ss_new > ss * (1.0 + 1e-10):
                break
            tangent_new = _tangent_norm(r_new, j_new, norms_new)
            if tangent_new >= tangent:
                break
            theta, ss, r, j, norms, tangent = trial, ss_new, r_new, j_new, norms_new, tangent_new
    except np.linalg.LinAlgError:  # pragma: no cover - SVD/QR failure
        return cand
    if ss <= cand.ss * (1.0 + 1e-10):
        cand.theta, cand.ss = theta, ss
        _judge(problem, cand)
    return cand


def _better(a: _Candidate | None, b: _Candidate | None) -> _Candidate | None:
    """Prefer converged fits, then fits not stuck at a bound, then the smaller SS."""
    if a is None:
        return b
    if b is None:
        return a

    def rank(c: _Candidate) -> tuple[int, int]:
        at_bound = bool(c.active is not None and np.any(c.active))
        return (0 if c.converged else 1, 1 if at_bound else 0)

    ra, rb = rank(a), rank(b)
    if ra[0] != rb[0]:
        return a if ra[0] < rb[0] else b
    # among equally converged fits an interior solution with (nearly) the same SS wins
    if b.ss < a.ss * (1.0 - 1e-10):
        return b
    if a.ss < b.ss * (1.0 - 1e-10):
        return a
    return a if ra[1] <= rb[1] else b


def _typical_size(values: np.ndarray) -> np.ndarray:
    """Magnitude of each starting value; a value of exactly 0 borrows the median magnitude of
    the others (1 if all are 0). Used instead of absolute floors such as 1e-3, which are
    huge for parameters expressed in small units (a Kd in mol/L) and negligible for large ones."""
    size = np.abs(np.asarray(values, dtype=float))
    nonzero = size[size > 0]
    fallback = float(np.median(nonzero)) if nonzero.size else 1.0
    return np.where(size > 0, size, fallback)


def _clip_start(start: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    out = start.astype(float).copy()
    size = _typical_size(out)
    for i in range(out.size):
        lo, hi = lower[i], upper[i]
        if math.isfinite(lo) and math.isfinite(hi):
            margin = 1e-6 * (hi - lo)
            out[i] = min(max(out[i], lo + margin), hi - margin)
        elif math.isfinite(lo) and out[i] <= lo:
            out[i] = lo + 1e-6 * max(abs(lo), size[i])
        elif math.isfinite(hi) and out[i] >= hi:
            out[i] = hi - 1e-6 * max(abs(hi), size[i])
    return out


def _solve(problem: _Problem, start: np.ndarray, restarts: bool = True,
           max_nfev: int = _MAX_NFEV) -> tuple[_Candidate | None, int]:
    """Fit from ``start``; returns (best candidate or None, number of restarts used)."""
    start = _clip_start(np.asarray(start, dtype=float), problem.lower, problem.upper)
    best = _attempt(problem, start, "trf", max_nfev)
    if not problem.bounded and start.size <= problem.n:
        best = _better(best, _attempt(problem, start, "lm", max_nfev))
    used = 0

    def satisfactory(c: _Candidate | None) -> bool:
        return c is not None and c.converged and not np.any(c.active)

    if restarts and not satisfactory(best):
        rng = np.random.default_rng(271828)
        scale = _typical_size(start)
        for i in range(N_RESTARTS):
            used += 1
            if i % 2 == 0 or best is None:
                base = start
            else:
                base = best.theta
            factor = np.exp(rng.normal(0.0, 0.5, size=start.size))
            jitter = rng.normal(0.0, 0.1, size=start.size) * scale
            trial = _clip_start(base * factor + jitter, problem.lower, problem.upper)
            method = "trf" if (problem.bounded or i % 4 < 2 or start.size > problem.n) else "lm"
            best = _better(best, _attempt(problem, trial, method, _MAX_NFEV_RESTART))
            if satisfactory(best):
                break
    if best is not None:
        best = _polish(problem, best)
    return best, used


def _covariance(jw: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """(J'J)^-1 and the parameter dependencies from the weighted Jacobian (column-scaled SVD).

    Returns (unscaled covariance or None when rank deficient, dependency per parameter,
    rank_deficient flag). Dependency_i = 1 - 1 / ((J'J)_ii ((J'J)^-1)_ii): the fraction of
    the variance of parameter i that is due to its entanglement with the other parameters.
    """
    n, p = jw.shape
    norms = np.linalg.norm(jw, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(jw)):
        return None, None, True
    try:
        _, sv, vt = np.linalg.svd(jw / norms, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None, None, True
    if sv.size < p or sv[-1] <= sv[0] * max(n, p) * np.finfo(float).eps * 10:
        return None, None, True
    inv_scaled = (vt.T / sv**2) @ vt
    cov = inv_scaled / np.outer(norms, norms)
    dependency = 1.0 - 1.0 / np.maximum(np.diag(inv_scaled), 1.0)
    return cov, dependency, False


@dataclass
class _Fit:
    """Everything known about one fitted curve (internal)."""

    model: CurveModel
    names: tuple[str, ...]
    theta: np.ndarray                 # all parameters (free and fixed)
    free: np.ndarray                  # bool mask
    lower: np.ndarray                 # bounds of all parameters
    upper: np.ndarray
    x: np.ndarray                     # original X
    tx: np.ndarray                    # X as passed to the model function
    y: np.ndarray
    w: np.ndarray                     # weights
    fitted: np.ndarray
    ss: float
    ss_tot: float
    cov: np.ndarray | None            # covariance of the free parameters
    dependency: np.ndarray | None
    converged: bool
    status: int
    message: str
    nfev: int
    restarts: int
    offset: float | None
    at_bound: list[str]
    rank_deficient: bool
    start: dict[str, float]
    fixed: dict[str, float]
    problem: _Problem | None = None

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def p(self) -> int:
        return int(self.free.sum())

    @property
    def df(self) -> int:
        return self.n - self.p

    @property
    def s2(self) -> float | None:
        return self.ss / self.df if self.df > 0 else None

    @property
    def free_names(self) -> list[str]:
        return [n for n, f in zip(self.names, self.free) if f]

    def values(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.theta)}

    def se(self) -> dict[str, float | None]:
        out: dict[str, float | None] = dict.fromkeys(self.names)
        if self.cov is None or self.s2 is None:
            return out
        diag = np.sqrt(np.maximum(np.diag(self.cov), 0.0))
        for name, value in zip(self.free_names, diag):
            out[name] = float(value) if math.isfinite(value) else None
        return out


def _exact_ss(w: np.ndarray, y: np.ndarray) -> float:
    """Weighted residual SS at or below which a fit is exact (no scatter left to analyse).

    The larger of ``NEAR_PERFECT_RATIO`` times the weighted, *centred* total sum of squares
    and the SS that rounding alone leaves behind, n * (4 ulp of the largest weighted |y|)^2.
    Both follow the unit of Y, and neither grows with a common offset: measured against the
    uncentred sum of y^2, a fit would be called "exact" whenever the mean of Y is some 1e10
    times its scatter.
    """
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    with np.errstate(all="ignore"):
        unit = _unit_scale(y)
        ys = y / unit
        total = float(np.sum(w))
        ybar = float(np.sum(w * ys) / total) if total > 0 else 0.0
        ss_tot = float(np.sum(w * (ys - ybar) ** 2)) * unit * unit
        top = float(np.max(np.abs(np.sqrt(w) * y)))
        rounding = y.size * (4.0 * float(np.spacing(top))) ** 2 if math.isfinite(top) else 0.0
    out = max(NEAR_PERFECT_RATIO * ss_tot, rounding)
    return out if math.isfinite(out) else 0.0


def _make_problem(
    model: CurveModel, tx: np.ndarray, y: np.ndarray, w: np.ndarray, theta_full: np.ndarray,
    free: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> _Problem:
    sqrt_w = np.sqrt(w)
    base = theta_full.astype(float).copy()
    idx = np.flatnonzero(free)

    def full(theta: np.ndarray) -> np.ndarray:
        out = base.copy()
        out[idx] = theta
        return out

    def residual(theta: np.ndarray) -> np.ndarray:
        return sqrt_w * (model.func(tx, full(theta)) - y)

    def jacobian(theta: np.ndarray) -> np.ndarray:
        return sqrt_w[:, None] * model.jac(tx, full(theta))[:, idx]

    return _Problem(residual=residual, jacobian=jacobian, lower=lower[idx], upper=upper[idx],
                    n=int(y.size), exact_ss=_exact_ss(w, y), unit=_unit_scale(sqrt_w * y))


def _run_fit(
    model: CurveModel, x: np.ndarray, y: np.ndarray, w: np.ndarray, start: dict[str, float],
    fixed: dict[str, float], bounds: dict[str, tuple[float | None, float | None]],
    restarts: bool = True,
) -> _Fit:
    names = model.parameters
    tx = model.x_transform(x) if model.x_transform is not None else x
    free = np.array([name not in fixed for name in names], dtype=bool)
    theta0 = np.array([fixed.get(name, start.get(name, 1.0)) for name in names], dtype=float)
    lower = np.array([-np.inf if bounds.get(n, (None, None))[0] is None else bounds[n][0]
                      for n in names], dtype=float)
    upper = np.array([np.inf if bounds.get(n, (None, None))[1] is None else bounds[n][1]
                      for n in names], dtype=float)
    problem = _make_problem(model, tx, y, w, theta0, free, lower, upper)
    cand, used = _solve(problem, theta0[free], restarts=restarts)
    if cand is None:
        raise StatsError(
            "The curve could not be fitted: the model gives undefined values (for example a "
            "logarithm of a negative number or a division by zero) at the starting values. "
            "Provide better initial values, check that the X values suit this model, or choose "
            "a different model."
        )
    theta = theta0.copy()
    theta[free] = cand.theta
    with np.errstate(all="ignore"):
        fitted = model.func(tx, theta)
        jw = np.sqrt(w)[:, None] * model.jac(tx, theta)[:, free]
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    cov_unscaled, dependency, rank_def = _covariance(jw)
    n, p = y.size, int(free.sum())
    cov = None
    if cov_unscaled is not None and n > p:
        cov = cov_unscaled * (cand.ss / (n - p))
    free_names = [nm for nm, f in zip(names, free) if f]
    at_bound = [nm for nm, act in zip(free_names, cand.active) if act]
    return _Fit(
        model=model, names=names, theta=theta, free=free, lower=lower, upper=upper, x=x, tx=tx,
        y=y, w=w, fitted=np.asarray(fitted, dtype=float), ss=cand.ss, ss_tot=ss_tot, cov=cov,
        dependency=dependency, converged=bool(cand.converged), status=cand.status,
        message=cand.message, nfev=cand.nfev, restarts=used, offset=cand.offset,
        at_bound=at_bound, rank_deficient=rank_def, start=dict(start), fixed=dict(fixed),
        problem=problem,
    )


# ------------------------------------------------------------- profile likelihood
def _profile_interval(fit: _Fit, name: str, level: float) -> tuple[float | None, float | None, list[str]]:
    """Profile-likelihood confidence limits of one free parameter (see module docstring)."""
    notes: list[str] = []
    df = fit.df
    if df < 1 or fit.ss <= 0:
        return None, None, ["Profile-likelihood limits need residual scatter and spare degrees of freedom."]
    names = fit.names
    j_full = names.index(name)
    target = fit.ss * (1.0 + float(stats.f.ppf(level, 1, df)) / df)
    best = float(fit.theta[j_full])
    free_other = fit.free.copy()
    free_other[j_full] = False
    se = fit.se().get(name)
    t_crit = float(stats.t.ppf(0.5 + level / 2.0, df))
    if se is not None and math.isfinite(se) and se > 0:
        step0 = se * t_crit
    else:  # no standard error: start from a tenth of the estimate (unit-free); 1e-3 only if it is 0
        step0 = abs(best) * 0.1 if best != 0 else 1e-3
    huge = target * 1e6 + 1.0

    def profile_ss(value: float, warm: np.ndarray) -> tuple[float, np.ndarray]:
        theta = fit.theta.copy()
        theta[j_full] = value
        if not np.any(free_other):
            with np.errstate(all="ignore"):
                r = np.sqrt(fit.w) * (fit.model.func(fit.tx, theta) - fit.y)
            ss = float(r @ r) if np.all(np.isfinite(r)) else huge
            return ss, warm
        problem = _make_problem(fit.model, fit.tx, fit.y, fit.w, theta, free_other,
                                fit.lower, fit.upper)
        cand, _ = _solve(problem, warm, restarts=False, max_nfev=_MAX_NFEV_RESTART)
        if cand is None or not math.isfinite(cand.ss):
            return huge, warm
        return min(cand.ss, huge), cand.theta

    limits: list[float | None] = []
    for sign in (-1.0, 1.0):
        bound = fit.lower[j_full] if sign < 0 else fit.upper[j_full]
        warm = fit.theta[free_other].copy()
        prev_v = best
        found: float | None = None
        hit_bound = False
        for i in range(16):
            v = best + sign * step0 * (0.5 * 2.0**i)
            if (sign < 0 and v <= bound) or (sign > 0 and v >= bound):
                v = float(bound)
                hit_bound = True
            ss_v, warm_new = profile_ss(v, warm)
            g = ss_v - target
            if g > 0:
                state = {"warm": warm}

                def objective(value: float, state: dict[str, np.ndarray] = state) -> float:
                    ss_i, warm_i = profile_ss(value, state["warm"])
                    state["warm"] = warm_i
                    return ss_i - target

                try:
                    # tolerance relative to the size of the parameter and of the search step
                    # (an absolute 1e-12 is 0.1% of a Kd of 1e-9)
                    found = float(optimize.brentq(objective, prev_v, v,
                                                  xtol=1e-12 * max(abs(best), abs(step0)),
                                                  rtol=1e-10, maxiter=100))
                except (ValueError, RuntimeError):
                    found = None
                break
            prev_v, warm = v, warm_new
            if hit_bound:
                break
        side = "lower" if sign < 0 else "upper"
        if found is None:
            if hit_bound and math.isfinite(bound):
                found = float(bound)
                notes.append(f"{name}: the {side} confidence limit is set by the bound you imposed.")
            else:
                notes.append(
                    f"{name}: the {side} profile-likelihood limit is open-ended (the fit hardly "
                    "gets worse however far this parameter is moved), so the data do not "
                    "determine it on that side."
                )
        limits.append(found)
    return limits[0], limits[1], notes


# =============================================================== data handling
def _prepare_xy(
    x: Any, y: Any, sd: Any, names: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    try:
        xa = np.asarray(x, dtype=float).ravel()
        ya = np.asarray(y, dtype=float).ravel()
        sda = None if sd is None else np.asarray(sd, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise StatsError(f"{names[0]} and {names[1]} must be lists of numbers.") from exc
    if xa.shape != ya.shape:
        raise StatsError(
            f"X and Y must have the same number of values ({names[0]} has {xa.size}, "
            f"{names[1]} has {ya.size})."
        )
    if sda is not None and sda.shape != ya.shape:
        raise StatsError("The SD values must match the Y values one to one.")
    ok = np.isfinite(xa) & np.isfinite(ya)
    notes = []
    dropped = int((~ok).sum())
    if dropped:
        notes.append(f"{dropped} point(s) with a missing X or Y value were ignored.")
    xa, ya = xa[ok], ya[ok]
    if sda is not None:
        sda = sda[ok]
    order = np.argsort(xa, kind="stable")
    return xa[order], ya[order], (None if sda is None else sda[order]), notes


def _weights(weighting: str, y: np.ndarray, sd: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if weighting == "none":
        return np.ones_like(y), notes
    if weighting in ("1/y", "1/y^2"):
        if np.any(y == 0):
            raise StatsError(
                f"Weighting by {weighting} is impossible because some Y values are exactly zero. "
                "Use no weighting, or remove the reason for the zeros (for example a subtracted "
                "baseline)."
            )
        with np.errstate(all="ignore"):
            w = 1.0 / np.abs(y) if weighting == "1/y" else 1.0 / (y * y)
        if not np.all(np.isfinite(w)) or float(np.max(w)) > 1e200:
            raise StatsError(
                f"Weighting by {weighting} is impossible because some Y values are too close to "
                "zero. Use no weighting."
            )
        notes.append(
            f"Points were weighted by {weighting} using the observed Y values "
            + ("(Poisson-like scatter: variance proportional to Y)." if weighting == "1/y"
               else "(relative weighting: scatter proportional to Y, i.e. a constant % error).")
        )
        return w, notes
    if sd is None:
        raise StatsError(
            "Weighting by 1/SD^2 needs the standard deviation of each point (replicates at "
            "every X)."
        )
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise StatsError(
            "Weighting by 1/SD^2 needs a positive SD for every point. Some X values have a "
            "single replicate or identical replicates; use a different weighting."
        )
    notes.append("Points were weighted by 1/SD^2 (the SD of the replicates at each X).")
    return 1.0 / (sd * sd), notes


def _looks_log_spaced(ux: np.ndarray) -> bool:
    pos = ux[ux > 0]
    if pos.size < 3 or np.any(ux < 0) or pos[-1] / pos[0] < 10:
        return False
    lin = np.diff(pos)
    log = np.diff(np.log10(pos))
    cv_lin = float(np.std(lin) / np.mean(lin))
    cv_log = float(np.std(log) / np.mean(log))
    return cv_log < cv_lin


def _curve_grid(model: CurveModel, x: np.ndarray) -> tuple[np.ndarray, bool]:
    ux = np.unique(x)
    lo, hi = float(ux[0]), float(ux[-1])
    if lo == hi:
        return np.array([lo]), False
    log_spaced = (model.family == "dose_response" and not model.x_is_log) or _looks_log_spaced(ux)
    pos = ux[ux > 0]
    if log_spaced and pos.size >= 2:
        grid = np.geomspace(float(pos[0]), float(pos[-1]), GRID_POINTS)
        if lo <= 0:
            grid = np.concatenate([ux[ux <= 0], grid])
        return grid, True
    return np.linspace(lo, hi, GRID_POINTS), False


# ================================================================== fit_curve
def _display(name: str, kind: str) -> str:
    if kind == "IC50":
        return {"LogEC50": "LogIC50", "EC50": "IC50"}.get(name, name)
    return name


def _display_equation(text: str, kind: str) -> str:
    return text.replace("EC50", "IC50") if kind == "IC50" else text


def _display_summary(text: str, kind: str) -> str:
    """Model summary with the fitted quantity named IC50 when that label was asked for
    (leaves 'stimulation (EC50) and inhibition (IC50)' style enumerations alone)."""
    if kind != "IC50":
        return text
    return text.replace("log10(EC50)", "log10(IC50)").replace("keeps EC50", "keeps IC50")


def _resolve_direction(model: CurveModel, direction: str, x: np.ndarray, y: np.ndarray,
                       fixed_user: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Default fixed values of the model (resolves the sign of the 3-parameter Hill slope)."""
    notes: list[str] = []
    if direction not in {"auto", "stimulation", "inhibition"}:
        raise StatsError("direction must be 'auto', 'stimulation' or 'inhibition'.")
    fixed = dict(model.default_fixed)
    if "HillSlope" in fixed and "HillSlope" not in fixed_user:
        if direction == "auto":
            ux, my, _ = mean_by_x(x, y)
            half = ux.size // 2
            rising = True
            if ux.size >= 2:
                rising = float(np.mean(my[-max(half, 1):])) >= float(np.mean(my[:max(half, 1)]))
            sign = 1.0 if rising else -1.0
            notes.append(
                f"The Hill slope was fixed at {sign:+.0f} because the response "
                f"{'rises' if rising else 'falls'} as X increases (direction chosen automatically)."
            )
        else:
            sign = 1.0 if direction == "stimulation" else -1.0
        fixed["HillSlope"] = sign
    return fixed, notes


@dataclass
class _Setup:
    model: CurveModel
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    sd: np.ndarray | None
    start: dict[str, float]
    fixed: dict[str, float]
    bounds: dict[str, tuple[float | None, float | None]]
    notes: list[str]
    warnings: list[str]


def _setup(
    x: Any, y: Any, model: str | CurveModel, *, equation: Any, initial: Any, fixed: Any,
    bounds: Any, weighting: str, sd: Any, direction: str, names: tuple[str, str],
) -> _Setup:
    if weighting not in WEIGHTINGS:
        raise StatsError(f"weighting must be one of: {', '.join(WEIGHTINGS)}.")
    if isinstance(equation, str) and not equation.strip():
        equation = None
    if not isinstance(model, CurveModel) and str(model).strip() != "user_defined" and equation is not None:
        raise StatsError(
            "An equation was given but the chosen model is not 'user_defined'. Choose the "
            "'user_defined' model to fit your own equation, or clear the equation box."
        )
    cm = get_model(model, equation)
    xa, ya, sda, notes = _prepare_xy(x, y, sd, names)
    notes.extend(cm.notes)  # e.g. "'e' was fitted as a parameter, not read as Euler's number"
    warns: list[str] = []
    fixed_user = _resolve_values(cm, fixed, "Fixed values")
    fixed_default, dir_notes = _resolve_direction(cm, direction, xa, ya, fixed_user) if xa.size else ({}, [])
    notes.extend(dir_notes)
    fixed_all = {**fixed_default, **fixed_user}
    n_free = cm.n_parameters - len(fixed_all)
    if n_free < 1:
        raise StatsError("Every parameter is fixed, so there is nothing left to fit.")
    if xa.size < n_free:
        raise StatsError(
            f"This fit estimates {n_free} parameter(s) but only {xa.size} usable point(s) were "
            f"given. At least {n_free} are needed (and more to get standard errors)."
        )
    if np.unique(xa).size < 2 and (xa.size > 1 or n_free > 1):
        raise StatsError(
            "All X values are identical, so a curve cannot be fitted. The data need several "
            "different X values."
        )
    if cm.family == "dose_response":
        if not cm.x_is_log:
            if np.any(xa < 0):
                raise StatsError(
                    "This model expects X to be a concentration (zero or positive), but some X "
                    "values are negative. If your X values are already log10(concentration), "
                    f"choose '{cm.key}_logx' instead."
                )
            if np.unique(xa[xa > 0]).size < 2:
                raise StatsError(
                    "A dose-response curve needs at least two different non-zero concentrations."
                )
            if np.any(xa == 0):
                notes.append(
                    "Points at X = 0 (no drug) were included; the curve approaches a plateau "
                    "there, so they help to define it."
                )
        elif np.all(xa > 0) and float(xa.max()) / float(xa.min()) >= 1000:
            warns.append(
                "The X values span more than three orders of magnitude and are all positive: "
                "they look like concentrations, but this model expects log10(concentration). "
                f"Either transform X or choose the model '{cm.key.replace('_logx', '')}'."
            )
    w, w_notes = _weights(weighting, ya, sda)
    notes.extend(w_notes)
    start_user = _resolve_values(cm, initial, "Initial values")
    overlap = sorted(set(start_user) & set(fixed_user))
    if overlap:
        warns.append(f"Initial value(s) for {', '.join(overlap)} were ignored because the "
                     "parameter is fixed.")
    try:
        tx = cm.x_transform(xa) if cm.x_transform is not None else xa
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = {k: float(v) for k, v in cm.guess(tx, ya, fixed_all).items()}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError,
            OverflowError):
        start = dict.fromkeys(cm.parameters, 1.0)
    for name in cm.parameters:
        if not math.isfinite(start.get(name, math.nan)):
            start[name] = 1.0
    if cm.key == "user_defined":
        missing = [n for n in cm.parameters if n not in start_user and n not in fixed_all]
        if missing:
            warns.append(
                f"No initial value was given for {', '.join(missing)}, so 1.0 was used. A "
                "user-defined equation has no automatic starting values, and a poor start can "
                "lead to a failed or wrong fit: give rough estimates under 'Initial values'."
            )
    start.update(start_user)
    bounds_all = _resolve_bounds(cm, bounds)
    for name, value in fixed_all.items():
        lo, hi = bounds_all.get(name, (None, None))
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            warns.append(f"The fixed value of {name} lies outside its bounds; the bounds were "
                         "ignored for this parameter.")
    return _Setup(model=cm, x=xa, y=ya, w=w, sd=sda, start=start, fixed=fixed_all,
                  bounds=bounds_all, notes=notes, warnings=warns)


def _goodness(fit: _Fit) -> dict[str, Any]:
    n, p, df = fit.n, fit.p, fit.df
    perfect = fit.ss <= _exact_ss(fit.w, fit.y)
    r2 = None
    if fit.ss_tot > 0:
        r2 = 1.0 - fit.ss / fit.ss_tot
    adj = None
    if r2 is not None and df > 0 and n > 1:
        adj = 1.0 - (fit.ss / df) / (fit.ss_tot / (n - 1))
    ic = information_criteria(fit.ss, n, p)
    return {
        "n": n, "n_parameters": p, "df": df, "ss": fit.ss, "ss_total": fit.ss_tot,
        "r_squared": r2, "adj_r_squared": adj,
        "sy_x": math.sqrt(fit.ss / df) if df > 0 else None,
        "aic": ic["aic"], "aicc": ic["aicc"], "bic": ic["bic"], "k": ic["k"],
        "perfect_fit": bool(perfect),
    }


def _parameter_table(
    fit: _Fit, ci_level: float, ci_method: str, kind: str
) -> tuple[list[Parameter], list[str], dict[str, tuple[float | None, float | None]]]:
    """Parameters (fitted, fixed, derived) plus notes; also returns the CI per internal name."""
    model = fit.model
    values = fit.values()
    se = fit.se()
    df = fit.df
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df)) if df > 0 else None
    notes: list[str] = []
    cis: dict[str, tuple[float | None, float | None]] = {}
    params: list[Parameter] = []
    for name in fit.names:
        label = _display(name, kind)
        if name in fit.fixed:
            params.append(Parameter(name=label, value=values[name], ci_level=ci_level,
                                    unit_note="Fixed at this value (not fitted)"))
            cis[name] = (None, None)
            continue
        lo = hi = None
        note = _display_equation(model.descriptions.get(name, ""), kind)
        if se[name] is not None and t_crit is not None:
            if ci_method == "profile":
                lo, hi, p_notes = _profile_interval(fit, name, ci_level)
                notes.extend(p_notes)
                note = (note + " " if note else "") + "CI: profile likelihood."
            else:
                lo, hi = values[name] - t_crit * se[name], values[name] + t_crit * se[name]
        cis[name] = (lo, hi)
        params.append(Parameter(name=label, value=values[name], se=se[name], ci_low=lo, ci_high=hi,
                                ci_level=ci_level, unit_note=note or None))
    free_names = fit.free_names
    for d in model.derived:
        label = _display(d.name, kind)
        source_values = {s: values[s] for s in d.sources}
        if d.needs_positive and any(v <= 0 for v in source_values.values()):
            params.append(Parameter(
                name=label, value=None, ci_level=ci_level,
                unit_note=f"Undefined because {d.sources[0]} is not positive"))
            continue
        try:
            value = float(d.value(values))
        except (OverflowError, ZeroDivisionError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            params.append(Parameter(name=label, value=None, ci_level=ci_level,
                                    unit_note="Could not be computed (out of numeric range)"))
            continue
        d_se = None
        if fit.cov is not None:
            try:
                grad = d.gradient(values)
                g = np.array([grad.get(n, 0.0) for n in free_names], dtype=float)
            except (OverflowError, ZeroDivisionError, ValueError):  # pragma: no cover
                g = np.full(len(free_names), math.nan)
            if np.all(np.isfinite(g)):
                var = float(g @ fit.cov @ g)
                if var >= 0 and math.isfinite(var):
                    d_se = math.sqrt(var)
        lo = hi = None
        if all(s in fit.fixed for s in d.sources):
            note = "Computed from fixed values"
            d_se = None
        elif d.monotone:
            src_lo, src_hi = cis[d.sources[0]]
            ends = []
            for end in (src_lo, src_hi):
                if end is None or (d.needs_positive and end <= 0):
                    ends.append(None)
                    continue
                try:
                    v = float(d.value({**values, d.sources[0]: end}))
                except (OverflowError, ZeroDivisionError, ValueError):
                    v = math.nan
                ends.append(v if math.isfinite(v) else None)
            if d.needs_positive:  # decreasing transform (1 / K): the limits swap
                lo, hi = ends[1], ends[0]
            else:
                lo, hi = ends[0], ends[1]
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
            note = d.note + ". SE by the delta method"
            if d.needs_positive and src_lo is not None and src_lo <= 0:
                note += (f". The CI of {d.sources[0]} includes zero, so there is no finite upper "
                         "limit")
        else:
            if d_se is not None and t_crit is not None:
                lo, hi = value - t_crit * d_se, value + t_crit * d_se
            note = d.note
        params.append(Parameter(name=label, value=value, se=d_se, ci_low=lo, ci_high=hi,
                                ci_level=ci_level, unit_note=_display_equation(note, kind)))
    return params, notes, cis


def _plain(values: Any) -> list[float | None]:
    """List of floats for JSON, with undefined (NaN / infinite) entries as ``None``."""
    arr = np.asarray(values, dtype=float).ravel()
    return [float(v) if math.isfinite(v) else None for v in arr]


def _bands(fit: _Fit, grid: np.ndarray, ci_level: float, weighting: str) -> dict[str, Any]:
    model = fit.model
    tgrid = model.x_transform(grid) if model.x_transform is not None else grid
    with np.errstate(all="ignore"):
        y_grid = np.asarray(model.func(tgrid, fit.theta), dtype=float)
    curve: dict[str, Any] = {"x": grid.tolist(), "y": _plain(y_grid)}
    if fit.cov is None or fit.df < 1 or fit.s2 is None or not np.all(np.isfinite(fit.cov)):
        return curve
    with np.errstate(all="ignore"):
        g = model.jac(tgrid, fit.theta)[:, fit.free]
        g = np.where(np.isfinite(g), g, 0.0)
        var = np.einsum("ij,jk,ik->i", g, fit.cov, g)
    var = np.maximum(var, 0.0)
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, fit.df))
    half = t_crit * np.sqrt(var)
    curve["ci_low"] = _plain(y_grid - half)
    curve["ci_high"] = _plain(y_grid + half)
    if weighting == "none":
        point_var = np.full_like(y_grid, fit.s2)
    elif weighting == "1/y":
        point_var = fit.s2 * np.abs(y_grid)
    elif weighting == "1/y^2":
        point_var = fit.s2 * y_grid**2
    else:
        return curve
    half_pred = t_crit * np.sqrt(var + point_var)
    curve["pred_low"] = _plain(y_grid - half_pred)
    curve["pred_high"] = _plain(y_grid + half_pred)
    return curve


def _scatter_check(fit: _Fit, weighting: str) -> AssumptionCheck:
    """Does the size of the (weighted) residuals change with the predicted Y?"""
    name = "Constant scatter (appropriate weighting)"
    resid = np.sqrt(fit.w) * (fit.y - fit.fitted)
    if fit.n < 6 or is_constant(fit.fitted) or is_constant(np.abs(resid)):
        return AssumptionCheck(name=name, status="info",
                               detail="Too few points to test whether the scatter changes along "
                                      "the curve. Look at the residual plot.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(np.abs(resid), np.abs(fit.fitted))
    if not math.isfinite(p):
        return AssumptionCheck(name=name, status="info",
                               detail="The scatter test could not be computed.")
    detail = (f"Spearman correlation between the size of the residuals and the predicted Y: "
              f"rho = {fmt(rho, 3)}, {p_phrase(p)}.")
    if p < 0.05:
        advice = (" The scatter grows with Y: consider relative weighting (1/Y^2)."
                  if rho > 0 and weighting == "none" else
                  " The scatter still changes along the curve; the weighting may not suit the data.")
        return AssumptionCheck(name=name, statistic=rho, p_value=p, status="warn",
                               detail=detail + advice)
    return AssumptionCheck(name=name, statistic=rho, p_value=p, status="pass",
                           detail=detail + " No evidence that the scatter changes along the curve.")


def _identifiability(
    fit: _Fit, cis: dict[str, tuple[float | None, float | None]], kind: str
) -> tuple[AssumptionCheck, list[str]]:
    """Checks that the data actually determine every parameter."""
    model = fit.model
    values = fit.values()
    se = fit.se()
    problems: list[str] = []
    ec = _display("EC50", kind)
    for name in fit.at_bound:
        problems.append(
            f"{_display(name, kind)} stopped at the bound you set ({fmt(values[name])}): the data "
            "would prefer a value outside the allowed range, and its standard error is not "
            "meaningful."
        )
    if fit.rank_deficient:
        problems.append(
            "The parameters are not all identifiable: different combinations of values fit the "
            "data equally well, so no standard errors can be given. Fix a parameter to a "
            "constant, collect data over a wider X range, or use a simpler model."
        )
    elif fit.dependency is not None:
        for name, dep in zip(fit.free_names, fit.dependency):
            if dep > 0.9999:
                problems.append(
                    f"{_display(name, kind)} is almost completely entangled with the other "
                    f"parameters (dependency {dep:.5f}): its value is ambiguous."
                )
    for name in fit.free_names:
        s = se[name]
        if s is None or name in fit.at_bound:
            continue
        label = _display(name, kind)
        if name == "LogEC50":
            lo, hi = cis.get(name, (None, None))
            width = (hi - lo) if (lo is not None and hi is not None) else 2 * 1.96 * s
            if width > 2.0:
                problems.append(
                    f"The confidence interval of {label} spans more than two log units "
                    f"(a 100-fold range of {ec}): the {ec} is poorly determined."
                )
        elif name in model.positive or name == "HillSlope":
            if values[name] != 0 and s > abs(values[name]):
                problems.append(
                    f"The standard error of {label} is larger than its value "
                    f"({fmt(values[name])} +/- {fmt(s)}): it is poorly determined by these data."
                )
    for name in model.positive:
        if name in fit.free_names and values[name] <= 0:
            word = "negative" if values[name] < 0 else "zero"
            if model.family == "exponential":
                problems.append(
                    f"{_display(name, kind)} came out {word} ({fmt(values[name])}): the data run "
                    "the opposite way to what this model describes (for example decay instead of "
                    "growth), so half-life / doubling time are undefined. Choose the matching model."
                )
            else:
                problems.append(
                    f"{_display(name, kind)} came out {word} ({fmt(values[name])}), which has no "
                    "physical meaning for this model. The data may not follow this model, or may "
                    "not cover enough of the curve."
                )
    if model.family == "dose_response":
        finite = np.isfinite(fit.tx)
        lo_x, hi_x = float(np.min(fit.tx[finite])), float(np.max(fit.tx[finite]))
        logec = values["LogEC50"]
        if "LogEC50" in fit.free_names and (logec < lo_x or logec > hi_x):
            side = "below the lowest" if logec < lo_x else "above the highest"
            problems.append(
                f"The {ec} is extrapolated: its best-fit value ({fmt(10.0 ** logec) if abs(logec) < 300 else 'out of range'}) "
                f"lies {side} concentration tested, so it depends on a plateau the data do not "
                "show. Treat it as a rough estimate at best."
            )
        span = values["Top"] - values["Bottom"]
        if span != 0 and np.all(np.isfinite(fit.fitted)):
            frac = (fit.fitted - values["Bottom"]) / span
            ux = np.unique(fit.x)
            frac_by_x = np.array([float(np.mean(frac[fit.x == u])) for u in ux])
            for pname, mask in (("Top", frac_by_x >= 0.9), ("Bottom", frac_by_x <= 0.1)):
                if pname in fit.free_names and int(mask.sum()) < 2:
                    count = int(mask.sum())
                    problems.append(
                        f"The {pname} plateau is not well defined: only {count} of the tested "
                        f"X values lie{'s' if count == 1 else ''} within 10% of it. Consider "
                        f"fixing {pname} to a control value (for example 0 or 100 for normalised "
                        "data) or testing a wider concentration range."
                    )
        if "HillSlope" in fit.free_names and "Top" in fit.free_names and "Bottom" in fit.free_names \
                and values["Top"] < values["Bottom"]:
            problems.append(
                "Top came out below Bottom: the curve is the mirror image of the usual "
                "convention (swap Top and Bottom and change the sign of the Hill slope)."
            )
    name = "Parameters well determined (identifiability)"
    if problems:
        return AssumptionCheck(name=name, status="warn", detail=" ".join(problems)), problems
    return AssumptionCheck(
        name=name, status="pass",
        detail="No parameter is at a bound, entangled with the others, or extrapolated beyond "
               "the data."), problems


def _replicate_summary(x: np.ndarray) -> tuple[int, int]:
    _, counts = np.unique(x, return_counts=True)
    return int(counts.size), int(counts.max())


def _fit_words(fit: _Fit, params: list[Parameter], gof: dict[str, Any], kind: str,
               ci_level: float, names: tuple[str, str]) -> str:
    model = fit.model
    by_name = {p.name: p for p in params}

    def describe(name: str) -> str:
        par = by_name.get(name)
        if par is None or par.value is None:
            return f"{name} could not be estimated"
        text = f"{name} = {fmt(par.value)}"
        if par.ci_low is not None and par.ci_high is not None:
            text += f" ({ci_phrase(par.ci_low, par.ci_high, ci_level)})"
        elif par.unit_note and par.unit_note.startswith("Fixed"):
            text += " (fixed)"
        return text

    r2 = gof["r_squared"]
    r2_text = (f" R squared = {fmt(r2, 4)}, so the curve accounts for about {fmt(100 * r2, 3)}% of "
               f"the variation in {names[1]}." if r2 is not None else "")
    if model.family == "dose_response":
        ec = _display("EC50", kind)
        par = by_name.get(ec)
        hill = fit.values()["HillSlope"]
        unit = ("in the units of your X values" if not model.x_is_log else
                "in the concentration units whose log10 you entered as X")
        ec_text = f"The best-fit {ec} is {fmt(par.value)}" if par and par.value is not None else \
            f"The {ec} could not be estimated"
        if par is not None and par.ci_low is not None and par.ci_high is not None:
            ec_text += f" ({ci_phrase(par.ci_low, par.ci_high, ci_level)})"
        ec_text += f", {unit}: the concentration giving a response halfway between Bottom and Top."
        if "HillSlope" in fit.fixed:
            hill_text = (f" The Hill slope was fixed at {fmt(hill)} "
                         f"({'response rises' if hill > 0 else 'response falls'} with dose).")
        else:
            steep = ("close to the standard slope of 1" if 0.8 <= abs(hill) <= 1.25 else
                     "steeper than the standard slope of 1" if abs(hill) > 1.25 else
                     "shallower than the standard slope of 1")
            hill_text = (f" The Hill slope is {fmt(hill)}: the response "
                         f"{'rises' if hill > 0 else 'falls'} as the dose increases, {steep}.")
        plateau = f" The curve runs between {describe('Bottom')} and {describe('Top')}."
        return ec_text + hill_text + plateau + r2_text
    listing = "; ".join(describe(_display(n, kind)) for n in fit.names)
    text = f"The best-fit values for the {model.label.lower()} model are: {listing}."
    for extra in ("HalfLife", "DoublingTime"):
        par = by_name.get(extra)
        if par is not None and par.value is not None:
            label = "half-life" if extra == "HalfLife" else "doubling time"
            text += f" The {label} is {fmt(par.value)}"
            if par.ci_low is not None and par.ci_high is not None:
                text += f" ({ci_phrase(par.ci_low, par.ci_high, ci_level)})"
            text += ", in the units of your X values."
    return text + r2_text


def _build_result(
    setup: _Setup, fit: _Fit, *, alpha: float, ci_level: float, ci_method: str, weighting: str,
    kind: str, names: tuple[str, str],
) -> StatResult:
    model = fit.model
    warns = list(setup.warnings)
    notes = list(setup.notes)
    gof = _goodness(fit)
    perfect = gof["perfect_fit"]
    params, p_notes, cis = _parameter_table(fit, ci_level, ci_method, kind)
    warns.extend(p_notes)
    resid = fit.y - fit.fitted
    wresid = np.sqrt(fit.w) * resid
    n_x, max_rep = _replicate_summary(fit.x)
    has_reps = n_x < fit.n

    # ---- convergence
    if not fit.converged:
        warns.insert(0, (
            "The fit did NOT converge to a clear best-fit solution (the optimiser stopped with: "
            f"{fit.message}). The reported values may be far from the best fit. Try better "
            "initial values, fix a parameter, or use a simpler model."
        ))
    if fit.df == 0:
        warns.append(
            "There are exactly as many points as fitted parameters, so the curve passes through "
            "every point and no standard errors, confidence intervals or goodness-of-fit "
            "measures can be computed."
        )
    elif perfect:
        warns.append(
            "The curve passes through the points essentially exactly (perfect fit). Standard "
            "errors are then meaninglessly small; real measurements always scatter, so check "
            "that the Y values were not generated from the model."
        )
    if is_constant(fit.y):
        warns.append("All Y values are identical: there is no curve to fit and the parameters "
                     "are arbitrary.")
    if fit.df > 0 and gof["aicc"] is None and not perfect:
        warns.append("AICc is undefined with this few points per parameter (it needs "
                     "N > number of parameters + 2).")
    if gof["r_squared"] is not None and gof["r_squared"] < 0:
        warns.append(
            "R squared is negative: the curve fits worse than a horizontal line through the "
            "mean of Y. This happens when the model (or a fixed parameter / bound) does not "
            "suit the data."
        )

    # ---- assumptions / diagnostics
    assumptions: list[AssumptionCheck] = [AssumptionCheck(
        name="Convergence", statistic=fit.offset,
        status="pass" if fit.converged else "fail",
        detail=("The fit converged" if fit.converged else "The fit did not converge")
               + (f" (relative offset {fit.offset:.1e}; values below {OFFSET_TOL:g} indicate a "
                  "true least-squares solution)" if fit.offset is not None else "")
               + (f" after {fit.restarts} restart(s) from perturbed starting values."
                  if fit.restarts else "."))]
    ident_check, ident_problems = _identifiability(fit, cis, kind)
    warns.extend(ident_problems)
    assumptions.append(ident_check)
    runs = lof = None
    if perfect or fit.df < 1:
        assumptions.append(AssumptionCheck(
            name="Residual diagnostics", status="warn",
            detail="The curve passes through the points exactly, so there are no residuals to "
                   "check for normality or systematic deviation."))
    else:
        assumptions.append(residual_normality(wresid))
        if has_reps:
            _, run_resid, _ = mean_by_x(fit.x, wresid)
        else:
            run_resid = wresid
        runs = runs_test(run_resid)
        run_check = runs_assumption(runs, "curve")
        if has_reps:
            run_check.detail += " (Computed on the mean residual at each X.)"
        lof = lack_of_fit(fit.x, fit.y, fit.fitted, fit.p, weights=fit.w)
        lof_check = lack_of_fit_assumption(lof, "curve")
        if lof_check is not None:
            assumptions.append(lof_check)
        assumptions.append(run_check)
        assumptions.append(_scatter_check(fit, weighting))
    assumptions.append(AssumptionCheck(
        name="Independent observations", status="info",
        detail="Each point must be an independent measurement. Replicates that are repeat "
               "readings of one sample are not independent: fit their mean instead."))
    assumptions.append(AssumptionCheck(
        name="X measured without error", status="info",
        detail="Least squares assumes X (dose, time, concentration) is known precisely and all "
               "the scatter is in Y."))

    # ---- curve, residuals, correlation
    grid, log_spaced = _curve_grid(model, fit.x)
    curve = _bands(fit, grid, ci_level, weighting)
    free_labels = [_display(n, kind) for n in fit.free_names]
    corr = None
    if fit.cov is not None:
        sd = np.sqrt(np.maximum(np.diag(fit.cov), 0.0))
        with np.errstate(all="ignore"):
            corr_m = fit.cov / np.outer(sd, sd)
        corr_m = np.where(np.isfinite(corr_m), np.clip(corr_m, -1.0, 1.0), np.nan)
        corr = {"labels": free_labels,
                "matrix": [[None if math.isnan(v) else float(v) for v in row] for row in corr_m]}
    dependency = None
    if fit.dependency is not None:
        dependency = {lab: float(d) for lab, d in zip(free_labels, fit.dependency)}

    equation = _display_equation(model.equation, kind)
    r2 = gof["r_squared"]
    effect_sizes = [EffectSize(name="R squared", value=r2, ci_level=ci_level)]
    if gof["adj_r_squared"] is not None:
        effect_sizes.append(EffectSize(name="adjusted R squared", value=gof["adj_r_squared"],
                                       ci_level=ci_level))
    citations = [_CITE_BW, _CITE_OFFSET, _CITE_AICC, _CITE_RUNS]
    if model.family == "dose_response":
        citations.insert(1, _CITE_4PL)
    if ci_method == "profile":
        citations.append(_CITE_PROFILE)
    result = StatResult(
        analysis="nonlinear_regression",
        test_name=f"Nonlinear regression: {model.label}",
        why_chosen=(_display_summary(model.summary, kind) + " " if model.summary else "")
                   + "Nonlinear regression finds the parameter values that bring the curve as "
                     "close as possible to the points (least squares).",
        assumptions=assumptions,
        df=float(fit.df),
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=params,
        n={names[1]: fit.n},
        descriptives=[{
            "group": names[1], "n": fit.n, "n_x_values": n_x, "max_replicates": max_rep,
            "x_min": float(fit.x.min()), "x_max": float(fit.x.max()),
            "y_min": float(fit.y.min()), "y_max": float(fit.y.max()),
            "mean": float(fit.y.mean()),
        }],
        warnings=warns,
        data_notes=notes,
        citations=citations,
        extra={
            "curve": curve,
            "residuals": {"x": fit.x.tolist(), "fitted": fit.fitted.tolist(),
                          "residual": resid.tolist(),
                          **({"weighted_residual": wresid.tolist()} if weighting != "none" else {})},
            "equation": equation,
            "model": model.key,
            "model_label": model.label,
            "x_is_log": bool(model.x_is_log),
            "x_log_spaced": bool(log_spaced),
            "goodness_of_fit": gof,
            "convergence": {
                "converged": bool(fit.converged), "status": fit.status, "message": fit.message,
                "function_evaluations": fit.nfev, "restarts": fit.restarts,
                "relative_offset": fit.offset, "parameters_at_bound": [
                    _display(n, kind) for n in fit.at_bound],
            },
            "parameter_correlation": corr,
            "parameter_dependency": dependency,
            "parameter_descriptions": {_display(k, kind): _display_equation(v, kind)
                                       for k, v in model.descriptions.items()},
            "initial_values": {_display(k, kind): float(v) for k, v in fit.start.items()
                               if k not in fit.fixed},
            "fixed": {_display(k, kind): float(v) for k, v in fit.fixed.items()},
            "bounds": {_display(n, kind): [None if not math.isfinite(lo) else float(lo),
                                           None if not math.isfinite(hi) else float(hi)]
                       for n, lo, hi in zip(fit.names, fit.lower, fit.upper)
                       if math.isfinite(lo) or math.isfinite(hi)},
            "weighting": weighting,
            "ci_method": ci_method,
            "runs_test": runs,
            "lack_of_fit": lof,
            "x_name": names[0],
            "y_name": names[1],
        },
    )
    text = _fit_words(fit, params, gof, kind, ci_level, names)
    cautions = []
    if not fit.converged:
        cautions.append("the fit did not converge, so these values are unreliable")
    if ident_check.status != "pass":
        cautions.append("at least one parameter is poorly determined by the data (see the warnings)")
    failing = [a.name for a in assumptions
               if a.status == "fail" and a.name not in ("Convergence",)]
    if failing:
        cautions.append("the diagnostics flag a problem (" + "; ".join(failing) + ")")
    if cautions:
        text += " Caution: " + "; ".join(cautions) + "."
    result.interpretation = text
    ci_words = ("profile-likelihood" if ci_method == "profile" else "asymptotic (t-based)")
    result.methods_text = (
        f"{names[1]} was fitted as a function of {names[0]} by nonlinear least-squares regression "
        f"({model.label}: {equation})"
        + (f" with {weighting} weighting" if weighting != "none" else "")
        + (f", with {', '.join(f'{_display(k, kind)} fixed at {fmt(v)}' for k, v in fit.fixed.items())}"
           if fit.fixed else "")
        + f"; {ci_level * 100:g}% {ci_words} confidence intervals are reported."
    )
    return result


def _check_kind(kind: str) -> str:
    if str(kind).upper() not in {"EC50", "IC50"}:
        raise StatsError("kind must be 'EC50' or 'IC50' (it only changes the parameter labels).")
    return str(kind).upper()


@guarded
def fit_curve(
    x: Any,
    y: Any,
    model: str | CurveModel = "dose_response_4pl_logx",
    *,
    equation: str | Equation | None = None,
    initial: Any = None,
    fixed: Any = None,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    weighting: str = "none",
    sd: Any = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    ci_method: str = "asymptotic",
    kind: str = "EC50",
    direction: str = "auto",
    names: tuple[str, str] = ("X", "Y"),
) -> StatResult:
    """Fit a curve to X-Y data by nonlinear least squares.

    Parameters
    ----------
    model:
        A key of :data:`MODELS`, or ``"user_defined"`` together with ``equation``.
    initial, fixed:
        ``{"name": value}`` dicts (or text such as ``"Top=100, Bottom=0"``). Fixed
        parameters are held constant. For dose-response models ``EC50`` / ``IC50`` may be
        given instead of ``LogEC50``. Built-in models find starting values automatically.
    bounds:
        ``{"HillSlope": (0, None)}``: lower/upper limits (``None`` = unlimited).
    weighting:
        ``"none"``; ``"1/y"`` or ``"1/y^2"`` (weights from the *observed* Y values);
        ``"1/sd^2"`` (needs ``sd``, one SD per point).
    ci_method:
        ``"asymptotic"`` (estimate +/- t * SE) or ``"profile"`` (profile likelihood).
    kind:
        ``"EC50"`` or ``"IC50"``: only changes the labels of dose-response parameters.
    direction:
        For the three-parameter dose-response models: ``"stimulation"`` (Hill slope +1),
        ``"inhibition"`` (-1) or ``"auto"`` (decided from the trend of the data).
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    kind = _check_kind(kind)
    if ci_method not in CI_METHODS:
        raise StatsError("ci_method must be 'asymptotic' or 'profile'.")
    setup = _setup(x, y, model, equation=equation, initial=initial, fixed=fixed, bounds=bounds,
                   weighting=weighting, sd=sd, direction=direction, names=names)
    fit = _run_fit(setup.model, setup.x, setup.y, setup.w, setup.start, setup.fixed, setup.bounds)
    return _build_result(setup, fit, alpha=alpha, ci_level=ci_level, ci_method=ci_method,
                         weighting=weighting, kind=kind, names=names)


# =============================================================== compare_fits
def _fixed_signature(model: CurveModel, fixed: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Express a model as (base model key, all constrained parameters) for nesting checks."""
    if model.nested_in and model.key.startswith("dose_response_3pl"):
        return model.nested_in[0], dict(fixed)
    return model.key, dict(fixed)


def _nesting(simple: _Fit, complex_: _Fit) -> bool | None:
    """True / False when nesting can be decided from the library, None when unknown."""
    if simple.model.key == "user_defined" or complex_.model.key == "user_defined":
        return None
    base_s, fixed_s = _fixed_signature(simple.model, simple.fixed)
    base_c, fixed_c = _fixed_signature(complex_.model, complex_.fixed)
    if base_s == base_c:
        if not set(fixed_c) < set(fixed_s):
            return False
        return all(math.isclose(fixed_c[k], fixed_s[k], rel_tol=1e-12, abs_tol=1e-12)
                   for k in fixed_c)
    chain = simple.model.nested_in
    if chain and chain[0] == base_c:
        # the simple model is the complex one with chain[1:] constrained; the complex model
        # must leave those (and everything else) free
        return not fixed_c and not (set(fixed_s) - set(simple.model.parameters))
    return False


def _fit_summary(fit: _Fit, kind: str, ci_level: float) -> dict[str, Any]:
    gof = _goodness(fit)
    params, _, _ = _parameter_table(fit, ci_level, "asymptotic", kind)
    return {
        "model": fit.model.key, "label": fit.model.label,
        "equation": _display_equation(fit.model.equation, kind),
        "n_parameters": fit.p, "df": fit.df, "ss": fit.ss, "r_squared": gof["r_squared"],
        "sy_x": gof["sy_x"], "aicc": gof["aicc"], "converged": bool(fit.converged),
        "fixed": {_display(k, kind): float(v) for k, v in fit.fixed.items()},
        "parameters": [{"name": p.name, "value": p.value, "se": p.se, "ci_low": p.ci_low,
                        "ci_high": p.ci_high} for p in params],
    }


def _akaike(aicc_simple: float | None, aicc_complex: float | None) -> dict[str, Any]:
    """AICc difference (complex - simple), Akaike weights and evidence ratio."""
    if aicc_simple is None or aicc_complex is None:
        return {"delta_aicc": None, "probability_simple": None, "probability_complex": None,
                "evidence_ratio": None}
    delta = aicc_complex - aicc_simple
    half = max(min(0.5 * delta, 700.0), -700.0)
    # w_i = exp(-delta_i / 2) / sum(exp(-delta_j / 2)); each computed directly to avoid
    # cancellation when one weight is very close to 1
    p_complex = 1.0 / (1.0 + math.exp(half))
    p_simple = 1.0 / (1.0 + math.exp(-half))
    return {"delta_aicc": delta, "probability_simple": p_simple,
            "probability_complex": p_complex, "evidence_ratio": math.exp(abs(half))}


@guarded
def compare_fits(
    x: Any,
    y: Any,
    model_simple: str | CurveModel,
    model_complex: str | CurveModel,
    *,
    equation_simple: str | None = None,
    equation_complex: str | None = None,
    initial_simple: Any = None,
    initial_complex: Any = None,
    fixed_simple: Any = None,
    fixed_complex: Any = None,
    bounds_simple: Any = None,
    bounds_complex: Any = None,
    weighting: str = "none",
    sd: Any = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    nested: bool | None = None,
    kind: str = "EC50",
    direction: str = "auto",
    names: tuple[str, str] = ("X", "Y"),
) -> StatResult:
    """Which of two models fits the data better?

    Nested models (the simpler one is a special case of the other) are compared with the
    extra-sum-of-squares F test,
    F = [(SS_simple - SS_complex) / (df_simple - df_complex)] / (SS_complex / df_complex),
    and the simpler model is kept unless P < alpha. In every case the AICc difference and the
    Akaike weights (the probability that each model is the better of the two) are reported;
    for non-nested models that is the only valid comparison.

    ``nested=None`` decides from the model library (same model with more parameters fixed,
    3- vs 4-parameter dose-response, line vs quadratic, specific vs total binding). For
    user-defined equations nesting cannot be verified, so pass ``nested=True`` if you know
    the simpler equation is a special case of the other.

    Wording: "simpler" / "more complex" are only used when the models have different numbers
    of fitted parameters; otherwise they are called "first" and "second". When the AICc
    values are equal (difference below 1e-6, e.g. two equations that are the same function)
    neither model is preferred: the result says that the data do not distinguish them,
    ``extra["preferred"]`` is ``"none"`` and the first model's curve is shown. An AICc
    difference below 2 is reported as weak evidence (Burnham & Anderson 2002, sec. 2.6).
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    kind = _check_kind(kind)
    common = {"weighting": weighting, "sd": sd, "direction": direction, "names": names}
    setup_s = _setup(x, y, model_simple, equation=equation_simple, initial=initial_simple,
                     fixed=fixed_simple, bounds=bounds_simple, **common)
    setup_c = _setup(x, y, model_complex, equation=equation_complex, initial=initial_complex,
                     fixed=fixed_complex, bounds=bounds_complex, **common)
    fit_s = _run_fit(setup_s.model, setup_s.x, setup_s.y, setup_s.w, setup_s.start,
                     setup_s.fixed, setup_s.bounds)
    fit_c = _run_fit(setup_c.model, setup_c.x, setup_c.y, setup_c.w, setup_c.start,
                     setup_c.fixed, setup_c.bounds)
    warns: list[str] = []
    swapped = False
    if fit_s.p > fit_c.p:
        fit_s, fit_c, setup_s, setup_c = fit_c, fit_s, setup_c, setup_s
        swapped = True
        warns.append("The model given as 'simple' has more fitted parameters than the 'complex' "
                     "one, so the two were swapped.")
    # "simpler" / "more complex" only mean something when the parameter counts differ
    same_size = fit_s.p == fit_c.p
    word_s, word_c = ("first", "second") if same_size else ("simpler", "more complex")
    for label, fit in ((word_s, fit_s), (word_c, fit_c)):
        if not fit.converged:
            warns.append(f"The fit of the {label} model did not converge; the comparison is "
                         "unreliable.")
    auto = _nesting(fit_s, fit_c)
    if nested is None:
        is_nested = bool(auto)
        nesting_note = (
            "The models are nested (the simpler one is a special case of the other)." if auto else
            "Nesting could not be verified for a user-defined equation, so only the AICc "
            "comparison is shown. Pass nested=True if the simpler equation is a special case of "
            "the other." if auto is None else
            "The models are not nested (neither is a special case of the other), so the F test "
            "does not apply and only the AICc comparison is valid.")
    else:
        is_nested = bool(nested)
        nesting_note = ("You declared the models to be nested." if is_nested else
                        "You declared the models not to be nested; only AICc is used.")
        if is_nested and auto is False:
            warns.append("You declared the models nested, but according to the model library "
                         "they are not: the F test is probably invalid.")
    gof_s, gof_c = _goodness(fit_s), _goodness(fit_c)
    akaike = _akaike(gof_s["aicc"], gof_c["aicc"])
    f_stat = p_value = None
    df_num = fit_s.df - fit_c.df
    df_den = fit_c.df
    f_possible = is_nested and df_num > 0 and df_den > 0
    if is_nested and df_num <= 0:
        warns.append("Both models have the same number of fitted parameters, so the F test "
                     "cannot be used; compare them by AICc.")
    if is_nested and df_den <= 0:
        warns.append("Too few points to compare the models with an F test.")
    if f_possible:
        if fit_c.ss > fit_s.ss * (1.0 + 1e-8) + 1e-300:
            warns.append(
                "The more complex model fits WORSE than the simpler one, which is impossible for "
                "nested models at their best fits: the complex fit is stuck in a poor solution. "
                "Provide better initial values."
            )
        elif fit_c.ss <= 0 or gof_c["perfect_fit"]:
            warns.append("The more complex model fits the data exactly, so the F ratio is "
                         "undefined.")
        else:
            f_stat = max(((fit_s.ss - fit_c.ss) / df_num) / (fit_c.ss / df_den), 0.0)
            p_value = float(stats.f.sf(f_stat, df_num, df_den))
    if akaike["delta_aicc"] is None:
        warns.append("AICc could not be computed for both models (too few points per "
                     "parameter, or an exact fit).")

    # identical AICc (to rounding): the data do not distinguish the models at all
    aicc_tie = akaike["delta_aicc"] is not None and abs(akaike["delta_aicc"]) < AICC_TIE
    undecided = False
    if p_value is not None:
        prefer_complex = p_value < alpha
        basis = "F test"
    elif akaike["delta_aicc"] is not None:
        prefer_complex = akaike["delta_aicc"] < 0 and not aicc_tie
        undecided = aicc_tie
        basis = "AICc (tie: neither model is preferred)" if aicc_tie else "AICc"
    else:
        prefer_complex = False
        basis = f"default ({word_s} model)"
    preferred, setup_p = (fit_c, setup_c) if prefer_complex else (fit_s, setup_s)
    label_s, label_c = fit_s.model.label, fit_c.model.label
    if fit_s.model.key == fit_c.model.key:
        label_s += " (more parameters fixed)"
    params, _, cis = _parameter_table(preferred, ci_level, "asymptotic", kind)
    grid, log_spaced = _curve_grid(preferred.model, preferred.x)
    effect_sizes = [
        EffectSize(name="R squared", value=gof_s["r_squared"], ci_level=ci_level,
                   target=f"{word_s} model: {label_s}"),
        EffectSize(name="R squared", value=gof_c["r_squared"], ci_level=ci_level,
                   target=f"{word_c} model: {label_c}"),
    ]
    comparison_rows = [
        {"model": word_s, "label": label_s, "n_parameters": fit_s.p, "df": fit_s.df,
         "ss": fit_s.ss, "r_squared": gof_s["r_squared"], "sy_x": gof_s["sy_x"],
         "aicc": gof_s["aicc"], "akaike_weight": akaike["probability_simple"]},
        {"model": word_c, "label": label_c, "n_parameters": fit_c.p, "df": fit_c.df,
         "ss": fit_c.ss, "r_squared": gof_c["r_squared"], "sy_x": gof_c["sy_x"],
         "aicc": gof_c["aicc"], "akaike_weight": akaike["probability_complex"]},
    ]
    assumptions = [
        AssumptionCheck(
            name="Nested models (needed for the F test)",
            status="pass" if is_nested and (nested is None or auto is not False) else
            "warn" if is_nested else "info",
            detail=nesting_note),
        AssumptionCheck(
            name="Both fits converged",
            status="pass" if fit_s.converged and fit_c.converged else "fail",
            detail="A comparison is only meaningful if each model was fitted to its true best "
                   "fit." + ("" if fit_s.converged and fit_c.converged else
                             " At least one fit did not converge.")),
        residual_normality(np.sqrt(preferred.w) * (preferred.y - preferred.fitted))
        if not _goodness(preferred)["perfect_fit"] and preferred.df > 0 else
        AssumptionCheck(name="Residual diagnostics", status="warn",
                        detail="The model shown fits exactly; residuals cannot be checked."),
        AssumptionCheck(
            name="Same data and weighting for both models", status="info",
            detail="Both models were fitted to exactly the same points with the same weighting, "
                   "as the F test and AICc require."),
    ]
    result = StatResult(
        analysis="compare_fits",
        test_name="Comparison of two curve fits (extra sum-of-squares F test and AICc)",
        why_chosen="A model with more parameters always fits at least as well; this analysis "
                   "asks whether the improvement is larger than expected by chance.",
        assumptions=assumptions,
        statistic_name="F" if f_possible else None,
        statistic=f_stat,
        df=float(df_num) if f_possible else None,
        df2=float(df_den) if f_possible else None,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=params,
        n={names[1]: preferred.n},
        descriptives=[{"group": names[1], "n": preferred.n,
                       "n_x_values": _replicate_summary(preferred.x)[0]}],
        anova_table=([
            {"source": "Improvement (simpler - complex)", "ss": fit_s.ss - fit_c.ss, "df": df_num,
             "ms": (fit_s.ss - fit_c.ss) / df_num, "f": f_stat, "p": p_value},
            {"source": "Residual (complex model)", "ss": fit_c.ss, "df": df_den,
             "ms": fit_c.ss / df_den, "f": None, "p": None},
            {"source": "Residual (simpler model)", "ss": fit_s.ss, "df": fit_s.df,
             "ms": fit_s.ss / fit_s.df if fit_s.df > 0 else None, "f": None, "p": None},
        ] if f_possible else []),
        warnings=[*setup_s.warnings, *[w for w in setup_c.warnings if w not in setup_s.warnings],
                  *warns],
        data_notes=[*setup_s.notes, *[n for n in setup_c.notes if n not in setup_s.notes]],
        citations=[_CITE_BW, _CITE_AICC, _CITE_BA],
        extra={
            "curve": _bands(preferred, grid, ci_level, weighting),
            "residuals": {"x": preferred.x.tolist(), "fitted": preferred.fitted.tolist(),
                          "residual": (preferred.y - preferred.fitted).tolist()},
            "equation": _display_equation(preferred.model.equation, kind),
            "model": preferred.model.key,
            "x_is_log": bool(preferred.model.x_is_log),
            "x_log_spaced": bool(log_spaced),
            "preferred": "none" if undecided else ("complex" if prefer_complex else "simple"),
            "shown": "complex" if prefer_complex else "simple",
            "preferred_basis": basis,
            "same_number_of_parameters": bool(same_size),
            "nested": bool(is_nested),
            "models_swapped": swapped,
            "comparison": comparison_rows,
            "akaike": akaike,
            "fits": {"simple": _fit_summary(fit_s, kind, ci_level),
                     "complex": _fit_summary(fit_c, kind, ci_level)},
        },
    )
    aic_text = ""
    if aicc_tie:
        aic_text = (
            " By AICc the two models are equally supported (difference in AICc = 0; 50% / 50%): "
            "the data do not distinguish them."
        )
    elif akaike["delta_aicc"] is not None:
        better = word_c if akaike["delta_aicc"] < 0 else word_s
        prob = akaike["probability_complex"] if akaike["delta_aicc"] < 0 else akaike["probability_simple"]
        weak = abs(akaike["delta_aicc"]) < 2.0
        aic_text = (
            f" By AICc the {better} model is more likely to be correct (difference in AICc = "
            f"{fmt(abs(akaike['delta_aicc']), 3)}; probability {fmt(100 * prob, 3)}% that it is "
            "the better of the two)"
            + ("; a difference below 2 is weak evidence, so the data do not clearly distinguish "
               "the two models." if weak else ".")
        )
    if p_value is not None:
        verdict = (
            f"The more complex model ({label_c}) fits significantly better than the simpler one "
            f"({label_s}): F({df_num}, {df_den}) = {fmt(f_stat)}, {p_phrase(p_value)}. There is "
            "evidence that the extra parameter(s) are needed, so the more complex model is "
            "preferred."
            if prefer_complex else
            f"The more complex model ({label_c}) does not fit significantly better than the "
            f"simpler one ({label_s}): F({df_num}, {df_den}) = {fmt(f_stat)}, "
            f"{p_phrase(p_value)}. The simpler model is preferred, because a simpler model is "
            f"kept unless P < {alpha:g}."
        )
    elif undecided:
        verdict = (
            f"{nesting_note} '{label_s}' and '{label_c}' fit these data equally well"
            + (" (they have the same number of fitted parameters and the same residual sum of "
               "squares)" if same_size else "")
            + f", so neither can be favoured over the other; the fit of the {word_s} model "
              f"('{label_s}') is shown."
        )
    elif basis == "AICc":
        chosen = label_c if prefer_complex else label_s
        verdict = (
            f"{nesting_note} Comparing '{label_s}' with '{label_c}' by AICc, the "
            f"{word_c if prefer_complex else word_s} model ('{chosen}') is preferred."
        )
    else:
        verdict = (f"The two models could not be compared formally; the {word_s} model "
                   f"({label_s}) is kept by default.")
    result.interpretation = verdict + aic_text + (
        " Choose between them on scientific grounds." if undecided else
        " A statistical preference is not proof: also check that the preferred fit is "
        "scientifically sensible.")
    result.methods_text = (
        f"The fits of two models ({label_s}; {label_c}) were compared with "
        + ("the extra sum-of-squares F test " if f_possible else "")
        + ("and " if f_possible else "")
        + f"the corrected Akaike information criterion (AICc); alpha = {alpha:g}."
    )
    return result


# ============================================================ compare_datasets
@guarded
def compare_datasets(
    datasets: dict[str, tuple[Any, Any]],
    model: str | CurveModel = "dose_response_4pl_logx",
    parameter: str | None = None,
    *,
    equation: str | None = None,
    initial: Any = None,
    fixed: Any = None,
    bounds: Any = None,
    weighting: str = "none",
    alpha: float = 0.05,
    ci_level: float = 0.95,
    kind: str = "EC50",
    direction: str = "auto",
) -> StatResult:
    """Does one parameter (for example LogEC50) differ between two or more data sets?

    Two fits are compared with the extra-sum-of-squares F test (Bates & Watts 1988;
    Motulsky & Christopoulos 2004, *Fitting Models to Biological Data Using Linear and
    Nonlinear Regression*, Oxford Univ. Press, ch. 27):

    * separate curves, every parameter free in every data set (SS_sep = sum of the
      individual SS, df_sep = sum of the individual df);
    * a global fit in which ``parameter`` is shared by all data sets and the other
      parameters stay separate (SS_shared, df_shared = df_sep + k - 1).

    F = [(SS_shared - SS_sep) / (k - 1)] / (SS_sep / df_sep). A small P value is evidence
    that the parameter differs between the data sets. The scatter is assumed to be the same
    in all data sets.
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    kind = _check_kind(kind)
    if not isinstance(datasets, dict) or len(datasets) < 2:
        raise StatsError("Comparing a parameter between data sets needs at least two data sets.")
    if weighting == "1/sd^2":
        raise StatsError("Weighting by 1/SD^2 is not available when comparing data sets.")
    setups: dict[str, _Setup] = {}
    fits: dict[str, _Fit] = {}
    resolved_direction = direction
    for name, pair in datasets.items():
        try:
            xv, yv = pair
        except (TypeError, ValueError):
            raise StatsError(f"Data set '{name}' must be a pair of (X values, Y values).") from None
        setup = _setup(xv, yv, model, equation=equation, initial=initial, fixed=fixed,
                       bounds=bounds, weighting=weighting, sd=None, direction=resolved_direction,
                       names=("X", str(name)))
        if resolved_direction == "auto" and "HillSlope" in setup.model.default_fixed:
            # all data sets must use the same fixed slope
            resolved_direction = "stimulation" if setup.fixed["HillSlope"] > 0 else "inhibition"
        setups[str(name)] = setup
        fits[str(name)] = _run_fit(setup.model, setup.x, setup.y, setup.w, setup.start,
                                   setup.fixed, setup.bounds)
    first = next(iter(fits.values()))
    cm = first.model
    if parameter is None or parameter == "":
        shared = "LogEC50" if cm.family == "dose_response" else first.free_names[-1]
    else:
        shared, _ = _canonical(cm, parameter)
    if shared not in first.free_names:
        raise StatsError(f"'{_display(shared, kind)}' is fixed, so it cannot be compared between "
                         "data sets.")
    names = list(fits)
    k = len(names)
    free_names = first.free_names
    other = [n for n in free_names if n != shared]
    n_other = len(other)
    idx_shared = cm.parameters.index(shared)
    idx_other = [cm.parameters.index(n) for n in other]
    warns: list[str] = []
    notes: list[str] = []
    for name in names:
        notes.extend(f"{name}: {t}" for t in setups[name].notes)
        warns.extend(f"{name}: {t}" for t in setups[name].warnings)
        if not fits[name].converged:
            warns.append(f"The separate fit of '{name}' did not converge; the comparison is "
                         "unreliable.")
    ss_sep = sum(f.ss for f in fits.values())
    df_sep = sum(f.df for f in fits.values())
    n_total = sum(f.n for f in fits.values())

    # ---- global fit with the parameter shared
    sizes = [fits[n].n for n in names]
    offsets = np.concatenate([[0], np.cumsum(sizes)])

    def unpack(theta: np.ndarray, i: int) -> np.ndarray:
        full = fits[names[i]].theta.copy()
        full[idx_shared] = theta[0]
        if n_other:
            full[idx_other] = theta[1 + i * n_other: 1 + (i + 1) * n_other]
        return full

    def residual(theta: np.ndarray) -> np.ndarray:
        parts = []
        for i, name in enumerate(names):
            f = fits[name]
            parts.append(np.sqrt(f.w) * (cm.func(f.tx, unpack(theta, i)) - f.y))
        return np.concatenate(parts)

    def jacobian(theta: np.ndarray) -> np.ndarray:
        out = np.zeros((n_total, 1 + k * n_other))
        for i, name in enumerate(names):
            f = fits[name]
            j = np.sqrt(f.w)[:, None] * cm.jac(f.tx, unpack(theta, i))
            rows = slice(offsets[i], offsets[i + 1])
            out[rows, 0] = j[:, idx_shared]
            if n_other:
                out[rows, 1 + i * n_other: 1 + (i + 1) * n_other] = j[:, idx_other]
        return out

    lower = np.concatenate([[first.lower[idx_shared]], np.tile(first.lower[idx_other], k)])
    upper = np.concatenate([[first.upper[idx_shared]], np.tile(first.upper[idx_other], k)])
    start = np.concatenate([[float(np.mean([fits[n].theta[idx_shared] for n in names]))],
                            *[fits[n].theta[idx_other] for n in names]])
    problem = _Problem(residual=residual, jacobian=jacobian, lower=lower, upper=upper, n=n_total,
                       exact_ss=sum(_exact_ss(f.w, f.y) for f in fits.values()),
                       unit=_unit_scale(np.concatenate([np.sqrt(f.w) * f.y for f in fits.values()])))
    cand, _ = _solve(problem, start)
    if cand is None:
        raise StatsError("The global fit with a shared parameter could not be computed.")
    if not cand.converged:
        warns.append("The global fit with the shared parameter did not converge; the comparison "
                     "is unreliable.")
    ss_shared = cand.ss
    df_shared = n_total - (1 + k * n_other)
    df_num = k - 1
    f_stat = p_value = None
    if df_sep < 1:
        warns.append("Too few points for an F test (no residual degrees of freedom).")
    elif ss_sep <= problem.exact_ss:
        warns.append("The separate curves fit the data exactly, so the F ratio is undefined.")
    else:
        f_stat = max(((ss_shared - ss_sep) / df_num) / (ss_sep / df_sep), 0.0)
        p_value = float(stats.f.sf(f_stat, df_num, df_sep))
    # ---- estimates
    label = _display(shared, kind)
    s2_pooled = ss_sep / df_sep if df_sep > 0 else None
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df_sep)) if df_sep > 0 else None
    parameters: list[Parameter] = []
    unscaled: dict[str, float | None] = {}
    for name in names:
        f = fits[name]
        value = float(f.theta[idx_shared])
        se = None
        unscaled[name] = None
        if f.cov is not None and f.s2 is not None and f.s2 > 0 and s2_pooled is not None:
            var_unscaled = float(f.cov[f.free_names.index(shared), f.free_names.index(shared)]) / f.s2
            unscaled[name] = var_unscaled
            se = math.sqrt(var_unscaled * s2_pooled)
        parameters.append(Parameter(
            name=f"{label} ({name})", value=value, se=se,
            ci_low=None if se is None else value - t_crit * se,
            ci_high=None if se is None else value + t_crit * se, ci_level=ci_level,
            unit_note="Separate fit; SE uses the scatter pooled over all data sets"))
    cov_shared, _, _ = _covariance(jacobian(cand.theta))
    se_shared = None
    if cov_shared is not None and df_shared > 0:
        se_shared = math.sqrt(max(cov_shared[0, 0] * ss_shared / df_shared, 0.0))
    t_shared = float(stats.t.ppf(0.5 + ci_level / 2.0, df_shared)) if df_shared > 0 else None
    parameters.append(Parameter(
        name=f"{label} (shared)", value=float(cand.theta[0]), se=se_shared,
        ci_low=None if se_shared is None else cand.theta[0] - t_shared * se_shared,
        ci_high=None if se_shared is None else cand.theta[0] + t_shared * se_shared,
        ci_level=ci_level,
        unit_note="Global fit with one value for all data sets (meaningful if they do not differ)"))
    diff_text = ""
    if k == 2:
        a, b = names
        diff = float(fits[b].theta[idx_shared] - fits[a].theta[idx_shared])
        se_diff = None
        if unscaled[a] is not None and unscaled[b] is not None and s2_pooled is not None:
            se_diff = math.sqrt((unscaled[a] + unscaled[b]) * s2_pooled)
        lo = None if se_diff is None else diff - t_crit * se_diff
        hi = None if se_diff is None else diff + t_crit * se_diff
        parameters.append(Parameter(
            name=f"Difference in {label} ({b} - {a})", value=diff, se=se_diff, ci_low=lo,
            ci_high=hi, ci_level=ci_level, unit_note="Asymptotic CI with pooled scatter"))
        if shared == "LogEC50" and abs(diff) < 300:
            ec = _display("EC50", kind)
            parameters.append(Parameter(
                name=f"{ec} ratio ({b} / {a})", value=10.0**diff,
                ci_low=None if lo is None or abs(lo) > 300 else 10.0**lo,
                ci_high=None if hi is None or abs(hi) > 300 else 10.0**hi, ci_level=ci_level,
                unit_note=f"Fold difference in {ec} = 10^(difference in {label})"))
            diff_text = (f" The {ec} of {b} is {fmt(10.0 ** diff, 3)}-fold that of {a}"
                         + (f" ({ci_phrase(10.0 ** lo, 10.0 ** hi, ci_level)})"
                            if lo is not None and abs(lo) < 300 and abs(hi) < 300 else "") + ".")
    ic_sep = information_criteria(ss_sep, n_total, k * len(free_names))
    ic_shared = information_criteria(ss_shared, n_total, 1 + k * n_other)
    akaike = _akaike(ic_shared["aicc"], ic_sep["aicc"])
    resid_all = np.concatenate([np.sqrt(f.w) * (f.y - f.fitted) for f in fits.values()])
    perfect = ss_sep <= problem.exact_ss
    result = StatResult(
        analysis="compare_datasets",
        test_name=f"Comparison of {label} between data sets (extra sum-of-squares F test)",
        why_chosen=f"Use this to ask whether {label} differs between data sets (for example two "
                   "formulations), by testing whether curves forced to share one value fit "
                   "clearly worse than curves with separate values.",
        assumptions=[
            AssumptionCheck(
                name="All fits converged",
                status="pass" if cand.converged and all(f.converged for f in fits.values()) else "fail",
                detail="Separate and shared fits must each reach their best fit."),
            residual_normality(resid_all) if not perfect and df_sep > 0 else AssumptionCheck(
                name="Residual diagnostics", status="warn",
                detail="The curves fit exactly; residuals cannot be checked."),
            AssumptionCheck(
                name="Similar scatter in every data set", status="info",
                detail="The F test pools the scatter around all curves, which assumes it is "
                       "similar in every data set."),
            AssumptionCheck(
                name="Independent observations", status="info",
                detail="Each point must be an independent measurement, and the data sets must be "
                       "independent of each other."),
        ],
        statistic_name="F",
        statistic=f_stat,
        df=float(df_num),
        df2=float(df_sep) if df_sep > 0 else None,
        p_value=p_value,
        alpha=alpha,
        effect_sizes=[EffectSize(
            name="R squared (separate curves)",
            value=(1.0 - ss_sep / sum(f.ss_tot for f in fits.values())
                   if sum(f.ss_tot for f in fits.values()) > 0 else None), ci_level=ci_level)],
        parameters=parameters,
        n={name: fits[name].n for name in names},
        descriptives=[{"group": name, "n": fits[name].n,
                       "n_x_values": _replicate_summary(fits[name].x)[0],
                       label: float(fits[name].theta[idx_shared])} for name in names],
        anova_table=([
            {"source": f"{label} shared vs separate", "ss": ss_shared - ss_sep, "df": df_num,
             "ms": (ss_shared - ss_sep) / df_num, "f": f_stat, "p": p_value},
            {"source": "Residual (separate curves)", "ss": ss_sep, "df": df_sep,
             "ms": ss_sep / df_sep, "f": None, "p": None},
            {"source": "Residual (shared parameter)", "ss": ss_shared, "df": df_shared,
             "ms": ss_shared / df_shared if df_shared > 0 else None, "f": None, "p": None},
        ] if df_sep > 0 else []),
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_BW, "Motulsky HJ, Christopoulos A (2004) Fitting Models to Biological "
                             "Data Using Linear and Nonlinear Regression, Oxford Univ. Press",
                   _CITE_AICC],
        extra={
            "model": cm.key,
            "equation": _display_equation(cm.equation, kind),
            "x_is_log": bool(cm.x_is_log),
            "shared_parameter": label,
            "ss": {"separate": ss_sep, "shared": ss_shared},
            "df": {"separate": df_sep, "shared": df_shared},
            "akaike": {"delta_aicc_separate_minus_shared": akaike["delta_aicc"],
                       "probability_shared": akaike["probability_simple"],
                       "probability_separate": akaike["probability_complex"]},
            "fits": {name: _fit_summary(fits[name], kind, ci_level) for name in names},
            "curves": {name: _bands(fits[name], _curve_grid(cm, fits[name].x)[0], ci_level,
                                    weighting) for name in names},
            "shared_fit": {
                "value": float(cand.theta[0]), "se": se_shared, "converged": bool(cand.converged),
                "other_parameters": {
                    name: {_display(o, kind): float(v) for o, v in
                           zip(other, cand.theta[1 + i * n_other: 1 + (i + 1) * n_other])}
                    for i, name in enumerate(names)},
            },
        },
    )
    listing = ", ".join(f"{name}: {fmt(fits[name].theta[idx_shared])}" for name in names)
    if p_value is None:
        result.interpretation = (
            f"The best-fit {label} values are {listing}, but the difference could not be tested."
        )
    else:
        differ = p_value < alpha
        result.interpretation = (
            f"The best-fit {label} values are {listing}. Forcing the data sets to share one "
            f"{label} makes the fit {'significantly' if differ else 'not significantly'} worse: "
            f"F({df_num}, {df_sep}) = {fmt(f_stat)}, which is {sig_phrase(p_value, alpha)}. "
            + (f"There is evidence that {label} differs between the data sets." if differ else
               f"There is no clear evidence that {label} differs between the data sets (which is "
               "not proof that it is the same).")
            + diff_text
        )
    result.methods_text = (
        f"{label} was compared between {', '.join(names)} by fitting the {cm.label} model with "
        f"{label} either shared among or separate for the data sets, and comparing the two fits "
        f"with the extra sum-of-squares F test (alpha = {alpha:g})."
    )
    return result


# ================================================================ table adapters
_MODEL_HELP_BASE = (
    "The equation to fit. Dose-response models come in two versions: '_logx' when your X "
    "values are log10(concentration), and without '_logx' when X is the concentration itself."
)
_MODEL_HELP = _MODEL_HELP_BASE + " Choose 'user_defined' to type your own equation."
#: compare_curve_parameter offers only the built-in models (its choices exclude 'user_defined')
_MODEL_HELP_SHARED = (
    _MODEL_HELP_BASE + " The same built-in equation is fitted to every data set; an equation you "
    "type yourself cannot be used in this analysis."
)
_WEIGHT_HELP = (
    "How much each point counts. 'none' treats all points equally (right when the scatter is "
    "similar everywhere). '1/y^2' suits data whose scatter is a constant percentage of Y; "
    "'1/y' is in between. '1/sd^2' weights each X by the SD of its replicates."
)
_ALPHA = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold for calling a test (such as a model comparison) significant. "
         "0.05 is the usual choice.")


def _fit_inputs(table: Any, series: Any, use_means: bool, weighting: str) -> tuple[
        str, np.ndarray, np.ndarray, np.ndarray | None, list[str], list[str]]:
    """(name, x, y, sd, notes, warnings) for one series of an XY table."""
    warns: list[str] = []
    force_means = weighting == "1/sd^2"
    name, xv, yv, notes, means = xy_series_points(table, series, bool(use_means) or force_means)
    sd = None
    try:
        n_rep = len(table.series_map[name])
    except (AttributeError, KeyError) as exc:  # pragma: no cover - defensive
        raise StatsError(str(exc)) from exc
    if force_means:
        sd = means["sd"].to_numpy(dtype=float)
        if n_rep < 2:
            raise StatsError(
                "Weighting by 1/SD^2 needs replicate Y values at every X, but this series has a "
                "single column. Choose a different weighting."
            )
        if not use_means:
            notes.append("Weighting by 1/SD^2 fits the mean at each X (weighted by the SD of its "
                         "replicates).")
    elif n_rep > 1 and not use_means:
        notes.append(
            "Each replicate was fitted as an individual point: this uses all the information "
            "about scatter and is right when replicates are independent samples."
        )
    elif n_rep > 1 and use_means:
        warns.append(
            "The curve was fitted to the mean at each X. This discards the information in the "
            "scatter of the replicates; prefer fitting individual replicates unless they are "
            "repeat readings of the same sample."
        )
    return name, xv, yv, sd, notes, warns


@register(
    key="nonlinear_regression", label="Nonlinear regression (curve fit)",
    category="Nonlinear regression", tables=("xy",),
    params=(
        ParamSpec("series", "Y series", kind="group", default=None, optional=True,
                  help="Which Y data set to fit. Leave empty to use the first one."),
        ParamSpec("model", "Model", kind="choice", default="dose_response_4pl",
                  choices=MODEL_KEYS, help=_MODEL_HELP),
        ParamSpec("equation", "Equation (user-defined model)", kind="text", default="",
                  optional=True,
                  help="Only for the 'user_defined' model. Type the equation with X as the "
                       "independent variable, e.g. Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope). "
                       "Every other name becomes a parameter to fit, including a lone 'e' (for "
                       "Euler's number 2.718 write exp(1), and exp(X) for an exponential). The only "
                       "built-in constant is 'pi' (3.14159)."),
        ParamSpec("initial", "Initial values", kind="text", default="", optional=True,
                  help="Optional starting guesses such as 'Top=100, Bottom=0, EC50=10'. Built-in "
                       "models find their own; a user-defined equation needs rough estimates."),
        ParamSpec("fixed", "Fixed parameters", kind="text", default="", optional=True,
                  help="Parameters to hold constant instead of fitting, such as 'Bottom=0' or "
                       "'Bottom=0, Top=100' for data normalised to 0-100%."),
        ParamSpec("weighting", "Weighting", kind="choice", default="none", choices=WEIGHTINGS,
                  help=_WEIGHT_HELP),
        ParamSpec("use_means", "Fit the mean of replicates", kind="bool", default=False,
                  help="Off (recommended): every replicate is fitted as its own point, which uses "
                       "all the information about scatter and gives correct confidence intervals "
                       "when replicates are independent. Tick only if replicates are repeat "
                       "readings of the same sample."),
        ParamSpec("ci_method", "Confidence interval method", kind="choice", default="asymptotic",
                  choices=CI_METHODS,
                  help="'asymptotic' is the fast, symmetric interval most programs report. "
                       "'profile' (profile likelihood) is slower but more accurate when the "
                       "uncertainty is lopsided."),
        ParamSpec("potency_name", "Name of the potency parameter", kind="choice", default="EC50",
                  choices=("EC50", "IC50"),
                  help="Dose-response models only: call the midpoint EC50 (stimulation) or IC50 "
                       "(inhibition). This changes labels only, not the numbers."),
        ParamSpec("direction", "Direction (three-parameter dose-response)", kind="choice",
                  default="auto", choices=("auto", "stimulation", "inhibition"),
                  help="Three-parameter dose-response models fix the Hill slope at +1 "
                       "(stimulation) or -1 (inhibition). 'auto' decides from the data."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Fits a curve (dose-response, enzyme kinetics, exponential, binding, or your own "
                "equation) to X-Y data and reports the best-fit parameters such as EC50 with "
                "confidence intervals.",
    citation=_CITE_BW,
)
def _run_nonlinear_regression(
    table: Any, *, series: str | None = None, model: str = "dose_response_4pl",
    equation: str = "", initial: Any = "", fixed: Any = "", weighting: str = "none",
    use_means: bool = False, ci_method: str = "asymptotic", potency_name: str = "EC50",
    direction: str = "auto", alpha: float = 0.05, ci_level: float = 0.95,
) -> StatResult:
    if weighting not in WEIGHTINGS:
        raise StatsError(f"weighting must be one of: {', '.join(WEIGHTINGS)}.")
    alpha = check_alpha(alpha)
    name, xv, yv, sd, notes, warns = _fit_inputs(table, series, bool(use_means), weighting)
    result = fit_curve(
        xv, yv, model, equation=equation or None, initial=initial or None, fixed=fixed or None,
        weighting=weighting, sd=sd, alpha=alpha, ci_level=ci_level, ci_method=ci_method,
        kind=potency_name, direction=direction, names=(table.x_name, name),
    )
    result.data_notes.extend(notes)
    result.warnings.extend(warns)
    result.extra["series"] = name
    return result


@register(
    key="compare_fits", label="Compare two curve fits (which model is better?)",
    category="Nonlinear regression", tables=("xy",),
    params=(
        ParamSpec("series", "Y series", kind="group", default=None, optional=True,
                  help="Which Y data set to fit. Leave empty to use the first one."),
        ParamSpec("model_simple", "Simpler model", kind="choice", default="dose_response_3pl",
                  choices=MODEL_KEYS,
                  help="The model with fewer parameters (for example the three-parameter "
                       "dose-response curve with a standard slope)."),
        ParamSpec("model_complex", "More complex model", kind="choice", default="dose_response_4pl",
                  choices=MODEL_KEYS,
                  help="The model with more parameters (for example the four-parameter curve "
                       "with a variable slope)."),
        ParamSpec("equation_simple", "Equation of the simpler model (user-defined)", kind="text",
                  default="", optional=True,
                  help="Only when the simpler model is 'user_defined'."),
        ParamSpec("equation_complex", "Equation of the complex model (user-defined)", kind="text",
                  default="", optional=True,
                  help="Only when the more complex model is 'user_defined'."),
        ParamSpec("fixed_simple", "Fixed parameters (simpler model)", kind="text", default="",
                  optional=True,
                  help="Parameters held constant in the simpler model, e.g. 'Bottom=0'. Fitting "
                       "the same model with and without a fixed parameter tests whether that "
                       "constraint is reasonable."),
        ParamSpec("fixed_complex", "Fixed parameters (complex model)", kind="text", default="",
                  optional=True,
                  help="Parameters held constant in the more complex model."),
        ParamSpec("use_means", "Fit the mean of replicates", kind="bool", default=False,
                  help="Off (recommended): every replicate is fitted as its own point."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Fits two models to the same data and tests whether the more complicated one "
                "fits enough better to justify its extra parameters.",
    citation=f"{_CITE_BW}; {_CITE_AICC}",
)
def _run_compare_fits(
    table: Any, *, series: str | None = None, model_simple: str = "dose_response_3pl",
    model_complex: str = "dose_response_4pl", equation_simple: str = "",
    equation_complex: str = "", fixed_simple: Any = "", fixed_complex: Any = "",
    use_means: bool = False, alpha: float = 0.05, ci_level: float = 0.95,
) -> StatResult:
    alpha = check_alpha(alpha)
    name, xv, yv, _, notes, warns = _fit_inputs(table, series, bool(use_means), "none")
    result = compare_fits(
        xv, yv, model_simple, model_complex, equation_simple=equation_simple or None,
        equation_complex=equation_complex or None, fixed_simple=fixed_simple or None,
        fixed_complex=fixed_complex or None, alpha=alpha, ci_level=ci_level,
        names=(table.x_name, name),
    )
    result.data_notes.extend(notes)
    result.warnings.extend(warns)
    result.extra["series"] = name
    return result


@register(
    key="compare_curve_parameter",
    label="Compare a curve parameter between data sets (e.g. EC50)",
    category="Nonlinear regression", tables=("xy",),
    params=(
        ParamSpec("series", "Y series to compare", kind="groups", default=None, optional=True,
                  help="Which data sets to compare. Leave empty to compare all of them."),
        ParamSpec("model", "Model", kind="choice", default="dose_response_4pl",
                  choices=tuple(MODELS), help=_MODEL_HELP_SHARED),
        ParamSpec("parameter", "Parameter to compare", kind="text", default="", optional=True,
                  help="The parameter that may differ between data sets, e.g. 'EC50' or 'Top'. "
                       "Leave empty to compare the EC50 (dose-response models)."),
        ParamSpec("fixed", "Fixed parameters", kind="text", default="", optional=True,
                  help="Parameters to hold constant in every data set, such as 'Bottom=0'."),
        ParamSpec("use_means", "Fit the mean of replicates", kind="bool", default=False,
                  help="Off (recommended): every replicate is fitted as its own point."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Tests whether a parameter such as the EC50 differs between two or more curves, "
                "by comparing fits with the parameter shared versus separate.",
    citation=_CITE_BW,
)
def _run_compare_curve_parameter(
    table: Any, *, series: list[str] | None = None, model: str = "dose_response_4pl",
    parameter: str = "", fixed: Any = "", use_means: bool = False, alpha: float = 0.05,
    ci_level: float = 0.95,
) -> StatResult:
    alpha = check_alpha(alpha)
    if isinstance(series, str):
        series = [s.strip() for s in series.split(",") if s.strip()]
    try:
        chosen = list(series) if series else list(table.series_names)
    except TableError as exc:  # pragma: no cover - defensive
        raise StatsError(str(exc)) from exc
    if len(chosen) < 2:
        raise StatsError("This comparison needs at least two Y series in the table.")
    datasets = {}
    notes: list[str] = []
    for item in chosen:
        name, xv, yv, _, n_notes, _ = _fit_inputs(table, item, bool(use_means), "none")
        datasets[name] = (xv, yv)
        notes.extend(f"{name}: {t}" for t in n_notes)
    result = compare_datasets(datasets, model, parameter or None, fixed=fixed or None,
                              alpha=alpha, ci_level=ci_level)
    result.data_notes.extend(n for n in notes if n not in result.data_notes)
    result.extra["series"] = chosen
    return result
