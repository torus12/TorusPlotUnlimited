"""Linear regression: simple (one X), comparison of slopes, and multiple regression.

Simple linear regression is computed here in closed form (ordinary least squares on
centred sums) and cross-checked in the tests against scipy and statsmodels and against
the NIST StRD certified values. Multiple regression uses statsmodels' OLS for the fit;
the design matrix (dummy coding, interactions), VIFs, standardized coefficients and
diagnostics are built here.

References
----------
* Draper NR, Smith H (1998) *Applied Regression Analysis*, 3rd ed., Wiley (least squares,
  confidence and prediction bands, pure-error lack-of-fit F test: ch. 1-3).
* Fieller EC (1954) J R Stat Soc B 16:175-185 (confidence limits of a ratio; used for the
  X-intercept).
* Wald A, Wolfowitz J (1940) Ann Math Stat 11:147-162 (runs test).
* Breusch TS, Pagan AR (1979) Econometrica 47:1287-1294; Koenker R (1981) J Econometrics
  17:107-112 (studentized score test for heteroscedasticity).
* Durbin J, Watson GS (1950) Biometrika 37:409-428.
* Shapiro SS, Wilk MB (1965) Biometrika 52:591-611.
* Hurvich CM, Tsai CL (1989) Biometrika 76:297-307 (AICc); Burnham KP, Anderson DR (2002)
  *Model Selection and Multimodel Inference*, 2nd ed., Springer (least-squares form).
* Belsley DA, Kuh E, Welsch RE (1980) *Regression Diagnostics*, Wiley (variance inflation).
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..data.tables import TableError
from ._utils import (
    StatsError,
    check_alpha,
    ci_phrase,
    clean_paired,
    fmt,
    p_phrase,
    sig_phrase,
)
from .effect_sizes import eta_squared_ci, magnitude_r
from .registry import ParamSpec, register
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

__all__ = [
    "breusch_pagan",
    "compare_slopes",
    "durbin_watson",
    "information_criteria",
    "lack_of_fit",
    "linear_regression",
    "multiple_regression",
    "runs_test",
]

_CITE_OLS = "Draper NR, Smith H (1998) Applied Regression Analysis, 3rd ed., Wiley"
_CITE_RUNS = "Wald A, Wolfowitz J (1940) Ann Math Stat 11:147-162"
_CITE_BP = "Koenker R (1981) J Econometrics 17:107-112"
_CITE_AICC = "Hurvich CM, Tsai CL (1989) Biometrika 76:297-307"
_CITE_FIELLER = "Fieller EC (1954) J R Stat Soc B 16:175-185"

#: Residual SS below this fraction of the total SS is treated as an exact (perfect) fit.
PERFECT_FIT_RATIO = 1e-24
#: Number of points on fitted curves returned for plotting.
GRID_POINTS = 200


# =========================================================== shared diagnostics
def _clean_value(obj: Any) -> Any:
    """Plain-Python copy of ``obj`` with NaN / infinite numbers replaced by ``None``."""
    if isinstance(obj, dict):
        return {str(k): _clean_value(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean_value(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean_value(v) for v in obj.tolist()]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def sanitize_result(result: StatResult) -> StatResult:
    """Replace every non-finite number in a result by ``None`` (undefined), so results
    always serialise to strict JSON and never show ``inf`` / ``nan`` to the user."""
    def fix(model: Any, fields: tuple[str, ...]) -> None:
        for name in fields:
            value = getattr(model, name)
            if isinstance(value, float) and not math.isfinite(value):
                setattr(model, name, None)

    fix(result, ("statistic", "df", "df2", "p_value"))
    for par in result.parameters:
        fix(par, ("value", "se", "ci_low", "ci_high", "statistic", "p_value"))
    for es in result.effect_sizes:
        fix(es, ("value", "ci_low", "ci_high"))
    for check in result.assumptions:
        fix(check, ("statistic", "p_value"))
    result.extra = _clean_value(result.extra)
    result.anova_table = _clean_value(result.anova_table)
    result.descriptives = _clean_value(result.descriptives)
    return result


def guarded(func: Callable[..., StatResult]) -> Callable[..., StatResult]:
    """Decorator for public analysis functions: numerical overflow / invalid-operation
    warnings raised by pathological data are silenced, and whatever became NaN or infinite
    is reported as undefined (``None``) instead of leaking into the result."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> StatResult:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(*args, **kwargs)
        return sanitize_result(result)
    return wrapper


def is_constant(arr: np.ndarray) -> bool:
    """True when all values are identical up to floating-point noise.

    Purely *relative* (unlike ``_utils.is_constant``, which has an absolute floor of 1e-12
    and would call concentrations expressed in molar, e.g. 1e-15 to 1e-13, constant).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return True
    spread = float(np.ptp(arr))
    return spread == 0.0 or spread <= 1e-13 * float(np.max(np.abs(arr)))


def check_level(ci_level: float) -> float:
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    return float(ci_level)


def runs_test(residuals: Iterable[float]) -> dict[str, Any] | None:
    """Wald-Wolfowitz runs test on the signs of residuals (already ordered by X).

    Asks whether the curve deviates *systematically* from the data: too few runs of
    same-sign residuals means the points sit above the curve in some regions and below it
    in others. The P value is one-sided (probability of this few runs or fewer), exact from
    the combinatorial distribution of the number of runs when n <= 2000, otherwise from the
    normal approximation with a continuity correction. Residuals of exactly zero are
    ignored. Returns ``None`` when all residuals have the same sign or none is non-zero.

    Exact distribution (Wald & Wolfowitz 1940), with n1 positive and n2 negative signs:
    P(R = 2k) = 2 C(n1-1, k-1) C(n2-1, k-1) / C(n, n1);
    P(R = 2k+1) = [C(n1-1, k) C(n2-1, k-1) + C(n1-1, k-1) C(n2-1, k)] / C(n, n1).
    """
    r = np.asarray(list(residuals), dtype=float)
    r = r[np.isfinite(r) & (r != 0)]
    signs = r > 0
    n1 = int(signs.sum())
    n2 = int(signs.size - n1)
    if n1 == 0 or n2 == 0:
        return None
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    n = n1 + n2
    expected = 2.0 * n1 * n2 / n + 1.0
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0)) if n > 1 else 0.0
    z = (runs - expected) / math.sqrt(variance) if variance > 0 else None
    if n <= 2000:
        total = math.comb(n, n1)
        acc = 0
        for count in range(2, runs + 1):
            k = count // 2
            if count % 2 == 0:
                acc += 2 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
            else:
                acc += (math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
                        + math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k))
        p = acc / total
        method = "exact"
    else:
        p = float(stats.norm.cdf((runs + 0.5 - expected) / math.sqrt(variance)))
        method = "normal approximation"
    return {"runs": runs, "n_positive": n1, "n_negative": n2, "expected_runs": expected,
            "z": z, "p_value": float(min(1.0, p)), "method": method}


def mean_by_x(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``values`` at each distinct x: (unique x ascending, means, counts)."""
    ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=values, minlength=ux.size)
    return ux, sums / counts, counts


def lack_of_fit(
    x: np.ndarray, y: np.ndarray, fitted: np.ndarray, n_params: int,
    weights: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Pure-error (replicates) lack-of-fit F test (Draper & Smith 1998, sec. 2.1).

    The scatter among replicate Y values at the same X estimates pure experimental error
    (SS_pe with N - m df, m = number of distinct X). What is left of the residual SS,
    SS_lof = SS_res - SS_pe with m - p df, measures how far the model is from the replicate
    means. F = (SS_lof / df_lof) / (SS_pe / df_pe). Returns ``None`` when there are no
    replicates or too few distinct X values (m <= p).
    """
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    m = int(ux.size)
    n = int(y.size)
    df_pe = n - m
    df_lof = m - int(n_params)
    if df_pe < 1 or df_lof < 1:
        return None
    wsum = np.bincount(inverse, weights=w, minlength=m)
    wmean = np.bincount(inverse, weights=w * y, minlength=m) / wsum
    ss_pe = float(np.sum(w * (y - wmean[inverse]) ** 2))
    ss_res = float(np.sum(w * (y - fitted) ** 2))
    ss_lof = max(ss_res - ss_pe, 0.0)
    out: dict[str, Any] = {
        "ss_lack_of_fit": ss_lof, "df_lack_of_fit": df_lof,
        "ss_pure_error": ss_pe, "df_pure_error": df_pe,
        "n_x_values": m, "f": None, "p_value": None,
    }
    if ss_pe > 0 and ss_pe > PERFECT_FIT_RATIO * max(ss_res, 1e-300):
        f = (ss_lof / df_lof) / (ss_pe / df_pe)
        out["f"] = float(f)
        out["p_value"] = float(stats.f.sf(f, df_lof, df_pe))
    return out


def breusch_pagan(residuals: np.ndarray, regressors: np.ndarray) -> dict[str, Any] | None:
    """Koenker's studentized Breusch-Pagan test for non-constant residual variance.

    Regress the squared residuals on the regressors (plus a constant); LM = n R^2 follows a
    chi-square distribution with (number of regressors) df under constant variance.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    z = np.asarray(regressors, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    n, k = z.shape
    if n <= k + 1:
        return None
    centred = e2 - e2.mean()
    ss_tot = float(centred @ centred)
    if ss_tot <= 0 or not math.isfinite(ss_tot):
        return None
    design = np.column_stack([np.ones(n), z])
    coef, *_ = np.linalg.lstsq(design, e2, rcond=None)
    resid = e2 - design @ coef
    r2 = max(0.0, 1.0 - float(resid @ resid) / ss_tot)
    lm = n * r2
    return {"lm": float(lm), "df": int(k), "p_value": float(stats.chi2.sf(lm, k))}


def durbin_watson(residuals: np.ndarray) -> float | None:
    """Durbin-Watson statistic sum((e_t - e_{t-1})^2) / sum(e_t^2); about 2 = no serial correlation."""
    e = np.asarray(residuals, dtype=float)
    denom = float(e @ e)
    if e.size < 2 or denom <= 0:
        return None
    return float(np.sum(np.diff(e) ** 2) / denom)


def information_criteria(ss: float, n: int, n_params: int) -> dict[str, float | None]:
    """Least-squares AIC / AICc / BIC.

    Convention (Burnham & Anderson 2002, sec. 2.2; Hurvich & Tsai 1989):
    AIC = N ln(SS / N) + 2K, AICc = AIC + 2K(K + 1) / (N - K - 1), BIC = N ln(SS / N) + K ln N,
    with K = number of fitted parameters + 1 (the residual variance counts as a parameter).
    Constant terms of the Gaussian log-likelihood are dropped, so only *differences* between
    models fitted to the same data are meaningful. AICc is undefined when N <= K + 1.
    """
    k = int(n_params) + 1
    out: dict[str, float | None] = {"aic": None, "aicc": None, "bic": None, "k": k}
    if n < 1 or not math.isfinite(ss) or ss <= 0:
        return out
    base = n * math.log(ss / n)
    out["aic"] = base + 2.0 * k
    out["bic"] = base + k * math.log(n)
    if n - k - 1 > 0:
        out["aicc"] = base + 2.0 * k + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    return out


def residual_normality(residuals: np.ndarray) -> AssumptionCheck:
    """Shapiro-Wilk test on residuals, wrapped as an assumption check."""
    name = "Normality of residuals (Shapiro-Wilk)"
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return AssumptionCheck(name=name, status="warn",
                               detail="Too few points (n < 3) to check the residuals for normality.")
    if is_constant(r) or float(np.max(np.abs(r))) == 0.0:
        return AssumptionCheck(name=name, status="warn",
                               detail="The residuals are all (nearly) identical, so normality "
                                      "cannot be assessed.")
    if r.size > 5000:
        return AssumptionCheck(name=name, status="info",
                               detail="Not tested: with more than 5000 points the test flags trivial "
                                      "departures. Inspect a QQ plot of the residuals instead.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, p = stats.shapiro(r)
    if not math.isfinite(p):
        return AssumptionCheck(name=name, status="warn",
                               detail="The normality test could not be computed for these residuals.")
    if p < 0.05:
        return AssumptionCheck(
            name=name, statistic=w, p_value=p, status="fail",
            detail=f"The residuals do not look normally distributed ({p_phrase(p)}). Look for "
                   "outliers, consider transforming Y or weighting; P values and confidence "
                   "intervals may be unreliable.")
    return AssumptionCheck(name=name, statistic=w, p_value=p, status="pass",
                           detail=f"No evidence against normally distributed residuals ({p_phrase(p)}).")


def runs_assumption(info: dict[str, Any] | None, what: str = "line") -> AssumptionCheck:
    name = "No systematic deviation (runs test)"
    if info is None:
        return AssumptionCheck(name=name, status="warn",
                               detail=f"Not computable: all residuals lie on one side of the {what} "
                                      "(or are zero).")
    detail = (f"{info['runs']} runs of same-sign residuals along X; "
              f"{fmt(info['expected_runs'], 3)} expected by chance ({p_phrase(info['p_value'])}, "
              f"{info['method']}).")
    if info["p_value"] < 0.05:
        return AssumptionCheck(
            name=name, statistic=info["runs"], p_value=info["p_value"], status="fail",
            detail=detail + f" The points deviate systematically from the {what}: the model is "
                            "probably the wrong shape for these data.")
    return AssumptionCheck(name=name, statistic=info["runs"], p_value=info["p_value"],
                           status="pass", detail=detail + " No sign of systematic deviation.")


def lack_of_fit_assumption(info: dict[str, Any] | None, what: str = "line") -> AssumptionCheck | None:
    name = "Model adequacy (replicates lack-of-fit F test)"
    if info is None:
        return None
    if info["p_value"] is None:
        return AssumptionCheck(name=name, status="warn",
                               detail="Replicates agree exactly at every X, so the lack-of-fit test "
                                      "cannot be computed.")
    detail = (f"F({info['df_lack_of_fit']}, {info['df_pure_error']}) = {fmt(info['f'], 4)}, "
              f"{p_phrase(info['p_value'])}.")
    if info["p_value"] < 0.05:
        return AssumptionCheck(
            name=name, statistic=info["f"], p_value=info["p_value"], status="fail",
            detail=detail + f" The {what} misses the replicate means by more than the scatter "
                            "among replicates explains: consider a different model.")
    return AssumptionCheck(name=name, statistic=info["f"], p_value=info["p_value"], status="pass",
                           detail=detail + " The scatter around the fit is consistent with the "
                                           "scatter among replicates.")


def homoscedasticity_assumption(info: dict[str, Any] | None) -> AssumptionCheck:
    name = "Constant scatter (Breusch-Pagan test)"
    if info is None:
        return AssumptionCheck(name=name, status="warn",
                               detail="Too few points (or a perfect fit) to test whether the scatter "
                                      "is constant.")
    detail = f"LM = {fmt(info['lm'], 4)}, df = {info['df']}, {p_phrase(info['p_value'])}."
    if info["p_value"] < 0.05:
        return AssumptionCheck(
            name=name, statistic=info["lm"], p_value=info["p_value"], status="fail",
            detail=detail + " The scatter of the points changes along the fit (heteroscedasticity). "
                            "Consider transforming Y (e.g. log) or a weighted fit.")
    return AssumptionCheck(name=name, statistic=info["lm"], p_value=info["p_value"], status="pass",
                           detail=detail + " No evidence that the scatter changes along the fit.")


def _independence(dw: float | None) -> AssumptionCheck:
    text = ("Each point must be an independent observation (not repeated measurements of the "
            "same sample, and not a time series where one value drives the next).")
    if dw is not None:
        text += (f" Durbin-Watson statistic for the residuals in X order: {fmt(dw, 3)} "
                 "(about 2 means no serial correlation; near 0 or 4 suggests neighbouring points "
                 "are related).")
    return AssumptionCheck(name="Independent observations", statistic=dw, status="info", detail=text)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _signed(value: float) -> str:
    return f"+ {fmt(value)}" if value >= 0 else f"- {fmt(abs(value))}"


# ====================================================== simple linear regression
@guarded
def linear_regression(
    x: Any,
    y: Any,
    *,
    names: tuple[str, str] = ("X", "Y"),
    alpha: float = 0.05,
    ci_level: float = 0.95,
    through_origin: bool = False,
    test_slope_against: float = 0.0,
) -> StatResult:
    """Ordinary least-squares straight line Y = intercept + slope * X.

    Closed form: slope = S_xy / S_xx, intercept = mean(y) - slope * mean(x),
    s^2 = SS_res / (n - 2), SE(slope) = s / sqrt(S_xx),
    SE(intercept) = s sqrt(1/n + mean(x)^2 / S_xx). Confidence band of the line at x0:
    yhat +/- t s sqrt(1/n + (x0 - mean(x))^2 / S_xx); prediction band adds 1 under the root.
    With ``through_origin=True`` the model is Y = slope * X (n - 1 df, uncentred R^2).
    The X-intercept (-intercept / slope) gets a Fieller (1954) confidence interval: the X
    values where the confidence band of the line includes Y = 0.
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    try:
        h0 = float(test_slope_against)
    except (TypeError, ValueError) as exc:
        raise StatsError("test_slope_against must be a number (usually 0).") from exc
    if not math.isfinite(h0):
        raise StatsError("test_slope_against must be a finite number (usually 0).")
    try:
        xa, ya, notes = clean_paired(x, y, names)
    except StatsError:
        raise
    except (TypeError, ValueError) as exc:
        raise StatsError(f"{names[0]} and {names[1]} must be lists of numbers.") from exc
    n = int(xa.size)
    n_par = 1 if through_origin else 2
    if n < n_par:
        raise StatsError(
            f"Linear regression needs at least {n_par} complete (X, Y) point(s); {n} found. "
            "Three or more are needed for standard errors and P values."
        )
    if through_origin:
        if not np.any(xa != 0):
            raise StatsError("All X values are zero, so a line through the origin has no defined slope.")
    elif is_constant(xa):
        raise StatsError(
            "All X values are identical, so the slope is undefined (the best-fit line would be "
            "vertical). Regression needs at least two different X values."
        )
    warns: list[str] = []
    order = np.argsort(xa, kind="stable")
    xa, ya = xa[order], ya[order]
    for arr in (xa, ya):
        top = float(np.max(np.abs(arr)))
        if top > 1e120 or (0 < top < 1e-120):
            raise StatsError(
                "The values are too large or too small to compute with reliably (beyond 1e120 "
                "or below 1e-120). Change the units of the data and try again."
            )
    xbar, ybar = float(xa.mean()), float(ya.mean())
    if through_origin:
        sxx = float(xa @ xa)
        slope = float(xa @ ya) / sxx
        intercept = 0.0
        ss_tot = float(ya @ ya)
    else:
        dx, dy = xa - xbar, ya - ybar
        sxx = float(dx @ dx)
        slope = float(dx @ dy) / sxx
        intercept = ybar - slope * xbar
        ss_tot = float(dy @ dy)
    fitted = intercept + slope * xa
    resid = ya - fitted
    ss_res = float(resid @ resid)
    scale_ref = max(ss_tot, float(ya @ ya), 1e-300)
    perfect = ss_res <= PERFECT_FIT_RATIO * scale_ref
    if perfect:
        ss_res = 0.0
    df = n - n_par
    constant_y = (not through_origin and is_constant(ya)) or ss_tot <= 0
    r2 = None if constant_y else max(0.0, 1.0 - ss_res / ss_tot)
    s2 = ss_res / df if df > 0 else None
    sy_x = math.sqrt(s2) if s2 is not None else None
    adj_r2 = None
    if r2 is not None and df > 0:
        adj_r2 = 1.0 - (ss_res / df) / (ss_tot / (n - (0 if through_origin else 1)))
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df)) if df > 0 else None

    se_slope = se_int = None
    if s2 is not None:
        se_slope = math.sqrt(s2 / sxx)
        if not through_origin:
            se_int = math.sqrt(s2 * (1.0 / n + xbar * xbar / sxx))

    def _t_and_p(estimate: float, se: float | None, null: float) -> tuple[float | None, float | None]:
        if se is None:
            return None, None
        if se == 0:
            if estimate == null:
                return None, None
            return None, 0.0
        t = (estimate - null) / se
        return t, float(2.0 * stats.t.sf(abs(t), df))

    t_slope, p_slope = _t_and_p(slope, se_slope, h0)
    t_int, p_int = _t_and_p(intercept, se_int, 0.0)

    def _ci(estimate: float, se: float | None) -> tuple[float | None, float | None]:
        if se is None or t_crit is None:
            return None, None
        return estimate - t_crit * se, estimate + t_crit * se

    slope_ci = _ci(slope, se_slope)
    int_ci = _ci(intercept, se_int)

    # ---- overall F test (slope = 0)
    ss_reg = max(ss_tot - ss_res, 0.0)
    f_stat = p_f = None
    if df > 0 and not constant_y:
        if ss_res > 0:
            f_stat = (ss_reg / 1.0) / (ss_res / df)
            p_f = float(stats.f.sf(f_stat, 1, df))
        else:
            p_f = 0.0

    # ---- degenerate-case messages
    if df == 0:
        warns.append(
            f"With only {n} point(s) the line passes exactly through the data: standard errors, "
            "confidence intervals and P values cannot be computed. Collect more points."
        )
    elif constant_y:
        warns.append("All Y values are identical: the best-fit slope is 0 and R squared is undefined.")
    elif perfect:
        warns.append("The points fall exactly on a straight line (perfect fit): standard errors are "
                     "zero. Check that Y was not computed from X.")
    if 0 < df < 3:
        warns.append(f"Only {n} points: the estimates are very imprecise.")

    # ---- X intercept with Fieller interval
    parameters = [
        Parameter(name="Slope", value=slope, se=se_slope, ci_low=slope_ci[0], ci_high=slope_ci[1],
                  ci_level=ci_level, statistic=t_slope, p_value=p_slope,
                  unit_note=f"Change in {names[1]} per one-unit increase in {names[0]}"
                            + (f"; tested against {fmt(h0)}" if h0 != 0 else "")),
    ]
    x_int = None
    if not through_origin:
        parameters.append(Parameter(
            name="Y-intercept", value=intercept, se=se_int, ci_low=int_ci[0], ci_high=int_ci[1],
            ci_level=ci_level, statistic=t_int, p_value=p_int,
            unit_note=f"Predicted {names[1]} when {names[0]} = 0"))
        if slope != 0:
            x_int = -intercept / slope
            lo = hi = None
            note = f"Where the line crosses {names[1]} = 0"
            if s2 is not None and t_crit is not None:
                g = t_crit * t_crit * s2
                qa = slope * slope - g / sxx
                qb = 2.0 * slope * intercept + 2.0 * g * xbar / sxx
                qc = intercept * intercept - g * (1.0 / n + xbar * xbar / sxx)
                disc = qb * qb - 4.0 * qa * qc
                if qa > 0 and disc >= 0:
                    root = math.sqrt(disc)
                    lo, hi = sorted(((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)))
                    note += "; CI by Fieller's theorem"
                else:
                    note += ("; no finite CI because the slope is not clearly different from zero")
            if x_int < float(xa.min()) or x_int > float(xa.max()):
                note += "; outside the range of the data (extrapolated)"
            parameters.append(Parameter(name="X-intercept", value=x_int, ci_low=lo, ci_high=hi,
                                        ci_level=ci_level, unit_note=note))
        else:
            parameters.append(Parameter(name="X-intercept", value=None, ci_level=ci_level,
                                        unit_note="Undefined: the line is horizontal"))

    # ---- curve + bands
    grid = np.linspace(float(xa.min()), float(xa.max()), GRID_POINTS)
    y_grid = intercept + slope * grid
    curve: dict[str, Any] = {"x": grid.tolist(), "y": y_grid.tolist()}
    if s2 is not None and t_crit is not None:
        if through_origin:
            leverage = grid * grid / sxx
        else:
            leverage = 1.0 / n + (grid - xbar) ** 2 / sxx
        half_ci = t_crit * np.sqrt(s2 * leverage)
        half_pred = t_crit * np.sqrt(s2 * (1.0 + leverage))
        curve.update({
            "ci_low": (y_grid - half_ci).tolist(), "ci_high": (y_grid + half_ci).tolist(),
            "pred_low": (y_grid - half_pred).tolist(), "pred_high": (y_grid + half_pred).tolist(),
        })

    # ---- diagnostics
    has_replicates = np.unique(xa).size < n
    if has_replicates:
        _, run_resid, _ = mean_by_x(xa, resid)
    else:
        run_resid = resid
    runs = None if perfect else runs_test(run_resid)
    lof = None if perfect else lack_of_fit(xa, ya, fitted, n_par)
    bp = None if (perfect or df < 2) else breusch_pagan(resid, xa)
    dw = None if perfect else durbin_watson(resid)
    assumptions = []
    if perfect or df == 0:
        assumptions.append(AssumptionCheck(
            name="Residual diagnostics", status="warn",
            detail="The line fits the points exactly, so there are no residuals to check for "
                   "normality, constant scatter or systematic deviation."))
    else:
        assumptions.append(residual_normality(resid))
        assumptions.append(homoscedasticity_assumption(bp))
        lof_check = lack_of_fit_assumption(lof, "straight line")
        if lof_check is not None:
            assumptions.append(lof_check)
        runs_check = runs_assumption(runs, "line")
        if has_replicates:
            runs_check.detail += " (Computed on the mean residual at each X.)"
        assumptions.append(runs_check)
    assumptions.append(AssumptionCheck(
        name="Linear relationship", status="info",
        detail="The model assumes Y changes by the same amount for every unit of X. Look at the "
               "residual plot: a curved pattern means a straight line is the wrong model."))
    assumptions.append(_independence(dw))
    assumptions.append(AssumptionCheck(
        name="X measured without error", status="info",
        detail="Ordinary regression assumes X is known precisely and all the scatter is in Y."))

    # ---- effect sizes
    effect_sizes = []
    r = None
    if r2 is not None:
        r2_lo = r2_hi = None
        method = None
        if f_stat is not None and not through_origin and df > 0:
            lo_hi = eta_squared_ci(f_stat, 1, df, ci_level)
            if not any(math.isnan(v) for v in lo_hi):
                r2_lo, r2_hi = lo_hi
                method = "noncentral F (Steiger 2004)"
        effect_sizes.append(EffectSize(
            name="R squared", value=r2, ci_low=r2_lo, ci_high=r2_hi, ci_level=ci_level,
            ci_method=method))
        if not through_origin:
            r = math.copysign(math.sqrt(r2), slope) if slope != 0 else 0.0
            r_lo = r_hi = None
            if n > 3 and abs(r) < 1:
                crit = float(stats.norm.ppf(0.5 + ci_level / 2.0))
                z = math.atanh(r)
                r_lo, r_hi = math.tanh(z - crit / math.sqrt(n - 3)), math.tanh(z + crit / math.sqrt(n - 3))
            effect_sizes.append(EffectSize(
                name="Pearson r", value=r, ci_low=r_lo, ci_high=r_hi, ci_level=ci_level,
                ci_method="Fisher z transformation" if r_lo is not None else None,
                magnitude=magnitude_r(r)))
    else:
        effect_sizes.append(EffectSize(name="R squared", value=None, ci_level=ci_level))

    if through_origin:
        equation = f"Y = {fmt(slope)}*X"
    else:
        equation = f"Y = {fmt(slope)}*X {_signed(intercept)}"

    anova = []
    if df > 0:
        anova = [
            {"source": "Regression", "ss": ss_reg, "df": 1, "ms": ss_reg, "f": f_stat, "p": p_f},
            {"source": "Residual", "ss": ss_res, "df": df, "ms": ss_res / df, "f": None, "p": None},
        ]
        if lof is not None:
            anova.append({"source": "  Lack of fit", "ss": lof["ss_lack_of_fit"],
                          "df": lof["df_lack_of_fit"],
                          "ms": lof["ss_lack_of_fit"] / lof["df_lack_of_fit"],
                          "f": lof["f"], "p": lof["p_value"]})
            anova.append({"source": "  Pure error", "ss": lof["ss_pure_error"],
                          "df": lof["df_pure_error"],
                          "ms": lof["ss_pure_error"] / lof["df_pure_error"], "f": None, "p": None})
        anova.append({"source": "Total", "ss": ss_tot, "df": n - (0 if through_origin else 1),
                      "ms": None, "f": None, "p": None})

    ic = information_criteria(ss_res, n, n_par)
    use_t_main = h0 != 0
    result = StatResult(
        analysis="linear_regression",
        test_name="Simple linear regression" + (" through the origin" if through_origin else ""),
        why_chosen="Linear regression finds the straight line that best predicts Y from X and "
                   "tests whether the slope differs from "
                   + ("zero (no relationship)." if h0 == 0 else f"{fmt(h0)}."),
        assumptions=assumptions,
        statistic_name="t" if use_t_main else "F",
        statistic=_finite_or_none(t_slope if use_t_main else f_stat),
        df=(float(df) if use_t_main else 1.0) if df > 0 else None,
        df2=None if use_t_main or df <= 0 else float(df),
        p_value=p_slope if use_t_main else p_f,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n={names[1]: n},
        descriptives=[{
            "group": names[1], "n": n, "n_x_values": int(np.unique(xa).size),
            "x_min": float(xa.min()), "x_max": float(xa.max()), "x_mean": xbar,
            "mean": ybar, "sd": float(ya.std(ddof=1)) if n > 1 else None,
        }],
        anova_table=anova,
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_OLS, _CITE_FIELLER, _CITE_RUNS, _CITE_BP],
        extra={
            "curve": curve,
            "residuals": {"x": xa.tolist(), "fitted": fitted.tolist(), "residual": resid.tolist()},
            "equation": equation,
            "model": "linear",
            "x_is_log": False,
            "goodness_of_fit": {
                "df": df, "r_squared": r2, "adj_r_squared": adj_r2, "ss_residual": ss_res,
                "ss_total": ss_tot, "sy_x": sy_x, "aic": ic["aic"], "aicc": ic["aicc"],
                "bic": ic["bic"], "n": n, "n_parameters": n_par,
            },
            "runs_test": runs,
            "lack_of_fit": lof,
            "breusch_pagan": bp,
            "durbin_watson": dw,
            "x_name": names[0],
            "y_name": names[1],
            "through_origin": bool(through_origin),
        },
    )
    # ---- words
    slope_text = f"{fmt(slope)}"
    if slope_ci[0] is not None:
        slope_text += f" ({ci_phrase(slope_ci[0], slope_ci[1], ci_level)})"
    if df == 0:
        result.interpretation = (
            f"The line through the {n} point(s) is {equation}. With no points to spare, nothing "
            "can be said about how reliable the slope is."
        )
    elif constant_y:
        result.interpretation = (
            f"{names[1]} is the same at every {names[0]}, so the best-fit line is flat "
            f"({equation}) and there is no relationship to test."
        )
    else:
        direction = "increases" if slope > 0 else "decreases"
        tested = ("differs from zero" if h0 == 0 else f"differs from {fmt(h0)}")
        p_main = p_slope
        result.interpretation = (
            f"The best-fit line is {equation}: {names[1]} {direction} by {fmt(abs(slope))} for each "
            f"one-unit increase in {names[0]} (slope {slope_text}). The test of whether the slope "
            f"{tested} is {sig_phrase(p_main, alpha)}. R squared = {fmt(r2, 3)}, meaning the line "
            f"accounts for about {fmt(100 * r2, 3)}% of the variation in {names[1]}"
            + (" (R squared is computed about zero for a line forced through the origin, so it is "
               "not comparable with an ordinary R squared)." if through_origin else ".")
            + " Predictions outside the range of the data are extrapolations."
        )
    result.methods_text = (
        f"{names[1]} was regressed on {names[0]} by ordinary least-squares linear regression"
        + (" constrained through the origin" if through_origin else "")
        + f"; the slope was tested against {fmt(h0)} with a two-tailed "
        + ("t test" if use_t_main else "F test")
        + f" (alpha = {alpha:g}), and {ci_level * 100:g}% confidence intervals are reported."
    )
    return result


# ============================================================== compare slopes
@guarded
def compare_slopes(
    datasets: dict[str, tuple[Any, Any]],
    *,
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> StatResult:
    """Do two or more regression lines have the same slope (and, if so, the same elevation)?

    Analysis-of-covariance comparison of nested linear models (Zar JH, 2010,
    *Biostatistical Analysis*, 5th ed., ch. 18; Draper & Smith 1998 extra-sum-of-squares
    principle):

    1. separate slope and intercept per data set (SS_sep, N - 2k df);
    2. one common slope, separate intercepts (SS_common, N - k - 1 df);
    3. one line for all data (SS_one, N - 2 df).

    Slopes: F = [(SS_common - SS_sep) / (k - 1)] / [SS_sep / (N - 2k)].
    Elevations (only meaningful when the slopes do not differ):
    F = [(SS_one - SS_common) / (k - 1)] / [SS_common / (N - k - 1)].
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    if not isinstance(datasets, dict) or len(datasets) < 2:
        raise StatsError("Comparing slopes needs at least two data sets.")
    cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    notes: list[str] = []
    for name, pair in datasets.items():
        try:
            xa, ya, n_notes = clean_paired(pair[0], pair[1], ("X", str(name)))
        except StatsError:
            raise
        except (TypeError, ValueError, IndexError) as exc:
            raise StatsError(f"Data set '{name}' must be a pair of (X values, Y values).") from exc
        notes.extend(f"{name}: {t}" for t in n_notes)
        if xa.size < 2 or is_constant(xa):
            raise StatsError(
                f"Data set '{name}' needs at least two points with different X values to "
                "have a slope."
            )
        cleaned[str(name)] = (xa, ya)
    k = len(cleaned)
    n_total = sum(xa.size for xa, _ in cleaned.values())
    df_sep = n_total - 2 * k
    if df_sep < 1:
        raise StatsError(
            "Too few points to compare slopes: the data sets together need more than "
            f"{2 * k} points."
        )
    ss_sep = 0.0
    sxx_sum = sxy_sum = 0.0
    rows = []
    for name, (xa, ya) in cleaned.items():
        dx, dy = xa - xa.mean(), ya - ya.mean()
        sxx, sxy, syy = float(dx @ dx), float(dx @ dy), float(dy @ dy)
        b = sxy / sxx
        ss = max(syy - b * sxy, 0.0)
        ss_sep += ss
        sxx_sum += sxx
        sxy_sum += sxy
        rows.append({"group": name, "n": int(xa.size), "slope": b,
                     "intercept": float(ya.mean() - b * xa.mean()),
                     "x_mean": float(xa.mean()), "mean": float(ya.mean()),
                     "ss_residual": ss, "sxx": sxx})
    common_slope = sxy_sum / sxx_sum
    ss_common = sum(
        float(np.sum(((ya - ya.mean()) - common_slope * (xa - xa.mean())) ** 2))
        for xa, ya in cleaned.values()
    )
    all_x = np.concatenate([xa for xa, _ in cleaned.values()])
    all_y = np.concatenate([ya for _, ya in cleaned.values()])
    dxa, dya = all_x - all_x.mean(), all_y - all_y.mean()
    one_slope = float(dxa @ dya) / float(dxa @ dxa)
    ss_one = float(np.sum((dya - one_slope * dxa) ** 2))
    df_common = n_total - k - 1
    warns: list[str] = []
    f_slope = p_slope = f_elev = p_elev = None
    ss_scale = max(float(dya @ dya), 1e-300)
    if ss_sep > PERFECT_FIT_RATIO * ss_scale:
        f_slope = ((ss_common - ss_sep) / (k - 1)) / (ss_sep / df_sep)
        f_slope = max(f_slope, 0.0)
        p_slope = float(stats.f.sf(f_slope, k - 1, df_sep))
    else:
        warns.append("Every data set lies exactly on its own line (zero residual scatter), so the "
                     "F tests cannot be computed.")
    if ss_common > PERFECT_FIT_RATIO * ss_scale:
        f_elev = max(((ss_one - ss_common) / (k - 1)) / (ss_common / df_common), 0.0)
        p_elev = float(stats.f.sf(f_elev, k - 1, df_common))
    se_common = math.sqrt(ss_common / df_common / sxx_sum) if df_common > 0 else None
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df_common)) if df_common > 0 else None
    parameters = [
        Parameter(name=f"Slope ({row['group']})", value=row["slope"],
                  se=(math.sqrt(row["ss_residual"] / (row["n"] - 2) / row["sxx"])
                      if row["n"] > 2 else None),
                  ci_level=ci_level)
        for row in rows
    ]
    for par, row in zip(parameters, rows):
        if par.se is not None:
            tc = float(stats.t.ppf(0.5 + ci_level / 2.0, row["n"] - 2))
            par.ci_low, par.ci_high = par.value - tc * par.se, par.value + tc * par.se
    parameters.append(Parameter(
        name="Common (pooled) slope", value=common_slope, se=se_common,
        ci_low=None if se_common is None else common_slope - t_crit * se_common,
        ci_high=None if se_common is None else common_slope + t_crit * se_common,
        ci_level=ci_level, unit_note="Meaningful only if the slopes do not differ"))
    slopes_differ = p_slope is not None and p_slope < alpha
    anova = [
        {"source": "Difference between slopes", "ss": ss_common - ss_sep, "df": k - 1,
         "ms": (ss_common - ss_sep) / (k - 1), "f": f_slope, "p": p_slope},
        {"source": "Residual (separate lines)", "ss": ss_sep, "df": df_sep,
         "ms": ss_sep / df_sep, "f": None, "p": None},
        {"source": "Difference between elevations", "ss": ss_one - ss_common, "df": k - 1,
         "ms": (ss_one - ss_common) / (k - 1), "f": f_elev, "p": p_elev},
        {"source": "Residual (common slope)", "ss": ss_common, "df": df_common,
         "ms": ss_common / df_common, "f": None, "p": None},
    ]
    names = list(cleaned)
    effect = EffectSize(name="partial eta squared (slopes)", value=None, ci_level=ci_level)
    if f_slope is not None:
        value = (f_slope * (k - 1)) / (f_slope * (k - 1) + df_sep)
        lo, hi = eta_squared_ci(f_slope, k - 1, df_sep, ci_level)
        effect = EffectSize(
            name="partial eta squared (slopes)", value=value,
            ci_low=None if math.isnan(lo) else lo, ci_high=None if math.isnan(hi) else hi,
            ci_level=ci_level, ci_method="noncentral F (Steiger 2004)")
    resid_all = np.concatenate([
        ya - (row["intercept"] + row["slope"] * xa)
        for (xa, ya), row in zip(cleaned.values(), rows)
    ])
    result = StatResult(
        analysis="compare_slopes",
        test_name="Comparison of regression slopes (analysis of covariance)",
        why_chosen="Use this to ask whether the relationship between X and Y is equally steep in "
                   "two or more data sets, and if so whether one line sits higher than the other.",
        assumptions=[
            residual_normality(resid_all),
            AssumptionCheck(
                name="Similar scatter in every data set", status="info",
                detail="The test pools the scatter around each line, which assumes it is similar "
                       "in all data sets. Compare the residual plots."),
            AssumptionCheck(
                name="Linear relationships", status="info",
                detail="Each data set must be reasonably described by a straight line."),
            AssumptionCheck(
                name="Independent observations", status="info",
                detail="Every point must be an independent observation."),
        ],
        statistic_name="F",
        statistic=_finite_or_none(f_slope),
        df=float(k - 1),
        df2=float(df_sep),
        p_value=p_slope,
        alpha=alpha,
        effect_sizes=[effect],
        parameters=parameters,
        n={row["group"]: row["n"] for row in rows},
        descriptives=[{key: row[key] for key in ("group", "n", "slope", "intercept", "x_mean", "mean")}
                      for row in rows],
        anova_table=anova,
        warnings=warns,
        data_notes=notes,
        citations=["Zar JH (2010) Biostatistical Analysis, 5th ed., Pearson, ch. 18", _CITE_OLS],
        extra={
            "slopes": {row["group"]: row["slope"] for row in rows},
            "intercepts": {row["group"]: row["intercept"] for row in rows},
            "common_slope": common_slope,
            "elevation_test": {"f": f_elev, "df": k - 1, "df2": df_common, "p_value": p_elev,
                               "valid": not slopes_differ},
            "ss": {"separate": ss_sep, "common_slope": ss_common, "one_line": ss_one},
        },
    )
    listing = ", ".join(f"{row['group']}: {fmt(row['slope'])}" for row in rows)
    if p_slope is None:
        result.interpretation = (
            f"The slopes are {listing}. Because the points lie exactly on their lines, the "
            "difference between slopes cannot be tested."
        )
    elif slopes_differ:
        result.interpretation = (
            f"The slopes ({listing}) differ between the data sets: F({k - 1}, {df_sep}) = "
            f"{fmt(f_slope)}, which is {sig_phrase(p_slope, alpha)}. There is evidence that the "
            "relationship between X and Y is not equally steep, so comparing the elevations "
            "(intercepts) of the lines is not meaningful."
        )
    else:
        elev = ("could not be tested" if p_elev is None
                else f"is {sig_phrase(p_elev, alpha)} (F({k - 1}, {df_common}) = {fmt(f_elev)})")
        result.interpretation = (
            f"The slopes ({listing}) do not differ detectably: F({k - 1}, {df_sep}) = "
            f"{fmt(f_slope)}, which is {sig_phrase(p_slope, alpha)}. Assuming one common slope of "
            f"{fmt(common_slope)}, the difference in elevation between the lines {elev}. A "
            "non-significant difference in slopes is not proof that the slopes are equal."
        )
    result.methods_text = (
        f"Slopes of the regression lines of {', '.join(names)} were compared by analysis of "
        f"covariance (extra-sum-of-squares F test, alpha = {alpha:g}); elevations were compared "
        "under a common slope when the slopes did not differ."
    )
    return result


# ========================================================= multiple regression
def _as_interactions(interactions: Any) -> list[tuple[str, str]]:
    if interactions is None or interactions == "":
        return []
    if isinstance(interactions, str):
        interactions = [part for part in interactions.replace(";", ",").split(",") if part.strip()]
    out = []
    for item in interactions:
        if isinstance(item, str):
            parts = [p.strip() for p in item.replace("*", ":").split(":")]
        else:
            parts = [str(p) for p in item]
        if len(parts) != 2 or not all(parts):
            raise StatsError(
                f"Interaction '{item}' is not understood. Write two predictor names joined by "
                "'*', for example 'Dose*Sex'."
            )
        out.append((parts[0], parts[1]))
    return out


def _build_design(
    df: pd.DataFrame, predictors: Sequence[str], interactions: list[tuple[str, str]]
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    """Design columns (without the constant), {term: its columns}, notes about coding."""
    notes: list[str] = []
    term_columns: dict[str, list[str]] = {}
    columns: dict[str, np.ndarray] = {}
    for pred in predictors:
        series = df[pred]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            columns[pred] = series.to_numpy(dtype=float)
            term_columns[pred] = [pred]
            continue
        labels = series.astype(str)
        levels = sorted(labels.unique())
        if len(levels) < 2:
            raise StatsError(
                f"Categorical predictor '{pred}' has only one level ('{levels[0]}') in the rows "
                "used, so its effect cannot be estimated. Remove it."
            )
        names = []
        for level in levels[1:]:
            col = f"{pred}[{level}]"
            columns[col] = (labels == level).to_numpy(dtype=float)
            names.append(col)
        term_columns[pred] = names
        notes.append(
            f"'{pred}' is categorical and was dummy-coded with '{levels[0]}' as the reference "
            f"level: each coefficient is the difference from '{levels[0]}'."
        )
    for a, b in interactions:
        for name in (a, b):
            if name not in term_columns:
                raise StatsError(
                    f"Interaction '{a}*{b}' uses '{name}', which is not among the predictors."
                )
        if a == b:
            raise StatsError(f"An interaction needs two different predictors (got '{a}*{b}').")
        names = []
        for col_a in term_columns[a]:
            for col_b in term_columns[b]:
                col = f"{col_a}:{col_b}"
                columns[col] = columns[col_a] * columns[col_b]
                names.append(col)
        term_columns[f"{a}:{b}"] = names
    return pd.DataFrame(columns), term_columns, notes


def _vif(design: np.ndarray) -> list[float | None]:
    """Variance inflation factor 1 / (1 - R_j^2) of each column regressed on the others."""
    n, k = design.shape
    if k == 1:
        return [1.0]
    out: list[float | None] = []
    for j in range(k):
        target = design[:, j]
        others = np.column_stack([np.ones(n), np.delete(design, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        centred = target - target.mean()
        ss_tot = float(centred @ centred)
        ss_res = float(resid @ resid)
        if ss_tot <= 0:
            out.append(None)
        elif ss_res <= 1e-14 * ss_tot:
            out.append(math.inf)
        else:
            out.append(ss_tot / ss_res)
    return out


@guarded
def multiple_regression(
    df: pd.DataFrame,
    y: str,
    predictors: Sequence[str],
    *,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    interactions: Any = None,
    standardize: bool = False,
) -> StatResult:
    """Multiple linear regression Y = b0 + b1 X1 + b2 X2 + ... by ordinary least squares.

    Categorical (text) predictors are dummy-coded automatically (first level in alphabetical
    order = reference). ``interactions`` is a list of pairs such as ``[("Dose", "Sex")]`` or
    text such as ``"Dose*Sex"``; product columns are added for each pair.
    ``standardize=True`` adds standardized coefficients (beta = b * SD(x) / SD(y)) so
    predictors measured in different units can be compared.
    """
    alpha = check_alpha(alpha)
    ci_level = check_level(ci_level)
    if not isinstance(df, pd.DataFrame):
        raise StatsError("Multiple regression needs a table with one column per variable.")
    if isinstance(predictors, str):
        predictors = [predictors]
    predictors = [str(p) for p in (predictors or [])]
    y = str(y)
    if not predictors:
        raise StatsError("Choose at least one predictor variable.")
    if len(set(predictors)) != len(predictors):
        raise StatsError("Each predictor may be listed only once.")
    missing = [c for c in [y, *predictors] if c not in df.columns]
    if missing:
        raise StatsError(f"Unknown column(s): {', '.join(missing)}. Available: "
                         f"{', '.join(map(str, df.columns))}.")
    if y in predictors:
        raise StatsError(f"'{y}' cannot be both the outcome and a predictor.")
    if not pd.api.types.is_numeric_dtype(df[y]) or pd.api.types.is_bool_dtype(df[y]):
        raise StatsError(f"The outcome '{y}' must be numeric. For a yes/no outcome use logistic "
                         "regression instead.")
    pairs = _as_interactions(interactions)
    work = df[[y, *predictors]].copy()
    numeric_cols = [c for c in work.columns
                    if pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_bool_dtype(work[c])]
    work[numeric_cols] = work[numeric_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    before = len(work)
    work = work.dropna(axis=0, how="any").reset_index(drop=True)
    notes: list[str] = []
    if len(work) < before:
        notes.append(f"{before - len(work)} row(s) with a missing value were ignored.")
    design_df, term_columns, coding_notes = (
        _build_design(work, predictors, pairs) if len(work) else (pd.DataFrame(), {}, [])
    )
    notes.extend(coding_notes)
    n = len(work)
    k = design_df.shape[1]
    if n < k + 2:
        raise StatsError(
            f"Multiple regression with {k} coefficient(s) plus an intercept needs at least "
            f"{k + 2} complete rows; {n} available. Use fewer predictors or more data."
        )
    yv = work[y].to_numpy(dtype=float)
    xmat = design_df.to_numpy(dtype=float)
    full = np.column_stack([np.ones(n), xmat])
    scaled = full / np.maximum(np.linalg.norm(full, axis=0), 1e-300)
    if np.linalg.matrix_rank(scaled) < full.shape[1]:
        raise StatsError(
            "The predictors are perfectly collinear: at least one of them is constant or can be "
            "computed exactly from the others, so its effect cannot be separated. Remove the "
            "redundant predictor."
        )
    warns: list[str] = []
    constant_y = is_constant(yv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.OLS(yv, full).fit()
    df_resid = int(fit.df_resid)
    df_model = int(k)
    resid = np.asarray(fit.resid, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    perfect = ss_res <= PERFECT_FIT_RATIO * max(ss_tot, float(yv @ yv), 1e-300)
    if perfect:
        ss_res = 0.0
    r2 = None if constant_y else max(0.0, 1.0 - ss_res / ss_tot)
    adj_r2 = None if r2 is None else 1.0 - (ss_res / df_resid) / (ss_tot / (n - 1))
    s2 = ss_res / df_resid
    f_stat = p_f = None
    if not constant_y:
        if ss_res > 0:
            f_stat = ((ss_tot - ss_res) / df_model) / s2
            p_f = float(stats.f.sf(f_stat, df_model, df_resid))
        else:
            p_f = 0.0
    if constant_y:
        warns.append(f"All values of {y} are identical: there is nothing for the predictors to explain.")
    elif perfect:
        warns.append("The model reproduces the data exactly (perfect fit): standard errors are zero. "
                     "Check that the outcome was not computed from the predictors.")
    if df_resid < 5 * (k + 1) and not perfect:
        warns.append(
            f"Only {n} rows for {k + 1} coefficients. A common rule of thumb is at least 10 rows "
            "per predictor; with fewer, the model is likely to overfit."
        )
    # coefficients from the statsmodels fit; SE recomputed from s2 so the perfect-fit case is clean
    params = np.asarray(fit.params, dtype=float)
    try:
        # (X'X)^-1 from the SVD of the column-scaled design (avoids squaring the condition number)
        _, sv, vt = np.linalg.svd(scaled, full_matrices=False)
        norms = np.linalg.norm(full, axis=0)
        cov_unscaled = ((vt.T / sv**2) @ vt) / np.outer(norms, norms)
    except np.linalg.LinAlgError:  # pragma: no cover - rank was checked above
        cov_unscaled = np.asarray(fit.normalized_cov_params, dtype=float)
    se = np.sqrt(np.maximum(np.diag(cov_unscaled) * s2, 0.0))
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df_resid))
    col_names = ["Intercept", *design_df.columns]
    vifs = _vif(xmat)
    sd_y = float(yv.std(ddof=1))
    parameters: list[Parameter] = []
    coef_rows: list[dict[str, Any]] = []
    for j, name in enumerate(col_names):
        b = float(params[j])
        sej = float(se[j])
        if sej > 0:
            t = b / sej
            p = float(2.0 * stats.t.sf(abs(t), df_resid))
        else:
            t = None
            p = None if b == 0 else 0.0
        lo, hi = b - t_crit * sej, b + t_crit * sej
        std_beta = None
        if j > 0 and sd_y > 0:
            std_beta = b * float(xmat[:, j - 1].std(ddof=1)) / sd_y
        vif = None if j == 0 else vifs[j - 1]
        parameters.append(Parameter(name=name, value=b, se=sej, ci_low=lo, ci_high=hi,
                                    ci_level=ci_level, statistic=t, p_value=p))
        coef_rows.append({
            "term": name, "estimate": b, "se": sej, "t": t, "p": p, "ci_low": lo, "ci_high": hi,
            "standardized_beta": std_beta if (standardize or j == 0) else None,
            "vif": None if vif is None or not math.isfinite(vif) else vif,
            "significant": None if p is None else bool(p < alpha),
        })
        if standardize and std_beta is not None:
            parameters.append(Parameter(
                name=f"Standardized beta: {name}", value=std_beta, ci_level=ci_level,
                unit_note="b * SD(predictor) / SD(outcome); compares predictors on one scale"))
    # ---- multicollinearity
    finite_vifs = [(name, v) for name, v in zip(design_df.columns, vifs) if v is not None]
    high = [(name, v) for name, v in finite_vifs if v > 10]
    if k == 1:
        vif_check = AssumptionCheck(name="No multicollinearity (VIF)", status="pass",
                                    detail="Only one predictor column, so collinearity cannot occur.")
    elif high:
        worst = ", ".join(f"{name} (VIF {fmt(v, 3)})" for name, v in high)
        vif_check = AssumptionCheck(
            name="No multicollinearity (VIF)", statistic=max(v for _, v in high), status="fail",
            detail=f"Strongly correlated predictors: {worst}. Their individual coefficients and P "
                   "values are unstable; remove or combine overlapping predictors."
                   + (" (Interaction terms are expected to correlate with their components; "
                      "centring the predictors reduces this.)" if pairs else ""))
        warns.append(f"Multicollinearity: variance inflation factor above 10 for {worst}.")
    else:
        top = max(v for _, v in finite_vifs) if finite_vifs else None
        vif_check = AssumptionCheck(
            name="No multicollinearity (VIF)", statistic=top,
            status="pass" if top is not None and top <= 5 else "warn",
            detail=f"Largest variance inflation factor: {fmt(top, 3)} (above 5 is a caution, above "
                   "10 a problem).")
    bp = None if perfect else breusch_pagan(resid, xmat)
    dw = None if perfect else durbin_watson(resid)
    assumptions = []
    if perfect:
        assumptions.append(AssumptionCheck(
            name="Residual diagnostics", status="warn",
            detail="The fit is exact, so there are no residuals to check."))
    else:
        assumptions.append(residual_normality(resid))
        assumptions.append(homoscedasticity_assumption(bp))
    assumptions.append(vif_check)
    assumptions.append(AssumptionCheck(
        name="Linear, additive effects", status="info",
        detail="Each numeric predictor is assumed to change the outcome by a constant amount per "
               "unit, regardless of the other predictors (unless an interaction is included). "
               "Check the residual-versus-fitted plot for curvature."))
    assumptions.append(_independence(dw))

    effect_sizes = []
    if r2 is not None:
        lo = hi = None
        method = None
        if f_stat is not None:
            lo_hi = eta_squared_ci(f_stat, df_model, df_resid, ci_level)
            if not any(math.isnan(v) for v in lo_hi):
                lo, hi = lo_hi
                method = "noncentral F (Steiger 2004)"
        effect_sizes.append(EffectSize(name="R squared", value=r2, ci_low=lo, ci_high=hi,
                                       ci_level=ci_level, ci_method=method))
        effect_sizes.append(EffectSize(name="adjusted R squared", value=adj_r2, ci_level=ci_level))
        if r2 < 1:
            effect_sizes.append(EffectSize(name="Cohen's f squared", value=r2 / (1.0 - r2),
                                           ci_level=ci_level))
    else:
        effect_sizes.append(EffectSize(name="R squared", value=None, ci_level=ci_level))
    ic = information_criteria(ss_res, n, k + 1)
    anova = [
        {"source": "Regression", "ss": ss_tot - ss_res, "df": df_model,
         "ms": (ss_tot - ss_res) / df_model, "f": f_stat, "p": p_f},
        {"source": "Residual", "ss": ss_res, "df": df_resid, "ms": s2, "f": None, "p": None},
        {"source": "Total", "ss": ss_tot, "df": n - 1, "ms": None, "f": None, "p": None},
    ]
    pieces = [fmt(params[0])] + [f"{_signed(float(b))}*{name}"
                                 for b, name in zip(params[1:], design_df.columns)]
    equation = f"{y} = " + " ".join(pieces)
    result = StatResult(
        analysis="multiple_regression",
        test_name="Multiple linear regression",
        why_chosen="Multiple regression estimates how an outcome depends on several predictors at "
                   "once, giving the effect of each one while holding the others constant.",
        assumptions=assumptions,
        statistic_name="F",
        statistic=_finite_or_none(f_stat),
        df=float(df_model),
        df2=float(df_resid),
        p_value=p_f,
        alpha=alpha,
        effect_sizes=effect_sizes,
        parameters=parameters,
        n={y: n},
        descriptives=[{"group": y, "n": n, "mean": float(yv.mean()), "sd": sd_y,
                       "n_predictors": len(predictors), "n_coefficients": k + 1}],
        anova_table=anova,
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_OLS, "Belsley DA, Kuh E, Welsch RE (1980) Regression Diagnostics, Wiley",
                   _CITE_BP, _CITE_AICC],
        extra={
            "coefficients": coef_rows,
            "terms": term_columns,
            "residuals": {"x": fitted.tolist(), "fitted": fitted.tolist(), "residual": resid.tolist()},
            "predicted_vs_actual": {"actual": yv.tolist(), "predicted": fitted.tolist()},
            "equation": equation,
            "model": "multiple_linear",
            "goodness_of_fit": {
                "df": df_resid, "r_squared": r2, "adj_r_squared": adj_r2, "ss_residual": ss_res,
                "ss_total": ss_tot, "sy_x": math.sqrt(s2), "aic": ic["aic"], "aicc": ic["aicc"],
                "bic": ic["bic"], "n": n, "n_parameters": k + 1,
            },
            "vif": {name: (None if v is None or not math.isfinite(v) else v)
                    for name, v in zip(design_df.columns, vifs)},
            "breusch_pagan": bp,
            "durbin_watson": dw,
            "outcome": y,
            "predictors": list(predictors),
        },
    )
    if constant_y:
        result.interpretation = (
            f"{y} does not vary in these data, so no predictor can explain it."
        )
    else:
        sig = [row["term"] for row in coef_rows[1:] if row["significant"]]
        sig_text = (f"Predictors with coefficients significantly different from zero: "
                    f"{', '.join(sig)}." if sig else
                    "No individual coefficient is significantly different from zero.")
        result.interpretation = (
            f"Together, the {len(predictors)} predictor(s) explain {fmt(100 * r2, 3)}% of the "
            f"variation in {y} (R squared = {fmt(r2, 3)}, adjusted R squared = {fmt(adj_r2, 3)}); "
            f"the overall model is {sig_phrase(p_f, alpha)}. {sig_text} Each coefficient is the "
            f"expected change in {y} for a one-unit increase in that predictor with the others "
            "held constant; this describes association, not necessarily cause."
        )
    result.methods_text = (
        f"{y} was modelled by multiple linear regression (ordinary least squares) on "
        f"{', '.join(predictors)}"
        + (f" with interaction(s) {', '.join(f'{a}*{b}' for a, b in pairs)}" if pairs else "")
        + f"; coefficients were tested with two-tailed t tests (alpha = {alpha:g}) and "
        "multicollinearity was assessed with variance inflation factors."
    )
    return result


# ================================================================ table adapters
def _series_name(table: Any, series: Any) -> str:
    names = table.series_names
    if series is None or series == "":
        return names[0]
    if str(series) not in names:
        raise StatsError(f"Unknown series '{series}'. Available: {', '.join(names)}.")
    return str(series)


def xy_series_points(table: Any, series: Any, use_means: bool) -> tuple[str, np.ndarray, np.ndarray,
                                                                         list[str], pd.DataFrame]:
    """(series name, x, y, notes, per-X summary) for one series of an XY table."""
    try:
        name = _series_name(table, series)
        means = table.xy_means(name)
        n_rep = len(table.series_map[name])
        notes: list[str] = []
        if use_means:
            xv = means["x"].to_numpy(dtype=float)
            yv = means["mean"].to_numpy(dtype=float)
            if n_rep > 1:
                notes.append(f"The mean of the replicates of {name} at each X was analysed "
                             "(one point per X).")
        else:
            xv, yv = table.xy_points(name)
            if n_rep > 1:
                notes.append(f"Every replicate of {name} was used as an individual point "
                             f"({xv.size} points at {len(means)} X values).")
    except TableError as exc:
        raise StatsError(str(exc)) from exc
    return name, xv, yv, notes, means


_ALPHA = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold for calling a result significant. 0.05 is the usual choice.")
CI_LEVEL_PARAM = ParamSpec(
    "ci_level", "Confidence level", kind="float", default=0.95,
    help="Confidence level of the reported intervals and bands. 0.95 gives 95% confidence "
         "intervals.")


@register(
    key="linear_regression", label="Simple linear regression", category="Regression",
    tables=("xy",),
    params=(
        ParamSpec("series", "Y series", kind="group", default=None, optional=True,
                  help="Which Y data set to fit. Leave empty to use the first one."),
        ParamSpec("use_means", "Fit the mean of replicates", kind="bool", default=False,
                  help="By default every replicate is used as its own point, which is right when "
                       "replicates are independent samples. Tick this to fit one mean per X instead "
                       "(for example when replicates are repeat readings of the same sample)."),
        ParamSpec("through_origin", "Force the line through the origin", kind="bool", default=False,
                  help="Tick only if theory says Y must be exactly 0 when X is 0. Otherwise leave "
                       "it off and let the data decide the intercept."),
        ParamSpec("test_slope_against", "Test the slope against", kind="float", default=0.0,
                  help="The slope value assumed by the null hypothesis. 0 asks 'is there any "
                       "relationship?'."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Fits the best straight line through X-Y data and reports the slope, intercept, "
                "how well the line fits, and bands showing the uncertainty.",
    citation=_CITE_OLS,
)
def _run_linear_regression(table: Any, *, series: str | None = None, use_means: bool = False,
                           through_origin: bool = False, test_slope_against: float = 0.0,
                           alpha: float = 0.05, ci_level: float = 0.95) -> StatResult:
    name, xv, yv, notes, _ = xy_series_points(table, series, bool(use_means))
    result = linear_regression(xv, yv, names=(table.x_name, name), alpha=alpha,
                               ci_level=ci_level, through_origin=bool(through_origin),
                               test_slope_against=test_slope_against)
    result.data_notes.extend(notes)
    result.extra["series"] = name
    return result


@register(
    key="compare_slopes", label="Compare slopes of regression lines", category="Regression",
    tables=("xy",),
    params=(
        ParamSpec("series", "Y series to compare", kind="groups", default=None, optional=True,
                  help="Which data sets to compare. Leave empty to compare all of them."),
        ParamSpec("use_means", "Fit the mean of replicates", kind="bool", default=False,
                  help="Tick to use one mean per X instead of every replicate."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Tests whether two or more straight lines have the same slope and, if they do, "
                "whether one line sits higher than the other.",
    citation="Zar JH (2010) Biostatistical Analysis, 5th ed., ch. 18",
)
def _run_compare_slopes(table: Any, *, series: list[str] | None = None, use_means: bool = False,
                        alpha: float = 0.05, ci_level: float = 0.95) -> StatResult:
    if isinstance(series, str):
        series = [s.strip() for s in series.split(",") if s.strip()]
    chosen = list(series) if series else list(table.series_names)
    if len(chosen) < 2:
        raise StatsError("Comparing slopes needs at least two Y series in the table.")
    datasets = {}
    notes: list[str] = []
    for item in chosen:
        name, xv, yv, n_notes, _ = xy_series_points(table, item, bool(use_means))
        datasets[name] = (xv, yv)
        notes.extend(n_notes)
    result = compare_slopes(datasets, alpha=alpha, ci_level=ci_level)
    result.data_notes.extend(notes)
    return result


@register(
    key="multiple_regression", label="Multiple linear regression", category="Regression",
    tables=("multiple_variables",),
    params=(
        ParamSpec("y", "Outcome (Y)", kind="column", default=None, optional=True,
                  help="The numeric variable you want to predict or explain. Leave empty to use "
                       "the first numeric column."),
        ParamSpec("predictors", "Predictors (X variables)", kind="columns", default=None,
                  optional=True,
                  help="The variables used to explain the outcome. Text columns are treated as "
                       "categories automatically. Leave empty to use every other column."),
        ParamSpec("interactions", "Interactions", kind="text", default="", optional=True,
                  help="Optional. Pairs of predictors whose effects may depend on each other, "
                       "written like 'Dose*Sex' and separated by commas."),
        ParamSpec("standardize", "Report standardized coefficients", kind="bool", default=False,
                  help="Adds coefficients rescaled to standard-deviation units so predictors "
                       "measured in different units can be compared."),
        _ALPHA,
        CI_LEVEL_PARAM,
    ),
    description="Explains one numeric outcome from several predictors at once, estimating the "
                "effect of each predictor while accounting for the others.",
    citation=_CITE_OLS,
)
def _run_multiple_regression(table: Any, *, y: str | None = None,
                             predictors: list[str] | None = None, interactions: Any = "",
                             standardize: bool = False, alpha: float = 0.05,
                             ci_level: float = 0.95) -> StatResult:
    try:
        frame = table.frame(dropna=False)
        numeric = table.numeric_columns
    except TableError as exc:
        raise StatsError(str(exc)) from exc
    if y is None or y == "":
        if not numeric:
            raise StatsError("The table has no numeric column to use as the outcome.")
        y = numeric[0]
    if isinstance(predictors, str):
        predictors = [p.strip() for p in predictors.split(",") if p.strip()]
    if not predictors:
        predictors = [c for c in frame.columns if c != y]
    return multiple_regression(frame, str(y), list(predictors), alpha=alpha, ci_level=ci_level,
                               interactions=interactions, standardize=bool(standardize))
