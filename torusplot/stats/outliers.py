"""Outlier detection. Outliers are only ever FLAGGED; nothing is removed or changed.

Whether a flagged value is excluded is always the user's explicit decision (and must be
reported). Results list the flagged values in ``extra["flagged"]`` as
``{"group", "row", "value"}`` (``row`` = zero-based row of the data table) so that the
application can offer an explicit, logged exclusion.

Grubbs' test
------------
Grubbs FE (1950) Sample criteria for testing outlying observations. *Ann Math Stat*
21:27-58; Grubbs FE (1969) Procedures for detecting outlying observations in samples.
*Technometrics* 11:1-21. See also the NIST/SEMATECH e-Handbook of Statistical Methods,
section 1.3.5.17. Two-sided test for a single outlier in a normal sample:

    G = max |x_i - mean| / sd
    G_crit = ((n - 1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2)),  t = t_{alpha / (2n), n - 2}

The P value is obtained by inverting the same relation (a Bonferroni-type bound,
n * two-sided t tail area, capped at 1); it is accurate for small P values and
conservative otherwise.

ROUT
----
Motulsky HJ, Brown RE (2006) Detecting outliers when fitting data with nonlinear
regression - a new method based on robust nonlinear regression and the false discovery
rate. *BMC Bioinformatics* 7:123.

1. Robust fit assuming the scatter follows a Lorentzian (Cauchy-like) distribution: the
   parameters minimise sum(ln(1 + (D_i / RSDR)^2)), where D_i are the residuals and RSDR,
   the robust standard deviation of the residuals, is re-estimated from the current
   residuals until the fit stops changing (an adaptive fit).
   RSDR = P68.27 * N / (N - K), where P68.27 is the 68.27th percentile of the absolute
   residuals (linear interpolation) and K the number of fitted parameters.
2. Outlier identification adapted from the Benjamini-Hochberg false discovery rate
   procedure: sort the absolute residuals from low to high; for i from int(0.70 N) to N
   compute alpha_i = Q (N - (i - 1)) / N, t = |residual_i| / RSDR and its two-tailed P
   value from the t distribution with N - K degrees of freedom. The first i with
   P < alpha_i, and every point with a larger absolute residual, is flagged.
3. Only on request (``refit=True``) an ordinary least-squares fit to the points that
   were not flagged is returned *alongside* the robust fit. Nothing is removed silently.

Q is the maximum desired false discovery rate among the flagged points (default 1%).

Implementation note: this module was written from the published description of the
method (the paper itself could not be consulted while writing it); the merit function
scale and the percentile interpolation rule follow the description above. Simulations in
the test-suite confirm the behaviour the paper reports (with Q = 1%, clean Gaussian data
give one or more false outliers in roughly 1-3% of data sets).
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from scipy import optimize, stats

from ..data.tables import DataTable
from ._utils import StatsError, check_alpha, fmt, is_constant, p_phrase
from .descriptives import _describe_clean, _name_list
from .registry import ParamSpec, register
from .result import AssumptionCheck, Parameter, StatResult

_CITE_GRUBBS = [
    "Grubbs FE (1950) Sample criteria for testing outlying observations. Ann Math Stat 21:27-58.",
    "Grubbs FE (1969) Procedures for detecting outlying observations in samples. Technometrics 11:1-21.",
]
_CITE_ROUT = [
    "Motulsky HJ, Brown RE (2006) Detecting outliers when fitting data with nonlinear regression - a "
    "new method based on robust nonlinear regression and the false discovery rate. BMC Bioinformatics 7:123.",
]

_REMINDER = (
    "Flagged values were not removed or changed. Excluding a value is your explicit decision; if you "
    "exclude any, say so (and why, and by which method) when you report the results."
)


def _finite_with_index(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """(finite values, their positions in the input, number of non-finite entries)."""
    if values is None:
        raise StatsError("No data given.")
    raw = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float).ravel()
    ok = np.isfinite(raw)
    return raw[ok], np.nonzero(ok)[0], int((~ok).sum())


# ========================================================================== Grubbs


def grubbs_critical_value(n: int, alpha: float = 0.05, *, two_sided: bool = True) -> float:
    """Critical value of Grubbs' statistic G for a sample of ``n`` values.

    G_crit = ((n - 1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2)) with t the upper
    alpha / (2n) point (two-sided; alpha / n for a one-sided test of only the maximum or
    only the minimum) of the t distribution with n - 2 degrees of freedom.
    """
    n = int(n)
    if n < 3:
        raise StatsError("Grubbs' test needs at least 3 values.")
    alpha = check_alpha(alpha)
    t = float(stats.t.isf(alpha / (2.0 * n) if two_sided else alpha / n, n - 2))
    return (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))


def grubbs_p_value(g: float, n: int, *, two_sided: bool = True) -> float:
    """Approximate P value of Grubbs' G: the critical-value formula solved for alpha.

    t^2 = (n - 2) q / (1 - q) with q = n G^2 / (n - 1)^2, then
    P = min(1, 2 n * P(T > t)) (n * P(T > t) for a one-sided test).
    """
    n = int(n)
    if n < 3:
        raise StatsError("Grubbs' test needs at least 3 values.")
    q = n * g * g / (n - 1) ** 2
    if q >= 1.0:
        return 0.0
    t = math.sqrt((n - 2) * q / (1.0 - q))
    tail = float(stats.t.sf(t, n - 2))
    return float(min(1.0, (2.0 if two_sided else 1.0) * n * tail))


def _grubbs_step(arr: np.ndarray, alpha: float) -> dict[str, Any]:
    n = int(arr.size)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    dev = np.abs(arr - mean)
    pos = int(np.argmax(dev))
    g = float(dev[pos] / sd)
    crit = grubbs_critical_value(n, alpha)
    return {"n": n, "mean": mean, "sd": sd, "position": pos, "value": float(arr[pos]), "g": g,
            "g_critical": crit, "p_value": grubbs_p_value(g, n), "is_outlier": bool(g > crit)}


def grubbs_test(x: Iterable[Any], *, alpha: float = 0.05, iterative: bool = False) -> dict[str, Any]:
    """Grubbs' two-sided test for a single outlier (see the module docstring).

    Returns a dict: ``n``, ``mean``, ``sd``, ``g`` (statistic of the most extreme value),
    ``g_critical``, ``p_value``, ``index`` (position of that value in the input, counting
    missing values), ``value``, ``is_outlier``, ``flagged`` (list of
    ``{"index", "value", "g", "g_critical", "p_value", "step"}``), ``steps`` and
    ``warnings``. Needs at least 3 values.

    ``iterative=True`` repeats the test after leaving the flagged value out *of the
    calculation only* (the data are untouched), flagging further values until the most
    extreme remaining value is no longer significant. Caveats: each round is tested at the
    same alpha (no adjustment for repeated testing), and two or more outliers can hide each
    other ("masking") because they inflate the SD, in which case even the first round
    finds nothing.
    """
    alpha = check_alpha(alpha)
    arr, index, _ = _finite_with_index(x)
    if arr.size < 3:
        raise StatsError(f"Grubbs' test needs at least 3 values; there are only {arr.size}.")
    n = int(arr.size)
    notes: list[str] = []
    out: dict[str, Any] = {
        "n": n, "mean": float(arr.mean()), "sd": None, "g": None, "g_critical": grubbs_critical_value(n, alpha),
        "p_value": None, "alpha": alpha, "index": None, "value": None, "is_outlier": False,
        "flagged": [], "steps": [], "iterative": bool(iterative), "warnings": notes,
    }
    if is_constant(arr):
        out["sd"] = 0.0
        notes.append("All values are identical, so there is no spread and no value can be an outlier.")
        return out
    remaining = arr.copy()
    remaining_index = index.copy()
    step_no = 0
    ties_flag: list[float] = []  # values flagged only because all the others are identical
    while remaining.size >= 3 and not is_constant(remaining):
        step_no += 1
        step = _grubbs_step(remaining, alpha)
        pos = step.pop("position")
        step.update(step=step_no, index=int(remaining_index[pos]))
        out["steps"].append(step)
        if step_no == 1:
            out.update(sd=step["sd"], g=step["g"], p_value=step["p_value"], index=step["index"],
                       value=step["value"], is_outlier=step["is_outlier"])
        if not step["is_outlier"]:
            break
        out["flagged"].append({k: step[k] for k in ("index", "value", "g", "g_critical", "p_value", "step")})
        if is_constant(np.delete(remaining, pos)):
            ties_flag.append(step["value"])
        if not iterative:
            break
        remaining = np.delete(remaining, pos)
        remaining_index = np.delete(remaining_index, pos)
    if ties_flag:
        notes.append(
            f"The flagged value {fmt(ties_flag[0])} is the only one that differs from the others, which "
            "are all identical. In that situation Grubbs' G takes its largest possible value, "
            "(n - 1) / sqrt(n), however small the difference is, so such a value is flagged whatever "
            "its size. The flag reflects identical (tied or rounded) values, i.e. limited measurement "
            "resolution, not evidence of an extreme value: check whether the data were recorded with "
            "enough decimals before treating it as an outlier."
        )
    elif n < 7:
        notes.append(
            f"With only {n} values Grubbs' test has little power: only a value very far from the others "
            "can be flagged"
            + (", so finding no outlier is weak evidence that there is none." if not out["flagged"] else ".")
        )
    if iterative and len(out["flagged"]) > 1:
        notes.append(
            "Several values were flagged by repeating Grubbs' test. Each round used the same alpha, so "
            "the overall false-positive risk is higher than alpha."
        )
    return out


def _normality_of_rest(arr: np.ndarray, target: str, n_flagged: int) -> AssumptionCheck:
    name = "Normal distribution apart from outliers (Shapiro-Wilk)"
    rest = "the values that were not flagged" if n_flagged else "the values"
    if arr.size < 3 or is_constant(arr):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail=f"Could not be checked for {rest} (too few values or no spread).")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        res = stats.shapiro(arr)
    w, p = float(res.statistic), float(res.pvalue)
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=w, p_value=p, status="fail",
            detail=f"Even {rest} deviate from a normal distribution ({p_phrase(p)}). Outlier tests that "
                   "assume normality can then flag values that simply belong to a skewed distribution "
                   "(for example lognormal data): consider analysing logarithms.",
        )
    return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="pass",
                           detail=f"No evidence against normality of {rest} ({p_phrase(p)}).")


def _row_lookup(row_index: dict[str, Sequence[int]] | None, group: str, size: int) -> np.ndarray:
    if row_index is None or group not in row_index:
        return np.arange(size)
    rows = np.asarray(row_index[group]).ravel()
    if rows.size != size:
        raise ValueError(f"row_index['{group}'] must have one entry per value of the group.")
    return rows


def _outlier_result(
    *, analysis: str, test_name: str, why: str, groups: dict[str, Iterable[Any]],
    row_index: dict[str, Sequence[int]] | None, minimum: int,
    run: Callable[[np.ndarray], dict[str, Any]], describe_group: Callable[[dict[str, Any]], dict[str, Any]],
    method_label: str, methods_text: str, citations: list[str], extra_assumptions: list[AssumptionCheck],
    alpha: float, headline: Callable[[dict[str, Any]], tuple[str, float | None, float | None]] | None,
    extra_more: dict[str, Any],
) -> StatResult:
    """Shared assembly of a multi-group outlier report."""
    if not groups:
        raise StatsError("No groups were given. Enter at least one column of numbers.")
    flagged: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    assumptions: list[AssumptionCheck] = []
    n: dict[str, int] = {}
    warnings: list[str] = []
    notes: list[str] = []
    per_group: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for name, values in groups.items():
        label = str(name)
        raw = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float).ravel()
        lookup = _row_lookup(row_index, label, raw.size)
        arr, index, n_bad = _finite_with_index(raw)
        if n_bad:
            notes.append(f"{label}: {n_bad} missing or non-finite value(s) ignored.")
        n[label] = int(arr.size)
        if arr.size:
            rows.append({"group": label, **_describe_clean(arr, 0.95)})
        if arr.size < minimum:
            skipped.append(label)
            table.append({"group": label, "n": int(arr.size), "n_flagged": None, "tested": False})
            continue
        res = run(raw)
        per_group[label] = res
        for w in res["warnings"]:
            warnings.append(f"{label}: {w}")
        flagged_positions = [int(f["index"]) for f in res["flagged"]]
        for f in res["flagged"]:
            flagged.append({"group": label, "row": int(lookup[int(f["index"])]), "value": float(f["value"])})
        keep = np.ones(raw.size, dtype=bool)
        keep[flagged_positions] = False
        assumptions.append(_normality_of_rest(raw[keep & np.isfinite(raw)], label, len(flagged_positions)))
        table.append({"group": label, "n": int(arr.size), "n_flagged": len(flagged_positions), "tested": True,
                      "flagged_rows": [int(lookup[i]) for i in flagged_positions],
                      "flagged_values": [float(raw[i]) for i in flagged_positions],
                      **describe_group(res)})
    if not per_group:
        raise StatsError(
            f"{method_label} needs at least {minimum} values in a group; no group has that many."
        )
    if skipped:
        warnings.append(
            f"Not tested (fewer than {minimum} values): {_name_list(skipped)}."
        )
    assumptions.extend(extra_assumptions)

    tested = list(per_group)
    if flagged:
        by_group: dict[str, list[str]] = {}
        for f in flagged:
            by_group.setdefault(f["group"], []).append(f"{fmt(f['value'])} (row {f['row'] + 1})")
        listing = "; ".join(f"{g}: {', '.join(v)}" for g, v in by_group.items())
        clean_groups = [g for g in tested if g not in by_group]
        interpretation = (
            f"{method_label} flagged {len(flagged)} possible outlier{'s' if len(flagged) != 1 else ''} "
            f"- {listing}."
        )
        if clean_groups:
            interpretation += f" No outliers were flagged in {_name_list(clean_groups)}."
        interpretation += (
            " A flagged value is unusually far from the rest given the method's assumptions; that is "
            "evidence it may come from a different process (a mistake, or real biology), not proof."
        )
    else:
        interpretation = (
            f"{method_label} did not flag any outlier in {_name_list(tested)}. This does not show "
            "that every value is valid, only that none is extreme enough to be flagged."
        )
    interpretation += " " + _REMINDER
    result = StatResult(
        analysis=analysis,
        test_name=test_name,
        why_chosen=why,
        assumptions=assumptions,
        alpha=alpha,
        n=n,
        descriptives=rows,
        warnings=warnings,
        data_notes=notes,
        interpretation=interpretation,
        methods_text=methods_text,
        citations=citations,
        extra={"flagged": flagged, "table": table, "n_flagged": len(flagged), "removed": False, **extra_more},
    )
    if headline is not None and len(per_group) == 1 and len(groups) == 1:
        stat_name, stat, p = headline(next(iter(per_group.values())))
        result.statistic_name, result.statistic, result.p_value = stat_name, stat, p
    return result


def grubbs_outliers(groups: dict[str, Iterable[Any]], *, alpha: float = 0.05, iterative: bool = False,
                    row_index: dict[str, Sequence[int]] | None = None) -> StatResult:
    """Grubbs' test on every group; flags (never removes) the outliers found.

    ``groups`` maps a name to its values *in table order, missing values included* so that
    the reported ``row`` is the position in the column; ``row_index`` can supply explicit
    row numbers (one per value) when the arrays are not complete table columns. With a
    single group the headline ``statistic`` (G) and ``p_value`` are filled.
    """
    alpha = check_alpha(alpha)

    def describe_group(res: dict[str, Any]) -> dict[str, Any]:
        return {"mean": res["mean"], "sd": res["sd"], "most_extreme_value": res["value"], "g": res["g"],
                "g_critical": res["g_critical"], "p_value": res["p_value"], "outlier": res["is_outlier"]}

    extra_assumptions = [
        AssumptionCheck(
            name="At most one outlier per group" if not iterative else "Outliers do not mask each other",
            status="info",
            detail=("Grubbs' test looks for a single outlier. Two or more outliers can hide each other "
                    "(masking) because they inflate the SD."
                    if not iterative else
                    "The test was repeated after setting aside each flagged value. Two or more outliers "
                    "can still hide each other (masking), and repeating the test raises the "
                    "false-positive risk above alpha."),
        ),
    ]
    return _outlier_result(
        analysis="outliers_grubbs",
        test_name="Grubbs' outlier test" + (" (repeated)" if iterative else ""),
        why=("Grubbs' test asks whether the single most extreme value in a group is too far from the "
             "mean to be plausible if all values came from one normal distribution."),
        groups=groups, row_index=row_index, minimum=3,
        run=lambda raw: grubbs_test(raw, alpha=alpha, iterative=iterative),
        describe_group=describe_group,
        method_label="Grubbs' test" + (" (repeated)" if iterative else ""),
        methods_text=(
            f"Outliers were identified with Grubbs' test (two-sided, alpha = {alpha:g}"
            + (", applied repeatedly" if iterative else "") + "); flagged values were not removed automatically."
        ),
        citations=list(_CITE_GRUBBS),
        extra_assumptions=extra_assumptions,
        alpha=alpha,
        headline=lambda res: ("G", res["g"], res["p_value"]),
        extra_more={"method": "grubbs", "alpha": alpha, "iterative": bool(iterative)},
    )


# ============================================================================ ROUT


def _check_q(q: float) -> float:
    try:
        value = float(q)
    except (TypeError, ValueError):
        raise StatsError("Q must be a number between 0 and 1 (for example 0.01 for 1%).") from None
    if not (0.0 < value < 1.0):
        raise StatsError("Q must be between 0 and 1 (for example 0.01 for 1%). If you meant a "
                         "percentage, divide it by 100.")
    return value


def robust_sd_of_residuals(residuals: Iterable[float], k: int) -> float:
    """RSDR = (68.27th percentile of the absolute residuals) * N / (N - K)."""
    r = np.abs(np.asarray(residuals, dtype=float))
    n = r.size
    if n - k < 1:
        raise StatsError("There must be more data points than fitted parameters.")
    return float(np.percentile(r, 68.27) * n / (n - k))


def _rout_detect(residuals: np.ndarray, rsdr: float, k: int, q: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Step 2 of ROUT. Returns (indices of flagged points, the tests that were made)."""
    n = residuals.size
    abs_res = np.abs(residuals)
    order = np.argsort(abs_res, kind="stable")
    tests: list[dict[str, Any]] = []
    flagged = order[:0]
    for i in range(max(int(0.70 * n), 1), n + 1):
        alpha_i = q * (n - (i - 1)) / n
        t = float(abs_res[order[i - 1]] / rsdr)
        p = float(2.0 * stats.t.sf(t, n - k))
        hit = bool(p < alpha_i)
        tests.append({"index": int(order[i - 1]), "rank": i, "t": t, "p_value": p, "threshold": alpha_i,
                      "outlier": hit})
        if hit:
            flagged = order[i - 1:]
            for j in range(i + 1, n + 1):
                t_j = float(abs_res[order[j - 1]] / rsdr)
                tests.append({"index": int(order[j - 1]), "rank": j, "t": t_j,
                              "p_value": float(2.0 * stats.t.sf(t_j, n - k)),
                              "threshold": q * (n - (j - 1)) / n, "outlier": True})
            break
    return np.sort(flagged), tests


def rout(
    x: Iterable[Any],
    y: Iterable[Any],
    model_func: Callable[..., Any],
    p0: Sequence[float],
    *,
    q: float = 0.01,
    refit: bool = False,
    max_iter: int = 100,
    _start: str = "ols",
) -> dict[str, Any]:
    """ROUT outlier detection for any curve ``y = model_func(x, *params)``.

    Parameters
    ----------
    x, y : data (points with a missing x or y are ignored; indices refer to the input).
    model_func : callable ``f(x_array, *params) -> y_array``.
    p0 : starting values of the parameters (their number defines K).
    q : maximum desired false discovery rate (default 0.01 = 1%).
    refit : when True, also return an ordinary least-squares fit to the points that were
        not flagged, under ``"ols_refit"``. The data are never modified.

    Returns a dict with ``n``, ``k``, ``q``, ``params_robust``, ``params_ols`` (ordinary
    least squares on *all* points, for comparison), ``rsdr``, ``residuals`` (from the
    robust fit, input order, NaN where the point was missing), ``outlier_indices``,
    ``outlier_mask``, ``flagged`` (``{"index", "value", "x", "residual", "t", "p_value"}``),
    ``tests``, ``converged``, ``iterations``, ``warnings`` and ``ols_refit``.
    """
    q = _check_q(q)
    if x is None or y is None:
        raise StatsError("No data given.")
    xr = np.asarray(x if isinstance(x, np.ndarray) else list(x), dtype=float).ravel()
    yr = np.asarray(y if isinstance(y, np.ndarray) else list(y), dtype=float).ravel()
    if xr.shape != yr.shape:
        raise StatsError(f"X and Y must have the same number of values (X has {xr.size}, Y has {yr.size}).")
    start = np.atleast_1d(np.asarray(p0, dtype=float))
    k = int(start.size)
    if k < 1 or not np.all(np.isfinite(start)):
        raise StatsError("Starting values for every parameter of the model are needed (finite numbers).")
    ok = np.isfinite(xr) & np.isfinite(yr)
    index = np.nonzero(ok)[0]
    xs, ys = xr[ok], yr[ok]
    n = int(xs.size)
    if n < max(k + 2, 3):
        raise StatsError(
            f"ROUT needs at least {max(k + 2, 3)} points for a model with {k} parameter(s); there are {n}."
        )
    notes: list[str] = []

    def residual(params: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
        xk, yk = (xs, ys) if keep is None else (xs[keep], ys[keep])
        with np.errstate(all="ignore"):
            return yk - np.asarray(model_func(xk, *params), dtype=float)

    def fit(initial: np.ndarray, keep: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                return optimize.least_squares(residual, initial, args=(keep,), xtol=1e-12, ftol=1e-12,
                                              gtol=1e-12, **kwargs).x
        except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise StatsError(
                "The model could not be fitted to these data (it returned undefined values or did not "
                f"converge). Try different starting values. Details: {exc}"
            ) from None

    params_ols = fit(start)
    params = params_ols if _start == "ols" else start
    converged = False
    iterations = 0
    rsdr = robust_sd_of_residuals(residual(params), k)
    scale = float(np.max(np.abs(ys))) or 1.0
    tiny = 1e-9 * scale  # scatter below the numerical noise of an iterative fit counts as none
    while iterations < int(max_iter):
        if not (rsdr > tiny):
            break
        iterations += 1
        new = fit(params, loss="cauchy", f_scale=rsdr)
        new_rsdr = robust_sd_of_residuals(residual(new), k)
        change = float(np.max(np.abs(new - params) / (np.abs(params) + 1e-12 * scale + 1e-300)))
        params = new
        if change < 1e-8 or abs(new_rsdr - rsdr) <= 1e-10 * rsdr:
            rsdr = new_rsdr
            converged = True
            break
        rsdr = new_rsdr
    res = residual(params)
    if not np.all(np.isfinite(res)):
        raise StatsError("The model returned undefined values for the fitted parameters.")
    residuals_full = np.full(xr.shape, np.nan)
    residuals_full[index] = res
    out: dict[str, Any] = {
        "n": n, "k": k, "q": q,
        "params_robust": [float(v) for v in params],
        "params_ols": [float(v) for v in params_ols],
        "rsdr": rsdr,
        "residuals": residuals_full,
        "outlier_indices": [],
        "outlier_mask": np.zeros(xr.shape, dtype=bool),
        "flagged": [],
        "tests": [],
        "converged": converged,
        "iterations": iterations,
        "warnings": notes,
        "ols_refit": None,
    }
    if not (rsdr > tiny):
        out["rsdr"] = 0.0
        out["converged"] = True
        notes.append(
            "More than two thirds of the points lie exactly on the fitted curve, so the robust SD of "
            "the residuals is zero and outliers cannot be assessed by ROUT."
        )
        return out
    if not converged:
        notes.append("The robust fit did not fully converge; treat the flagged points with caution.")
    local, tests = _rout_detect(res, rsdr, k, q)
    for t in tests:
        t["index"] = int(index[t["index"]])
    by_index = {t["index"]: t for t in tests}
    out["tests"] = tests
    out["outlier_indices"] = [int(index[i]) for i in local]
    out["outlier_mask"][index[local]] = True
    out["flagged"] = [
        {"index": int(index[i]), "value": float(ys[i]), "x": float(xs[i]), "residual": float(res[i]),
         "t": by_index[int(index[i])]["t"], "p_value": by_index[int(index[i])]["p_value"]}
        for i in local
    ]
    if n - k < 5:
        notes.append("There are very few points for this model, so outlier detection has little power.")
    if refit:
        keep = np.ones(n, dtype=bool)
        keep[local] = False
        if int(keep.sum()) > k:
            refit_params = fit(params, keep)
            out["ols_refit"] = {
                "params": [float(v) for v in refit_params], "n_used": int(keep.sum()),
                "n_excluded_from_fit": int(local.size),
                "note": "Least-squares fit to the points that were not flagged; the data were not changed.",
            }
        else:
            notes.append("Too few points remain for a least-squares fit without the flagged points.")
    return out


def _robust_location(arr: np.ndarray, max_iter: int = 500) -> tuple[float, float, bool, int]:
    """Adaptive Lorentzian fit of a constant: (location, RSDR, converged, iterations).

    For a fixed scale the minimiser of sum(ln(1 + ((x - m) / s)^2)) satisfies
    m = sum(w x) / sum(w) with w = 1 / (1 + ((x - m) / s)^2); iterating that (with the
    RSDR re-estimated each round) converges from the median.
    """
    n = arr.size
    m = float(np.median(arr))
    scale = max(float(np.max(np.abs(arr))), 1e-300)
    rsdr = 0.0
    for it in range(1, max_iter + 1):
        d = arr - m
        rsdr = float(np.percentile(np.abs(d), 68.27) * n / (n - 1))
        if not (rsdr > 1e-12 * scale):
            return m, 0.0, True, it
        w = 1.0 / (1.0 + (d / rsdr) ** 2)
        new = float(np.sum(w * arr) / np.sum(w))
        if abs(new - m) <= 1e-13 * scale:
            m = new
            d = arr - m
            return m, float(np.percentile(np.abs(d), 68.27) * n / (n - 1)), True, it
        m = new
    return m, rsdr, False, max_iter


def rout_column(values: Iterable[Any], *, q: float = 0.01) -> dict[str, Any]:
    """ROUT for a plain column of values: the "model" is a single constant (K = 1).

    Returns a dict with ``n``, ``k`` (1), ``q``, ``location`` (the robust centre), ``mean``,
    ``sd``, ``rsdr``, ``residuals`` (input order, NaN for missing values),
    ``outlier_indices`` (positions in the input), ``outlier_mask``, ``flagged``
    (``{"index", "value", "residual", "t", "p_value"}``), ``tests``, ``converged`` and
    ``warnings``. Needs at least 3 values.
    """
    q = _check_q(q)
    arr, index, n_bad = _finite_with_index(values)
    n = int(arr.size)
    if n < 3:
        raise StatsError(f"ROUT needs at least 3 values; there are only {n}.")
    raw_size = n + n_bad
    notes: list[str] = []
    location, rsdr, converged, iterations = _robust_location(arr)
    res = arr - location
    residuals_full = np.full(raw_size, np.nan)
    residuals_full[index] = res
    out: dict[str, Any] = {
        "n": n, "k": 1, "q": q, "location": location, "mean": float(arr.mean()),
        "sd": 0.0 if is_constant(arr) else float(arr.std(ddof=1)), "rsdr": rsdr,
        "residuals": residuals_full, "outlier_indices": [],
        "outlier_mask": np.zeros(raw_size, dtype=bool), "flagged": [], "tests": [],
        "converged": converged, "iterations": iterations, "warnings": notes,
    }
    if rsdr <= 0.0:
        notes.append(
            "All values are identical." if is_constant(arr) else
            "More than two thirds of the values are identical, so the robust SD is zero and outliers "
            "cannot be assessed by ROUT."
        )
        return out
    if not converged:
        notes.append("The robust fit did not fully converge; treat the flagged values with caution.")
    if n < 6:
        notes.append(f"With only {n} values outlier detection has very little power.")
    local, tests = _rout_detect(res, rsdr, 1, q)
    for t in tests:
        t["index"] = int(index[t["index"]])
    by_index = {t["index"]: t for t in tests}
    out["tests"] = tests
    out["outlier_indices"] = [int(index[i]) for i in local]
    out["outlier_mask"][index[local]] = True
    out["flagged"] = [
        {"index": int(index[i]), "value": float(arr[i]), "residual": float(res[i]),
         "t": by_index[int(index[i])]["t"], "p_value": by_index[int(index[i])]["p_value"]}
        for i in local
    ]
    return out


def rout_outliers(groups: dict[str, Iterable[Any]], *, q: float = 0.01,
                  row_index: dict[str, Sequence[int]] | None = None) -> StatResult:
    """ROUT (Q = ``q``) on every group; flags (never removes) the outliers found.

    ``groups`` maps a name to its values *in table order, missing values included*, so
    that the reported ``row`` is the position in the column (or pass ``row_index``).
    Unlike Grubbs' test, ROUT can flag several outliers at once and is not fooled by
    outliers masking each other, because the centre and spread are estimated robustly.
    """
    q = _check_q(q)

    def describe_group(res: dict[str, Any]) -> dict[str, Any]:
        return {"robust_location": res["location"], "rsdr": res["rsdr"], "mean": res["mean"], "sd": res["sd"]}

    result = _outlier_result(
        analysis="outliers_rout",
        test_name=f"ROUT outlier detection (Q = {q * 100:g}%)",
        why=("ROUT finds outliers by first fitting the data robustly (so that outliers barely influence "
             "the centre and spread) and then asking which values are too far from that fit, while "
             "keeping the false discovery rate at or below Q. It can flag several outliers at once."),
        groups=groups, row_index=row_index, minimum=3,
        run=lambda raw: rout_column(raw, q=q),
        describe_group=describe_group,
        method_label=f"ROUT (Q = {q * 100:g}%)",
        methods_text=(
            f"Outliers were identified with the ROUT method (Q = {q * 100:g}%); flagged values were not "
            "removed automatically."
        ),
        citations=list(_CITE_ROUT),
        extra_assumptions=[
            AssumptionCheck(
                name="Meaning of Q", status="info",
                detail=f"Q = {q * 100:g}% is the maximum desired false discovery rate: on average no more "
                       f"than about {q * 100:g}% of flagged values are expected to be false alarms. When "
                       "there are truly no outliers, about 1-3% of data sets still get at least one "
                       "value flagged at Q = 1%.",
            ),
        ],
        alpha=0.05,
        headline=None,
        extra_more={"method": "rout", "q": q},
    )
    for row in result.extra["table"]:
        if row.get("tested"):
            result.parameters.append(Parameter(
                name=f"Robust centre of {row['group']}", value=row["robust_location"],
                unit_note=f"Lorentzian robust fit; robust SD (RSDR) = {fmt(row['rsdr'])}.",
            ))
    return result


# ------------------------------------------------------------------ table adapters


def _table_columns(table: DataTable, groups: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Selected columns with missing values kept, plus the table row of every entry."""
    frame = table.analysis_frame()
    available = table.group_names
    if groups is None or groups == "" or (isinstance(groups, list | tuple) and len(groups) == 0):
        names = list(available)
    else:
        names = [groups] if isinstance(groups, str) else [str(g) for g in groups]
        unknown = [g for g in names if g not in available]
        if unknown:
            raise StatsError(f"Unknown group(s): {unknown}. Available groups: {available}.")
        names = list(dict.fromkeys(names))
    rows = frame.index.to_numpy()
    return ({g: frame[g].to_numpy(dtype=float) for g in names}, {g: rows for g in names})


_GROUPS_PARAM = ParamSpec(
    "groups", "Groups to check", kind="groups", default=None, optional=True,
    help="Which columns to look for outliers in. Leave empty to check every column (each column is "
         "checked on its own).",
)


@register(
    key="outliers_grubbs",
    label="Identify outliers (Grubbs)",
    category="Outliers",
    tables=("column",),
    params=(
        _GROUPS_PARAM,
        ParamSpec(
            "alpha", "Significance level (alpha)", kind="float", default=0.05,
            help="How strict the test is. With 0.05, a group with no real outlier still has a 5% chance "
                 "of having a value flagged.",
        ),
        ParamSpec(
            "iterative", "Look for more than one outlier", kind="bool", default=False,
            help="Grubbs' test finds one outlier at a time. Tick this to repeat it after setting aside "
                 "each flagged value. Values are only flagged, never deleted.",
        ),
    ),
    description=(
        "Flags the single most extreme value in each group if it is too far from the mean for normally "
        "distributed data. Values are only flagged; you decide whether to exclude them."
    ),
    citation="Grubbs (1969) Technometrics 11:1-21",
)
def _run_outliers_grubbs(table: DataTable, groups: Any = None, alpha: float = 0.05, iterative: bool = False,
                         **_: Any) -> StatResult:
    data, rows = _table_columns(table, groups)
    return grubbs_outliers(data, alpha=alpha, iterative=bool(iterative), row_index=rows)


@register(
    key="outliers_rout",
    label="Identify outliers (ROUT)",
    category="Outliers",
    tables=("column",),
    params=(
        _GROUPS_PARAM,
        ParamSpec(
            "q", "Q (maximum false discovery rate)", kind="float", default=0.01,
            help="Q is the maximum desired false discovery rate: the largest fraction of flagged values "
                 "you accept being false alarms. 0.01 (1%) is the usual choice; a larger Q (e.g. 0.05) "
                 "flags outliers more readily, a smaller Q (e.g. 0.001) only the most extreme values.",
        ),
    ),
    description=(
        "Flags outliers using a robust fit, so that several outliers can be found at once without "
        "hiding each other. Values are only flagged; you decide whether to exclude them."
    ),
    citation="Motulsky & Brown (2006) BMC Bioinformatics 7:123",
)
def _run_outliers_rout(table: DataTable, groups: Any = None, q: float = 0.01, **_: Any) -> StatResult:
    data, rows = _table_columns(table, groups)
    return rout_outliers(data, q=q, row_index=rows)
