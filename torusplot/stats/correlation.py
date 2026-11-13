"""Correlation: Pearson, Spearman and correlation matrices.

References
----------
* Pearson K (1896) Philos Trans R Soc A 187:253-318 (product-moment correlation).
* Fisher RA (1915) Biometrika 10:507-521; Fisher RA (1921) Metron 1:3-32 (z transformation
  used for the confidence interval of r).
* Spearman C (1904) Am J Psychol 15:72-101 (rank correlation).
* Bonett DG, Wright TA (2000) Psychometrika 65:23-28 (standard error
  sqrt((1 + rho^2 / 2) / (n - 3)) for the Fisher-z interval of Spearman's rho).
* Kendall MG, Gibbons JD (1990) *Rank Correlation Methods*, 5th ed. (exact null
  distribution of Spearman's rho by enumeration of all permutations).
"""

from __future__ import annotations

import math
import warnings
from functools import cache
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..data.tables import TableError
from ._utils import (
    StatsError,
    alternative_label,
    check_alpha,
    check_alternative,
    ci_phrase,
    clean_paired,
    fmt,
    p_phrase,
    sig_phrase,
)
from .effect_sizes import magnitude_r
from .multiplicity import METHOD_LABELS, adjust_pvalues
from .registry import ParamSpec, register
from .result import AssumptionCheck, EffectSize, Parameter, StatResult

__all__ = ["correlation_matrix", "pearson", "spearman"]

#: Largest n for which the exact permutation distribution of Spearman's rho is enumerated.
EXACT_SPEARMAN_MAX_N = 9

_CITE_PEARSON = "Pearson K (1896) Philos Trans R Soc A 187:253-318"
_CITE_FISHER = "Fisher RA (1921) Metron 1:3-32"
_CITE_SPEARMAN = "Spearman C (1904) Am J Psychol 15:72-101"
_CITE_BW = "Bonett DG, Wright TA (2000) Psychometrika 65:23-28"


# ----------------------------------------------------------------------- helpers
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


def _check_level(ci_level: float) -> float:
    if not (0 < float(ci_level) < 1):
        raise StatsError("ci_level must be between 0 and 1 (typically 0.95).")
    return float(ci_level)


def _r_value(x: np.ndarray, y: np.ndarray) -> float:
    """Product-moment correlation from centred sums (clipped to [-1, 1])."""
    dx = x - x.mean()
    dy = y - y.mean()
    # scale before squaring so very large / very small data do not overflow
    sx = float(np.max(np.abs(dx)))
    sy = float(np.max(np.abs(dy)))
    if sx == 0 or sy == 0:
        return math.nan
    dx = dx / sx
    dy = dy / sy
    denom = math.sqrt(float(dx @ dx) * float(dy @ dy))
    if denom == 0:
        return math.nan
    return float(max(-1.0, min(1.0, float(dx @ dy) / denom)))


def _t_pvalue(r: float, n: int, alternative: str) -> tuple[float | None, float | None]:
    """t statistic and p value for H0: correlation = 0 (df = n - 2)."""
    df = n - 2
    if df < 1 or not math.isfinite(r):
        return None, None
    if abs(r) >= 1.0:
        t = math.copysign(math.inf, r)
    else:
        t = r * math.sqrt(df / ((1.0 - r) * (1.0 + r)))
    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(t), df))
    elif alternative == "greater":
        p = float(stats.t.sf(t, df))
    else:
        p = float(stats.t.cdf(t, df))
    return t, min(1.0, p)


def _fisher_interval(
    r: float, se_z: float, level: float, alternative: str = "two-sided"
) -> tuple[float | None, float | None]:
    """Fisher-z confidence interval, one-sided when the test is one-sided."""
    if not math.isfinite(r) or not math.isfinite(se_z):
        return None, None
    if abs(r) >= 1.0:
        return r, r
    z = math.atanh(r)
    if alternative == "two-sided":
        crit = float(stats.norm.ppf(0.5 + level / 2.0))
        return math.tanh(z - crit * se_z), math.tanh(z + crit * se_z)
    crit = float(stats.norm.ppf(level))
    if alternative == "greater":
        return math.tanh(z - crit * se_z), 1.0
    return -1.0, math.tanh(z + crit * se_z)


def _shapiro(values: np.ndarray, target: str) -> AssumptionCheck:
    name = "Normality (Shapiro-Wilk)"
    if values.size < 3:
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="Too few values (n < 3) to check normality.")
    if is_constant(values):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="All values are identical, so normality cannot be assessed.")
    if values.size > 5000:
        return AssumptionCheck(name=name, target=target, status="info",
                               detail="Not tested: with more than 5000 values the test flags trivial "
                                      "departures. Inspect a QQ plot instead.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, p = stats.shapiro(values)
    if not math.isfinite(p):
        return AssumptionCheck(name=name, target=target, status="warn",
                               detail="The normality test could not be computed for these values.")
    if p < 0.05:
        return AssumptionCheck(
            name=name, target=target, statistic=w, p_value=p, status="fail",
            detail=f"{target} does not look normally distributed ({p_phrase(p)}). "
                   "Consider Spearman correlation, which does not assume normality.")
    return AssumptionCheck(name=name, target=target, statistic=w, p_value=p, status="pass",
                           detail=f"No evidence against normality ({p_phrase(p)}).")


def _leave_one_out(x: np.ndarray, y: np.ndarray, r: float, alpha: float) -> AssumptionCheck:
    """Sensitivity of r to single points (drop each point in turn)."""
    name = "No single influential point (leave-one-out)"
    n = x.size
    if n < 4 or n > 5000:
        return AssumptionCheck(
            name=name, status="info",
            detail="Pearson's r can be dominated by one unusual point. Always look at the "
                   "scatter plot. (Not checked numerically for this sample size.)")
    values = []
    idx = np.arange(n)
    for i in range(n):
        keep = idx != i
        xi, yi = x[keep], y[keep]
        if is_constant(xi) or is_constant(yi):
            return AssumptionCheck(
                name=name, status="warn",
                detail="Without one particular point, one of the variables does not vary at all: "
                       "the whole correlation rests on that single point. Do not rely on it.")
        values.append(_r_value(xi, yi))
    arr = np.asarray(values)
    lo, hi = float(arr.min()), float(arr.max())
    change = float(np.max(np.abs(arr - r)))
    # shift of the estimate on the Fisher z scale, in units of its standard error
    limit = 1.0 - 1e-12
    z_all = np.arctanh(np.clip(arr, -limit, limit))
    z_shift = float(np.max(np.abs(z_all - math.atanh(max(-limit, min(limit, r)))))) * math.sqrt(n - 3)
    _, p_full = _t_pvalue(r, n, "two-sided")
    flips = False
    if p_full is not None:
        for ri in arr:
            _, pi = _t_pvalue(float(ri), n - 1, "two-sided")
            if pi is not None and (pi < alpha) != (p_full < alpha):
                flips = True
                break
    detail = (f"Dropping one point at a time gives r between {fmt(lo, 3)} and {fmt(hi, 3)} "
              f"(largest change {fmt(change, 3)}).")
    if z_shift > 2.0 or flips:
        extra = " The conclusion changes when one point is removed." if flips else ""
        return AssumptionCheck(
            name=name, statistic=change, status="warn",
            detail=detail + extra + " One point has a large influence (it moves the estimate by "
                   "more than two standard errors, or changes the conclusion): inspect the "
                   "scatter plot and consider Spearman correlation.")
    return AssumptionCheck(name=name, statistic=change, status="pass", detail=detail)


def _strength(r: float) -> str:
    label = magnitude_r(r)
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    if r == 0:
        return "no"
    return f"{label} {direction}" if label else direction


def _prepare(x: Any, y: Any, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        xa, ya, notes = clean_paired(x, y, names)
    except StatsError:
        raise
    except (TypeError, ValueError) as exc:
        raise StatsError(f"{names[0]} and {names[1]} must be lists of numbers.") from exc
    if xa.size < 2:
        raise StatsError(
            f"Correlation needs at least 2 complete pairs of values ({xa.size} found); "
            "3 or more are needed for a P value."
        )
    for arr, name in ((xa, names[0]), (ya, names[1])):
        if is_constant(arr):
            raise StatsError(
                f"All values of {name} are identical, so a correlation cannot be computed "
                "(a variable that does not vary cannot co-vary with anything)."
            )
    return xa, ya, notes


# ------------------------------------------------------------------------ Pearson
def pearson(
    x: Any,
    y: Any,
    *,
    names: tuple[str, str] = ("X", "Y"),
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ci_level: float = 0.95,
) -> StatResult:
    """Pearson product-moment correlation with a t test and a Fisher-z confidence interval.

    r = S_xy / sqrt(S_xx S_yy); t = r sqrt((n - 2) / (1 - r^2)) on n - 2 degrees of freedom;
    CI: tanh(atanh(r) +/- z / sqrt(n - 3)) (Fisher 1921). With a one-sided alternative the
    interval is one-sided too.
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    ci_level = _check_level(ci_level)
    xa, ya, notes = _prepare(x, y, names)
    n = int(xa.size)
    warns: list[str] = []
    r = _r_value(xa, ya)
    if not math.isfinite(r):
        raise StatsError("The correlation could not be computed (the values do not vary enough).")
    t, p = _t_pvalue(r, n, alternative)
    perfect = abs(r) >= 1.0 - 1e-15
    if n == 2:
        warns.append("With only 2 pairs the correlation is always +1 or -1 and no P value or "
                     "confidence interval exists. Collect more data.")
    elif perfect:
        warns.append("The points fall exactly on a straight line (r = +/-1). Check that the two "
                     "variables are not the same measurement or computed from each other.")
    ci_low = ci_high = None
    if n > 3:
        ci_low, ci_high = _fisher_interval(r, 1.0 / math.sqrt(n - 3), ci_level, alternative)
    elif n == 3:
        warns.append("A confidence interval for r needs at least 4 pairs.")
    if 3 <= n < 10:
        warns.append(f"Only {n} pairs: the estimate of r is very imprecise (see the wide CI).")

    assumptions = [
        _shapiro(xa, names[0]),
        _shapiro(ya, names[1]),
        AssumptionCheck(
            name="Linear relationship", status="info",
            detail="Pearson's r measures straight-line association only. A curved relationship "
                   "can give r near 0; check the scatter plot."),
        _leave_one_out(xa, ya, r, alpha),
        AssumptionCheck(
            name="Independent pairs", status="info",
            detail="Each pair must come from a different subject/sample; replicate measurements "
                   "of the same sample are not independent."),
    ]
    stat_t = t if (t is not None and math.isfinite(t)) else None
    r2 = r * r
    result = StatResult(
        analysis="correlation",
        test_name="Pearson correlation",
        why_chosen="Pearson correlation quantifies how closely two continuous measurements "
                   "follow a straight-line relationship, when both are roughly normally distributed.",
        assumptions=assumptions,
        statistic_name="t",
        statistic=stat_t,
        df=float(n - 2) if n > 2 else None,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[
            EffectSize(name="Pearson r", value=r, ci_low=ci_low, ci_high=ci_high, ci_level=ci_level,
                       ci_method="Fisher z transformation" if ci_low is not None else None,
                       magnitude=magnitude_r(r)),
            EffectSize(name="r squared", value=r2,
                       ci_low=_square_interval(ci_low, ci_high)[0] if alternative == "two-sided" else None,
                       ci_high=_square_interval(ci_low, ci_high)[1] if alternative == "two-sided" else None,
                       ci_level=ci_level,
                       ci_method="squared limits of the Fisher z interval" if ci_low is not None
                       and alternative == "two-sided" else None),
        ],
        parameters=[Parameter(name="r", value=r, ci_low=ci_low, ci_high=ci_high, ci_level=ci_level,
                              statistic=stat_t, p_value=p)],
        n={f"{names[0]} / {names[1]} pairs": n},
        descriptives=_describe(xa, ya, names),
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_PEARSON, _CITE_FISHER],
        extra={"r": r, "r_squared": r2, "method": "pearson", "x_name": names[0], "y_name": names[1],
               "scatter": {"x": xa.tolist(), "y": ya.tolist()}},
    )
    ci_text = f", {ci_phrase(ci_low, ci_high, ci_level)}" if ci_low is not None else ""
    if p is None:
        result.interpretation = (
            f"The Pearson correlation between {names[0]} and {names[1]} is r = {fmt(r, 3)} "
            f"(n = {n} pairs), but with so few pairs it cannot be tested."
        )
    else:
        result.interpretation = (
            f"{names[0]} and {names[1]} show a {_strength(r)} linear correlation "
            f"(Pearson r = {fmt(r, 3)}{ci_text}; n = {n} pairs), which is {sig_phrase(p, alpha)}. "
            f"r squared = {fmt(r2, 3)}: about {fmt(100 * r2, 3)}% of the variance of one variable "
            "is shared with the other. Correlation alone is not evidence that one variable causes "
            "the other."
        )
    result.methods_text = (
        f"The association between {names[0]} and {names[1]} was assessed with the Pearson "
        f"correlation coefficient ({alternative_label(alternative)} t test, alpha = {alpha:g}); the "
        f"{ci_level * 100:g}% confidence interval was obtained with the Fisher z transformation."
    )
    return result


def _square_interval(lo: float | None, hi: float | None) -> tuple[float | None, float | None]:
    """Interval for r^2 implied by an interval (lo, hi) for r."""
    if lo is None or hi is None:
        return None, None
    if lo <= 0.0 <= hi:
        return 0.0, max(lo * lo, hi * hi)
    return min(lo * lo, hi * hi), max(lo * lo, hi * hi)


def _describe(xa: np.ndarray, ya: np.ndarray, names: tuple[str, str]) -> list[dict[str, Any]]:
    rows = []
    for arr, name in ((xa, names[0]), (ya, names[1])):
        n = int(arr.size)
        scale = float(np.max(np.abs(arr)))  # scale first so huge values do not overflow
        sd = (scale * float((arr / scale).std(ddof=1)) if scale > 0 else 0.0) if n > 1 else None
        rows.append({
            "group": name, "n": n, "mean": float(arr.mean()), "sd": sd,
            "sem": sd / math.sqrt(n) if sd is not None else None,
            "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
        })
    return rows


# ----------------------------------------------------------------------- Spearman
@cache
def _spearman_null(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact null distribution of S = sum(i * pi(i)) over all n! permutations.

    Returns (distinct S values ascending, counts). rho is a linear function of S, so this
    is the exact permutation distribution of Spearman's rho when there are no ties.
    """
    base = np.arange(1, n + 1, dtype=np.int64)
    perms = np.array(list(permutations(range(1, n + 1))), dtype=np.int64)
    s = perms @ base
    values, counts = np.unique(s, return_counts=True)
    return values, counts


def _spearman_exact_p(rank_x: np.ndarray, rank_y: np.ndarray, alternative: str) -> float:
    n = rank_x.size
    values, counts = _spearman_null(n)
    total = float(counts.sum())
    s_obs = int(round(float(rank_x @ rank_y)))
    centre = n * (n + 1) ** 2 / 4.0  # E[S] under H0
    if alternative == "greater":
        hit = values >= s_obs
    elif alternative == "less":
        hit = values <= s_obs
    else:
        hit = np.abs(values - centre) >= abs(s_obs - centre) - 1e-9
    return float(counts[hit].sum() / total)


def spearman(
    x: Any,
    y: Any,
    *,
    names: tuple[str, str] = ("X", "Y"),
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ci_level: float = 0.95,
) -> StatResult:
    """Spearman rank correlation (Pearson's r computed on average ranks, so ties are handled).

    P value: the exact permutation distribution when n <= 9 and there are no ties, otherwise
    the t approximation t = rho sqrt((n - 2) / (1 - rho^2)) with n - 2 df (the result says
    which). CI: Fisher z with the Bonett & Wright (2000) standard error
    sqrt((1 + rho^2 / 2) / (n - 3)).
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    ci_level = _check_level(ci_level)
    xa, ya, notes = _prepare(x, y, names)
    n = int(xa.size)
    warns: list[str] = []
    rx = stats.rankdata(xa, method="average")
    ry = stats.rankdata(ya, method="average")
    ties_x = int(n - np.unique(xa).size)
    ties_y = int(n - np.unique(ya).size)
    has_ties = bool(ties_x or ties_y)
    rho = _r_value(rx, ry)
    if not math.isfinite(rho):
        raise StatsError("The rank correlation could not be computed (the values do not vary enough).")
    t, p_t = _t_pvalue(rho, n, alternative)
    if n <= EXACT_SPEARMAN_MAX_N and not has_ties:
        p = _spearman_exact_p(rx, ry, alternative)
        p_method = "exact permutation distribution"
    else:
        p = p_t
        p_method = "t approximation with n - 2 degrees of freedom"
        if n <= EXACT_SPEARMAN_MAX_N and has_ties:
            warns.append("Tied values are present, so the exact P value is unavailable and the "
                         "t approximation was used; with so few pairs it is only rough.")
        elif n < 10:
            warns.append("Small sample: the t approximation to the P value is rough.")
    if n == 2:
        warns.append("With only 2 pairs the rank correlation is always +1 or -1; nothing can be "
                     "concluded. Collect more data.")
    ci_low = ci_high = None
    if n > 3:
        se_z = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
        ci_low, ci_high = _fisher_interval(rho, se_z, ci_level, alternative)
    elif n == 3:
        warns.append("A confidence interval for rho needs at least 4 pairs.")
    if has_ties:
        notes = [*notes, f"Tied values received average ranks ({ties_x} tie(s) in {names[0]}, "
                         f"{ties_y} in {names[1]})."]

    assumptions = [
        AssumptionCheck(
            name="Monotonic relationship", status="info",
            detail="Spearman's rho measures whether one variable consistently increases (or "
                   "decreases) as the other increases, whatever the shape. It misses relationships "
                   "that go up and then down."),
        AssumptionCheck(
            name="Independent pairs", status="info",
            detail="Each pair must come from a different subject/sample."),
        AssumptionCheck(
            name="P value method", status="pass" if p_method.startswith("exact") else "info",
            detail=f"P value from the {p_method}."
                   + (" Ties prevent an exact calculation." if has_ties else "")),
    ]
    stat_t = t if (t is not None and math.isfinite(t)) else None
    result = StatResult(
        analysis="correlation",
        test_name="Spearman rank correlation",
        why_chosen="Spearman correlation works on ranks, so it does not need normally distributed "
                   "data and is not thrown off by outliers; it detects any consistently increasing "
                   "or decreasing relationship.",
        assumptions=assumptions,
        statistic_name="rho",
        statistic=rho,
        df=float(n - 2) if (n > 2 and not p_method.startswith("exact")) else None,
        p_value=p,
        alpha=alpha,
        alternative=alternative,
        effect_sizes=[
            EffectSize(name="Spearman rho", value=rho, ci_low=ci_low, ci_high=ci_high,
                       ci_level=ci_level,
                       ci_method="Fisher z with the Bonett-Wright (2000) standard error"
                       if ci_low is not None else None,
                       magnitude=magnitude_r(rho)),
        ],
        parameters=[Parameter(name="rho", value=rho, ci_low=ci_low, ci_high=ci_high,
                              ci_level=ci_level, statistic=stat_t, p_value=p)],
        n={f"{names[0]} / {names[1]} pairs": n},
        descriptives=_describe(xa, ya, names),
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_SPEARMAN, _CITE_BW],
        extra={"rho": rho, "method": "spearman", "p_method": p_method, "t_approximation": stat_t,
               "p_t_approximation": p_t, "x_name": names[0], "y_name": names[1],
               "scatter": {"x": xa.tolist(), "y": ya.tolist()}},
    )
    ci_text = f", {ci_phrase(ci_low, ci_high, ci_level)}" if ci_low is not None else ""
    result.interpretation = (
        f"{names[0]} and {names[1]} show a {_strength(rho)} monotonic association "
        f"(Spearman rho = {fmt(rho, 3)}{ci_text}; n = {n} pairs), which is {sig_phrase(p, alpha)} "
        f"({p_method}). A positive rho means that higher values of one variable tend to go with "
        "higher values of the other; it does not show that one causes the other."
    )
    result.methods_text = (
        f"The association between {names[0]} and {names[1]} was assessed with the Spearman rank "
        f"correlation ({alternative_label(alternative)}, alpha = {alpha:g}; P value from the "
        f"{p_method}); the {ci_level * 100:g}% confidence interval used the Fisher z transformation "
        "with the Bonett-Wright standard error."
    )
    return result


# ------------------------------------------------------------------------- matrix
def correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    adjust: str = "none",
    alpha: float = 0.05,
    ci_level: float = 0.95,
) -> StatResult:
    """All pairwise correlations among the numeric columns of ``df``.

    Each pair uses the rows where *both* variables are present (pairwise-complete), so n
    can differ between pairs. P values are two-sided (t approximation for Spearman) and can
    be adjusted across the unique pairs with any method of
    :func:`torusplot.stats.multiplicity.adjust_pvalues`.
    """
    alpha = check_alpha(alpha)
    ci_level = _check_level(ci_level)
    method = str(method).lower()
    if method not in {"pearson", "spearman"}:
        raise StatsError("method must be 'pearson' or 'spearman'.")
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except (TypeError, ValueError) as exc:
            raise StatsError("The data for a correlation matrix must be a table of columns.") from exc
    notes: list[str] = []
    numeric: dict[str, np.ndarray] = {}
    skipped = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            skipped.append(str(col))
            continue
        arr = series.to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        numeric[str(col)] = arr
    if skipped:
        notes.append(f"Non-numeric column(s) left out: {', '.join(skipped)}.")
    labels = list(numeric)
    k = len(labels)
    if k < 2:
        raise StatsError("A correlation matrix needs at least two numeric columns.")
    try:
        adjust_pvalues([0.5], adjust)
    except ValueError as exc:
        raise StatsError(str(exc)) from exc

    r_mat = [[None] * k for _ in range(k)]
    p_mat = [[None] * k for _ in range(k)]
    n_mat = [[0] * k for _ in range(k)]
    pairs: list[dict[str, Any]] = []
    warns: list[str] = []
    effect_sizes: list[EffectSize] = []
    for i, a in enumerate(labels):
        ok = ~np.isnan(numeric[a])
        n_mat[i][i] = int(ok.sum())
        r_mat[i][i] = 1.0 if (n_mat[i][i] >= 2 and not is_constant(numeric[a][ok])) else None
    for i in range(k):
        for j in range(i + 1, k):
            a, b = labels[i], labels[j]
            ok = ~np.isnan(numeric[a]) & ~np.isnan(numeric[b])
            xa, ya = numeric[a][ok], numeric[b][ok]
            n = int(ok.sum())
            n_mat[i][j] = n_mat[j][i] = n
            r = t = p = lo = hi = None
            if n < 3:
                warns.append(f"{a} vs {b}: only {n} complete pair(s); correlation not computed.")
            elif is_constant(xa) or is_constant(ya):
                warns.append(f"{a} vs {b}: one variable is constant; correlation is undefined.")
            else:
                if method == "spearman":
                    xa = stats.rankdata(xa, method="average")
                    ya = stats.rankdata(ya, method="average")
                rv = _r_value(xa, ya)
                if math.isfinite(rv):
                    r = rv
                    t, p = _t_pvalue(rv, n, "two-sided")
                    if n > 3:
                        se = (math.sqrt((1.0 + rv * rv / 2.0) / (n - 3)) if method == "spearman"
                              else 1.0 / math.sqrt(n - 3))
                        lo, hi = _fisher_interval(rv, se, ci_level)
            r_mat[i][j] = r_mat[j][i] = r
            p_mat[i][j] = p_mat[j][i] = p
            pairs.append({"a": a, "b": b, "r": r, "n": n, "p": p, "ci_low": lo, "ci_high": hi,
                          "t": t if (t is not None and math.isfinite(t)) else None,
                          "df": n - 2 if r is not None else None})
            effect_sizes.append(EffectSize(
                name="Spearman rho" if method == "spearman" else "Pearson r", value=r,
                ci_low=lo, ci_high=hi, ci_level=ci_level,
                ci_method=None if lo is None else (
                    "Fisher z with the Bonett-Wright (2000) standard error" if method == "spearman"
                    else "Fisher z transformation"),
                magnitude=magnitude_r(r) if r is not None else None, target=f"{a} vs {b}"))
    raw = np.array([math.nan if pr["p"] is None else pr["p"] for pr in pairs], dtype=float)
    adjusted = adjust_pvalues(raw, adjust)
    padj_mat = [[None] * k for _ in range(k)]
    idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            value = None if math.isnan(adjusted[idx]) else float(adjusted[idx])
            pairs[idx]["p_adjusted"] = value
            pairs[idx]["significant"] = None if value is None else bool(value < alpha)
            padj_mat[i][j] = padj_mat[j][i] = value
            idx += 1
    n_values = sorted({pr["n"] for pr in pairs})
    if len(n_values) > 1:
        notes.append(
            f"Missing values were handled pair by pair (pairwise-complete): n ranges from "
            f"{n_values[0]} to {n_values[-1]} across pairs."
        )
    n_pairs = len(pairs)
    label = "Spearman" if method == "spearman" else "Pearson"
    symbol = "rho" if method == "spearman" else "r"
    valid = [pr for pr in pairs if pr["r"] is not None]
    n_sig = sum(1 for pr in valid if pr["significant"])
    adj_key = str(adjust).lower()
    correction = METHOD_LABELS.get(adj_key, str(adjust))
    if adj_key == "none" and n_pairs > 1:
        warns.append(
            f"{n_pairs} correlations were tested without adjusting for multiple comparisons, so "
            "some 'significant' results are expected by chance alone. Consider the Holm or "
            "Benjamini-Hochberg adjustment."
        )
    assumptions = [
        AssumptionCheck(
            name="Linear relationship" if method == "pearson" else "Monotonic relationship",
            status="info",
            detail=("Pearson's r measures straight-line association and assumes roughly normal "
                    "data; inspect scatter plots of the pairs you care about."
                    if method == "pearson" else
                    "Spearman's rho measures any consistently increasing or decreasing "
                    "relationship and does not assume normality.")),
        AssumptionCheck(
            name="Independent rows", status="info",
            detail="Each row must be a different subject/sample."),
        AssumptionCheck(
            name="Multiple comparisons", status="info" if adj_key != "none" or n_pairs == 1 else "warn",
            detail=f"{n_pairs} unique pair(s) tested; P value adjustment: {correction}."),
    ]
    result = StatResult(
        analysis="correlation_matrix",
        test_name=f"{label} correlation matrix",
        why_chosen="A correlation matrix screens every pair of variables at once to see which "
                   "ones rise and fall together.",
        assumptions=assumptions,
        alpha=alpha,
        effect_sizes=effect_sizes,
        n={lab: n_mat[i][i] for i, lab in enumerate(labels)},
        n_total=int(len(df)),
        comparison_family=f"all {n_pairs} pairwise correlations among {k} variables",
        correction=correction,
        warnings=warns,
        data_notes=notes,
        citations=[_CITE_SPEARMAN, _CITE_BW] if method == "spearman" else [_CITE_PEARSON, _CITE_FISHER],
        extra={
            "matrix": {"labels": labels, "r": r_mat, "p": p_mat, "n": n_mat,
                       "p_adjusted": padj_mat, "method": method, "adjust": adj_key},
            "pairs": pairs,
        },
    )
    if valid:
        top = max(valid, key=lambda pr: abs(pr["r"]))
        strongest = (f"The strongest association is between {top['a']} and {top['b']} "
                     f"({symbol} = {fmt(top['r'], 3)}, n = {top['n']}, "
                     f"{'adjusted ' if adj_key != 'none' else ''}{p_phrase(top['p_adjusted'])}).")
    else:
        strongest = "No correlation could be computed."
    result.interpretation = (
        f"{label} correlations were computed for all {n_pairs} pair(s) among {k} variables. "
        f"{n_sig} of {len(valid)} computable pair(s) are statistically significant at alpha = "
        f"{alpha:g} ({correction}). {strongest} Correlation does not show causation."
    )
    result.methods_text = (
        f"Pairwise {label} correlation coefficients were computed using pairwise-complete "
        f"observations; two-tailed P values"
        + (f" were adjusted for multiple comparisons with the {correction} method"
           if adj_key != "none" else " were not adjusted for multiple comparisons")
        + f" (alpha = {alpha:g})."
    )
    return result


# ---------------------------------------------------------------- table adapters
def _pick(names: list[str], wanted: Any, what: str, default_index: int) -> str:
    if wanted is None or wanted == "":
        if len(names) <= default_index:
            raise StatsError(f"The table does not have enough {what}s for this analysis.")
        return names[default_index]
    if str(wanted) not in names:
        raise StatsError(f"Unknown {what} '{wanted}'. Available: {', '.join(names)}.")
    return str(wanted)


_METHOD_PARAM = ParamSpec(
    "method", "Method", kind="choice", default="pearson", choices=("pearson", "spearman"),
    help="Pearson measures straight-line association and assumes roughly normal data. "
         "Spearman works on ranks: use it for skewed data, outliers, or curved-but-steady trends.")
_ALPHA_PARAM = ParamSpec(
    "alpha", "Significance level (alpha)", kind="float", default=0.05,
    help="The P value threshold for calling a result significant. 0.05 is the usual choice.")
_CI_PARAM = ParamSpec(
    "ci_level", "Confidence level", kind="float", default=0.95,
    help="Confidence level of the reported intervals and bands. 0.95 gives 95% confidence "
         "intervals.")


@register(
    key="correlation", label="Correlation (Pearson or Spearman)", category="Correlation",
    tables=("xy", "column", "multiple_variables"),
    params=(
        ParamSpec("series", "Y series (XY tables)", kind="group", default=None, optional=True,
                  help="XY tables: which Y data set to correlate with X. Leave empty to use the first."),
        ParamSpec("use_means", "Use the mean of replicates (XY tables)", kind="bool", default=False,
                  help="XY tables with replicates: tick to correlate X with the mean Y at each X. "
                       "Replicates of one sample are not independent, so using means is usually "
                       "the safer choice."),
        ParamSpec("group_a", "First column (Column tables)", kind="group", default=None, optional=True,
                  help="Column tables: the first variable. Each row must be one subject measured "
                       "on both columns. Leave empty to use the first column."),
        ParamSpec("group_b", "Second column (Column tables)", kind="group", default=None, optional=True,
                  help="Column tables: the second variable. Leave empty to use the second column."),
        ParamSpec("x", "First variable (Multiple variables tables)", kind="column", default=None,
                  optional=True,
                  help="Multiple variables tables: the first numeric variable. Leave empty to use "
                       "the first numeric column."),
        ParamSpec("y", "Second variable (Multiple variables tables)", kind="column", default=None,
                  optional=True,
                  help="Multiple variables tables: the second numeric variable. Leave empty to use "
                       "the second numeric column."),
        _METHOD_PARAM,
        ParamSpec("alternative", "Alternative hypothesis", kind="choice", default="two-sided",
                  choices=("two-sided", "greater", "less"),
                  help="Two-sided asks whether there is any correlation. 'greater' tests only for a "
                       "positive correlation and 'less' only for a negative one; choose these only "
                       "if you decided the direction before seeing the data."),
        _ALPHA_PARAM,
        _CI_PARAM,
    ),
    description="Measures how strongly two measurements rise and fall together, with a P value "
                "and a confidence interval.",
    citation=f"{_CITE_PEARSON}; {_CITE_SPEARMAN}",
)
def _run_correlation(table: Any, *, series: str | None = None, use_means: bool = False,
                     group_a: str | None = None, group_b: str | None = None,
                     x: str | None = None, y: str | None = None, method: str = "pearson",
                     alternative: str = "two-sided", alpha: float = 0.05,
                     ci_level: float = 0.95) -> StatResult:
    method = str(method).lower()
    if method not in {"pearson", "spearman"}:
        raise StatsError("method must be 'pearson' or 'spearman'.")
    notes: list[str] = []
    warns: list[str] = []
    try:
        if table.kind == "xy":
            name = _pick(table.series_names, series, "series", 0)
            names = (table.x_name, name)
            n_rep = len(table.series_map[name])
            if use_means:
                means = table.xy_means(name)
                xv, yv = means["x"].to_numpy(dtype=float), means["mean"].to_numpy(dtype=float)
                if n_rep > 1:
                    notes.append(f"Replicates of {name} were averaged at each X before the analysis.")
            else:
                xv, yv = table.xy_points(name)
                if n_rep > 1:
                    warns.append(
                        "Each replicate was treated as an independent point. If replicates are "
                        "repeat measurements of the same sample (pseudo-replication), the P value "
                        "is too small and the CI too narrow: tick 'Use the mean of replicates'."
                    )
        elif table.kind == "column":
            a = _pick(table.group_names, group_a, "column", 0)
            b = _pick(table.group_names, group_b, "column", 1)
            if a == b:
                raise StatsError("Choose two different columns to correlate.")
            frame = table.analysis_frame()[[a, b]]
            xv, yv = frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float)
            names = (a, b)
            notes.append("Rows were treated as pairs (one subject per row).")
        else:
            numeric = table.numeric_columns
            a = _pick(numeric, x, "numeric column", 0)
            b = _pick(numeric, y, "numeric column", 1)
            if a == b:
                raise StatsError("Choose two different variables to correlate.")
            frame = table.frame([a, b], dropna=False)
            xv, yv = frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float)
            names = (a, b)
    except TableError as exc:
        raise StatsError(str(exc)) from exc
    func = spearman if method == "spearman" else pearson
    result = func(xv, yv, names=names, alpha=alpha, alternative=alternative, ci_level=ci_level)
    result.data_notes.extend(notes)
    result.warnings.extend(warns)
    return result


@register(
    key="correlation_matrix", label="Correlation matrix", category="Correlation",
    tables=("multiple_variables", "column"),
    params=(
        ParamSpec("columns", "Variables", kind="columns", default=None, optional=True,
                  help="Which numeric variables to include. Leave empty to use all of them."),
        _METHOD_PARAM,
        ParamSpec("adjust", "Multiple-comparison adjustment", kind="choice", default="none",
                  choices=("none", "bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh"),
                  help="With many pairs, some look significant by chance. Holm controls the chance "
                       "of any false positive; fdr_bh (Benjamini-Hochberg) controls the expected "
                       "fraction of false positives."),
        _ALPHA_PARAM,
        _CI_PARAM,
    ),
    description="Computes the correlation between every pair of variables and shows them as a "
                "matrix, optionally adjusting the P values for the number of pairs.",
    citation=f"{_CITE_PEARSON}; {_CITE_SPEARMAN}",
)
def _run_correlation_matrix(table: Any, *, columns: list[str] | None = None,
                            method: str = "pearson", adjust: str = "none",
                            alpha: float = 0.05, ci_level: float = 0.95) -> StatResult:
    try:
        if table.kind == "column":
            frame = table.analysis_frame()
            available = table.group_names
        else:
            frame = table.frame(dropna=False)
            available = table.numeric_columns
    except TableError as exc:
        raise StatsError(str(exc)) from exc
    if columns:
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(",") if c.strip()]
        unknown = [c for c in columns if c not in available]
        if unknown:
            raise StatsError(
                f"Unknown or non-numeric variable(s): {', '.join(map(str, unknown))}. "
                f"Available: {', '.join(available)}."
            )
        selected = [str(c) for c in columns]
    else:
        selected = list(available)
    if len(selected) < 2:
        raise StatsError("A correlation matrix needs at least two numeric variables.")
    result = correlation_matrix(frame[selected], method=method, adjust=adjust, alpha=alpha,
                                ci_level=ci_level)
    if table.kind == "column":
        result.data_notes.append("Rows were treated as subjects measured on every column.")
    return result
