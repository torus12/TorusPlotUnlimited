"""Effect sizes with confidence intervals (all implemented in TorusPlot).

References
----------
* Cohen J (1988) *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
  (d, f, eta-squared conventions).
* Hedges LV (1981) Distribution theory for Glass's estimator of effect size and
  related estimators. *J Educ Stat* 6:107-128 (small-sample correction, g).
* Steiger JH, Fouladi RT (1997) Noncentrality interval estimation and the evaluation
  of statistical models. In: *What if there were no significance tests?*
* Cumming G, Finch S (2001) A primer on the understanding, use, and calculation of
  confidence intervals that are based on central and noncentral distributions.
  *Educ Psychol Meas* 61:532-574 (noncentral-t CI for d).
* Steiger JH (2004) Beyond the F test: effect size confidence intervals and tests of
  close fit in the analysis of variance and contrast analysis. *Psychol Methods*
  9:164-182 (noncentral-F CI for eta-squared).
* Olejnik S, Algina J (2003) Generalized eta and omega squared statistics.
  *Psychol Methods* 8:434-447; Hays WL (1963) *Statistics for Psychologists* (omega²).
* Kerby DS (2014) The simple difference formula: an approach to teaching
  nonparametric correlation. *Compr Psychol* 3:11.IT.3.1 (rank-biserial r).
* Cureton EE (1956) Rank-biserial correlation. *Psychometrika* 21:287-290.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special, stats

from .result import EffectSize

# ----------------------------------------------------------------- magnitude labels


def magnitude_d(d: float | None) -> str | None:
    """Cohen (1988) benchmarks for standardized mean differences."""
    if d is None or math.isnan(d):
        return None
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def magnitude_eta2(eta2: float | None) -> str | None:
    """Cohen (1988) benchmarks for eta-squared-type measures (0.01 / 0.06 / 0.14)."""
    if eta2 is None or math.isnan(eta2):
        return None
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


def magnitude_r(r: float | None) -> str | None:
    """Cohen (1988) benchmarks for correlation-type measures (0.1 / 0.3 / 0.5)."""
    if r is None or math.isnan(r):
        return None
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


# ------------------------------------------------------------- noncentral intervals


def _nct_ncp_limits(t_obs: float, df: float, level: float) -> tuple[float, float]:
    """Noncentrality parameters whose noncentral-t distributions put ``t_obs`` at the
    upper / lower ``(1 - level) / 2`` tail (Steiger & Fouladi 1997)."""
    if not np.isfinite(t_obs) or df <= 0:
        return (math.nan, math.nan)
    alpha = 1.0 - level
    span = abs(t_obs) + 10.0 + 4.0 * math.sqrt(1.0 + t_obs * t_obs / max(df, 1.0))

    def solve(target: float) -> float:
        def f(ncp: float) -> float:
            # The CDF decreases monotonically in ncp; scipy returns NaN in the extreme
            # tails, where the true value is 0 (ncp far above t) or 1 (far below).
            value = float(stats.nct.cdf(t_obs, df, ncp))
            if math.isnan(value):
                value = 0.0 if ncp > t_obs else 1.0
            return value - target

        lo, hi = t_obs - span, t_obs + span
        try:
            if f(lo) * f(hi) > 0:
                return math.nan
            return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200))
        except (ValueError, RuntimeError):
            return math.nan

    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)


def _ncf_ncp_limits(f_obs: float, df1: float, df2: float, level: float) -> tuple[float, float]:
    """Lower / upper noncentrality parameter of the noncentral F for ``f_obs``
    (Steiger 2004). The lower limit is 0 when F is not significant at that level."""
    if not np.isfinite(f_obs) or f_obs < 0 or df1 <= 0 or df2 <= 0:
        return (math.nan, math.nan)
    alpha = 1.0 - level

    def cdf(ncp: float) -> float:
        if ncp <= 0:
            return float(stats.f.cdf(f_obs, df1, df2))
        return float(stats.ncf.cdf(f_obs, df1, df2, ncp))

    def solve(target: float) -> float:
        if cdf(0.0) <= target:
            return 0.0
        hi = max(10.0, 3.0 * f_obs * df1 + 10.0)
        tries = 0
        while cdf(hi) > target and tries < 60:
            hi *= 2.0
            tries += 1
        if cdf(hi) > target:
            return math.nan
        try:
            return float(optimize.brentq(lambda v: cdf(v) - target, 0.0, hi, xtol=1e-10, maxiter=300))
        except (ValueError, RuntimeError):
            return math.nan

    return solve(1.0 - alpha / 2.0), solve(alpha / 2.0)


# --------------------------------------------------- standardized mean differences


def hedges_correction(df: float) -> float:
    """Exact small-sample bias correction J(df) of Hedges (1981)."""
    if df <= 1:
        return math.nan
    return float(math.exp(special.gammaln(df / 2.0) - special.gammaln((df - 1.0) / 2.0)) / math.sqrt(df / 2.0))


def _smd_result(
    name: str, d: float, t_obs: float, df: float, scale: float, level: float, correct: bool, target: str | None
) -> EffectSize:
    """Build an EffectSize for a standardized mean difference.

    ``scale`` converts a noncentrality parameter into d (d = ncp * scale).
    """
    lo_ncp, hi_ncp = _nct_ncp_limits(t_obs, df, level)
    lo, hi = lo_ncp * scale, hi_ncp * scale
    j = hedges_correction(df) if correct else 1.0
    if math.isnan(j):  # Hedges' correction is undefined for df <= 1 (e.g. n = 2 paired)
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    value = d * j
    return EffectSize(
        name=name,
        value=value,
        ci_low=lo * j,
        ci_high=hi * j,
        ci_level=level,
        ci_method="noncentral t (Steiger & Fouladi 1997; Cumming & Finch 2001)",
        magnitude=magnitude_d(value),
        target=target,
    )


def cohen_d_independent(
    x: np.ndarray, y: np.ndarray, level: float = 0.95, hedges: bool = False, target: str | None = None
) -> EffectSize:
    """Cohen's d (or Hedges' g) for two independent groups, pooled SD.

    d = (mean(x) - mean(y)) / s_pooled, with the CI obtained by inverting the
    noncentral t distribution. ``hedges=True`` applies Hedges' exact correction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    name = "Hedges' g" if hedges else "Cohen's d"
    df = n1 + n2 - 2
    if n1 < 2 or n2 < 2:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / df
    if sp2 <= 0:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    d = float((x.mean() - y.mean()) / math.sqrt(sp2))
    scale = math.sqrt(1.0 / n1 + 1.0 / n2)
    return _smd_result(name, d, d / scale, df, scale, level, hedges, target)


def cohen_d_paired(
    x: np.ndarray, y: np.ndarray, level: float = 0.95, hedges: bool = False, target: str | None = None
) -> EffectSize:
    """Cohen's d_z for paired data: mean of the differences / SD of the differences."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return cohen_d_one_sample(
        diff, 0.0, level=level, hedges=hedges, target=target,
        name="Hedges' g_z (paired)" if hedges else "Cohen's d_z (paired)",
    )


def cohen_d_one_sample(
    x: np.ndarray, popmean: float = 0.0, level: float = 0.95, hedges: bool = False,
    target: str | None = None, name: str | None = None,
) -> EffectSize:
    """Cohen's d for one sample against a hypothetical mean: (mean - mu) / SD."""
    x = np.asarray(x, dtype=float)
    n = x.size
    name = name or ("Hedges' g" if hedges else "Cohen's d")
    if n < 2:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    sd = x.std(ddof=1)
    if sd <= 0:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    d = float((x.mean() - popmean) / sd)
    scale = 1.0 / math.sqrt(n)
    return _smd_result(name, d, d / scale, n - 1, scale, level, hedges, target)


def cohen_d_from_t(
    t: float, n1: int, n2: int, level: float = 0.95, hedges: bool = False, target: str | None = None
) -> EffectSize:
    """d from a pooled-variance t statistic (used by post-hoc tables)."""
    scale = math.sqrt(1.0 / n1 + 1.0 / n2)
    name = "Hedges' g" if hedges else "Cohen's d"
    return _smd_result(name, t * scale, t, n1 + n2 - 2, scale, level, hedges, target)


# ------------------------------------------------------------- variance explained


def eta_squared_ci(f_obs: float, df1: float, df2: float, level: float = 0.95) -> tuple[float, float]:
    """CI for the population proportion of variance (eta² / partial eta² / omega²)
    from an F statistic, by inverting the noncentral F: eta² = ncp / (ncp + df1 + df2 + 1)."""
    lo, hi = _ncf_ncp_limits(f_obs, df1, df2, level)
    n = df1 + df2 + 1.0
    return (lo / (lo + n) if not math.isnan(lo) else math.nan,
            hi / (hi + n) if not math.isnan(hi) else math.nan)


def eta_squared(ss_effect: float, ss_total: float, f_obs: float | None = None, df1: float | None = None,
                df2: float | None = None, level: float = 0.95, target: str | None = None) -> EffectSize:
    """eta² = SS_effect / SS_total. A CI is given when F and its df are supplied *and*
    the design is one-way (where eta² equals partial eta²)."""
    value = float(ss_effect / ss_total) if ss_total > 0 else math.nan
    lo = hi = None
    method = None
    if f_obs is not None and df1 is not None and df2 is not None:
        lo, hi = eta_squared_ci(f_obs, df1, df2, level)
        method = "noncentral F (Steiger 2004)"
    return EffectSize(name="eta squared", value=value, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method=method, magnitude=magnitude_eta2(value), target=target)


def partial_eta_squared(ss_effect: float, ss_error: float, df1: float, df2: float, level: float = 0.95,
                        target: str | None = None) -> EffectSize:
    """Partial eta² = SS_effect / (SS_effect + SS_error), CI by noncentral F."""
    denom = ss_effect + ss_error
    value = float(ss_effect / denom) if denom > 0 else math.nan
    lo = hi = None
    method = None
    if ss_error > 0 and df1 > 0 and df2 > 0:
        f_obs = (ss_effect / df1) / (ss_error / df2)
        lo, hi = eta_squared_ci(f_obs, df1, df2, level)
        method = "noncentral F (Steiger 2004)"
    return EffectSize(name="partial eta squared", value=value, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method=method, magnitude=magnitude_eta2(value), target=target)


def partial_eta_squared_from_f(f_obs: float, df1: float, df2: float, level: float = 0.95,
                               target: str | None = None) -> EffectSize:
    """Partial eta² from F: F*df1 / (F*df1 + df2)."""
    if not np.isfinite(f_obs) or df1 <= 0 or df2 <= 0:
        return EffectSize(name="partial eta squared", value=None, ci_level=level, target=target)
    value = float(f_obs * df1 / (f_obs * df1 + df2))
    lo, hi = eta_squared_ci(f_obs, df1, df2, level)
    return EffectSize(name="partial eta squared", value=value, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method="noncentral F (Steiger 2004)", magnitude=magnitude_eta2(value), target=target)


def omega_squared(ss_effect: float, df_effect: float, ms_error: float, ss_total: float,
                  f_obs: float | None = None, df_error: float | None = None, level: float = 0.95,
                  target: str | None = None) -> EffectSize:
    """omega² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error) (Hays 1963).

    Less biased than eta². Negative estimates are reported as 0 (the usual convention).
    The CI (one-way designs) is the noncentral-F interval for the population value.
    """
    denom = ss_total + ms_error
    raw = float((ss_effect - df_effect * ms_error) / denom) if denom > 0 else math.nan
    value = max(raw, 0.0) if not math.isnan(raw) else raw
    lo = hi = None
    method = None
    if f_obs is not None and df_error is not None:
        lo, hi = eta_squared_ci(f_obs, df_effect, df_error, level)
        method = "noncentral F (Steiger 2004)"
    return EffectSize(name="omega squared", value=value, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method=method, magnitude=magnitude_eta2(value), target=target)


def partial_omega_squared(f_obs: float, df1: float, df2: float, n_total: int, level: float = 0.95,
                          target: str | None = None) -> EffectSize:
    """Partial omega² = df1 (F - 1) / (df1 (F - 1) + N) (Olejnik & Algina 2003)."""
    if not np.isfinite(f_obs) or n_total <= 0:
        return EffectSize(name="partial omega squared", value=None, ci_level=level, target=target)
    raw = df1 * (f_obs - 1.0) / (df1 * (f_obs - 1.0) + n_total)
    value = max(float(raw), 0.0)
    lo, hi = eta_squared_ci(f_obs, df1, df2, level)
    return EffectSize(name="partial omega squared", value=value, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method="noncentral F (Steiger 2004)", magnitude=magnitude_eta2(value), target=target)


def cohen_f_from_eta2(eta2: float) -> float:
    """Cohen's f = sqrt(eta² / (1 - eta²)); used for ANOVA power calculations."""
    if eta2 is None or math.isnan(eta2) or eta2 >= 1 or eta2 < 0:
        return math.nan
    return math.sqrt(eta2 / (1.0 - eta2))


# ------------------------------------------------------------------- rank-based


def _fisher_ci(r: float, se_z: float, level: float) -> tuple[float, float]:
    if not np.isfinite(r) or not np.isfinite(se_z):
        return (math.nan, math.nan)
    if abs(r) >= 1.0:
        return (r, r)
    z = math.atanh(r)
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    return (math.tanh(z - crit * se_z), math.tanh(z + crit * se_z))


def rank_biserial_independent(x: np.ndarray, y: np.ndarray, level: float = 0.95,
                              target: str | None = None) -> EffectSize:
    """Rank-biserial correlation for two independent groups (Cureton 1956; Kerby 2014).

    r = (favourable pairs - unfavourable pairs) / (n1 * n2) = 2*U_x / (n1*n2) - 1,
    where U_x counts pairs with x > y (ties count 1/2). Positive values mean ``x``
    tends to be larger. CI: Fisher z transform with SE = sqrt((n1 + n2 + 1) / (3 n1 n2)).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 1 or n2 < 1:
        return EffectSize(name="rank-biserial r", value=None, ci_level=level, target=target)
    ranks = stats.rankdata(np.concatenate([x, y]))
    u_x = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    r = 2.0 * u_x / (n1 * n2) - 1.0
    se = math.sqrt((n1 + n2 + 1.0) / (3.0 * n1 * n2))
    lo, hi = _fisher_ci(r, se, level)
    return EffectSize(name="rank-biserial r", value=r, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method="Fisher z approximation", magnitude=magnitude_r(r), target=target)


def rank_biserial_paired(diff: np.ndarray, level: float = 0.95, target: str | None = None) -> EffectSize:
    """Matched-pairs rank-biserial correlation (Kerby 2014).

    r = (W+ - W-) / (W+ + W-) using ranks of the non-zero absolute differences.
    CI: Fisher z with SE = sqrt((2n^3 + 3n^2 + n) / 6) / ((n^2 + n) / 2).
    """
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    n = d.size
    name = "matched-pairs rank-biserial r"
    if n < 1:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    ranks = stats.rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = w_plus + w_minus
    r = (w_plus - w_minus) / total
    se = math.sqrt((2.0 * n**3 + 3.0 * n**2 + n) / 6.0) / ((n**2 + n) / 2.0)
    lo, hi = _fisher_ci(r, se, level)
    return EffectSize(name=name, value=r, ci_low=lo, ci_high=hi, ci_level=level,
                      ci_method="Fisher z approximation", magnitude=magnitude_r(r), target=target)


def epsilon_squared_kw(h: float, n_total: int, k: int, target: str | None = None) -> EffectSize:
    """Rank epsilon-squared for Kruskal-Wallis: H / (N - 1) (Kelley 1935; Tomczak & Tomczak 2014)."""
    value = float(h / (n_total - 1)) if n_total > 1 else math.nan
    return EffectSize(name="epsilon squared (rank)", value=value, magnitude=magnitude_eta2(value), target=target)


def kendall_w(chi2: float, n_subjects: int, k: int, target: str | None = None) -> EffectSize:
    """Kendall's W for Friedman's test: chi² / (n (k - 1))."""
    value = float(chi2 / (n_subjects * (k - 1))) if n_subjects > 0 and k > 1 else math.nan
    return EffectSize(name="Kendall's W", value=value, magnitude=magnitude_r(value), target=target)
