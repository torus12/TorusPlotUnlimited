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
* Newcombe RG (2006) Confidence intervals for an effect size measure based on the
  Mann-Whitney statistic. Part 1: general issues and tail-area-based methods; Part 2:
  asymptotic methods and evaluation. *Stat Med* 25:543-557 and 559-573 (score-type
  interval for the Mann-Whitney probability, "method 5"; CI of the rank-biserial r).
* Hanley JA, McNeil BJ (1982) The meaning and use of the area under a receiver
  operating characteristic (ROC) curve. *Radiology* 143:29-36 (variance of the
  Mann-Whitney probability).
* DeLong ER, DeLong DM, Clarke-Pearson DL (1988) Comparing the areas under two or more
  correlated receiver operating characteristic curves: a nonparametric approach.
  *Biometrics* 44:837-845; Sen PK (1960) On some convergence properties of U-statistics.
  *Calcutta Stat Assoc Bull* 10:1-18 (distribution-free variance from placement values).
* Hollander M, Wolfe DA (1999) *Nonparametric Statistical Methods*, 2nd ed. (variance of
  the signed-rank statistic under alternatives; Walsh averages).

All standardized effect sizes are computed on internally rescaled data (see
``_utils.rescale``), so they are unaffected by the unit or offset of the measurements.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special, stats

from ._utils import difference_tolerance, differences_constant, is_constant, rescale
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


_SMD_CI_METHOD = "noncentral t (Steiger & Fouladi 1997; Cumming & Finch 2001)"
_HEDGES_CI_METHOD = (
    "noncentral t interval for the population standardized difference (Steiger & Fouladi 1997; "
    "Cumming & Finch 2001), the same interval as for Cohen's d: the small-sample correction is "
    "applied to the point estimate only, not to the limits"
)


def _smd_result(
    name: str, d: float, t_obs: float, df: float, scale: float, level: float, correct: bool, target: str | None
) -> EffectSize:
    """Build an EffectSize for a standardized mean difference.

    ``scale`` converts a noncentrality parameter into d (d = ncp * scale).

    The interval is the noncentral-t interval for the *population* standardized
    difference delta. For Hedges' g the point estimate is multiplied by the bias
    correction J(df) but the limits are not: J removes the small-sample bias of the
    estimator, it does not change the parameter, and limits shrunk by J < 1 would cover
    delta less often than stated (about 89% instead of 95% with n = 3 per group).
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
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        ci_method=_HEDGES_CI_METHOD if correct else _SMD_CI_METHOD,
        magnitude=magnitude_d(value),
        target=target,
    )


def _zero_safe_var(z: np.ndarray, constant: bool) -> float:
    return 0.0 if constant else float(np.var(z, ddof=1))


def cohen_d_independent(
    x: np.ndarray, y: np.ndarray, level: float = 0.95, hedges: bool = False, target: str | None = None
) -> EffectSize:
    """Cohen's d (or Hedges' g) for two independent groups, pooled SD.

    d = (mean(x) - mean(y)) / s_pooled, with the CI obtained by inverting the
    noncentral t distribution. ``hedges=True`` applies Hedges' exact correction to the
    point estimate (the interval stays the interval for the population value).

    The value is ``None`` (undefined) when the pooled SD is zero, i.e. when both groups
    are constant to within floating-point rounding (the same rule the t tests use).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    name = "Hedges' g" if hedges else "Cohen's d"
    df = n1 + n2 - 2
    if n1 < 2 or n2 < 2:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    zx, zy = rescale(x, y).arrays
    sp2 = ((n1 - 1) * _zero_safe_var(zx, is_constant(x)) + (n2 - 1) * _zero_safe_var(zy, is_constant(y))) / df
    if not sp2 > 0:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    d = float((zx.mean() - zy.mean()) / math.sqrt(sp2))
    scale = math.sqrt(1.0 / n1 + 1.0 / n2)
    return _smd_result(name, d, d / scale, df, scale, level, hedges, target)


def cohen_d_paired(
    x: np.ndarray, y: np.ndarray, level: float = 0.95, hedges: bool = False, target: str | None = None
) -> EffectSize:
    """Cohen's d_z for paired data: mean of the differences / SD of the differences.

    Undefined (``None``) when every pair has the same difference to within the rounding
    error of the original values (e.g. 100.1 - 100 and 1.1 - 1 are both "0.1").
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    name = "Hedges' g_z (paired)" if hedges else "Cohen's d_z (paired)"
    if x.shape != y.shape or x.size < 2:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    with np.errstate(over="ignore"):
        diff = x - y
    if not np.all(np.isfinite(diff)) or differences_constant(diff, difference_tolerance(x, y)):
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    rs = rescale(diff)
    return _one_sample_d(rs.arrays[0], rs.to_scaled_location(0.0), level, hedges, target, name, constant=False)


def _one_sample_d(z: np.ndarray, mu: float, level: float, hedges: bool, target: str | None, name: str,
                  constant: bool) -> EffectSize:
    """One-sample d from values of order 1 (already rescaled)."""
    n = z.size
    sd = 0.0 if constant else float(z.std(ddof=1))
    if not sd > 0:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    d = float((z.mean() - mu) / sd)
    scale = 1.0 / math.sqrt(n)
    return _smd_result(name, d, d / scale, n - 1, scale, level, hedges, target)


def cohen_d_one_sample(
    x: np.ndarray, popmean: float = 0.0, level: float = 0.95, hedges: bool = False,
    target: str | None = None, name: str | None = None,
) -> EffectSize:
    """Cohen's d for one sample against a hypothetical mean: (mean - mu) / SD.

    Undefined (``None``) when the values are constant to within rounding (SD = 0).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    name = name or ("Hedges' g" if hedges else "Cohen's d")
    if n < 2:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    rs = rescale(x)
    return _one_sample_d(rs.arrays[0], rs.to_scaled_location(popmean), level, hedges, target, name,
                         constant=is_constant(x))


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
    """Fisher z interval (kept for callers outside this module; not used for rank-biserial r)."""
    if not np.isfinite(r) or not np.isfinite(se_z):
        return (math.nan, math.nan)
    if abs(r) >= 1.0:
        return (r, r)
    z = math.atanh(r)
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    return (math.tanh(z - crit * se_z), math.tanh(z + crit * se_z))


RANK_BISERIAL_CI_METHOD = (
    "score interval for the Mann-Whitney probability P(X > Y) = (r + 1) / 2 (Newcombe 2006, method 5: "
    "Hanley-McNeil variance evaluated at the limits), widened when the distribution-free variance "
    "from placement values (DeLong et al. 1988) is larger; approximate, one-sided when the groups "
    "do not overlap"
)
RANK_BISERIAL_PAIRED_CI_METHOD = (
    "score interval from the variance of the signed-rank statistic (Hollander & Wolfe 1999) under a "
    "normal-shift working model evaluated at the limits (in the manner of Newcombe 2006), widened "
    "when the jackknife variance is larger; approximate, one-sided when all differences have the "
    "same sign"
)


def _hanley_mcneil_variance(theta: float, m: int, n: int) -> float:
    """Variance of the Mann-Whitney probability estimate at ``theta`` (Hanley & McNeil 1982,
    exponential-model form), with both group sizes replaced by their mean as recommended by
    Newcombe (2006, part 2, method 5)."""
    n_star = (m + n) / 2.0
    return theta * (1.0 - theta) * (2.0 * n_star - 1.0 - (3.0 * n_star - 3.0) / ((2.0 - theta) * (1.0 + theta))) / (m * n)


def _score_limits(theta_hat: float, crit: float, variance) -> tuple[float, float]:
    """Solve (theta_hat - theta)^2 = crit^2 * variance(theta) for theta in [0, 1].

    ``variance`` vanishes at 0 and 1, so the limit on the far side of a boundary estimate
    (theta_hat = 0 or 1) is that boundary and the other limit lies strictly inside.
    """
    eps = 1e-12

    def g(t: float) -> float:
        return (theta_hat - t) ** 2 - crit * crit * variance(t)

    lo = hi = math.nan
    try:
        if theta_hat <= 0.0:
            lo = 0.0
        else:
            upper = min(theta_hat, 1.0 - eps)
            lo = float(optimize.brentq(g, 0.0, upper, xtol=1e-12)) if g(0.0) * g(upper) <= 0 else 0.0
        if theta_hat >= 1.0:
            hi = 1.0
        else:
            lower = max(theta_hat, eps)
            hi = float(optimize.brentq(g, lower, 1.0, xtol=1e-12)) if g(lower) * g(1.0) <= 0 else 1.0
    except (ValueError, RuntimeError):  # pragma: no cover - brentq on a continuous bracketed function
        return (math.nan, math.nan)
    return lo, hi


def rank_biserial_independent(x: np.ndarray, y: np.ndarray, level: float = 0.95,
                              target: str | None = None) -> EffectSize:
    """Rank-biserial correlation for two independent groups (Cureton 1956; Kerby 2014).

    r = (favourable pairs - unfavourable pairs) / (n1 * n2) = 2*U_x / (n1*n2) - 1,
    where U_x counts pairs with x > y (ties count 1/2). Positive values mean ``x``
    tends to be larger. r = 2 theta - 1 with theta = U_x / (n1 n2), the estimate of the
    probability of superiority P(X > Y) + P(X = Y) / 2 (the area under the ROC curve).

    Confidence interval: the score-type interval that Newcombe (2006, Stat Med
    25:559-573, "method 5") recommends for theta, mapped to r. Its limits solve
    ``(theta_hat - theta)^2 = z^2 V(theta)`` with the Hanley-McNeil (1982) variance

        V(theta) = theta (1 - theta) [2 n* - 1 - (3 n* - 3) / ((2 - theta)(1 + theta))] / (n1 n2),
        n* = (n1 + n2) / 2,

    evaluated at the limit, not at the estimate. Unlike a Fisher-z interval built from the
    null-hypothesis SE, it widens correctly for strong effects, respects the range
    [-1, 1], and with complete separation (r = +/-1) gives a one-sided interval of
    positive width. Because V(theta) is derived from a working model, it is multiplied by
    max(1, V_placement / V(theta_hat)), where V_placement is the distribution-free
    variance from the placement values (Sen 1960; DeLong et al. 1988), so the interval is
    not too narrow when the data are more variable than the model assumes (e.g. a small
    group with a much larger spread than a big one).

    Limits: the interval is approximate. In simulations (normal, exponential, heavy-tailed
    and rounded data, n = 3 to 100 per group, no effect to near-complete separation) the
    95% interval covered the true value in 93-99% of samples; with only 3 or 4 values per
    group, or very unequal variances together with unequal group sizes, coverage can drop
    to about 90%.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 1 or n2 < 1:
        return EffectSize(name="rank-biserial r", value=None, ci_level=level, target=target)
    ranks = stats.rankdata(np.concatenate([x, y]))
    u_x = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    theta = u_x / (n1 * n2)
    r = 2.0 * theta - 1.0
    inflate = 1.0
    if 0.0 < theta < 1.0 and n1 > 1 and n2 > 1:
        # placement values: share of the other group lying below each observation
        v10 = (ranks[:n1] - stats.rankdata(x)) / n2
        v01 = 1.0 - (ranks[n1:] - stats.rankdata(y)) / n1
        v_placement = float(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n2)
        v_model = _hanley_mcneil_variance(theta, n1, n2)
        if v_model > 0 and math.isfinite(v_placement):
            inflate = max(1.0, v_placement / v_model)
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    lo, hi = _score_limits(theta, crit, lambda t: inflate * _hanley_mcneil_variance(t, n1, n2))
    return EffectSize(name="rank-biserial r", value=r, ci_low=2.0 * lo - 1.0, ci_high=2.0 * hi - 1.0,
                      ci_level=level, ci_method=RANK_BISERIAL_CI_METHOD, magnitude=magnitude_r(r),
                      target=target)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_DELTA_MAX = 8.0


def _signed_rank_model(delta: float, n: int) -> tuple[float, float]:
    """Mean and variance of W+ / (n (n + 1) / 2) for n differences D ~ N(delta, 1).

    W+ is the number of positive Walsh averages (D_i + D_j) / 2, i <= j, so with
    p1 = P(D > 0), p2 = P(D1 + D2 > 0), p3 = P(D1 > 0, D1 + D2 > 0) and
    p4 = P(D1 + D2 > 0, D1 + D3 > 0) (Hollander & Wolfe 1999, section 3.1):

        E[W+]   = n p1 + C(n, 2) p2
        Var[W+] = n p1 (1 - p1) + C(n, 2) p2 (1 - p2) + 2 n (n - 1) (p3 - p1 p2)
                  + n (n - 1)(n - 2) (p4 - p2^2)

    The probabilities are evaluated in complementary form at abs(delta) (so nothing is
    lost to cancellation when they are close to 1) and mirrored for negative delta.
    """
    a = abs(delta)
    q1 = float(special.ndtr(-a))                    # 1 - p1
    q2 = float(special.ndtr(-math.sqrt(2.0) * a))   # 1 - p2
    z = 9.0 * _GL_X
    w = 9.0 * _GL_W * np.exp(-0.5 * z * z) / _SQRT_2PI
    q = special.ndtr(-(z + 2.0 * a))                # P(D + D' <= 0 | D = z + a)
    var_pair = max(float(w @ (q * q)) - q2 * q2, 0.0)   # p4 - p2^2
    hi = min(-a, 9.0)
    if hi > -9.0:
        zz = 0.5 * (hi + 9.0) * _GL_X + 0.5 * (hi - 9.0)
        ww = 0.5 * (hi + 9.0) * _GL_W * np.exp(-0.5 * zz * zz) / _SQRT_2PI
        cov = float(ww @ special.ndtr(-(zz + 2.0 * a))) - q1 * q2   # p3 - p1 p2
    else:
        cov = 0.0
    m = n * (n + 1) / 2.0
    var_w = (n * q1 * (1.0 - q1) + n * (n - 1) / 2.0 * q2 * (1.0 - q2) + 2.0 * n * (n - 1) * cov
             + n * (n - 1) * (n - 2.0) * var_pair)
    below = (n * q1 + n * (n - 1) / 2.0 * q2) / m
    theta = 1.0 - below if delta >= 0 else below
    return theta, max(var_w, 0.0) / (m * m)


def _signed_rank_jackknife_variance(d: np.ndarray) -> float:
    """Delete-one jackknife variance of W+ / (W+ + W-) for non-zero differences ``d``.

    W+ equals the number of pairs i <= j with d_i + d_j > 0 (a pair with d_i + d_j = 0
    counts 1/2), also with mid-ranks for ties, so leaving out observation i removes
    row_i = #{j: d_j > -d_i} + #{j: d_j = -d_i} / 2 from it.
    """
    n = d.size
    ordered = np.sort(d)
    right = np.searchsorted(ordered, -d, side="right")
    left = np.searchsorted(ordered, -d, side="left")
    row = (n - right) + 0.5 * (right - left)
    w_plus = 0.5 * (float(row.sum()) + float(np.count_nonzero(d > 0)))
    theta_i = (w_plus - row) / ((n - 1) * n / 2.0)
    return float((n - 1) / n * np.sum((theta_i - theta_i.mean()) ** 2))


def _signed_rank_score_limits(theta_hat: float, n: int, crit: float, inflate: float) -> tuple[float, float]:
    def f(delta: float, sign: float) -> float:
        theta, var = _signed_rank_model(delta, n)
        return (theta_hat - theta) - sign * crit * math.sqrt(inflate * var)

    def solve(sign: float) -> float:
        f_lo, f_hi = f(-_DELTA_MAX, sign), f(_DELTA_MAX, sign)
        if f_lo * f_hi > 0:  # no crossing inside the range: the limit is at the boundary
            return 0.0 if f_lo < 0 else 1.0
        root = float(optimize.brentq(f, -_DELTA_MAX, _DELTA_MAX, args=(sign,), xtol=1e-10))
        return _signed_rank_model(root, n)[0]

    try:
        lo = 0.0 if theta_hat <= 0.0 else solve(1.0)
        hi = 1.0 if theta_hat >= 1.0 else solve(-1.0)
    except (ValueError, RuntimeError):  # pragma: no cover
        return (math.nan, math.nan)
    return min(lo, theta_hat), max(hi, theta_hat)


def rank_biserial_paired(diff: np.ndarray, level: float = 0.95, target: str | None = None) -> EffectSize:
    """Matched-pairs rank-biserial correlation (Kerby 2014).

    r = (W+ - W-) / (W+ + W-) using ranks of the non-zero absolute differences (mid-ranks
    for ties). With theta = W+ / (W+ + W-), r = 2 theta - 1, and W+ is the number of
    positive Walsh averages (d_i + d_j) / 2, i <= j.

    Confidence interval: a score-type interval in the manner Newcombe (2006) recommends
    for the Mann-Whitney measure. Its limits are the values theta_n(delta) for which
    ``|theta_hat - theta_n(delta)| = z sqrt(V_n(delta))``, where theta_n and V_n are the
    exact mean and variance of the statistic for n differences (Hollander & Wolfe 1999)
    under a normal-shift working model D ~ N(delta, 1) (see ``_signed_rank_model``). The
    variance is therefore evaluated at each limit, the interval respects [-1, 1], and
    when all differences have the same sign (r = +/-1) it is one-sided with positive
    width. To protect against data that are more variable than the working model
    (heavy tails), V_n is multiplied by max(1, V_jackknife / V_n(delta_hat)) with the
    delete-one jackknife variance of the estimate.

    What is covered: the population value of r *for samples of this size*,
    2 [n P(D > 0) + C(n, 2) P(D1 + D2 > 0)] / [n (n + 1) / 2] - 1, which tends to
    2 P(D1 + D2 > 0) - 1 as n grows.

    Limits: approximate. In simulations (normal, t3, Cauchy, lognormal, uniform, bimodal
    and rounded differences; n = 5 to 100; no effect to near-complete separation) the 95%
    interval covered the true value in 93-99% of samples, dropping to about 90% for
    heavy-tailed differences with very strong effects. A Fisher-z interval with the
    null-hypothesis SE (used by some packages) covers only 20-75% for strong effects.
    """
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    n = int(d.size)
    name = "matched-pairs rank-biserial r"
    if n < 1:
        return EffectSize(name=name, value=None, ci_level=level, target=target)
    ranks = stats.rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = w_plus + w_minus
    r = (w_plus - w_minus) / total
    theta = w_plus / total
    inflate = 1.0
    if 0.0 < theta < 1.0 and n >= 3:
        try:
            delta_hat = float(optimize.brentq(lambda v: _signed_rank_model(v, n)[0] - theta,
                                              -_DELTA_MAX, _DELTA_MAX, xtol=1e-10))
            v_model = _signed_rank_model(delta_hat, n)[1]
            v_jack = _signed_rank_jackknife_variance(d)
            if v_model > 0 and math.isfinite(v_jack):
                inflate = max(1.0, v_jack / v_model)
        except (ValueError, RuntimeError):
            inflate = 1.0
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    lo, hi = _signed_rank_score_limits(theta, n, crit, inflate)
    return EffectSize(name=name, value=r, ci_low=2.0 * lo - 1.0, ci_high=2.0 * hi - 1.0, ci_level=level,
                      ci_method=RANK_BISERIAL_PAIRED_CI_METHOD, magnitude=magnitude_r(r), target=target)


def epsilon_squared_kw(h: float, n_total: int, k: int, target: str | None = None) -> EffectSize:
    """Rank epsilon-squared for Kruskal-Wallis: H / (N - 1) (Kelley 1935; Tomczak & Tomczak 2014)."""
    value = float(h / (n_total - 1)) if n_total > 1 else math.nan
    return EffectSize(name="epsilon squared (rank)", value=value, magnitude=magnitude_eta2(value), target=target)


def kendall_w(chi2: float, n_subjects: int, k: int, target: str | None = None) -> EffectSize:
    """Kendall's W for Friedman's test: chi² / (n (k - 1))."""
    value = float(chi2 / (n_subjects * (k - 1))) if n_subjects > 0 and k > 1 else math.nan
    return EffectSize(name="Kendall's W", value=value, magnitude=magnitude_r(value), target=target)
