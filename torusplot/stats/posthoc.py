"""Multiple-comparison (post-hoc) procedures, written for TorusPlot from the primary papers.

Every function takes ``groups``: a mapping ``{group name: 1-D array of finite values}``
(already cleaned) and returns a :class:`PosthocResult`, which unpacks as
``(comparisons, family description, correction label)`` and also carries ``warnings``,
``citations`` and a ``method`` description.

In every :class:`~torusplot.stats.result.Comparison`:

* ``estimate`` is mean(A) - mean(B) (difference of mean ranks for rank procedures),
* ``p_value`` is the unadjusted two-group P value where one is defined,
* ``p_adjusted`` is the multiplicity-adjusted P value (for Tukey, Dunnett and
  Games-Howell this is the procedure's own P value),
* ``ci_low`` / ``ci_high`` are simultaneous confidence limits when the procedure defines
  them (see each function).

References
----------
* Tukey JW (1949) Comparing individual means in the analysis of variance.
  *Biometrics* 5:99-114.
* Kramer CY (1956) Extension of multiple range tests to group means with unequal
  numbers of replications. *Biometrics* 12:307-310.
* Dunnett CW (1955) A multiple comparison procedure for comparing several treatments
  with a control. *J Am Stat Assoc* 50:1096-1121.
* Dunnett CW (1964) New tables for multiple comparisons with a control.
  *Biometrics* 20:482-491.
* Games PA, Howell JF (1976) Pairwise multiple comparison procedures with unequal n's
  and/or variances: a Monte Carlo study. *J Educ Stat* 1:113-125.
* Welch BL (1947) The generalization of 'Student's' problem when several different
  population variances are involved. *Biometrika* 34:28-35.
* Satterthwaite FE (1946) An approximate distribution of estimates of variance
  components. *Biometrics Bull* 2:110-114.
* Fisher RA (1935) *The Design of Experiments* (least significant difference).
* Dunn OJ (1961) Multiple comparisons among means. *J Am Stat Assoc* 56:52-64.
* Dunn OJ (1964) Multiple comparisons using rank sums. *Technometrics* 6:241-252.
* Siegel S, Castellan NJ (1988) *Nonparametric Statistics for the Behavioral
  Sciences*, 2nd ed. (rank-sum comparisons after the Friedman test).
* Demsar J (2006) Statistical comparisons of classifiers over multiple data sets.
  *J Mach Learn Res* 7:1-30.
* Wilcoxon F (1945) Individual comparisons by ranking methods. *Biometrics Bull* 1:80-83.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy import optimize, special, stats

from . import effect_sizes as es
from ._utils import (
    _TOO_LARGE,
    StatsError,
    check_alpha,
    check_alternative,
    difference_tolerance,
    differences_constant,
    is_constant,
    pow2_back,
    pow2_scaled,
    rescale,
    snap_differences,
)
from .multiplicity import METHOD_LABELS, adjust_pvalues
from .result import Comparison, EffectSize
from .two_group import EXACT_TIES_MAX_N, _signed_rank_core

__all__ = [
    "POSTHOC_METHODS",
    "PosthocResult",
    "dunn",
    "dunnett",
    "friedman_posthoc",
    "games_howell",
    "pairwise_t",
    "run_posthoc",
    "tukey_hsd",
]

CITE_TUKEY = "Tukey JW (1949) Biometrics 5:99-114"
CITE_KRAMER = "Kramer CY (1956) Biometrics 12:307-310"
CITE_DUNNETT = "Dunnett CW (1955) J Am Stat Assoc 50:1096-1121"
CITE_GAMES_HOWELL = "Games PA, Howell JF (1976) J Educ Stat 1:113-125"
CITE_DUNN = "Dunn OJ (1964) Technometrics 6:241-252"
CITE_SIEGEL = (
    "Siegel S, Castellan NJ (1988) Nonparametric Statistics for the Behavioral Sciences, 2nd ed."
)
CITE_WILCOXON = "Wilcoxon F (1945) Biometrics Bull 1:80-83"
CITE_FISHER = "Fisher RA (1935) The Design of Experiments"
_ADJUST_CITATIONS = {
    "none": CITE_FISHER,
    "bonferroni": "Dunn OJ (1961) J Am Stat Assoc 56:52-64",
    "sidak": "Sidak Z (1967) J Am Stat Assoc 62:626-633",
    "holm": "Holm S (1979) Scand J Stat 6:65-70",
    "holm-sidak": "Holm S (1979) Scand J Stat 6:65-70",
    "fdr_bh": "Benjamini Y, Hochberg Y (1995) J R Stat Soc B 57:289-300",
}

#: Method names understood by :func:`run_posthoc`.
POSTHOC_METHODS = (
    "tukey", "dunnett", "games-howell", "bonferroni", "sidak", "holm", "holm-sidak",
    "fdr_bh", "lsd", "none",
)

_SQRT2 = math.sqrt(2.0)


class PosthocResult(tuple):
    """Outcome of one post-hoc procedure.

    Behaves as the 3-tuple ``(comparisons, family, correction)`` (so it can be unpacked
    directly) and additionally exposes ``warnings`` (things the user must know),
    ``citations`` (primary references) and ``method`` (what was computed, in words).
    """

    warnings: list[str]
    citations: list[str]
    method: str

    def __new__(
        cls,
        comparisons: list[Comparison],
        family: str,
        correction: str,
        *,
        warnings: Sequence[str] = (),
        citations: Sequence[str] = (),
        method: str = "",
    ) -> PosthocResult:
        self = super().__new__(cls, (comparisons, family, correction))
        self.warnings = list(dict.fromkeys(warnings))
        self.citations = list(dict.fromkeys(citations))
        self.method = method
        return self

    @property
    def comparisons(self) -> list[Comparison]:
        return self[0]

    @property
    def family(self) -> str:
        return self[1]

    @property
    def correction(self) -> str:
        return self[2]


# ------------------------------------------------------------------------- helpers


def _near_constant(arr: np.ndarray) -> bool:
    """True when all values agree to within rounding error.

    This is the shared :func:`~torusplot.stats._utils.is_constant` rule (spread within a
    few units in the last place of the largest magnitude, no absolute floor), so data
    such as 1e12 + {0.2, ..., 0.6} or 1e-80 * {1, 2, 3} are *not* constant. Paired
    differences are judged against the rounding error of the original values instead
    (see :func:`pairwise_t`).
    """
    return is_constant(arr)


def _mean(arr: np.ndarray) -> float:
    """Mean; computed on values scaled by a power of two so the sum cannot overflow."""
    if _near_constant(arr):
        return float(arr[0])
    scaled, e = pow2_scaled(arr)
    return pow2_back(float(np.mean(scaled)), e)


def _variance(arr: np.ndarray) -> float:
    """Sample variance; exactly 0 for a constant group; NaN when n < 2.

    Computed on power-of-two scaled values (exact), so squares neither overflow nor
    underflow; a StatsError is raised only when the variance itself exceeds the largest
    representable number."""
    if arr.size < 2:
        return math.nan
    if _near_constant(arr):
        return 0.0
    scaled, e = pow2_scaled(arr - _mean(arr) if math.isfinite(float(np.ptp(arr))) else arr)
    if not np.all(np.isfinite(scaled)):
        raise StatsError(_TOO_LARGE)
    return pow2_back(float(np.var(scaled, ddof=1)), e, 2)


def _no_difference(diff: float, a: np.ndarray, b: np.ndarray) -> bool:
    """True when two means differ by no more than rounding error."""
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return abs(diff) <= 8.0 * np.finfo(float).eps * scale


class _Summary:
    """Names, arrays, n, means and variances of the groups."""

    def __init__(self, groups: Mapping[str, Any], what: str, min_groups: int = 2) -> None:
        if not isinstance(groups, Mapping):
            raise TypeError("groups must be a mapping {name: array of values}.")
        self.names = [str(name) for name in groups]
        self.arrays = [np.asarray(v, dtype=float).ravel() for v in groups.values()]
        if len(self.names) < min_groups:
            raise StatsError(f"{what} needs at least {min_groups} groups.")
        for name, arr in zip(self.names, self.arrays):
            if arr.size == 0:
                raise StatsError(
                    f"Group '{name}' has no values; remove it from the analysis or add data."
                )
            if not np.all(np.isfinite(arr)):
                raise StatsError(f"Group '{name}' contains missing or infinite values.")
        self.k = len(self.names)
        self.n = np.array([a.size for a in self.arrays], dtype=float)
        self.n_total = int(self.n.sum())
        self.means = np.array([_mean(a) for a in self.arrays])
        self.variances = np.array([_variance(a) for a in self.arrays])

    def index(self, name: str | None, role: str = "control") -> int:
        if name is None or str(name) not in self.names:
            raise StatsError(
                f"The {role} group '{name}' is not one of the groups ({', '.join(self.names)})."
            )
        return self.names.index(str(name))

    def pairs(self, control: str | None) -> list[tuple[int, int]]:
        """(A, B) index pairs: all pairs, or every other group (A) against the control (B)."""
        if control is None:
            return list(combinations(range(self.k), 2))
        c = self.index(control)
        return [(i, c) for i in range(self.k) if i != c]

    def family(self, control: str | None, m: int) -> str:
        if control is None:
            noun = "comparison" if m == 1 else "comparisons"
            return f"all {m} pairwise {noun} among {self.k} groups"
        noun = "comparison" if m == 1 else "comparisons"
        return f"{m} {noun} of each other group against the control group '{control}'"


def _pooled_error(
    s: _Summary, mse: float | None, df_error: float | None, what: str
) -> tuple[float, float]:
    """Pooled within-group variance (MSE) and its df, or the supplied override."""
    if (mse is None) != (df_error is None):
        raise ValueError("mse and df_error must be given together.")
    if mse is not None and df_error is not None:
        if not (math.isfinite(mse) and mse >= 0 and math.isfinite(df_error) and df_error > 0):
            raise StatsError("The supplied error mean square / degrees of freedom are not valid.")
        return float(mse), float(df_error)
    df = float(s.n_total - s.k)
    if df < 1:
        raise StatsError(
            f"{what} needs replicates: at least one group must have 2 or more values so the "
            "within-group variability can be estimated."
        )
    ss = sum((a.size - 1) * v for a, v in zip(s.arrays, s.variances) if a.size > 1)
    return float(ss / df), df


def _srange_sf(q: float, k: int, df: float) -> float:
    """Upper tail of the studentized range distribution, clipped to [0, 1]."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        p = float(stats.studentized_range.sf(q, k, df))
    if math.isnan(p):
        return math.nan
    return min(max(p, 0.0), 1.0)


def _srange_ppf(prob: float, k: int, df: float) -> float:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return float(stats.studentized_range.ppf(prob, k, df))


def _t_two_sided(t: float, df: float) -> float:
    return float(2.0 * stats.t.sf(abs(t), df))


def _t_pvalue(t: float, df: float, alternative: str) -> float:
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    return _t_two_sided(t, df)


def _hedges_g(a: np.ndarray, b: np.ndarray, target: str) -> EffectSize:
    if _near_constant(a) and _near_constant(b):
        return EffectSize(name="Hedges' g", value=None, target=target)
    return es.cohen_d_independent(a, b, hedges=True, target=target)


_ZERO_SE_WARNING = (
    "At least one comparison has a standard error of zero (no variability within the groups "
    "involved), so its statistic is infinite (P reported as 0) when the means differ and "
    "undefined when they are equal. Check whether the data were rounded or duplicated."
)


def _normalise_adjust(adjust: str | None) -> str:
    key = "none" if adjust is None else str(adjust).strip().lower().replace("_", "-")
    key = {"lsd": "none", "fdr-bh": "fdr_bh", "bh": "fdr_bh", "fdr": "fdr_bh",
           "holmsidak": "holm-sidak", "šidák": "sidak", "holm-šidák": "holm-sidak"}.get(key, key)
    if key not in METHOD_LABELS:
        raise StatsError(
            f"Unknown P value adjustment '{adjust}'. Choose one of: bonferroni, sidak, holm, "
            "holm-sidak, fdr_bh or none."
        )
    return key


def _adjust(pvalues: Sequence[float | None], key: str) -> list[float | None]:
    """Adjusted P values (None stays None).

    The shared ``adjust_pvalues`` takes log(1 - p), which makes numpy emit a
    divide-by-zero warning for the Sidak-type methods when a raw P value is exactly 1
    (the adjusted value, 1, is still correct), so the call is guarded here.
    """
    raw = [math.nan if p is None else p for p in pvalues]
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = adjust_pvalues(raw, key)
    return [None if math.isnan(a) else float(a) for a in adjusted]


def _per_comparison_alpha(adjust: str, alpha: float, m: int) -> tuple[float, str | None]:
    """Alpha used for each CI, and a note when the CIs are not simultaneous."""
    if m <= 1 or adjust == "none":
        return alpha, None
    if adjust == "bonferroni":
        return alpha / m, None
    if adjust == "sidak":
        return -math.expm1(math.log1p(-alpha) / m), None
    return alpha, (
        f"Confidence intervals are unadjusted {100 * (1 - alpha):g}% intervals: the "
        f"{METHOD_LABELS[adjust]} adjustment applies to P values only and has no matching "
        "simultaneous interval."
    )


# ------------------------------------------------------------------- Tukey / Kramer


def tukey_hsd(
    groups: Mapping[str, Any],
    *,
    alpha: float = 0.05,
    mse: float | None = None,
    df_error: float | None = None,
) -> PosthocResult:
    """Tukey's honestly significant difference test (Tukey 1949), with the Tukey-Kramer
    extension to unequal group sizes (Kramer 1956).

    For each pair, ``q = |mean_i - mean_j| / sqrt(MSE / 2 * (1/n_i + 1/n_j))`` is referred
    to the studentized range distribution with ``k`` groups and the error df. The
    simultaneous ``1 - alpha`` intervals are ``diff +/- q_crit / sqrt(2) * SE``.

    ``mse`` / ``df_error`` let a caller supply the error mean square of a larger model
    (for example a factorial ANOVA); by default the one-way pooled variance is used.
    The reported ``statistic`` is q; ``p_value`` is the unadjusted pooled-variance t test.
    """
    alpha = check_alpha(alpha)
    s = _Summary(groups, "Tukey's test")
    mse_value, df = _pooled_error(s, mse, df_error, "Tukey's test")
    pairs = s.pairs(None)
    q_crit = _srange_ppf(1.0 - alpha, s.k, df) if mse_value > 0 else math.nan
    notes: list[str] = []
    comparisons: list[Comparison] = []
    for i, j in pairs:
        a, b = s.arrays[i], s.arrays[j]
        diff = float(s.means[i] - s.means[j])
        se = math.sqrt(mse_value * (1.0 / s.n[i] + 1.0 / s.n[j]))
        target = f"{s.names[i]} vs {s.names[j]}"
        if se > 0:
            q = abs(diff) / se * _SQRT2
            p_adj: float | None = _srange_sf(q, s.k, df)
            p_raw: float | None = _t_two_sided(diff / se, df)
            half = q_crit / _SQRT2 * se
            lo, hi = diff - half, diff + half
        else:
            notes.append(_ZERO_SE_WARNING)
            if _no_difference(diff, a, b):
                q, p_adj, p_raw, lo, hi = None, None, None, None, None
            else:
                q, p_adj, p_raw, lo, hi = math.inf, 0.0, 0.0, diff, diff
        comparisons.append(Comparison(
            id="", group_a=s.names[i], group_b=s.names[j], estimate=diff, se=se,
            ci_low=lo, ci_high=hi, statistic_name="q", statistic=q, df=df,
            p_value=p_raw, p_adjusted=p_adj, effect_size=_hedges_g(a, b, target),
        ))
    unequal = len(set(s.n.tolist())) > 1
    return PosthocResult(
        comparisons, s.family(None, len(pairs)), "Tukey-Kramer" if unequal else "Tukey HSD",
        warnings=notes,
        citations=[CITE_TUKEY, CITE_KRAMER] if unequal else [CITE_TUKEY],
        method="Tukey's HSD test (studentized range, pooled error variance)"
        + (" with the Tukey-Kramer adjustment for unequal group sizes" if unequal else ""),
    )


# ------------------------------------------------------------------------- Dunnett

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _composite_nodes(a: float, b: float, panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of a composite 16-point Gauss-Legendre rule on [a, b]."""
    edges = np.linspace(a, b, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    x = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    w = (half[:, None] * _GL_W[None, :]).ravel()
    return x, w


def _dunnett_cdf(c: float, n_treat: np.ndarray, n_control: float, df: float, two_sided: bool) -> float:
    """P(max_i |T_i| <= c) (two-sided) or P(max_i T_i <= c) (one-sided) under the null.

    Dunnett (1955): with a common control, ``T_i = (Z_i - Z_0) / (S sqrt(1/n_i + 1/n_0))``
    has the product correlation structure ``rho_ij = b_i b_j`` with
    ``b_i = sqrt(n_i / (n_i + n_0))``. Conditioning on the shared control term ``z`` and
    on ``u = S / sigma`` makes the T_i independent, which leaves the double integral

        integral f_u(u) integral phi(z) prod_i [Phi((b_i z + c u) / s_i)
                                                - Phi((b_i z - c u) / s_i)] dz du

    with ``s_i = sqrt(1 - b_i^2)`` and ``u ~ chi_df / sqrt(df)`` (the second Phi is
    dropped for the one-sided case). It is evaluated with deterministic composite
    Gauss-Legendre quadrature (absolute error about 1e-11 or better).
    """
    if np.size(n_treat) == 1:
        # One treatment vs control is an ordinary t test: use the exact t distribution
        # instead of quadrature (identical in theory, exact to machine precision).
        if two_sided:
            return 0.0 if c <= 0 else float(1.0 - 2.0 * stats.t.sf(c, df))
        return float(stats.t.cdf(c, df))
    if c <= 0:
        return 0.0 if two_sided else _dunnett_cdf_raw(c, n_treat, n_control, df, False)
    return _dunnett_cdf_raw(c, n_treat, n_control, df, two_sided)


def _dunnett_cdf_raw(c: float, n_treat: np.ndarray, n_control: float, df: float, two_sided: bool) -> float:
    b = np.sqrt(n_treat / (n_treat + n_control))
    sd = np.sqrt(1.0 - b * b)
    # panel width at most twice the narrowest transition width (s_i / b_i) of the integrand
    z_panels = int(min(max(math.ceil(17.0 / (2.0 * float(np.min(sd / b)))), 24), 800))
    z, zw = _composite_nodes(-8.5, 8.5, z_panels)
    zw = zw * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    root_df = math.sqrt(df)
    lo = float(stats.chi.ppf(1e-15, df)) / root_df
    hi = float(stats.chi.isf(1e-15, df)) / root_df
    u, uw = _composite_nodes(lo, hi, 12)
    uw = uw * stats.chi.pdf(u * root_df, df) * root_df
    cu = c * u[:, None]
    prod = np.ones((u.size, z.size))
    for bi, si in zip(b, sd):
        term = special.ndtr((bi * z[None, :] + cu) / si)
        if two_sided:
            term = term - special.ndtr((bi * z[None, :] - cu) / si)
        prod *= term
    return float(min(max(uw @ (prod @ zw), 0.0), 1.0))


def _dunnett_critical(alpha: float, n_treat: np.ndarray, n_control: float, df: float,
                      two_sided: bool) -> float:
    """Critical value c with P(max <= c) = 1 - alpha (between the single-comparison
    t quantile and the Bonferroni t quantile)."""
    m = n_treat.size
    tail = alpha / 2.0 if two_sided else alpha
    lower = float(stats.t.isf(tail, df))
    if m == 1:
        return lower
    upper = float(stats.t.isf(tail / m, df))

    def f(c: float) -> float:
        return _dunnett_cdf(c, n_treat, n_control, df, two_sided) - (1.0 - alpha)

    f_lo, f_hi = f(lower), f(upper)
    if f_lo >= 0:
        return lower
    if f_hi <= 0:
        return upper
    return float(optimize.brentq(f, lower, upper, xtol=1e-10, rtol=1e-12, maxiter=200))


def dunnett(
    groups: Mapping[str, Any],
    control: str,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    mse: float | None = None,
    df_error: float | None = None,
) -> PosthocResult:
    """Dunnett's (1955) many-to-one comparisons of every group with a control.

    ``t_i = (mean_i - mean_control) / sqrt(MSE (1/n_i + 1/n_control))`` with the pooled
    error variance of *all* groups. The adjusted P value of a comparison is the
    probability, under the null, that the largest of the ``k - 1`` statistics exceeds
    the observed one (single-step procedure); the simultaneous confidence limits use the
    matching critical value. The joint distribution (a multivariate t with a product
    correlation structure) is integrated numerically by deterministic quadrature, so
    results are reproducible and need no random seed.

    ``alternative='greater'`` tests mean(group) > mean(control); the intervals are then
    one-sided (the other limit is infinite).
    """
    alpha = check_alpha(alpha)
    alternative = check_alternative(alternative)
    s = _Summary(groups, "Dunnett's test")
    c = s.index(control)
    mse_value, df = _pooled_error(s, mse, df_error, "Dunnett's test")
    pairs = s.pairs(str(control))
    treat = [i for i, _ in pairs]
    n_treat = s.n[treat]
    n_control = float(s.n[c])
    two_sided = alternative == "two-sided"
    crit = _dunnett_critical(alpha, n_treat, n_control, df, two_sided) if mse_value > 0 else math.nan
    notes: list[str] = []
    comparisons: list[Comparison] = []
    for i in treat:
        a, b = s.arrays[i], s.arrays[c]
        diff = float(s.means[i] - s.means[c])
        se = math.sqrt(mse_value * (1.0 / s.n[i] + 1.0 / n_control))
        target = f"{s.names[i]} vs {s.names[c]}"
        lo: float | None
        hi: float | None
        if se > 0:
            t: float | None = diff / se
            if two_sided:
                p_adj: float | None = 1.0 - _dunnett_cdf(abs(t), n_treat, n_control, df, True)
                lo, hi = diff - crit * se, diff + crit * se
            elif alternative == "greater":
                p_adj = 1.0 - _dunnett_cdf(t, n_treat, n_control, df, False)
                lo, hi = diff - crit * se, math.inf
            else:
                p_adj = 1.0 - _dunnett_cdf(-t, n_treat, n_control, df, False)
                lo, hi = -math.inf, diff + crit * se
            p_adj = min(max(p_adj, 0.0), 1.0)
            p_raw: float | None = _t_pvalue(t, df, alternative)
        else:
            notes.append(_ZERO_SE_WARNING)
            if _no_difference(diff, a, b):
                t, p_adj, p_raw, lo, hi = None, None, None, None, None
            else:
                t = math.copysign(math.inf, diff)
                favours = two_sided or (diff > 0) == (alternative == "greater")
                p_adj = p_raw = 0.0 if favours else 1.0
                lo, hi = diff, diff
        comparisons.append(Comparison(
            id="", group_a=s.names[i], group_b=s.names[c], estimate=diff, se=se,
            ci_low=lo, ci_high=hi, statistic_name="t", statistic=t, df=df,
            p_value=p_raw, p_adjusted=p_adj, effect_size=_hedges_g(a, b, target),
        ))
    sided = "" if two_sided else f", one-sided ({alternative})"
    return PosthocResult(
        comparisons, s.family(str(control), len(pairs)), "Dunnett",
        warnings=notes, citations=[CITE_DUNNETT],
        method=f"Dunnett's many-to-one test against '{control}' (pooled error variance{sided}; "
               "multivariate t probabilities by numerical integration)",
    )


# -------------------------------------------------------------------- Games-Howell


def _welch_pair(vi: float, ni: float, vj: float, nj: float) -> tuple[float, float]:
    """Welch standard error of a difference and its Welch-Satterthwaite df."""
    a, b = vi / ni, vj / nj
    se = math.sqrt(a + b)
    denom = a * a / (ni - 1.0) + b * b / (nj - 1.0)
    df = (a + b) ** 2 / denom if denom > 0 else math.nan
    return se, df


def games_howell(groups: Mapping[str, Any], *, alpha: float = 0.05) -> PosthocResult:
    """Games-Howell (1976) pairwise comparisons, which do not assume equal variances.

    For each pair, ``SE = sqrt(s_i^2/n_i + s_j^2/n_j)`` with Welch-Satterthwaite df, and
    ``q = sqrt(2) |diff| / SE`` is referred to the studentized range distribution with
    ``k`` groups and that df. Simultaneous intervals: ``diff +/- q_crit(k, df)/sqrt(2) * SE``.
    Every group needs at least 2 values. ``p_value`` is the unadjusted Welch t test.
    """
    alpha = check_alpha(alpha)
    s = _Summary(groups, "The Games-Howell test")
    for name, arr in zip(s.names, s.arrays):
        if arr.size < 2:
            raise StatsError(
                f"Group '{name}' has only {arr.size} value; the Games-Howell test needs at "
                "least 2 values in every group to estimate each group's own variance."
            )
    pairs = s.pairs(None)
    notes: list[str] = []
    comparisons: list[Comparison] = []
    for i, j in pairs:
        a, b = s.arrays[i], s.arrays[j]
        diff = float(s.means[i] - s.means[j])
        se, df = _welch_pair(s.variances[i], s.n[i], s.variances[j], s.n[j])
        target = f"{s.names[i]} vs {s.names[j]}"
        if se > 0:
            q = abs(diff) / se * _SQRT2
            p_adj: float | None = _srange_sf(q, s.k, df)
            p_raw: float | None = _t_two_sided(diff / se, df)
            half = _srange_ppf(1.0 - alpha, s.k, df) / _SQRT2 * se
            lo, hi = diff - half, diff + half
            df_out: float | None = df
        else:
            notes.append(_ZERO_SE_WARNING)
            df_out = None
            if _no_difference(diff, a, b):
                q, p_adj, p_raw, lo, hi = None, None, None, None, None
            else:
                q, p_adj, p_raw, lo, hi = math.inf, 0.0, 0.0, diff, diff
        comparisons.append(Comparison(
            id="", group_a=s.names[i], group_b=s.names[j], estimate=diff, se=se,
            ci_low=lo, ci_high=hi, statistic_name="q", statistic=q, df=df_out,
            p_value=p_raw, p_adjusted=p_adj, effect_size=_hedges_g(a, b, target),
        ))
    return PosthocResult(
        comparisons, s.family(None, len(pairs)), "Games-Howell",
        warnings=notes, citations=[CITE_GAMES_HOWELL],
        method="Games-Howell test (studentized range with separate variances and "
               "Welch-Satterthwaite degrees of freedom)",
    )


# --------------------------------------------------------------- pairwise t tests


def pairwise_t(
    groups: Mapping[str, Any],
    *,
    pooled: bool = True,
    paired: bool = False,
    adjust: str = "holm",
    control: str | None = None,
    alpha: float = 0.05,
    mse: float | None = None,
    df_error: float | None = None,
) -> PosthocResult:
    """Pairwise t tests with a P value adjustment.

    * ``pooled=True`` (the usual choice after an ANOVA): every comparison uses the error
      variance pooled over *all* groups and its df (or the supplied ``mse``/``df_error``).
      With ``adjust='none'`` this is Fisher's least significant difference.
    * ``pooled=False``: each pair gets its own Welch t test (separate variances).
    * ``paired=True``: paired t tests on within-row differences (groups must have equal
      length, values aligned by position); ``pooled`` is then ignored.
    * ``control``: compare every other group with this one instead of all pairs.

    ``adjust`` is passed to :func:`torusplot.stats.multiplicity.adjust_pvalues`.
    Confidence intervals are simultaneous for ``bonferroni`` and ``sidak`` (per-comparison
    alpha adjusted to match); for the step-wise and FDR methods, which have no matching
    interval, they are unadjusted and a warning says so.
    """
    alpha = check_alpha(alpha)
    key = _normalise_adjust(adjust)
    s = _Summary(groups, "Pairwise t tests")
    pairs = s.pairs(None if control is None else str(control))
    m = len(pairs)
    notes: list[str] = []
    if paired:
        if len({a.size for a in s.arrays}) != 1:
            raise StatsError(
                "Paired comparisons need the same number of values in every group "
                "(one value per subject in each group)."
            )
        if s.arrays[0].size < 2:
            raise StatsError("Paired comparisons need at least 2 subjects (rows).")
        mse_value, df_pool = math.nan, math.nan
    elif pooled:
        mse_value, df_pool = _pooled_error(s, mse, df_error, "Pairwise t tests with a pooled SD")
    else:
        for name, arr in zip(s.names, s.arrays):
            if arr.size < 2:
                raise StatsError(
                    f"Group '{name}' has only {arr.size} value; Welch (separate-variance) "
                    "comparisons need at least 2 values in every group."
                )
        mse_value, df_pool = math.nan, math.nan
    ci_alpha, ci_note = _per_comparison_alpha(key, alpha, m)
    if ci_note:
        notes.append(ci_note)

    rows: list[dict[str, Any]] = []
    for i, j in pairs:
        a, b = s.arrays[i], s.arrays[j]
        target = f"{s.names[i]} vs {s.names[j]}"
        df: float | None
        if paired:
            d, tol = _paired_diff(a, b)
            # judged at the rounding error of the ORIGINAL values (100.1 - 100 and 1.1 - 1
            # are the same difference), as in the paired t test
            constant = differences_constant(d, tol)
            if constant:
                diff = float(snap_differences(d, tol)[int(np.argmin(tol))])
                se = 0.0
            else:
                rs = rescale(d)
                dz = rs.arrays[0]
                diff = rs.location(float(dz.mean()))
                se = rs.spread(float(dz.std(ddof=1)) / math.sqrt(d.size))
            df = float(d.size - 1)
            zero = constant and diff == 0.0
            effect = (EffectSize(name="Hedges' g_z (paired)", value=None, target=target)
                      if constant else es.cohen_d_paired(a, b, hedges=True, target=target))
        else:
            diff = float(s.means[i] - s.means[j])
            if pooled:
                se, df = math.sqrt(mse_value * (1.0 / s.n[i] + 1.0 / s.n[j])), df_pool
            else:
                se, df = _welch_pair(s.variances[i], s.n[i], s.variances[j], s.n[j])
            zero = _no_difference(diff, a, b)
            effect = _hedges_g(a, b, target)
        if se > 0:
            t: float | None = diff / se
            p: float | None = _t_two_sided(diff / se, df)
            half = float(stats.t.isf(ci_alpha / 2.0, df)) * se
            lo, hi = diff - half, diff + half
        else:
            notes.append(_ZERO_SE_WARNING)
            if not paired and not pooled:
                df = None
            if zero:
                t, p, lo, hi = None, None, None, None
            else:
                t, p, lo, hi = math.copysign(math.inf, diff), 0.0, diff, diff
        rows.append({"i": i, "j": j, "diff": diff, "se": se, "df": df, "t": t, "p": p,
                     "lo": lo, "hi": hi, "effect": effect})

    adjusted = _adjust([r["p"] for r in rows], key)
    comparisons = [
        Comparison(
            id="", group_a=s.names[r["i"]], group_b=s.names[r["j"]], estimate=r["diff"],
            se=r["se"], ci_low=r["lo"], ci_high=r["hi"], statistic_name="t", statistic=r["t"],
            df=r["df"], p_value=r["p"], p_adjusted=p_adj, effect_size=r["effect"],
        )
        for r, p_adj in zip(rows, adjusted)
    ]
    if paired:
        test = "paired t tests"
    elif pooled:
        test = "t tests using the pooled within-group SD"
    else:
        test = "Welch t tests (separate variances)"
    if key == "none":
        if paired:
            label = "Paired t tests (unadjusted)"
        elif pooled:
            label = "Fisher's LSD (unadjusted)"
        else:
            label = "Welch t tests (unadjusted)"
        if m > 1:
            notes.append(
                "P values are not adjusted for multiple comparisons, so the chance of at least "
                "one false positive across the family is larger than alpha."
            )
    else:
        label = METHOD_LABELS[key]
    return PosthocResult(
        comparisons, s.family(None if control is None else str(control), m), label,
        warnings=notes, citations=[_ADJUST_CITATIONS[key]],
        method=f"{test}, {METHOD_LABELS[key]} adjustment" if key != "none" else f"{test}, no adjustment",
    )


# ---------------------------------------------------------------------------- Dunn


def dunn(
    groups: Mapping[str, Any],
    *,
    adjust: str = "bonferroni",
    control: str | None = None,
    alpha: float = 0.05,
) -> PosthocResult:
    """Dunn's (1964) rank-sum multiple comparisons, used after a Kruskal-Wallis test.

    All values are ranked together (average ranks for ties) and for each pair

        z = (Rbar_i - Rbar_j) / sqrt((N(N+1)/12 - sum(t^3 - t) / (12 (N-1))) (1/n_i + 1/n_j))

    where ``t`` runs over the sizes of the groups of tied values. ``estimate`` is the
    difference in mean ranks, ``p_value`` the two-sided normal P value, ``p_adjusted``
    its adjustment. No confidence interval is given: the standard error is derived under
    the null hypothesis, so an interval around the mean-rank difference would not be
    valid. The effect size of each pair is the rank-biserial correlation of the two
    groups' own values.
    """
    check_alpha(alpha)
    key = _normalise_adjust(adjust)
    s = _Summary(groups, "Dunn's test")
    pairs = s.pairs(None if control is None else str(control))
    n_total = s.n_total
    ranks = stats.rankdata(np.concatenate(s.arrays))
    bounds = np.concatenate([[0], np.cumsum(s.n.astype(int))])
    mean_ranks = [float(np.mean(ranks[bounds[g]:bounds[g + 1]])) for g in range(s.k)]
    _, counts = np.unique(ranks, return_counts=True)
    ties = float(np.sum(counts.astype(float) ** 3 - counts))
    base = n_total * (n_total + 1) / 12.0 - (ties / (12.0 * (n_total - 1)) if n_total > 1 else 0.0)
    notes: list[str] = []
    if base <= 1e-12:
        notes.append(
            "All values are identical, so the ranks carry no information and Dunn's "
            "statistics are undefined."
        )
    rows = []
    for i, j in pairs:
        diff = mean_ranks[i] - mean_ranks[j]
        if base > 1e-12:
            se: float | None = math.sqrt(base * (1.0 / s.n[i] + 1.0 / s.n[j]))
            z: float | None = diff / se
            p: float | None = float(2.0 * stats.norm.sf(abs(z)))
        else:
            se, z, p = None, None, None
        rows.append((i, j, diff, se, z, p))
    adjusted = _adjust([r[5] for r in rows], key)
    comparisons = [
        Comparison(
            id="", group_a=s.names[i], group_b=s.names[j], estimate=diff, se=se,
            statistic_name="z", statistic=z, p_value=p, p_adjusted=p_adj,
            effect_size=es.rank_biserial_independent(
                s.arrays[i], s.arrays[j], target=f"{s.names[i]} vs {s.names[j]}"),
        )
        for (i, j, diff, se, z, p), p_adj in zip(rows, adjusted)
    ]
    return PosthocResult(
        comparisons, s.family(None if control is None else str(control), len(pairs)),
        f"Dunn's test, {METHOD_LABELS[key]}" if key != "none" else "Dunn's test (unadjusted)",
        warnings=notes, citations=[CITE_DUNN, _ADJUST_CITATIONS[key]] if key != "none" else [CITE_DUNN],
        method="Dunn's rank-sum comparisons (joint ranks, tie-corrected variance)",
    )


# ------------------------------------------------------- after the Friedman test


def _paired_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a - b, rounding tolerance of each difference); StatsError on overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        d = a - b
    if not np.all(np.isfinite(d)):
        raise StatsError(_TOO_LARGE)
    return d, difference_tolerance(a, b)


def _walsh_median(d: np.ndarray) -> float:
    """Hodges-Lehmann estimate of the typical difference: median of all Walsh averages."""
    i, j = np.triu_indices(d.size)
    return float(np.median(d[i] / 2.0 + d[j] / 2.0))


def friedman_posthoc(
    matrix: Any,
    names: Sequence[str],
    *,
    adjust: str = "bonferroni",
    alpha: float = 0.05,
    method: str = "dunn",
) -> PosthocResult:
    """Pairwise comparisons of related samples after the Friedman test.

    ``matrix`` has one row per subject (block) and one column per condition, with no
    missing values; ``names`` labels the columns.

    ``method='dunn'`` (default): values are ranked within each subject and the rank sums
    compared with ``z = (R_i - R_j) / sqrt(n k (k + 1) / 6)`` (Dunn 1964; Siegel &
    Castellan 1988; the same z underlies the Nemenyi test in Demsar 2006). ``estimate``
    is the difference in mean ranks; no confidence interval is defined. The classical
    variance has no tie correction, which makes the comparisons slightly conservative
    when there are ties within subjects.

    ``method='wilcoxon'``: a Wilcoxon (1945) matched-pairs signed-rank test for each pair,
    computed by the same engine as :func:`torusplot.stats.two_group.wilcoxon_signed_rank`
    (zero differences discarded; differences compared at the rounding error of the
    original values; exact P value without ties for up to 50 non-zero differences, exact
    permutation P value of the tied mid-ranks for up to 6 (``EXACT_TIES_MAX_N``), otherwise the normal
    approximation with tie correction and continuity correction, never below the
    smallest attainable P value), so each comparison has exactly the P value of that
    two-group test. ``estimate`` is the Hodges-Lehmann median of the paired differences
    and the statistic is W = W+ - W- (sum of the signed ranks), as in the two-group test.
    """
    check_alpha(alpha)
    key = _normalise_adjust(adjust)
    if method not in {"dunn", "wilcoxon"}:
        raise ValueError("method must be 'dunn' or 'wilcoxon'.")
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise StatsError("Repeated-measures comparisons need a table with rows = subjects and "
                         "columns = conditions.")
    n, k = data.shape
    labels = [str(name) for name in names]
    if len(labels) != k:
        raise ValueError("names must have one entry per column of the matrix.")
    if k < 2:
        raise StatsError("Comparisons after the Friedman test need at least 2 conditions.")
    if n < 2:
        raise StatsError("Comparisons after the Friedman test need at least 2 subjects (rows).")
    if not np.all(np.isfinite(data)):
        raise StatsError("The repeated-measures table contains missing values; every subject "
                         "needs a value in every condition.")
    pairs = list(combinations(range(k), 2))
    noun = "comparison" if len(pairs) == 1 else "comparisons"
    family = f"all {len(pairs)} pairwise {noun} among {k} related conditions"
    notes: list[str] = []
    rows = []
    if method == "dunn":
        ranks = np.apply_along_axis(stats.rankdata, 1, data)
        rank_sums = ranks.sum(axis=0)
        se_sum = math.sqrt(n * k * (k + 1) / 6.0)
        for i, j in pairs:
            z = float((rank_sums[i] - rank_sums[j]) / se_sum)
            rows.append((i, j, float((rank_sums[i] - rank_sums[j]) / n), se_sum / n, "z", z,
                         float(2.0 * stats.norm.sf(abs(z)))))
        if any(np.unique(row).size < k for row in data):
            notes.append(
                "Some subjects have tied values across conditions; the rank-sum comparisons "
                "use the classical variance without a tie correction (slightly conservative)."
            )
        label = f"Dunn's test, {METHOD_LABELS[key]}" if key != "none" else "Dunn's test (unadjusted)"
        cites = [CITE_DUNN, CITE_SIEGEL]
        what = "Dunn-type comparisons of within-subject rank sums"
    else:
        for i, j in pairs:
            raw, tol = _paired_diff(data[:, i], data[:, j])
            d = snap_differences(raw, tol)
            if not np.any(d != 0):
                rows.append((i, j, 0.0, None, "W", None, None))
                notes.append("At least one pair of conditions has identical values for every "
                             "subject; its Wilcoxon test is undefined.")
                continue
            core = _signed_rank_core(d, zero_method="wilcox", method="auto", alternative="two-sided")
            rows.append((i, j, _walsh_median(d), None, "W", float(core["w"]), core["p"]))
        label = (f"Wilcoxon signed-rank tests, {METHOD_LABELS[key]}" if key != "none"
                 else "Wilcoxon signed-rank tests (unadjusted)")
        cites = [CITE_WILCOXON]
        what = "pairwise Wilcoxon matched-pairs signed-rank tests"
        notes.append(
            "Each Wilcoxon comparison is the same test as the two-group Wilcoxon matched-pairs "
            "signed-rank test (zero differences discarded; W = sum of the signed ranks). P values "
            "are exact when no two non-zero differences are tied and there are at most 50 of them, "
            f"exact for the tied ranks (permutation distribution) when there are ties and at most "
            f"{EXACT_TIES_MAX_N} non-zero differences, and otherwise from the normal approximation with tie correction "
            "and continuity correction."
        )
    if key != "none":
        cites.append(_ADJUST_CITATIONS[key])
    adjusted = _adjust([r[6] for r in rows], key)
    comparisons = [
        Comparison(
            id="", group_a=labels[i], group_b=labels[j], estimate=est, se=se,
            statistic_name=stat_name, statistic=stat, p_value=p, p_adjusted=p_adj,
            # differences with rounding noise removed, as in the two-group signed-rank test
            effect_size=es.rank_biserial_paired(
                snap_differences(*_paired_diff(data[:, i], data[:, j])), target=f"{labels[i]} vs {labels[j]}"),
        )
        for (i, j, est, se, stat_name, stat, p), p_adj in zip(rows, adjusted)
    ]
    return PosthocResult(comparisons, family, label, warnings=notes, citations=cites,
                         method=what)


# ---------------------------------------------------------------------- dispatcher


def run_posthoc(
    groups: Mapping[str, Any],
    method: str,
    *,
    control: str | None = None,
    alpha: float = 0.05,
    pooled: bool = True,
    alternative: str = "two-sided",
    mse: float | None = None,
    df_error: float | None = None,
) -> PosthocResult:
    """Run a mean-comparison post-hoc procedure by name.

    ``method`` is one of ``tukey``, ``dunnett`` (needs ``control``), ``games-howell``,
    or a pairwise-t adjustment: ``bonferroni``, ``sidak``, ``holm``, ``holm-sidak``,
    ``fdr_bh``, ``lsd`` / ``none`` (unadjusted, Fisher's LSD). For the pairwise-t
    methods ``control`` switches to many-to-one comparisons and ``pooled=False`` uses
    Welch t tests. Returns ``(comparisons, family description, correction label)``
    (a :class:`PosthocResult`).

    Note that ``"none"`` here means "no P value adjustment" (the comparisons are still
    computed); callers that offer "no post-hoc tests" must simply not call this function.
    """
    key = str(method).strip().lower().replace("_", "-").replace(" ", "-")
    key = {"tukey-hsd": "tukey", "tukey-kramer": "tukey", "gameshowell": "games-howell",
           "fdr-bh": "fdr_bh", "fisher-lsd": "lsd"}.get(key, key)
    if key == "tukey":
        return tukey_hsd(groups, alpha=alpha, mse=mse, df_error=df_error)
    if key == "dunnett":
        if control is None:
            raise StatsError("Dunnett's test needs a control group to compare the others with.")
        return dunnett(groups, control, alpha=alpha, alternative=alternative, mse=mse,
                       df_error=df_error)
    if key == "games-howell":
        return games_howell(groups, alpha=alpha)
    if key in {"bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh", "lsd", "none"}:
        return pairwise_t(groups, pooled=pooled, adjust=key, control=control, alpha=alpha,
                          mse=mse if pooled else None, df_error=df_error if pooled else None)
    raise StatsError(
        f"Unknown post-hoc method '{method}'. Choose one of: {', '.join(POSTHOC_METHODS)}."
    )
