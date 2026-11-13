"""Post-hoc multiple comparisons: published values, cross-implementation and edge cases.

Sources for the reference values are recorded in ``tests/fixtures/sources/oneway.md``.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import scikit_posthocs as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, special, stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from torusplot.stats import StatsError
from torusplot.stats import posthoc as ph

# ------------------------------------------------------------------------- data

# Dobson (1983) Table 7.1: dried weight of plants under a control and two treatments.
PLANT = {
    "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
    "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
    "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
}

# Tippett (1950) warp breaks per loom: wool type (A, B) x tension (L, M, H), 9 looms each.
WARPBREAKS = {
    ("A", "L"): [26, 30, 54, 25, 70, 52, 51, 26, 67],
    ("A", "M"): [18, 21, 29, 17, 12, 18, 35, 30, 36],
    ("A", "H"): [36, 21, 24, 18, 10, 43, 28, 15, 26],
    ("B", "L"): [27, 14, 29, 19, 29, 31, 41, 20, 44],
    ("B", "M"): [42, 26, 19, 16, 39, 28, 21, 39, 29],
    ("B", "H"): [20, 21, 24, 17, 13, 15, 15, 16, 28],
}

# Unequal group sizes and spreads, with some ties (made up for cross-checks).
UNEQUAL = {
    "A": np.array([10.1, 12.3, 9.8, 11.4, 10.9, 12.0]),
    "B": np.array([13.2, 15.1, 14.8, 12.9, 16.3, 15.5, 14.1, 13.8]),
    "C": np.array([9.1, 8.4, 10.9, 9.9]),
    "D": np.array([11.0, 18.5, 14.2, 20.3, 9.7, 16.6, 12.3]),
}
TIED = {
    "A": np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0]),
    "B": np.array([2.0, 4.0, 5.0, 5.0, 6.0]),
    "C": np.array([4.0, 6.0, 7.0, 7.0, 7.0, 8.0, 9.0]),
}


def _long(groups):
    return pd.DataFrame({
        "value": np.concatenate(list(groups.values())),
        "group": np.concatenate([[name] * len(v) for name, v in groups.items()]),
    })


def _pooled(groups):
    arrays = list(groups.values())
    df = sum(a.size for a in arrays) - len(arrays)
    return sum((a.size - 1) * a.var(ddof=1) for a in arrays) / df, df


# ------------------------------------------------------------- result container


def test_posthoc_result_unpacks_as_three_tuple():
    res = ph.tukey_hsd(PLANT)
    comparisons, family, correction = res
    assert comparisons is res.comparisons and family == res.family and correction == res.correction
    assert family == "all 3 pairwise comparisons among 3 groups"
    assert correction == "Tukey HSD"
    assert res.citations and "Tukey" in res.citations[0] and res.method
    assert [(c.group_a, c.group_b) for c in comparisons] == [
        ("ctrl", "trt1"), ("ctrl", "trt2"), ("trt1", "trt2")]


# --------------------------------------------------------------------------- Tukey


def test_tukey_published_plantgrowth():
    """Dobson's plant-growth data: values printed by R's TukeyHSD(aov(weight ~ group))
    (differences are B - A there, A - B here)."""
    res = ph.tukey_hsd(PLANT)
    expected = {
        ("ctrl", "trt1"): (0.371, -0.3202161, 1.0622161, 0.3908711),
        ("ctrl", "trt2"): (-0.494, -1.1852161, 0.1972161, 0.1979960),
        ("trt1", "trt2"): (-0.865, -1.5562161, -0.1737839, 0.0120064),
    }
    for c in res.comparisons:
        diff, lo, hi, p = expected[(c.group_a, c.group_b)]
        assert c.estimate == pytest.approx(diff, abs=5e-8)
        assert c.ci_low == pytest.approx(lo, abs=5e-8)
        assert c.ci_high == pytest.approx(hi, abs=5e-8)
        assert c.p_adjusted == pytest.approx(p, abs=5e-8)
        assert c.statistic_name == "q" and c.df == 27


def test_tukey_published_warpbreaks_with_model_error_override():
    """R's documented example ``TukeyHSD(aov(breaks ~ wool + tension), "tension")``:
    the error mean square comes from the additive two-factor model (50 df), supplied
    through ``mse`` / ``df_error``."""
    values = np.concatenate([np.asarray(v, float) for v in WARPBREAKS.values()])
    grand = values.mean()
    tension = {t: np.concatenate([WARPBREAKS[(w, t)] for w in "AB"]).astype(float) for t in "LMH"}
    wool = {w: np.concatenate([WARPBREAKS[(w, t)] for t in "LMH"]).astype(float) for w in "AB"}
    ss_total = ((values - grand) ** 2).sum()
    ss_tension = sum(v.size * (v.mean() - grand) ** 2 for v in tension.values())
    ss_wool = sum(v.size * (v.mean() - grand) ** 2 for v in wool.values())
    df_error = values.size - 1 - 1 - 2
    mse = (ss_total - ss_tension - ss_wool) / df_error
    res = ph.tukey_hsd(tension, mse=mse, df_error=df_error)
    expected = {  # diff, lwr, upr, p adj (printed to 7 significant digits)
        ("M", "H"): (4.722222, -4.6311985, 14.07564, 0.4474210),
        ("L", "H"): (14.722222, 5.3688015, 24.07564, 0.0011218),
        ("L", "M"): (10.000000, 0.6465793, 19.35342, 0.0336262),
    }
    for c in res.comparisons:
        diff, lo, hi, p = expected[(c.group_a, c.group_b)]
        assert c.estimate == pytest.approx(diff, abs=5e-7)
        assert c.ci_low == pytest.approx(lo, abs=5e-7)
        assert c.ci_high == pytest.approx(hi, abs=5e-6)
        assert c.p_adjusted == pytest.approx(p, abs=5e-8)
        assert c.df == 50


def test_studentized_range_critical_values_match_published_table():
    """Upper 5% points of the studentized range (Harter 1960 tables, as reprinted in
    most textbooks): q(k, df)."""
    table = {(3, 30): 3.49, (3, 10): 3.88, (5, 20): 4.23, (4, 20): 3.96, (2, 10): 3.15}
    for (k, df), q in table.items():
        assert round(ph._srange_ppf(0.95, k, df), 2) == q


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
def test_tukey_matches_scipy_and_first_principles(groups):
    res = ph.tukey_hsd(groups)
    arrays = list(groups.values())
    ref = stats.tukey_hsd(*arrays)
    ci = ref.confidence_interval(0.95)
    mse, df = _pooled(groups)
    names = list(groups)
    k = len(names)
    for c in res.comparisons:
        i, j = names.index(c.group_a), names.index(c.group_b)
        a, b = arrays[i], arrays[j]
        assert c.estimate == pytest.approx(a.mean() - b.mean(), rel=1e-12)
        se = math.sqrt(mse * (1 / a.size + 1 / b.size))
        assert c.se == pytest.approx(se, rel=1e-12)
        q = abs(a.mean() - b.mean()) / math.sqrt(mse / 2 * (1 / a.size + 1 / b.size))
        assert c.statistic == pytest.approx(q, rel=1e-12)
        assert c.p_adjusted == pytest.approx(stats.studentized_range.sf(q, k, df), rel=1e-9)
        assert c.p_adjusted == pytest.approx(ref.pvalue[i, j], rel=1e-6, abs=1e-12)
        assert c.ci_low == pytest.approx(ci.low[i, j], rel=1e-6)
        assert c.ci_high == pytest.approx(ci.high[i, j], rel=1e-6)
        # unadjusted P value = pooled-variance t test with the ANOVA error df
        assert c.p_value == pytest.approx(2 * stats.t.sf(abs(c.estimate) / se, df), rel=1e-10)
        assert c.p_adjusted >= c.p_value
        # effect size: Hedges' g of the two groups
        sp2 = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
        d = (a.mean() - b.mean()) / math.sqrt(sp2)
        nu = a.size + b.size - 2
        j_corr = math.exp(special.gammaln(nu / 2) - special.gammaln((nu - 1) / 2)) / math.sqrt(nu / 2)
        assert c.effect_size.name == "Hedges' g"
        assert c.effect_size.value == pytest.approx(d * j_corr, rel=1e-10)


def test_tukey_kramer_matches_statsmodels():
    res = ph.tukey_hsd(UNEQUAL)
    assert res.correction == "Tukey-Kramer" and any("Kramer" in c for c in res.citations)
    frame = _long(UNEQUAL)
    ref = pairwise_tukeyhsd(frame["value"], frame["group"])
    pairs = [(a, b) for i, a in enumerate(ref.groupsunique) for b in ref.groupsunique[i + 1:]]
    for (a, b), diff, p, (lo, hi) in zip(pairs, ref.meandiffs, ref.pvalues, ref.confint):
        c = res.comparisons[[(x.group_a, x.group_b) for x in res.comparisons].index((a, b))]
        # statsmodels reports group2 - group1
        assert c.estimate == pytest.approx(-diff, rel=1e-9)
        assert c.p_adjusted == pytest.approx(p, abs=1e-6)
        assert c.ci_low == pytest.approx(-hi, rel=1e-5)
        assert c.ci_high == pytest.approx(-lo, rel=1e-5)


def test_tukey_two_groups_equals_t_test():
    two = {"A": UNEQUAL["A"], "B": UNEQUAL["B"]}
    c = ph.tukey_hsd(two).comparisons[0]
    ref = stats.ttest_ind(two["A"], two["B"])
    assert c.p_adjusted == pytest.approx(ref.pvalue, rel=1e-8)
    assert c.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert c.statistic == pytest.approx(abs(ref.statistic) * math.sqrt(2), rel=1e-10)


def test_tukey_ci_excludes_zero_exactly_when_significant():
    rng = np.random.default_rng(5)
    for _ in range(15):
        groups = {f"g{i}": rng.normal(i * 0.5, 1.0, rng.integers(3, 9)) for i in range(4)}
        for c in ph.tukey_hsd(groups, alpha=0.1).comparisons:
            assert (c.ci_low > 0 or c.ci_high < 0) == (c.p_adjusted < 0.1)


def test_tukey_error_override_validation():
    with pytest.raises(ValueError):
        ph.tukey_hsd(PLANT, mse=1.0)
    with pytest.raises(StatsError):
        ph.tukey_hsd(PLANT, mse=-1.0, df_error=10)
    with pytest.raises(StatsError):
        ph.tukey_hsd(PLANT, mse=1.0, df_error=0)


# ------------------------------------------------------------------------- Dunnett


def _dunnett_by_quadpack(c, n_treat, n_control, df, two_sided=True):
    """Independent evaluation of Dunnett's (1955) double integral with adaptive
    QUADPACK quadrature (the module uses fixed composite Gauss-Legendre rules)."""
    b = np.sqrt(np.asarray(n_treat, float) / (np.asarray(n_treat, float) + n_control))
    s = np.sqrt(1 - b * b)

    def inner(u):
        def g(z):
            term = special.ndtr((b * z + c * u) / s)
            if two_sided:
                term = term - special.ndtr((b * z - c * u) / s)
            return np.prod(term) * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return integrate.quad(g, -9, 9, epsabs=1e-12, epsrel=1e-12, limit=300)[0]

    def outer(u):
        return inner(u) * stats.chi.pdf(u * math.sqrt(df), df) * math.sqrt(df)

    hi = stats.chi.isf(1e-13, df) / math.sqrt(df)
    return integrate.quad(outer, 0, hi, epsabs=1e-11, epsrel=1e-11, limit=300)[0]


def test_dunnett_critical_values_match_published_tables():
    """Two-sided 5% points from the tables of Dunnett (1964) and one-sided 5% points from
    the tables of Dunnett (1955), for equal group sizes (k = number of treatments compared
    with the control; the infinite-df row is approximated by df = 1e7)."""
    two_sided = {10: [2.23, 2.57, 2.76, 2.89], 20: [2.09, 2.38, 2.54, 2.65],
                 1e7: [1.96, 2.21, 2.35, 2.44]}
    one_sided = {10: [1.81, 2.15, 2.34, 2.47], 20: [1.72, 2.03, 2.19, 2.30]}
    for two, table in ((True, two_sided), (False, one_sided)):
        for df, row in table.items():
            for k, published in zip((1, 2, 3, 4), row):
                crit = ph._dunnett_critical(0.05, np.full(k, 6.0), 6.0, df, two)
                assert round(crit, 2) == pytest.approx(published, abs=1e-9), (two, df, k, crit)


@pytest.mark.parametrize("case", [
    (2.38, [5, 5], 5, 20), (0.5, [5, 5, 5], 5, 12), (4.0, [4, 5], 6, 12),
    (2.0, [40, 3], 2, 3), (2.5, [2] * 6, 30, 1), (3.0, [10] * 5, 10, 400),
])
@pytest.mark.parametrize("two_sided", [True, False])
def test_dunnett_probability_matches_independent_integration(case, two_sided):
    c, n_treat, n_control, df = case
    mine = ph._dunnett_cdf(c, np.asarray(n_treat, float), float(n_control), df, two_sided)
    assert mine == pytest.approx(_dunnett_by_quadpack(c, n_treat, n_control, df, two_sided), abs=2e-9)


def test_dunnett_single_comparison_is_the_t_distribution():
    for c in (0.3, 1.7, 3.2):
        assert ph._dunnett_cdf(c, np.array([7.0]), 4.0, 9, True) == pytest.approx(
            1 - 2 * stats.t.sf(c, 9), abs=1e-11)
        assert ph._dunnett_cdf(c, np.array([7.0]), 4.0, 9, False) == pytest.approx(
            stats.t.cdf(c, 9), abs=1e-11)
    assert ph._dunnett_cdf(0.0, np.array([5.0, 5.0]), 5.0, 9, True) == 0.0
    assert ph._dunnett_cdf(-0.4, np.array([7.0]), 4.0, 9, False) == pytest.approx(
        stats.t.cdf(-0.4, 9), abs=1e-11)


@pytest.mark.parametrize("groups,control", [(PLANT, "ctrl"), (UNEQUAL, "C"), (TIED, "A")])
def test_dunnett_matches_scipy(groups, control):
    """scipy.stats.dunnett integrates the multivariate t by randomised quasi-Monte-Carlo,
    so its P values and intervals carry noise of roughly 1e-4 (they change with the
    seed). The t statistics are deterministic and must agree exactly; P values and CIs
    are compared with a tolerance that reflects scipy's integration noise."""
    res = ph.dunnett(groups, control)
    others = [name for name in groups if name != control]
    ref = stats.dunnett(*[groups[n] for n in others], control=groups[control], rng=12345)
    ci = ref.confidence_interval(0.95)
    mse, df = _pooled(groups)
    assert [c.group_a for c in res.comparisons] == others
    for idx, c in enumerate(res.comparisons):
        a, b = groups[c.group_a], groups[control]
        assert c.group_b == control and c.df == df
        t = (a.mean() - b.mean()) / math.sqrt(mse * (1 / a.size + 1 / b.size))
        assert c.statistic == pytest.approx(t, rel=1e-12)
        assert c.statistic == pytest.approx(ref.statistic[idx], rel=1e-10)
        assert c.estimate == pytest.approx(a.mean() - b.mean(), rel=1e-12)
        assert c.p_adjusted == pytest.approx(ref.pvalue[idx], abs=1e-3)
        assert c.ci_low == pytest.approx(ci.low[idx], abs=5e-3 * c.se)
        assert c.ci_high == pytest.approx(ci.high[idx], abs=5e-3 * c.se)
        assert c.p_value == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-10)
        assert c.p_value <= c.p_adjusted <= min(1.0, len(others) * c.p_value) + 1e-12
        # exact reference for the adjusted P value
        n_treat = [groups[n].size for n in others]
        exact = 1 - _dunnett_by_quadpack(abs(t), n_treat, b.size, df)
        assert c.p_adjusted == pytest.approx(exact, abs=2e-9)


def test_dunnett_is_deterministic_and_ci_matches_p():
    first = ph.dunnett(UNEQUAL, "A", alpha=0.05)
    second = ph.dunnett(UNEQUAL, "A", alpha=0.05)
    for a, b in zip(first.comparisons, second.comparisons):
        assert (a.p_adjusted, a.ci_low, a.ci_high) == (b.p_adjusted, b.ci_low, b.ci_high)
        assert (a.ci_low > 0 or a.ci_high < 0) == (a.p_adjusted < 0.05)
        # the simultaneous interval is wider than the unadjusted t interval and narrower
        # than the Bonferroni interval
        half = (a.ci_high - a.ci_low) / 2
        assert stats.t.isf(0.025, a.df) * a.se < half < stats.t.isf(0.025 / 3, a.df) * a.se


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_dunnett_one_sided(alternative):
    res = ph.dunnett(UNEQUAL, "A", alternative=alternative)
    others = [n for n in UNEQUAL if n != "A"]
    ref = stats.dunnett(*[UNEQUAL[n] for n in others], control=UNEQUAL["A"],
                        alternative=alternative, rng=7)
    mse, df = _pooled(UNEQUAL)
    n_treat = [UNEQUAL[n].size for n in others]
    for idx, c in enumerate(res.comparisons):
        assert c.p_adjusted == pytest.approx(ref.pvalue[idx], abs=1e-3)
        sign = 1.0 if alternative == "greater" else -1.0
        exact = 1 - _dunnett_by_quadpack(sign * c.statistic, n_treat, UNEQUAL["A"].size, df, False)
        assert c.p_adjusted == pytest.approx(exact, abs=2e-9)
        if alternative == "greater":
            assert c.ci_high == math.inf and c.ci_low < c.estimate
            assert c.p_value == pytest.approx(stats.t.sf(c.statistic, df), rel=1e-10)
        else:
            assert c.ci_low == -math.inf and c.ci_high > c.estimate
            assert c.p_value == pytest.approx(stats.t.cdf(c.statistic, df), rel=1e-10)
    assert "one-sided" in res.method


def test_dunnett_bad_input():
    with pytest.raises(StatsError, match="control"):
        ph.dunnett(PLANT, "placebo")
    with pytest.raises(StatsError):
        ph.dunnett(PLANT, "ctrl", alternative="bigger")
    with pytest.raises(StatsError, match="replicates"):
        ph.dunnett({"a": np.array([1.0]), "b": np.array([2.0])}, "a")


# -------------------------------------------------------------------- Games-Howell


def test_games_howell_hand_computed_example():
    """A = (1, 2, 3), B = (2, 4, 6), C = (10, 11, 15). By hand: means 2, 4, 12;
    variances 1, 4, 7. A vs B: SE = sqrt(1/3 + 4/3) = sqrt(5/3);
    df = (5/3)^2 / ((1/3)^2/2 + (4/3)^2/2) = 50/17; q = sqrt(2) * 2 / sqrt(5/3)."""
    groups = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([2.0, 4.0, 6.0]),
              "C": np.array([10.0, 11.0, 15.0])}
    res = ph.games_howell(groups)
    ab, ac, bc = res.comparisons
    assert ab.estimate == pytest.approx(-2.0) and ab.se == pytest.approx(math.sqrt(5 / 3))
    assert ab.df == pytest.approx(50 / 17)
    assert ab.statistic == pytest.approx(math.sqrt(2) * 2 / math.sqrt(5 / 3))
    assert ab.p_adjusted == pytest.approx(
        stats.studentized_range.sf(math.sqrt(2) * 2 / math.sqrt(5 / 3), 3, 50 / 17), rel=1e-9)
    # A vs C: SE = sqrt(1/3 + 7/3), df = (8/3)^2 / ((1/9 + 49/9) / 2) = 64/25
    assert ac.estimate == pytest.approx(-10.0) and ac.se == pytest.approx(math.sqrt(8 / 3))
    assert ac.df == pytest.approx(64 / 25)
    # B vs C: SE = sqrt(4/3 + 7/3), df = (11/3)^2 / ((16/9 + 49/9) / 2) = 242/65
    assert bc.se == pytest.approx(math.sqrt(11 / 3)) and bc.df == pytest.approx(242 / 65)
    assert res.correction == "Games-Howell"


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
def test_games_howell_first_principles_and_scikit_posthocs(groups):
    res = ph.games_howell(groups)
    ref = sp.posthoc_games_howell(_long(groups), val_col="value", group_col="group")
    k = len(groups)
    for c in res.comparisons:
        a, b = groups[c.group_a], groups[c.group_b]
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        welch = stats.ttest_ind(a, b, equal_var=False)
        assert c.se == pytest.approx(se, rel=1e-12) and c.df == pytest.approx(df, rel=1e-12)
        assert c.df == pytest.approx(welch.df, rel=1e-10)
        assert c.statistic == pytest.approx(abs(welch.statistic) * math.sqrt(2), rel=1e-10)
        assert c.p_value == pytest.approx(welch.pvalue, rel=1e-9)
        p = stats.studentized_range.sf(c.statistic, k, df)
        assert c.p_adjusted == pytest.approx(p, rel=1e-9)
        assert c.p_adjusted == pytest.approx(ref.loc[c.group_a, c.group_b], abs=1e-6)
        half = stats.studentized_range.ppf(0.95, k, df) / math.sqrt(2) * se
        assert c.ci_low == pytest.approx(c.estimate - half, rel=1e-8)
        assert c.ci_high == pytest.approx(c.estimate + half, rel=1e-8)
        assert (c.ci_low > 0 or c.ci_high < 0) == (c.p_adjusted < 0.05)
        assert c.p_adjusted >= c.p_value


def test_games_howell_needs_two_values_per_group():
    with pytest.raises(StatsError, match="at least 2 values"):
        ph.games_howell({"A": np.array([1.0]), "B": np.array([1.0, 2.0]), "C": np.array([2.0, 3.0])})


# --------------------------------------------------------------- pairwise t tests


@pytest.mark.parametrize("adjust,sm_name", [
    ("bonferroni", "bonferroni"), ("sidak", "sidak"), ("holm", "holm"),
    ("holm-sidak", "holm-sidak"), ("fdr_bh", "fdr_bh"),
])
def test_pairwise_t_pooled_matches_formula_and_statsmodels(adjust, sm_name):
    res = ph.pairwise_t(UNEQUAL, adjust=adjust)
    mse, df = _pooled(UNEQUAL)
    raw = []
    for c in res.comparisons:
        a, b = UNEQUAL[c.group_a], UNEQUAL[c.group_b]
        t = (a.mean() - b.mean()) / math.sqrt(mse * (1 / a.size + 1 / b.size))
        assert c.statistic == pytest.approx(t, rel=1e-12) and c.df == df
        assert c.p_value == pytest.approx(2 * stats.t.sf(abs(t), df), rel=1e-10)
        raw.append(2 * stats.t.sf(abs(t), df))
    expected = multipletests(raw, method=sm_name)[1]
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(expected, rel=1e-9)
    ref = sp.posthoc_ttest(_long(UNEQUAL), val_col="value", group_col="group", pool_sd=True,
                           p_adjust=sm_name)
    for c in res.comparisons:
        assert c.p_adjusted == pytest.approx(ref.loc[c.group_a, c.group_b], rel=1e-6)
        assert c.p_adjusted >= c.p_value - 1e-15


def test_pairwise_t_confidence_intervals():
    mse, df = _pooled(UNEQUAL)
    m = 6
    for adjust, per_alpha in (("bonferroni", 0.05 / m), ("sidak", 1 - 0.95 ** (1 / m)),
                              ("holm", 0.05), ("none", 0.05)):
        res = ph.pairwise_t(UNEQUAL, adjust=adjust)
        for c in res.comparisons:
            half = stats.t.isf(per_alpha / 2, df) * c.se
            assert c.ci_low == pytest.approx(c.estimate - half, rel=1e-10)
            assert c.ci_high == pytest.approx(c.estimate + half, rel=1e-10)
        noted = any("unadjusted" in w and "interval" in w for w in res.warnings)
        assert noted == (adjust == "holm")
    # Bonferroni: the simultaneous CI excludes zero exactly when adjusted P < alpha
    for c in ph.pairwise_t(UNEQUAL, adjust="bonferroni").comparisons:
        assert (c.ci_low > 0 or c.ci_high < 0) == (c.p_adjusted < 0.05)


def test_fisher_lsd_label_and_aliases():
    lsd = ph.pairwise_t(PLANT, adjust="none")
    assert lsd.correction == "Fisher's LSD (unadjusted)"
    assert any("not adjusted" in w for w in lsd.warnings)
    assert all(c.p_adjusted == c.p_value for c in lsd.comparisons)
    for alias in ("lsd", "LSD", None):
        other = ph.pairwise_t(PLANT, adjust=alias)
        assert [c.p_adjusted for c in other.comparisons] == [c.p_adjusted for c in lsd.comparisons]
    assert ph.pairwise_t(PLANT, adjust="none", pooled=False).correction == "Welch t tests (unadjusted)"
    assert ph.pairwise_t(PLANT, adjust="none", paired=True).correction == "Paired t tests (unadjusted)"
    with pytest.raises(StatsError, match="adjustment"):
        ph.pairwise_t(PLANT, adjust="scheffe")


def test_pairwise_t_welch_matches_scipy():
    res = ph.pairwise_t(UNEQUAL, pooled=False, adjust="holm")
    raw = []
    for c in res.comparisons:
        ref = stats.ttest_ind(UNEQUAL[c.group_a], UNEQUAL[c.group_b], equal_var=False)
        assert c.statistic == pytest.approx(ref.statistic, rel=1e-10)
        assert c.df == pytest.approx(ref.df, rel=1e-10)
        assert c.p_value == pytest.approx(ref.pvalue, rel=1e-9)
        ci = ref.confidence_interval(0.95)
        assert (c.ci_low, c.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-8)
        raw.append(ref.pvalue)
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(
        multipletests(raw, method="holm")[1], rel=1e-9)
    assert "Welch" in res.method
    with pytest.raises(StatsError, match="at least 2 values"):
        ph.pairwise_t({"A": np.array([1.0]), "B": np.array([1.0, 2.0])}, pooled=False)


def test_pairwise_t_paired_matches_scipy():
    rng = np.random.default_rng(11)
    base = rng.normal(10, 2, 9)
    groups = {"pre": base, "mid": base + rng.normal(0.8, 1, 9), "post": base + rng.normal(1.5, 1, 9)}
    res = ph.pairwise_t(groups, paired=True, adjust="bonferroni")
    for c in res.comparisons:
        ref = stats.ttest_rel(groups[c.group_a], groups[c.group_b])
        assert c.statistic == pytest.approx(ref.statistic, rel=1e-10)
        assert c.p_value == pytest.approx(ref.pvalue, rel=1e-9) and c.df == 8
        assert c.p_adjusted == pytest.approx(min(1.0, 3 * ref.pvalue), rel=1e-9)
        ci = ref.confidence_interval(1 - 0.05 / 3)
        assert (c.ci_low, c.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-8)
        assert c.effect_size.name == "Hedges' g_z (paired)"
    assert "paired" in res.method
    with pytest.raises(StatsError, match="same number"):
        ph.pairwise_t({"a": np.arange(4.0), "b": np.arange(5.0)}, paired=True)
    with pytest.raises(StatsError, match="2 subjects"):
        ph.pairwise_t({"a": np.array([1.0]), "b": np.array([2.0])}, paired=True)


def test_pairwise_t_paired_constant_differences():
    groups = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 3.0, 4.0]),
              "c": np.array([1.0, 2.0, 3.0])}
    res = ph.pairwise_t(groups, paired=True, adjust="holm")
    ab, ac, bc = res.comparisons
    assert ab.statistic == -math.inf and ab.p_value == 0.0 and ab.se == 0.0
    assert ac.statistic is None and ac.p_value is None and ac.p_adjusted is None
    assert bc.statistic == math.inf
    assert any("standard error of zero" in w for w in res.warnings)


def test_pairwise_t_many_to_one():
    res = ph.pairwise_t(UNEQUAL, control="B", adjust="holm")
    assert [(c.group_a, c.group_b) for c in res.comparisons] == [("A", "B"), ("C", "B"), ("D", "B")]
    assert res.family == "3 comparisons of each other group against the control group 'B'"
    raw = [c.p_value for c in res.comparisons]
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(
        multipletests(raw, method="holm")[1], rel=1e-9)
    with pytest.raises(StatsError, match="control"):
        ph.pairwise_t(UNEQUAL, control="Z")


# ---------------------------------------------------------------------------- Dunn


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
@pytest.mark.parametrize("adjust,sp_name", [
    ("none", None), ("bonferroni", "bonferroni"), ("sidak", "sidak"), ("holm", "holm"),
    ("holm-sidak", "holm-sidak"), ("fdr_bh", "fdr_bh"),
])
def test_dunn_matches_scikit_posthocs_and_formula(groups, adjust, sp_name):
    res = ph.dunn(groups, adjust=adjust)
    ref = sp.posthoc_dunn(_long(groups), val_col="value", group_col="group", p_adjust=sp_name)
    pooled = np.concatenate(list(groups.values()))
    ranks = stats.rankdata(pooled)
    n_total = pooled.size
    _, counts = np.unique(pooled, return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    base = n_total * (n_total + 1) / 12 - ties / (12 * (n_total - 1))
    mean_rank, start = {}, 0
    for name, values in groups.items():
        mean_rank[name] = ranks[start:start + values.size].mean()
        start += values.size
    for c in res.comparisons:
        na, nb = groups[c.group_a].size, groups[c.group_b].size
        z = (mean_rank[c.group_a] - mean_rank[c.group_b]) / math.sqrt(base * (1 / na + 1 / nb))
        assert c.estimate == pytest.approx(mean_rank[c.group_a] - mean_rank[c.group_b], rel=1e-12)
        assert c.statistic == pytest.approx(z, rel=1e-12) and c.statistic_name == "z"
        assert c.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
        assert c.p_adjusted == pytest.approx(ref.loc[c.group_a, c.group_b], rel=1e-6, abs=1e-12)
        assert c.p_adjusted >= c.p_value - 1e-15
        assert c.ci_low is None and c.ci_high is None and c.df is None
        assert c.effect_size.name == "rank-biserial r"
        u = stats.mannwhitneyu(groups[c.group_a], groups[c.group_b]).statistic
        assert c.effect_size.value == pytest.approx(2 * u / (na * nb) - 1, rel=1e-10, abs=1e-12)


def test_dunn_many_to_one_and_degenerate():
    res = ph.dunn(TIED, control="A", adjust="holm")
    assert [(c.group_a, c.group_b) for c in res.comparisons] == [("B", "A"), ("C", "A")]
    assert res.correction == "Dunn's test, Holm"
    assert ph.dunn(TIED, adjust="none").correction == "Dunn's test (unadjusted)"
    same = ph.dunn({"a": np.full(3, 2.0), "b": np.full(4, 2.0), "c": np.full(2, 2.0)})
    assert all(c.statistic is None and c.p_value is None and c.p_adjusted is None
               for c in same.comparisons)
    assert any("identical" in w for w in same.warnings)


# -------------------------------------------------------------- Friedman post hoc

# Hollander & Wolfe (1973), p. 140 ff.: rounding-first-base times of 22 players for three
# running methods (round out, narrow angle, wide angle).
ROUNDING_TIMES = np.array([
    [5.40, 5.50, 5.55], [5.85, 5.70, 5.75], [5.20, 5.60, 5.50], [5.55, 5.50, 5.40],
    [5.90, 5.85, 5.70], [5.45, 5.55, 5.60], [5.40, 5.40, 5.35], [5.45, 5.50, 5.35],
    [5.25, 5.15, 5.00], [5.85, 5.80, 5.70], [5.25, 5.20, 5.10], [5.65, 5.55, 5.45],
    [5.60, 5.35, 5.45], [5.05, 5.00, 4.95], [5.50, 5.50, 5.40], [5.45, 5.55, 5.50],
    [5.55, 5.55, 5.35], [5.45, 5.50, 5.55], [5.50, 5.45, 5.25], [5.65, 5.60, 5.40],
    [5.70, 5.65, 5.55], [6.30, 6.30, 6.25],
])
ROUNDING_NAMES = ["Round Out", "Narrow Angle", "Wide Angle"]


def test_friedman_posthoc_rank_sums_hand_check():
    """The within-player rank sums of the rounding-times data are 53, 47 and 32 (they
    add up to n k (k + 1) / 2 = 132); z = (R_i - R_j) / sqrt(n k (k + 1) / 6) with
    n = 22, k = 3, i.e. the denominator is sqrt(44)."""
    res = ph.friedman_posthoc(ROUNDING_TIMES, ROUNDING_NAMES, adjust="bonferroni")
    expected = {("Round Out", "Narrow Angle"): 6, ("Round Out", "Wide Angle"): 21,
                ("Narrow Angle", "Wide Angle"): 15}
    for c in res.comparisons:
        d = expected[(c.group_a, c.group_b)]
        z = d / math.sqrt(44)
        assert c.estimate == pytest.approx(d / 22) and c.statistic == pytest.approx(z)
        assert c.se == pytest.approx(math.sqrt(44) / 22)
        assert c.p_value == pytest.approx(2 * stats.norm.sf(z), rel=1e-10)
        assert c.p_adjusted == pytest.approx(min(1, 3 * 2 * stats.norm.sf(z)), rel=1e-10)
        assert c.effect_size.name == "matched-pairs rank-biserial r"
    ranks = np.array([stats.rankdata(row) for row in ROUNDING_TIMES])
    assert ranks.sum(axis=0).tolist() == [53.0, 47.0, 32.0]
    assert res.family == "all 3 pairwise comparisons among 3 related conditions"
    assert any("tied" in w for w in res.warnings)  # several players have tied times


@pytest.mark.parametrize("adjust,sp_name", [("none", None), ("bonferroni", "bonferroni"),
                                            ("holm", "holm"), ("fdr_bh", "fdr_bh")])
def test_friedman_posthoc_matches_scikit_posthocs(adjust, sp_name):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(12, 5)) + np.arange(5) * 0.5
    names = list("abcde")
    res = ph.friedman_posthoc(data, names, adjust=adjust)
    ref = sp.posthoc_siegel_friedman(data, p_adjust=sp_name)
    for c in res.comparisons:
        i, j = names.index(c.group_a), names.index(c.group_b)
        assert c.p_adjusted == pytest.approx(ref.iloc[i, j], rel=1e-6, abs=1e-12)
    assert not res.warnings


def test_friedman_posthoc_wilcoxon_option():
    res = ph.friedman_posthoc(ROUNDING_TIMES, ROUNDING_NAMES, adjust="holm", method="wilcoxon")
    raw = []
    for c in res.comparisons:
        i, j = ROUNDING_NAMES.index(c.group_a), ROUNDING_NAMES.index(c.group_b)
        d = ROUNDING_TIMES[:, i] - ROUNDING_TIMES[:, j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = stats.wilcoxon(d)
        assert c.statistic == pytest.approx(ref.statistic) and c.statistic_name == "W"
        assert c.p_value == pytest.approx(ref.pvalue, rel=1e-10)
        walsh = [(d[a] + d[b]) / 2 for a in range(d.size) for b in range(a, d.size)]
        assert c.estimate == pytest.approx(np.median(walsh))
        raw.append(ref.pvalue)
    assert [c.p_adjusted for c in res.comparisons] == pytest.approx(
        multipletests(raw, method="holm")[1], rel=1e-9)
    assert res.correction == "Wilcoxon signed-rank tests, Holm"
    assert ph.friedman_posthoc(ROUNDING_TIMES, ROUNDING_NAMES, adjust="none",
                               method="wilcoxon").correction.endswith("(unadjusted)")
    # identical columns: the signed-rank test is undefined for that pair
    data = np.column_stack([np.arange(6.0), np.arange(6.0), np.arange(6.0)[::-1]])
    same = ph.friedman_posthoc(data, ["a", "b", "c"], method="wilcoxon")
    assert same.comparisons[0].p_value is None and same.comparisons[0].statistic is None
    assert same.comparisons[1].p_value is not None


def test_friedman_posthoc_bad_input():
    with pytest.raises(StatsError):
        ph.friedman_posthoc(np.arange(5.0), ["a"])
    with pytest.raises(StatsError, match="2 subjects"):
        ph.friedman_posthoc(np.ones((1, 3)), list("abc"))
    with pytest.raises(StatsError, match="2 conditions"):
        ph.friedman_posthoc(np.ones((4, 1)), ["a"])
    with pytest.raises(StatsError, match="missing"):
        ph.friedman_posthoc(np.array([[1.0, np.nan], [2.0, 3.0]]), ["a", "b"])
    with pytest.raises(ValueError):
        ph.friedman_posthoc(np.ones((4, 3)), ["a", "b"])
    with pytest.raises(ValueError):
        ph.friedman_posthoc(np.ones((4, 3)), list("abc"), method="nemenyi")


# ---------------------------------------------------------------------- dispatcher


def test_run_posthoc_dispatch():
    assert ph.run_posthoc(PLANT, "tukey").correction == "Tukey HSD"
    assert ph.run_posthoc(PLANT, "Tukey HSD").correction == "Tukey HSD"
    assert ph.run_posthoc(PLANT, "dunnett", control="ctrl").correction == "Dunnett"
    assert ph.run_posthoc(PLANT, "games-howell").correction == "Games-Howell"
    assert ph.run_posthoc(PLANT, "games_howell").correction == "Games-Howell"
    for method, label in (("bonferroni", "Bonferroni"), ("sidak", "Šidák"), ("holm", "Holm"),
                          ("holm-sidak", "Holm-Šidák"), ("fdr_bh", "Benjamini-Hochberg FDR"),
                          ("lsd", "Fisher's LSD (unadjusted)"), ("none", "Fisher's LSD (unadjusted)")):
        comparisons, family, correction = ph.run_posthoc(PLANT, method)
        assert correction == label and len(comparisons) == 3 and "pairwise" in family
    assert set(ph.POSTHOC_METHODS) >= {"tukey", "dunnett", "games-howell", "lsd", "none"}
    many = ph.run_posthoc(PLANT, "holm", control="ctrl")
    assert len(many.comparisons) == 2
    welch = ph.run_posthoc(PLANT, "holm", pooled=False)
    assert welch.comparisons[0].df != 27
    override = ph.run_posthoc(PLANT, "bonferroni", mse=2.0, df_error=40)
    assert override.comparisons[0].df == 40
    with pytest.raises(StatsError, match="control"):
        ph.run_posthoc(PLANT, "dunnett")
    with pytest.raises(StatsError, match="Unknown post-hoc"):
        ph.run_posthoc(PLANT, "scheffe")


# ---------------------------------------------------------------- degenerate input


def test_input_validation():
    with pytest.raises(TypeError):
        ph.tukey_hsd([np.arange(3.0), np.arange(3.0)])
    with pytest.raises(StatsError, match="at least 2 groups"):
        ph.tukey_hsd({"only": np.arange(4.0)})
    with pytest.raises(StatsError, match="no values"):
        ph.tukey_hsd({"a": np.arange(4.0), "b": np.array([])})
    with pytest.raises(StatsError, match="missing or infinite"):
        ph.dunn({"a": np.arange(4.0), "b": np.array([1.0, np.nan])})
    with pytest.raises(StatsError, match="replicates"):
        ph.tukey_hsd({"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([4.0])})
    with pytest.raises(StatsError, match="alpha"):
        ph.tukey_hsd(PLANT, alpha=1.5)


@pytest.mark.parametrize("func", [
    lambda g: ph.tukey_hsd(g), lambda g: ph.dunnett(g, "a"), lambda g: ph.games_howell(g),
    lambda g: ph.pairwise_t(g, adjust="holm"), lambda g: ph.pairwise_t(g, pooled=False),
])
def test_zero_variance_everywhere(func):
    """All groups constant: infinite statistic and P = 0 where the means differ,
    undefined (None) where they are equal, plus an explicit warning; no crash, no
    numpy warnings."""
    groups = {"a": np.full(3, 0.1), "b": np.full(4, 0.1), "c": np.full(3, 0.3)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = func(groups)
    by_pair = {(c.group_a, c.group_b): c for c in res.comparisons}
    same = by_pair.get(("a", "b")) or by_pair.get(("b", "a"))
    diff = by_pair.get(("a", "c")) or by_pair.get(("c", "a"))
    assert same.statistic is None and same.p_value is None and same.p_adjusted is None
    assert same.ci_low is None and same.effect_size.value is None
    assert math.isinf(diff.statistic) and diff.p_adjusted == 0.0
    assert diff.ci_low == diff.ci_high == diff.estimate
    assert any("standard error of zero" in w for w in res.warnings)
    assert len(res.warnings) == len(set(res.warnings))


def test_dunnett_one_sided_with_zero_variance():
    groups = {"a": np.full(3, 1.0), "b": np.full(3, 2.0), "c": np.full(3, 0.5)}
    res = ph.dunnett(groups, "a", alternative="greater")
    b, c = res.comparisons
    assert b.statistic == math.inf and b.p_adjusted == 0.0
    assert c.statistic == -math.inf and c.p_adjusted == 1.0


def test_one_constant_group_is_fine():
    groups = {"a": np.full(4, 5.0), "b": np.array([4.0, 6.0, 7.0, 9.0]), "c": np.array([1.0, 2.0, 2.5])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gh = ph.games_howell(groups)
        tk = ph.tukey_hsd(groups)
    ab = gh.comparisons[0]
    assert ab.df == pytest.approx(3.0)  # only the varying group contributes
    assert all(0 <= c.p_adjusted <= 1 for c in gh.comparisons + tk.comparisons)
    assert not gh.warnings and not tk.warnings


def test_single_value_groups_allowed_with_pooled_error():
    groups = {"a": np.array([1.0]), "b": np.array([2.0, 3.0, 5.0]), "c": np.array([6.0, 9.0])}
    res = ph.tukey_hsd(groups)
    assert all(0 <= c.p_adjusted <= 1 and c.df == 3 for c in res.comparisons)
    assert res.comparisons[0].effect_size.value is None  # no SD for a single value
    dn = ph.dunnett(groups, "a")
    assert all(0 <= c.p_adjusted <= 1 for c in dn.comparisons)


def test_large_offset_groups_are_not_treated_as_constant():
    """1e12 + small scatter: must behave like the same data without the offset."""
    base = {"a": np.array([0.25, 0.5, 0.75, 1.0]), "b": np.array([1.0, 1.5, 1.25, 2.0]),
            "c": np.array([0.5, 0.25, 1.5, 1.75])}
    shifted = {k: v + 1e12 for k, v in base.items()}  # exactly representable
    for a, b in zip(ph.tukey_hsd(base).comparisons, ph.tukey_hsd(shifted).comparisons):
        assert b.p_adjusted == pytest.approx(a.p_adjusted, rel=1e-9)
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12)


# ------------------------------------------------------------------ property tests

_values = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False).map(
    lambda v: round(v, 1))  # one decimal: plenty of ties, zeros and constant groups


@st.composite
def _groups(draw, min_groups=2, max_groups=4, min_n=2, max_n=6):
    k = draw(st.integers(min_groups, max_groups))
    return {f"g{i}": np.array(draw(st.lists(_values, min_size=min_n, max_size=max_n)))
            for i in range(k)}


def _check_comparisons(res, n_expected):
    assert len(res.comparisons) == n_expected
    for c in res.comparisons:
        for p in (c.p_value, c.p_adjusted):
            assert p is None or 0.0 <= p <= 1.0
        if c.p_value is not None and c.p_adjusted is not None:
            assert c.p_adjusted >= c.p_value - 1e-12
        if c.ci_low is not None and c.ci_high is not None:
            assert c.ci_low <= c.estimate <= c.ci_high
        assert c.estimate is not None and not math.isnan(c.estimate)
        assert c.statistic is None or not math.isnan(c.statistic)


@settings(max_examples=40, deadline=None)
@given(_groups(), st.sampled_from(["tukey", "bonferroni", "sidak", "holm", "holm-sidak",
                                   "fdr_bh", "lsd"]))
def test_property_pooled_procedures(groups, method):
    k = len(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ph.run_posthoc(groups, method)
    _check_comparisons(res, k * (k - 1) // 2)
    # reversing the group order flips the sign of each difference, nothing else
    flipped = ph.run_posthoc(dict(reversed(list(groups.items()))), method)
    lookup = {(c.group_b, c.group_a): c for c in flipped.comparisons}
    for c in res.comparisons:
        other = lookup[(c.group_a, c.group_b)]
        assert other.estimate == pytest.approx(-c.estimate, abs=1e-9)
        if c.p_adjusted is None:
            assert other.p_adjusted is None
        else:
            assert other.p_adjusted == pytest.approx(c.p_adjusted, rel=1e-7, abs=1e-10)


@settings(max_examples=25, deadline=None)
@given(_groups(max_groups=3, max_n=5))
def test_property_games_howell(groups):
    k = len(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ph.games_howell(groups)
        welch = ph.pairwise_t(groups, pooled=False, adjust="holm")
    _check_comparisons(res, k * (k - 1) // 2)
    _check_comparisons(welch, k * (k - 1) // 2)


@settings(max_examples=30, deadline=None)
@given(_groups(min_groups=2, max_groups=4, min_n=1, max_n=5),
       st.sampled_from(["two-sided", "greater", "less"]))
def test_property_dunnett(groups, alternative):
    k = len(groups)
    n_total = sum(v.size for v in groups.values())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if n_total - k < 1:
            with pytest.raises(StatsError):
                ph.dunnett(groups, "g0", alternative=alternative)
            return
        res = ph.dunnett(groups, "g0", alternative=alternative)
    assert len(res.comparisons) == k - 1
    for c in res.comparisons:
        assert c.p_adjusted is None or 0.0 <= c.p_adjusted <= 1.0
        if c.p_value is not None:
            assert c.p_adjusted >= c.p_value - 1e-9
            # single-step Dunnett is never more conservative than Bonferroni
            assert c.p_adjusted <= min(1.0, (k - 1) * c.p_value) + 1e-9


@settings(max_examples=40, deadline=None)
@given(_groups(min_n=1), st.sampled_from(["bonferroni", "holm", "fdr_bh", "none"]),
       st.floats(min_value=-5, max_value=5), st.floats(min_value=0.5, max_value=20))
def test_property_dunn_rank_invariance(groups, adjust, shift, scale):
    k = len(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ph.dunn(groups, adjust=adjust)
        moved = ph.dunn({n: np.round(v * 10) * scale + shift for n, v in groups.items()},
                        adjust=adjust)
    _check_comparisons(res, k * (k - 1) // 2)
    for a, b in zip(res.comparisons, moved.comparisons):
        # ranks are unchanged by a strictly increasing transformation
        if a.statistic is None:
            assert b.statistic is None
        else:
            assert b.statistic == pytest.approx(a.statistic, rel=1e-9, abs=1e-12)


@settings(max_examples=30, deadline=None)
@given(st.integers(2, 7), st.integers(2, 4), st.data())
def test_property_friedman_posthoc(n, k, data):
    matrix = np.array(data.draw(st.lists(
        st.lists(_values, min_size=k, max_size=k), min_size=n, max_size=n)))
    names = [f"c{j}" for j in range(k)]
    for method in ("dunn", "wilcoxon"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = ph.friedman_posthoc(matrix, names, adjust="holm", method=method)
        assert len(res.comparisons) == k * (k - 1) // 2
        for c in res.comparisons:
            for p in (c.p_value, c.p_adjusted):
                assert p is None or 0.0 <= p <= 1.0
            if c.p_value is not None:
                assert c.p_adjusted >= c.p_value - 1e-12
