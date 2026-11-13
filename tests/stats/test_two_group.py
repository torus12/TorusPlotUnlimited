"""Two-group comparisons: t tests, Mann-Whitney, Wilcoxon signed-rank.

Validation sources (see tests/fixtures/sources/basics.md):
* Published: Student (1908) sleep data (the paper's own summary numbers, and the results
  printed by R's documented ``t.test`` / ``wilcox.test`` examples on the ``sleep`` data).
* Cross-checks against scipy (``ttest_ind``, ``ttest_rel``, ``ttest_1samp`` including
  ``confidence_interval()``, ``mannwhitneyu``, ``wilcoxon``) and statsmodels
  (``ttest_ind``, ``CompareMeans``, ``DescrStatsW``).
* First-principles computations written in this file (formulas; brute-force enumeration
  of the exact rank-test null distributions; brute-force Hodges-Lehmann estimates).
* hypothesis property tests and registry tests.
"""

import itertools
import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind

from torusplot.data.tables import ColumnTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import effect_sizes as es
from torusplot.stats import two_group as tg

# Student (1908) Biometrika 6:1-25, Table "additional hours of sleep gained" (Cushny & Peebles)
SLEEP_1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
SLEEP_2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])

RNG = np.random.default_rng(42)
A = RNG.normal(10.0, 2.0, 12)
B = RNG.normal(12.0, 3.5, 9)
P1 = RNG.normal(5.0, 1.0, 11)
P2 = P1 + RNG.normal(0.6, 0.8, 11)
TIES_A = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 7.0, 8.0])
TIES_B = np.array([2.0, 4.0, 5.0, 6.0, 6.0, 9.0, 9.0, 10.0, 11.0])
ALTERNATIVES = ("two-sided", "greater", "less")
SM_ALT = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)


def _full(res):
    """Checks every result must satisfy."""
    res = res.finalize()
    assert res.test_name and res.why_chosen and res.interpretation and res.methods_text
    assert res.assumptions and res.citations and res.descriptives
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    json.loads(res.model_dump_json())
    return res


# ================================================================ published values


def test_student_1908_sleep_data_original_summary():
    """Student (1908), section IX: for the difference between the two soporifics he gives
    mean +1.58, S.D. 1.17 (divisor n) and z = mean / S.D. = 1.35, P = 0.9985."""
    res = tg.ttest_paired(SLEEP_2, SLEEP_1, names=("Drug 2", "Drug 1"))
    d = SLEEP_2 - SLEEP_1
    assert res.parameters[0].value == pytest.approx(1.58, abs=0.005)
    sd_n = math.sqrt(((d - d.mean()) ** 2).mean())
    assert sd_n == pytest.approx(1.17, abs=0.005)
    z = res.parameters[0].value / sd_n
    assert z == pytest.approx(1.35, abs=0.005)
    # Student's z relates to the modern t by t = z sqrt(n - 1); his P is the one-sided probability
    assert res.statistic == pytest.approx(z * math.sqrt(9), rel=1e-12)
    assert 1 - res.p_value / 2 == pytest.approx(0.9985, abs=0.0002)


def test_sleep_data_paired_t_published_r_output():
    """R ``t.test(extra ~ group, data = sleep, paired = TRUE)``:
    t = -4.0621, df = 9, p-value = 0.002833, 95% CI -2.4598858 -0.7001142, mean diff -1.58."""
    res = _full(tg.ttest_paired(SLEEP_1, SLEEP_2, names=("Drug 1", "Drug 2")))
    assert res.statistic == pytest.approx(-4.0621, abs=5e-5)
    assert res.df == 9
    assert res.p_value == pytest.approx(0.002833, abs=5e-7)
    par = res.parameters[0]
    assert par.value == pytest.approx(-1.58, abs=1e-9)
    assert (par.ci_low, par.ci_high) == pytest.approx((-2.4598858, -0.7001142), abs=5e-8)
    assert res.significant is True and res.n == {"Drug 1": 10, "Drug 2": 10} and res.n_total == 10


def test_sleep_data_welch_and_student_published_r_output():
    """R ``t.test(extra ~ group, data = sleep)`` (Welch): t = -1.8608, df = 17.776,
    p-value = 0.07939, 95% CI -3.3654832 0.2054832, means 0.75 and 2.33.
    With ``var.equal = TRUE``: t = -1.8608, df = 18, p-value = 0.07919."""
    welch = _full(tg.ttest_welch(SLEEP_1, SLEEP_2))
    assert welch.statistic == pytest.approx(-1.8608, abs=5e-5)
    assert welch.df == pytest.approx(17.776, abs=5e-4)
    assert welch.p_value == pytest.approx(0.07939, abs=5e-6)
    par = welch.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((-3.3654832, 0.2054832), abs=5e-8)
    assert welch.parameter("Mean of A").value == pytest.approx(0.75)
    assert welch.parameter("Mean of B").value == pytest.approx(2.33)
    assert welch.significant is False
    student = _full(tg.ttest_unpaired(SLEEP_1, SLEEP_2))
    assert student.statistic == pytest.approx(-1.8608, abs=5e-5) and student.df == 18
    assert student.p_value == pytest.approx(0.07919, abs=5e-6)


def test_sleep_data_rank_tests_published_r_output():
    """R ``wilcox.test(extra ~ group, data = sleep)``: W = 25.5, p-value = 0.06933 (normal
    approximation with continuity correction, because of ties);
    with ``paired = TRUE``: V = 0, p-value = 0.009091."""
    mw = _full(tg.mann_whitney(SLEEP_1, SLEEP_2))
    assert mw.extra["u_a"] == 25.5 and mw.statistic == 25.5  # R's W is U of the first group
    assert mw.p_value == pytest.approx(0.06933, abs=5e-6)
    assert mw.extra["exact"] is False and "asymptotic" in mw.extra["p_method"]
    sr = _full(tg.wilcoxon_signed_rank(SLEEP_1, SLEEP_2))
    assert sr.extra["w_plus"] == 0.0  # R's V is the sum of positive ranks
    assert sr.p_value == pytest.approx(0.009091, abs=5e-7)
    assert sr.extra["n_zero"] == 1 and sr.extra["n_nonzero"] == 9 and sr.statistic == -45.0


# ================================================================== unpaired t tests


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("welch", [False, True])
def test_independent_t_matches_scipy_and_statsmodels(alternative, welch):
    func = tg.ttest_welch if welch else tg.ttest_unpaired
    res = _full(func(A, B, names=("ctl", "trt"), alternative=alternative))
    ref = stats.ttest_ind(A, B, equal_var=not welch, alternative=alternative)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert res.df == pytest.approx(ref.df, rel=1e-10)
    t_sm, p_sm, df_sm = sm_ttest_ind(A, B, alternative=SM_ALT[alternative],
                                     usevar="unequal" if welch else "pooled")
    assert res.statistic == pytest.approx(t_sm, rel=1e-10)
    assert res.p_value == pytest.approx(p_sm, rel=1e-8)
    assert res.df == pytest.approx(df_sm, rel=1e-10)
    # the CI is always two-sided (documented), so compare with the two-sided references
    ci = stats.ttest_ind(A, B, equal_var=not welch).confidence_interval(0.95)
    par = res.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    cm = CompareMeans(DescrStatsW(A), DescrStatsW(B))
    lo, hi = cm.tconfint_diff(alpha=0.05, usevar="unequal" if welch else "pooled")
    assert (par.ci_low, par.ci_high) == pytest.approx((lo, hi), rel=1e-9)
    assert res.alternative == alternative
    assert (len(res.warnings) == 0) == (alternative == "two-sided")
    if alternative != "two-sided":
        assert "two-sided" in res.warnings[-1] and "one-tailed" in res.interpretation


def test_independent_t_first_principles():
    n1, n2 = A.size, B.size
    m1, m2, v1, v2 = A.mean(), B.mean(), A.var(ddof=1), B.var(ddof=1)
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    se_pooled = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    student = tg.ttest_unpaired(A, B)
    assert student.statistic == pytest.approx((m1 - m2) / se_pooled, rel=1e-12)
    assert student.df == n1 + n2 - 2
    assert student.p_value == pytest.approx(2 * stats.t.sf(abs(student.statistic), n1 + n2 - 2), rel=1e-12)
    assert student.parameters[0].se == pytest.approx(se_pooled, rel=1e-12)
    se_w = math.sqrt(v1 / n1 + v2 / n2)
    df_w = se_w**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))  # Welch-Satterthwaite
    welch = tg.ttest_welch(A, B)
    assert welch.statistic == pytest.approx((m1 - m2) / se_w, rel=1e-12)
    assert welch.df == pytest.approx(df_w, rel=1e-12)
    tcrit = stats.t.ppf(0.975, df_w)
    par = welch.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((m1 - m2 - tcrit * se_w, m1 - m2 + tcrit * se_w), rel=1e-12)
    # 90% CI
    par90 = tg.ttest_welch(A, B, ci_level=0.90).parameters[0]
    t90 = stats.t.ppf(0.95, df_w)
    assert (par90.ci_low, par90.ci_high) == pytest.approx((m1 - m2 - t90 * se_w, m1 - m2 + t90 * se_w), rel=1e-12)
    assert par90.ci_level == 0.90


def test_independent_t_effect_sizes_descriptives_assumptions():
    res = tg.ttest_unpaired(A, B, names=("ctl", "trt"))
    d = es.cohen_d_independent(A, B)
    g = es.cohen_d_independent(A, B, hedges=True)
    assert [e.name for e in res.effect_sizes] == ["Cohen's d", "Hedges' g"]
    assert res.effect("Cohen's d").value == pytest.approx(d.value)
    assert (res.effect("Cohen's d").ci_low, res.effect("Cohen's d").ci_high) == pytest.approx((d.ci_low, d.ci_high))
    assert res.effect("Hedges' g").value == pytest.approx(g.value)
    # d and t are linked: d = t sqrt(1/n1 + 1/n2)
    assert d.value == pytest.approx(res.statistic * math.sqrt(1 / A.size + 1 / B.size), rel=1e-10)
    rows = {r["group"]: r for r in res.descriptives}
    assert rows["ctl"]["n"] == 12 and rows["trt"]["mean"] == pytest.approx(B.mean())
    assert rows["ctl"]["sd"] == pytest.approx(A.std(ddof=1))
    names = [(a.name, a.target) for a in res.assumptions]
    assert names == [("Normality (Shapiro-Wilk)", "ctl"), ("Normality (Shapiro-Wilk)", "trt"),
                     ("Equal variances (Brown-Forsythe)", None), ("Independent observations", None)]
    assert res.assumptions[0].p_value == pytest.approx(stats.shapiro(A).pvalue)
    bf = stats.levene(A, B, center="median")
    assert res.assumptions[2].statistic == pytest.approx(bf.statistic)
    assert res.assumptions[2].p_value == pytest.approx(bf.pvalue)
    welch = tg.ttest_welch(A, B)
    assert welch.assumptions[2].status == "info" and "not required" in welch.assumptions[2].name
    assert "pooled SD" in welch.effect("Cohen's d").ci_method
    assert welch.effect("Cohen's d").value == pytest.approx(d.value)


def test_unequal_variances_flagged_and_welch_recommended():
    rng = np.random.default_rng(3)
    x, y = rng.normal(0, 1, 30), rng.normal(0, 6, 30)
    res = tg.ttest_unpaired(x, y)
    check = res.assumptions[2]
    assert check.status == "fail" and "Welch" in check.detail
    assert res.worst_assumption_status() == "fail"
    skewed = tg.ttest_unpaired(rng.lognormal(0, 1.5, 30), x)
    assert skewed.assumptions[0].status == "fail" and "Mann-Whitney" in skewed.assumptions[0].detail


def test_independent_t_degenerate_inputs():
    for func in (tg.ttest_unpaired, tg.ttest_welch):
        both_const = _full(func([2.0, 2.0, 2.0], [5.0, 5.0, 5.0]))
        assert both_const.statistic is None and both_const.p_value is None and both_const.df is None
        assert both_const.parameters[0].value == -3.0 and both_const.parameters[0].ci_low is None
        assert "no variability" in both_const.warnings[0] and "cannot be computed" in both_const.interpretation
        assert both_const.effect("Cohen's d").value is None and both_const.significant is None
        one_const = _full(func([2.0, 2.0, 2.0], [4.0, 5.0, 7.0]))
        assert one_const.p_value is not None and any("no variability" in w for w in one_const.warnings)
        tiny = _full(func([1.0, 2.0], [2.0, 5.0]))
        assert any("only 2 values" in w for w in tiny.warnings)
        assert tiny.assumptions[0].status == "warn" and tiny.assumptions[2].status in {"warn", "info"}
        same = _full(func([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]))
        assert same.statistic == 0.0 and same.p_value == pytest.approx(1.0) and "identical" in same.interpretation
        with_nan = func([*A, np.nan, np.inf], B)
        assert with_nan.statistic == pytest.approx(func(A, B).statistic)
        assert len(with_nan.data_notes) == 2
        for bad in (([1.0], [1.0, 2.0]), ([], [1.0, 2.0]), ([np.nan, np.nan, 1.0], [1.0, 2.0])):
            with pytest.raises(StatsError):
                func(*bad)
        with pytest.raises(StatsError):
            func(None, B)
        for kw in ({"alpha": 0}, {"alternative": "bigger"}, {"ci_level": 1}):
            with pytest.raises(StatsError):
                func(A, B, **kw)
        with pytest.raises(TypeError):
            func(A, B, names=("only one",))
    assert list(tg.ttest_unpaired(A, B, names=("x", "x")).n) == ["x", "x (2)"]


# ============================================================================ paired


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_paired_t_matches_scipy_and_statsmodels(alternative):
    res = _full(tg.ttest_paired(P1, P2, names=("before", "after"), alternative=alternative))
    ref = stats.ttest_rel(P1, P2, alternative=alternative)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert res.df == ref.df == P1.size - 1
    d = DescrStatsW(P1 - P2)
    t_sm, p_sm, _ = d.ttest_mean(0.0, alternative=SM_ALT[alternative])
    assert res.statistic == pytest.approx(t_sm, rel=1e-10)
    assert res.p_value == pytest.approx(p_sm, rel=1e-8)
    ci = stats.ttest_rel(P1, P2).confidence_interval(0.95)
    par = res.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    assert (par.ci_low, par.ci_high) == pytest.approx(d.tconfint_mean(0.05), rel=1e-9)


def test_paired_t_first_principles_and_extras():
    res = tg.ttest_paired(P1, P2, names=("before", "after"))
    d = P1 - P2
    se = d.std(ddof=1) / math.sqrt(d.size)
    assert res.statistic == pytest.approx(d.mean() / se, rel=1e-12)
    assert res.parameters[0].se == pytest.approx(se, rel=1e-12)
    dz = es.cohen_d_paired(P1, P2)
    assert res.effect_sizes[0].name == "Cohen's d_z (paired)"
    assert res.effect_sizes[0].value == pytest.approx(d.mean() / d.std(ddof=1), rel=1e-12)
    assert res.effect_sizes[0].ci_low == pytest.approx(dz.ci_low)
    assert res.effect_sizes[1].name == "Hedges' g_z (paired)"
    # effectiveness of pairing: Pearson r with the usual two-sided P and Fisher-z CI
    r_ref = stats.pearsonr(P1, P2)
    corr = res.parameter("Correlation between pairs (Pearson r)")
    assert corr.value == pytest.approx(r_ref.statistic, rel=1e-10)
    assert corr.p_value == pytest.approx(r_ref.pvalue, rel=1e-8)
    ci = r_ref.confidence_interval(0.95)
    assert (corr.ci_low, corr.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-8)
    pairing = res.assumptions[1]
    assert pairing.status == "pass" and "effective" in pairing.detail
    assert pairing.p_value == pytest.approx(stats.pearsonr(P1, P2, alternative="greater").pvalue, rel=1e-8)
    assert res.assumptions[0].target == "Difference (before - after)"
    assert res.assumptions[0].p_value == pytest.approx(stats.shapiro(d).pvalue)
    assert [r["group"] for r in res.descriptives] == ["before", "after", "Difference (before - after)"]
    assert res.descriptives[2]["mean"] == pytest.approx(d.mean())
    assert res.extra["differences"] == pytest.approx(list(d)) and res.extra["n_pairs"] == 11


def test_paired_t_pairing_checks_and_degenerate_inputs():
    rng = np.random.default_rng(8)
    x = rng.normal(size=15)
    anti = _full(tg.ttest_paired(x, -x + rng.normal(0, 0.1, 15)))
    assert anti.assumptions[1].status == "warn" and "not positively correlated" in anti.assumptions[1].detail
    weak = None
    for seed in range(50):
        r2 = np.random.default_rng(seed)
        u, v = r2.normal(size=12), r2.normal(size=12)
        res = tg.ttest_paired(u, v)
        if res.assumptions[1].status == "info":
            weak = res
            break
    assert weak is not None and "weakly" in weak.assumptions[1].detail
    perfect = _full(tg.ttest_paired([1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 6.0, 10.0]))
    assert perfect.parameter("Correlation between pairs (Pearson r)").value == pytest.approx(1.0)

    const_diff = _full(tg.ttest_paired([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]))
    assert const_diff.statistic is None and const_diff.p_value is None
    assert "same difference" in const_diff.warnings[0] and "always -1" in const_diff.interpretation
    two = _full(tg.ttest_paired([1.0, 2.0], [3.0, 7.0]))
    assert any("Only 2 pairs" in w for w in two.warnings) and two.assumptions[1].status == "warn"
    zero = _full(tg.ttest_paired([1.0, 2.0, 4.0], [2.0, 1.0, 4.0]))
    assert zero.statistic == 0.0 and "exactly 0" in zero.interpretation
    with_nan = tg.ttest_paired([*P1, 1.0, np.nan], [*P2, np.nan, 2.0])
    assert with_nan.statistic == pytest.approx(tg.ttest_paired(P1, P2).statistic)
    assert "2 pair(s) with a missing value" in with_nan.data_notes[0]
    with pytest.raises(StatsError, match="same number"):
        tg.ttest_paired([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(StatsError, match="at least 2 complete pairs"):
        tg.ttest_paired([1.0, np.nan], [2.0, 3.0])


# ======================================================================== one sample


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_one_sample_t_matches_scipy_and_statsmodels(alternative):
    res = _full(tg.ttest_one_sample(A, 9.0, name="ctl", alternative=alternative))
    ref = stats.ttest_1samp(A, 9.0, alternative=alternative)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert res.df == A.size - 1
    t_sm, p_sm, _ = DescrStatsW(A).ttest_mean(9.0, alternative=SM_ALT[alternative])
    assert res.statistic == pytest.approx(t_sm, rel=1e-10) and res.p_value == pytest.approx(p_sm, rel=1e-8)
    ci = stats.ttest_1samp(A, 9.0).confidence_interval(0.95)
    mean_par = res.parameter("Mean of ctl")
    assert (mean_par.ci_low, mean_par.ci_high) == pytest.approx((ci.low, ci.high), rel=1e-9)
    diff_par = res.parameter("Difference from hypothetical value")
    assert diff_par.value == pytest.approx(A.mean() - 9.0)
    assert (diff_par.ci_low, diff_par.ci_high) == pytest.approx((ci.low - 9.0, ci.high - 9.0), rel=1e-9)
    assert res.effect("Cohen's d").value == pytest.approx((A.mean() - 9.0) / A.std(ddof=1))
    assert res.extra["hypothetical_value"] == 9.0


def test_one_sample_t_degenerate_inputs():
    const = _full(tg.ttest_one_sample([3.0, 3.0, 3.0], 1.0))
    assert const.statistic is None and "identical" in const.warnings[0]
    exact = _full(tg.ttest_one_sample([1.0, 2.0, 3.0], 2.0))
    assert exact.statistic == 0.0 and "exactly" in exact.interpretation
    assert any("Only 2 values" in w for w in tg.ttest_one_sample([1.0, 3.0]).warnings)
    default = tg.ttest_one_sample(A)
    assert default.statistic == pytest.approx(stats.ttest_1samp(A, 0.0).statistic)
    for bad in ("x", float("nan"), float("inf"), None):
        with pytest.raises(StatsError):
            tg.ttest_one_sample(A, bad)
    with pytest.raises(StatsError):
        tg.ttest_one_sample([1.0])


def test_one_sided_interpretations_name_the_direction():
    up = tg.ttest_unpaired(B, A, alternative="greater")
    assert up.p_value < 0.05 and "is higher than that of" in up.interpretation
    down = tg.ttest_paired(P1, P2, alternative="less")
    assert "systematically lower" in down.interpretation
    one = tg.ttest_one_sample(A, 9.0, alternative="greater")
    assert "higher than 9" in one.interpretation


# ====================================================================== Mann-Whitney


def _brute_mwu_pmf(n1, n2):
    counts = np.zeros(n1 * n2 + 1)
    for subset in itertools.combinations(range(1, n1 + n2 + 1), n1):
        counts[sum(subset) - n1 * (n1 + 1) // 2] += 1
    return counts / counts.sum()


@pytest.mark.parametrize("n1,n2", [(1, 1), (2, 3), (4, 4), (3, 7), (6, 5)])
def test_mann_whitney_null_distribution_vs_enumeration(n1, n2):
    pmf = tg.mann_whitney_null_pmf(n1, n2)
    assert pmf == pytest.approx(_brute_mwu_pmf(n1, n2), abs=1e-14)
    assert pmf.sum() == pytest.approx(1.0) and pmf == pytest.approx(pmf[::-1])  # symmetric


def test_mann_whitney_null_distribution_large_and_invalid():
    pmf = tg.mann_whitney_null_pmf(50, 50)
    assert pmf.size == 2501 and pmf.sum() == pytest.approx(1.0)
    mean = (np.arange(2501) * pmf).sum()
    var = ((np.arange(2501) - mean) ** 2 * pmf).sum()
    assert mean == pytest.approx(1250.0) and var == pytest.approx(50 * 50 * 101 / 12, rel=1e-10)
    with pytest.raises(ValueError):
        tg.mann_whitney_null_pmf(0, 3)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_exact_matches_scipy(alternative):
    res = _full(tg.mann_whitney(A, B, alternative=alternative))
    ref = stats.mannwhitneyu(A, B, alternative=alternative, method="exact")
    assert res.extra["u_a"] == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.extra["exact"] is True and res.extra["p_method"] == "exact"
    assert res.extra["u_a"] + res.extra["u_b"] == A.size * B.size
    assert res.statistic == min(res.extra["u_a"], res.extra["u_b"]) == res.extra["u_min"]
    # first principles: U_A counts the pairs with a > b
    assert res.extra["u_a"] == sum(1.0 for a in A for b in B if a > b)
    assert "exact P value" in res.interpretation and "exact P value" in res.methods_text


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_mann_whitney_asymptotic_with_ties_matches_scipy(alternative):
    res = _full(tg.mann_whitney(TIES_A, TIES_B, alternative=alternative))
    ref = stats.mannwhitneyu(TIES_A, TIES_B, alternative=alternative, method="asymptotic", use_continuity=True)
    assert res.extra["u_a"] == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.extra["exact"] is False and res.extra["ties"] is True
    assert any("tied values" in w for w in res.warnings)
    # forcing the asymptotic method on tie-free data also matches
    forced = tg.mann_whitney(A, B, alternative=alternative, method="asymptotic")
    ref2 = stats.mannwhitneyu(A, B, alternative=alternative, method="asymptotic", use_continuity=True)
    assert forced.p_value == pytest.approx(ref2.pvalue, rel=1e-9) and forced.extra["z"] is not None


def test_mann_whitney_asymptotic_first_principles():
    x, y = TIES_A, TIES_B
    n1, n2 = x.size, y.size
    n = n1 + n2
    u_a = sum((a > b) + 0.5 * (a == b) for a in x for b in y)
    _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1))))
    z = (abs(u_a - n1 * n2 / 2) - 0.5) / sigma
    res = tg.mann_whitney(x, y)
    assert res.extra["u_a"] == u_a
    assert res.p_value == pytest.approx(2 * stats.norm.sf(z), rel=1e-12)
    assert abs(res.extra["z"]) == pytest.approx(z, rel=1e-12) and res.extra["z"] < 0  # A tends to be smaller
    assert res.extra["probability_of_superiority"] == pytest.approx(u_a / (n1 * n2))
    rb = es.rank_biserial_independent(x, y)
    assert res.effect_sizes[0].name == "rank-biserial r"
    assert res.effect_sizes[0].value == pytest.approx(rb.value) == pytest.approx(2 * u_a / (n1 * n2) - 1)
    ranks = stats.rankdata(np.concatenate([x, y]))
    assert res.descriptives[0]["rank_sum"] == pytest.approx(ranks[:n1].sum())
    assert res.descriptives[1]["mean_rank"] == pytest.approx(ranks[n1:].mean())


def test_mann_whitney_method_selection():
    big_a, big_b = np.arange(60.0), np.arange(60.0) + 0.5
    auto = tg.mann_whitney(big_a, big_b)
    assert auto.extra["exact"] is False  # more than 50 per group -> normal approximation
    forced = tg.mann_whitney(big_a, big_b, method="exact")
    assert forced.extra["exact"] is False and any("exact P value was requested" in w for w in forced.warnings)
    forced_ties = tg.mann_whitney(TIES_A, TIES_B, method="exact")
    assert any("because there are tied values" in w for w in forced_ties.warnings)
    at_limit = tg.mann_whitney(np.arange(50.0), np.arange(50.0) + 0.5)
    assert at_limit.extra["exact"] is True
    assert at_limit.p_value == pytest.approx(
        stats.mannwhitneyu(np.arange(50.0), np.arange(50.0) + 0.5, method="exact").pvalue, rel=1e-8)
    with pytest.raises(StatsError):
        tg.mann_whitney(A, B, method="bootstrap")


def test_hodges_lehmann_shift_estimate_and_exact_ci():
    res = tg.mann_whitney(A, B, names=("ctl", "trt"))
    diffs = sorted(a - b for a in A for b in B)
    par = res.parameter("Hodges-Lehmann difference (ctl - trt)")
    assert par.value == pytest.approx(np.median(diffs), rel=1e-12)
    # exact CI by enumeration of the null distribution of U: largest k with P(U <= k - 1) <= 0.025
    cdf = np.cumsum(tg.mann_whitney_null_pmf(A.size, B.size))
    k = int(np.nonzero(cdf <= 0.025)[0][-1]) + 1
    assert (par.ci_low, par.ci_high) == pytest.approx((diffs[k - 1], diffs[len(diffs) - k]), rel=1e-12)
    assert res.extra["hodges_lehmann_ci_coverage"] == pytest.approx(1 - 2 * cdf[k - 1])
    assert res.extra["hodges_lehmann_ci_coverage"] >= 0.95
    assert "exact" in res.extra["hodges_lehmann_ci_method"]
    # classical table check: n1 = n2 = 10, two-sided 5% critical value of U is 23, so k = 24
    cdf10 = np.cumsum(tg.mann_whitney_null_pmf(10, 10))
    assert cdf10[23] <= 0.025 < cdf10[24]
    med = res.parameter("Difference between medians (ctl - trt)")
    assert med.value == pytest.approx(np.median(A) - np.median(B)) and med.ci_low is None
    # removing the estimated shift balances the two samples
    balanced = tg.mann_whitney(A - par.value, B)
    assert abs(balanced.effect_sizes[0].value) <= 2 / (A.size * B.size) + 1e-12


def test_hodges_lehmann_ci_is_dual_to_the_exact_test():
    """The exact CI excludes 0 exactly when the exact two-sided test has P <= alpha."""
    rng = np.random.default_rng(77)
    seen = set()
    for _ in range(150):
        n1, n2 = int(rng.integers(4, 11)), int(rng.integers(4, 11))
        x, y = rng.normal(0, 1, n1), rng.normal(rng.choice([0.0, 1.0, 2.0]), 1, n2)
        res = tg.mann_whitney(x, y)
        par = res.parameters[0]
        if par.ci_low is None:
            continue
        excluded = not (par.ci_low <= 0.0 <= par.ci_high)
        assert excluded == (res.p_value <= 0.05 + 1e-12)
        assert par.ci_low <= par.value <= par.ci_high
        seen.add(excluded)
    assert seen == {True, False}


def test_hodges_lehmann_ci_coverage_simulation():
    """Fixed-seed simulation under a pure location shift: the 95% intervals (exact, and the
    normal-approximation version used with larger samples) cover the true shift about 95%."""
    rng = np.random.default_rng(2024)
    hits_exact = hits_normal = 0
    n_sim = 300
    for _ in range(n_sim):
        x, y = rng.exponential(1.0, 9) + 1.5, rng.exponential(1.0, 8)
        par = tg.mann_whitney(x, y).parameters[0]
        hits_exact += par.ci_low <= 1.5 <= par.ci_high
        par2 = tg.mann_whitney(x, y, method="asymptotic").parameters[0]
        hits_normal += par2.ci_low <= 1.5 <= par2.ci_high
    assert 0.93 <= hits_exact / n_sim <= 0.99
    assert 0.92 <= hits_normal / n_sim <= 0.99


def test_mann_whitney_normal_approximation_ci_rank():
    res = tg.mann_whitney(TIES_A, TIES_B)
    n1, n2 = TIES_A.size, TIES_B.size
    diffs = sorted(a - b for a in TIES_A for b in TIES_B)
    k = math.floor(n1 * n2 / 2 + 0.5 - stats.norm.ppf(0.975) * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12))
    par = res.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((diffs[k - 1], diffs[len(diffs) - k]))
    assert "normal approximation" in res.extra["hodges_lehmann_ci_method"]
    assert res.extra["hodges_lehmann_ci_coverage"] is None


def test_mann_whitney_wording_assumptions_and_edge_cases():
    res = tg.mann_whitney(A, B, names=("ctl", "trt"))
    assert "compares the distributions" in res.interpretation and "same shape" in res.interpretation
    assert "medians only when" in res.why_chosen
    assert [a.status for a in res.assumptions[:2]] == ["info", "info"]
    assert res.assumptions[2].name.startswith("Similar distribution shape")
    rng = np.random.default_rng(4)
    spread = tg.mann_whitney(rng.normal(0, 1, 30), rng.normal(0, 8, 30))
    assert spread.assumptions[2].status == "warn" and "do not interpret" in spread.assumptions[2].detail

    tiny = _full(tg.mann_whitney([1.0, 2.0], [3.0, 4.0]))
    assert tiny.p_value == pytest.approx(2 / 6) and tiny.statistic == 0.0
    assert any("never reach significance" in w for w in tiny.warnings)
    assert tiny.parameters[0].ci_low is None and any("too small for a 95%" in w for w in tiny.warnings)
    assert tiny.assumptions[2].status == "info"
    identical = _full(tg.mann_whitney([3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]))
    assert identical.p_value == 1.0 and identical.extra["p_method"] == "all values identical"
    assert "evenly interleaved" in identical.interpretation
    larger = tg.mann_whitney(B, A)
    assert "tended to be larger" in larger.interpretation
    assert larger.statistic == tg.mann_whitney(A, B).statistic  # U (the smaller one) ignores the order
    assert larger.p_value == pytest.approx(tg.mann_whitney(A, B).p_value)
    with_nan = tg.mann_whitney([*A, np.nan], B)
    assert with_nan.p_value == pytest.approx(res.p_value) and with_nan.data_notes
    for bad in (([1.0], [1.0, 2.0]), ([1.0, 2.0], []), ([np.nan, 1.0], [1.0, 2.0])):
        with pytest.raises(StatsError):
            tg.mann_whitney(*bad)


def test_large_samples_use_the_normal_approximation_without_overflow():
    rng = np.random.default_rng(31)
    x, y = rng.normal(0.0, 1.0, 700), rng.normal(0.15, 1.0, 650)
    res = _full(tg.mann_whitney(x, y))
    ref = stats.mannwhitneyu(x, y, method="asymptotic", use_continuity=True)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9) and res.extra["exact"] is False
    par = res.parameters[0]
    assert par.ci_low < par.value < par.ci_high and par.value == pytest.approx(-0.15, abs=0.15)
    paired = _full(tg.wilcoxon_signed_rank(x[:650], y))
    ref2 = stats.wilcoxon(x[:650], y, correction=True, method="asymptotic")
    assert paired.p_value == pytest.approx(ref2.pvalue, rel=1e-9)
    big = rng.normal(0.0, 1.0, 5200)
    t = _full(tg.ttest_unpaired(big, big[::-1] + 0.01))  # > 5000 values: no warning leaks out
    assert t.assumptions[0].p_value is not None


def test_mann_whitney_too_many_pairs_skips_estimate(monkeypatch):
    monkeypatch.setattr(tg, "_MAX_PAIRS", 10)
    res = _full(tg.mann_whitney(A, B))
    assert res.parameters[0].value is None and any("too large" in w for w in res.warnings)
    sr = _full(tg.wilcoxon_signed_rank(P1, P2))
    assert sr.parameters[0].value is None and any("too many values" in w for w in sr.warnings)


# ============================================================== Wilcoxon signed-rank


def _brute_signed_rank_pmf(ranks):
    total = sum(ranks)
    counts = np.zeros(total + 1)
    for signs in itertools.product((0, 1), repeat=len(ranks)):
        counts[sum(r for r, s in zip(ranks, signs) if s)] += 1
    return counts / counts.sum()


@pytest.mark.parametrize("ranks", [[1], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6], [2, 5, 9, 10]])
def test_signed_rank_null_distribution_vs_enumeration(ranks):
    assert tg.signed_rank_null_pmf(ranks) == pytest.approx(_brute_signed_rank_pmf(ranks), abs=1e-14)


def test_signed_rank_null_distribution_moments():
    n = 50
    pmf = tg.signed_rank_null_pmf(range(1, n + 1))
    w = np.arange(pmf.size)
    mean = (w * pmf).sum()
    assert pmf.sum() == pytest.approx(1.0) and mean == pytest.approx(n * (n + 1) / 4)
    assert ((w - mean) ** 2 * pmf).sum() == pytest.approx(n * (n + 1) * (2 * n + 1) / 24, rel=1e-10)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_wilcoxon_exact_matches_scipy(alternative):
    res = _full(tg.wilcoxon_signed_rank(P1, P2, alternative=alternative))
    ref = stats.wilcoxon(P1, P2, alternative=alternative, method="exact")
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.extra["exact"] is True and res.extra["p_method"] == "exact"
    d = P1 - P2
    ranks = stats.rankdata(np.abs(d))
    w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    assert (res.extra["w_plus"], res.extra["w_minus"]) == (w_plus, w_minus)
    assert res.statistic == w_plus - w_minus and res.extra["t_min"] == min(w_plus, w_minus)
    # scipy reports min(W+, W-) for a two-sided test and W+ otherwise
    assert ref.statistic == (res.extra["t_min"] if alternative == "two-sided" else res.extra["w_plus"])
    assert res.extra["w_plus"] + res.extra["w_minus"] == d.size * (d.size + 1) / 2


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
def test_wilcoxon_asymptotic_with_ties_and_zeros_matches_scipy(alternative, zero_method):
    x = np.array([5.0, 7.0, 3.0, 8.0, 6.0, 9.0, 4.0, 7.0, 5.0, 6.0, 8.0, 10.0])
    y = np.array([4.0, 5.0, 3.0, 9.0, 4.0, 6.0, 4.0, 5.0, 6.0, 3.0, 5.0, 7.0])
    res = _full(tg.wilcoxon_signed_rank(x, y, alternative=alternative, zero_method=zero_method))
    ref = stats.wilcoxon(x, y, alternative=alternative, zero_method=zero_method, correction=True,
                         method="asymptotic")
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.extra["exact"] is False and res.extra["n_zero"] == 2 and res.extra["zero_method"] == zero_method
    assert any("tied in size" in w for w in res.warnings)
    assert any("exactly zero" in w for w in res.warnings)
    assert ("Pratt" in " ".join(res.citations)) == (zero_method == "pratt")
    assert ("Pratt" in res.methods_text) == (zero_method == "pratt")


def test_wilcoxon_asymptotic_first_principles_wilcox_and_pratt():
    x = np.array([5.0, 7.0, 3.0, 8.0, 6.0, 9.0, 4.0, 7.0, 5.0, 6.0, 8.0, 10.0])
    y = np.array([4.0, 5.0, 3.0, 9.0, 4.0, 6.0, 4.0, 5.0, 6.0, 3.0, 5.0, 7.0])
    d = x - y
    nz = d[d != 0]
    # Wilcoxon: drop zeros, rank the rest
    ranks = stats.rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    n = nz.size
    _, t = np.unique(np.abs(nz), return_counts=True)
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (t**3 - t).sum() / 48)
    z = (abs(w_plus - n * (n + 1) / 4) - 0.5) / sd
    res = tg.wilcoxon_signed_rank(x, y)
    assert res.extra["w_plus"] == w_plus and res.p_value == pytest.approx(2 * stats.norm.sf(z), rel=1e-12)
    # Pratt: rank all |d| including zeros, then drop the zero ranks
    ranks_all = stats.rankdata(np.abs(d))
    w_plus_p = ranks_all[d > 0].sum()
    big_n, zeros = d.size, int((d == 0).sum())
    mean_p = (big_n * (big_n + 1) - zeros * (zeros + 1)) / 4
    var_p = (big_n * (big_n + 1) * (2 * big_n + 1) - zeros * (zeros + 1) * (2 * zeros + 1)) / 24 - (t**3 - t).sum() / 48
    z_p = (abs(w_plus_p - mean_p) - 0.5) / math.sqrt(var_p)
    pratt = tg.wilcoxon_signed_rank(x, y, zero_method="pratt")
    assert pratt.extra["w_plus"] == w_plus_p
    assert pratt.p_value == pytest.approx(2 * stats.norm.sf(z_p), rel=1e-12)
    assert pratt.p_value > res.p_value  # Pratt's handling is the more conservative here


def test_wilcoxon_exact_with_zeros_wilcox_and_pratt_vs_enumeration():
    d = np.array([0.0, 1.5, -0.7, 2.2, 3.1, 0.0, -0.4, 4.0, 2.9])
    x, y = d + 10.0, np.full(d.size, 10.0)
    nz = d[d != 0]
    # Wilcoxon's method: exact test on the non-zero differences (scipy agrees)
    res = tg.wilcoxon_signed_rank(x, y)
    ref = stats.wilcoxon(nz, method="exact")
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.extra["p_method"] == "exact (zero differences discarded)"
    # Pratt's method: non-zero ranks are 3..9; enumerate all sign patterns
    ranks = stats.rankdata(np.abs(d))[d != 0]
    assert sorted(ranks) == [3, 4, 5, 6, 7, 8, 9]
    w_plus = ranks[nz > 0].sum()
    pmf = _brute_signed_rank_pmf([int(r) for r in ranks])
    p = min(1.0, 2 * min(pmf[: int(w_plus) + 1].sum(), pmf[int(w_plus):].sum()))
    pratt = tg.wilcoxon_signed_rank(x, y, zero_method="pratt")
    assert pratt.extra["w_plus"] == w_plus and pratt.p_value == pytest.approx(p, rel=1e-12)
    assert "Pratt" in pratt.extra["p_method"] and pratt.extra["exact"] is True
    for alternative in ("greater", "less"):
        one = tg.wilcoxon_signed_rank(x, y, zero_method="pratt", alternative=alternative)
        expect = pmf[int(w_plus):].sum() if alternative == "greater" else pmf[: int(w_plus) + 1].sum()
        assert one.p_value == pytest.approx(expect, rel=1e-12)


def test_wilcoxon_pseudo_median_and_exact_ci():
    res = tg.wilcoxon_signed_rank(P1, P2, names=("before", "after"))
    d = P1 - P2
    walsh = sorted((d[i] + d[j]) / 2 for i in range(d.size) for j in range(i, d.size))
    par = res.parameter("Hodges-Lehmann median difference (before - after)")
    assert par.value == pytest.approx(np.median(walsh), rel=1e-12)
    cdf = np.cumsum(_brute_signed_rank_pmf(list(range(1, d.size + 1))))
    k = int(np.nonzero(cdf <= 0.025)[0][-1]) + 1
    assert (par.ci_low, par.ci_high) == pytest.approx((walsh[k - 1], walsh[len(walsh) - k]), rel=1e-12)
    assert res.extra["hodges_lehmann_ci_coverage"] == pytest.approx(1 - 2 * cdf[k - 1])
    # classical table: n = 11, two-sided 5% critical value of T is 10, so k = 11
    assert k == 11
    assert res.parameter("Median of differences").value == pytest.approx(np.median(d))
    rho = stats.spearmanr(P1, P2)
    sp = res.parameter("Correlation between pairs (Spearman r)")
    assert sp.value == pytest.approx(rho.statistic, rel=1e-10) and sp.p_value == pytest.approx(rho.pvalue, rel=1e-8)
    rb = es.rank_biserial_paired(d)
    assert res.effect_sizes[0].value == pytest.approx(rb.value)
    assert res.effect_sizes[0].value == pytest.approx(res.statistic / (d.size * (d.size + 1) / 2))
    assert res.assumptions[0].name == "Symmetric distribution of the differences"
    assert res.assumptions[0].status == "info" and "skewness" in res.assumptions[0].detail


def test_wilcoxon_ci_is_dual_to_the_exact_test_and_covers():
    rng = np.random.default_rng(5)
    seen = set()
    hits = 0
    n_sim = 200
    for _ in range(n_sim):
        n = int(rng.integers(7, 15))
        d = rng.normal(0.8, 1.0, n)
        res = tg.wilcoxon_one_sample(d, 0.0)
        par = res.parameters[0]
        excluded = not (par.ci_low <= 0.0 <= par.ci_high)
        assert excluded == (res.p_value <= 0.05 + 1e-12)
        assert par.ci_low <= par.value <= par.ci_high
        seen.add(excluded)
        hits += par.ci_low <= 0.8 <= par.ci_high
    assert seen == {True, False}
    assert 0.92 <= hits / n_sim <= 0.99


def test_wilcoxon_normal_approximation_ci_rank_with_ties():
    d = np.array([1.0, 2.0, 2.0, -1.0, 3.0, 4.0, 4.0, -2.0, 5.0, 6.0, 1.0, 3.0])
    res = tg.wilcoxon_one_sample(d, 0.0)
    n = d.size
    walsh = sorted((d[i] + d[j]) / 2 for i in range(n) for j in range(i, n))
    k = math.floor(n * (n + 1) / 4 + 0.5 - stats.norm.ppf(0.975) * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))
    par = res.parameters[0]
    assert (par.ci_low, par.ci_high) == pytest.approx((walsh[k - 1], walsh[len(walsh) - k]))
    assert "normal approximation" in res.extra["hodges_lehmann_ci_method"]
    assert res.extra["hodges_lehmann_ci_coverage"] is None


@pytest.mark.parametrize("alternative", ALTERNATIVES)
def test_wilcoxon_one_sample_matches_scipy_and_paired(alternative):
    m0 = 9.25
    res = _full(tg.wilcoxon_one_sample(A, m0, name="ctl", alternative=alternative))
    ref = stats.wilcoxon(A - m0, alternative=alternative, method="exact")
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    paired = tg.wilcoxon_signed_rank(A, np.full(A.size, m0), alternative=alternative)
    assert res.statistic == paired.statistic and res.p_value == pytest.approx(paired.p_value)
    # the reported estimate is the pseudo-median of the sample itself (not of x - m0)
    walsh = [(A[i] + A[j]) / 2 for i in range(A.size) for j in range(i, A.size)]
    par = res.parameter("Hodges-Lehmann pseudo-median of ctl")
    assert par.value == pytest.approx(np.median(walsh), rel=1e-12)
    assert par.ci_low == pytest.approx(paired.parameters[0].ci_low + m0)
    assert res.parameter("Median of ctl").value == pytest.approx(np.median(A))
    assert res.parameter("Median minus hypothetical value").value == pytest.approx(np.median(A) - m0)
    assert res.extra["hypothetical_value"] == m0 and res.n == {"ctl": 12}


def test_wilcoxon_method_selection_and_degenerate_inputs():
    rng = np.random.default_rng(12)
    big = rng.normal(0.2, 1.0, 60)
    auto = tg.wilcoxon_one_sample(big)
    assert auto.extra["exact"] is False
    assert auto.p_value == pytest.approx(
        stats.wilcoxon(big, correction=True, method="asymptotic").pvalue, rel=1e-9)
    forced = tg.wilcoxon_one_sample(big, method="exact")
    assert any("exact P value was requested" in w for w in forced.warnings)
    forced_ties = tg.wilcoxon_one_sample([1.0, 1.0, 2.0, -3.0, 4.0, 5.0], method="exact")
    assert any("tied in size" in w for w in forced_ties.warnings)
    asym = tg.wilcoxon_signed_rank(P1, P2, method="asymptotic")
    assert asym.extra["exact"] is False and asym.extra["z"] is not None
    assert asym.p_value == pytest.approx(stats.wilcoxon(P1, P2, correction=True, method="asymptotic").pvalue, rel=1e-9)
    assert "exact" in asym.extra["hodges_lehmann_ci_method"]  # the CI stays exact when it can be

    all_zero = _full(tg.wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert all_zero.statistic is None and all_zero.p_value is None
    assert "exactly zero" in all_zero.warnings[0] and "cannot be computed" in all_zero.interpretation
    assert all_zero.effect_sizes[0].value is None
    assert _full(tg.wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], zero_method="pratt")).p_value is None
    few = _full(tg.wilcoxon_signed_rank([1.0, 2.0, 5.0], [0.5, 2.0, 3.0]))
    assert few.p_value == pytest.approx(0.5) and any("cannot reach significance" in w for w in few.warnings)
    assert few.parameters[0].ci_low is None and any("too few values" in w for w in few.warnings)
    balanced = _full(tg.wilcoxon_one_sample([-2.0, -1.0, 1.0, 2.0]))
    assert balanced.statistic == 0.0 and "balanced out" in balanced.interpretation
    assert "tended to be larger" in tg.wilcoxon_signed_rank(P2, P1).interpretation
    for kw in ({"zero_method": "zsplit"}, {"method": "mc"}, {"alternative": "x"}, {"alpha": 2}):
        with pytest.raises(StatsError):
            tg.wilcoxon_signed_rank(P1, P2, **kw)
    with pytest.raises(StatsError):
        tg.wilcoxon_signed_rank([1.0], [2.0])
    with pytest.raises(StatsError):
        tg.wilcoxon_one_sample([1.0])
    with pytest.raises(StatsError):
        tg.wilcoxon_one_sample(A, "abc")


def _brute_rank_sum_p(a, b, alternative):
    """Exact conditional (permutation) P value of the rank sum with ties, by enumerating
    every split of the pooled values into groups of the observed sizes."""
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    n1 = len(a)
    observed = ranks[:n1].sum()
    sums = np.array([ranks[list(idx)].sum() for idx in itertools.combinations(range(pooled.size), n1)])
    upper = np.mean(sums >= observed - 1e-9)
    lower = np.mean(sums <= observed + 1e-9)
    if alternative == "greater":
        return float(upper)
    if alternative == "less":
        return float(lower)
    return float(min(1.0, 2.0 * min(upper, lower)))


def _brute_signed_rank_p(d, alternative):
    """Exact conditional P value of W+ with tied |d| (zeros discarded), by enumerating
    every pattern of signs of the observed mid-ranks."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    ranks = stats.rankdata(np.abs(d))
    observed = ranks[d > 0].sum()
    totals = np.array([sum(r for r, s in zip(ranks, signs) if s)
                       for signs in itertools.product((0, 1), repeat=d.size)])
    upper = np.mean(totals >= observed - 1e-9)
    lower = np.mean(totals <= observed + 1e-9)
    if alternative == "greater":
        return float(upper)
    if alternative == "less":
        return float(lower)
    return float(min(1.0, 2.0 * min(upper, lower)))


@pytest.mark.parametrize("alt", ALTERNATIVES)
def test_rank_tests_exact_p_with_ties_equals_brute_force_enumeration(alt):
    """Small tied samples get the exact conditional P value (permutation distribution of
    the mid-ranks); it must equal brute-force enumeration."""
    cases = [([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]), ([1.0, 2.0, 2.0, 3.0], [2.0, 3.0, 3.0, 5.0, 5.0]),
             ([0.0, 1.0, 1.0, 1.0, 4.0, 4.0, 6.0], [1.0, 4.0, 5.0, 5.0, 6.0, 7.0, 7.0])]
    for a, b in cases:
        res = tg.mann_whitney(a, b, alternative=alt)
        assert res.extra["exact"] and res.extra["ties"]
        assert res.p_value == pytest.approx(_brute_rank_sum_p(np.array(a), np.array(b), alt), rel=1e-12)
    for d in ([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 2.0, 2.0, 3.0, 0.0], [2.0, 2.0, -2.0, 3.0, 5.0, 5.0, 5.0]):
        res = tg.wilcoxon_one_sample(d, 0.0, alternative=alt)
        assert res.extra["exact"] and res.extra["ties"]
        assert res.p_value == pytest.approx(_brute_signed_rank_p(d, alt), rel=1e-12)
    # the five tied differences of +1: exact two-sided P = 2 / 2^5
    assert tg.wilcoxon_one_sample([1.0] * 5, 0.0).p_value == pytest.approx(0.0625)


def test_rank_tests_with_random_ties_and_zeros_match_scipy():
    """Randomised cross-check (fixed seed) on small integer data full of ties and zeros:
    every P value equals scipy's whenever both use the same method (exact without ties, or
    normal approximation with tie + continuity correction), for all alternatives and both
    zero-handling methods. Small tied samples get the exact conditional P value instead
    (``extra["p_method"]`` says so), which scipy does not compute; those are compared with
    brute-force enumeration."""
    rng = np.random.default_rng(0)
    compared = enumerated = 0
    for _ in range(400):
        n1, n2 = rng.integers(2, 12, 2)
        a = rng.integers(0, 6, n1).astype(float)
        b = rng.integers(0, 6, n2).astype(float)
        alt = str(rng.choice(ALTERNATIVES))
        res = tg.mann_whitney(a, b, alternative=alt)
        if res.extra["p_method"] == "all values identical":
            assert res.p_value == 1.0
            continue
        if res.extra["exact"] and res.extra["ties"]:
            assert res.extra["p_method"] == tg._P_EXACT_TIES
            assert res.p_value == pytest.approx(_brute_rank_sum_p(a, b, alt), rel=1e-9, abs=1e-12)
            enumerated += 1
            continue
        ref = stats.mannwhitneyu(a, b, alternative=alt, use_continuity=True,
                                 method="exact" if res.extra["exact"] else "asymptotic")
        assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)
        compared += 1
    assert compared > 200 and enumerated > 30
    compared = enumerated = 0
    for _ in range(400):
        d = rng.integers(-5, 6, int(rng.integers(2, 14))).astype(float)
        alt = str(rng.choice(ALTERNATIVES))
        zero_method = str(rng.choice(["wilcox", "pratt"]))
        res = tg.wilcoxon_one_sample(d, 0.0, alternative=alt, zero_method=zero_method)
        if res.p_value is None:
            assert not np.any(d != 0)
            continue
        if res.extra["exact"]:
            if zero_method == "pratt" and res.extra["n_zero"]:
                continue  # exact Pratt test: checked against enumeration above
            if res.extra["ties"]:
                assert res.extra["p_method"].startswith(tg._P_EXACT_TIES)
                assert res.p_value == pytest.approx(_brute_signed_rank_p(d, alt), rel=1e-9, abs=1e-12)
                enumerated += 1
                continue
            ref = stats.wilcoxon(d[d != 0], alternative=alt, method="exact")
        else:
            ref = stats.wilcoxon(d, alternative=alt, zero_method=zero_method, correction=True,
                                 method="asymptotic")
        assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9, abs=1e-12)
        compared += 1
    assert compared > 150 and enumerated > 30


# ----------------------------------------------------------------------- registry

TABLE = ColumnTable.from_dict("study", {
    "Control": [*P1, 4.4, np.nan],
    "Treated": [*P2, np.nan, 6.1],
    "Other": [*(P1 * 2), 1.0, 2.0],
})
ALL_KEYS = ("ttest_unpaired", "ttest_welch", "ttest_paired", "ttest_one_sample", "mann_whitney",
            "wilcoxon_signed_rank", "wilcoxon_one_sample")


@pytest.mark.parametrize("key", ALL_KEYS)
def test_registry_defaults(key):
    info = registry.REGISTRY[key]
    assert info.category == "Compare two groups" and info.tables == ("column",)
    assert info.description and info.citation and all(p.help for p in info.params)
    res = run_analysis(TABLE, key)
    assert res.analysis == key and res.table == "study"
    assert res.interpretation and res.why_chosen and len(res.assumptions) >= 1
    assert res.significant is not None and res.params["alpha"] == 0.05
    json.loads(res.model_dump_json())
    one_sample = key in {"ttest_one_sample", "wilcoxon_one_sample"}
    assert list(res.n) == (["Control"] if one_sample else ["Control", "Treated"])
    if one_sample:
        assert res.params["hypothetical_value"] == 0.0 and res.params["group"] is None
    else:
        assert res.params["group_a"] is None


def test_registry_unpaired_uses_all_values_paired_uses_complete_rows():
    unpaired = run_analysis(TABLE, "ttest_unpaired")
    assert unpaired.n == {"Control": 12, "Treated": 12}
    ref = stats.ttest_ind([*P1, 4.4], [*P2, 6.1])
    assert unpaired.statistic == pytest.approx(ref.statistic)
    for key, func in (("ttest_paired", tg.ttest_paired), ("wilcoxon_signed_rank", tg.wilcoxon_signed_rank)):
        res = run_analysis(TABLE, key)
        assert res.n == {"Control": 11, "Treated": 11}
        assert res.statistic == pytest.approx(func(P1, P2).statistic)
        assert res.data_notes[0].startswith("2 incomplete pair(s)") and "11 complete pair(s)" in res.data_notes[0]
    one_missing = run_analysis(TABLE, "ttest_paired", group_a="Control", group_b="Other")
    assert one_missing.data_notes[0].startswith("1 incomplete pair(s)") and one_missing.n["Other"] == 12
    full = ColumnTable.from_dict("full", {"Before": P1, "After": P2})
    complete = run_analysis(full, "ttest_paired")
    assert complete.data_notes[0] == "All 11 pair(s) were complete; no incomplete pairs were dropped."
    assert complete.n == {"Before": 11, "After": 11}


def test_registry_group_selection_and_options():
    res = run_analysis(TABLE, "ttest_welch", group_a="Treated", group_b="Control", alternative="greater",
                       alpha=0.01, ci_level=0.9)
    assert list(res.n) == ["Treated", "Control"] and res.alternative == "greater" and res.alpha == 0.01
    assert res.parameters[0].ci_level == 0.9 and res.parameters[0].name == "Mean difference (Treated - Control)"
    only_b = run_analysis(TABLE, "mann_whitney", group_b="Control")  # A defaults to the first other column
    assert list(only_b.n) == ["Treated", "Control"]
    only_a = run_analysis(TABLE, "mann_whitney", group_a="Other")
    assert list(only_a.n) == ["Other", "Control"]
    one = run_analysis(TABLE, "ttest_one_sample", group="Treated", hypothetical_value=5.0)
    assert one.statistic == pytest.approx(stats.ttest_1samp([*P2, 6.1], 5.0).statistic)
    w1 = run_analysis(TABLE, "wilcoxon_one_sample", group="Treated", hypothetical_value=5.0, zero_method="pratt")
    assert w1.extra["zero_method"] == "pratt" and w1.extra["hypothetical_value"] == 5.0
    wp = run_analysis(TABLE, "wilcoxon_signed_rank", zero_method="pratt")
    assert wp.extra["zero_method"] == "pratt"
    TABLE2 = ColumnTable.from_dict("x", {"Control": [1.0, 2.0, 3.0, 4.0, 50.0], "Treated": [2.0, 3.0, 4.0, 5.0, 6.0]})
    TABLE2.exclude([(4, "Control")])
    excluded = run_analysis(TABLE2, "ttest_unpaired")
    assert excluded.n["Control"] == 4 and any("explicitly excluded" in n for n in excluded.data_notes)
    paired_excl = run_analysis(TABLE2, "ttest_paired")
    assert paired_excl.n["Control"] == 4 and paired_excl.data_notes[0].startswith("1 incomplete pair(s)")


def test_registry_errors():
    single = ColumnTable.from_dict("one", {"Only": [1.0, 2.0, 3.0]})
    for key in ("ttest_unpaired", "ttest_welch", "ttest_paired", "mann_whitney", "wilcoxon_signed_rank"):
        with pytest.raises(StatsError, match="only one column"):
            run_analysis(single, key)
        with pytest.raises(StatsError, match="Unknown group"):
            run_analysis(TABLE, key, group_a="Nope")
        with pytest.raises(StatsError, match="two different columns"):
            run_analysis(TABLE, key, group_a="Control", group_b="Control")
        with pytest.raises(StatsError, match="Unknown option"):
            run_analysis(TABLE, key, bogus=1)
    for key in ("ttest_one_sample", "wilcoxon_one_sample"):
        assert run_analysis(single, key).n == {"Only": 3}
        with pytest.raises(StatsError, match="Unknown group"):
            run_analysis(TABLE, key, group="Nope")
        with pytest.raises(StatsError):
            run_analysis(TABLE, key, hypothetical_value="abc")
    empty = ColumnTable.from_dict("e", {"A": [np.nan, 1.0], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(StatsError):
        run_analysis(empty, "ttest_unpaired")
    with pytest.raises(StatsError):
        run_analysis(empty, "ttest_paired")


# ------------------------------------------------------------------ property tests

_value = st.one_of(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    st.integers(min_value=-3, max_value=3).map(float),
    st.just(float("nan")),
)
_sample = st.lists(_value, min_size=0, max_size=14)
_alt = st.sampled_from(ALTERNATIVES)


def _finite(values):
    return [v for v in values if not math.isnan(v)]


def _check_result(res):
    res = res.finalize()
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    assert (res.p_value is None) == (res.statistic is None)
    if res.p_value is None:
        assert res.warnings
    for par in res.parameters:
        if par.ci_low is not None and par.ci_high is not None and par.value is not None:
            assert par.ci_low <= par.value + 1e-9 * (1 + abs(par.value))
            assert par.value <= par.ci_high + 1e-9 * (1 + abs(par.value))
    for effect in res.effect_sizes:
        assert effect.value is None or math.isfinite(effect.value)
    assert res.interpretation and res.assumptions
    json.loads(res.model_dump_json())
    return res


@settings(max_examples=60, deadline=None)
@given(_sample, _sample, _alt)
def test_independent_tests_never_crash(a, b, alternative):
    ok = len(_finite(a)) >= 2 and len(_finite(b)) >= 2
    for func in (tg.ttest_unpaired, tg.ttest_welch, tg.mann_whitney):
        if not ok:
            with pytest.raises(StatsError):
                func(a, b, alternative=alternative)
            continue
        res = _check_result(func(a, b, alternative=alternative))
        assert sum(res.n.values()) == len(_finite(a)) + len(_finite(b))
        if func is tg.mann_whitney:
            assert res.p_value is not None
            assert res.extra["u_a"] + res.extra["u_b"] == pytest.approx(len(_finite(a)) * len(_finite(b)))
            assert -1 - 1e-12 <= res.effect_sizes[0].value <= 1 + 1e-12


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(_value, _value), min_size=0, max_size=14), _alt, st.sampled_from(["wilcox", "pratt"]))
def test_paired_tests_never_crash(pairs, alternative, zero_method):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    complete = sum(1 for x, y in pairs if not (math.isnan(x) or math.isnan(y)))
    calls = (
        lambda: tg.ttest_paired(a, b, alternative=alternative),
        lambda: tg.wilcoxon_signed_rank(a, b, alternative=alternative, zero_method=zero_method),
    )
    for call in calls:
        if complete < 2:
            with pytest.raises(StatsError):
                call()
            continue
        res = _check_result(call())
        assert res.n_total == complete


@settings(max_examples=60, deadline=None)
@given(_sample, st.floats(min_value=-10, max_value=10), _alt, st.sampled_from(["wilcox", "pratt"]))
def test_one_sample_tests_never_crash(x, mu, alternative, zero_method):
    calls = (
        lambda: tg.ttest_one_sample(x, mu, alternative=alternative),
        lambda: tg.wilcoxon_one_sample(x, mu, alternative=alternative, zero_method=zero_method),
    )
    for call in calls:
        if len(_finite(x)) < 2:
            with pytest.raises(StatsError):
                call()
            continue
        _check_result(call())


_distinct = st.lists(st.integers(min_value=-200, max_value=200), min_size=6, max_size=24, unique=True)


@settings(max_examples=50, deadline=None)
@given(_distinct, st.integers(min_value=3, max_value=12), st.floats(min_value=-50, max_value=50),
       st.floats(min_value=0.1, max_value=20))
def test_swap_and_location_scale_invariance(values, split, shift, scale):
    split = min(split, len(values) - 3)
    a = np.array(values[:split], dtype=float)
    b = np.array(values[split:], dtype=float)
    for func in (tg.ttest_unpaired, tg.ttest_welch):
        r, swapped = func(a, b), func(b, a)
        assert swapped.statistic == pytest.approx(-r.statistic, rel=1e-9, abs=1e-12)
        assert swapped.p_value == pytest.approx(r.p_value, rel=1e-9, abs=1e-15)
        assert swapped.parameters[0].ci_low == pytest.approx(-r.parameters[0].ci_high, rel=1e-9, abs=1e-9)
        moved = func(a * scale + shift, b * scale + shift)
        assert moved.statistic == pytest.approx(r.statistic, rel=1e-6, abs=1e-9)
        assert moved.parameters[0].value == pytest.approx(r.parameters[0].value * scale, rel=1e-6, abs=1e-6)
        # one-sided P values are complementary
        assert func(a, b, alternative="greater").p_value + func(a, b, alternative="less").p_value == pytest.approx(1.0)
    r, swapped = tg.mann_whitney(a, b), tg.mann_whitney(b, a)
    assert swapped.statistic == r.statistic and swapped.p_value == pytest.approx(r.p_value, rel=1e-9)
    assert swapped.extra["u_a"] == r.extra["u_b"]
    assert swapped.effect_sizes[0].value == pytest.approx(-r.effect_sizes[0].value, abs=1e-12)
    assert swapped.parameters[0].value == pytest.approx(-r.parameters[0].value, abs=1e-9)
    # rank tests are unchanged by any increasing transformation of the data
    cubed = tg.mann_whitney(a**3, b**3)
    assert cubed.statistic == r.statistic and cubed.p_value == pytest.approx(r.p_value, rel=1e-12)
    # exact two-sided P agrees with scipy for every tie-free configuration
    assert r.p_value == pytest.approx(stats.mannwhitneyu(a, b, method="exact").pvalue, rel=1e-9)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(min_value=-300, max_value=300).filter(lambda v: v != 0), min_size=4, max_size=20,
                unique_by=abs),
       st.floats(min_value=0.1, max_value=20))
def test_signed_rank_properties(diffs, scale):
    d = np.array(diffs, dtype=float)
    base = np.linspace(1.0, 2.0, d.size)
    r = tg.wilcoxon_signed_rank(base + d, base)
    assert r.extra["exact"] is True
    assert r.p_value == pytest.approx(stats.wilcoxon(d, method="exact").pvalue, rel=1e-9)
    swapped = tg.wilcoxon_signed_rank(base, base + d)
    assert swapped.statistic == -r.statistic and swapped.p_value == pytest.approx(r.p_value, rel=1e-12)
    assert swapped.parameters[0].value == pytest.approx(-r.parameters[0].value, abs=1e-9)
    scaled = tg.wilcoxon_one_sample(d * scale, 0.0)
    assert scaled.statistic == r.statistic and scaled.p_value == pytest.approx(r.p_value, rel=1e-12)
    assert scaled.parameters[0].value == pytest.approx(r.parameters[0].value * scale, rel=1e-6, abs=1e-9)
    t = tg.ttest_paired(base + d, base)
    t1 = tg.ttest_one_sample(d, 0.0)
    assert (t.statistic is None) == (t1.statistic is None)
    if t.statistic is not None:
        assert t.statistic == pytest.approx(t1.statistic, rel=1e-6, abs=1e-9)
        assert tg.ttest_paired(base, base + d).statistic == pytest.approx(-t.statistic, rel=1e-6, abs=1e-9)
