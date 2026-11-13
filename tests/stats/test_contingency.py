"""Contingency tables: chi-square, Fisher's exact test, trend test and 2x2 measures.

Validation layers (sources in tests/fixtures/sources/survival_contingency_nested.md):
* published worked examples (Fisher's tea-tasting table, the aspirin / myocardial
  infarction table, Newcombe's risk-difference example, two textbook tables);
* cross-implementation: TorusPlot vs scipy vs statsmodels;
* from-first-principles computations written out in this file;
* property-based edge cases (hypothesis).
"""

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.contingency_tables import Table, Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from torusplot.data.tables import ContingencyTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import contingency as ct

# Physicians' Health Study: myocardial infarction (fatal + non-fatal) by treatment
ASPIRIN = [[189, 11034 - 189], [104, 11037 - 104]]  # rows: placebo, aspirin; columns: MI, no MI
TEA = [[3, 1], [1, 3]]
# Party identification by gender (2000 General Social Survey), a standard textbook table
GENDER_GAP = [[762, 327, 468], [484, 239, 477]]
# Infant malformation (rows: present, absent) by maternal alcohol consumption (drinks / day)
MALFORMATION = [[48, 38, 5, 1, 1], [17066, 14464, 788, 126, 37]]


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    """The module under test registered itself on import; do not require sibling analysis
    modules (written by other people, possibly not importable yet) to load."""
    monkeypatch.setattr(registry, "_loaded", True)


def _fisher_first_principles(table, alternative="two-sided"):
    """Hypergeometric P value from factorials (no scipy)."""
    (a, b), (c, d) = table
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def prob(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    support = range(max(0, c1 - r2), min(r1, c1) + 1)
    if alternative == "greater":
        return sum(prob(x) for x in support if x >= a)
    if alternative == "less":
        return sum(prob(x) for x in support if x <= a)
    p_obs = prob(a)
    return sum(prob(x) for x in support if prob(x) <= p_obs * (1 + 1e-7))


def _rxc_brute_force(table):
    """Fisher-Freeman-Halton P by enumerating every way of dealing the column labels to
    the subjects (rows fixed): each distinct arrangement is equally likely under the null."""
    table = np.asarray(table)
    n_rows, n_cols = table.shape
    row_of = np.repeat(np.arange(n_rows), table.sum(axis=1))
    col_totals = table.sum(axis=0)
    counts: dict[tuple, int] = {}

    def deal(col, free, current):
        if col == n_cols - 1:
            t = current.copy()
            for subject in free:
                t[row_of[subject], col] += 1
            counts[tuple(t.ravel())] = counts.get(tuple(t.ravel()), 0) + 1
            return
        for chosen in itertools.combinations(free, int(col_totals[col])):
            t = current.copy()
            for subject in chosen:
                t[row_of[subject], col] += 1
            deal(col + 1, [s for s in free if s not in chosen], t)

    deal(0, list(range(int(table.sum()))), np.zeros_like(table))
    total = sum(counts.values())
    p_obs = counts[tuple(table.ravel())] / total
    return sum(v / total for v in counts.values() if v / total <= p_obs * (1 + 1e-7))


# ====================================================== published worked examples
def test_lady_tasting_tea_published_p_values():
    two = ct.fisher_exact_test(TEA)
    one = ct.fisher_exact_test(TEA, alternative="greater")
    assert round(two.p_value, 4) == 0.4857  # 34/70
    assert round(one.p_value, 4) == 0.2429  # 17/70
    assert two.p_value == pytest.approx(34 / 70, rel=1e-12)
    assert one.p_value == pytest.approx(17 / 70, rel=1e-12)
    assert ct.fisher_exact_test(TEA, alternative="less").p_value == pytest.approx(69 / 70, rel=1e-12)
    assert two.statistic_name == "odds ratio" and two.statistic == pytest.approx(9.0)
    assert one.alternative == "greater" and "one-tailed" in one.interpretation
    assert any("more likely" in note for note in one.data_notes)


def test_aspirin_myocardial_infarction_published_measures():
    res = ct.chi_square_test(ASPIRIN, row_names=["Placebo", "Aspirin"], column_names=["MI", "No MI"],
                             yates=False)
    odds, rr, rd = res.parameter("Odds ratio"), res.parameter("Relative risk"), res.parameter("Risk difference")
    assert round(rr.value, 2) == 1.82  # placebo relative to aspirin
    assert round(1 / rr.value, 2) == 0.55  # aspirin relative to placebo
    assert round(odds.value, 2) == 1.83
    assert (round(odds.ci_low, 2), round(odds.ci_high, 2)) == (1.44, 2.33)  # Woolf interval
    assert round(rd.value, 4) == 0.0077
    wald = res.extra["measures"]["risk_difference"]
    assert (round(wald["wald_ci_low"], 3), round(wald["wald_ci_high"], 3)) == (0.005, 0.011)
    assert res.df == 1 and res.p_value < 0.0001  # (statistic itself is cross-checked against scipy below)
    assert round(res.parameter("Risk of MI in Placebo").value, 4) == 0.0171
    assert round(res.parameter("Risk of MI in Aspirin").value, 4) == 0.0094
    # swapping the rows turns every ratio into its reciprocal
    swapped = ct.chi_square_test(ASPIRIN[::-1], yates=False)
    assert swapped.parameter("Relative risk").value == pytest.approx(1 / rr.value, rel=1e-12)
    assert swapped.parameter("Odds ratio").ci_low == pytest.approx(1 / odds.ci_high, rel=1e-12)
    assert swapped.parameter("Risk difference").value == pytest.approx(-rd.value, rel=1e-12)
    assert swapped.statistic == pytest.approx(res.statistic, rel=1e-12)
    assert "Placebo relative to Aspirin" in res.interpretation and "189 of 11034" in res.interpretation


def test_newcombe_1998_risk_difference_example():
    # Newcombe (1998), 56/70 vs 48/80: method 10 gives +0.0524 to +0.3339
    diff, low, high = ct.risk_difference_newcombe(56, 70, 48, 80)
    assert diff == pytest.approx(0.2, abs=1e-12)
    assert (round(low, 4), round(high, 4)) == (0.0524, 0.3339)
    rd = ct.fisher_exact_test([[56, 14], [48, 32]]).parameter("Risk difference")
    assert (round(rd.ci_low, 4), round(rd.ci_high, 4)) == (0.0524, 0.3339)


def test_gender_gap_textbook_table():
    res = ct.chi_square_test(GENDER_GAP, row_names=["Female", "Male"],
                             column_names=["Democrat", "Independent", "Republican"])
    assert round(res.statistic, 1) == 30.1 and res.df == 2
    assert "Yates" not in res.test_name  # no continuity correction beyond 2x2
    assert np.round(res.extra["expected"], 1).tolist() == [[703.7, 319.6, 533.7], [542.3, 246.4, 411.3]]
    assert np.round(res.extra["adjusted_residuals"], 2).tolist() == [[4.50, 0.70, -5.32], [-4.50, -0.70, 5.32]]
    assert res.parameters == []
    assert "Male / Republican" in res.interpretation or "Female / Republican" in res.interpretation


def test_malformation_trend_textbook_table():
    scores = [0, 0.5, 1.5, 4.0, 7.0]
    res = ct.chi_square_trend(MALFORMATION, scores=scores, row_names=["Present", "Absent"],
                              column_names=["0", "<1", "1-2", "3-5", ">=6"])
    n = 32574
    assert res.n_total is None and sum(res.n.values()) == n
    assert round(res.effect_sizes[0].value, 4) == 0.0142  # correlation between score and outcome
    # the textbook statistic is M^2 = (n - 1) r^2 = 6.57; the Cochran-Armitage form is n r^2
    assert round(res.statistic * (n - 1) / n, 2) == 6.57 and round(res.p_value, 3) == 0.010
    assert round(res.extra["chi2_total"], 1) == 12.1
    assert res.extra["departure_from_trend"]["df"] == 3


# ================================================= chi-square: cross-implementation
@pytest.mark.parametrize("table", [ASPIRIN, TEA, GENDER_GAP, [[12, 5, 7], [3, 9, 11], [8, 8, 2]],
                                   [[20, 15], [30, 25], [10, 40], [5, 5]]])
def test_chi_square_matches_scipy_and_statsmodels(table):
    arr = np.array(table, float)
    is_2x2 = arr.shape == (2, 2)
    res = ct.chi_square_test(table)
    ref = stats.chi2_contingency(arr, correction=is_2x2)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    assert res.df == ref.dof
    assert np.allclose(res.extra["expected"], ref.expected_freq, rtol=1e-12)
    plain = stats.chi2_contingency(arr, correction=False)
    assert res.extra["chi2_uncorrected"] == pytest.approx(plain.statistic, rel=1e-10)
    assert res.extra["p_uncorrected"] == pytest.approx(plain.pvalue, rel=1e-8)
    assert ct.chi_square_test(table, yates=False).statistic == pytest.approx(plain.statistic, rel=1e-10)
    sm = Table(arr, shift_zeros=False)
    assert np.allclose(res.extra["pearson_residuals"], sm.resid_pearson, rtol=1e-10)
    assert np.allclose(res.extra["adjusted_residuals"], sm.standardized_resids, rtol=1e-10)
    assert np.allclose(res.extra["chi2_contributions"], sm.chi2_contribs, rtol=1e-10)
    v = res.effect("Cramér's V")
    assert v.value == pytest.approx(stats.contingency.association(np.array(table), method="cramer"), rel=1e-10)
    # first principles
    n = arr.sum()
    expected = np.outer(arr.sum(1), arr.sum(0)) / n
    assert res.extra["chi2_uncorrected"] == pytest.approx(((arr - expected) ** 2 / expected).sum(), rel=1e-12)


def test_yates_correction_rules():
    default = ct.chi_square_test(TEA)
    assert default.extra["yates"] is True and "Yates" in default.test_name
    a, b, c, d = 3, 1, 1, 3
    n = 8
    classic = n * (abs(a * d - b * c) - n / 2) ** 2 / (4 * 4 * 4 * 4)  # textbook Yates formula
    assert default.statistic == pytest.approx(classic, rel=1e-12)
    assert ct.chi_square_test(TEA, yates=False).extra["yates"] is False
    # the correction never pushes |O - E| past zero
    near = ct.chi_square_test([[10, 10], [10, 11]])
    assert near.statistic == pytest.approx(0.0, abs=1e-12) and near.p_value == pytest.approx(1.0)
    # asking for Yates on a larger table is refused with a note, not applied
    big = ct.chi_square_test(GENDER_GAP, yates=True)
    assert big.extra["yates"] is False and any("only for 2x2" in n for n in big.data_notes)
    assert big.statistic == pytest.approx(ct.chi_square_test(GENDER_GAP).statistic)


def test_cochran_rule_statuses():
    ok = ct.chi_square_test(GENDER_GAP).assumptions[0]
    assert ok.status == "pass" and "Cochran" in ok.name
    # one of ten expected counts below 5 (10%): allowed by the rule but flagged
    warn = ct.chi_square_test([[30, 30, 30, 30, 4], [30, 30, 30, 30, 5]]).assumptions[0]
    assert warn.status == "warn"
    fail = ct.chi_square_test(TEA)
    assert fail.assumptions[0].status == "fail" and "Fisher's exact test" in fail.assumptions[0].detail
    assert any("Fisher's exact test" in w for w in fail.warnings)
    assert "prefer Fisher's exact test" in fail.interpretation
    tiny = ct.chi_square_test([[1, 0, 9], [0, 1, 9], [9, 9, 50]])
    assert tiny.assumptions[0].status == "fail" and "merge sparse categories" in tiny.assumptions[0].detail


def test_effect_sizes_phi_and_cramers_v():
    res = ct.chi_square_test(ASPIRIN)
    phi = res.effect("phi")
    # phi is the Pearson correlation between the two binary variables
    (a, b), (c, d) = ASPIRIN
    x = np.repeat([1, 1, 0, 0], [a, b, c, d])
    y = np.repeat([1, 0, 1, 0], [a, b, c, d])
    assert phi.value == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-9)
    assert res.effect_sizes[0].name == "phi" and phi.value > 0
    assert ct.chi_square_test(ASPIRIN[::-1]).effect("phi").value == pytest.approx(-phi.value, rel=1e-12)
    v = res.effect("Cramér's V")
    assert v.value == pytest.approx(abs(phi.value), rel=1e-12)
    # CI limits invert the noncentral chi-square distribution of the uncorrected statistic
    n, chi2 = np.sum(ASPIRIN), res.extra["chi2_uncorrected"]
    assert stats.ncx2.cdf(chi2, 1, v.ci_low**2 * n) == pytest.approx(0.975, abs=1e-6)
    assert stats.ncx2.cdf(chi2, 1, v.ci_high**2 * n) == pytest.approx(0.025, abs=1e-6)
    assert v.ci_low < v.value < v.ci_high
    # not significant -> lower limit 0
    ns = ct.chi_square_test([[20, 22], [21, 19]]).effect("Cramér's V")
    assert ns.ci_low == 0.0 and 0 < ns.ci_high < 1
    # Bergsma (2013) bias correction, from the formula
    arr = np.array(GENDER_GAP, float)
    res = ct.chi_square_test(arr)
    n, (r, c) = arr.sum(), arr.shape
    phi2 = max(0.0, res.statistic / n - (r - 1) * (c - 1) / (n - 1))
    rr, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    bc = res.effect("Cramér's V (bias-corrected)")
    assert bc.value == pytest.approx(math.sqrt(phi2 / min(rr - 1, cc - 1)), rel=1e-12)
    assert bc.value < res.effect("Cramér's V").value
    # independent table: corrected V is exactly 0
    assert ct.chi_square_test([[10, 20], [20, 40]]).effect("Cramér's V (bias-corrected)").value == 0.0


# ====================================================================== 2x2 measures
@pytest.mark.parametrize("table", [ASPIRIN, TEA, [[56, 14], [48, 32]], [[7, 93], [21, 79]], [[40, 3], [2, 51]]])
def test_two_by_two_measures_match_statsmodels_and_scipy(table):
    m = ct.two_by_two_measures(table)
    sm = Table2x2(np.array(table, float), shift_zeros=False)
    (a, b), (c, d) = table
    assert m["odds_ratio"]["value"] == pytest.approx(sm.oddsratio, rel=1e-10)
    assert (m["odds_ratio"]["ci_low"], m["odds_ratio"]["ci_high"]) == pytest.approx(sm.oddsratio_confint(), rel=1e-10)
    assert m["odds_ratio"]["se_log"] == pytest.approx(sm.log_oddsratio_se, rel=1e-10)
    assert m["relative_risk"]["value"] == pytest.approx(sm.riskratio, rel=1e-10)
    assert (m["relative_risk"]["ci_low"], m["relative_risk"]["ci_high"]) == pytest.approx(sm.riskratio_confint(), rel=1e-10)
    sp = stats.contingency.relative_risk(a, a + b, c, c + d)
    ci = sp.confidence_interval(0.95)
    assert m["relative_risk"]["value"] == pytest.approx(sp.relative_risk, rel=1e-10)
    assert (m["relative_risk"]["ci_low"], m["relative_risk"]["ci_high"]) == pytest.approx((ci.low, ci.high), rel=1e-10)
    # Newcombe hybrid score and Wilson intervals
    low, high = confint_proportions_2indep(a, a + b, c, c + d, method="newcomb", compare="diff")
    assert (m["risk_difference"]["ci_low"], m["risk_difference"]["ci_high"]) == pytest.approx((low, high), rel=1e-9)
    assert m["risk1_ci"] == pytest.approx(proportion_confint(a, a + b, method="wilson"), rel=1e-10)
    assert m["risk2_ci"] == pytest.approx(proportion_confint(c, c + d, method="wilson"), rel=1e-10)
    wald = confint_proportions_2indep(a, a + b, c, c + d, method="wald", compare="diff")
    assert (m["risk_difference"]["wald_ci_low"], m["risk_difference"]["wald_ci_high"]) == pytest.approx(wald, rel=1e-9)
    # definitions
    assert m["odds_ratio"]["value"] == pytest.approx(a * d / (b * c), rel=1e-12)
    assert m["relative_risk"]["value"] == pytest.approx((a / (a + b)) / (c / (c + d)), rel=1e-12)
    assert m["risk_difference"]["value"] == pytest.approx(a / (a + b) - c / (c + d), abs=1e-14)
    assert m["nnt"]["value"] == pytest.approx(1 / abs(m["risk_difference"]["value"]), rel=1e-12)
    # other confidence levels
    m90 = ct.two_by_two_measures(table, ci_level=0.90)
    assert (m90["odds_ratio"]["ci_low"], m90["odds_ratio"]["ci_high"]) == pytest.approx(sm.oddsratio_confint(alpha=0.10), rel=1e-10)


def test_wilson_interval_formula_and_edges():
    z = stats.norm.ppf(0.975)
    x, n = 7, 20
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert ct.wilson_interval(x, n) == pytest.approx((centre - half, centre + half), rel=1e-12)
    lo, hi = ct.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(z * z / (10 + z * z), rel=1e-12)
    lo, hi = ct.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(10 / (10 + z * z), rel=1e-12)
    assert all(math.isnan(v) for v in ct.wilson_interval(0, 0))


def test_nnt_and_its_interval():
    # CI of the risk difference excludes zero -> a single NNT interval from its reciprocals
    m = ct.two_by_two_measures(ASPIRIN)
    rd, nnt = m["risk_difference"], m["nnt"]
    assert nnt["value"] == pytest.approx(1 / rd["value"])
    assert (nnt["ci_low"], nnt["ci_high"]) == pytest.approx((1 / rd["ci_high"], 1 / rd["ci_low"]))
    assert nnt["ci_crosses_zero"] is False
    # negative difference: still positive NNT with ordered limits
    neg = ct.two_by_two_measures(ASPIRIN[::-1])["nnt"]
    assert (neg["value"], neg["ci_low"], neg["ci_high"]) == pytest.approx((nnt["value"], nnt["ci_low"], nnt["ci_high"]))
    # CI includes zero -> no single interval, explained in a note
    m = ct.two_by_two_measures(TEA)
    assert m["nnt"]["value"] == pytest.approx(2.0) and m["nnt"]["ci_low"] is None
    assert m["nnt"]["ci_crosses_zero"] is True
    assert m["nnt"]["bound_same_direction"] == pytest.approx(1 / m["risk_difference"]["ci_high"])
    assert m["nnt"]["bound_opposite_direction"] == pytest.approx(1 / abs(m["risk_difference"]["ci_low"]))
    assert any("through infinity" in n for n in m["notes"])
    res = ct.chi_square_test(TEA)
    assert "infinity" in res.parameter("Number needed to treat (NNT)").unit_note
    # identical risks -> NNT undefined
    same = ct.two_by_two_measures([[5, 5], [10, 10]])
    assert same["nnt"]["value"] is None and any("infinite" in n for n in same["notes"])


def test_zero_cells_are_flagged():
    m = ct.two_by_two_measures([[3, 0], [1, 3]])
    odds = m["odds_ratio"]
    assert odds["corrected"] is True and odds["uncorrected"] is None
    assert odds["value"] == pytest.approx((3.5 * 3.5) / (0.5 * 1.5), rel=1e-12)  # Haldane-Anscombe
    se = math.sqrt(1 / 3.5 + 1 / 0.5 + 1 / 1.5 + 1 / 3.5)
    assert odds["ci_low"] == pytest.approx(odds["value"] * math.exp(-stats.norm.ppf(0.975) * se), rel=1e-12)
    assert any("Haldane-Anscombe" in n for n in m["notes"])
    assert m["relative_risk"]["value"] == pytest.approx(4.0)
    # zero outcomes in the first row: OR "0" corrected, RR = 0 without a CI
    m = ct.two_by_two_measures([[0, 5], [3, 4]])
    assert m["odds_ratio"]["uncorrected"] == 0.0 and m["odds_ratio"]["corrected"] is True
    assert m["relative_risk"]["value"] == 0.0 and m["relative_risk"]["ci_low"] is None
    # zero outcomes in the second row: RR infinite -> None with a note
    m = ct.two_by_two_measures([[3, 4], [0, 5]])
    assert m["relative_risk"]["value"] is None and any("relative risk is undefined" in n for n in m["notes"])
    res = ct.fisher_exact_test([[3, 0], [1, 3]])
    assert res.statistic is None  # infinite sample odds ratio is not reported as a number
    cond = res.parameter("Odds ratio (conditional MLE)")
    assert cond.value is None and cond.ci_high is None and cond.ci_low == pytest.approx(0.3700601673910439, rel=1e-6)
    assert any("infinite" in n for n in res.data_notes)
    assert any("Haldane-Anscombe" in w for w in res.warnings)
    assert "0.5 added to each cell" in res.interpretation
    json.loads(res.model_dump_json())
    with pytest.raises(StatsError, match="2x2"):
        ct.two_by_two_measures(GENDER_GAP)


def test_diagnostic_measures():
    table = [[90, 30], [10, 170]]  # rows: test + / -, columns: disease present / absent
    res = ct.fisher_exact_test(table, row_names=["Test +", "Test -"], column_names=["Disease", "Healthy"],
                               diagnostic=True)
    expect = {"Sensitivity": (90, 100), "Specificity": (170, 200),
              "Positive predictive value": (90, 120), "Negative predictive value": (170, 180)}
    for name, (num, den) in expect.items():
        par = res.parameter(name)
        assert par.value == pytest.approx(num / den)
        assert (par.ci_low, par.ci_high) == pytest.approx(proportion_confint(num, den, method="wilson"), rel=1e-10)
    assert any("Diagnostic measures read the table as" in n for n in res.data_notes)
    assert res.extra["measures"]["diagnostic"]["sensitivity"]["denominator"] == 100
    assert ct.chi_square_test(table, diagnostic=True).parameter("Specificity").value == pytest.approx(0.85)
    assert ct.chi_square_test(table).parameter("Specificity") is None
    assert any("need a 2x2 table" in n for n in ct.chi_square_test(GENDER_GAP, diagnostic=True).data_notes)
    assert any("need a 2x2 table" in n for n in ct.fisher_exact_test([[3, 1, 2], [1, 3, 4]], diagnostic=True).data_notes)


# ============================================================== Fisher's exact test
@pytest.mark.parametrize("table", [TEA, [[8, 2], [1, 5]], [[6, 2], [1, 4]], [[0, 7], [5, 2]], [[12, 3], [9, 14]],
                                   [[1, 1], [1, 1]], [[30, 5], [22, 17]]])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_fisher_2x2_matches_scipy_and_first_principles(table, alternative):
    res = ct.fisher_exact_test(table, alternative=alternative)
    assert res.p_value == pytest.approx(stats.fisher_exact(table, alternative=alternative).pvalue, rel=1e-10)
    assert res.p_value == pytest.approx(_fisher_first_principles(table, alternative), rel=1e-9)
    assert res.extra["p_method"] == "exact"
    ref = stats.contingency.odds_ratio(table, kind="conditional")
    cond = res.parameter("Odds ratio (conditional MLE)")
    if math.isfinite(ref.statistic):
        assert cond.value == pytest.approx(ref.statistic, rel=1e-8)
    ci = ref.confidence_interval(0.95)
    assert cond.ci_low == pytest.approx(ci.low, rel=1e-8)
    assert (cond.ci_high is None) if math.isinf(ci.high) else cond.ci_high == pytest.approx(ci.high, rel=1e-8)


def test_fisher_2x2_result_content():
    res = ct.fisher_exact_test([[8, 2], [1, 5]], row_names=["Drug", "Placebo"],
                               column_names=["Improved", "Not improved"], row_label="Treatment",
                               column_label="Response").finalize()
    assert round(res.p_value, 4) == 0.035  # scipy documentation example: 0.0350
    assert res.significant is True and res.test_name == "Fisher's exact test"
    assert res.n == {"Drug": 10, "Placebo": 6} and res.n_total == 16
    assert res.descriptives[0]["Improved"] == 8 and res.descriptives[0]["% Improved"] == pytest.approx(80.0)
    assert res.effect("Odds ratio").value == pytest.approx(20.0)
    assert res.effect("phi").value > 0
    assert "Improved occurred in 8 of 10 (80%) in Drug and 1 of 6 (16.7%) in Placebo" in res.interpretation
    assert any("rows 'Drug' vs 'Placebo'; first column 'Improved'" in n for n in res.data_notes)
    assert res.extra["orientation"].startswith("Table orientation")
    assert [a.status for a in res.assumptions] == ["info", "info", "pass"]
    json.loads(res.model_dump_json())


@pytest.mark.parametrize("table", [[[3, 1, 2], [1, 2, 3]], [[2, 0, 1], [0, 3, 1], [1, 1, 0]], [[4, 1], [1, 2], [0, 2]]])
def test_fisher_rxc_exact_matches_brute_force(table):
    res = ct.fisher_exact_test(table)
    assert res.extra["p_method"] == "exact" and res.extra["tables_enumerated"] >= 1
    assert res.p_value == pytest.approx(_rxc_brute_force(table), rel=1e-9)
    assert ct.fisher_exact_test(np.array(table).T).p_value == pytest.approx(res.p_value, rel=1e-10)
    assert "Freeman-Halton" in res.test_name and res.statistic is None
    assert res.effect("Cramér's V").value > 0 and res.parameters == []


def test_fisher_rxc_enumeration_agrees_with_2x2_and_monte_carlo():
    # the general enumerator reproduces the 2x2 hypergeometric answer
    for table in ([[8, 2], [1, 5]], [[12, 3], [9, 14]], TEA):
        p, n_tables = ct._rxc_exact_p(np.array(table, float), 10**6)
        assert p == pytest.approx(stats.fisher_exact(table).pvalue, rel=1e-9)
        assert n_tables == min(sum(table[0]), sum(table[1]), table[0][0] + table[1][0], table[0][1] + table[1][1]) + 1
    table = [[10, 12, 9], [8, 15, 20], [7, 3, 11]]
    exact = ct.fisher_exact_test(table)
    assert exact.extra["p_method"] == "exact" and not exact.warnings
    mc = stats.fisher_exact(table, method=stats.MonteCarloMethod(n_resamples=200_000, rng=np.random.default_rng(5)))
    se = math.sqrt(exact.p_value * (1 - exact.p_value) / 200_000)
    assert abs(exact.p_value - mc.pvalue) < 5 * se
    # forced Monte Carlo fallback: reproducible, flagged, close to the exact value
    approx = ct.fisher_exact_test(table, max_tables=50)
    assert approx.extra["p_method"] == "monte_carlo" and approx.extra["monte_carlo"]["n_resamples"] == 100_000
    assert any("Monte Carlo estimate" in w for w in approx.warnings)
    assert abs(approx.p_value - exact.p_value) < 5 * approx.extra["monte_carlo"]["p_standard_error"]
    assert ct.fisher_exact_test(table, max_tables=50).p_value == approx.p_value  # fixed seed
    with pytest.raises(StatsError, match="only for 2x2"):
        ct.fisher_exact_test(table, alternative="greater")
    with pytest.raises(StatsError):
        ct.fisher_exact_test(TEA, alternative="bigger")


# ====================================================================== trend test
def test_trend_equals_n_r_squared_and_statsmodels():
    table = np.array([[10, 40], [15, 35], [25, 25], [30, 20]])  # 4 ordered doses x responder yes/no
    res = ct.chi_square_trend(table, row_names=["0 mg", "1 mg", "2 mg", "3 mg"], column_names=["Yes", "No"],
                              row_label="Dose")
    score = np.repeat([1, 2, 3, 4], table.sum(axis=1))
    outcome = np.concatenate([np.repeat([1, 0], row) for row in table])
    r = np.corrcoef(score, outcome)[0, 1]
    n = table.sum()
    assert res.statistic == pytest.approx(n * r * r, rel=1e-10)
    assert res.effect_sizes[0].value == pytest.approx(r, rel=1e-10)
    assert res.p_value == pytest.approx(stats.chi2.sf(n * r * r, 1), rel=1e-10)
    # statsmodels' linear-by-linear test uses (n - 1) r^2
    sm = Table(table).test_ordinal_association(row_scores=np.array([1.0, 2, 3, 4]), col_scores=np.array([1.0, 0.0]))
    assert res.statistic * (n - 1) / n == pytest.approx(sm.zscore**2, rel=1e-9)
    assert res.extra["z"] == pytest.approx(sm.zscore * math.sqrt(n / (n - 1)), rel=1e-9)
    # slope = least-squares slope of the outcome on the score
    assert res.parameter("Slope").value == pytest.approx(np.polyfit(score, outcome, 1)[0], rel=1e-9)
    assert "increased across Dose" in res.interpretation and res.df == 1
    dep = res.extra["departure_from_trend"]
    assert dep["statistic"] == pytest.approx(stats.chi2_contingency(table, correction=False).statistic - res.statistic)
    assert dep["df"] == 2 and not res.warnings
    # the same table laid out as 2 x k gives the same test
    flipped = ct.chi_square_trend(table.T)
    assert flipped.statistic == pytest.approx(res.statistic, rel=1e-12)
    # rescaling the scores does not matter; unequal spacing does
    assert ct.chi_square_trend(table, scores=[0, 10, 20, 30]).statistic == pytest.approx(res.statistic, rel=1e-12)
    assert ct.chi_square_trend(table, scores=[0, 1, 2, 10]).statistic != pytest.approx(res.statistic, rel=1e-3)
    # decreasing trend has a negative slope and correlation
    down = ct.chi_square_trend(table[::-1])
    assert down.parameter("Slope").value < 0 and down.effect_sizes[0].value < 0 and "decreased" in down.interpretation


def test_trend_edge_cases():
    # for a 2x2 table the trend test is the uncorrected chi-square test
    assert ct.chi_square_trend(TEA).statistic == pytest.approx(ct.chi_square_test(TEA, yates=False).statistic)
    assert ct.chi_square_trend(TEA).extra["departure_from_trend"] is None
    # a U-shaped pattern: no linear trend, but a clear departure from linearity
    u_shape = ct.chi_square_trend([[40, 10], [10, 40], [40, 10]])
    assert u_shape.statistic == pytest.approx(0.0, abs=1e-12) and "did not change" in u_shape.interpretation
    assert u_shape.effect_sizes[0].value == 0.0
    assert any("depart from a straight-line trend" in w for w in u_shape.warnings)
    assert ct.chi_square_trend([[3, 1], [2, 2], [1, 3]]).assumptions[0].status == "warn"
    with pytest.raises(StatsError, match="exactly two"):
        ct.chi_square_trend([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(StatsError, match="one numeric score"):
        ct.chi_square_trend([[3, 1], [2, 2], [1, 3]], scores=[1, 2])
    with pytest.raises(StatsError, match="identical"):
        ct.chi_square_trend([[3, 1], [2, 2], [1, 3]], scores=[2, 2, 2])


# ================================================================ input validation
@pytest.mark.parametrize("func", [ct.chi_square_test, ct.fisher_exact_test, ct.chi_square_trend])
def test_invalid_tables_raise_clear_errors(func):
    with pytest.raises(StatsError, match="whole numbers"):
        func([[12.5, 30], [40, 17.5]])  # percentages / means are not counts
    with pytest.raises(StatsError, match="negative"):
        func([[5, -1], [3, 4]])
    with pytest.raises(StatsError, match="missing"):
        func([[5, np.nan], [3, 4]])
    with pytest.raises(StatsError, match="at least 2 rows and 2 columns"):
        func([[5, 1, 3]])
    with pytest.raises(StatsError, match="at least 2 rows and 2 columns"):
        func([5, 1, 3])
    with pytest.raises(StatsError, match="numbers only"):
        func([["a", "b"], ["c", "d"]])
    with pytest.raises(StatsError, match="No subjects at all in row"):
        func([[0, 0], [3, 4]])
    with pytest.raises(StatsError, match=r"column\(s\) Dead"):
        func([[5, 0], [3, 0]], column_names=["Alive", "Dead"])
    with pytest.raises(StatsError, match="empty"):
        func([[0, 0], [0, 0]])
    with pytest.raises(StatsError, match="names does not match"):
        func([[5, 1], [3, 4]], row_names=["only one"])
    with pytest.raises(StatsError, match="alpha"):
        func([[5, 1], [3, 4]], alpha=0)


def test_dataframe_input_and_levels():
    frame = pd.DataFrame(TEA, index=["Milk first", "Tea first"], columns=["Guess milk", "Guess tea"])
    res = ct.fisher_exact_test(frame)
    assert list(res.n) == ["Milk first", "Tea first"] and res.extra["column_names"] == ["Guess milk", "Guess tea"]
    assert ct.chi_square_test(frame).descriptives[1]["group"] == "Tea first"
    with pytest.raises(StatsError, match="ci_level"):
        ct.chi_square_test(TEA, ci_level=95)
    with pytest.raises(StatsError, match="ci_level"):
        ct.two_by_two_measures(TEA, ci_level=0)
    # whole numbers stored as floats are fine
    assert ct.chi_square_test([[3.0, 1.0], [1.0, 3.0]]).statistic == pytest.approx(ct.chi_square_test(TEA).statistic)


# ======================================================================= registry
def _ctable(counts=TEA, rows=("Milk first", "Tea first"), cols=("Guess milk", "Guess tea")):
    return ContingencyTable.from_counts("Tea", counts, list(rows), list(cols), row_label="Poured",
                                        column_label="Guess")


@pytest.mark.parametrize("key", ["chi_square", "fisher_exact", "chi_square_trend"])
def test_registry_defaults(key):
    info = registry.REGISTRY[key]
    assert info.category == "Contingency tables" and info.tables == ("contingency",)
    assert all(p.help for p in info.params) and info.description and info.citation
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run_analysis(_ctable(), key)
    assert res.analysis == key and res.table == "Tea"
    assert res.interpretation and res.why_chosen and len(res.assumptions) >= 1
    assert res.significant is False and 0 <= res.p_value <= 1
    assert "Poured" in res.interpretation
    payload = json.loads(res.model_dump_json())
    assert payload["extra"]["observed"] == [[3.0, 1.0], [1.0, 3.0]]
    if key != "chi_square_trend":
        assert res.parameter("Odds ratio") is not None  # 2x2 measures included


def test_registry_options_and_table_checks():
    assert run_analysis(_ctable(), "fisher_exact", alternative="greater").p_value == pytest.approx(17 / 70)
    plain = run_analysis(_ctable(), "chi_square", yates="no")
    assert plain.statistic == pytest.approx(2.0) and plain.extra["yates"] is False
    assert run_analysis(_ctable(), "chi_square", yates="yes").extra["yates"] is True
    assert run_analysis(_ctable(), "chi_square", yates=False).extra["yates"] is False
    assert run_analysis(_ctable(), "chi_square", yates=None).extra["yates"] is True
    assert run_analysis(_ctable(), "chi_square", diagnostic=True).parameter("Sensitivity").value == pytest.approx(0.75)
    with pytest.raises(StatsError, match="yates must be"):
        run_analysis(_ctable(), "chi_square", yates="maybe")
    # r x c table through the registry
    big = _ctable(GENDER_GAP, ("Female", "Male"), ("Democrat", "Independent", "Republican"))
    assert round(run_analysis(big, "chi_square").statistic, 1) == 30.1
    assert 0 < run_analysis(_ctable([[3, 1, 2], [1, 3, 4]], ("a", "b"), ("x", "y", "z")), "fisher_exact").p_value < 1
    # percentages typed into a contingency table are refused
    pct = _ctable([[37.5, 62.5], [80.0, 20.0]])
    assert pct.has_non_integer_counts()
    with pytest.raises(StatsError, match="whole numbers"):
        run_analysis(pct, "chi_square")
    # blank cells are counted as zero and reported
    frame = pd.DataFrame({"Group": ["a", "b"], "Yes": [5, 3], "No": [np.nan, 6]})
    res = run_analysis(ContingencyTable("blank", frame), "fisher_exact")
    assert any("empty cell(s)" in n for n in res.data_notes) and res.n == {"a": 5, "b": 9}


# ================================================================ property-based
_count = st.one_of(st.integers(0, 12), st.integers(0, 400))


@settings(max_examples=60, deadline=None)
@given(st.tuples(_count, _count, _count, _count), st.sampled_from(["two-sided", "greater", "less"]))
def test_property_2x2_tables(cells, alternative):
    a, b, c, d = cells
    table = [[a, b], [c, d]]
    degenerate = min(a + b, c + d, a + c, b + d) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if degenerate:
            for func in (ct.chi_square_test, ct.fisher_exact_test):
                with pytest.raises(StatsError):
                    func(table)
            return
        chi = ct.chi_square_test(table).finalize()
        fisher = ct.fisher_exact_test(table, alternative=alternative).finalize()
    for res in (chi, fisher):
        assert 0.0 <= res.p_value <= 1.0
        text = res.model_dump_json()
        assert "Infinity" not in text and "NaN" not in text
        for par in res.parameters:
            if par.value is not None and par.ci_low is not None and par.ci_high is not None:
                assert par.ci_low <= par.value + 1e-9 and par.value <= par.ci_high + 1e-9
    assert chi.statistic >= 0 and chi.statistic <= chi.extra["chi2_uncorrected"] + 1e-9
    if min(a, b, c, d) == 0:
        assert any("Haldane-Anscombe" in w for w in chi.warnings)
    assert fisher.p_value == pytest.approx(stats.fisher_exact(table, alternative=alternative).pvalue, rel=1e-9, abs=1e-12)
    # invariances: transposing or swapping rows / columns never changes the chi-square statistic
    for other in ([[c, d], [a, b]], [[b, a], [d, c]], [[a, c], [b, d]]):
        assert ct.chi_square_test(other).statistic == pytest.approx(chi.statistic, rel=1e-9, abs=1e-9)
    two_sided = ct.fisher_exact_test(table).p_value
    assert ct.fisher_exact_test([[c, d], [a, b]]).p_value == pytest.approx(two_sided, rel=1e-9)
    # swapping the rows negates the risk difference and mirrors its interval
    rd = chi.parameter("Risk difference")
    rd_swapped = ct.chi_square_test([[c, d], [a, b]]).parameter("Risk difference")
    assert rd_swapped.value == pytest.approx(-rd.value, abs=1e-12)
    assert rd_swapped.ci_low == pytest.approx(-rd.ci_high, abs=1e-12)
    assert -1.0 <= rd.ci_low <= rd.value <= rd.ci_high <= 1.0


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 4), st.integers(2, 4), st.data())
def test_property_rxc_tables(n_rows, n_cols, data):
    cells = data.draw(st.lists(st.integers(0, 6), min_size=n_rows * n_cols, max_size=n_rows * n_cols))
    table = np.array(cells).reshape(n_rows, n_cols)
    degenerate = (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if degenerate:
            with pytest.raises(StatsError):
                ct.chi_square_test(table)
            with pytest.raises(StatsError):
                ct.fisher_exact_test(table)
            return
        chi = ct.chi_square_test(table).finalize()
        fisher = ct.fisher_exact_test(table, max_tables=20_000).finalize()
    assert 0.0 <= chi.p_value <= 1.0 and 0.0 <= fisher.p_value <= 1.0
    assert chi.df == (n_rows - 1) * (n_cols - 1)
    v = chi.effect("Cramér's V")
    assert 0.0 <= v.value <= 1.0 + 1e-12
    assert v.ci_low is None or v.ci_low <= v.value + 1e-9
    json.loads(chi.model_dump_json())
    json.loads(fisher.model_dump_json())
    # permuting rows and columns leaves both tests unchanged
    perm = table[::-1, ::-1]
    assert ct.chi_square_test(perm).statistic == pytest.approx(chi.statistic, rel=1e-9, abs=1e-9)
    assert ct.chi_square_test(table.T).statistic == pytest.approx(chi.statistic, rel=1e-9, abs=1e-9)
    if fisher.extra["p_method"] == "exact":
        assert ct.fisher_exact_test(perm, max_tables=20_000).p_value == pytest.approx(fisher.p_value, rel=1e-8)


@settings(max_examples=25, deadline=None)
@given(st.tuples(*[st.integers(1, 10**9)] * 4))
def test_property_huge_counts(cells):
    a, b, c, d = cells
    table = [[a, b], [c, d]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chi = ct.chi_square_test(table)
        measures = ct.two_by_two_measures(table)
    ref = stats.chi2_contingency(np.array(table, float), correction=True)
    assert chi.statistic == pytest.approx(ref.statistic, rel=1e-8)
    assert 0.0 <= chi.p_value <= 1.0
    assert measures["odds_ratio"]["value"] == pytest.approx((a / b) * (d / c), rel=1e-9)
    assert measures["odds_ratio"]["ci_low"] <= measures["odds_ratio"]["value"] <= measures["odds_ratio"]["ci_high"]


def test_fisher_with_large_counts_runs():
    # above 5,000 subjects the (slow) exact CI is skipped, with a note; the P value is still exact
    table = [[3000, 3100], [3000, 2900]]
    res = ct.fisher_exact_test(table)
    assert res.p_value == pytest.approx(stats.fisher_exact(table).pvalue, rel=1e-9)
    cond = res.parameter("Odds ratio (conditional MLE)")
    assert cond.value == pytest.approx(stats.contingency.odds_ratio(table).statistic, rel=1e-8)
    assert cond.ci_low is None and any("not computed" in n for n in res.data_notes)
    assert res.parameter("Odds ratio").ci_low < cond.value < res.parameter("Odds ratio").ci_high
    # counts beyond the 32-bit range: no conditional estimate, everything else still works
    huge = ct.fisher_exact_test([[10**9, 10**9 + 50_000], [10**9, 10**9]])
    assert 0.0 <= huge.p_value <= 1.0
    assert huge.parameter("Odds ratio (conditional MLE)").value is None
    assert any("could not be computed" in n for n in huge.data_notes)
    assert huge.parameter("Odds ratio").value == pytest.approx(1e9 / (1e9 + 5e4), rel=1e-12)


# =============================================================================
# Review item c: custom category scores for the trend test
# =============================================================================
def test_trend_scores_option_of_the_registered_analysis():
    spec = registry.get_analysis("chi_square_trend")
    param = next(p for p in spec.params if p.name == "scores")
    assert param.kind == "text" and param.default == "" and param.optional
    assert "0, 1, 5" in param.help and "equally spaced" in param.help
    counts = [[2, 18], [5, 15], [9, 11], [15, 5]]
    table = ContingencyTable.from_counts("t", counts, ["0 mg", "1 mg", "5 mg", "25 mg"], ["Responder", "None"])
    default = run_analysis(table, "chi_square_trend")
    assert default.extra["scores"] == [1.0, 2.0, 3.0, 4.0] and default.extra["scores_source"].startswith("equally")
    assert "enter the real values" in next(a for a in default.assumptions if a.name.startswith("Ordered")).detail
    assert run_analysis(table, "chi_square_trend", scores="  ").statistic == pytest.approx(default.statistic)

    res = run_analysis(table, "chi_square_trend", scores="0, 1, 5, 25")
    assert res.extra["scores"] == [0.0, 1.0, 5.0, 25.0] and res.params["scores"] == "0, 1, 5, 25"
    # reference: N r^2 between outcome (1 = responder) and the score, on the expanded data
    y = np.concatenate([[1] * a + [0] * b for a, b in counts])
    s = np.concatenate([[v] * 20 for v in (0, 1, 5, 25)])
    assert res.statistic == pytest.approx(y.size * np.corrcoef(y, s)[0, 1] ** 2, rel=1e-10)
    assert res.p_value == pytest.approx(stats.chi2.sf(res.statistic, 1), rel=1e-12)
    assert res.statistic != pytest.approx(default.statistic, rel=1e-3)
    # statsmodels' ordinal association test with the same scores gives the same statistic up to N / (N - 1)
    sm_res = Table(np.array(counts)).test_ordinal_association(row_scores=np.array([0.0, 1, 5, 25]),
                                                               col_scores=np.array([1.0, 0.0]))
    assert res.statistic == pytest.approx(sm_res.zscore**2 * y.size / (y.size - 1), rel=1e-9)
    assert "scores you entered (0, 1, 5, 25" in next(a for a in res.assumptions if a.name.startswith("Ordered")).detail
    assert "category scores 0, 1, 5, 25" in res.methods_text
    assert any("entered by the user" in n and "25 mg = 25" in n for n in res.data_notes)
    assert "per unit of the entered scores" in res.interpretation
    # separators, lists and a linear change of the scores
    for alt in ("0;1;5;25", "0 1 5 25", [0, 1, 5, 25], "10, 12, 20, 60"):
        assert run_analysis(table, "chi_square_trend", scores=alt).statistic == pytest.approx(res.statistic, rel=1e-10)
    json.loads(res.model_dump_json())


@pytest.mark.parametrize("bad, message", [
    ("0, 1, 5", "one numeric score for each of the 4"),
    ("0, 1, 5, 25, 100", "one numeric score for each of the 4"),
    ("0, low, 5, 25", "'low' is not a number"),
    ("0, 1, inf, 25", "finite"),
    ("3, 3, 3, 3", "identical"),
])
def test_trend_scores_are_validated_with_a_clear_message(bad, message):
    table = ContingencyTable.from_counts("t", [[2, 18], [5, 15], [9, 11], [15, 5]],
                                         ["a", "b", "c", "d"], ["Yes", "No"])
    with pytest.raises(StatsError, match=message):
        run_analysis(table, "chi_square_trend", scores=bad)


def test_trend_scores_out_of_order_are_allowed_but_flagged():
    table = ContingencyTable.from_counts("t", [[2, 18], [5, 15], [9, 11], [15, 5]],
                                         ["a", "b", "c", "d"], ["Yes", "No"])
    res = run_analysis(table, "chi_square_trend", scores="1, 3, 2, 4")
    assert any("not in steadily increasing" in w for w in res.warnings)
    assert not any("steadily increasing" in w for w in run_analysis(table, "chi_square_trend", scores="4, 3, 2, 1").warnings)
