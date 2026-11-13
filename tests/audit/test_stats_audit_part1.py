"""Independent statistical audit of torusplot.stats, part 1.

Scope: descriptives, assumptions, two_group, oneway, posthoc, effect_sizes, multiplicity,
outliers, power, result, _utils, registry and the NIST one-way ANOVA fixtures.

Section 1 holds one or more FAILING tests per defect found (each docstring states the
defect, the expected value and how that value was derived). Section 2 holds a few tests
that PASS today and guard behaviour that was verified independently and is easy to
"fix" into something wrong.

Severity key used in the docstrings:
  A = wrong numbers / wrong statistical conclusion      B = misleading text or labels
  C = crash / unhandled input                           D = documentation or minor issue
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.stats._nist import load_nist_anova
from torusplot.data import ColumnTable, MultipleVariablesTable
from torusplot.stats import StatsError, get_analysis, run_analysis
from torusplot.stats import assumptions as am
from torusplot.stats import descriptives as de
from torusplot.stats import effect_sizes as es
from torusplot.stats import multiplicity as mu
from torusplot.stats import oneway as ow
from torusplot.stats import outliers as ol
from torusplot.stats import posthoc as ph
from torusplot.stats import power as pw
from torusplot.stats import two_group as tg

# =====================================================================================
# SECTION 1 - DEFECTS (every test below fails on the audited code)
# =====================================================================================


# ------------------------------------------------------------------ defect 1 (A)
# effect_sizes.py:328-380  rank-biserial confidence intervals


def test_d01_rank_biserial_independent_ci_coverage():
    """DEFECT 1 (A): the "95% CI" of the rank-biserial correlation does not have 95% coverage.

    ``rank_biserial_independent`` builds the interval as tanh(atanh(r) +/- z * SE0) with
    SE0 = sqrt((n1 + n2 + 1) / (3 n1 n2)), the standard deviation of r under the NULL
    hypothesis. Away from r = 0 that is not the sampling SD of atanh(r), so the interval
    is far too narrow.

    Expected: for X ~ N(delta, 1), Y ~ N(0, 1) the estimator 2 U / (n1 n2) - 1 is unbiased
    for 2 P(X > Y) - 1 = 2 Phi(delta / sqrt(2)) - 1. With n1 = n2 = 10 and delta = 2
    (true r = 0.843) a 95% interval must cover that value in about 95% of samples;
    allowing generously for an approximate method, at least 90%. Fixed-seed simulation
    (2000 samples): the audited code covers in only ~74% of samples.
    """
    rng = np.random.default_rng(20240501)
    delta, n = 2.0, 10
    truth = 2.0 * stats.norm.cdf(delta / math.sqrt(2.0)) - 1.0
    reps, hits = 2000, 0
    for _ in range(reps):
        e = es.rank_biserial_independent(rng.normal(delta, 1, n), rng.normal(0, 1, n))
        hits += e.ci_low <= truth <= e.ci_high
    assert hits / reps >= 0.90, f"95% CI covered the true value in only {hits / reps:.1%} of samples"


def test_d01_rank_biserial_paired_ci_coverage():
    """DEFECT 1 (A), paired version: ``rank_biserial_paired`` has the same flaw.

    Expected: for differences D ~ N(delta, 1), W+ equals the number of positive Walsh
    averages, so E[W+] = n P(D > 0) + C(n, 2) P(D1 + D2 > 0) and the estimator's target is
    2 E[W+] / (n (n + 1) / 2) - 1 with P(D > 0) = Phi(delta), P(D1 + D2 > 0) =
    Phi(sqrt(2) delta). With n = 20, delta = 1 (target 0.828) a 95% CI must cover it in
    about 95% of samples (>= 90% required here). The audited code covers ~75%.
    """
    rng = np.random.default_rng(20240502)
    delta, n = 1.0, 20
    e_wplus = n * stats.norm.cdf(delta) + math.comb(n, 2) * stats.norm.cdf(math.sqrt(2.0) * delta)
    truth = 2.0 * e_wplus / (n * (n + 1) / 2.0) - 1.0
    reps, hits = 2000, 0
    for _ in range(reps):
        e = es.rank_biserial_paired(rng.normal(delta, 1, n))
        hits += e.ci_low <= truth <= e.ci_high
    assert hits / reps >= 0.90, f"95% CI covered the true value in only {hits / reps:.1%} of samples"


def test_d01_rank_biserial_ci_is_not_a_single_point_for_three_values_per_group():
    """DEFECT 1 (A/B): with complete separation the interval collapses to [-1, -1].

    ``_fisher_ci`` returns (r, r) when |r| = 1, so three values per group that do not
    overlap are reported as "rank-biserial r = -1 (95% CI -1 to -1)": a zero-width 95%
    interval from six observations. Complete separation of 3 vs 3 values happens with
    probability 2/20 = 10% even when the two populations are identical, so the true
    value is certainly not known exactly. Expected: no interval (None) or an interval of
    positive width.
    """
    e = es.rank_biserial_independent(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert e.value == -1.0
    assert e.ci_low is None or e.ci_high > e.ci_low, f"degenerate CI [{e.ci_low}, {e.ci_high}]"


# ------------------------------------------------------------------ defect 2 (A)
# effect_sizes.py:155-165  Hedges' g interval


def test_d02_hedges_g_ci_coverage_small_samples():
    """DEFECT 2 (A, small n): the CI reported for Hedges' g is the noncentral-t CI of
    Cohen's d with both limits multiplied by the bias-correction factor J.

    The noncentral-t interval (Steiger & Fouladi 1997; Cumming & Finch 2001, the
    ``ci_method`` quoted) is an exact interval for the population delta. Multiplying its
    limits by J < 1 gives an interval for J * delta instead, which under-covers delta;
    J corrects the bias of the point estimate, not the parameter. With n = 3 per group
    J = 0.798, the typical "n = 3" bench experiment.

    Expected: >= 95% coverage of delta (the unscaled interval is exact; >= 93% is
    required here to allow for Monte-Carlo error, 2000 samples, SE 0.5%). Fixed-seed
    simulation, n1 = n2 = 3, delta = 3: the audited code covers in ~89% of samples,
    while the Cohen's d interval from the same function covers in ~95.5%.
    """
    rng = np.random.default_rng(20240503)
    delta, reps, hits = 3.0, 2000, 0
    for _ in range(reps):
        g = es.cohen_d_independent(rng.normal(delta, 1, 3), rng.normal(0, 1, 3), hedges=True)
        hits += g.ci_low <= delta <= g.ci_high
    assert hits / reps >= 0.93, f"Hedges' g 95% CI covered delta in only {hits / reps:.1%} of samples"


# ------------------------------------------------------------------ defect 3 (A)
# two_group.py:312-313, 616-617 / effect_sizes.py:188-190, 217-219


def test_d03_effect_size_undefined_when_t_test_says_no_variability():
    """DEFECT 3 (A): a nonsensical effect size (about -1.8e16, "large") is reported for
    data the same result calls constant.

    ``_independent_t`` treats a group whose values agree to rounding error as constant
    (variance 0), so with x = [0.1 + 0.2, 0.3, 0.3] and y = [1, 1, 1] it reports "both
    groups have no variability, t cannot be computed". But ``cohen_d_independent`` has no
    such guard: it divides by the pooled SD of the rounding noise (3e-17) and returns
    d = -1.78e16 with a CI and magnitude "large".

    Expected: the effect size is undefined (value None) whenever the SD it standardises
    by is zero, exactly as the paired test already does.
    """
    res = tg.ttest_unpaired([0.1 + 0.2, 0.3, 0.3], [1.0, 1.0, 1.0])
    assert res.statistic is None  # the test itself correctly refuses
    for effect in res.effect_sizes:
        assert effect.value is None, f"{effect.name} = {effect.value!r} although the SD is zero"


def test_d03_one_sample_effect_size_undefined_when_no_variability():
    """DEFECT 3 (A), one-sample version: ``ttest_one_sample([0.1 + 0.2, 0.3, 0.3], 0)``
    reports "all values are identical" and Cohen's d = 5.4e15. Expected: None."""
    res = tg.ttest_one_sample([0.1 + 0.2, 0.3, 0.3], 0.0)
    assert res.statistic is None
    for effect in res.effect_sizes:
        assert effect.value is None, f"{effect.name} = {effect.value!r} although the SD is zero"


# ------------------------------------------------------------------ defect 4 (A)
# two_group.py:486-489, 999-1012 / posthoc.py:669-676  rounding noise in paired differences


def test_d04_paired_t_constant_differences_are_not_rounding_noise():
    """DEFECT 4 (A): ``ttest_paired`` turns floating-point rounding noise into a t statistic.

    The pairs (100.1, 100), (200.1, 200), (300.1, 300), (1.1, 1) all differ by exactly
    0.1, so the SD of the differences is 0 and t is undefined (the module's own rule:
    "Every pair has exactly the same difference ... cannot be computed"). In binary
    floating point the four differences come out as 0.0999999999999943 ...
    0.10000000000000009; ``is_constant(d)`` compares their spread with 4 ulp of 0.1
    instead of the rounding error of the inputs (ulp of 300), so the code reports
    t = 1.5e13, SE = 7e-15, P = 7e-40 and a CI of width 4e-14.

    Expected (exact decimal arithmetic, checked below with Fraction): SD = 0, hence no
    t statistic (None; or infinite under the post-hoc module's zero-SE convention).
    """
    a = ["100.1", "200.1", "300.1", "1.1"]
    b = ["100", "200", "300", "1"]
    exact = {Fraction(x) - Fraction(y) for x, y in zip(a, b)}
    assert exact == {Fraction(1, 10)}  # the data: every difference is exactly 0.1
    res = tg.ttest_paired([float(v) for v in a], [float(v) for v in b])
    assert res.statistic is None or math.isinf(res.statistic), f"t = {res.statistic!r} from rounding noise"


def test_d04_signed_rank_ties_detected_despite_rounding_noise():
    """DEFECT 4 (A): the Wilcoxon signed-rank test ranks rounding noise.

    Six pairs whose differences are exactly +0.1, -0.1, +0.1, -0.1, +0.1, -0.1. All
    |d| are tied, every rank is 3.5, so W+ = W- = 10.5 and W = 0. The audited code ranks
    the binary rounding errors of the differences instead (0.0999999999999943 <
    0.09999999999999964 < 0.10000000000000009) and reports W+ = 11, W- = 10, W = 1.

    Expected values from exact rational arithmetic (computed in the test).
    """
    x = ["1.1", "2.0", "3.1", "4.0", "100.1", "6.0"]
    y = ["1.0", "2.1", "3.0", "4.1", "100.0", "6.1"]
    d = [Fraction(a) - Fraction(b) for a, b in zip(x, y)]
    ranks = stats.rankdata([float(abs(v)) for v in d])  # |d| are all exactly 1/10 -> all 3.5
    assert len({abs(v) for v in d}) == 1
    w_plus = float(sum(r for r, v in zip(ranks, d) if v > 0))
    w_minus = float(sum(r for r, v in zip(ranks, d) if v < 0))
    assert (w_plus, w_minus) == (10.5, 10.5)
    res = tg.wilcoxon_signed_rank([float(v) for v in x], [float(v) for v in y])
    assert res.extra["w_plus"] == pytest.approx(w_plus)
    assert res.extra["w_minus"] == pytest.approx(w_minus)
    assert res.statistic == pytest.approx(0.0)


def test_d04_signed_rank_tie_correction_for_one_decimal_data():
    """DEFECT 4 (A): wrong P value for ordinary one-decimal data.

    before = 5.1 4.9 6.2 5.8 6.0 5.5 5.3 4.7, after = 5.4 5.2 6.0 6.1 6.3 5.3 5.6 5.0.
    The differences after - before are 0.3 (six times) and -0.2 (twice): two groups of
    ties (t = 6 and t = 2). Hence W+ = 33, W- = 3, mean 18,
    variance = 8*9*17/24 - ((6^3 - 6) + (2^3 - 2))/48 = 51 - 4.5 = 46.5,
    z = (33 - 18 - 0.5)/sqrt(46.5) = 2.1264, two-sided P = 0.03347 (normal approximation
    with tie and continuity correction, the method this module documents; identical to
    scipy.stats.wilcoxon(correction=True) on the exact differences).

    The audited code sees three distinct |d| because 5.4 - 5.1 and 5.2 - 4.9 differ in
    the last bit, uses the tie term of the wrong tie groups and reports P = 0.0371.
    """
    before = ["5.1", "4.9", "6.2", "5.8", "6.0", "5.5", "5.3", "4.7"]
    after = ["5.4", "5.2", "6.0", "6.1", "6.3", "5.3", "5.6", "5.0"]
    d = [Fraction(a) - Fraction(b) for a, b in zip(after, before)]
    assert sorted(d) == [Fraction(-1, 5)] * 2 + [Fraction(3, 10)] * 6
    var = 8 * 9 * 17 / 24.0 - ((6**3 - 6) + (2**3 - 2)) / 48.0
    z = (33.0 - 18.0 - 0.5) / math.sqrt(var)
    expected = 2.0 * stats.norm.sf(z)
    assert expected == pytest.approx(0.033471, abs=1e-6)
    res = tg.wilcoxon_signed_rank([float(v) for v in after], [float(v) for v in before])
    assert res.p_value == pytest.approx(expected, rel=1e-9)


def test_d04_paired_posthoc_constant_differences():
    """DEFECT 4 (A), post-hoc version (``pairwise_t(paired=True)``, posthoc.py:669-676).

    Same data as the paired t test above: every difference is exactly 0.1, so the SE is
    0 and, by this module's documented zero-SE convention, t is infinite and P = 0.
    The audited code reports SE = 7e-15 and a finite t = 1.5e13 computed from rounding
    noise (``_near_constant(d)`` uses the scale of d, not of the data).
    """
    groups = {"A": np.array([100.1, 200.1, 300.1, 1.1]), "B": np.array([100.0, 200.0, 300.0, 1.0])}
    comp = ph.pairwise_t(groups, paired=True, adjust="none").comparisons[0]
    assert comp.se == 0.0 and math.isinf(comp.statistic), f"SE = {comp.se!r}, t = {comp.statistic!r}"


# ------------------------------------------------------------------ defect 5 (A/B)
# two_group.py:840-862 and 1052-1070  tied tiny samples


def _exact_permutation_p_mann_whitney(x, y) -> float:
    """Two-sided exact permutation P value of the rank-sum statistic, valid with ties:
    every split of the pooled values into groups of the observed sizes is equally likely."""
    pooled = np.concatenate([x, y])
    n1, total = len(x), len(pooled)
    ranks = stats.rankdata(pooled)
    centre = n1 * (total + 1) / 2.0
    observed = abs(ranks[:n1].sum() - centre)
    count = splits = 0
    for idx in itertools.combinations(range(total), n1):
        splits += 1
        count += abs(ranks[list(idx)].sum() - centre) >= observed - 1e-9
    return count / splits


def test_d05_mann_whitney_tied_3_vs_3_cannot_be_significant():
    """DEFECT 5 (A/B): Mann-Whitney on [1, 1, 1] vs [2, 2, 2] is declared "statistically
    significant (P = 0.0469)" while the same result warns "this test can never reach
    significance at alpha = 0.05".

    With three values per group the most extreme arrangement has permutation probability
    1/C(6,3) = 1/20 per tail, so no two-sided P value below 0.10 is attainable, with or
    without ties (enumerated in ``_exact_permutation_p_mann_whitney``: exact P = 0.10).
    The audited code switches to the tie-corrected normal approximation because of the
    ties and reports 0.0469, below the attainable minimum.

    Expected: P = 0.10 (exact permutation value); in any case P >= 0.10 and not
    significant at 0.05.
    """
    x, y = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    exact = _exact_permutation_p_mann_whitney(x, y)
    assert exact == pytest.approx(0.1)
    res = tg.mann_whitney(x, y)
    assert any("never reach significance" in w for w in res.warnings)  # the result says so itself
    assert res.p_value >= exact - 1e-12, f"P = {res.p_value:.4f} is below the attainable minimum {exact}"


def test_d05_signed_rank_five_tied_differences_cannot_be_significant():
    """DEFECT 5 (A/B), signed-rank version: five differences, all +1.

    Under the null each difference is + or - with probability 1/2, so the smallest
    attainable two-sided P value is 2 / 2^5 = 0.0625 whatever the ranks (ties included).
    The audited code reports P = 0.0369 ("statistically significant") from the normal
    approximation, together with the warning "the smallest possible two-tailed exact P
    value is 0.0625; the test cannot reach significance at alpha = 0.05".

    Expected: P = 0.0625 (exact sign-flip distribution); in any case P >= 0.0625.
    """
    res = tg.wilcoxon_one_sample([1.0, 1.0, 1.0, 1.0, 1.0], 0.0)
    assert any("cannot reach significance" in w for w in res.warnings)
    assert res.p_value >= 2.0 / 2**5 - 1e-12, f"P = {res.p_value:.4f} is below the attainable minimum 0.0625"


# ------------------------------------------------------------------ defect 6 (A/B)
# posthoc.py:881-900  Wilcoxon comparisons after the Friedman test


def _signed_rank_normal_p(d: np.ndarray) -> float:
    """Two-sided signed-rank P value by the normal approximation with tie correction and
    continuity correction, zeros discarded (the method documented in two_group.py and
    used by R's wilcox.test when an exact P value is not available)."""
    d = d[d != 0]
    n = d.size
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(counts**3 - counts)) / 48.0
    z = max(abs(w_plus - n * (n + 1) / 4.0) - 0.5, 0.0) / math.sqrt(var)
    return float(2.0 * stats.norm.sf(z))


def test_d06_friedman_wilcoxon_posthoc_uses_documented_p_value_method():
    """DEFECT 6 (A/B): the pairwise Wilcoxon comparisons offered after the Friedman test
    give different P values from the application's own Wilcoxon test on the same pair,
    and the accompanying note misdescribes what was computed.

    ``friedman_posthoc(method="wilcoxon")`` delegates to ``scipy.stats.wilcoxon`` with
    default settings. For tied differences without zeros scipy uses the *untied* exact
    null distribution (the statistic can then be a half-integer that this distribution
    cannot even take); with zeros or n > 50 it uses a normal approximation *without*
    continuity correction. The result note nevertheless says "exact for small samples
    without ties or zero differences and use the normal approximation otherwise", and
    two_group.py documents "normal approximation with tie correction and continuity
    correction" as the convention "identical for every test".

    Data: 10 subjects, 3 integer-scored conditions (ties in every pair). Expected P
    values: the documented method, computed from first principles in
    ``_signed_rank_normal_p``: X vs Y 0.1405, X vs Z 0.0050, Y vs Z 0.0528. The audited
    code gives 0.1719, 0.0020 and 0.0625.
    """
    data = np.array([
        [1, 1, 2], [0, 1, 2], [2, 1, 3], [1, 3, 2], [0, 0, 1],
        [1, 2, 4], [-1, 1, 1], [0, 2, 1], [1, 0, 3], [2, 2, 4],
    ], dtype=float)
    res = ph.friedman_posthoc(data, ["X", "Y", "Z"], adjust="none", method="wilcoxon")
    for comp in res.comparisons:
        i, j = "XYZ".index(comp.group_a), "XYZ".index(comp.group_b)
        d = data[:, i] - data[:, j]
        assert np.unique(np.abs(d[d != 0])).size < d[d != 0].size  # tied differences
        expected = _signed_rank_normal_p(d)
        # the application's own two-group test agrees with the first-principles value
        assert tg.wilcoxon_signed_rank(data[:, i], data[:, j]).p_value == pytest.approx(expected, rel=1e-9)
        assert comp.p_value == pytest.approx(expected, rel=1e-9), (
            f"{comp.group_a} vs {comp.group_b}: post-hoc P = {comp.p_value}, documented method gives {expected}")


def test_d06_friedman_wilcoxon_posthoc_large_sample_continuity_correction():
    """DEFECT 6 (A/B): with more than 50 subjects (no ties) the post-hoc Wilcoxon P value
    is the normal approximation *without* continuity correction, whereas the documented
    convention (and ``wilcoxon_signed_rank`` on the same pair) uses the correction.
    Expected: equality with the first-principles value of ``_signed_rank_normal_p``."""
    rng = np.random.default_rng(20240506)
    data = rng.normal(0, 1, (60, 3)) + np.array([0.0, 0.2, 0.3])
    res = ph.friedman_posthoc(data, ["X", "Y", "Z"], adjust="none", method="wilcoxon")
    for comp in res.comparisons:
        i, j = "XYZ".index(comp.group_a), "XYZ".index(comp.group_b)
        expected = _signed_rank_normal_p(data[:, i] - data[:, j])
        assert comp.p_value == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ defect 7 (C/B)
# assumptions.py:552-561  Levene / Brown-Forsythe with a large offset


def _levene_f(groups: list[np.ndarray], center) -> float:
    z = [np.abs(g - center(g)) for g in groups]
    n = sum(v.size for v in z)
    k = len(z)
    grand = np.concatenate(z).mean()
    between = sum(v.size * (v.mean() - grand) ** 2 for v in z)
    within = sum(((v - v.mean()) ** 2).sum() for v in z)
    return float((n - k) / (k - 1) * between / within)


@pytest.mark.parametrize("center", ["median", "mean"])
def test_d07_levene_with_large_offset_is_computed(center):
    """DEFECT 7 (C/B): Levene / Brown-Forsythe refuse data with a large constant offset
    and explain it wrongly ("The spread is identical in every group").

    ``levene_test`` declares the statistic undefined when the within-group sum of
    squares of the absolute deviations is below N * (1e-12 * max|x|)^2. At an offset of
    1e12 that threshold is N * 1, so any data whose spread is below about 1 unit are
    called spread-less, although ``is_constant`` (and the t tests / ANOVA, see the NIST
    SmLs07-09 tests) treat the same numbers as perfectly analysable.

    Data: offset 1e12 plus deviations that are multiples of 1/1024, so every value is
    exactly representable (spacing of doubles at 1e12 is 1/8192) and the offset can be
    removed without any rounding. Group b is ~50 times more spread out than group a.
    Expected: Levene's F is invariant to the offset, so it must equal the F of the
    deviations alone, computed here from the defining formula and by scipy.stats.levene
    (median-centred F = 12.3, P = 0.003): clearly unequal spread, not "identical".
    """
    dev_a = np.array([0, 8, 3, 12, 5, 10, 1, 7]) / 1024.0
    dev_b = np.array([0, 400, 150, 600, 250, 500, 50, 350]) / 1024.0
    a, b = 1e12 + dev_a, 1e12 + dev_b
    assert np.array_equal(a - 1e12, dev_a) and np.array_equal(b - 1e12, dev_b)  # exact
    expected = _levene_f([dev_a, dev_b], np.median if center == "median" else np.mean)
    assert expected == pytest.approx(stats.levene(dev_a, dev_b, center=center).statistic, rel=1e-10)
    res = am.levene_test({"a": a, "b": b}, center=center)
    assert res["statistic"] is not None, res["warnings"]
    assert res["statistic"] == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------ defect 8 (C/B)
# assumptions.py:586-593 and oneway.py:257-265, 296-301  Bartlett with equal variances


def test_d08_bartlett_with_exactly_equal_variances():
    """DEFECT 8 (C/B): Bartlett's test "could not be computed" when the group variances
    are equal to within rounding.

    Bartlett's statistic T = [(N - k) ln s_p^2 - sum (n_i - 1) ln s_i^2] / C is exactly 0
    when all s_i^2 are equal, so P = 1. Rounding can make the numerator about -1e-13;
    scipy then returns statistic 0 with P = NaN, and ``bartlett_test`` turns that into
    "Bartlett's test could not be computed for these values".

    Data: one sample and two shifted copies of it (shift 1000 and 1e6), whose variances
    are mathematically identical. Expected: T = 0 (to rounding), P = 1.
    """
    base = np.array([1.1, 2.3, 3.2, 4.8, 5.9])
    res = am.bartlett_test({"a": base, "b": base + 1000.0, "c": base + 1e6})
    assert res["statistic"] is not None, res["warnings"]
    assert res["statistic"] == pytest.approx(0.0, abs=1e-6)
    assert res["p_value"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ["SmLs01", "SmLs04"])
def test_d08_bartlett_check_on_nist_smls(name):
    """DEFECT 8 (C/B) on the NIST StRD fixtures: in the SmLs data sets every one of the
    nine treatments has the same variance (0.01), so Bartlett's statistic is 0 and
    P = 1. For SmLs01 and SmLs04 the one-way ANOVA reports instead the assumption check
    "Equal variances (Bartlett): Cannot be tested (a group has fewer than 2 values or no
    variability)" - neither of which is true (n = 21 per group, SD = 0.1). (For
    SmLs02/03/05-09 the rounding happens to fall the other way and the check passes
    with P = 1.)

    Expected: status "pass", statistic ~ 0, P ~ 1 (equal sample variances by
    construction of the NIST data; verified with the offset removed in exact decimal
    arithmetic: every group variance equals 0.01 to 1e-12).
    """
    ref = load_nist_anova(name)
    variances = [float(np.var(v, ddof=1)) for v in ref.groups_offset_removed().values()]
    assert variances == pytest.approx([0.01] * len(variances), rel=1e-12)
    result = ow.anova_oneway(ref.groups(), posthoc="none")
    check = next(a for a in result.assumptions if a.name == "Equal variances (Bartlett)")
    assert check.status == "pass", check.detail
    assert check.p_value == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------ defect 9 (B)
# two_group.py:1066-1070  signed-rank small-sample warning ignores alpha


def test_d09_signed_rank_warning_contradicts_significant_result_at_alpha_010():
    """DEFECT 9 (B): the small-sample warning of the signed-rank test hard-codes 0.05.

    Five untied positive differences give the exact two-sided P = 2/2^5 = 0.0625. With
    the user's alpha = 0.10 the result is significant (0.0625 < 0.10), yet the warnings
    say "the test cannot reach significance at alpha = 0.05" - a statement about an alpha
    the user did not choose, next to a significant result.

    Expected: no "cannot reach significance" warning when the smallest attainable P
    value (0.0625) is below the alpha in use (the Mann-Whitney code already compares the
    minimum with the actual alpha).
    """
    res = tg.wilcoxon_one_sample([1.0, 2.0, 3.0, 4.0, 5.0], 0.0, alpha=0.10)
    assert res.p_value == pytest.approx(0.0625)
    assert res.p_value < res.alpha
    assert not any("cannot reach significance" in w for w in res.warnings), res.warnings


def test_d09_signed_rank_warns_when_alpha_001_is_unreachable():
    """DEFECT 9 (B), the other direction: with seven non-zero differences the smallest
    attainable two-sided exact P value is 2/2^7 = 0.0156, so at alpha = 0.01 the test can
    never be significant. The warning is only issued for fewer than 6 differences
    (hard-coded for alpha = 0.05), so the user is not told.

    Expected: a warning that significance cannot be reached at the chosen alpha.
    """
    res = tg.wilcoxon_one_sample([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 0.0, alpha=0.01)
    assert res.p_value == pytest.approx(2.0 / 2**7)
    assert any("reach significance" in w for w in res.warnings), res.warnings


# ------------------------------------------------------------------ defect 10 (B)
# outliers.py:169-173  Grubbs with n - 1 identical values


def test_d10_grubbs_three_values_two_identical_warning_contradicts_result():
    """DEFECT 10 (B): for [10.1, 10.1, 10.2] Grubbs' test flags 10.2 with P = 2e-8 while
    warning "With only 3 values Grubbs' test has little power: a single value can hardly
    ever be flagged, however extreme it looks".

    When n - 1 values are identical, G equals its mathematical maximum (n - 1)/sqrt(n)
    (1.1547 for n = 3), which exceeds the 5% critical value (1.1543) for ANY size of the
    deviation, so with rounded triplicates the "outlier" is flagged every time: the
    opposite of what the warning says. (The same happens for [10.1, 10.1, 10.3, 10.1].)
    The flag reflects rounding/ties, not an extreme value.

    Expected: the message must not claim that a value "can hardly ever be flagged" when
    one was flagged; the user should instead be told that the flag is an artefact of
    identical (rounded) values.
    """
    res = ol.grubbs_test([10.1, 10.1, 10.2])
    assert res["g"] == pytest.approx(2.0 / math.sqrt(3.0))  # the maximum possible G for n = 3
    contradictory = res["is_outlier"] and any("hardly ever be flagged" in w for w in res["warnings"])
    assert not contradictory, res["warnings"]


# ------------------------------------------------------------------ defect 11 (B)
# power.py:366-370 and 616-621  wording of power results


def test_d11_power_curve_unequal_groups_does_not_say_per_group():
    """DEFECT 11 (B): ``power_curve(effect_size=0.5, ratio=2)`` says "80.0% power needs
    48 per group". With n2 = 2 n1 the design needs 48 in group 1 and 96 in group 2
    (144 in total; values confirmed independently: noncentral t, df = 142,
    ncp = 0.5 / sqrt(1/48 + 1/96) gives power 0.802).

    Expected: the sentence must not state 48 "per group".
    """
    df, nc = 142, 0.5 / math.sqrt(1 / 48 + 1 / 96)
    crit = stats.t.isf(0.025, df)
    assert stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc) == pytest.approx(0.802, abs=1e-3)
    res = pw.power_curve(effect_size=0.5, ratio=2.0)
    assert "48 per group" not in res.interpretation, res.interpretation


def test_d11_power_curve_paired_design_does_not_say_per_group():
    """DEFECT 11 (B): for a paired design there are no groups, only pairs, but the curve
    is described as "a function of the sample size per group ... needs 34 per group".
    Expected: pairs (34 pairs is the correct number: G*Power / noncentral t)."""
    res = pw.power_curve(effect_size=0.5, design="paired")
    assert res.extra["target"]["n"] == 34
    assert "per group" not in res.interpretation, res.interpretation


def test_d11_detectable_effect_for_lower_tail_is_not_described_as_or_larger():
    """DEFECT 11 (B): solving for the detectable effect of a one-tailed "less" test gives
    d = -0.801, described as "power to detect effects of size d = -0.801 ... or larger.
    Smaller true effects will usually be missed". For a negative effect "or larger"
    means d closer to zero or positive, for which the power is *lower* (0.05 at d = 0).

    Expected: wording in terms of magnitude / "more negative", not "or larger".
    """
    res = pw.power_ttest(n=20, power=0.8, alternative="less")
    assert res.extra["effect_size"] == pytest.approx(-0.8007, abs=1e-3)
    assert "or larger" not in res.interpretation, res.interpretation


# ------------------------------------------------------------------ defect 12 (D/B)
# oneway.py:1160-1163  tooltip reused for the Kruskal-Wallis test


def test_d12_kruskal_wallis_groups_tooltip():
    """DEFECT 12 (D/B): the "Groups to compare" tooltip of the Kruskal-Wallis test says
    "with two, a t test gives the same answer". That is true for the one-way ANOVA
    (F = t^2) but false for Kruskal-Wallis, whose two-group equivalent is the
    Mann-Whitney test (as the result's own warning says).

    Expected: the Kruskal-Wallis tooltip must not promise that a t test gives the same
    answer.
    """
    groups_param = next(p for p in get_analysis("kruskal_wallis").params if p.name == "groups")
    assert "t test gives the same answer" not in groups_param.help


# ------------------------------------------------------------------ defect 13 (C)
# oneway.py:117-141, assumptions.py:551-553, two_group.py:244-255  overflow


def _scaled_groups(scale: float) -> dict[str, np.ndarray]:
    return {
        "a": np.array([1.0, 2.0, 3.0, 4.5]) * scale,
        "b": np.array([2.0, 3.0, 4.0, 7.5]) * scale,
        "c": np.array([5.0, 6.0, 8.0, 9.5]) * scale,
    }


def test_d13_anova_is_scale_invariant_for_very_large_values():
    """DEFECT 13 (C): with values of order 1e200 (finite, valid doubles) the one-way
    ANOVA returns F = NaN and P = NaN and the interpretation "The difference among the
    group means was could not be evaluated, F(2, 9) = n/a" because squares overflow.

    Expected: F is invariant to the unit of measurement, so it must equal the F of the
    same data divided by 1e200 (5.2409, cross-checked with scipy.stats.f_oneway); a
    clear StatsError would also be acceptable, a silent NaN is not.
    """
    reference = float(stats.f_oneway(*_scaled_groups(1.0).values()).statistic)
    try:
        res = ow.anova_oneway(_scaled_groups(1e200), posthoc="none")
    except StatsError:
        return
    assert res.statistic is not None and math.isfinite(res.statistic), f"F = {res.statistic!r}"
    assert res.statistic == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("call", ["equal_variance", "ttest_unpaired", "anova_1e308"])
def test_d13_no_raw_overflow_error(call):
    """DEFECT 13 (C): very large values crash with a raw ``OverflowError`` ("Numerical
    result out of range" / "intermediate overflow in fsum") instead of a result or a
    user-facing ``StatsError``: ``equal_variance_tests`` and ``ttest_unpaired`` at 1e200,
    ``anova_oneway`` at 1e308.

    Expected: a finite result (the tests are scale-invariant) or a StatsError.
    """
    groups = _scaled_groups(1e200)
    try:
        if call == "equal_variance":
            am.equal_variance_tests(groups)
        elif call == "ttest_unpaired":
            tg.ttest_unpaired(groups["a"], groups["b"])
        else:
            ow.anova_oneway({"a": [1e308, 1e308, -1e308], "b": [1.0, 2.0, 3.0], "c": [1.0, 2.0, 4.0]},
                            posthoc="none")
    except StatsError:
        pass
    except OverflowError as exc:  # the defect
        pytest.fail(f"raw OverflowError leaked to the caller: {exc}")


# ------------------------------------------------------------------ defect 14 (A, extreme inputs)
# _utils.py:77-93  spreads below 1e-75 are called "constant"


def test_d14_tiny_but_real_spread_is_not_zero():
    """DEFECT 14 (A, only for extreme magnitudes): ``is_constant`` treats any sample
    whose range is below 1e-75 as constant, so ``describe([1e-80, 2e-80, 3e-80])``
    reports SD = 0, SEM = 0, CV = 0% and a zero-width CI, and the one-sample t test says
    "All values are identical".

    The values differ by 100% of their size; their SD is exactly 1e-80 and its square
    (1e-160) is far inside the double range. Expected: SD = 1e-80 and
    t = mean / (SD / sqrt(3)) = 2 sqrt(3) = 3.4641 (the t statistic is scale-invariant:
    same as for [1, 2, 3]).
    """
    d = de.describe([1e-80, 2e-80, 3e-80])
    assert d["sd"] == pytest.approx(1e-80, rel=1e-9, abs=0.0), f"SD reported as {d['sd']!r}"
    res = tg.ttest_one_sample([1e-80, 2e-80, 3e-80], 0.0)
    assert res.statistic == pytest.approx(2.0 * math.sqrt(3.0), rel=1e-9)
    assert res.statistic == pytest.approx(tg.ttest_one_sample([1.0, 2.0, 3.0], 0.0).statistic, rel=1e-9)


# ------------------------------------------------------------------ defect 15 (D)
# two_group.py / assumptions.py  array-level results are not finalized


@pytest.mark.parametrize("func", ["ttest_unpaired", "ttest_welch", "mann_whitney", "ttest_paired",
                                  "wilcoxon_signed_rank"])
def test_d15_array_level_two_group_results_have_significance_flag(func):
    """DEFECT 15 (D): the array-level two-group functions (documented as public API in
    torusplot/stats/__init__.py) return a result whose ``significant`` flag is ``None``
    although P < alpha and the interpretation says "statistically significant". The
    one-way functions (``anova_oneway`` etc.) call ``finalize()`` and do set the flag, so
    ``if result.significant:`` silently behaves differently between modules.

    Expected: ``significant is True`` for clearly different groups (P < 0.05 in every
    test here), as for ``anova_oneway`` on the same data.
    """
    a = [5.1, 6.2, 7.3, 8.4, 9.5, 10.6, 11.7, 12.8]
    b = [1.0, 2.1, 3.3, 4.2, 5.6, 6.1, 7.2, 8.9]
    res = getattr(tg, func)(a, b)
    assert res.p_value < 0.05 and res.alpha == 0.05
    assert ow.anova_oneway({"A": a, "B": b}, posthoc="none").significant is True
    assert res.significant is True, f"significant = {res.significant!r} with P = {res.p_value:.2g}"


# ------------------------------------------------------------------ defect 16 (C/B)
# assumptions.py:845-855  'groups' silently ignored on multiple-variables tables


def test_d16_normality_groups_option_not_silently_ignored_on_multiple_variables_table():
    """DEFECT 16 (C/B): the normality analysis offers the option "Groups to check"
    ("Which columns to check") for both table types it accepts, but on a
    multiple-variables table the option is silently ignored: asking for column "y"
    tests the first numeric column ("dose") instead and reports its result without any
    note.

    Expected: the requested column is tested (or the option is rejected with a
    StatsError); testing a different variable than the one requested is not acceptable.
    """
    frame = pd.DataFrame({
        "dose": [1, 1, 2, 2, 2, 1, 1, 2, 2, 1],
        "y": [1.0, 2.5, 3.1, 4.7, 5.2, 6.8, 7.3, 8.9, 9.4, 10.6],
    })
    table = MultipleVariablesTable("mv", frame)
    try:
        res = run_analysis(table, "normality", groups=["y"])
    except StatsError:
        return
    assert list(res.n) == ["y"], f"asked for 'y', got a test of {list(res.n)}"


# =====================================================================================
# SECTION 2 - GUARDS (these PASS today; behaviour verified independently in the audit)
# =====================================================================================


def test_guard_pratt_exact_p_is_conditional_on_nonzero_ranks_not_scipy():
    """GUARD (passes). With Pratt's zero handling the exact null distribution is over the
    signs of the NON-ZERO ranks only. d = [0, 1.5, -2.5, 3.5, 4.5]: Pratt ranks of the
    non-zero differences are 2, 3, 4, 5; W+ = 2 + 4 + 5 = 11. Enumerating the 16 sign
    patterns gives P(W+ >= 11) = 3/16 ({2,4,5}, {3,4,5}, {2,3,4,5}), two-sided P = 0.375.
    scipy.stats.wilcoxon(zero_method="pratt", method="exact") returns 0.4375 here: it
    uses the distribution of ranks 1..5 as if the zero carried a sign. Do not "fix" the
    code to match scipy.
    """
    d = np.array([0.0, 1.5, -2.5, 3.5, 4.5])
    ranks = [2, 3, 4, 5]
    signs = [1, -1, 1, 1]
    observed = sum(r for r, s in zip(ranks, signs) if s > 0)
    totals = [sum(r for r, s in zip(ranks, pattern) if s) for pattern in itertools.product([0, 1], repeat=4)]
    upper = sum(t >= observed for t in totals) / 16.0
    lower = sum(t <= observed for t in totals) / 16.0
    expected = min(1.0, 2.0 * min(upper, lower))
    res = tg.wilcoxon_one_sample(d, 0.0, zero_method="pratt")
    assert res.extra["w_plus"] == observed == 11
    assert expected == pytest.approx(0.375)
    assert res.p_value == pytest.approx(expected)


def test_guard_hodges_lehmann_exact_ci_is_the_inverted_exact_test():
    """GUARD (passes). The exact Hodges-Lehmann CI of the Mann-Whitney test must be the
    set of shifts not rejected by the exact two-sided test: just inside each limit the
    exact P value of (x - shift) vs y exceeds 1 - level, just outside it does not.
    Checked against scipy's exact Mann-Whitney P value (independent implementation)."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        n1, n2 = rng.integers(4, 10, 2)
        x, y = rng.normal(0, 1, n1), rng.normal(0, 1, n2)
        par = tg.mann_whitney(x, y, ci_level=0.95).parameters[0]
        if par.ci_low is None:
            continue

        def p_at(shift: float, x: np.ndarray = x, y: np.ndarray = y) -> float:
            return float(stats.mannwhitneyu(x - shift, y, method="exact").pvalue)

        eps = 1e-7
        assert p_at(par.ci_low + eps) > 0.05 and p_at(par.ci_high - eps) > 0.05
        assert p_at(par.ci_low - eps) <= 0.05 + 1e-12 and p_at(par.ci_high + eps) <= 0.05 + 1e-12


def test_guard_dunnett_critical_values_match_published_tables():
    """GUARD (passes). Two-sided Dunnett critical values (Dunnett 1964, Table II/III,
    equal n): 3 treatments, 20 df, alpha 0.05 -> 2.54; 5 treatments, 24 df, alpha 0.01
    -> 3.40. One-sided (Dunnett 1955): 3 treatments, 20 df, alpha 0.05 -> 2.19."""
    n3, n5 = np.full(3, 5.0), np.full(5, 5.0)
    assert ph._dunnett_critical(0.05, n3, 5.0, 20.0, True) == pytest.approx(2.54, abs=0.005)
    assert ph._dunnett_critical(0.01, n5, 5.0, 24.0, True) == pytest.approx(3.40, abs=0.005)
    assert ph._dunnett_critical(0.05, n3, 5.0, 20.0, False) == pytest.approx(2.19, abs=0.005)


def test_guard_adjustments_match_statsmodels_including_zeros_ones_and_ties():
    """GUARD (passes). All five adjustments agree with statsmodels.multipletests to
    1e-12 on random families that include P = 0, P = 1 and tied P values."""
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(1)
    for trial in range(100):
        m = int(rng.integers(1, 15))
        p = rng.uniform(0, 1, m) ** int(rng.integers(1, 6))
        if trial % 5 == 0:
            p[rng.integers(0, m)] = 0.0
        if trial % 7 == 0:
            p[rng.integers(0, m)] = 1.0
        if trial % 3 == 0 and m > 2:
            p[0] = p[1]
        for method in ("bonferroni", "sidak", "holm", "holm-sidak", "fdr_bh"):
            np.testing.assert_allclose(mu.adjust_pvalues(p, method), multipletests(p, method=method)[1],
                                       rtol=0, atol=1e-12)


def test_guard_registry_runs_two_group_analysis_with_a_minus_b_convention():
    """GUARD (passes). Through the registry the difference is first column minus second
    column and 'greater' means first > second (checked with scipy.stats.ttest_ind)."""
    table = ColumnTable.from_dict("t", {"Control": [5.0, 6.0, 7.0, 8.0, 9.0], "Drug": [1.0, 2.0, 3.0, 4.0, 5.5]})
    res = run_analysis(table, "ttest_unpaired", alternative="greater")
    ref = stats.ttest_ind(table.groups()["Control"], table.groups()["Drug"], alternative="greater")
    assert res.parameters[0].value == pytest.approx(7.0 - 3.1)
    assert res.p_value == pytest.approx(float(ref.pvalue), rel=1e-10)
    assert res.significant is True
