"""Power and sample-size calculations.

Validation sources (see tests/fixtures/sources/basics.md):
* Published: Cohen (1988) sample-size tables as summarised in Cohen (1992) "A power
  primer", Table 2 (N per group for power .80 at alpha .05).
* Independent noncentral-t / noncentral-F computations written in this file (scipy).
* Cross-check against statsmodels' own solver, a Monte-Carlo simulation, and hypothesis
  property tests.
"""

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower

from torusplot.stats import StatsError
from torusplot.stats import power as pw

# ====================================================== independent reference formulas


def _t_power_ref(d, n1, alpha=0.05, alternative="two-sided", ratio=1.0, design="unpaired"):
    """Power of a t test straight from the noncentral t distribution."""
    if design == "unpaired":
        n2 = n1 * ratio
        df = n1 + n2 - 2
        ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
    else:
        df = n1 - 1
        ncp = d * math.sqrt(n1)
    if alternative == "two-sided":
        crit = stats.t.ppf(1 - alpha / 2, df)
        return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    if alternative == "greater":
        return stats.nct.sf(stats.t.ppf(1 - alpha, df), df, ncp)
    return stats.nct.cdf(stats.t.ppf(alpha, df), df, ncp)


def _anova_power_ref(f, n_per_group, k, alpha=0.05):
    """Power of the one-way ANOVA F test from the noncentral F distribution."""
    n_total = n_per_group * k
    df1, df2 = k - 1, n_total - k
    return stats.ncf.sf(stats.f.ppf(1 - alpha, df1, df2), df1, df2, f * f * n_total)


# ================================================================= published values


@pytest.mark.parametrize("d,n_published", [(0.5, 64), (0.8, 26)])
def test_cohen_t_test_sample_sizes(d, n_published):
    """Cohen (1988) Table 2.4.1 / Cohen (1992) Table 2: two-tailed alpha = .05, power = .80:
    d = .50 needs n = 64 per group, d = .80 needs n = 26 per group."""
    res = pw.power_ttest(effect_size=d, power=0.80, alpha=0.05)
    assert res.extra["n"] == n_published and res.extra["n2"] == n_published
    assert res.extra["n_total"] == 2 * n_published
    assert n_published - 1 < res.extra["n_exact"] <= n_published
    assert res.extra["achieved_power"] >= 0.80
    assert res.n == {"Group 1": n_published, "Group 2": n_published}
    # one subject fewer per group is not enough
    assert pw.power_ttest(effect_size=d, n=n_published - 1).extra["power"] < 0.80


def test_cohen_1992_power_primer_table():
    """Cohen (1992) Table 2, alpha = .05, power = .80. The table entries are the exact
    solutions rounded to the nearest integer (so "rounded up" can be one more).
    t test (d = .2, .5, .8): 393, 64, 26. ANOVA (f = .10, .25, .40):
    3 groups 322, 52, 21; 4 groups 274, 45, 18; 5 groups 240, 39, 16; 6 groups 215, 35, 14;
    7 groups (medium, large) 32, 13."""
    for d, published in ((0.2, 393), (0.5, 64), (0.8, 26)):
        assert round(pw.power_ttest(effect_size=d, power=0.80).extra["n_exact"]) == published
    table = {3: (322, 52, 21), 4: (274, 45, 18), 5: (240, 39, 16), 6: (215, 35, 14)}
    for k, row in table.items():
        for f, published in zip((0.10, 0.25, 0.40), row):
            res = pw.power_anova(k_groups=k, effect_size=f, power=0.80)
            assert round(res.extra["n_exact"]) == published, (k, f)
            assert res.extra["n"] in (published, published + 1) and res.extra["n_total"] == res.extra["n"] * k
    for f, published in ((0.25, 32), (0.40, 13)):
        assert round(pw.power_anova(k_groups=7, effect_size=f, power=0.80).extra["n_exact"]) == published


# ============================================================ cross-implementation


@pytest.mark.parametrize("design", ["unpaired", "paired", "one-sample"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_power_matches_noncentral_t(design, alternative):
    for d, n in ((0.3, 20), (0.5, 64), (0.8, 12), (1.5, 5), (0.2, 150)):
        d_signed = -d if alternative == "less" else d
        res = pw.power_ttest(design=design, effect_size=d_signed, n=n, alternative=alternative, alpha=0.01)
        ref = _t_power_ref(d_signed, n, 0.01, alternative, design=design)
        assert res.extra["power"] == pytest.approx(ref, rel=1e-9)
        assert res.extra["solved_for"] == "power" and res.alpha == 0.01 and res.alternative == alternative


def test_unequal_group_sizes_match_noncentral_t_and_statsmodels():
    res = pw.power_ttest(effect_size=0.6, n=20, ratio=2.5)
    assert res.extra["power"] == pytest.approx(_t_power_ref(0.6, 20, ratio=2.5), rel=1e-9)
    assert res.extra["power"] == pytest.approx(TTestIndPower().power(0.6, 20, 0.05, ratio=2.5), rel=1e-12)
    assert res.extra["n2"] == 50 and res.extra["n_total"] == 70 and res.n == {"Group 1": 20, "Group 2": 50}
    solved = pw.power_ttest(effect_size=0.6, power=0.9, ratio=0.5)
    n1 = solved.extra["n_exact"]
    assert _t_power_ref(0.6, n1, ratio=0.5) == pytest.approx(0.9, abs=1e-8)
    assert solved.extra["n"] == math.ceil(n1) and solved.extra["n2"] == math.ceil(0.5 * math.ceil(n1))
    assert solved.extra["achieved_power"] >= 0.9
    # paired / one-sample designs ignore the ratio
    assert pw.power_ttest(design="paired", effect_size=0.5, n=30, ratio=3).extra["power"] == pytest.approx(
        _t_power_ref(0.5, 30, design="paired"), rel=1e-9)


def test_solving_for_n_matches_statsmodels_solver_and_reference():
    for d, target, alt in ((0.5, 0.8, "two-sided"), (0.35, 0.9, "two-sided"), (0.7, 0.95, "greater")):
        res = pw.power_ttest(effect_size=d, power=target, alternative=alt)
        sm = TTestIndPower().solve_power(d, None, 0.05, target, 1.0,
                                         {"two-sided": "two-sided", "greater": "larger"}[alt])
        assert res.extra["n_exact"] == pytest.approx(sm, rel=1e-6)
        assert _t_power_ref(d, res.extra["n_exact"], alternative=alt) == pytest.approx(target, abs=1e-8)
        assert res.extra["n"] == math.ceil(res.extra["n_exact"])
        assert res.extra["achieved_power"] == pytest.approx(_t_power_ref(d, res.extra["n"], alternative=alt), rel=1e-9)
    paired = pw.power_ttest(design="paired", effect_size=0.5, power=0.8)
    assert paired.extra["n_exact"] == pytest.approx(TTestPower().solve_power(0.5, None, 0.05, 0.8), rel=1e-6)
    assert paired.extra["n"] == 34 and paired.n == {"Pairs": 34} and paired.extra["n2"] is None
    one = pw.power_ttest(design="one-sample", effect_size=-0.5, power=0.9, alternative="less")
    assert _t_power_ref(-0.5, one.extra["n_exact"], alternative="less", design="one-sample") == pytest.approx(0.9, abs=1e-8)
    assert one.n == {"Sample": one.extra["n"]}


def test_solving_for_effect_size():
    res = pw.power_ttest(n=64, power=0.8)
    assert res.extra["solved_for"] == "effect_size" and res.extra["effect_size"] > 0
    assert _t_power_ref(res.extra["effect_size"], 64) == pytest.approx(0.8, abs=1e-8)
    assert res.extra["effect_size"] == pytest.approx(0.5, abs=0.005)  # the mirror image of Cohen's table
    less = pw.power_ttest(n=20, power=0.8, alternative="less")
    assert less.extra["effect_size"] < 0
    assert _t_power_ref(less.extra["effect_size"], 20, alternative="less") == pytest.approx(0.8, abs=1e-8)
    greater = pw.power_ttest(n=20, power=0.8, alternative="greater")
    assert greater.extra["effect_size"] == pytest.approx(-less.extra["effect_size"], rel=1e-8)
    assert "or larger" in res.interpretation


def test_power_agrees_with_monte_carlo_simulation():
    """Fraction of significant simulated experiments ~ the calculated power (fixed seed)."""
    rng = np.random.default_rng(1988)
    n_sim, n, d = 4000, 18, 0.9
    a = rng.normal(0.0, 1.0, (n_sim, n))
    b = rng.normal(d, 1.0, (n_sim, n))
    observed = np.mean(stats.ttest_ind(b, a, axis=1).pvalue < 0.05)
    calc = pw.power_ttest(effect_size=d, n=n).extra["power"]
    assert abs(observed - calc) < 4 * math.sqrt(calc * (1 - calc) / n_sim)
    diffs = rng.normal(0.5, 1.0, (n_sim, 25))
    observed_paired = np.mean(stats.ttest_1samp(diffs, 0.0, axis=1).pvalue < 0.05)
    calc_paired = pw.power_ttest(design="paired", effect_size=0.5, n=25).extra["power"]
    assert abs(observed_paired - calc_paired) < 4 * math.sqrt(calc_paired * (1 - calc_paired) / n_sim)


def test_extreme_inputs_do_not_produce_nan():
    """scipy's noncentral t returns NaN in the far tail opposite to the effect; the power
    function must still give the right answer (1 for a huge effect)."""
    for d in (2.0, 4.0, 25.0, -4.0):
        assert pw.power_ttest(effect_size=d, n=64).extra["power"] == pytest.approx(1.0, abs=1e-9)
    assert pw.power_ttest(effect_size=4.0, n=64, design="paired", alternative="less").extra["power"] == pytest.approx(0.0, abs=1e-9)
    assert pw.power_ttest(effect_size=-4.0, n=64, design="paired", alternative="greater").extra["power"] == pytest.approx(0.0, abs=1e-9)
    big = pw.power_ttest(effect_size=0.01, power=0.9)
    assert _t_power_ref(0.01, big.extra["n_exact"]) == pytest.approx(0.9, abs=1e-6)
    assert 200_000 < big.extra["n"] < 220_000  # ~ 2 (z_.975 + z_.90)^2 / d^2 = 210 148
    huge = pw.power_ttest(effect_size=5.0, power=0.8)
    assert huge.extra["n"] == 3 and any("so few subjects" in w for w in huge.warnings)
    minimal = pw.power_ttest(effect_size=50.0, power=0.8)
    assert minimal.extra["n_exact"] == 2.0 and any("smallest allowed sample size" in w for w in minimal.warnings)
    assert pw.power_anova(k_groups=3, effect_size=50.0, n=1000).extra["power"] == pytest.approx(1.0)
    # statsmodels' ANOVA power is NaN here with scipy < 1.18 (noncentrality 1470; the noncentral
    # F tail underflows); newer scipy returns the true power of 1. TorusPlot must give 1 either way.
    reference = float(FTestAnovaPower().power(1.75, 480, 0.01, k_groups=3))
    assert math.isnan(reference) or reference == pytest.approx(1.0)
    assert pw.power_anova(k_groups=3, effect_size=1.75, n=160, alpha=0.01).extra["power"] == pytest.approx(1.0)
    assert any("smallest allowed" in w for w in pw.power_anova(k_groups=3, effect_size=50.0, power=0.8).warnings)


# =========================================================================== ANOVA


def test_anova_power_matches_noncentral_f_and_statsmodels():
    for f, n, k, alpha in ((0.25, 20, 3, 0.05), (0.4, 8, 5, 0.01), (0.1, 100, 4, 0.05), (0.6, 4, 2, 0.1)):
        res = pw.power_anova(k_groups=k, effect_size=f, n=n, alpha=alpha)
        assert res.extra["power"] == pytest.approx(_anova_power_ref(f, n, k, alpha), rel=1e-9)
        # statsmodels' nobs is the TOTAL sample size
        assert res.extra["power"] == pytest.approx(FTestAnovaPower().power(f, n * k, alpha, k_groups=k), rel=1e-12)
        assert res.extra["n_total"] == n * k and res.n == {f"Group {i + 1}": n for i in range(k)}
    solved = pw.power_anova(k_groups=4, effect_size=0.25, power=0.8)
    assert _anova_power_ref(0.25, solved.extra["n_exact"], 4) == pytest.approx(0.8, abs=1e-8)
    assert solved.extra["n"] == 45 and solved.extra["n_total"] == 180
    sm_total = FTestAnovaPower().solve_power(0.25, None, 0.05, 0.8, k_groups=4)
    assert solved.extra["n_exact"] * 4 == pytest.approx(sm_total, rel=1e-6)
    assert solved.extra["achieved_power"] == pytest.approx(_anova_power_ref(0.25, 45, 4), rel=1e-9)
    assert solved.extra["eta_squared"] == pytest.approx(0.25**2 / (1 + 0.25**2))
    effect = pw.power_anova(k_groups=3, n=52, power=0.8)
    assert _anova_power_ref(effect.extra["effect_size"], 52, 3) == pytest.approx(0.8, abs=1e-8)
    assert effect.extra["effect_size"] == pytest.approx(0.25, abs=0.005)
    # two groups: the F test is the two-sided t test with f = d / 2
    two = pw.power_anova(k_groups=2, effect_size=0.25, n=64)
    assert two.extra["power"] == pytest.approx(pw.power_ttest(effect_size=0.5, n=64).extra["power"], rel=1e-8)


def test_anova_result_content_and_errors():
    res = pw.power_anova(k_groups=4, effect_size=0.25, power=0.8).finalize()
    assert res.test_name == "Power analysis: one-way ANOVA" and res.analysis == "power_anova"
    assert "45 subjects per group (180 in total)" in res.interpretation and "medium" in res.interpretation
    assert "overall F test only" in res.interpretation
    assert res.parameter("Sample size per group, rounded up").value == 45
    assert res.parameter("Total sample size").value == 180
    assert res.parameter("Power").value == pytest.approx(res.extra["achieved_power"])
    assert res.p_value is None and res.significant is None and res.assumptions and res.citations
    json.loads(res.model_dump_json())
    frac = pw.power_anova(k_groups=3, effect_size=0.25, n=10.5)
    assert frac.n == {} and any("not a whole number" in w for w in frac.warnings)
    assert any("below 50%" in w for w in frac.warnings)
    for bad in (
        {"k_groups": 1, "effect_size": 0.25, "n": 10}, {"k_groups": 2.5, "effect_size": 0.25, "n": 10},
        {"k_groups": "x", "effect_size": 0.25, "n": 10}, {"k_groups": 3, "effect_size": 0.25},
        {"k_groups": 3, "effect_size": 0.25, "n": 10, "power": 0.8}, {"k_groups": 3, "effect_size": -0.1, "n": 10},
        {"k_groups": 3, "effect_size": 0.0, "power": 0.8}, {"k_groups": 3, "effect_size": 0.25, "n": 1},
        {"k_groups": 3, "effect_size": 0.25, "power": 0.03}, {"k_groups": 3, "effect_size": 1e-7, "power": 0.8},
        {"k_groups": 3, "effect_size": 0.25, "n": 10, "alpha": 1.0}, {"k_groups": 3, "effect_size": "big", "n": 10},
    ):
        with pytest.raises(StatsError):
            pw.power_anova(**bad)


# ================================================================ result and errors


def test_ttest_result_content():
    res = pw.power_ttest(effect_size=0.5, power=0.8).finalize()
    assert res.analysis == "power_ttest" and res.test_name == "Power analysis: unpaired t test"
    assert "64 subjects per group (128 in total)" in res.interpretation
    assert "80.0% power" in res.interpretation and "medium" in res.interpretation
    assert res.parameter("Effect size (Cohen's d)").value == 0.5
    assert res.parameter("Sample size, exact (per group)").value == pytest.approx(63.7656, abs=1e-3)
    assert res.parameter("Sample size, rounded up (per group)").value == 64
    assert res.parameter("Total sample size").value == 128
    assert res.parameter("Power").value == pytest.approx(0.8015, abs=1e-4)
    assert res.parameter("Alpha").value == 0.05
    assert res.p_value is None and res.significant is None
    assert [a.status for a in res.assumptions] == ["info", "info"] and res.why_chosen and res.methods_text
    json.loads(res.model_dump_json())
    given_n = pw.power_ttest(effect_size=0.5, n=20)
    assert "20 subjects per group (40 in total)" in given_n.interpretation
    assert any("below 50%" in w for w in given_n.warnings)
    assert "chance of missing it" in given_n.interpretation
    frac = pw.power_ttest(effect_size=0.5, n=10.5)
    assert frac.n == {} and frac.extra["n"] == 10.5 and any("not a whole number" in w for w in frac.warnings)
    assert pw.power_ttest(design="paired", effect_size=0.5, n=10.5).extra["n_total"] == 10.5
    unequal = pw.power_ttest(effect_size=0.5, power=0.8, ratio=2)
    assert "48 subjects in group 1 and 96 in group 2 (144 in total)" in unequal.interpretation
    assert "pairs" in pw.power_ttest(design="paired", effect_size=0.5, n=30).interpretation
    assert "values" in pw.power_ttest(design="one-sample", effect_size=0.5, power=0.8).interpretation
    assert "one-tailed" in pw.power_ttest(effect_size=0.5, n=30, alternative="greater").interpretation


def test_ttest_errors():
    for bad in (
        {"effect_size": 0.5}, {"effect_size": 0.5, "n": 20, "power": 0.8}, {},
        {"effect_size": 0.0, "power": 0.8}, {"effect_size": 0.5, "power": 0.8, "alternative": "less"},
        {"effect_size": -0.5, "power": 0.8, "alternative": "greater"},
        {"effect_size": 1e-7, "power": 0.8}, {"effect_size": 0.5, "n": 1},
        {"effect_size": 0.5, "n": 3, "ratio": 0.5},  # group 2 would have fewer than 2 subjects
        {"effect_size": 0.5, "power": 1.2}, {"effect_size": 0.5, "power": 0.0},
        {"effect_size": 0.5, "power": 0.04}, {"effect_size": 0.5, "n": 20, "alpha": 0.0},
        {"effect_size": 0.5, "n": 20, "alternative": "both"}, {"effect_size": 0.5, "n": 20, "design": "crossover"},
        {"effect_size": 0.5, "n": 20, "ratio": 0.0}, {"effect_size": 0.5, "n": 20, "ratio": "x"},
        {"effect_size": "large", "n": 20}, {"effect_size": float("nan"), "n": 20},
        {"effect_size": 0.5, "n": float("inf")},
    ):
        with pytest.raises(StatsError):
            pw.power_ttest(**bad)


def test_effect_size_from_means():
    assert pw.effect_size_from_means(5.0, 10.0) == 0.5
    assert pw.effect_size_from_means(-3.0, 2.0) == -1.5
    d = pw.effect_size_from_means(12.0, 15.0)
    assert pw.power_ttest(effect_size=d, power=0.8).extra["n"] == 26  # d = 0.8 again
    for bad in ((1.0, 0.0), (1.0, -2.0), ("a", 1.0), (float("nan"), 1.0), (1.0, float("inf"))):
        with pytest.raises(StatsError):
            pw.effect_size_from_means(*bad)


# =========================================================================== curve


def test_power_curve_ttest():
    res = pw.power_curve(effect_size=0.5).finalize()
    curve = res.extra["curve"]
    assert set(curve) == {"x", "y"} and len(curve["x"]) == len(curve["y"]) > 20
    assert curve["x"][0] == 2.0 and curve["x"] == sorted(curve["x"])
    assert all(b >= a for a, b in zip(curve["y"], curve["y"][1:]))  # power grows with n
    assert curve["y"][-1] >= 0.99
    for n, p in list(zip(curve["x"], curve["y"]))[::15]:
        assert p == pytest.approx(_t_power_ref(0.5, n), rel=1e-9)
    assert res.extra["target"] == {"power": 0.8, "n": 64, "n_exact": pytest.approx(63.7656, abs=1e-3)}
    assert curve["y"][curve["x"].index(64.0)] >= 0.8 > curve["y"][curve["x"].index(63.0)]
    assert res.extra["x_label"] == "Sample size per group" and res.extra["y_label"] == "Power"
    assert "64 per group" in res.interpretation and res.assumptions and res.why_chosen
    json.loads(res.model_dump_json())
    paired = pw.power_curve(effect_size=0.5, design="paired", n_max=40, target_power=0.9, alternative="greater")
    assert paired.extra["curve"]["x"] == [float(v) for v in range(2, 41)]
    assert paired.extra["curve"]["y"][-1] == pytest.approx(_t_power_ref(0.5, 40, alternative="greater", design="paired"))
    assert paired.extra["x_label"] == "Sample size" and paired.extra["target"]["power"] == 0.9
    custom = pw.power_curve(effect_size=0.8, n_values=[30, 5, 10, 10, 20])
    assert custom.extra["curve"]["x"] == [5.0, 10.0, 20.0, 30.0]
    small = pw.power_curve(effect_size=0.05)  # a long curve is thinned to at most 300 points
    assert len(small.extra["curve"]["x"]) <= 300 and small.extra["curve"]["y"][-1] >= 0.98
    ratio = pw.power_curve(effect_size=0.5, ratio=0.5, n_max=30)
    assert ratio.extra["curve"]["x"][0] == 4.0  # group 2 (half the size) needs at least 2
    assert ratio.extra["curve"]["y"][-1] == pytest.approx(_t_power_ref(0.5, 30, ratio=0.5), rel=1e-9)


def test_power_curve_anova_and_errors():
    res = pw.power_curve(effect_size=0.25, design="anova", k_groups=4, n_max=60).finalize()
    curve = res.extra["curve"]
    assert curve["x"] == [float(v) for v in range(2, 61)]
    assert curve["y"][43] == pytest.approx(_anova_power_ref(0.25, 45, 4), rel=1e-9)
    assert res.extra["target"]["n"] == 45 and "one-way ANOVA (4 groups)" in res.test_name
    auto = pw.power_curve(effect_size=0.4, design="anova", k_groups=3)
    assert auto.extra["curve"]["y"][-1] >= 0.99
    for bad in (
        {"effect_size": 0.0}, {"effect_size": 0.25, "design": "anova"}, {"effect_size": "x"},
        {"effect_size": 0.5, "n_values": []}, {"effect_size": 0.5, "n_values": [1, 5]},
        {"effect_size": 0.5, "n_max": 2}, {"effect_size": 0.5, "target_power": 1.0},
        {"effect_size": 0.5, "design": "nope"}, {"effect_size": 0.5, "alpha": 2},
    ):
        with pytest.raises(StatsError):
            pw.power_curve(**bad)


# ------------------------------------------------------------------ property tests

_design = st.sampled_from(["unpaired", "paired", "one-sample"])
_alternative = st.sampled_from(["two-sided", "greater", "less"])


@settings(max_examples=60, deadline=None)
@given(_design, _alternative, st.floats(min_value=-3, max_value=3), st.floats(min_value=2, max_value=500),
       st.sampled_from([0.001, 0.01, 0.05, 0.1]), st.floats(min_value=0.2, max_value=5))
def test_power_is_a_probability_and_matches_reference(design, alternative, d, n, alpha, ratio):
    if design == "unpaired" and n * ratio < 2:
        with pytest.raises(StatsError):
            pw.power_ttest(design=design, effect_size=d, n=n, alpha=alpha, alternative=alternative, ratio=ratio)
        return
    res = pw.power_ttest(design=design, effect_size=d, n=n, alpha=alpha, alternative=alternative, ratio=ratio)
    power = res.extra["power"]
    assert 0.0 <= power <= 1.0 and res.interpretation
    ref = _t_power_ref(d, n, alpha, alternative, ratio if design == "unpaired" else 1.0, design)
    if math.isfinite(ref):
        assert power == pytest.approx(ref, rel=1e-7, abs=1e-10)
    if alternative == "two-sided":
        assert power >= alpha * (1 - 1e-6)  # never below alpha, symmetric in the sign of d
        mirrored = pw.power_ttest(design=design, effect_size=-d, n=n, alpha=alpha, ratio=ratio)
        assert mirrored.extra["power"] == pytest.approx(power, rel=1e-7, abs=1e-10)
    json.loads(res.finalize().model_dump_json())


@settings(max_examples=50, deadline=None)
@given(_design, st.sampled_from(["two-sided", "greater"]), st.floats(min_value=0.05, max_value=3),
       st.floats(min_value=0.5, max_value=0.99), st.sampled_from([0.01, 0.05]), st.floats(min_value=0.25, max_value=4))
def test_solved_sample_size_reaches_the_target(design, alternative, d, target, alpha, ratio):
    res = pw.power_ttest(design=design, effect_size=d, power=target, alpha=alpha, alternative=alternative, ratio=ratio)
    n_exact, n = res.extra["n_exact"], res.extra["n"]
    assert n == math.ceil(n_exact - 1e-9) and n >= 2
    assert res.extra["achieved_power"] >= target - 1e-6
    r = ratio if design == "unpaired" else 1.0
    floor = max(2.0, 2.0 / r) if design == "unpaired" else 2.0
    if n_exact > floor + 1e-6:  # an interior solution sits exactly on the target
        assert _t_power_ref(d, n_exact, alpha, alternative, r, design) == pytest.approx(target, abs=1e-6)
    # round trip: solving for the effect size at that n gives the effect back
    if n_exact > floor + 1e-6:
        back = pw.power_ttest(design=design, n=n_exact, power=target, alpha=alpha, alternative=alternative, ratio=ratio)
        assert back.extra["effect_size"] == pytest.approx(d, rel=1e-5)
    # more subjects, more power
    more = pw.power_ttest(design=design, effect_size=d, n=n + 5, alpha=alpha, alternative=alternative, ratio=ratio)
    assert more.extra["power"] >= res.extra["achieved_power"] - 1e-9


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=2, max_value=8), st.floats(min_value=0.05, max_value=2),
       st.floats(min_value=2, max_value=300), st.floats(min_value=0.5, max_value=0.99),
       st.sampled_from([0.01, 0.05]))
def test_anova_properties(k, f, n, target, alpha):
    res = pw.power_anova(k_groups=k, effect_size=f, n=n, alpha=alpha)
    power = res.extra["power"]
    assert alpha * (1 - 1e-6) <= power <= 1.0
    ref = _anova_power_ref(f, n, k, alpha)
    if math.isfinite(ref):
        assert power == pytest.approx(ref, rel=1e-7, abs=1e-10)
    solved = pw.power_anova(k_groups=k, effect_size=f, power=target, alpha=alpha)
    assert solved.extra["n"] == math.ceil(solved.extra["n_exact"] - 1e-9) >= 2
    assert solved.extra["achieved_power"] >= target - 1e-6
    assert solved.extra["n_total"] == solved.extra["n"] * k
    json.loads(solved.finalize().model_dump_json())


@settings(max_examples=30, deadline=None)
@given(st.sampled_from(["unpaired", "paired", "one-sample", "anova"]), st.floats(min_value=0.1, max_value=2.5))
def test_power_curve_properties(design, effect):
    res = pw.power_curve(effect_size=effect, design=design, k_groups=3 if design == "anova" else None)
    xs, ys = res.extra["curve"]["x"], res.extra["curve"]["y"]
    assert len(xs) == len(ys) >= 9 and xs == sorted(xs) and len(set(xs)) == len(xs)
    assert all(0.0 <= y <= 1.0 for y in ys)
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
    json.loads(res.finalize().model_dump_json())
