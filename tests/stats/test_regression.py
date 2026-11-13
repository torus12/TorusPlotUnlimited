"""Linear regression (simple, comparison of slopes, multiple).

Reference values: NIST StRD Norris (certified) and Longley (see tests/fixtures/sources/
regression.md for provenance), Anscombe (1973), plus cross-checks against scipy and
statsmodels and first-principles computations written here.
"""

import itertools
import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson as sm_durbin_watson

from tests.fixtures.nist.strd_regression import load_lls
from torusplot.data.tables import MultipleVariablesTable, XYTable
from torusplot.stats import run_analysis
from torusplot.stats._utils import StatsError
from torusplot.stats.regression import (
    breusch_pagan,
    compare_slopes,
    durbin_watson,
    information_criteria,
    is_constant,
    lack_of_fit,
    linear_regression,
    multiple_regression,
    runs_test,
)


def assert_strict_json(res):
    """The result must serialise to strict JSON: no NaN / Infinity anywhere."""
    json.dumps(json.loads(res.finalize().model_dump_json()), allow_nan=False)


RNG = np.random.default_rng(11)
X = np.repeat(np.arange(1.0, 9.0), 3)
Y = 2.0 + 0.7 * X + RNG.normal(0, 1.0, X.size)

ANSCOMBE_X123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ANSCOMBE = {
    "I": (ANSCOMBE_X123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (ANSCOMBE_X123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (ANSCOMBE_X123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}


# ------------------------------------------------------------- NIST / published
def test_nist_norris_certified_values():
    d = load_lls("Norris")
    res = linear_regression(d.x[:, 0], d.y)
    slope, intercept = res.parameter("Slope"), res.parameter("Y-intercept")
    gof = res.extra["goodness_of_fit"]
    assert slope.value == pytest.approx(d.certified["B1"], rel=1e-11)
    assert intercept.value == pytest.approx(d.certified["B0"], rel=1e-10)
    assert slope.se == pytest.approx(d.certified_sd["B1"], rel=1e-10)
    assert intercept.se == pytest.approx(d.certified_sd["B0"], rel=1e-10)
    assert gof["sy_x"] == pytest.approx(d.residual_sd, rel=1e-10)
    assert gof["r_squared"] == pytest.approx(d.r_squared, rel=1e-12)
    assert res.statistic == pytest.approx(d.f_statistic, rel=1e-10)
    assert (res.df, res.df2) == (d.df_regression, d.df_residual)
    table = {row["source"]: row for row in res.anova_table}
    assert table["Regression"]["ss"] == pytest.approx(d.ss_regression, rel=1e-11)
    assert table["Residual"]["ss"] == pytest.approx(d.ss_residual, rel=1e-10)
    assert table["Residual"]["ms"] == pytest.approx(d.ss_residual / d.df_residual, rel=1e-10)


def test_nist_longley_multiple_regression():
    d = load_lls("Longley")
    cols = [f"x{i}" for i in range(1, 7)]
    df = pd.DataFrame(d.x, columns=cols)
    df["y"] = d.y
    res = multiple_regression(df, "y", cols)
    for i, par in enumerate(res.parameters):
        assert par.value == pytest.approx(d.certified[f"B{i}"], rel=1e-8)
        assert par.se == pytest.approx(d.certified_sd[f"B{i}"], rel=1e-8)
    gof = res.extra["goodness_of_fit"]
    assert gof["sy_x"] == pytest.approx(d.residual_sd, rel=1e-10)
    assert gof["r_squared"] == pytest.approx(d.r_squared, rel=1e-12)
    assert res.statistic == pytest.approx(d.f_statistic, rel=1e-9)
    assert (res.df, res.df2) == (6, 9)
    assert gof["ss_residual"] == pytest.approx(d.ss_residual, rel=1e-9)
    # Longley is the textbook multicollinearity example: the check must say so
    vif = next(a for a in res.assumptions if "VIF" in a.name)
    assert vif.status == "fail" and any("Multicollinearity" in w for w in res.warnings)
    assert res.extra["vif"]["x2"] > 1000


@pytest.mark.parametrize("name", list(ANSCOMBE))
def test_anscombe_quartet_published_summary(name):
    # Anscombe FJ (1973) Am Stat 27:17-21: for all four data sets "equation of regression
    # line y = 3 + 0.5x", regression SS = 27.50 (1 df), residual SS = 13.75 (9 df),
    # estimated standard error of b1 = 0.118, multiple R^2 = 0.667. The four sets were built
    # to share these summaries only approximately (e.g. the residual SS is 13.74 to 13.78
    # across the sets), which is what sets the tolerances below.
    x, y = ANSCOMBE[name]
    res = linear_regression(x, y)
    assert res.parameter("Slope").value == pytest.approx(0.500, abs=1e-3)
    assert res.parameter("Y-intercept").value == pytest.approx(3.00, abs=5e-3)
    assert res.parameter("Slope").se == pytest.approx(0.118, abs=5e-4)
    table = {row["source"]: row for row in res.anova_table}
    assert table["Regression"]["ss"] == pytest.approx(27.50, abs=0.04)
    assert table["Residual"]["ss"] == pytest.approx(13.75, abs=0.04)
    assert table["Residual"]["df"] == 9
    assert res.effect("R squared").value == pytest.approx(0.667, abs=1e-3)
    assert res.effect("Pearson r").value == pytest.approx(0.816, abs=1e-3)


def test_anscombe_diagnostics_tell_the_four_apart():
    def status(name, key):
        res = linear_regression(*ANSCOMBE[name])
        return next(a.status for a in res.assumptions if key in a.name)

    assert status("I", "runs test") == "pass"
    assert status("II", "runs test") == "fail"       # a curve, not a line
    assert status("III", "Normality") == "fail"      # one outlier
    assert status("I", "Normality") == "pass"


# ---------------------------------------------------------- cross-implementation
def test_matches_scipy_linregress_and_statsmodels():
    res = linear_regression(X, Y)
    lr = stats.linregress(X, Y)
    slope, intercept = res.parameter("Slope"), res.parameter("Y-intercept")
    assert slope.value == pytest.approx(lr.slope, rel=1e-12)
    assert intercept.value == pytest.approx(lr.intercept, rel=1e-12)
    assert slope.se == pytest.approx(lr.stderr, rel=1e-10)
    assert intercept.se == pytest.approx(lr.intercept_stderr, rel=1e-10)
    assert slope.p_value == pytest.approx(lr.pvalue, rel=1e-8)
    assert res.p_value == pytest.approx(lr.pvalue, rel=1e-8)   # F test == slope t test
    assert res.statistic == pytest.approx(slope.statistic**2, rel=1e-10)
    assert res.effect("R squared").value == pytest.approx(lr.rvalue**2, rel=1e-12)
    assert res.effect("Pearson r").value == pytest.approx(lr.rvalue, rel=1e-12)
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    ci = ols.conf_int()
    assert (intercept.ci_low, intercept.ci_high) == pytest.approx(tuple(ci[0]), rel=1e-10)
    assert (slope.ci_low, slope.ci_high) == pytest.approx(tuple(ci[1]), rel=1e-10)
    assert intercept.p_value == pytest.approx(ols.pvalues[0], rel=1e-8)
    gof = res.extra["goodness_of_fit"]
    assert gof["adj_r_squared"] == pytest.approx(ols.rsquared_adj, rel=1e-12)
    assert gof["sy_x"] == pytest.approx(math.sqrt(ols.scale), rel=1e-12)
    assert res.statistic == pytest.approx(ols.fvalue, rel=1e-10)
    np.testing.assert_allclose(res.extra["residuals"]["residual"], ols.resid, atol=1e-10)
    np.testing.assert_allclose(res.extra["residuals"]["fitted"], ols.fittedvalues, rtol=1e-12)
    assert res.extra["equation"].startswith("Y = ")


@pytest.mark.parametrize("level", [0.95, 0.99])
def test_confidence_and_prediction_bands_match_statsmodels(level):
    res = linear_regression(X, Y, ci_level=level)
    curve = res.extra["curve"]
    grid = np.array(curve["x"])
    assert grid.size == 200 and grid[0] == X.min() and grid[-1] == X.max()
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    frame = ols.get_prediction(sm.add_constant(grid)).summary_frame(alpha=1 - level)
    np.testing.assert_allclose(curve["y"], frame["mean"], rtol=1e-12)
    np.testing.assert_allclose(curve["ci_low"], frame["mean_ci_lower"], rtol=1e-10)
    np.testing.assert_allclose(curve["ci_high"], frame["mean_ci_upper"], rtol=1e-10)
    np.testing.assert_allclose(curve["pred_low"], frame["obs_ci_lower"], rtol=1e-10)
    np.testing.assert_allclose(curve["pred_high"], frame["obs_ci_upper"], rtol=1e-10)


def test_through_origin_matches_statsmodels_without_constant():
    res = linear_regression(X, Y, through_origin=True)
    ols = sm.OLS(Y, X).fit()
    slope = res.parameter("Slope")
    assert slope.value == pytest.approx(ols.params[0], rel=1e-12)
    assert slope.se == pytest.approx(ols.bse[0], rel=1e-10)
    assert slope.p_value == pytest.approx(ols.pvalues[0], rel=1e-8)
    assert (slope.ci_low, slope.ci_high) == pytest.approx(tuple(ols.conf_int()[0]), rel=1e-10)
    assert res.effect("R squared").value == pytest.approx(ols.rsquared, rel=1e-12)  # uncentred
    assert res.df2 == X.size - 1
    assert res.parameter("Y-intercept") is None and res.parameter("X-intercept") is None
    grid = np.array(res.extra["curve"]["x"])
    frame = ols.get_prediction(grid[:, None]).summary_frame(alpha=0.05)
    np.testing.assert_allclose(res.extra["curve"]["ci_low"], frame["mean_ci_lower"], rtol=1e-9,
                               atol=1e-12)
    np.testing.assert_allclose(res.extra["curve"]["pred_high"], frame["obs_ci_upper"], rtol=1e-9)
    assert "through the origin" in res.test_name and "not comparable" in res.interpretation
    assert res.extra["equation"].count("X") == 1 and "+" not in res.extra["equation"]


def test_slope_tested_against_a_nonzero_value():
    res = linear_regression(X, Y, test_slope_against=0.5)
    slope = res.parameter("Slope")
    t = (slope.value - 0.5) / slope.se
    assert slope.statistic == pytest.approx(t, rel=1e-12)
    assert slope.p_value == pytest.approx(2 * stats.t.sf(abs(t), X.size - 2), rel=1e-10)
    assert res.statistic_name == "t" and res.statistic == pytest.approx(t)
    assert res.df == X.size - 2 and res.df2 is None
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    assert slope.p_value == pytest.approx(float(ols.t_test("x1 = 0.5").pvalue), rel=1e-8)
    assert "0.5" in res.methods_text
    for bad in ("abc", math.inf, None):
        with pytest.raises(StatsError):
            linear_regression(X, Y, test_slope_against=bad)


def test_x_intercept_fieller_limits_are_where_the_band_crosses_zero():
    x = np.arange(2.0, 14.0)
    y = -4.0 + 1.5 * x + np.array([0.3, -0.5, 0.2, 0.6, -0.4, 0.1, -0.2, 0.5, -0.6, 0.2, 0.1, -0.3])
    res = linear_regression(x, y)
    xi = res.parameter("X-intercept")
    slope, intercept = res.parameter("Slope").value, res.parameter("Y-intercept").value
    assert xi.value == pytest.approx(-intercept / slope, rel=1e-12)
    assert xi.ci_low < xi.value < xi.ci_high
    n, xbar = x.size, x.mean()
    sxx = np.sum((x - xbar) ** 2)
    s = res.extra["goodness_of_fit"]["sy_x"]
    tcrit = stats.t.ppf(0.975, n - 2)
    for limit in (xi.ci_low, xi.ci_high):
        half = tcrit * s * math.sqrt(1 / n + (limit - xbar) ** 2 / sxx)
        assert abs(intercept + slope * limit) == pytest.approx(half, rel=1e-9)
    assert "Fieller" in xi.unit_note
    # slope not clearly different from zero -> no finite interval
    flat = linear_regression(x, np.array([1, 3, 2, 4, 1, 3, 2, 4, 1, 3, 2, 4.2]))
    assert flat.parameter("X-intercept").ci_low is None
    assert "no finite CI" in flat.parameter("X-intercept").unit_note
    assert "extrapolated" in flat.parameter("X-intercept").unit_note


def test_breusch_pagan_and_durbin_watson_match_statsmodels():
    x = np.linspace(1, 10, 40)
    y = 1 + 2 * x + RNG.normal(0, 0.2 * x)  # scatter grows with x
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    lm, lm_p, _, _ = het_breuschpagan(ols.resid, sm.add_constant(x))
    mine = breusch_pagan(ols.resid, x)
    assert mine["lm"] == pytest.approx(lm, rel=1e-9) and mine["p_value"] == pytest.approx(lm_p, rel=1e-7)
    assert mine["df"] == 1
    res = linear_regression(x, y)
    check = next(a for a in res.assumptions if "Breusch" in a.name)
    assert check.statistic == pytest.approx(lm, rel=1e-9) and check.status == "fail"
    assert durbin_watson(ols.resid) == pytest.approx(sm_durbin_watson(ols.resid), rel=1e-12)
    assert res.extra["durbin_watson"] == pytest.approx(sm_durbin_watson(ols.resid), rel=1e-9)
    assert breusch_pagan(np.array([1.0, -1.0]), np.array([1.0, 2.0])) is None
    assert breusch_pagan(np.ones(6), np.arange(6.0)) is None
    assert durbin_watson(np.zeros(4)) is None
    multi = breusch_pagan(ols.resid, np.column_stack([x, x**2]))
    lm2, lm2_p, _, _ = het_breuschpagan(ols.resid, np.column_stack([np.ones_like(x), x, x**2]))
    assert multi["lm"] == pytest.approx(lm2, rel=1e-8) and multi["df"] == 2


def _runs_brute_force(n1, n2):
    """Exact distribution of the number of runs by enumerating all arrangements."""
    counts = {}
    for positions in itertools.combinations(range(n1 + n2), n1):
        signs = np.zeros(n1 + n2, dtype=bool)
        signs[list(positions)] = True
        runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
        counts[runs] = counts.get(runs, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


@pytest.mark.parametrize(("n1", "n2"), [(3, 4), (5, 5), (2, 7), (6, 3), (1, 5)])
def test_runs_test_exact_distribution_matches_enumeration(n1, n2):
    dist = _runs_brute_force(n1, n2)
    for runs_obs in sorted(dist):
        # build a sequence with exactly this many runs: enumerate until one is found
        for positions in itertools.combinations(range(n1 + n2), n1):
            signs = -np.ones(n1 + n2)
            signs[list(positions)] = 1.0
            if 1 + int(np.count_nonzero(signs[1:] != signs[:-1])) == runs_obs:
                break
        info = runs_test(signs)
        expected_p = sum(p for r, p in dist.items() if r <= runs_obs)
        assert info["runs"] == runs_obs and info["method"] == "exact"
        assert info["p_value"] == pytest.approx(expected_p, abs=1e-12)
        assert (info["n_positive"], info["n_negative"]) == (n1, n2)
    mean = sum(r * p for r, p in dist.items())
    var = sum((r - mean) ** 2 * p for r, p in dist.items())
    info = runs_test(signs)
    assert info["expected_runs"] == pytest.approx(mean, rel=1e-12)
    if var > 0:
        assert (info["runs"] - info["expected_runs"]) / info["z"] == pytest.approx(math.sqrt(var), rel=1e-9)


def test_runs_test_details_and_normal_approximation():
    from statsmodels.sandbox.stats.runs import runstest_1samp

    seq = RNG.normal(size=60)
    info = runs_test(seq)
    z_sm, _ = runstest_1samp(seq > 0, cutoff=0.5, correction=False)
    assert info["z"] == pytest.approx(z_sm, rel=1e-10)
    assert runs_test([1.0, 2.0, 3.0]) is None                  # all one side
    assert runs_test([0.0, 0.0]) is None
    assert runs_test([1, 0, -1, np.nan, 1])["runs"] == 3        # zeros and NaN are dropped
    clustered = np.r_[np.ones(15), -np.ones(15)]
    assert runs_test(clustered)["p_value"] < 1e-6
    alternating = np.tile([1.0, -1.0], 15)
    assert runs_test(alternating)["p_value"] == pytest.approx(1.0)
    big = np.r_[np.ones(1100), -np.ones(1100)]
    approx = runs_test(big)
    assert approx["method"] == "normal approximation" and approx["p_value"] < 1e-12
    # the normal approximation agrees with the exact value for a moderately large sample
    seq2 = np.where(RNG.random(1800) < 0.5, 1.0, -1.0)
    exact = runs_test(seq2)
    n1, n2 = exact["n_positive"], exact["n_negative"]
    n = n1 + n2
    sd = math.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1)))
    normal_p = stats.norm.cdf((exact["runs"] + 0.5 - exact["expected_runs"]) / sd)
    assert exact["method"] == "exact" and exact["p_value"] == pytest.approx(normal_p, abs=2e-3)


def test_lack_of_fit_matches_anova_of_line_versus_one_mean_per_x():
    x = np.repeat([1.0, 2, 3, 4, 5, 6], 4)
    y = 3 + 0.2 * (x - 3.5) ** 2 + x + RNG.normal(0, 0.4, x.size)   # curved truth
    res = linear_regression(x, y)
    lof = res.extra["lack_of_fit"]
    df = pd.DataFrame({"x": x, "y": y})
    line = smf.ols("y ~ x", df).fit()
    saturated = smf.ols("y ~ C(x)", df).fit()
    table = anova_lm(line, saturated)
    assert lof["f"] == pytest.approx(table["F"][1], rel=1e-9)
    assert lof["p_value"] == pytest.approx(table["Pr(>F)"][1], rel=1e-7)
    assert lof["ss_pure_error"] == pytest.approx(saturated.ssr, rel=1e-10)
    assert (lof["df_lack_of_fit"], lof["df_pure_error"]) == (4, 18)
    check = next(a for a in res.assumptions if "lack-of-fit" in a.name)
    assert check.status == "fail" and "different model" in check.detail
    rows = {row["source"].strip(): row for row in res.anova_table}
    assert rows["Lack of fit"]["ss"] + rows["Pure error"]["ss"] == pytest.approx(rows["Residual"]["ss"])
    assert "mean residual at each X" in next(a for a in res.assumptions if "runs" in a.name).detail
    # no replicates -> no test; identical replicates -> cannot be computed
    assert lack_of_fit(np.arange(6.0), np.arange(6.0) ** 2, np.arange(6.0), 2) is None
    exact = lack_of_fit(np.repeat([1.0, 2, 3], 2), np.repeat([1.0, 4, 9], 2),
                        np.repeat([0.5, 4.5, 8.5], 2), 2)
    assert exact["f"] is None and exact["ss_pure_error"] == 0.0
    weighted = lack_of_fit(x, y, line.fittedvalues.to_numpy(), 2, weights=np.full(x.size, 2.0))
    assert weighted["f"] == pytest.approx(lof["f"], rel=1e-9)


def test_information_criteria_convention():
    res = linear_regression(X, Y)
    gof = res.extra["goodness_of_fit"]
    n, ss, k = X.size, gof["ss_residual"], 3
    assert gof["aic"] == pytest.approx(n * math.log(ss / n) + 2 * k)
    assert gof["aicc"] == pytest.approx(n * math.log(ss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))
    assert gof["bic"] == pytest.approx(n * math.log(ss / n) + k * math.log(n))
    # same model ranking as the likelihood-based AIC of statsmodels: the two differ by a
    # constant that depends on n only (statsmodels does not count the variance parameter)
    ols = sm.OLS(Y, sm.add_constant(X)).fit()
    constant = n * (1 + math.log(2 * math.pi)) - 2
    assert gof["aic"] + constant == pytest.approx(ols.aic, rel=1e-10)
    assert information_criteria(0.0, 10, 2)["aicc"] is None
    assert information_criteria(5.0, 4, 2)["aicc"] is None and information_criteria(5.0, 4, 2)["aic"] is not None


# ------------------------------------------------------------- degenerate inputs
def test_degenerate_inputs():
    two = linear_regression([1, 2], [3, 7])
    assert two.parameter("Slope").value == pytest.approx(4.0)
    assert two.parameter("Slope").se is None and two.p_value is None and two.statistic is None
    assert any("exactly through the data" in w for w in two.warnings)
    assert "ci_low" not in two.extra["curve"]
    three = linear_regression([1, 2, 3], [1, 3, 2])
    assert three.parameter("Slope").se is not None
    assert any("very imprecise" in w for w in three.warnings)
    const_y = linear_regression([1, 2, 3, 4], [5, 5, 5, 5])
    assert const_y.parameter("Slope").value == 0.0 and const_y.effect("R squared").value is None
    assert const_y.p_value is None and "flat" in const_y.interpretation
    assert const_y.parameter("X-intercept").value is None
    perfect = linear_regression([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    assert perfect.parameter("Slope").value == pytest.approx(2.0)
    assert perfect.parameter("Slope").se == 0.0 and perfect.p_value == 0.0
    assert perfect.statistic is None and perfect.effect("R squared").value == 1.0
    assert any("perfect fit" in w for w in perfect.warnings)
    assert perfect.assumptions[0].name == "Residual diagnostics"
    with_nan = linear_regression([1, 2, np.nan, 4, 5, 6], [2, 4, 5, np.nan, 11, 12])
    assert with_nan.n == {"Y": 4} and any("missing" in n for n in with_nan.data_notes)
    origin_one = linear_regression([2.0], [6.0], through_origin=True)
    assert origin_one.parameter("Slope").value == 3.0
    with pytest.raises(StatsError, match="identical"):
        linear_regression([2, 2, 2], [1, 2, 3])
    with pytest.raises(StatsError, match="at least 2"):
        linear_regression([1], [1])
    with pytest.raises(StatsError, match="at least 2"):
        linear_regression([np.nan, 1], [1, np.nan])
    with pytest.raises(StatsError, match="All X values are zero"):
        linear_regression([0, 0, 0], [1, 2, 3], through_origin=True)
    with pytest.raises(StatsError, match="same number"):
        linear_regression([1, 2, 3], [1, 2])
    with pytest.raises(StatsError, match="numbers"):
        linear_regression(["a", "b"], [1, 2])
    with pytest.raises(StatsError, match="too large or too small"):
        linear_regression([1e200, 2e200, 3e200], [1, 2, 4])
    with pytest.raises(StatsError):
        linear_regression(X, Y, alpha=0)
    with pytest.raises(StatsError):
        linear_regression(X, Y, ci_level=1.2)
    horizontal = linear_regression([1, 2, 3, 4], [1, 2, 2, 1])
    assert horizontal.parameter("Slope").value == pytest.approx(0.0, abs=1e-15)
    assert is_constant(np.array([5.0])) and is_constant(np.array([1e-15, 1e-15]))
    assert not is_constant(np.array([1e-15, 3e-15]))


def test_tiny_magnitude_data_are_not_mistaken_for_constants():
    # concentrations in molar: 1 fM .. 10 pM
    x = np.array([1e-15, 1e-14, 1e-13, 1e-12, 1e-11])
    y = 3e-9 * (x / 1e-12) + 1e-10
    res = linear_regression(x, y + np.array([1, -1, 0.5, -0.5, 0.2]) * 1e-11)
    assert res.parameter("Slope").value == pytest.approx(3e3, rel=0.1)
    assert res.effect("R squared").value > 0.99


finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.one_of(finite, st.just(float("nan"))), st.one_of(finite, st.integers(-2, 2).map(float))),
                min_size=0, max_size=12), st.booleans())
def test_property_never_crashes(pairs, through_origin):
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = linear_regression(x, y, through_origin=through_origin)
        except StatsError:
            return
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    for par in res.parameters:
        assert par.p_value is None or 0.0 <= par.p_value <= 1.0
        if par.ci_low is not None and par.ci_high is not None:
            assert par.ci_low <= par.ci_high
    r2 = res.effect("R squared").value
    assert r2 is None or -1e-9 <= r2 <= 1.0 + 1e-9
    assert len(res.extra["curve"]["x"]) == len(res.extra["curve"]["y"])
    assert_strict_json(res)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.integers(-20, 20), st.integers(-20, 20)), min_size=4, max_size=15),
       st.floats(0.1, 50), st.floats(-100, 100), st.floats(0.1, 50), st.floats(-100, 100))
def test_property_location_scale_equivariance(pairs, ax, bx, ay, by):
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    try:
        base = linear_regression(x, y)
        moved = linear_regression(ax * x + bx, ay * y + by)
    except StatsError:
        return
    if base.effect("R squared").value is None or base.parameter("Slope").se in (None, 0.0):
        return
    assert moved.parameter("Slope").value == pytest.approx(base.parameter("Slope").value * ay / ax,
                                                           rel=1e-6, abs=1e-9)
    assert moved.effect("R squared").value == pytest.approx(base.effect("R squared").value, abs=1e-7)
    if base.p_value is not None and base.p_value > 1e-12:
        assert moved.p_value == pytest.approx(base.p_value, rel=1e-5, abs=1e-12)


# --------------------------------------------------------------- compare slopes
def _two_lines():
    rng = np.random.default_rng(3)
    x = np.tile(np.arange(1.0, 11.0), 2)
    ya = 1.0 + 0.9 * x + rng.normal(0, 0.8, x.size)
    yb = 2.5 + 1.4 * x + rng.normal(0, 0.8, x.size)
    yc = 4.0 + 0.9 * x + rng.normal(0, 0.8, x.size)
    return x, ya, yb, yc


def test_compare_slopes_matches_statsmodels_ancova():
    x, ya, yb, yc = _two_lines()
    res = compare_slopes({"A": (x, ya), "B": (x, yb)})
    df = pd.DataFrame({"x": np.r_[x, x], "y": np.r_[ya, yb], "g": ["A"] * x.size + ["B"] * x.size})
    full = smf.ols("y ~ x * g", df).fit()
    common = smf.ols("y ~ x + g", df).fit()
    one = smf.ols("y ~ x", df).fit()
    slopes = anova_lm(common, full)
    assert res.statistic == pytest.approx(slopes["F"][1], rel=1e-9)
    assert res.p_value == pytest.approx(slopes["Pr(>F)"][1], rel=1e-7)
    assert (res.df, res.df2) == (1, 2 * x.size - 4)
    elev = anova_lm(one, common)
    assert res.extra["elevation_test"]["f"] == pytest.approx(elev["F"][1], rel=1e-9)
    assert res.extra["elevation_test"]["p_value"] == pytest.approx(elev["Pr(>F)"][1], rel=1e-7)
    assert res.extra["common_slope"] == pytest.approx(common.params["x"], rel=1e-10)
    assert res.parameter("Common (pooled) slope").se == pytest.approx(common.bse["x"], rel=1e-9)
    assert res.parameter("Slope (A)").value == pytest.approx(stats.linregress(x, ya).slope, rel=1e-12)
    assert res.parameter("Slope (B)").se == pytest.approx(stats.linregress(x, yb).stderr, rel=1e-10)
    assert res.p_value < 0.05 and "differ between the data sets" in res.interpretation
    assert res.extra["elevation_test"]["valid"] is False
    assert res.effect_sizes[0].value == pytest.approx(
        res.statistic / (res.statistic + res.df2), rel=1e-12)
    assert_strict_json(res)


def test_compare_slopes_parallel_lines_three_groups_and_errors():
    x, ya, yb, yc = _two_lines()
    res = compare_slopes({"A": (x, ya), "C": (x, yc)})
    assert res.p_value > 0.05 and "do not differ detectably" in res.interpretation
    assert res.extra["elevation_test"]["p_value"] < 0.001 and res.extra["elevation_test"]["valid"]
    three = compare_slopes({"A": (x, ya), "B": (x[:12], yb[:12]), "C": (x, yc)})
    df = pd.DataFrame({"x": np.r_[x, x[:12], x], "y": np.r_[ya, yb[:12], yc],
                       "g": ["A"] * x.size + ["B"] * 12 + ["C"] * x.size})
    table = anova_lm(smf.ols("y ~ x + g", df).fit(), smf.ols("y ~ x * g", df).fit())
    assert three.statistic == pytest.approx(table["F"][1], rel=1e-9)
    assert three.df == 2 and three.n == {"A": 20, "B": 12, "C": 20}
    exact = compare_slopes({"A": ([1, 2, 3, 4], [2, 4, 6, 8]), "B": ([1, 2, 3, 4], [3, 6, 9, 12])})
    assert exact.p_value is None and any("exactly" in w for w in exact.warnings)
    assert "cannot be tested" in exact.interpretation
    with pytest.raises(StatsError, match="at least two data sets"):
        compare_slopes({"A": (x, ya)})
    with pytest.raises(StatsError, match="different X"):
        compare_slopes({"A": (x, ya), "B": ([1, 1, 1], [1, 2, 3])})
    with pytest.raises(StatsError, match="Too few points"):
        compare_slopes({"A": ([1, 2], [1, 2]), "B": ([1, 2], [2, 3])})
    with pytest.raises(StatsError, match="pair of"):
        compare_slopes({"A": (x, ya), "B": 5})
    with_nan = compare_slopes({"A": (np.r_[x, np.nan], np.r_[ya, 1.0]), "B": (x, yb)})
    assert any(note.startswith("A:") for note in with_nan.data_notes)


# --------------------------------------------------------- multiple regression
def _mv_frame(n=60, seed=8):
    rng = np.random.default_rng(seed)
    dose = rng.uniform(0, 10, n)
    age = rng.normal(40, 8, n)
    sex = rng.choice(["F", "M"], n)
    strain = rng.choice(["wt", "ko", "het"], n)
    y = (5 + 1.2 * dose - 0.1 * age + 2.0 * (sex == "M") + 1.5 * (strain == "ko")
         + 0.4 * dose * (sex == "M") + rng.normal(0, 1.5, n))
    return pd.DataFrame({"y": y, "dose": dose, "age": age, "sex": sex, "strain": strain})


def test_multiple_regression_matches_statsmodels_formula():
    df = _mv_frame()
    res = multiple_regression(df, "y", ["dose", "age", "sex", "strain"], standardize=True)
    ref = smf.ols("y ~ dose + age + C(sex) + C(strain)", df).fit()
    mapping = {"Intercept": "Intercept", "dose": "dose", "age": "age", "sex[M]": "C(sex)[T.M]",
               "strain[ko]": "C(strain)[T.ko]", "strain[wt]": "C(strain)[T.wt]"}
    ci = ref.conf_int()
    for mine, theirs in mapping.items():
        par = res.parameter(mine)
        assert par.value == pytest.approx(ref.params[theirs], rel=1e-9)
        assert par.se == pytest.approx(ref.bse[theirs], rel=1e-9)
        assert par.statistic == pytest.approx(ref.tvalues[theirs], rel=1e-8)
        assert par.p_value == pytest.approx(ref.pvalues[theirs], rel=1e-6)
        assert (par.ci_low, par.ci_high) == pytest.approx(tuple(ci.loc[theirs]), rel=1e-8)
    assert res.statistic == pytest.approx(ref.fvalue, rel=1e-9)
    assert res.p_value == pytest.approx(ref.f_pvalue, rel=1e-6)
    assert (res.df, res.df2) == (ref.df_model, ref.df_resid)
    gof = res.extra["goodness_of_fit"]
    assert gof["r_squared"] == pytest.approx(ref.rsquared, rel=1e-12)
    assert gof["adj_r_squared"] == pytest.approx(ref.rsquared_adj, rel=1e-12)
    n, k = len(df), 7
    assert gof["aicc"] == pytest.approx(
        n * math.log(ref.ssr / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1), rel=1e-12)
    assert gof["aic"] + n * (1 + math.log(2 * math.pi)) - 2 == pytest.approx(ref.aic, rel=1e-10)
    np.testing.assert_allclose(res.extra["residuals"]["residual"], ref.resid, atol=1e-9)
    np.testing.assert_allclose(res.extra["predicted_vs_actual"]["predicted"], ref.fittedvalues, rtol=1e-10)
    np.testing.assert_allclose(res.extra["predicted_vs_actual"]["actual"], df["y"])
    assert res.effect("Cohen's f squared").value == pytest.approx(ref.rsquared / (1 - ref.rsquared))
    r2 = res.effect("R squared")
    assert r2.ci_low < r2.value < r2.ci_high
    assert any("reference level" in note and "'F'" in note for note in res.data_notes)
    assert res.extra["terms"]["strain"] == ["strain[ko]", "strain[wt]"]
    assert res.extra["equation"].startswith("y = ")
    assert_strict_json(res)


def test_multiple_regression_vif_and_standardized_betas():
    df = _mv_frame()
    res = multiple_regression(df, "y", ["dose", "age", "sex"], standardize=True)
    design = np.column_stack([np.ones(len(df)), df["dose"], df["age"], (df["sex"] == "M").astype(float)])
    for j, name in enumerate(["dose", "age", "sex[M]"], start=1):
        assert res.extra["vif"][name] == pytest.approx(variance_inflation_factor(design, j), rel=1e-9)
    # standardized beta = coefficient after z-scoring outcome and predictors
    z = pd.DataFrame({
        "y": stats.zscore(df["y"], ddof=1), "dose": stats.zscore(df["dose"], ddof=1),
        "age": stats.zscore(df["age"], ddof=1),
        "sex": stats.zscore((df["sex"] == "M").astype(float), ddof=1)})
    zfit = smf.ols("y ~ dose + age + sex", z).fit()
    rows = {row["term"]: row for row in res.extra["coefficients"]}
    assert rows["dose"]["standardized_beta"] == pytest.approx(zfit.params["dose"], rel=1e-9)
    assert rows["sex[M]"]["standardized_beta"] == pytest.approx(zfit.params["sex"], rel=1e-9)
    assert res.parameter("Standardized beta: age").value == pytest.approx(zfit.params["age"], rel=1e-9)
    plain = multiple_regression(df, "y", ["dose", "age", "sex"])
    assert plain.parameter("Standardized beta: age") is None
    assert next(a for a in plain.assumptions if "VIF" in a.name).status == "pass"
    single = multiple_regression(df, "y", "dose")
    assert single.extra["vif"] == {"dose": 1.0}
    lr = stats.linregress(df["dose"], df["y"])
    assert single.parameter("dose").value == pytest.approx(lr.slope, rel=1e-10)
    assert single.parameter("dose").se == pytest.approx(lr.stderr, rel=1e-9)


@pytest.mark.parametrize("spec", ["dose*sex", [("dose", "sex")], "dose:sex; age*strain"])
def test_multiple_regression_interactions(spec):
    df = _mv_frame()
    res = multiple_regression(df, "y", ["dose", "age", "sex", "strain"], interactions=spec)
    formula = "y ~ dose + age + C(sex) + C(strain) + dose:C(sex)"
    if isinstance(spec, str) and "strain" in spec:
        formula += " + age:C(strain)"
    ref = smf.ols(formula, df).fit()
    assert res.parameter("dose:sex[M]").value == pytest.approx(ref.params["dose:C(sex)[T.M]"], rel=1e-8)
    assert res.parameter("dose:sex[M]").p_value == pytest.approx(ref.pvalues["dose:C(sex)[T.M]"], rel=1e-6)
    assert res.statistic == pytest.approx(ref.fvalue, rel=1e-8)
    assert "interaction" in res.methods_text
    if isinstance(spec, str) and "strain" in spec:
        assert res.extra["terms"]["age:strain"] == ["age:strain[ko]", "age:strain[wt]"]


def test_multiple_regression_collinearity_warning_and_errors():
    df = _mv_frame()
    df["dose2"] = df["dose"] * 2 + np.random.default_rng(1).normal(0, 0.05, len(df))
    res = multiple_regression(df, "y", ["dose", "dose2", "age"])
    assert next(a for a in res.assumptions if "VIF" in a.name).status == "fail"
    assert any("variance inflation factor above 10" in w for w in res.warnings)
    df["dose3"] = df["dose"] * 3
    with pytest.raises(StatsError, match="perfectly collinear"):
        multiple_regression(df, "y", ["dose", "dose3"])
    df["flat"] = 1.0
    with pytest.raises(StatsError, match="perfectly collinear"):
        multiple_regression(df, "y", ["dose", "flat"])
    with pytest.raises(StatsError, match="at least one predictor"):
        multiple_regression(df, "y", [])
    with pytest.raises(StatsError, match="Unknown column"):
        multiple_regression(df, "y", ["nope"])
    with pytest.raises(StatsError, match="both the outcome and a predictor"):
        multiple_regression(df, "y", ["y", "dose"])
    with pytest.raises(StatsError, match="only once"):
        multiple_regression(df, "y", ["dose", "dose"])
    with pytest.raises(StatsError, match="must be numeric"):
        multiple_regression(df, "sex", ["dose"])
    with pytest.raises(StatsError, match="not among the predictors"):
        multiple_regression(df, "y", ["dose"], interactions="dose*age")
    with pytest.raises(StatsError, match="two different predictors"):
        multiple_regression(df, "y", ["dose"], interactions="dose*dose")
    with pytest.raises(StatsError, match="not understood"):
        multiple_regression(df, "y", ["dose", "age"], interactions="dose*age*sex")
    with pytest.raises(StatsError, match="only one level"):
        multiple_regression(df.assign(one="k"), "y", ["dose", "one"])
    with pytest.raises(StatsError, match="at least"):
        multiple_regression(df.head(3), "y", ["dose", "age"])
    with pytest.raises(StatsError, match="table"):
        multiple_regression([1, 2, 3], "y", ["x"])
    with pytest.raises(StatsError, match="at least"):
        multiple_regression(df.assign(dose=np.nan), "y", ["dose"])


def test_multiple_regression_degenerate_cases():
    df = _mv_frame(20)
    with_missing = df.copy()
    with_missing.loc[[0, 5], "age"] = np.nan
    with_missing.loc[3, "sex"] = None
    with_missing.loc[4, "y"] = np.inf
    res = multiple_regression(with_missing, "y", ["dose", "age", "sex"])
    assert res.n == {"y": 16}
    assert any("4 row(s) with a missing value" in note for note in res.data_notes)
    assert any("rule of thumb" in w for w in res.warnings)
    exact = df.assign(y=3 + 2 * df["dose"] - df["age"])
    perfect = multiple_regression(exact, "y", ["dose", "age"])
    assert perfect.parameter("dose").value == pytest.approx(2.0, rel=1e-9)
    assert perfect.p_value == 0.0 and perfect.statistic is None
    assert any("perfect fit" in w for w in perfect.warnings)
    assert perfect.assumptions[0].name == "Residual diagnostics"
    const = multiple_regression(df.assign(y=4.0), "y", ["dose", "age"])
    assert const.p_value is None and const.effect("R squared").value is None
    assert "does not vary" in const.interpretation
    json.loads(const.finalize().model_dump_json())
    moderate = df.assign(mix=df["dose"] + np.random.default_rng(2).normal(0, 1.1, len(df)))
    check = next(a for a in multiple_regression(moderate, "y", ["dose", "mix"]).assumptions
                 if "VIF" in a.name)
    assert check.status in {"pass", "warn"}


@settings(max_examples=30, deadline=None)
@given(st.integers(4, 25), st.integers(0, 10_000), st.booleans())
def test_property_multiple_regression_never_crashes(n, seed, with_ties):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, n).astype(float) if with_ties else rng.normal(size=n)
    b = rng.normal(size=n)
    y = rng.integers(0, 2, n).astype(float) if with_ties else rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": a, "b": b, "g": rng.choice(["u", "v"], n)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = multiple_regression(df, "y", ["a", "b", "g"])
        except StatsError:
            return
    assert res.p_value is None or 0 <= res.p_value <= 1
    assert all(p.p_value is None or 0 <= p.p_value <= 1 for p in res.parameters)
    assert_strict_json(res)


# ----------------------------------------------------------------------- registry
def _xy_table():
    x = np.arange(1.0, 9.0)
    rng = np.random.default_rng(4)
    a = np.column_stack([1 + 2 * x + rng.normal(0, 0.5, x.size) for _ in range(3)])
    b = np.column_stack([3 + 2.1 * x + rng.normal(0, 0.5, x.size) for _ in range(3)])
    a[2, 1] = np.nan
    return XYTable.from_arrays("lines", x, {"Control": a, "Treated": b}, x_name="Time")


def test_registry_linear_regression():
    table = _xy_table()
    res = run_analysis(table, "linear_regression")
    assert res.analysis == "linear_regression" and res.table == "lines"
    assert res.extra["series"] == "Control" and res.n == {"Control": 23}
    assert res.interpretation and res.why_chosen and res.assumptions and res.significant is True
    assert any("individual point" in note for note in res.data_notes)
    assert {"x", "y", "ci_low", "ci_high", "pred_low", "pred_high"} <= set(res.extra["curve"])
    assert isinstance(res.extra["curve"]["x"], list)
    json.loads(res.model_dump_json())
    x, y = table.xy_points("Control")
    assert res.parameter("Slope").value == pytest.approx(stats.linregress(x, y).slope, rel=1e-12)
    means = run_analysis(table, "linear_regression", series="Treated", use_means=True,
                         through_origin=True, test_slope_against=2.0, alpha=0.01, ci_level=0.99)
    assert means.n == {"Treated": 8} and means.alpha == 0.01
    assert means.parameter("Slope").ci_level == 0.99
    assert any("mean of the replicates" in note for note in means.data_notes)
    with pytest.raises(StatsError, match="Unknown series"):
        run_analysis(table, "linear_regression", series="Nope")
    table.exclude([(0, "Control:1")])
    assert run_analysis(table, "linear_regression").n == {"Control": 22}


def test_registry_compare_slopes():
    table = _xy_table()
    res = run_analysis(table, "compare_slopes")
    assert res.n == {"Control": 23, "Treated": 24}
    assert res.interpretation and res.why_chosen and res.assumptions
    json.loads(res.model_dump_json())
    same = run_analysis(table, "compare_slopes", series="Control, Treated", use_means=True,
                        ci_level=0.9)
    assert same.n == {"Control": 8, "Treated": 8} and same.parameters[0].ci_level == 0.9
    with pytest.raises(StatsError, match="at least two"):
        run_analysis(table, "compare_slopes", series=["Control"])
    with pytest.raises(StatsError, match="Unknown series"):
        run_analysis(table, "compare_slopes", series=["Control", "X"])


def test_registry_multiple_regression():
    table = MultipleVariablesTable("mv", _mv_frame(40))
    res = run_analysis(table, "multiple_regression")
    assert res.extra["outcome"] == "y" and res.extra["predictors"] == ["dose", "age", "sex", "strain"]
    assert res.interpretation and res.why_chosen and res.assumptions and res.significant is True
    json.loads(res.model_dump_json())
    chosen = run_analysis(table, "multiple_regression", y="age", predictors="dose, sex",
                          interactions="dose*sex", standardize=True, ci_level=0.9)
    assert chosen.extra["outcome"] == "age" and "dose:sex[M]" in chosen.extra["vif"]
    assert chosen.parameter("dose").ci_level == 0.9
    with pytest.raises(StatsError, match="Unknown column"):
        run_analysis(table, "multiple_regression", predictors=["nope"])
    text_only = MultipleVariablesTable("t", pd.DataFrame({"a": list("xyzxyz"), "b": list("uvwuvw")}))
    with pytest.raises(StatsError, match="no numeric column"):
        run_analysis(text_only, "multiple_regression")
    table.exclude([(0, "*"), (1, "dose")])
    assert run_analysis(table, "multiple_regression").n == {"y": 38}


# =============================================================================
# Audit part 2 (C1): offset and unit invariance of the regression computations
# =============================================================================
def _line_summary(res):
    slope = res.parameter("Slope")
    gof = res.extra["goodness_of_fit"]
    return [res.statistic, res.p_value, slope.statistic, slope.p_value, gof["r_squared"], gof["adj_r_squared"],
            res.extra["durbin_watson"], res.extra["breusch_pagan"]["lm"], res.extra["breusch_pagan"]["p_value"],
            *[a.p_value for a in res.assumptions if a.p_value is not None]]


_OFF_X = np.arange(1.0, 13.0)
_OFF_Y = np.array([2.3, 3.1, 7.2, 7.9, 9.4, 13.8, 13.1, 17.6, 17.2, 21.5, 21.9, 25.0])


@pytest.mark.parametrize("offset", [1e9, 1e12, 1e15])
def test_linear_regression_does_not_depend_on_a_large_offset_in_x(offset):
    """X = offset + (1..12): exactly representable, and slope, SE, tests, residuals, bands and
    the Fieller interval are invariant to adding a constant to X. A plain x - mean(x) loses the
    centring at 1e15 (the running sum of twelve values exceeds 2**53)."""
    big = _OFF_X + offset
    assert np.array_equal(big - offset, _OFF_X)
    base, res = linear_regression(_OFF_X, _OFF_Y), linear_regression(big, _OFF_Y)
    assert _line_summary(res) == pytest.approx(_line_summary(base), rel=1e-9)
    for name in ("value", "se", "ci_low", "ci_high"):
        assert getattr(res.parameter("Slope"), name) == pytest.approx(getattr(base.parameter("Slope"), name), rel=1e-9)
    assert res.extra["residuals"]["residual"] == pytest.approx(base.extra["residuals"]["residual"], rel=1e-9, abs=1e-12)
    for key in ("y", "ci_low", "ci_high", "pred_low", "pred_high"):
        assert res.extra["curve"][key] == pytest.approx(base.extra["curve"][key], rel=1e-9)
    # X-intercept and its Fieller limits shift by the offset (to the resolution of doubles there)
    xi, xb = res.parameter("X-intercept"), base.parameter("X-intercept")
    resolution = 4 * np.spacing(offset)
    assert xi.value - offset == pytest.approx(xb.value, abs=resolution)
    assert xi.ci_low - offset == pytest.approx(xb.ci_low, abs=resolution)
    assert xi.ci_high - offset == pytest.approx(xb.ci_high, abs=resolution)
    # exact rational check of the slope from the centred sums
    dx, dy = _OFF_X - _OFF_X.mean(), _OFF_Y - _OFF_Y.mean()
    assert res.parameter("Slope").value == pytest.approx(float(dx @ dy) / float(dx @ dx), rel=1e-12)
    assert_strict_json(res)


def test_linear_regression_large_offset_in_y_too():
    base = linear_regression(_OFF_X, _OFF_Y)
    res = linear_regression(_OFF_X + 1e12, np.round(_OFF_Y * 8) / 8 + 1e12)
    ref = linear_regression(_OFF_X, np.round(_OFF_Y * 8) / 8)
    assert res.parameter("Slope").value == pytest.approx(ref.parameter("Slope").value, rel=1e-9)
    assert res.p_value == pytest.approx(ref.p_value, rel=1e-8)
    assert res.extra["goodness_of_fit"]["r_squared"] == pytest.approx(ref.extra["goodness_of_fit"]["r_squared"], rel=1e-9)
    assert base.p_value is not None


@pytest.mark.parametrize("sx, sy", [(1e-13, 1.0), (1.0, 1e-13), (1e9, 1e-9), (1e100, 1e100), (1e-100, 1e100)])
def test_linear_regression_tests_do_not_depend_on_the_units(sx, sy):
    base = linear_regression(_OFF_X, _OFF_Y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = linear_regression(_OFF_X * sx, _OFF_Y * sy)
    assert _line_summary(res) == pytest.approx(_line_summary(base), rel=1e-8)
    assert res.parameter("Slope").value == pytest.approx(base.parameter("Slope").value * sy / sx, rel=1e-9)
    assert res.parameter("Y-intercept").value == pytest.approx(base.parameter("Y-intercept").value * sy, rel=1e-9)
    assert res.parameter("X-intercept").value == pytest.approx(base.parameter("X-intercept").value * sx, rel=1e-9)


@pytest.mark.parametrize("sx, sy, offset", [(1.0, 1.0, 1e12), (1e-13, 1e-13, 0.0), (1e9, 1e100, 0.0)])
def test_compare_slopes_does_not_depend_on_units_or_offset(sx, sy, offset):
    rng = np.random.default_rng(12)
    x = np.arange(1.0, 11.0)
    ya, yb = 2.0 * x + rng.normal(0, 1, 10), 1.2 * x + 3 + rng.normal(0, 1, 10)
    base = compare_slopes({"a": (x, ya), "b": (x, yb)})
    res = compare_slopes({"a": (x * sx + offset, ya * sy), "b": (x * sx + offset, yb * sy)})
    assert res.statistic == pytest.approx(base.statistic, rel=1e-8)
    assert res.p_value == pytest.approx(base.p_value, rel=1e-7)
    elev, ref = res.extra["elevation_test"], base.extra["elevation_test"]
    assert elev["f"] == pytest.approx(ref["f"], rel=1e-8)
    assert res.assumptions[0].p_value == pytest.approx(base.assumptions[0].p_value, rel=1e-6)
    assert res.parameters[0].value == pytest.approx(base.parameters[0].value * sy / sx, rel=1e-9)


def test_centre_is_accurate_with_a_huge_offset_and_constant_is_ulp_based():
    from torusplot.stats.regression import centre

    x = np.arange(1.0, 13.0) + 1e15
    mean, dev = centre(x)
    assert dev == pytest.approx(np.arange(1.0, 13.0) - 6.5, abs=1e-12)  # a plain x - x.mean() is off by 0.5
    assert mean == pytest.approx(1e15 + 6.5, abs=0.25)
    assert not is_constant(x) and not is_constant(np.array([1e-200, 3e-200]))
    assert is_constant(np.array([0.1 + 0.2, 0.3])) and is_constant(np.array([1e15, 1e15]))
    with pytest.raises(StatsError, match="identical"):
        linear_regression([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    # Breusch-Pagan and Durbin-Watson are unit- and offset-free
    resid = np.array([0.3, -0.2, 0.5, -0.6, 0.1, 0.4, -0.5, 0.2])
    z = np.arange(8.0)
    ref = breusch_pagan(resid, z)
    for scale, shift in ((1e-150, 0.0), (1e150, 0.0), (1.0, 1e15)):
        got = breusch_pagan(resid * scale, z + shift)
        assert got["lm"] == pytest.approx(ref["lm"], rel=1e-9) and got["p_value"] == pytest.approx(ref["p_value"], rel=1e-9)
        assert durbin_watson(resid * scale) == pytest.approx(durbin_watson(resid), rel=1e-12)
