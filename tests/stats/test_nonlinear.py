"""Nonlinear regression: model library, fitter, intervals, diagnostics, model comparison.

Reference values: published Bates & Watts (1988) examples (Puromycin, BOD); cross-checks
against scipy.optimize.curve_fit / least_squares, statsmodels and first-principles
formulas written here; synthetic-data parameter recovery and CI coverage. The NIST StRD
benchmarks live in test_nist_nls.py.
"""

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
from scipy import optimize, stats
from statsmodels.stats.anova import anova_lm

from torusplot.data.tables import XYTable
from torusplot.stats import run_analysis
from torusplot.stats._utils import StatsError
from torusplot.stats.nonlinear import (
    MODEL_KEYS,
    MODELS,
    compare_datasets,
    compare_fits,
    fit_curve,
    get_model,
    parse_assignments,
)
from torusplot.stats.regression import compare_slopes, linear_regression


def assert_strict_json(res):
    """The result must serialise to strict JSON: no NaN / Infinity anywhere."""
    json.dumps(json.loads(res.finalize().model_dump_json()), allow_nan=False)


LN10 = math.log(10)


def hill(lx, bottom, top, logec50, slope):
    return bottom + (top - bottom) / (1 + 10 ** ((logec50 - lx) * slope))


def dose_data(seed=1, sd=3.0, reps=3, logec50=-6.3, slope=1.1, bottom=4.0, top=96.0):
    rng = np.random.default_rng(seed)
    lx = np.repeat(np.arange(-9.0, -3.75, 0.5), reps)
    y = hill(lx, bottom, top, logec50, slope) + rng.normal(0, sd, lx.size)
    return lx, y


#: (x values, true parameters) used for recovery / cross-checks of every built-in model
CASES = {
    "dose_response_4pl_logx": (np.linspace(-9, -4, 11),
                               {"Bottom": 2, "Top": 98, "LogEC50": -6.4, "HillSlope": 1.3}),
    "dose_response_4pl": (np.r_[0, 10 ** np.linspace(-2, 2, 11)],
                          {"Bottom": 2, "Top": 98, "LogEC50": 0.3, "HillSlope": -0.8}),
    "dose_response_3pl_logx": (np.linspace(-9, -4, 11),
                               {"Bottom": 2, "Top": 98, "LogEC50": -6.4, "HillSlope": -1}),
    "dose_response_3pl": (10 ** np.linspace(-2, 2, 11),
                          {"Bottom": 2, "Top": 98, "LogEC50": 0.3, "HillSlope": 1}),
    "michaelis_menten": (np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0]), {"Vmax": 120, "Km": 6}),
    "one_phase_decay": (np.array([0, 1, 2, 4, 8, 16, 32, 64.0]),
                        {"Y0": 100, "Plateau": 10, "K": 0.15}),
    "one_phase_association": (np.array([0, 1, 2, 4, 8, 16, 32, 64.0]),
                              {"Y0": 5, "Plateau": 80, "K": 0.2}),
    "one_site_binding_specific": (np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0]), {"Bmax": 1500, "Kd": 4}),
    "one_site_binding_total": (np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64.0]),
                               {"Bmax": 1500, "Kd": 4, "NS": 6, "Background": 40}),
    "exponential_growth": (np.linspace(0, 8, 9), {"Y0": 3, "K": 0.6}),
    "gaussian": (np.linspace(-3, 7, 15), {"Amplitude": 50, "Mean": 2.2, "SD": 1.4}),
    "linear": (np.linspace(0, 10, 8), {"Intercept": 3, "Slope": -0.7}),
    "quadratic": (np.linspace(0, 10, 8), {"B0": 3, "B1": -0.7, "B2": 0.05}),
}


def model_values(key, x, params):
    m = MODELS[key]
    tx = m.x_transform(x) if m.x_transform is not None else x
    return m.func(tx, np.array([params[p] for p in m.parameters], dtype=float))


def noisy(key, seed=0, rel=0.03, reps=2):
    x, true = CASES[key]
    x = np.repeat(x, reps)
    clean = model_values(key, x, true)
    rng = np.random.default_rng(seed)
    return x, clean + rng.normal(0, rel * (np.max(clean) - np.min(clean)), x.size), true


# ------------------------------------------------------------ published examples
PUROMYCIN_CONC = [0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10]
PUROMYCIN_RATE = [76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200]


def test_puromycin_michaelis_menten_bates_watts():
    # Treloar (1974) data, treated enzyme; Bates & Watts (1988) Example Puromycin:
    # Vmax = 212.68 (SE 6.947), Km = 0.06412 (SE 0.008281), residual SE 10.93 on 10 df.
    res = fit_curve(PUROMYCIN_CONC, PUROMYCIN_RATE, "michaelis_menten")
    vmax, km = res.parameter("Vmax"), res.parameter("Km")
    assert round(vmax.value, 2) == 212.68 and round(vmax.se, 3) == 6.947
    assert round(km.value, 5) == 0.06412 and round(km.se, 6) == 0.008281
    assert round(res.extra["goodness_of_fit"]["sy_x"], 2) == 10.93
    assert res.df == 10 and res.extra["convergence"]["converged"]
    assert res.extra["convergence"]["restarts"] == 0
    # replicates at every concentration -> lack-of-fit F test is reported
    assert res.extra["lack_of_fit"]["df_pure_error"] == 6
    assert any("lack-of-fit" in a.name for a in res.assumptions)
    assert "Vmax = 212.7" in res.interpretation


def test_bod_bates_watts():
    # Bates & Watts (1988), BOD data (Appendix 1, A1.4): theta = (19.143, 0.5311),
    # residual SS 25.99, s^2 = 6.498 on 4 df.
    x, y = [1, 2, 3, 4, 5, 7], [8.3, 10.3, 19.0, 16.0, 15.6, 19.8]
    res = fit_curve(x, y, "user_defined", equation="Y = t1 * (1 - exp(-t2 * X))",
                    initial={"t1": 20, "t2": 0.24})
    assert round(res.parameter("t1").value, 3) == 19.143
    assert round(res.parameter("t2").value, 4) == 0.5311
    gof = res.extra["goodness_of_fit"]
    assert round(gof["ss"], 2) == 25.99 and round(gof["ss"] / gof["df"], 3) == 6.498 and gof["df"] == 4
    built_in = fit_curve(x, y, "one_phase_association", fixed={"Y0": 0})
    assert built_in.parameter("Plateau").value == pytest.approx(res.parameter("t1").value, rel=1e-9)
    assert built_in.parameter("K").value == pytest.approx(res.parameter("t2").value, rel=1e-9)
    assert built_in.parameter("Y0").se is None and "Fixed" in built_in.parameter("Y0").unit_note


# ------------------------------------------------- recovery and cross-implementation
@pytest.mark.parametrize("key", list(CASES))
def test_noise_free_parameter_recovery(key):
    x, true = CASES[key]
    y = model_values(key, x, true)
    res = fit_curve(x, y, key)
    assert res.extra["convergence"]["converged"]
    for name in MODELS[key].parameters:
        assert res.parameter(name).value == pytest.approx(true[name], rel=1e-6, abs=1e-9), name
    assert res.effect("R squared").value == pytest.approx(1.0, abs=1e-12)
    assert any("perfect fit" in w for w in res.warnings)
    assert res.assumptions[2].name == "Residual diagnostics"
    assert_strict_json(res)


@pytest.mark.parametrize("key", list(CASES))
def test_analytic_jacobians_match_finite_differences(key):
    x, true = CASES[key]
    m = MODELS[key]
    tx = m.x_transform(x) if m.x_transform is not None else x
    tx = tx[np.isfinite(tx)]
    p0 = np.array([true[p] for p in m.parameters], dtype=float) * 1.07 + 0.013
    jac = m.jac(tx, p0)
    num = np.empty_like(jac)
    for i in range(p0.size):
        h = 1e-6 * max(1.0, abs(p0[i]))
        up, dn = p0.copy(), p0.copy()
        up[i] += h
        dn[i] -= h
        num[:, i] = (m.func(tx, up) - m.func(tx, dn)) / (2 * h)
    np.testing.assert_allclose(jac, num, rtol=1e-6, atol=1e-7 * np.max(np.abs(jac)))


@pytest.mark.parametrize("key", list(CASES))
def test_noisy_fit_matches_scipy_curve_fit(key):
    x, y, true = noisy(key)
    m = MODELS[key]
    res = fit_curve(x, y, key)
    free = [p for p in m.parameters if p not in res.extra["fixed"]]
    fixed = {p: res.extra["fixed"][p] for p in m.parameters if p in res.extra["fixed"]}

    def f(xv, *theta):
        full = {**dict(zip(free, theta)), **fixed}
        return model_values(key, xv, full)

    p0 = [res.parameter(p).value * 1.02 + 1e-3 for p in free]
    popt, pcov = optimize.curve_fit(f, x, y, p0=p0, xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                    maxfev=20000)
    for i, name in enumerate(free):
        par = res.parameter(name)
        # documented tolerance for two independent iterative fits: 1e-6 (values), 1e-4 (SE;
        # curve_fit differentiates numerically)
        assert par.value == pytest.approx(popt[i], rel=1e-6, abs=1e-8), name
        assert par.se == pytest.approx(math.sqrt(pcov[i, i]), rel=1e-4), name
    ss_ref = float(np.sum((y - f(x, *popt)) ** 2))
    gof = res.extra["goodness_of_fit"]
    assert gof["ss"] == pytest.approx(ss_ref, rel=1e-9)
    n, p = x.size, len(free)
    assert gof["df"] == n - p and res.df == n - p
    assert gof["sy_x"] == pytest.approx(math.sqrt(ss_ref / (n - p)), rel=1e-9)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert gof["r_squared"] == pytest.approx(1 - ss_ref / ss_tot, rel=1e-10)
    assert gof["adj_r_squared"] == pytest.approx(1 - (ss_ref / (n - p)) / (ss_tot / (n - 1)), rel=1e-10)
    k = p + 1
    assert gof["k"] == k
    assert gof["aicc"] == pytest.approx(n * math.log(ss_ref / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                                        rel=1e-9)
    assert gof["bic"] == pytest.approx(n * math.log(ss_ref / n) + k * math.log(n), rel=1e-9)
    # correlation matrix of the estimates
    sd = np.sqrt(np.diag(pcov))
    np.testing.assert_allclose(np.array(res.extra["parameter_correlation"]["matrix"], dtype=float),
                               pcov / np.outer(sd, sd), atol=2e-4)
    # asymptotic CI = estimate +/- t * SE
    tcrit = stats.t.ppf(0.975, n - p)
    first = res.parameter(free[0])
    assert first.ci_high - first.value == pytest.approx(tcrit * first.se, rel=1e-10)
    # the automatic starting values were good enough: no restarts, true values recovered
    assert res.extra["convergence"]["restarts"] == 0
    for name in free:
        assert abs(res.parameter(name).value - true[name]) < 6 * res.parameter(name).se + 1e-9


def test_linear_model_equals_linear_regression_including_bands():
    x, y, _ = noisy("linear", seed=3, reps=3)
    res = fit_curve(x, y, "linear")
    lin = linear_regression(x, y)
    assert res.parameter("Slope").value == pytest.approx(lin.parameter("Slope").value, rel=1e-10)
    assert res.parameter("Slope").se == pytest.approx(lin.parameter("Slope").se, rel=1e-9)
    assert res.parameter("Intercept").ci_low == pytest.approx(lin.parameter("Y-intercept").ci_low, rel=1e-9)
    for key in ("x", "y", "ci_low", "ci_high", "pred_low", "pred_high"):
        np.testing.assert_allclose(res.extra["curve"][key], lin.extra["curve"][key], rtol=1e-8,
                                   atol=1e-10)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    frame = ols.get_prediction(sm.add_constant(np.array(res.extra["curve"]["x"]))).summary_frame()
    np.testing.assert_allclose(res.extra["curve"]["ci_low"], frame["mean_ci_lower"], rtol=1e-8)
    np.testing.assert_allclose(res.extra["curve"]["pred_high"], frame["obs_ci_upper"], rtol=1e-8)
    assert res.extra["lack_of_fit"]["f"] == pytest.approx(lin.extra["lack_of_fit"]["f"], rel=1e-8)
    assert res.extra["runs_test"]["runs"] == lin.extra["runs_test"]["runs"]
    assert res.extra["goodness_of_fit"]["aicc"] == pytest.approx(lin.extra["goodness_of_fit"]["aicc"])


def test_confidence_band_is_the_delta_method():
    lx, y = dose_data()
    res = fit_curve(lx, y, "dose_response_4pl_logx")
    popt, pcov = optimize.curve_fit(hill, lx, y, p0=[0, 100, -6, 1], xtol=1e-14, ftol=1e-14)
    grid = np.array(res.extra["curve"]["x"])
    assert grid.size == 200 and grid[0] == lx.min() and grid[-1] == lx.max()
    np.testing.assert_allclose(res.extra["curve"]["y"], hill(grid, *popt), rtol=1e-7)
    grad = np.empty((grid.size, 4))
    for i in range(4):
        h = 1e-6
        up, dn = popt.copy(), popt.copy()
        up[i] += h
        dn[i] -= h
        grad[:, i] = (hill(grid, *up) - hill(grid, *dn)) / (2 * h)
    se_fit = np.sqrt(np.einsum("ij,jk,ik->i", grad, pcov, grad))
    tcrit = stats.t.ppf(0.975, lx.size - 4)
    half = (np.array(res.extra["curve"]["ci_high"]) - np.array(res.extra["curve"]["ci_low"])) / 2
    np.testing.assert_allclose(half, tcrit * se_fit, rtol=2e-4)
    s2 = res.extra["goodness_of_fit"]["sy_x"] ** 2
    half_pred = (np.array(res.extra["curve"]["pred_high"]) - np.array(res.extra["curve"]["pred_low"])) / 2
    np.testing.assert_allclose(half_pred, tcrit * np.sqrt(se_fit**2 + s2), rtol=2e-4)
    assert np.all(half_pred > half)


def test_ci_coverage_simulation():
    # 200 simulated dose-response experiments: the 95% CI of LogEC50 and of the Hill slope
    # should contain the truth about 95% of the time (loose bounds: 0.89 to 0.99)
    hits = {"LogEC50": 0, "HillSlope": 0, "Top": 0}
    truth = {"LogEC50": -6.3, "HillSlope": 1.1, "Top": 96.0}
    n_sim = 200
    for seed in range(n_sim):
        lx, y = dose_data(seed=1000 + seed, sd=4.0)
        res = fit_curve(lx, y, "dose_response_4pl_logx")
        assert res.extra["convergence"]["converged"]
        for name in hits:
            par = res.parameter(name)
            hits[name] += par.ci_low <= truth[name] <= par.ci_high
    for name, count in hits.items():
        assert 0.89 <= count / n_sim <= 0.99, (name, count)


# ------------------------------------------------------------- dose-response details
def test_ec50_is_transformed_from_logec50_with_asymmetric_ci():
    lx, y = dose_data()
    res = fit_curve(lx, y, "dose_response_4pl_logx")
    log_par, ec = res.parameter("LogEC50"), res.parameter("EC50")
    assert ec.value == pytest.approx(10**log_par.value, rel=1e-12)
    assert ec.ci_low == pytest.approx(10**log_par.ci_low, rel=1e-12)
    assert ec.ci_high == pytest.approx(10**log_par.ci_high, rel=1e-12)
    assert ec.ci_high - ec.value > ec.value - ec.ci_low           # asymmetric on the linear scale
    assert ec.se == pytest.approx(LN10 * ec.value * log_par.se, rel=1e-10)   # delta method
    assert "asymmetric" in ec.unit_note
    span = res.parameter("Span")
    assert span.value == pytest.approx(res.parameter("Top").value - res.parameter("Bottom").value)
    corr = res.extra["parameter_correlation"]
    i, j = corr["labels"].index("Top"), corr["labels"].index("Bottom")
    var = (res.parameter("Top").se ** 2 + res.parameter("Bottom").se ** 2
           - 2 * corr["matrix"][i][j] * res.parameter("Top").se * res.parameter("Bottom").se)
    assert span.se == pytest.approx(math.sqrt(var), rel=1e-9)
    assert "units" in res.interpretation and "EC50" in res.interpretation
    assert res.extra["x_is_log"] is True and res.extra["model"] == "dose_response_4pl_logx"
    assert res.extra["equation"] == "Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) * HillSlope))"


def test_linear_x_model_is_the_same_fit_as_log_x_model():
    lx, y = dose_data(seed=5)
    conc = 10**lx
    a = fit_curve(lx, y, "dose_response_4pl_logx")
    b = fit_curve(conc, y, "dose_response_4pl")
    for name in ("Bottom", "Top", "LogEC50", "HillSlope", "EC50"):
        assert b.parameter(name).value == pytest.approx(a.parameter(name).value, rel=1e-8)
        assert b.parameter(name).se == pytest.approx(a.parameter(name).se, rel=1e-6)
        assert b.parameter(name).ci_low == pytest.approx(a.parameter(name).ci_low, rel=1e-6)
    assert b.extra["goodness_of_fit"]["ss"] == pytest.approx(a.extra["goodness_of_fit"]["ss"], rel=1e-10)
    assert b.parameter("EC50").value > 0 and b.parameter("EC50").ci_low > 0   # log-symmetric CI
    assert b.extra["x_is_log"] is False and b.extra["x_log_spaced"] is True
    grid = np.array(b.extra["curve"]["x"])
    assert grid[0] == pytest.approx(conc.min()) and grid[-1] == pytest.approx(conc.max())
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0])                       # logarithmic spacing
    assert b.extra["equation"] == "Y = Bottom + (Top - Bottom) / (1 + (EC50 / X)^HillSlope)"
    # agrees with an independent curve_fit of the textbook linear-X equation
    def four_pl(x, bottom, top, ec50, slope):
        return bottom + (top - bottom) / (1 + (ec50 / x) ** slope)
    popt, _ = optimize.curve_fit(four_pl, conc, y, p0=[0, 100, 1e-6, 1], xtol=1e-14, ftol=1e-14)
    assert b.parameter("EC50").value == pytest.approx(popt[2], rel=1e-6)
    assert b.parameter("HillSlope").value == pytest.approx(popt[3], rel=1e-6)


def test_zero_dose_is_handled_without_nan():
    lx, y = dose_data(seed=2)
    conc = np.r_[0.0, 0.0, 0.0, 10**lx]
    yy = np.r_[3.5, 5.0, 4.2, y]
    res = fit_curve(conc, yy, "dose_response_4pl")
    assert res.extra["convergence"]["converged"]
    assert all(np.isfinite(res.extra["residuals"]["fitted"]))
    # the fitted value at X = 0 is the plateau the curve starts from
    assert res.extra["residuals"]["fitted"][0] == pytest.approx(res.parameter("Bottom").value, rel=1e-12)
    curve_x = res.extra["curve"]["x"]
    assert curve_x[0] == 0.0 and all(np.isfinite(res.extra["curve"]["y"]))
    assert all(np.isfinite(res.extra["curve"]["ci_low"]))
    assert any("X = 0" in note for note in res.data_notes)
    m = MODELS["dose_response_4pl"]
    tx = m.x_transform(np.array([0.0, 1.0]))
    assert m.func(tx, np.array([1.0, 9.0, 0.0, 2.0]))[0] == 1.0     # Hill > 0 -> Bottom
    assert m.func(tx, np.array([1.0, 9.0, 0.0, -2.0]))[0] == 9.0    # Hill < 0 -> Top
    assert m.func(tx, np.array([1.0, 9.0, 0.0, 0.0]))[0] == 5.0     # Hill = 0 -> midpoint
    assert np.all(np.isfinite(m.jac(tx, np.array([1.0, 9.0, 0.0, 2.0]))))


def test_inhibition_curve_and_ic50_labels():
    lx, y = dose_data(seed=7, slope=-1.2)
    res = fit_curve(lx, y, "dose_response_4pl_logx", kind="IC50")
    names = [p.name for p in res.parameters]
    assert names == ["Bottom", "Top", "LogIC50", "HillSlope", "IC50", "Span"]
    assert res.parameter("HillSlope").value < 0 and res.parameter("Top").value > res.parameter("Bottom").value
    assert "IC50" in res.interpretation and "falls as the dose increases" in res.interpretation
    assert "LogIC50" in res.extra["equation"] and "LogIC50" in res.extra["parameter_correlation"]["labels"]
    plain = fit_curve(lx, y, "dose_response_4pl_logx")
    assert plain.parameter("EC50").value == pytest.approx(res.parameter("IC50").value, rel=1e-12)
    with pytest.raises(StatsError, match="kind"):
        fit_curve(lx, y, "dose_response_4pl_logx", kind="ED50")


def test_three_parameter_models_and_direction():
    lx, y = dose_data(seed=9, slope=-1.0)
    auto = fit_curve(lx, y, "dose_response_3pl_logx")
    assert auto.extra["fixed"] == {"HillSlope": -1.0}
    assert any("fixed at -1" in note for note in auto.data_notes)
    assert auto.parameter("HillSlope").se is None and auto.df == lx.size - 3
    assert "fixed at -1" in auto.interpretation
    inhib = fit_curve(lx, y, "dose_response_3pl_logx", direction="inhibition")
    assert inhib.parameter("LogEC50").value == pytest.approx(auto.parameter("LogEC50").value, rel=1e-10)
    # same as the 4-parameter model with the slope fixed by hand
    by_hand = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"HillSlope": -1})
    assert by_hand.extra["goodness_of_fit"]["ss"] == pytest.approx(auto.extra["goodness_of_fit"]["ss"], rel=1e-10)
    # the wrong direction gives the mirror-image solution with Top below Bottom
    wrong = fit_curve(lx, y, "dose_response_3pl_logx", direction="stimulation")
    assert wrong.parameter("Top").value < wrong.parameter("Bottom").value
    assert wrong.extra["goodness_of_fit"]["ss"] == pytest.approx(auto.extra["goodness_of_fit"]["ss"], rel=1e-8)
    up_x, up_y = dose_data(seed=9, slope=1.0)
    assert fit_curve(10**up_x, up_y, "dose_response_3pl").extra["fixed"] == {"HillSlope": 1.0}
    with pytest.raises(StatsError, match="direction"):
        fit_curve(lx, y, "dose_response_3pl_logx", direction="sideways")


def test_fixed_parameters_bounds_and_aliases():
    lx, y = dose_data(seed=11)
    res = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"Bottom": 0, "Top": 100})
    assert res.extra["fixed"] == {"Bottom": 0.0, "Top": 100.0} and res.df == lx.size - 2
    popt, pcov = optimize.curve_fit(lambda x, c, h: hill(x, 0, 100, c, h), lx, y, p0=[-6, 1],
                                    xtol=1e-14, ftol=1e-14)
    assert res.parameter("LogEC50").value == pytest.approx(popt[0], rel=1e-7)
    assert res.parameter("HillSlope").se == pytest.approx(math.sqrt(pcov[1, 1]), rel=1e-4)
    assert res.parameter("Span").value == 100.0 and res.parameter("Span").se is None
    assert "Bottom fixed at 0" in res.methods_text
    assert res.extra["parameter_correlation"]["labels"] == ["LogEC50", "HillSlope"]
    # text form, case-insensitive names, EC50 alias converted to its logarithm
    text = fit_curve(lx, y, "dose_response_4pl_logx", fixed="bottom=0; TOP: 100")
    assert text.parameter("LogEC50").value == pytest.approx(res.parameter("LogEC50").value, rel=1e-10)
    alias = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"EC50": 1e-6})
    assert alias.extra["fixed"] == {"LogEC50": -6.0} and alias.parameter("EC50").value == pytest.approx(1e-6)
    assert alias.parameter("EC50").se is None
    init = fit_curve(lx, y, "dose_response_4pl_logx", initial="IC50=1e-7, Hill=2, Top=90")
    assert init.extra["initial_values"]["LogEC50"] == -7.0 and init.extra["initial_values"]["HillSlope"] == 2.0
    assert init.parameter("LogEC50").value == pytest.approx(
        fit_curve(lx, y, "dose_response_4pl_logx").parameter("LogEC50").value, rel=1e-8)
    # bounds: an inactive bound changes nothing; an active one is reported
    loose = fit_curve(lx, y, "dose_response_4pl_logx", bounds={"HillSlope": (0, None)})
    assert loose.parameter("HillSlope").value == pytest.approx(
        fit_curve(lx, y, "dose_response_4pl_logx").parameter("HillSlope").value, rel=1e-7)
    assert loose.extra["bounds"] == {"HillSlope": [0.0, None]}
    # (an upper bound alone would be evaded by the mirror solution Top <-> Bottom, Hill -> -Hill)
    tight = fit_curve(lx, y, "dose_response_4pl_logx", bounds={"HillSlope": (0, 0.5), "EC50": (1e-9, 1e-3)})
    assert tight.parameter("HillSlope").value == pytest.approx(0.5)
    assert tight.extra["convergence"]["parameters_at_bound"] == ["HillSlope"]
    assert any("stopped at the bound" in w for w in tight.warnings)
    assert tight.extra["bounds"]["LogEC50"] == [-9.0, -3.0]
    ref = optimize.curve_fit(lambda x, b, t, c: hill(x, b, t, c, 0.5), lx, y, p0=[0, 100, -6])[0]
    assert tight.parameter("LogEC50").value == pytest.approx(ref[2], rel=1e-5)
    ignored = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"Bottom": 0}, initial={"Bottom": 5})
    assert any("ignored because the parameter is fixed" in w for w in ignored.warnings)
    out = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"HillSlope": -3}, bounds={"HillSlope": (0, 5)})
    assert any("outside its bounds" in w for w in out.warnings)


def test_option_errors():
    lx, y = dose_data()
    cases = [
        ({"fixed": {"Nope": 1}}, "not a parameter"),
        ({"fixed": {"Bottom": 0, "bottom": 1}}, "twice"),
        ({"fixed": {"EC50": -1}}, "greater than zero"),
        ({"fixed": {"Bottom": float("inf")}}, "finite"),
        ({"fixed": {"Bottom": "abc"}}, "not a number"),
        ({"fixed": "Bottom"}, "Could not understand"),
        ({"fixed": "Bottom=zero"}, "not a number"),
        ({"fixed": 5}, "text such as"),
        ({"fixed": {"Bottom": 0, "Top": 1, "LogEC50": 2, "HillSlope": 1}}, "nothing left to fit"),
        ({"bounds": {"HillSlope": 3}}, "pair"),
        ({"bounds": {"HillSlope": ("a", 1)}}, "numbers or None"),
        ({"bounds": {"HillSlope": (2, 1)}}, "below the upper"),
        ({"bounds": [1, 2]}, "dict"),
        ({"bounds": {"Zed": (0, 1)}}, "not a parameter"),
        ({"weighting": "1/x"}, "weighting"),
        ({"ci_method": "bootstrap"}, "ci_method"),
        ({"alpha": 2}, "alpha"),
        ({"ci_level": 0}, "ci_level"),
        ({"equation": "A*x"}, "not 'user_defined'"),
    ]
    for kwargs, message in cases:
        with pytest.raises(StatsError, match=message):
            fit_curve(lx, y, "dose_response_4pl_logx", **kwargs)
    with pytest.raises(StatsError, match="Unknown model"):
        fit_curve(lx, y, "five_pl")
    with pytest.raises(StatsError, match="needs an equation"):
        fit_curve(lx, y, "user_defined")
    with pytest.raises(StatsError, match="no parameters"):
        fit_curve(lx, y, "user_defined", equation="2*x")
    assert parse_assignments(None) == {} and parse_assignments("  ") == {}
    assert parse_assignments("A=1, B : 2e-3;\nC=-4") == {"A": 1.0, "B": 0.002, "C": -4.0}
    assert parse_assignments({"A": "3"}) == {"A": 3.0}
    assert get_model(MODELS["linear"]) is MODELS["linear"]
    assert MODEL_KEYS[-1] == "user_defined" and len(MODEL_KEYS) == len(MODELS) + 1
    for model in MODELS.values():   # library metadata is complete
        assert set(model.descriptions) == set(model.parameters)
        assert model.label and model.summary and model.equation.startswith("Y = ")


# --------------------------------------------------------------------- weighting
@pytest.mark.parametrize(("weighting", "sigma"), [("1/y", lambda y: np.sqrt(np.abs(y))),
                                                  ("1/y^2", lambda y: np.abs(y))])
def test_weighted_fit_matches_curve_fit_sigma(weighting, sigma):
    rng = np.random.default_rng(4)
    x = np.repeat([0.5, 1, 2, 4, 8, 16, 32, 64.0], 3)
    clean = 120 * x / (6 + x)
    y = clean * (1 + rng.normal(0, 0.08, x.size))
    res = fit_curve(x, y, "michaelis_menten", weighting=weighting)
    popt, pcov = optimize.curve_fit(lambda xv, v, k: v * xv / (k + xv), x, y, p0=[100, 5],
                                    sigma=sigma(y), absolute_sigma=False, xtol=1e-14, ftol=1e-14)
    assert res.parameter("Vmax").value == pytest.approx(popt[0], rel=1e-7)
    assert res.parameter("Km").value == pytest.approx(popt[1], rel=1e-7)
    assert res.parameter("Km").se == pytest.approx(math.sqrt(pcov[1, 1]), rel=1e-4)
    w = 1 / sigma(y) ** 2
    fitted = popt[0] * x / (popt[1] + x)
    wss = float(np.sum(w * (y - fitted) ** 2))
    assert res.extra["goodness_of_fit"]["ss"] == pytest.approx(wss, rel=1e-9)
    ybar = np.sum(w * y) / np.sum(w)
    assert res.effect("R squared").value == pytest.approx(1 - wss / np.sum(w * (y - ybar) ** 2), rel=1e-9)
    assert "weighted_residual" in res.extra["residuals"]
    np.testing.assert_allclose(res.extra["residuals"]["weighted_residual"],
                               np.sqrt(w) * (y - fitted), atol=1e-6)
    assert any("observed Y" in note for note in res.data_notes)
    assert weighting in res.methods_text
    unweighted = fit_curve(x, y, "michaelis_menten")
    assert unweighted.parameter("Km").value != pytest.approx(res.parameter("Km").value, rel=1e-6)
    # prediction band widens with Y under these weightings
    half = np.array(res.extra["curve"]["pred_high"]) - np.array(res.extra["curve"]["pred_low"])
    assert half[-1] > half[0]


def test_sd_weighting_and_weighting_errors():
    x = np.array([1, 2, 4, 8, 16, 32.0])
    y = np.array([17.5, 29.0, 49.5, 67.0, 88.0, 100.5])
    sd = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    res = fit_curve(x, y, "michaelis_menten", weighting="1/sd^2", sd=sd)
    popt, pcov = optimize.curve_fit(lambda xv, v, k: v * xv / (k + xv), x, y, p0=[100, 5],
                                    sigma=sd, absolute_sigma=False, xtol=1e-14, ftol=1e-14)
    assert res.parameter("Vmax").value == pytest.approx(popt[0], rel=1e-7)
    assert res.parameter("Vmax").se == pytest.approx(math.sqrt(pcov[0, 0]), rel=1e-4)
    assert "pred_low" not in res.extra["curve"] and "ci_low" in res.extra["curve"]
    with pytest.raises(StatsError, match="needs the standard deviation"):
        fit_curve(x, y, "michaelis_menten", weighting="1/sd^2")
    with pytest.raises(StatsError, match="positive SD"):
        fit_curve(x, y, "michaelis_menten", weighting="1/sd^2", sd=[1, 0, 1, 1, 1, 1])
    with pytest.raises(StatsError, match="one to one"):
        fit_curve(x, y, "michaelis_menten", weighting="1/sd^2", sd=[1, 2])
    with pytest.raises(StatsError, match="exactly zero"):
        fit_curve(x, np.r_[0.0, y[1:]], "michaelis_menten", weighting="1/y")
    nan_sd = fit_curve(np.r_[x, np.nan], np.r_[y, 5.0], "michaelis_menten", weighting="1/sd^2",
                       sd=np.r_[sd, 1.0])
    assert nan_sd.parameter("Vmax").value == pytest.approx(res.parameter("Vmax").value, rel=1e-10)


# ------------------------------------------------------------- derived quantities
def test_half_life_tau_and_span():
    x, y, _ = noisy("one_phase_decay", seed=6, reps=3)
    res = fit_curve(x, y, "one_phase_decay")
    k = res.parameter("K")
    half, tau, span = res.parameter("HalfLife"), res.parameter("Tau"), res.parameter("Span")
    assert half.value == pytest.approx(math.log(2) / k.value, rel=1e-12)
    assert tau.value == pytest.approx(1 / k.value, rel=1e-12)
    assert (half.ci_low, half.ci_high) == pytest.approx((math.log(2) / k.ci_high, math.log(2) / k.ci_low))
    assert half.se == pytest.approx(math.log(2) / k.value**2 * k.se, rel=1e-10)
    assert span.value == pytest.approx(res.parameter("Y0").value - res.parameter("Plateau").value)
    assert span.ci_low < span.value < span.ci_high
    assert "half-life" in res.interpretation
    assoc_x, assoc_y, _ = noisy("one_phase_association", seed=6, reps=3)
    assoc = fit_curve(assoc_x, assoc_y, "one_phase_association")
    assert assoc.parameter("Span").value == pytest.approx(
        assoc.parameter("Plateau").value - assoc.parameter("Y0").value)
    gx, gy, _ = noisy("exponential_growth", seed=6, reps=3)
    growth = fit_curve(gx, gy, "exponential_growth")
    assert growth.parameter("DoublingTime").value == pytest.approx(
        math.log(2) / growth.parameter("K").value, rel=1e-12)
    assert "doubling time" in growth.interpretation
    px, py, _ = noisy("gaussian", seed=6, reps=3)
    peak = fit_curve(px, py, "gaussian")
    assert peak.parameter("FWHM").value == pytest.approx(2.354820045 * peak.parameter("SD").value, rel=1e-9)
    assert peak.extra["bounds"] == {"SD": [0.0, None]}


def test_derived_quantities_when_rate_is_not_positive_or_ci_spans_zero():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7.0])
    rng = np.random.default_rng(3)
    # decaying data fitted with the growth model: K is negative, so there is no doubling time
    decaying = 10 * np.exp(-0.02 * x) + rng.normal(0, 0.6, x.size)
    res = fit_curve(x, decaying, "exponential_growth")
    assert res.parameter("K").value < 0
    assert res.parameter("DoublingTime").value is None and res.parameter("Tau").value is None
    assert "not positive" in res.parameter("DoublingTime").unit_note
    assert any("opposite way" in w for w in res.warnings)
    # slow, noisy growth: K > 0 but its CI includes zero -> no finite upper limit
    growing = 10 * np.exp(0.02 * x) + rng.normal(0, 0.6, x.size)
    slow = fit_curve(x, growing, "exponential_growth")
    k, dt = slow.parameter("K"), slow.parameter("DoublingTime")
    assert k.value > 0 and k.ci_low < 0
    assert dt.value == pytest.approx(math.log(2) / k.value) and dt.ci_high is None
    assert dt.ci_low == pytest.approx(math.log(2) / k.ci_high)
    assert "no finite upper limit" in dt.unit_note
    assert any("larger than its value" in w for w in slow.warnings)
    # a negative Km has no physical meaning and is reported as such
    odd = fit_curve([1, 2, 3, 4, 5.0], [9.0, 4.1, 2.9, 2.6, 2.3], "michaelis_menten")
    assert odd.parameter("Km").value < 0
    assert any("no physical meaning" in w for w in odd.warnings)
    json.loads(odd.finalize().model_dump_json())


# ------------------------------------------------------------------- diagnostics
def test_wrong_model_is_flagged_by_runs_and_lack_of_fit_tests():
    x = np.repeat(np.arange(1.0, 13.0), 3)
    rng = np.random.default_rng(2)
    y = 5 + 0.5 * (x - 6) ** 2 + rng.normal(0, 1.0, x.size)      # a parabola
    res = fit_curve(x, y, "linear")
    status = {a.name: a.status for a in res.assumptions}
    assert status["No systematic deviation (runs test)"] == "fail"
    assert status["Model adequacy (replicates lack-of-fit F test)"] == "fail"
    assert "diagnostics flag a problem" in res.interpretation
    good = fit_curve(x, y, "quadratic")
    status_good = {a.name: a.status for a in good.assumptions}
    assert status_good["No systematic deviation (runs test)"] == "pass"
    assert status_good["Model adequacy (replicates lack-of-fit F test)"] == "pass"
    # lack of fit against an independent computation
    lof = good.extra["lack_of_fit"]
    df = pd.DataFrame({"x": x, "y": y})
    table = anova_lm(smf.ols("y ~ x + I(x**2)", df).fit(), smf.ols("y ~ C(x)", df).fit())
    assert lof["f"] == pytest.approx(table["F"][1], rel=1e-7)
    assert lof["p_value"] == pytest.approx(table["Pr(>F)"][1], rel=1e-6)


def test_scatter_check_suggests_weighting():
    rng = np.random.default_rng(8)
    x = np.repeat([0.5, 1, 2, 4, 8, 16, 32, 64.0], 6)
    clean = 5 + 500 * x / (6 + x)
    y = clean * (1 + rng.normal(0, 0.15, x.size))
    res = fit_curve(x, y, "one_site_binding_total", fixed={"NS": 0})
    check = next(a for a in res.assumptions if a.name.startswith("Constant scatter"))
    assert check.status == "warn" and "relative weighting" in check.detail
    weighted = fit_curve(x, y, "one_site_binding_total", fixed={"NS": 0}, weighting="1/y^2")
    assert next(a for a in weighted.assumptions if a.name.startswith("Constant scatter")).status == "pass"
    few = fit_curve(x[::10], y[::10], "michaelis_menten")
    assert next(a for a in few.assumptions if a.name.startswith("Constant scatter")).status == "info"


def test_identifiability_warnings_for_an_incomplete_curve():
    # only the bottom third of the curve was measured: the EC50 lies beyond the last dose
    lx = np.repeat(np.arange(-9.0, -6.4, 0.5), 3)
    rng = np.random.default_rng(1)
    y = hill(lx, 2, 100, -6.0, 1.0) + rng.normal(0, 1.0, lx.size)
    res = fit_curve(lx, y, "dose_response_4pl_logx")
    joined = " ".join(res.warnings)
    assert "is extrapolated" in joined
    assert "Top plateau is not well defined" in joined
    check = next(a for a in res.assumptions if "identifiability" in a.name)
    assert check.status == "warn"
    assert "poorly determined" in res.interpretation
    # fixing the unsupported plateau, as the warning suggests, tames the fit
    fixed = fit_curve(lx, y, "dose_response_4pl_logx", fixed={"Top": 100})
    assert fixed.parameter("LogEC50").se < res.parameter("LogEC50").se
    complete_x, complete_y = dose_data()
    ok = fit_curve(complete_x, complete_y, "dose_response_4pl_logx")
    assert next(a for a in ok.assumptions if "identifiability" in a.name).status == "pass"
    assert not ok.warnings


def test_dependency_and_rank_deficiency():
    lx, y = dose_data(seed=3)
    res = fit_curve(lx, y, "dose_response_4pl_logx")
    # dependency_i = 1 - 1 / VIF_i of the Jacobian columns
    m = MODELS["dose_response_4pl_logx"]
    theta = np.array([res.parameter(p).value for p in m.parameters])
    jac = m.jac(lx, theta)
    inv = np.linalg.inv(jac.T @ jac)
    expected = 1 - 1 / (np.diag(inv) * np.diag(jac.T @ jac))
    got = [res.extra["parameter_dependency"][p] for p in m.parameters]
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    # a product of two parameters cannot be separated
    x = np.arange(1.0, 9.0)
    amb = fit_curve(x, 2 * x + np.sin(5 * x), "user_defined", equation="A*B*x")
    assert amb.parameter("A").se is None and amb.extra["parameter_correlation"] is None
    assert any("not all identifiable" in w for w in amb.warnings)
    assert "ci_low" not in amb.extra["curve"]
    assert amb.parameter("A").value * amb.parameter("B").value == pytest.approx(
        float(np.sum(x * (2 * x + np.sin(5 * x))) / np.sum(x * x)), rel=1e-8)
    assert any("1.0 was used" in w for w in amb.warnings)


def test_x_scale_warning_for_log_model_given_concentrations():
    lx, y = dose_data()
    res = fit_curve(10**lx * 1e9, y, "dose_response_4pl_logx")      # nM, not log
    assert any("look like concentrations" in w for w in res.warnings)
    with pytest.raises(StatsError, match="negative"):
        fit_curve(lx, y, "dose_response_4pl")                         # logs given to the linear model
    with pytest.raises(StatsError, match="two different non-zero"):
        fit_curve([0, 0, 5, 5, 5], [1, 2, 3, 4, 5], "dose_response_4pl", fixed={"Bottom": 0, "Top": 9})


# ----------------------------------------------------------- profile likelihood
def test_profile_interval_equals_t_interval_for_a_linear_model():
    x, y, _ = noisy("quadratic", seed=12, reps=2)
    asym = fit_curve(x, y, "quadratic")
    prof = fit_curve(x, y, "quadratic", ci_method="profile")
    for name in ("B0", "B1", "B2"):
        a, p = asym.parameter(name), prof.parameter(name)
        assert p.ci_low == pytest.approx(a.ci_low, rel=1e-6, abs=1e-8)
        assert p.ci_high == pytest.approx(a.ci_high, rel=1e-6, abs=1e-8)
        assert "profile likelihood" in p.unit_note
    assert prof.extra["ci_method"] == "profile" and "profile-likelihood" in prof.methods_text
    assert any("Venzon" in c for c in prof.citations)


def test_profile_limits_sit_exactly_on_the_ss_threshold():
    x = np.array(PUROMYCIN_CONC)
    y = np.array(PUROMYCIN_RATE, dtype=float)
    res = fit_curve(x, y, "michaelis_menten", ci_method="profile")
    asym = fit_curve(x, y, "michaelis_menten")
    n, p = x.size, 2
    ss_best = res.extra["goodness_of_fit"]["ss"]
    target = ss_best * (1 + stats.f.ppf(0.95, 1, n - p) / (n - p))

    def profiled_ss_km(km):
        # Vmax enters linearly: its conditional least-squares estimate is closed form
        basis = x / (km + x)
        vmax = float(basis @ y) / float(basis @ basis)
        return float(np.sum((y - vmax * basis) ** 2))

    def profiled_ss_vmax(vmax):
        best = optimize.minimize_scalar(lambda km: float(np.sum((y - vmax * x / (km + x)) ** 2)),
                                        bounds=(1e-4, 1.0), method="bounded",
                                        options={"xatol": 1e-12})
        return float(best.fun)

    km = res.parameter("Km")
    assert profiled_ss_km(km.ci_low) == pytest.approx(target, rel=1e-7)
    assert profiled_ss_km(km.ci_high) == pytest.approx(target, rel=1e-7)
    vmax = res.parameter("Vmax")
    assert profiled_ss_vmax(vmax.ci_low) == pytest.approx(target, rel=1e-6)
    assert profiled_ss_vmax(vmax.ci_high) == pytest.approx(target, rel=1e-6)
    # the likelihood interval of Km is asymmetric (longer upper arm), unlike the asymptotic one
    assert km.ci_high - km.value > km.value - km.ci_low
    assert asym.parameter("Km").ci_high - asym.parameter("Km").value == pytest.approx(
        asym.parameter("Km").value - asym.parameter("Km").ci_low, rel=1e-9)
    assert km.se == pytest.approx(asym.parameter("Km").se, rel=1e-12)


def test_profile_interval_transforms_to_ec50_and_handles_open_ended_and_bounded_cases():
    lx, y = dose_data(seed=21)
    res = fit_curve(lx, y, "dose_response_4pl_logx", ci_method="profile")
    log_par, ec = res.parameter("LogEC50"), res.parameter("EC50")
    assert (ec.ci_low, ec.ci_high) == pytest.approx((10**log_par.ci_low, 10**log_par.ci_high))
    asym = fit_curve(lx, y, "dose_response_4pl_logx")
    assert log_par.ci_low == pytest.approx(asym.parameter("LogEC50").ci_low, abs=0.05)
    # incomplete curve: the upper limit of LogEC50 / Top is not determined by the data
    part = np.repeat(np.arange(-9.0, -6.4, 0.5), 2)
    rng = np.random.default_rng(5)
    yp = hill(part, 2, 100, -6.0, 1.0) + rng.normal(0, 2.0, part.size)
    open_res = fit_curve(part, yp, "dose_response_4pl_logx", ci_method="profile")
    assert open_res.parameter("Top").ci_high is None
    assert any("open-ended" in w for w in open_res.warnings)
    # a bound that cuts the profile off becomes the limit
    x = np.array(PUROMYCIN_CONC)
    yv = np.array(PUROMYCIN_RATE, dtype=float)
    bounded = fit_curve(x, yv, "michaelis_menten", ci_method="profile", bounds={"Km": (0.055, None)})
    assert bounded.parameter("Km").ci_low == pytest.approx(0.055)
    assert any("set by the bound" in w for w in bounded.warnings)
    one = fit_curve(x, yv, "michaelis_menten", ci_method="profile", fixed={"Vmax": 212.68})
    assert one.parameter("Km").ci_low < one.parameter("Km").value < one.parameter("Km").ci_high
    exact = fit_curve(x, 200 * x / (0.05 + x), "michaelis_menten", ci_method="profile")
    assert any("need residual scatter" in w for w in exact.warnings)


# ------------------------------------------------------------- degenerate inputs
def test_degenerate_inputs():
    x = np.arange(1.0, 9.0)
    with pytest.raises(StatsError, match="only 3 usable point"):
        fit_curve(x[:3], [1, 3, 8], "dose_response_4pl_logx")
    with pytest.raises(StatsError, match="only 0 usable"):
        fit_curve([], [], "linear")
    with pytest.raises(StatsError, match="only 0 usable"):
        fit_curve([np.nan, 1], [1, np.nan], "linear")
    with pytest.raises(StatsError, match="All X values are identical"):
        fit_curve(np.ones(5), np.arange(5.0), "michaelis_menten")
    with pytest.raises(StatsError, match="same number"):
        fit_curve([1, 2, 3], [1, 2], "linear")
    with pytest.raises(StatsError, match="numbers"):
        fit_curve(["a", "b"], [1, 2], "linear")
    with pytest.raises(StatsError, match="undefined values"):
        fit_curve(x, x, "user_defined", equation="A*log(x - B)", initial={"A": 1, "B": 20})
    exact = fit_curve(x[:4], [1, 3, 8, 9], "dose_response_4pl_logx")      # N == P
    assert exact.df == 0 and exact.parameter("Top").se is None
    assert any("exactly as many points" in w for w in exact.warnings)
    assert "ci_low" not in exact.extra["curve"]
    const = fit_curve(x, np.full(8, 5.0), "dose_response_4pl_logx")
    assert any("All Y values are identical" in w for w in const.warnings)
    assert const.effect("R squared").value is None
    with_nan = fit_curve(np.r_[x, np.nan], np.r_[2 * x + np.sin(x), 3.0], "linear")
    assert with_nan.n == {"Y": 8} and any("missing" in note for note in with_nan.data_notes)
    few = fit_curve(x[:3], [2.0, 2.9, 3.6], "michaelis_menten")           # df = 1: no AICc
    assert any("AICc is undefined" in w for w in few.warnings)
    bad = fit_curve(x, 10 - x + np.sin(x), "dose_response_4pl_logx", fixed={"Bottom": 20, "Top": 30})
    assert any("R squared is negative" in w for w in bad.warnings)
    tiny = fit_curve(x * 1e-15, 5 + 2e15 * x * 1e-15 + np.sin(x), "linear")   # fM-scale X
    assert tiny.parameter("Slope").value == pytest.approx(2e15, rel=0.2)
    json.loads(const.finalize().model_dump_json())
    json.loads(exact.finalize().model_dump_json())


def test_unfittable_data_report_non_convergence_instead_of_crashing():
    x = np.arange(1.0, 9.0)
    y = np.array([5, 5.1, 4.9, 5, 5.05, 4.95, 5.02, 4.98])        # flat noise, 4 parameters
    res = fit_curve(x, y, "dose_response_4pl_logx")
    assert res.interpretation and res.extra["convergence"]["restarts"] >= 0
    if not res.extra["convergence"]["converged"]:
        assert res.warnings[0].startswith("The fit did NOT converge")
        assert next(a for a in res.assumptions if a.name == "Convergence").status == "fail"
        assert "did not converge" in res.interpretation
    # a straight line fitted with a saturation curve: Vmax and Km run off together
    line = fit_curve(x[:5], [1.0, 2.1, 2.9, 4.2, 5.0], "michaelis_menten")
    assert line.warnings
    json.loads(line.finalize().model_dump_json())


def test_pathological_inputs_do_not_leak_warnings_or_non_finite_numbers():
    # cases found by fuzzing: overflow inside the covariance / profile / grid computations
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = compare_fits([0.0, -1.0, 154.0, 155.0], [1.0, 0.0, 0.0, 0.0],
                         "dose_response_3pl_logx", "dose_response_4pl_logx")
        b = fit_curve([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], "exponential_growth", ci_method="profile")
        c = compare_datasets({"A": ([0.0, 1.0, 2.0], [0.0, 0.0, 1.0]),
                              "B": ([1.0, 2.7e-303, 1e6, 5.0], [0.0, 0.0, 0.0, 1.0])}, "linear")
        d = fit_curve([1e-300, 1.0, 1e300, 5.0, 7.0], [1.0, 2.0, 3.0, 2.5, 2.7], "michaelis_menten")
    for res in (a, b, c, d):
        assert_strict_json(res)
        assert res.interpretation


@settings(max_examples=30, deadline=None)
@given(
    st.sampled_from(["dose_response_4pl_logx", "dose_response_4pl", "michaelis_menten",
                     "one_phase_decay", "exponential_growth", "gaussian", "linear",
                     "one_site_binding_total", "dose_response_3pl_logx"]),
    st.lists(st.tuples(st.one_of(st.integers(0, 8).map(float), st.floats(0.01, 100), st.just(float("nan"))),
                       st.one_of(st.integers(-3, 3).map(float), st.floats(-100, 100), st.floats(1, 50))),
             min_size=2, max_size=12),
    st.sampled_from(["none", "none", "1/y", "1/y^2"]),
)
def test_property_never_crashes(model, pairs, weighting):
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = fit_curve(x, y, model, weighting=weighting)
        except StatsError:
            return
    assert res.p_value is None
    gof = res.extra["goodness_of_fit"]
    assert gof["ss"] >= 0 and gof["df"] >= 0
    for par in res.parameters:
        if par.ci_low is not None and par.ci_high is not None:
            assert par.ci_low <= par.ci_high
        assert par.se is None or par.se >= 0
    assert len(res.extra["curve"]["x"]) == len(res.extra["curve"]["y"])
    for a in res.assumptions:
        assert a.p_value is None or 0 <= a.p_value <= 1
    assert_strict_json(res)


@settings(max_examples=25, deadline=None)
@given(st.floats(0.5, 20), st.floats(-50, 50), st.integers(0, 1000))
def test_property_y_scale_and_shift_equivariance(scale, shift, seed):
    lx, y = dose_data(seed=seed, sd=2.0, reps=2)
    base = fit_curve(lx, y, "dose_response_4pl_logx")
    moved = fit_curve(lx, scale * y + shift, "dose_response_4pl_logx")
    assert moved.parameter("LogEC50").value == pytest.approx(base.parameter("LogEC50").value, abs=1e-6)
    assert moved.parameter("HillSlope").value == pytest.approx(base.parameter("HillSlope").value, rel=1e-5)
    assert moved.parameter("Top").value == pytest.approx(scale * base.parameter("Top").value + shift,
                                                         rel=1e-6, abs=1e-6)
    assert moved.effect("R squared").value == pytest.approx(base.effect("R squared").value, abs=1e-9)


# ------------------------------------------------------------------ compare_fits
def test_compare_fits_f_test_and_aicc_for_3pl_versus_4pl():
    lx, y = dose_data(seed=31, slope=1.9)        # clearly steeper than the standard slope
    res = compare_fits(lx, y, "dose_response_3pl_logx", "dose_response_4pl_logx")
    p3 = optimize.curve_fit(lambda x, b, t, c: hill(x, b, t, c, 1.0), lx, y, p0=[0, 100, -6],
                            xtol=1e-14, ftol=1e-14)[0]
    p4 = optimize.curve_fit(hill, lx, y, p0=[0, 100, -6, 1], xtol=1e-14, ftol=1e-14)[0]
    ss3 = float(np.sum((y - hill(lx, *p3, 1.0)) ** 2))
    ss4 = float(np.sum((y - hill(lx, *p4)) ** 2))
    n = lx.size
    f = ((ss3 - ss4) / 1) / (ss4 / (n - 4))
    assert res.statistic == pytest.approx(f, rel=1e-7)
    assert (res.df, res.df2) == (1, n - 4)
    assert res.p_value == pytest.approx(stats.f.sf(f, 1, n - 4), rel=1e-6)
    aicc3 = n * math.log(ss3 / n) + 2 * 4 + 2 * 4 * 5 / (n - 5)
    aicc4 = n * math.log(ss4 / n) + 2 * 5 + 2 * 5 * 6 / (n - 6)
    ak = res.extra["akaike"]
    assert ak["delta_aicc"] == pytest.approx(aicc4 - aicc3, rel=1e-7)
    w4 = math.exp(-0.5 * (aicc4 - min(aicc3, aicc4)))
    w3 = math.exp(-0.5 * (aicc3 - min(aicc3, aicc4)))
    assert ak["probability_complex"] == pytest.approx(w4 / (w3 + w4), rel=1e-7)
    assert ak["probability_simple"] + ak["probability_complex"] == pytest.approx(1.0)
    assert ak["evidence_ratio"] == pytest.approx(max(w3, w4) / min(w3, w4), rel=1e-6)
    assert res.extra["preferred"] == "complex" and res.extra["preferred_basis"] == "F test"
    assert res.extra["nested"] is True and res.extra["model"] == "dose_response_4pl_logx"
    assert "fits significantly better" in res.interpretation and "AICc" in res.interpretation
    assert [p.name for p in res.parameters][:4] == ["Bottom", "Top", "LogEC50", "HillSlope"]
    assert res.extra["fits"]["simple"]["fixed"] == {"HillSlope": 1.0}
    assert len(res.extra["comparison"]) == 2 and res.anova_table[0]["f"] == pytest.approx(f, rel=1e-7)
    assert {"x", "y", "ci_low"} <= set(res.extra["curve"])
    assert_strict_json(res)


def test_compare_fits_keeps_the_simpler_model_when_the_gain_is_small():
    lx, y = dose_data(seed=32, slope=1.0)
    res = compare_fits(lx, y, "dose_response_3pl_logx", "dose_response_4pl_logx", alpha=0.01)
    assert res.p_value > 0.01 and res.extra["preferred"] == "simple"
    assert "simpler model is preferred" in res.interpretation and "unless P < 0.01" in res.interpretation
    assert res.extra["model"] == "dose_response_3pl_logx"
    # linear-X pair as well
    res_lin = compare_fits(10**lx, y, "dose_response_3pl", "dose_response_4pl")
    assert res_lin.extra["nested"] is True and res_lin.statistic is not None


def test_compare_fits_line_versus_quadratic_matches_statsmodels():
    x, y, _ = noisy("quadratic", seed=40, reps=3)
    res = compare_fits(x, y, "linear", "quadratic")
    df = pd.DataFrame({"x": x, "y": y})
    table = anova_lm(smf.ols("y ~ x", df).fit(), smf.ols("y ~ x + I(x**2)", df).fit())
    assert res.statistic == pytest.approx(table["F"][1], rel=1e-8)
    assert res.p_value == pytest.approx(table["Pr(>F)"][1], rel=1e-6)
    # arguments in the wrong order are swapped, with a note
    swapped = compare_fits(x, y, "quadratic", "linear")
    assert swapped.statistic == pytest.approx(res.statistic, rel=1e-10)
    assert swapped.extra["models_swapped"] and any("swapped" in w for w in swapped.warnings)


def test_compare_fits_same_model_with_a_constraint():
    lx, y = dose_data(seed=33, bottom=4.0)
    res = compare_fits(lx, y, "dose_response_4pl_logx", "dose_response_4pl_logx",
                       fixed_simple={"Bottom": 0})
    assert res.extra["nested"] is True and (res.df, res.df2) == (1, lx.size - 4)
    assert "more parameters fixed" in res.effect_sizes[0].target
    both = compare_fits(lx, y, "dose_response_4pl_logx", "dose_response_4pl_logx",
                        fixed_simple={"Bottom": 0, "Top": 100}, fixed_complex={"Bottom": 0})
    assert both.extra["nested"] is True and both.df == 1
    # different fixed VALUES are not nested
    not_nested = compare_fits(lx, y, "dose_response_4pl_logx", "dose_response_4pl_logx",
                              fixed_simple={"Bottom": 0, "Top": 100}, fixed_complex={"Bottom": 5})
    assert not_nested.extra["nested"] is False and not_nested.p_value is None
    same = compare_fits(lx, y, "dose_response_4pl_logx", "dose_response_4pl_logx")
    assert same.p_value is None and same.extra["nested"] is False


def test_compare_fits_non_nested_models_use_aicc_only():
    x, y, _ = noisy("one_phase_association", seed=41, reps=3)
    res = compare_fits(x, y, "michaelis_menten", "one_phase_association")
    assert res.statistic is None and res.p_value is None and res.anova_table == []
    assert res.extra["nested"] is False and res.extra["preferred_basis"] == "AICc"
    assert "not nested" in res.interpretation and "only the AICc comparison is valid" in res.interpretation
    nest = next(a for a in res.assumptions if a.name.startswith("Nested"))
    assert nest.status == "info"
    declared = compare_fits(x, y, "michaelis_menten", "one_phase_association", nested=True)
    assert declared.statistic is not None
    assert any("probably invalid" in w for w in declared.warnings)
    forced_off = compare_fits(x, y, "linear", "quadratic", nested=False)
    assert forced_off.p_value is None and "declared" in forced_off.assumptions[0].detail
    # known nested pairs from the library
    bx, by, _ = noisy("one_site_binding_total", seed=42, reps=3)
    binding = compare_fits(bx, by, "one_site_binding_specific", "one_site_binding_total")
    assert binding.extra["nested"] is True and binding.df == 2 and binding.p_value < 0.05
    cross = compare_fits(*dose_data(seed=1), "dose_response_3pl_logx", "michaelis_menten")
    assert cross.extra["nested"] is False


def test_compare_fits_user_defined_equations():
    x, y, _ = noisy("one_phase_decay", seed=43, reps=3)
    kwargs = {"equation_simple": "A*exp(-K*x)", "equation_complex": "A*exp(-K*x) + C",
              "initial_simple": {"A": 100, "K": 0.1}, "initial_complex": {"A": 100, "K": 0.1, "C": 5}}
    unknown = compare_fits(x, y, "user_defined", "user_defined", **kwargs)
    assert unknown.p_value is None and "could not be verified" in unknown.interpretation
    declared = compare_fits(x, y, "user_defined", "user_defined", nested=True, **kwargs)
    built_in = compare_fits(x, y, "one_phase_decay", "one_phase_decay", fixed_simple={"Plateau": 0})
    assert declared.statistic == pytest.approx(built_in.statistic, rel=1e-6)
    assert declared.p_value < 0.05 and declared.extra["preferred"] == "complex"


def test_compare_fits_edge_cases():
    x = np.array([1, 2, 3, 4.0])
    few = compare_fits(x, [1.0, 2.1, 2.9, 4.2], "linear", "quadratic")
    assert few.df2 == 1 and few.extra["akaike"]["delta_aicc"] is None
    assert any("AICc could not be computed" in w for w in few.warnings)
    tiny = compare_fits(x[:3], [1.0, 2.1, 2.9], "linear", "quadratic")
    assert tiny.p_value is None and any("Too few points" in w for w in tiny.warnings)
    assert "kept by default" in tiny.interpretation
    exact = compare_fits(np.arange(8.0), 1 + 2 * np.arange(8.0) + 0.5 * np.arange(8.0) ** 2,
                         "linear", "quadratic")
    assert exact.p_value is None and any("fits the data exactly" in w for w in exact.warnings)
    with pytest.raises(StatsError):
        compare_fits(x, [1, 2, 3, 4], "linear", "nope")


# -------------------------------------------------------------- compare_datasets
def test_compare_datasets_against_an_independent_global_fit():
    lx, ya = dose_data(seed=51, logec50=-6.3)
    _, yb = dose_data(seed=52, logec50=-5.8)
    res = compare_datasets({"Free drug": (lx, ya), "Nanoparticle": (lx, yb)},
                           "dose_response_4pl_logx", "LogEC50", kind="IC50")
    pa = optimize.curve_fit(hill, lx, ya, p0=[0, 100, -6, 1], xtol=1e-14, ftol=1e-14)[0]
    pb = optimize.curve_fit(hill, lx, yb, p0=[0, 100, -6, 1], xtol=1e-14, ftol=1e-14)[0]
    ss_sep = float(np.sum((ya - hill(lx, *pa)) ** 2) + np.sum((yb - hill(lx, *pb)) ** 2))

    def shared_resid(t):
        return np.r_[ya - hill(lx, t[1], t[2], t[0], t[3]), yb - hill(lx, t[4], t[5], t[0], t[6])]

    start = [(pa[2] + pb[2]) / 2, pa[0], pa[1], pa[3], pb[0], pb[1], pb[3]]
    glob = optimize.least_squares(shared_resid, start, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    ss_shared = float(np.sum(glob.fun**2))
    df_sep = 2 * lx.size - 8
    f = (ss_shared - ss_sep) / (ss_sep / df_sep)
    assert res.extra["ss"]["separate"] == pytest.approx(ss_sep, rel=1e-9)
    assert res.extra["ss"]["shared"] == pytest.approx(ss_shared, rel=1e-8)
    assert res.statistic == pytest.approx(f, rel=1e-6)
    assert (res.df, res.df2) == (1, df_sep)
    assert res.p_value == pytest.approx(stats.f.sf(f, 1, df_sep), rel=1e-5)
    assert res.parameter("LogIC50 (shared)").value == pytest.approx(glob.x[0], rel=1e-7)
    assert res.parameter("LogIC50 (Free drug)").value == pytest.approx(pa[2], rel=1e-7)
    diff = res.parameter("Difference in LogIC50 (Nanoparticle - Free drug)")
    assert diff.value == pytest.approx(pb[2] - pa[2], rel=1e-6)
    assert diff.ci_low < diff.value < diff.ci_high and diff.ci_low > 0
    ratio = res.parameter("IC50 ratio (Nanoparticle / Free drug)")
    assert ratio.value == pytest.approx(10 ** (pb[2] - pa[2]), rel=1e-6)
    assert (ratio.ci_low, ratio.ci_high) == pytest.approx((10**diff.ci_low, 10**diff.ci_high))
    assert res.p_value < 0.001 and "evidence that LogIC50 differs" in res.interpretation
    assert "fold" in res.interpretation
    assert res.n == {"Free drug": lx.size, "Nanoparticle": lx.size}
    assert set(res.extra["curves"]) == {"Free drug", "Nanoparticle"}
    assert res.extra["akaike"]["probability_separate"] > 0.99
    assert_strict_json(res)


def test_compare_datasets_shared_slope_of_lines_equals_compare_slopes():
    rng = np.random.default_rng(60)
    x = np.tile(np.arange(1.0, 11.0), 2)
    data = {"A": (x, 1 + 0.9 * x + rng.normal(0, 0.8, x.size)),
            "B": (x, 2 + 1.3 * x + rng.normal(0, 0.8, x.size)),
            "C": (x[:14], 0.5 + 1.0 * x[:14] + rng.normal(0, 0.8, 14))}
    res = compare_datasets(data, "linear", "Slope")
    ref = compare_slopes(data)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-8)
    assert res.p_value == pytest.approx(ref.p_value, rel=1e-6)
    assert (res.df, res.df2) == (ref.df, ref.df2)
    assert res.parameter("Slope (shared)").value == pytest.approx(ref.extra["common_slope"], rel=1e-9)
    assert res.parameter("Slope (shared)").se == pytest.approx(
        ref.parameter("Common (pooled) slope").se, rel=1e-8)
    assert res.parameter("Difference in Slope (B - A)") is None        # only reported for two sets


def test_compare_datasets_no_difference_defaults_and_errors():
    lx, ya = dose_data(seed=61)
    _, yb = dose_data(seed=62)
    res = compare_datasets({"A": (lx, ya), "B": (lx, yb)})
    assert res.extra["shared_parameter"] == "LogEC50"                   # default parameter
    assert res.p_value > 0.05 and "no clear evidence" in res.interpretation
    top = compare_datasets({"A": (lx, ya), "B": (lx, yb + 15)}, "dose_response_4pl_logx", "Top")
    assert top.p_value < 0.01 and top.parameter("Difference in Top (B - A)").value == pytest.approx(15, abs=6)
    three_pl = compare_datasets({"A": (lx, ya), "B": (lx, yb)}, "dose_response_3pl_logx",
                                fixed={"Bottom": 0})
    assert three_pl.df2 == 2 * lx.size - 4
    mm_x, mm_y, _ = noisy("michaelis_menten", seed=63, reps=3)
    mm = compare_datasets({"wt": (mm_x, mm_y), "mut": (mm_x, mm_y * 0.6)}, "michaelis_menten",
                          weighting="1/y^2")
    assert mm.extra["shared_parameter"] == "Km" and mm.p_value > 0.9      # same Km, different Vmax
    exact = compare_datasets({"A": (lx, hill(lx, 0, 100, -6, 1)), "B": (lx, hill(lx, 0, 100, -7, 1))})
    assert exact.p_value is None and any("exactly" in w for w in exact.warnings)
    with pytest.raises(StatsError, match="at least two data sets"):
        compare_datasets({"A": (lx, ya)})
    with pytest.raises(StatsError, match="pair of"):
        compare_datasets({"A": (lx, ya), "B": 3})
    with pytest.raises(StatsError, match="is fixed"):
        compare_datasets({"A": (lx, ya), "B": (lx, yb)}, fixed={"LogEC50": -6})
    with pytest.raises(StatsError, match="not a parameter"):
        compare_datasets({"A": (lx, ya), "B": (lx, yb)}, parameter="Km")
    with pytest.raises(StatsError, match="not available"):
        compare_datasets({"A": (lx, ya), "B": (lx, yb)}, weighting="1/sd^2")


# ----------------------------------------------------------------------- registry
def _dose_table():
    rng = np.random.default_rng(70)
    conc = 10 ** np.arange(-9.0, -3.75, 0.5)
    lx = np.log10(conc)
    free = np.column_stack([hill(lx, 5, 95, -6.4, -1.0) + rng.normal(0, 3, lx.size) for _ in range(3)])
    nano = np.column_stack([hill(lx, 5, 95, -5.7, -1.0) + rng.normal(0, 3, lx.size) for _ in range(3)])
    free[4, 2] = np.nan
    return XYTable.from_arrays("viability", conc, {"Free drug": free, "Nanoparticle": nano},
                               x_name="Concentration (M)")


def test_registry_nonlinear_regression_defaults():
    table = _dose_table()
    res = run_analysis(table, "nonlinear_regression")
    assert res.analysis == "nonlinear_regression" and res.table == "viability"
    assert res.extra["series"] == "Free drug" and res.extra["model"] == "dose_response_4pl"
    assert res.n == {"Free drug": 32}                       # every replicate is a point
    assert any("individual point" in note for note in res.data_notes)
    assert res.interpretation and res.why_chosen and res.assumptions and res.methods_text
    assert res.parameter("EC50").value == pytest.approx(10**-6.4, rel=0.5)
    assert set(res.extra["curve"]) >= {"x", "y", "ci_low", "ci_high", "pred_low", "pred_high"}
    assert isinstance(res.extra["curve"]["y"], list) and res.extra["x_is_log"] is False
    assert res.significant is None
    json.loads(res.model_dump_json())


def test_registry_nonlinear_regression_options():
    table = _dose_table()
    res = run_analysis(table, "nonlinear_regression", series="Nanoparticle", potency_name="IC50",
                       fixed="Bottom=0", initial="Top=100, IC50=1e-6", alpha=0.1, ci_level=0.9,
                       ci_method="profile")
    assert res.parameter("IC50") is not None and res.parameter("IC50").ci_level == pytest.approx(0.9)
    assert res.extra["fixed"] == {"Bottom": 0.0} and res.alpha == 0.1
    means = run_analysis(table, "nonlinear_regression", use_means=True)
    assert means.n == {"Free drug": 11}
    assert any("discards the information" in w for w in means.warnings)
    sd_weighted = run_analysis(table, "nonlinear_regression", weighting="1/sd^2")
    assert sd_weighted.n == {"Free drug": 11} and sd_weighted.extra["weighting"] == "1/sd^2"
    assert any("1/SD^2" in note for note in sd_weighted.data_notes)
    three = run_analysis(table, "nonlinear_regression", model="dose_response_3pl", direction="inhibition")
    assert three.extra["fixed"] == {"HillSlope": -1.0}
    user = run_analysis(table, "nonlinear_regression", model="user_defined",
                        equation="Y = Bottom + (Top - Bottom) / (1 + (X / IC50)^Hill)",
                        initial="Bottom=0, Top=100, IC50=1e-6, Hill=1")
    default = run_analysis(table, "nonlinear_regression")
    assert user.parameter("IC50").value == pytest.approx(default.parameter("EC50").value, rel=1e-5)
    assert user.extra["goodness_of_fit"]["ss"] == pytest.approx(default.extra["goodness_of_fit"]["ss"], rel=1e-8)
    single = XYTable.from_arrays("s", [1, 2, 4, 8, 16.0], {"Y": [10, 18, 30, 41, 47.0]})
    assert run_analysis(single, "nonlinear_regression", model="michaelis_menten").n == {"Y": 5}
    with pytest.raises(StatsError, match="needs replicate"):
        run_analysis(single, "nonlinear_regression", model="michaelis_menten", weighting="1/sd^2")
    with pytest.raises(StatsError, match="weighting"):
        run_analysis(table, "nonlinear_regression", weighting="bogus")
    with pytest.raises(StatsError, match="Unknown series"):
        run_analysis(table, "nonlinear_regression", series="Nope")
    with pytest.raises(StatsError, match="Unknown function"):
        run_analysis(table, "nonlinear_regression", model="user_defined", equation="__import__('os')")
    with pytest.raises(StatsError, match="Unknown option"):
        run_analysis(table, "nonlinear_regression", bogus=1)


def test_registry_compare_fits_and_compare_curve_parameter():
    table = _dose_table()
    res = run_analysis(table, "compare_fits")
    assert res.analysis == "compare_fits" and res.extra["series"] == "Free drug"
    assert res.extra["nested"] is True and res.statistic is not None
    assert res.interpretation and res.why_chosen and res.assumptions
    json.loads(res.model_dump_json())
    other = run_analysis(table, "compare_fits", series="Nanoparticle", model_simple="dose_response_4pl",
                         model_complex="dose_response_4pl", fixed_simple="Bottom=0", use_means=True)
    assert other.df == 1 and other.n == {"Nanoparticle": 11}
    user = run_analysis(table, "compare_fits", model_simple="user_defined", model_complex="dose_response_4pl",
                        equation_simple="Y = A + B*log10(X)")
    assert user.p_value is None and user.extra["preferred_basis"] == "AICc"

    cmp_res = run_analysis(table, "compare_curve_parameter")
    assert cmp_res.analysis == "compare_curve_parameter" and cmp_res.extra["series"] == ["Free drug", "Nanoparticle"]
    assert cmp_res.p_value < 0.001 and cmp_res.significant is True
    assert cmp_res.interpretation and cmp_res.why_chosen and cmp_res.assumptions
    assert cmp_res.parameter("EC50 ratio (Nanoparticle / Free drug)").value == pytest.approx(5, rel=0.5)
    json.loads(cmp_res.model_dump_json())
    top = run_analysis(table, "compare_curve_parameter", parameter="Top", fixed="Bottom=0",
                       series="Free drug, Nanoparticle", use_means=True)
    assert top.extra["shared_parameter"] == "Top" and top.p_value > 0.01
    with pytest.raises(StatsError, match="at least two"):
        run_analysis(table, "compare_curve_parameter", series=["Free drug"])


# =============================================================================
# Audit part 2: B7 ('e' is a parameter), B9 (ties), D1 (IC50 labels), units
# =============================================================================
def test_user_equation_parameter_named_e_is_fitted_and_explained():
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 25)
    y = 1 + 2 * x - 0.5 * x**2 + 0.3 * x**3 + 0.7 * x**4 + rng.normal(0, 0.05, 25)
    res = fit_curve(x, y, "user_defined", equation="Y = a + b*x + c*x^2 + d*x^3 + e*x^4")
    assert [p.name for p in res.parameters] == ["a", "b", "c", "d", "e"]
    coef = np.polyfit(x, y, 4)[::-1]  # independent least-squares solution
    assert [p.value for p in res.parameters] == pytest.approx(coef, rel=1e-6)
    assert res.extra["goodness_of_fit"]["r_squared"] > 0.99
    assert any("'e'" in n and "Euler" in n and "exp(1)" in n for n in res.data_notes)
    assert "not Euler" in res.parameter("e").unit_note
    # Euler's number is exp(1); pi stays a constant and the result says so
    growth = fit_curve(x, 3.0 * np.exp(0.4 * x) + rng.normal(0, 0.01, 25), "user_defined",
                       equation="A * exp(1)^(k*x)", initial="A=1, k=0.1")
    assert [p.name for p in growth.parameters[:2]] == ["A", "k"]
    assert growth.parameter("k").value == pytest.approx(0.4, abs=0.01)
    assert not any("Euler" in n for n in growth.data_notes)
    wave = fit_curve(x, 2.0 * np.sin(2 * np.pi * x / 4) + rng.normal(0, 0.01, 25), "user_defined",
                     equation="A * sin(2 * pi * x / 4)")
    assert [p.name for p in wave.parameters] == ["A"]
    assert any("3.14159" in n for n in wave.data_notes)
    # the tooltip of the equation box tells users about both names
    spec = next(p for p in __import__("torusplot.stats.registry", fromlist=["x"]).get_analysis(
        "nonlinear_regression").params if p.name == "equation")
    assert "exp(1)" in spec.help and "pi" in spec.help


def _binding_data(seed=2):
    rng = np.random.default_rng(seed)
    x = np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0] * 2)
    return x, 100 * x / (5 + x) + rng.normal(0, 2, x.size)


def test_compare_fits_declares_no_winner_between_identical_fits():
    """'One site: specific binding' and 'Michaelis-Menten' are the same function."""
    x, y = _binding_data()
    res = compare_fits(x, y, "one_site_binding_specific", "michaelis_menten")
    ak = res.extra["akaike"]
    assert abs(ak["delta_aicc"]) < 1e-6 and ak["probability_simple"] == pytest.approx(0.5, abs=1e-6)
    assert res.extra["preferred"] == "none" and res.extra["shown"] == "simple"
    assert res.extra["same_number_of_parameters"] is True and "tie" in res.extra["preferred_basis"]
    text = res.interpretation
    assert "fit these data equally well" in text and "do not distinguish" in text
    for banned in ("is preferred", "more likely to be correct", "simpler", "more complex"):
        assert banned not in text
    assert [row["model"] for row in res.extra["comparison"]] == ["first", "second"]
    assert [e.target.split(":")[0] for e in res.effect_sizes] == ["first model", "second model"]
    assert res.extra["model"] == "one_site_binding_specific"  # the first model's curve is shown
    assert_strict_json(res)


def test_compare_fits_wording_for_equal_parameter_counts_and_weak_evidence():
    x, y = _binding_data()
    # same number of parameters, clearly different fits: a preference, but no "simpler"
    res = compare_fits(x, y, "michaelis_menten", "exponential_growth")
    assert res.extra["same_number_of_parameters"] is True and res.extra["preferred"] in {"simple", "complex"}
    assert "simpler" not in res.interpretation and "more complex" not in res.interpretation
    assert ("first model" in res.interpretation) or ("second model" in res.interpretation)
    assert "is preferred" in res.interpretation
    # different numbers of parameters keep the usual words
    nested = compare_fits(x, y, "one_site_binding_specific", "one_site_binding_total")
    assert "simpler" in nested.interpretation and nested.extra["same_number_of_parameters"] is False
    assert [row["model"] for row in nested.extra["comparison"]] == ["simpler", "more complex"]
    # an AICc difference below 2 is reported as weak evidence
    rng = np.random.default_rng(11)
    xs = np.linspace(1, 10, 12)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        ys = 2 + 0.5 * xs + rng.normal(0, 1, xs.size)
        weak = compare_fits(xs, ys, "linear", "one_phase_association", nested=False)
        delta = weak.extra["akaike"]["delta_aicc"]
        if delta is not None and 1e-3 < abs(delta) < 2:
            assert "weak evidence" in weak.interpretation
            break
    else:  # pragma: no cover - the loop finds a case within a few seeds
        pytest.fail("no data set with a small AICc difference was generated")


def test_ic50_label_covers_descriptions_and_summary():
    rng = np.random.default_rng(21)
    conc = np.repeat(np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]), 3)
    y = 5 + 95 / (1 + (conc / 1e-7) ** 1.2) + rng.normal(0, 4, conc.size)
    res = fit_curve(conc, y, "dose_response_4pl", kind="IC50")
    assert res.parameter("LogIC50").unit_note == "log10 of IC50 (the parameter actually fitted)."
    assert "EC50" not in json.dumps(res.extra["parameter_descriptions"])
    assert all("EC50" not in (p.unit_note or "") and "EC50" not in p.name for p in res.parameters)
    assert "log10(IC50)" in res.why_chosen and "EC50" not in res.why_chosen
    assert "IC50" in res.extra["equation"] and "EC50" not in res.extra["equation"]
    ec = fit_curve(conc, y, "dose_response_4pl")
    assert ec.parameter("LogEC50").unit_note == "log10 of EC50 (the parameter actually fitted)."
    assert ec.parameter("LogEC50").value == pytest.approx(res.parameter("LogIC50").value, rel=1e-12)


def test_compare_curve_parameter_has_its_own_model_tooltip():
    from torusplot.stats import registry

    shared = next(p for p in registry.get_analysis("compare_curve_parameter").params if p.name == "model")
    single = next(p for p in registry.get_analysis("nonlinear_regression").params if p.name == "model")
    assert "user_defined" not in shared.choices and "user_defined" not in shared.help
    assert "cannot be used in this analysis" in shared.help
    assert "user_defined" in single.choices and "Choose 'user_defined'" in single.help


_NL_X = np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0] * 3)
_NL_Y = 100 * _NL_X / (5 + _NL_X) + np.random.default_rng(3).normal(0, 3, _NL_X.size)
_NL_XG = np.linspace(-3, 3, 13).repeat(2) + 5
_NL_YG = 50 * np.exp(-0.5 * ((_NL_XG - 5.3) / 0.9) ** 2) + np.random.default_rng(4).normal(0, 2, _NL_XG.size)
# parameter -> (power of the X unit, power of the Y unit)
_NL_UNITS = {
    "michaelis_menten": (_NL_X, _NL_Y, {"Vmax": (0, 1), "Km": (1, 0)}),
    "gaussian": (_NL_XG, _NL_YG, {"Amplitude": (0, 1), "Mean": (1, 0), "SD": (1, 0)}),  # SD is bounded at 0
    "one_phase_decay": (_NL_X, 100 - _NL_Y, {"Y0": (0, 1), "Plateau": (0, 1), "K": (-1, 0)}),
    "dose_response_4pl": (_NL_X, 100 - _NL_Y, {"Bottom": (0, 1), "Top": (0, 1), "HillSlope": (0, 0)}),
}


@pytest.mark.parametrize("model", list(_NL_UNITS))
@pytest.mark.parametrize("sx, sy", [(1e-9, 1.0), (1.0, 1e-12), (1e-9, 1e-12), (1e6, 1e9), (1.0, 1e120)])
def test_curve_fits_do_not_depend_on_the_units_of_x_and_y(model, sx, sy):
    """Fit, goodness of fit, tests and profile-likelihood limits are the same in any unit;
    parameters simply carry the unit. (The optimiser works on rescaled residuals: its absolute
    gradient threshold used to stop bounded fits at the starting values for small Y units.)"""
    x, y, units = _NL_UNITS[model]
    base = fit_curve(x, y, model, ci_method="profile")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_curve(x * sx, y * sy, model, ci_method="profile")
    gb, gr = base.extra["goodness_of_fit"], res.extra["goodness_of_fit"]
    assert gr["r_squared"] == pytest.approx(gb["r_squared"], rel=1e-8)
    assert gr["sy_x"] == pytest.approx(gb["sy_x"] * sy, rel=1e-6)
    assert res.extra["convergence"]["converged"] and base.extra["convergence"]["converged"]
    assert res.extra["runs_test"]["p_value"] == pytest.approx(base.extra["runs_test"]["p_value"], rel=1e-9)
    for check, ref in zip(res.assumptions, base.assumptions):
        assert check.name == ref.name and check.status == ref.status
        assert (check.p_value is None) == (ref.p_value is None)
        if ref.p_value is not None:
            assert check.p_value == pytest.approx(ref.p_value, rel=1e-5), check.name
    for name, (px, py) in units.items():
        factor = sx**px * sy**py
        new, old = res.parameter(name), base.parameter(name)
        for attr in ("value", "se", "ci_low", "ci_high"):
            assert getattr(new, attr) == pytest.approx(getattr(old, attr) * factor, rel=2e-5), (name, attr)
    assert_strict_json(res)


@pytest.mark.parametrize("sx, sy", [(1e-9, 1e-12), (1e6, 1e9)])
def test_fit_comparisons_do_not_depend_on_the_units(sx, sy):
    x, y, _ = _NL_UNITS["dose_response_4pl"]
    base = compare_fits(x, y, "dose_response_3pl", "dose_response_4pl")
    res = compare_fits(x * sx, y * sy, "dose_response_3pl", "dose_response_4pl")
    assert res.statistic == pytest.approx(base.statistic, rel=1e-5)
    assert res.p_value == pytest.approx(base.p_value, rel=1e-5)
    assert res.extra["akaike"]["delta_aicc"] == pytest.approx(base.extra["akaike"]["delta_aicc"], rel=1e-5)
    data = {"a": (x, y), "b": (x * 1.7, y)}
    scaled = {k: (xx * sx, yy * sy) for k, (xx, yy) in data.items()}
    base_d = compare_datasets(data, "dose_response_4pl")
    res_d = compare_datasets(scaled, "dose_response_4pl")
    assert res_d.statistic == pytest.approx(base_d.statistic, rel=1e-4)
    assert res_d.p_value == pytest.approx(base_d.p_value, rel=1e-4)


def test_a_large_offset_in_y_is_not_mistaken_for_a_perfect_fit():
    """'Exact fit' is judged against the centred total sum of squares and the rounding level,
    not against sum(y^2): with Y = 1e11 + signal the residual scatter is still real."""
    x = np.arange(1.0, 21.0)
    rng = np.random.default_rng(6)
    noise = np.round(rng.normal(0, 1, 20) * 64) / 64
    y = 3.0 * x + noise
    base = fit_curve(x, y, "linear")
    res = fit_curve(x, y + 1e11, "linear")
    assert not res.extra["goodness_of_fit"]["perfect_fit"]
    assert res.parameter("Slope").value == pytest.approx(base.parameter("Slope").value, rel=1e-6)
    assert res.parameter("Slope").se == pytest.approx(base.parameter("Slope").se, rel=1e-4)
    line = linear_regression(x, y + 1e11)
    assert line.parameter("Slope").se == pytest.approx(linear_regression(x, y).parameter("Slope").se, rel=1e-6)
    assert not any("perfect" in w for w in line.warnings)
    # exact data are still recognised as an exact fit, with or without the offset
    for offset in (0.0, 1e11):
        exact = linear_regression(x, 3.0 * x + 1.0 + offset)
        assert any("perfect fit" in w for w in exact.warnings) and exact.parameter("Slope").se == 0.0
        assert fit_curve(x, 3.0 * x + 1.0 + offset, "linear").extra["goodness_of_fit"]["perfect_fit"]
