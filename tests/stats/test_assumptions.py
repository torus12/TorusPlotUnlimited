"""Normality and equal-variance tests.

Validation sources (see tests/fixtures/sources/basics.md):
* Published worked example: Shapiro & Wilk (1965), weights of 11 men, W = 0.79.
* Cross-checks against scipy and statsmodels, and against from-first-principles
  re-implementations written in this file (Royston's Shapiro-Wilk algorithm, the
  D'Agostino / Anscombe-Glynn z scores, the Kolmogorov-Smirnov distance, the
  Anderson-Darling statistic, Filliben order-statistic medians, Levene, Bartlett, F ratio).
* Simulations with fixed seeds (type I error near alpha, power for skewed data).
* hypothesis property tests.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad

from torusplot.data.tables import ColumnTable, MultipleVariablesTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import assumptions as am

RNG = np.random.default_rng(1)
NORMAL = RNG.normal(10.0, 2.0, 40)
SKEWED = RNG.lognormal(0.0, 1.0, 40)
SMALL = np.array([4.1, 5.3, 3.9, 6.8, 5.0, 4.4, 7.9, 5.6, 4.7])


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)


# ============================================================ first-principles helpers


def _royston_shapiro(x):
    """Shapiro-Wilk W and P by Royston's (1992, 1995) algorithm, n >= 4."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    mm = float(m @ m)
    c = m / math.sqrt(mm)
    u = 1 / math.sqrt(n)
    a = np.empty(n)
    an = c[-1] + 0.221157 * u - 0.147981 * u**2 - 2.071190 * u**3 + 4.434685 * u**4 - 2.706056 * u**5
    if n > 5:
        an1 = c[-2] + 0.042981 * u - 0.293762 * u**2 - 1.752461 * u**3 + 5.682633 * u**4 - 3.582633 * u**5
        phi = (mm - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (1 - 2 * an**2 - 2 * an1**2)
        a[2:-2] = m[2:-2] / math.sqrt(phi)
        a[-1], a[-2], a[0], a[1] = an, an1, -an, -an1
    else:
        phi = (mm - 2 * m[-1] ** 2) / (1 - 2 * an**2)
        a[1:-1] = m[1:-1] / math.sqrt(phi)
        a[-1], a[0] = an, -an
    w = float(a @ x) ** 2 / float(((x - x.mean()) ** 2).sum())
    ln = math.log(n)
    if n >= 12:
        mu = -1.5861 - 0.31082 * ln - 0.083751 * ln**2 + 0.0038915 * ln**3
        sigma = math.exp(-0.4803 - 0.082676 * ln + 0.0030302 * ln**2)
        z = (math.log(1 - w) - mu) / sigma
    else:
        gamma = -2.273 + 0.459 * n
        mu = 0.5440 - 0.39978 * n + 0.025054 * n**2 - 6.714e-4 * n**3
        sigma = math.exp(1.3822 - 0.77857 * n + 0.062767 * n**2 - 0.0020322 * n**3)
        z = (-math.log(gamma - math.log(1 - w)) - mu) / sigma
    return w, float(stats.norm.sf(z))


def _dagostino(x):
    """K2 from the D'Agostino (1970) skewness z and the Anscombe-Glynn (1983) kurtosis z."""
    x = np.asarray(x, float)
    n = x.size
    d = x - x.mean()
    m2 = (d**2).mean()
    b1 = (d**3).mean() / m2**1.5
    b2 = (d**4).mean() / m2**2
    y = b1 * math.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
    beta2 = 3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2 / (w2 - 1))
    z1 = delta * math.log(y / alpha + math.sqrt((y / alpha) ** 2 + 1))
    mean_b2 = 3 * (n - 1) / (n + 1)
    var_b2 = 24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    xx = (b2 - mean_b2) / math.sqrt(var_b2)
    sb = 6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * math.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    a = 6 + 8 / sb * (2 / sb + math.sqrt(1 + 4 / sb**2))
    z2 = ((1 - 2 / (9 * a)) - np.cbrt((1 - 2 / a) / (1 + xx * math.sqrt(2 / (a - 4))))) / math.sqrt(2 / (9 * a))
    k2 = z1 * z1 + z2 * z2
    return float(k2), float(stats.chi2.sf(k2, 2)), float(z1), float(z2)


def _ks_distance(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    cdf = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)))


def _levene(groups, center):
    """Levene's W as a one-way ANOVA F on absolute deviations."""
    z = [np.abs(np.asarray(g, float) - center(np.asarray(g, float))) for g in groups]
    return stats.f_oneway(*z)


# ===================================================================== Shapiro-Wilk


def test_shapiro_wilk_published_example():
    """Shapiro & Wilk (1965) Biometrika 52:591-611, section 4: weights (lb) of 11 men;
    the paper reports W = 0.79, below the 1% point."""
    weights = [148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236]
    res = am.shapiro_wilk(weights)
    assert res["statistic"] == pytest.approx(0.79, abs=0.005)
    assert res["p_value"] < 0.01 and res["normal"] is False and res["n"] == 11


@pytest.mark.parametrize("data", [NORMAL, SKEWED, SMALL, SKEWED[:4], NORMAL[:5], NORMAL[:12]])
def test_shapiro_wilk_matches_scipy_and_royston_algorithm(data):
    res = am.shapiro_wilk(data)
    ref = stats.shapiro(data)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)
    w, p = _royston_shapiro(data)
    assert res["statistic"] == pytest.approx(w, rel=1e-6)
    assert res["p_value"] == pytest.approx(p, rel=1e-4, abs=1e-9)


def test_shapiro_wilk_n3_exact_formula():
    x = np.array([1.0, 2.0, 4.5])
    res = am.shapiro_wilk(x)
    xs = np.sort(x)
    w = (math.sqrt(0.5) * (xs[2] - xs[0])) ** 2 / ((xs - xs.mean()) ** 2).sum()
    p = 6 / math.pi * (math.asin(math.sqrt(w)) - math.asin(math.sqrt(0.75)))
    assert res["statistic"] == pytest.approx(w, rel=1e-6)
    assert res["p_value"] == pytest.approx(p, rel=1e-4)


def test_shapiro_wilk_edge_cases():
    with pytest.raises(StatsError, match="at least 3"):
        am.shapiro_wilk([1.0, 2.0])
    with pytest.raises(StatsError):
        am.shapiro_wilk([1.0, 2.0, np.nan])
    const = am.shapiro_wilk([3.0] * 6)
    assert const["statistic"] is None and const["p_value"] is None and const["normal"] is None
    assert "identical" in const["warnings"][0]
    big = am.shapiro_wilk(np.random.default_rng(1).normal(size=5001))
    assert any("approximate" in w for w in big["warnings"]) and 0 <= big["p_value"] <= 1
    with pytest.raises(StatsError):
        am.shapiro_wilk(NORMAL, alpha=1.5)
    assert am.shapiro_wilk(NORMAL, alpha=0.999)["normal"] is False  # alpha is honoured


# ============================================================== D'Agostino-Pearson


@pytest.mark.parametrize("data", [NORMAL, SKEWED, SMALL, NORMAL[:8], SKEWED[:25]])
def test_dagostino_matches_scipy_and_first_principles(data):
    res = am.dagostino_pearson(data)
    ref = stats.normaltest(data)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert res["z_skewness"] == pytest.approx(stats.skewtest(data).statistic, rel=1e-10)
    assert res["z_kurtosis"] == pytest.approx(stats.kurtosistest(data).statistic, rel=1e-10)
    k2, p, z1, z2 = _dagostino(data)
    assert res["statistic"] == pytest.approx(k2, rel=1e-8)
    assert res["p_value"] == pytest.approx(p, rel=1e-6)
    assert res["z_skewness"] == pytest.approx(z1, rel=1e-8)
    assert res["z_kurtosis"] == pytest.approx(z2, rel=1e-8)


def test_dagostino_exactly_symmetric_sample_has_zero_skewness_z():
    """For perfectly symmetric data the sample skewness is exactly 0, so its z score must be
    0 and K2 reduces to the squared kurtosis z. (scipy.stats.skewtest special-cases a zero
    skewness and returns a non-zero z there, which is why K2 is computed in-house; the
    statistic must also not change when the data are merely rescaled.)"""
    x = np.arange(8.0)
    res = am.dagostino_pearson(x)
    assert res["z_skewness"] == 0.0
    z_kurt = stats.kurtosistest(x).statistic
    assert res["z_kurtosis"] == pytest.approx(z_kurt, rel=1e-10)
    assert res["statistic"] == pytest.approx(z_kurt**2, rel=1e-10)
    assert res["p_value"] == pytest.approx(stats.chi2.sf(z_kurt**2, 2), rel=1e-10)
    for scale in (0.01, 0.1, 3.0):
        assert am.dagostino_pearson(x * scale + 1.0)["statistic"] == pytest.approx(res["statistic"], rel=1e-8)


def test_dagostino_edge_cases():
    with pytest.raises(StatsError, match="at least 8"):
        am.dagostino_pearson(NORMAL[:7])
    assert am.dagostino_pearson([2.0] * 10)["p_value"] is None
    assert any("fewer than 20" in w for w in am.dagostino_pearson(SMALL)["warnings"])
    assert am.dagostino_pearson(NORMAL)["warnings"] == []


# ===================================================== Kolmogorov-Smirnov / Lilliefors


@pytest.mark.parametrize("data", [NORMAL, SKEWED, SMALL, NORMAL[:5]])
def test_ks_distance_from_first_principles_and_lilliefors_p(data):
    res = am.kolmogorov_smirnov_normal(data)
    assert res["statistic"] == pytest.approx(_ks_distance(data), rel=1e-10)
    d, p = lilliefors(data, dist="norm", pvalmethod="table")
    assert res["statistic"] == pytest.approx(d, rel=1e-12)
    assert res["p_value"] == pytest.approx(p, rel=1e-12)
    assert "Lilliefors" in res["p_note"]
    # scipy's KS distance against the fitted normal is the same number...
    ks = stats.kstest(data, "norm", args=(np.mean(data), np.std(data, ddof=1)))
    assert res["statistic"] == pytest.approx(ks.statistic, rel=1e-10)
    # ...but its classical P value ignores that the parameters were estimated: too large
    assert ks.pvalue > res["p_value"] or res["p_value"] >= 0.99


def test_uncorrected_ks_p_value_is_invalid_lilliefors_is_calibrated():
    """Under normality the classical KS P value with estimated parameters almost never
    rejects (hugely conservative); the Lilliefors P value rejects about alpha of the time."""
    rng = np.random.default_rng(7)
    naive = corrected = 0
    n_sim = 400
    for _ in range(n_sim):
        x = rng.normal(size=30)
        naive += stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))).pvalue < 0.05
        corrected += am.kolmogorov_smirnov_normal(x)["p_value"] < 0.05
    assert naive / n_sim < 0.01
    assert 0.02 <= corrected / n_sim <= 0.09


def test_ks_edge_cases_and_table_limits():
    with pytest.raises(StatsError, match="at least 5"):
        am.kolmogorov_smirnov_normal([1.0, 2.0, 3.0, 4.0])
    assert am.kolmogorov_smirnov_normal([1.0] * 7)["p_value"] is None
    extreme = am.kolmogorov_smirnov_normal(np.random.default_rng(3).exponential(size=300) ** 3)
    assert extreme["p_value"] == pytest.approx(0.001) and "0.001 or smaller" in extreme["p_note"]
    perfect = stats.norm.ppf((np.arange(1, 201) - 0.5) / 200)  # an ideal normal sample
    assert "0.99 or larger" in am.kolmogorov_smirnov_normal(perfect)["p_note"]


# ================================================================= Anderson-Darling


@pytest.mark.parametrize("data", [NORMAL, SKEWED, SMALL, NORMAL[:8]])
def test_anderson_darling_matches_statsmodels_and_scipy(data):
    res = am.anderson_darling(data)
    a2, p = normal_ad(np.asarray(data))
    assert res["statistic"] == pytest.approx(a2, rel=1e-9)
    assert res["p_value"] == pytest.approx(p, rel=1e-6, abs=1e-12)
    sp = stats.anderson(data, "norm", method="interpolate")
    assert res["statistic"] == pytest.approx(sp.statistic, rel=1e-9)
    n = len(data)
    assert res["statistic_modified"] == pytest.approx(res["statistic"] * (1 + 0.75 / n + 2.25 / n**2))
    assert res["exceeds_5pct_critical"] == (res["statistic_modified"] > 0.752)
    # the 5% critical value and the P value formula agree with each other
    assert res["exceeds_5pct_critical"] == (res["p_value"] < 0.05)


def test_anderson_p_value_formula_branches_and_edges():
    # D'Agostino & Stephens (1986) Table 4.9: P(A* = 0.752) is 0.05 by construction
    assert am._anderson_p(0.752) == pytest.approx(0.05, abs=0.001)
    assert am._anderson_p(0.631) == pytest.approx(0.10, abs=0.002)
    assert am._anderson_p(1.035) == pytest.approx(0.01, abs=0.0005)
    values = [am._anderson_p(a) for a in (0.05, 0.15, 0.25, 0.4, 0.7, 2.0, 20.0, 200.0)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values, reverse=True) and values[-1] == 0.0
    with pytest.raises(StatsError, match="at least 8"):
        am.anderson_darling(NORMAL[:7])
    assert am.anderson_darling([5.0] * 9)["p_value"] is None


# ========================================================================= QQ data


def test_qq_data_matches_probplot_and_filliben_medians():
    qq = am.qq_data(SMALL)
    (osm, osr), (slope, intercept, r) = stats.probplot(SMALL, dist="norm")
    assert qq["theoretical"] == pytest.approx(list(osm)) and qq["ordered"] == pytest.approx(list(osr))
    assert (qq["slope"], qq["intercept"], qq["r"]) == pytest.approx((slope, intercept, r))
    n = SMALL.size
    med = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
    med[0], med[-1] = 1 - 0.5 ** (1 / n), 0.5 ** (1 / n)
    assert qq["theoretical"] == pytest.approx(list(stats.norm.ppf(med)), rel=1e-10)
    assert qq["ordered"] == sorted(SMALL.tolist())
    # least-squares line through the points
    fit = np.polyfit(qq["theoretical"], qq["ordered"], 1)
    assert (qq["slope"], qq["intercept"]) == pytest.approx(tuple(fit), rel=1e-8)
    const = am.qq_data([4.0, 4.0, 4.0])
    assert const["slope"] == 0.0 and const["intercept"] == 4.0 and const["r"] is None
    assert len(am.qq_data([1.0, np.nan, 2.0])["ordered"]) == 2
    with pytest.raises(StatsError):
        am.qq_data([1.0])


# ================================================================= combined result


def test_normality_tests_result_structure():
    groups = {"normal": NORMAL, "skewed": SKEWED, "tiny": [1.0, 2.0, 3.5], "one": [1.0],
              "flat": [2.0] * 9, "gap": [np.nan, 1.0, 2.0, 4.0, 3.0, 2.5]}
    res = am.normality_tests(groups, tests=("shapiro", "dagostino", "ks", "anderson")).finalize()
    assert res.test_name == "Normality tests" and res.p_value is None
    assert len(res.assumptions) == 6 * 4 and len(res.extra["table"]) == 24
    by = {(a.target, a.name): a for a in res.assumptions}
    assert by[("normal", "Normality (Shapiro-Wilk)")].status == "pass"
    assert by[("skewed", "Normality (Shapiro-Wilk)")].status == "fail"
    assert by[("skewed", "Normality (Anderson-Darling)")].status == "fail"
    assert by[("tiny", "Normality (Shapiro-Wilk)")].status == "pass"
    assert by[("tiny", "Normality (D'Agostino-Pearson)")].status == "warn"
    assert by[("one", "Normality (Shapiro-Wilk)")].status == "warn"
    assert by[("flat", "Normality (Shapiro-Wilk)")].status == "warn"
    sw = by[("skewed", "Normality (Shapiro-Wilk)")]
    assert sw.statistic == pytest.approx(stats.shapiro(SKEWED).statistic)
    assert sw.p_value == pytest.approx(stats.shapiro(SKEWED).pvalue)
    assert "Lilliefors" in by[("normal", "Normality (Kolmogorov-Smirnov (Lilliefors P value))")].detail
    assert set(res.extra["qq"]) == {"normal", "skewed", "tiny", "flat", "gap"}
    assert set(res.extra["qq"]["normal"]) == {"theoretical", "ordered", "slope", "intercept", "r"}
    assert res.n["gap"] == 5 and any("gap: 1 missing" in n for n in res.data_notes)
    assert "skewed (" in res.interpretation
    assert "No deviation was detected in normal, tiny, gap." in res.interpretation
    assert "Could not be tested: one, flat" in res.interpretation
    assert "very few values" in res.interpretation and "very many values" in res.interpretation
    assert any("fewer than 8" in w for w in res.warnings)
    assert len(res.citations) == 4 and len(res.descriptives) == 6
    row = next(r for r in res.extra["table"] if r["group"] == "skewed" and r["test"] == "Shapiro-Wilk")
    assert row["passed"] is False and row["statistic_name"] == "W"
    json.loads(res.model_dump_json())


def test_normality_tests_options_and_errors():
    res = am.normality_tests({"a": NORMAL}, tests="Shapiro-Wilk", alpha=0.01)
    assert len(res.assumptions) == 1 and res.alpha == 0.01
    assert "No test found evidence against a normal distribution in a" in res.interpretation
    assert "Lilliefors" not in res.methods_text
    assert "Lilliefors" in am.normality_tests({"a": NORMAL}).methods_text
    none = am.normality_tests({"a": [1.0, 2.0], "b": [3.0]})
    assert "None of the groups could be tested" in none.interpretation
    assert all(a.status == "warn" for a in none.assumptions)
    for bad in ({"tests": ("nope",)}, {"tests": ()}, {"alpha": 0}):
        with pytest.raises(StatsError):
            am.normality_tests({"a": NORMAL}, **bad)
    with pytest.raises(StatsError):
        am.normality_tests({})
    with pytest.raises(StatsError):
        am.normality_tests({"a": [np.nan], "b": []})


def test_normality_tests_type_one_error_and_power():
    """Fixed-seed simulation: normal samples are rejected about alpha of the time, strongly
    skewed (exponential) samples most of the time."""
    rng = np.random.default_rng(11)
    funcs = {"sw": am.shapiro_wilk, "dp": am.dagostino_pearson, "ks": am.kolmogorov_smirnov_normal,
             "ad": am.anderson_darling}
    n_sim = 300
    false_pos = dict.fromkeys(funcs, 0)
    power = dict.fromkeys(funcs, 0)
    for _ in range(n_sim):
        x, y = rng.normal(5.0, 2.0, 30), rng.exponential(1.0, 40)
        for key, func in funcs.items():
            false_pos[key] += func(x)["normal"] is False
            power[key] += func(y)["normal"] is False
    for key in funcs:
        assert 0.015 <= false_pos[key] / n_sim <= 0.10, (key, false_pos[key] / n_sim)
    assert power["sw"] / n_sim > 0.95 and power["ad"] / n_sim > 0.93
    assert power["dp"] / n_sim > 0.80 and power["ks"] / n_sim > 0.75
    # the well-known power ordering: Shapiro-Wilk beats the Kolmogorov-Smirnov test
    assert power["sw"] >= power["ks"]


# ================================================================== equal variances

A = np.array([8.2, 9.4, 7.7, 10.1, 9.0, 8.8, 9.9, 8.5])
B = np.array([12.1, 6.3, 15.8, 9.9, 4.2, 13.7, 7.5, 11.0, 16.4])
C = np.array([9.1, 9.6, 8.7, 10.4, 9.9, 9.3])


@pytest.mark.parametrize("groups", [[A, B], [A, B, C], [A, C], [SKEWED[:10], NORMAL[:15], SMALL]])
def test_levene_and_brown_forsythe_first_principles_vs_scipy(groups):
    for center, func, ours in (("mean", np.mean, am.levene_test(groups, center="mean")),
                               ("median", np.median, am.brown_forsythe_test(groups))):
        ref = stats.levene(*groups, center=center)
        assert ours["statistic"] == pytest.approx(ref.statistic, rel=1e-10)
        assert ours["p_value"] == pytest.approx(ref.pvalue, rel=1e-8)
        anova = _levene(groups, func)  # second, independent route: ANOVA on |deviations|
        assert ours["statistic"] == pytest.approx(anova.statistic, rel=1e-10)
        assert ours["p_value"] == pytest.approx(anova.pvalue, rel=1e-8)
        assert ours["df1"] == len(groups) - 1 and ours["df2"] == sum(len(g) for g in groups) - len(groups)
    assert am.brown_forsythe_test(groups)["center"] == "median"
    assert am.levene_test(groups)["test"] == "Levene"


@pytest.mark.parametrize("groups", [[A, B], [A, B, C]])
def test_bartlett_vs_scipy_and_formula(groups):
    res = am.bartlett_test(groups)
    ref = stats.bartlett(*groups)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-12)
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    v = np.array([np.var(g, ddof=1) for g in groups])
    big_n = n.sum()
    sp2 = ((n - 1) * v).sum() / (big_n - k)
    t = ((big_n - k) * math.log(sp2) - ((n - 1) * np.log(v)).sum()) / (
        1 + (1 / (3 * (k - 1))) * ((1 / (n - 1)).sum() - 1 / (big_n - k)))
    assert res["statistic"] == pytest.approx(t, rel=1e-10)
    assert res["p_value"] == pytest.approx(stats.chi2.sf(t, k - 1), rel=1e-8)
    assert res["df1"] == k - 1


def test_f_test_two_variances_first_principles():
    res = am.f_test_two_variances(B, A)
    f = np.var(B, ddof=1) / np.var(A, ddof=1)
    df1, df2 = B.size - 1, A.size - 1
    assert res["statistic"] == pytest.approx(f, rel=1e-12)
    assert (res["df1"], res["df2"]) == (df1, df2)
    assert res["p_value"] == pytest.approx(2 * stats.f.sf(f, df1, df2), rel=1e-10)  # f > 1 here
    # swapping the groups inverts F and leaves the two-sided P value unchanged
    swapped = am.f_test_two_variances(A, B)
    assert swapped["statistic"] == pytest.approx(1 / f, rel=1e-12)
    assert swapped["p_value"] == pytest.approx(res["p_value"], rel=1e-10)
    # CI of the variance ratio: the limits are where the two-sided test has P = 0.05
    for limit in (res["ratio_ci_low"], res["ratio_ci_high"]):
        scaled = f / limit
        p = 2 * min(stats.f.cdf(scaled, df1, df2), stats.f.sf(scaled, df1, df2))
        assert p == pytest.approx(0.05, abs=1e-9)
    assert res["ratio_ci_low"] < f < res["ratio_ci_high"]
    assert am.f_test_two_variances(A, A)["p_value"] == pytest.approx(1.0)
    with pytest.raises(StatsError):
        am.f_test_two_variances(A, B, ci_level=1.0)


def test_variance_tests_degenerate_inputs():
    # two values per group: |deviations| identical within every group -> undefined, not F = inf
    two = am.levene_test([[1.0, 3.0], [1.0, 5.0]])
    assert two["statistic"] is None and two["p_value"] is None and "2 values" in two["warnings"][0]
    flat = am.brown_forsythe_test([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert flat["statistic"] is None and "identical in every group" in flat["warnings"][0]
    assert am.bartlett_test([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])["p_value"] is None
    assert am.f_test_two_variances([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["statistic"] is None
    # one constant group is fine for Levene-type tests
    assert am.levene_test([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 9.0]])["p_value"] is not None
    for bad in ([[1.0, 2.0, 3.0]], [[1.0], [1.0, 2.0, 3.0]], [[np.nan, 1.0], [1.0, 2.0]], None):
        with pytest.raises(StatsError):
            am.levene_test(bad)
    with pytest.raises(StatsError):
        am.levene_test([A, B], center="trimmed")
    # NaN are dropped
    with_nan = am.levene_test({"a": [*A, np.nan], "b": B})
    assert with_nan["statistic"] == pytest.approx(am.levene_test([A, B])["statistic"])


def test_equal_variance_tests_result():
    res = am.equal_variance_tests({"A": A, "B": B}).finalize()
    bf = stats.levene(A, B, center="median")
    assert res.statistic_name == "F" and res.statistic == pytest.approx(bf.statistic)
    assert res.p_value == pytest.approx(bf.pvalue) and (res.df, res.df2) == (1, 15)
    assert res.significant is True
    names = [a.name for a in res.assumptions]
    assert names == ["Equal variances (Brown-Forsythe)", "Equal variances (Levene)",
                     "Equal variances (Bartlett)", "Equal variances (F test of two variances)",
                     "Independent observations"]
    assert [a.status for a in res.assumptions[:4]] == ["fail"] * 4
    assert "sensitive to non-normal" in res.assumptions[2].detail
    assert "Welch" in res.interpretation and "statistically significant" in res.interpretation
    ratio = res.parameter("SD ratio (largest / smallest)")
    assert ratio.value == pytest.approx(np.std(B, ddof=1) / np.std(A, ddof=1))
    var_ratio = res.parameter("Variance ratio (A / B)")
    assert var_ratio.value == pytest.approx(np.var(A, ddof=1) / np.var(B, ddof=1))
    assert var_ratio.ci_low < var_ratio.value < var_ratio.ci_high
    assert res.descriptives[0]["variance"] == pytest.approx(np.var(A, ddof=1))
    assert [r["test"] for r in res.extra["table"]] == ["Brown-Forsythe", "Levene", "Bartlett",
                                                       "F test of two variances"]
    assert res.n == {"A": 8, "B": 9} and len(res.citations) == 3
    json.loads(res.model_dump_json())

    three = am.equal_variance_tests({"A": A, "C": C, "A2": A + 1.0}).finalize()
    assert len(three.assumptions) == 4 and three.significant is False
    assert "no evidence" in three.interpretation and three.df == 2
    assert all(a.status == "pass" for a in three.assumptions[:3])


def test_equal_variance_tests_degenerate_and_disagreement():
    res = am.equal_variance_tests({"a": [1.0, 3.0], "b": [1.0, 5.0]})
    assert res.p_value is None and res.assumptions[0].status == "warn"
    assert "could not be computed" in res.interpretation
    assert any("fewer than 5" in w for w in res.warnings)
    const = am.equal_variance_tests({"a": [2.0, 2.0, 2.0, 2.0, 2.0], "b": [1.0, 2.0, 3.0, 4.0, 9.0]})
    assert const.parameter("SD ratio (largest / smallest)").value is None
    assert any("Bartlett" in w for w in const.warnings)
    # heavy-tailed but equal-scale data: Bartlett rejects, Brown-Forsythe does not
    rng = np.random.default_rng(5)
    found = None
    for _ in range(200):
        g = {"x": rng.standard_t(2, 25), "y": rng.standard_t(2, 25)}
        r = am.equal_variance_tests(g)
        if r.p_value >= 0.05 and r.extra["table"][2]["p_value"] < 0.05:
            found = r
            break
    assert found is not None and "The tests disagree" in found.interpretation
    with pytest.raises(StatsError):
        am.equal_variance_tests({"a": A})
    with pytest.raises(StatsError):
        am.equal_variance_tests({"a": A, "b": [1.0]})


def test_bartlett_is_sensitive_to_non_normality_brown_forsythe_is_not():
    """Simulation behind the advice in the docstrings: with equal variances but heavy-tailed
    data, Bartlett's false-positive rate is far above alpha; Brown-Forsythe stays near it."""
    rng = np.random.default_rng(99)
    n_sim, bart, bf = 300, 0, 0
    for _ in range(n_sim):
        g = [rng.standard_t(3, 20), rng.standard_t(3, 20), rng.standard_t(3, 20)]
        bart += am.bartlett_test(g)["p_value"] < 0.05
        bf += am.brown_forsythe_test(g)["p_value"] < 0.05
    assert bart / n_sim > 0.20
    assert bf / n_sim < 0.09


# ----------------------------------------------------------------------- registry


def _check(res, key):
    assert res.analysis == key and res.interpretation and res.why_chosen
    assert len(res.assumptions) >= 1
    json.loads(res.model_dump_json())


def test_registry_normality_column_table():
    table = ColumnTable.from_dict("col", {"N": NORMAL, "S": SKEWED, "T": [1.0, 2.0, 3.5]})
    res = run_analysis(table, "normality")
    _check(res, "normality")
    assert len(res.assumptions) == 9 and set(res.extra["qq"]) == {"N", "S", "T"}
    assert res.table == "col" and res.params["alpha"] == 0.05
    res = run_analysis(table, "normality", groups=["S"], include_anderson=True, alpha=0.01)
    assert len(res.assumptions) == 4 and res.alpha == 0.01 and list(res.n) == ["S"]
    assert run_analysis(table, "normality", groups="N").n == {"N": 40}
    assert list(run_analysis(table, "normality", groups=[]).n) == ["N", "S", "T"]
    with pytest.raises(StatsError, match="Unknown group"):
        run_analysis(table, "normality", groups=["nope"])


def test_registry_normality_multiple_variables_table():
    frame = pd.DataFrame({
        "trt": ["a"] * 10 + ["b"] * 10 + [None],
        "y": list(NORMAL[:10]) + list(SKEWED[:10]) + [3.0],
        "z": list(range(21)),
    })
    table = MultipleVariablesTable("mv", frame)
    res = run_analysis(table, "normality")  # defaults: first numeric column, no grouping
    _check(res, "normality")
    assert res.n == {"y": 21}
    res = run_analysis(table, "normality", column="y", by="trt")
    assert res.n == {"a": 10, "b": 10} and any("no value in 'trt'" in n for n in res.data_notes)
    res = run_analysis(table, "normality", by="y")  # default column skips the grouping column
    assert sum(res.n.values()) == 21 and len(res.n) == 21
    for bad in ({"column": "nope"}, {"column": "trt"}, {"column": "y", "by": "nope"}):
        with pytest.raises(StatsError):
            run_analysis(table, "normality", **bad)
    with pytest.raises(StatsError):
        run_analysis(MultipleVariablesTable("t", pd.DataFrame({"a": ["x", "y"]})), "normality")
    with pytest.raises(StatsError):
        run_analysis(MultipleVariablesTable("t", pd.DataFrame({"y": [1.0, 2.0], "g": ["", ""]})),
                     "normality", column="y", by="g")


def test_registry_equal_variance():
    table = ColumnTable.from_dict("col", {"A": A, "B": B, "C": C})
    res = run_analysis(table, "equal_variance")
    _check(res, "equal_variance")
    assert res.statistic == pytest.approx(stats.levene(A, B, C, center="median").statistic)
    assert res.significant is True and res.n_total == 23
    two = run_analysis(table, "equal_variance", groups=["A", "C"])
    assert two.statistic == pytest.approx(stats.levene(A, C, center="median").statistic)
    assert len(two.assumptions) == 5
    with pytest.raises(StatsError):
        run_analysis(table, "equal_variance", groups=["A"])
    with pytest.raises(StatsError):
        run_analysis(table, "equal_variance", groups=["A", "Z"])


# ------------------------------------------------------------------ property tests

_sample = st.lists(
    st.one_of(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
              st.integers(min_value=-3, max_value=3).map(float), st.just(float("nan"))),
    min_size=0, max_size=30,
)


def _n_finite(values):
    return sum(1 for v in values if not math.isnan(v))


@settings(max_examples=60, deadline=None)
@given(_sample)
def test_normality_functions_never_crash(values):
    minimum = {am.shapiro_wilk: 3, am.dagostino_pearson: 8, am.kolmogorov_smirnov_normal: 5,
               am.anderson_darling: 8, am.qq_data: 2}
    for func, need in minimum.items():
        if _n_finite(values) < need:
            with pytest.raises(StatsError):
                func(values)
            continue
        res = func(values)
        if func is am.qq_data:
            assert len(res["theoretical"]) == len(res["ordered"]) == _n_finite(values)
            assert res["r"] is None or -1 - 1e-9 <= res["r"] <= 1 + 1e-9
            continue
        assert res["p_value"] is None or 0.0 <= res["p_value"] <= 1.0
        assert (res["p_value"] is None) == (res["statistic"] is None) == (res["normal"] is None)
        if res["p_value"] is None:
            assert res["warnings"]


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=-50, max_value=50), min_size=8, max_size=25, unique=True),
       st.floats(min_value=-100, max_value=100), st.floats(min_value=0.01, max_value=100))
def test_normality_tests_are_location_scale_invariant(values, shift, scale):
    x = np.array(values, dtype=float)
    y = x * scale + shift
    for func in (am.shapiro_wilk, am.dagostino_pearson, am.kolmogorov_smirnov_normal, am.anderson_darling):
        a, b = func(x), func(y)
        assert b["statistic"] == pytest.approx(a["statistic"], rel=1e-6, abs=1e-9)
        assert b["p_value"] == pytest.approx(a["p_value"], rel=1e-5, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.dictionaries(st.sampled_from(["a", "b", "c"]), _sample, min_size=0, max_size=3))
def test_normality_tests_result_never_crashes(groups):
    if sum(_n_finite(v) for v in groups.values()) == 0:
        with pytest.raises(StatsError):
            am.normality_tests(groups)
        return
    res = am.normality_tests(groups, tests=("shapiro", "dagostino", "ks", "anderson")).finalize()
    assert len(res.assumptions) == 4 * len(groups)
    for a in res.assumptions:
        assert a.p_value is None or 0.0 <= a.p_value <= 1.0
        assert a.status in {"pass", "fail", "warn"}
    json.loads(res.model_dump_json())


@settings(max_examples=60, deadline=None)
@given(st.lists(_sample, min_size=0, max_size=4), st.sampled_from(["mean", "median"]))
def test_variance_tests_never_crash_and_are_order_invariant(groups, center):
    valid = len(groups) >= 2 and all(_n_finite(g) >= 2 for g in groups)
    funcs = [lambda g: am.levene_test(g, center=center), am.bartlett_test]
    if not valid:
        for func in funcs:
            with pytest.raises(StatsError):
                func(groups)
        with pytest.raises(StatsError):
            am.equal_variance_tests({f"g{i}": g for i, g in enumerate(groups)})
        return
    for func in funcs:
        res, rev = func(groups), func(groups[::-1])
        assert res["p_value"] is None or 0.0 <= res["p_value"] <= 1.0
        assert (res["p_value"] is None) == (res["statistic"] is None)
        if res["statistic"] is not None and rev["statistic"] is not None:
            assert rev["statistic"] == pytest.approx(res["statistic"], rel=1e-6, abs=1e-9)
        if res["p_value"] is None:
            assert res["warnings"]
    if len(groups) == 2:
        f = am.f_test_two_variances(groups[0], groups[1])
        assert f["p_value"] is None or 0.0 <= f["p_value"] <= 1.0
    out = am.equal_variance_tests({f"g{i}": g for i, g in enumerate(groups)}).finalize()
    assert out.p_value is None or 0.0 <= out.p_value <= 1.0
    json.loads(out.model_dump_json())


@settings(max_examples=30, deadline=None)
@given(st.lists(st.lists(st.integers(-40, 40), min_size=3, max_size=12, unique=True), min_size=2, max_size=4),
       st.floats(min_value=-50, max_value=50), st.floats(min_value=0.1, max_value=20))
def test_variance_tests_location_scale_invariance(groups, shift, scale):
    x = [np.array(g, dtype=float) for g in groups]
    y = [g * scale + shift for g in x]
    for func in (am.levene_test, am.brown_forsythe_test, am.bartlett_test):
        a, b = func(x), func(y)
        if a["statistic"] is None or b["statistic"] is None:
            continue
        assert b["statistic"] == pytest.approx(a["statistic"], rel=1e-6, abs=1e-8)
