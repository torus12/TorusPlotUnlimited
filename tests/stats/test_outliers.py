"""Outlier detection (Grubbs, ROUT). Outliers are flagged, never removed.

Validation sources (see tests/fixtures/sources/basics.md):
* Published: Grubbs critical values (ASTM E178 / Grubbs & Beck 1972 tables) and the worked
  example of the NIST/SEMATECH e-Handbook, section 1.3.5.17 (Tietjen-Moore data).
* ROUT (Motulsky & Brown 2006): every step re-implemented from the published description
  inside this file, plus fixed-seed simulations of the behaviour the paper reports.
* Cross-checks against scipy (t distribution, optimisers) and hypothesis property tests.
"""

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import optimize, stats

from torusplot.data.tables import ColumnTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import outliers as ol

# NIST/SEMATECH e-Handbook 1.3.5.17 (data: Tietjen & Moore 1972, mass spectrometer measurements)
NIST = np.array([199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57])
MASKED = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 30.0, 31.0])  # two outliers hiding each other


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)


# ============================================================================ Grubbs


def test_grubbs_critical_values_published_tables():
    """Two-sided 5% critical values (= one-sided 2.5% column of ASTM E178 / Grubbs & Beck
    1972): n = 10 -> 2.290, n = 50 -> 3.128; one-sided 5%: n = 8 -> 2.032, two-sided: 2.126
    (given as 2.127 in some tables; formula value 2.1266)."""
    assert ol.grubbs_critical_value(10, 0.05) == pytest.approx(2.290, abs=5e-4)
    assert ol.grubbs_critical_value(50, 0.05) == pytest.approx(3.128, abs=5e-4)
    assert ol.grubbs_critical_value(8, 0.05, two_sided=False) == pytest.approx(2.032, abs=5e-4)
    assert ol.grubbs_critical_value(8, 0.05) == pytest.approx(2.1266, abs=5e-5)
    # formula written out independently
    for n, alpha in ((5, 0.05), (20, 0.01), (100, 0.1)):
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        expected = (n - 1) / math.sqrt(n) * math.sqrt(t**2 / (n - 2 + t**2))
        assert ol.grubbs_critical_value(n, alpha) == pytest.approx(expected, rel=1e-12)
    # G can never exceed (n - 1) / sqrt(n); the critical value approaches it for tiny n
    assert ol.grubbs_critical_value(3, 0.05) < 2 / math.sqrt(3)
    with pytest.raises(StatsError):
        ol.grubbs_critical_value(2)
    with pytest.raises(StatsError):
        ol.grubbs_critical_value(10, 0.0)


def test_grubbs_nist_handbook_example():
    """NIST e-Handbook: G = 2.4687 for the maximum 245.57; the 5% critical value of the
    upper one-sided test is 2.032, so the maximum is declared an outlier."""
    res = ol.grubbs_test(NIST)
    assert res["g"] == pytest.approx(2.4687, abs=1e-4)
    assert res["value"] == 245.57 and res["index"] == 7 and res["n"] == 8
    assert res["g"] > ol.grubbs_critical_value(8, 0.05, two_sided=False)
    assert res["is_outlier"] is True  # also by the two-sided test used here (critical value 2.127)
    assert res["g_critical"] == pytest.approx(2.1266, abs=5e-5)
    assert res["mean"] == pytest.approx(NIST.mean()) and res["sd"] == pytest.approx(NIST.std(ddof=1))
    assert res["flagged"] == [{"index": 7, "value": 245.57, "g": res["g"], "g_critical": res["g_critical"],
                               "p_value": res["p_value"], "step": 1}]


def test_grubbs_p_value_inverts_the_critical_value():
    for n in (3, 5, 8, 20, 60):
        for alpha in (0.001, 0.01, 0.05, 0.2):
            crit = ol.grubbs_critical_value(n, alpha)
            assert ol.grubbs_p_value(crit, n) == pytest.approx(alpha, rel=1e-8)
            one = ol.grubbs_critical_value(n, alpha, two_sided=False)
            assert ol.grubbs_p_value(one, n, two_sided=False) == pytest.approx(alpha, rel=1e-8)
    assert ol.grubbs_p_value(0.5, 20) == 1.0  # capped
    assert ol.grubbs_p_value(19 / math.sqrt(20), 20) == 0.0  # the largest possible G
    assert ol.grubbs_p_value(1.0, 10) > ol.grubbs_p_value(2.0, 10) > ol.grubbs_p_value(2.5, 10)
    with pytest.raises(StatsError):
        ol.grubbs_p_value(1.0, 2)
    # decision by P value and by critical value always agree
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.standard_t(3, int(rng.integers(3, 30)))
        res = ol.grubbs_test(x)
        assert res["is_outlier"] == (res["p_value"] < 0.05)
        assert res["g"] == pytest.approx(np.max(np.abs(x - x.mean())) / x.std(ddof=1), rel=1e-12)


def test_grubbs_false_positive_rate_simulation():
    """For normal samples the two-sided test flags a value in about alpha of the data sets."""
    rng = np.random.default_rng(321)
    n_sim = 1500
    hits = sum(ol.grubbs_test(rng.normal(50, 5, 20))["is_outlier"] for _ in range(n_sim))
    assert 0.03 <= hits / n_sim <= 0.07


def test_grubbs_iterative_and_masking():
    x = [9.8, 10.1, 10.3, 9.9, 10.0, 10.2, 9.7, 10.4, 10.05, 9.95, 14.0, 19.0]
    single = ol.grubbs_test(x)
    assert [f["value"] for f in single["flagged"]] == [19.0] and len(single["steps"]) == 1
    multi = ol.grubbs_test(x, iterative=True)
    assert [f["value"] for f in multi["flagged"]] == [19.0, 14.0]
    assert [f["index"] for f in multi["flagged"]] == [11, 10] and [f["step"] for f in multi["flagged"]] == [1, 2]
    assert len(multi["steps"]) == 3 and multi["steps"][-1]["is_outlier"] is False
    # the second round is Grubbs' test on the data without the first flagged value
    second = ol.grubbs_test(x[:11])
    assert multi["flagged"][1]["g"] == pytest.approx(second["g"]) and multi["steps"][1]["n"] == 11
    assert any("same alpha" in w for w in multi["warnings"])
    assert multi["g"] == single["g"] and multi["value"] == 19.0  # headline = first round
    # masking: two similar outliers inflate the SD, Grubbs finds nothing even when repeated
    assert ol.grubbs_test(MASKED, iterative=True)["flagged"] == []


def test_grubbs_edge_cases():
    with pytest.raises(StatsError, match="at least 3"):
        ol.grubbs_test([1.0, 2.0])
    with pytest.raises(StatsError):
        ol.grubbs_test([1.0, 2.0, np.nan])
    with pytest.raises(StatsError):
        ol.grubbs_test(None)
    with pytest.raises(StatsError):
        ol.grubbs_test(NIST, alpha=1.0)
    const = ol.grubbs_test([4.0] * 5)
    assert const["g"] is None and const["p_value"] is None and const["flagged"] == [] and const["sd"] == 0.0
    assert "identical" in const["warnings"][0]
    small = ol.grubbs_test([1.0, 2.0, 3.0, 50.0])
    assert any("little power" in w for w in small["warnings"])
    # indices refer to positions in the input, missing values included
    with_gaps = ol.grubbs_test([np.nan, *NIST[:4], np.nan, *NIST[4:]])
    assert with_gaps["index"] == 9 and with_gaps["g"] == pytest.approx(ol.grubbs_test(NIST)["g"])
    # a strict alpha no longer flags a moderate outlier
    moderate = [10.0, 10.2, 9.9, 10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 11.6]
    assert ol.grubbs_test(moderate, alpha=0.05)["is_outlier"] is True
    assert ol.grubbs_test(moderate, alpha=0.0001)["is_outlier"] is False
    # the input is never modified
    data = NIST.copy()
    ol.grubbs_test(data, iterative=True)
    assert np.array_equal(data, NIST)


def test_grubbs_flag_caused_by_identical_values_is_explained():
    """When n - 1 values are identical, G equals its maximum (n - 1) / sqrt(n) whatever the
    size of the deviation, so the flag reflects ties / measurement resolution; the warning
    says so and does not claim that a value 'can hardly ever be flagged'."""
    for values in ([10.1, 10.1, 10.2], [10.1, 10.1, 10.3, 10.1], [5.0, 5.0, 5.0, 5.0, 5.0, 5.001]):
        res = ol.grubbs_test(values)
        n = len(values)
        assert res["g"] == pytest.approx((n - 1) / math.sqrt(n))
        assert res["is_outlier"] is True
        assert any("resolution" in w and "identical" in w for w in res["warnings"])
        assert not any("hardly ever" in w for w in res["warnings"])
    plain = ol.grubbs_test([1.0, 2.0, 3.0])
    assert plain["is_outlier"] is False and any("little power" in w for w in plain["warnings"])
    assert not any("resolution" in w for w in plain["warnings"])


def test_outliers_underflow_safe_for_tiny_values():
    """[0, 0, 1.1e-153] once crashed (squares underflow to 0 and the SD came out 0): the
    computations are rescaled, so the spread is real and the result is finite."""
    res = ol.grubbs_test([0.0, 0.0, 1.1e-153])
    assert res["sd"] == pytest.approx(1.1e-153 / math.sqrt(3.0), rel=1e-12)
    assert res["g"] == pytest.approx(2.0 / math.sqrt(3.0), rel=1e-12)
    out = ol.grubbs_outliers({"a": [0.0, 0.0, 1.1e-153]}).finalize()
    assert out.extra["flagged"][0]["row"] == 2
    json.loads(out.model_dump_json())


def test_grubbs_outliers_result():
    groups = {"A": [np.nan, *NIST], "B": MASKED, "C": [1.0, 2.0], "D": [3.0, 3.0, 3.0, 3.0]}
    res = ol.grubbs_outliers(groups).finalize()
    assert res.test_name == "Grubbs' outlier test" and res.analysis == "outliers_grubbs"
    assert res.extra["flagged"] == [{"group": "A", "row": 8, "value": 245.57}]
    assert res.extra["removed"] is False and res.extra["n_flagged"] == 1 and res.extra["method"] == "grubbs"
    assert res.n == {"A": 8, "B": 9, "C": 2, "D": 4}  # nothing was removed from the counts
    assert res.descriptives[0]["max"] == 245.57  # ...nor from the summaries
    assert "245.6 (row 9)" in res.interpretation and "No outliers were flagged in B, D" in res.interpretation
    assert "not removed" in res.interpretation and "your explicit decision" in res.interpretation
    assert "report" in res.interpretation
    assert any("Not tested (fewer than 3 values): C" in w for w in res.warnings)
    assert any(w.startswith("D:") and "identical" in w for w in res.warnings)
    assert any("A: 1 missing" in n for n in res.data_notes)
    by = {a.target: a for a in res.assumptions if a.target}
    assert by["A"].status == "pass" and by["A"].p_value == pytest.approx(stats.shapiro(NIST[:7]).pvalue)
    assert by["D"].status == "warn"
    assert res.assumptions[-1].name == "At most one outlier per group" and res.assumptions[-1].status == "info"
    row = res.extra["table"][0]
    assert row["flagged_rows"] == [8] and row["g"] == pytest.approx(2.4688, abs=1e-4) and row["outlier"] is True
    assert res.extra["table"][2] == {"group": "C", "n": 2, "n_flagged": None, "tested": False}
    assert res.statistic is None and res.p_value is None  # several groups: no single headline test
    assert "not removed automatically" in res.methods_text and len(res.citations) == 2
    json.loads(res.model_dump_json())

    single = ol.grubbs_outliers({"A": NIST}, alpha=0.01).finalize()
    assert single.statistic_name == "G" and single.statistic == pytest.approx(2.4688, abs=1e-4)
    assert single.p_value == pytest.approx(ol.grubbs_p_value(single.statistic, 8)) and single.alpha == 0.01
    none = ol.grubbs_outliers({"B": MASKED}, iterative=True).finalize()
    assert none.extra["flagged"] == [] and "did not flag any outlier" in none.interpretation
    assert "your explicit decision" in none.interpretation and none.test_name.endswith("(repeated)")
    assert none.assumptions[-1].name == "Outliers do not mask each other"
    skewed = ol.grubbs_outliers({"S": np.random.default_rng(2).lognormal(0, 1.2, 40)})
    assert skewed.assumptions[0].status == "fail" and "logarithms" in skewed.assumptions[0].detail

    custom = ol.grubbs_outliers({"A": NIST}, row_index={"A": np.arange(8) + 100})
    assert custom.extra["flagged"][0]["row"] == 107
    with pytest.raises(ValueError):
        ol.grubbs_outliers({"A": NIST}, row_index={"A": [1, 2]})
    with pytest.raises(StatsError):
        ol.grubbs_outliers({})
    with pytest.raises(StatsError, match="no group has that many"):
        ol.grubbs_outliers({"A": [1.0, 2.0], "B": [np.nan]})


# ============================================================================== ROUT


def _rout_reference(residuals, k, q):
    """ROUT step 2 written directly from the published description (1-based ranks)."""
    residuals = np.asarray(residuals, float)
    n = residuals.size
    rsdr = np.percentile(np.abs(residuals), 68.27) * n / (n - k)
    order = np.argsort(np.abs(residuals), kind="stable")
    for i in range(int(0.70 * n), n + 1):
        if i < 1:
            continue
        alpha_i = q * (n - (i - 1)) / n
        t = abs(residuals[order[i - 1]]) / rsdr
        p = 2 * stats.t.sf(t, n - k)
        if p < alpha_i:
            return sorted(int(j) for j in order[i - 1:])
    return []


def test_rsdr_definition():
    r = np.array([-0.5, 0.1, 0.3, -1.2, 2.0, 0.7, -0.2, 0.05, 1.1, -0.9])
    assert ol.robust_sd_of_residuals(r, 2) == pytest.approx(np.percentile(np.abs(r), 68.27) * 10 / 8, rel=1e-12)
    # for Gaussian residuals 68.27% lie within one SD, so the RSDR estimates the SD
    z = np.random.default_rng(1).normal(0, 3.0, 200_000)
    assert ol.robust_sd_of_residuals(z, 1) == pytest.approx(3.0, rel=0.01)
    with pytest.raises(StatsError):
        ol.robust_sd_of_residuals([1.0, 2.0], 2)


def test_rout_column_robust_fit_minimises_lorentzian_merit():
    rng = np.random.default_rng(10)
    x = np.concatenate([rng.normal(20, 2, 25), [45.0, 60.0]])
    res = ol.rout_column(x)
    assert res["converged"] and res["k"] == 1 and res["n"] == 27
    # self-consistency: the RSDR is the robust SD of the final residuals...
    assert res["rsdr"] == pytest.approx(ol.robust_sd_of_residuals(x - res["location"], 1), rel=1e-12)
    # ...and the location minimises sum(ln(1 + (D / RSDR)^2)) for that RSDR
    best = optimize.minimize_scalar(lambda m: np.sum(np.log1p(((x - m) / res["rsdr"]) ** 2)),
                                    bounds=(10, 30), method="bounded", options={"xatol": 1e-10})
    assert res["location"] == pytest.approx(best.x, abs=1e-6)
    assert res["residuals"] == pytest.approx(x - res["location"])
    # (c) the robust centre is barely moved by the outliers, the mean is
    clean_mean = x[:25].mean()
    assert abs(res["location"] - clean_mean) < 0.5 < abs(res["mean"] - clean_mean)
    assert res["rsdr"] < 0.5 * res["sd"]
    # (b) the planted outliers are flagged, nothing else
    assert res["outlier_indices"] == [25, 26]
    assert [f["value"] for f in res["flagged"]] == [45.0, 60.0]
    assert res["outlier_mask"].tolist() == [False] * 25 + [True, True]
    assert res["outlier_indices"] == _rout_reference(x - res["location"], 1, 0.01)
    for f in res["flagged"]:
        t = abs(f["residual"]) / res["rsdr"]
        assert f["t"] == pytest.approx(t) and f["p_value"] == pytest.approx(2 * stats.t.sf(t, 26))


def test_rout_detection_step_matches_published_procedure():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(4, 40))
        x = rng.standard_t(2, n)  # heavy tails: plenty of data sets with and without outliers
        q = float(rng.choice([0.001, 0.01, 0.05, 0.1]))
        res = ol.rout_column(x, q=q)
        assert res["outlier_indices"] == _rout_reference(x - res["location"], 1, q)
        tested = {t["index"] for t in res["tests"]}
        assert set(res["outlier_indices"]) <= tested
        # flagged points are exactly the ones with the largest absolute residuals
        if res["outlier_indices"]:
            abs_res = np.abs(res["residuals"])
            assert abs_res[res["outlier_indices"]].min() >= np.delete(abs_res, res["outlier_indices"]).max()
    # thresholds follow alpha_i = Q (N - (i - 1)) / N and testing starts at int(0.70 N)
    res = ol.rout_column(np.arange(20.0), q=0.01)
    assert [t["rank"] for t in res["tests"]] == list(range(14, 21))
    assert [t["threshold"] for t in res["tests"]] == pytest.approx([0.01 * (20 - (i - 1)) / 20 for i in range(14, 21)])


def test_rout_false_discovery_behaviour_on_clean_gaussian_data():
    """(a) Motulsky & Brown report that with Q = 1% and purely Gaussian scatter, one or more
    outliers are (falsely) found in only about 1-3% of simulated data sets."""
    rng = np.random.default_rng(2006)
    n_sim = 400
    with_any = n_flagged = 0
    for i in range(n_sim):
        x = rng.normal(100.0, 15.0, (10, 20, 30, 50)[i % 4])
        k = len(ol.rout_column(x, q=0.01)["outlier_indices"])
        with_any += k > 0
        n_flagged += k
    assert with_any / n_sim < 0.06
    assert n_flagged <= 2 * with_any + 5  # and never whole swathes of points


def test_rout_power_and_q_monotonicity():
    """(b) Gross planted outliers are flagged; a larger Q never flags fewer points."""
    rng = np.random.default_rng(77)
    found = false_extra = 0
    n_sim = 200
    for _ in range(n_sim):
        x = rng.normal(0.0, 1.0, 25)
        x[7] += 8.0
        x[19] -= 9.0
        flagged = set(ol.rout_column(x)["outlier_indices"])
        found += {7, 19} <= flagged
        false_extra += len(flagged - {7, 19}) > 0
        sizes = [len(ol.rout_column(x, q=q)["outlier_indices"]) for q in (0.001, 0.01, 0.05, 0.2)]
        assert sizes == sorted(sizes)
    assert found / n_sim > 0.95
    assert false_extra / n_sim < 0.10
    # masking does not fool ROUT (Grubbs' test finds nothing in these data, see above)
    assert ol.rout_column(MASKED)["outlier_indices"] == [7, 8]


def test_rout_general_curve_fit():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 25)
    y = 2.0 + 3.0 * x + rng.normal(0, 1.0, 25)
    y[5] += 40.0
    y[20] -= 30.0

    def line(xv, a, b):
        return a + b * xv

    res = ol.rout(x, y, line, [0.0, 1.0], refit=True)
    assert res["k"] == 2 and res["n"] == 25 and res["converged"]
    assert res["outlier_indices"] == [5, 20]
    assert [f["x"] for f in res["flagged"]] == pytest.approx([x[5], x[20]])
    # ordinary least squares on all points (reported for comparison) equals numpy's polyfit
    slope, intercept = np.polyfit(x, y, 1)
    assert res["params_ols"] == pytest.approx([intercept, slope], rel=1e-6)
    # (c) the robust fit stays near the truth (2, 3); least squares is dragged away
    assert abs(res["params_robust"][0] - 2.0) < 1.0 and abs(res["params_robust"][1] - 3.0) < 0.2
    assert abs(res["params_ols"][0] - 2.0) > 3.0
    # self-consistency of the adaptive fit
    resid = y - line(x, *res["params_robust"])
    assert res["residuals"] == pytest.approx(resid)
    assert res["rsdr"] == pytest.approx(ol.robust_sd_of_residuals(resid, 2), rel=1e-9)
    merit = lambda p: float(np.sum(np.log1p(((y - line(x, *p)) / res["rsdr"]) ** 2)))  # noqa: E731
    best = optimize.minimize(merit, res["params_robust"], method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-14})
    assert best.fun >= merit(res["params_robust"]) - 1e-8
    assert res["outlier_indices"] == _rout_reference(resid, 2, 0.01)
    # the least-squares refit without the flagged points is returned alongside, on request only
    keep = np.ones(25, bool)
    keep[[5, 20]] = False
    slope2, intercept2 = np.polyfit(x[keep], y[keep], 1)
    assert res["ols_refit"]["params"] == pytest.approx([intercept2, slope2], rel=1e-6)
    assert res["ols_refit"]["n_used"] == 23 and res["ols_refit"]["n_excluded_from_fit"] == 2
    assert ol.rout(x, y, line, [0.0, 1.0])["ols_refit"] is None
    # nonlinear model with missing values: indices still refer to the input
    xe = np.linspace(0, 5, 30)
    ye = 10.0 * np.exp(-0.7 * xe) + rng.normal(0, 0.15, 30)
    ye[12] += 6.0
    xe2, ye2 = np.insert(xe, 3, np.nan), np.insert(ye, 3, 1.0)
    out = ol.rout(xe2, ye2, lambda xv, a, k: a * np.exp(-k * xv), [5.0, 1.0])
    assert out["outlier_indices"] == [13] and np.isnan(out["residuals"][3]) and out["n"] == 30
    assert out["params_robust"] == pytest.approx([10.0, 0.7], rel=0.05)


def test_rout_constant_model_agrees_with_column_shortcut():
    rng = np.random.default_rng(4)
    for _ in range(10):
        v = rng.normal(5.0, 1.0, 30)
        v[3] += rng.choice([0.0, 9.0])
        a = ol.rout_column(v)
        b = ol.rout(np.arange(30.0), v, lambda xv, m: m + 0.0 * xv, [float(np.median(v))], _start="p0")
        assert b["params_robust"][0] == pytest.approx(a["location"], abs=1e-5)
        assert b["rsdr"] == pytest.approx(a["rsdr"], rel=1e-5)
        assert b["outlier_indices"] == a["outlier_indices"]
        assert b["params_ols"][0] == pytest.approx(v.mean(), rel=1e-8)


def test_rout_edge_cases():
    with pytest.raises(StatsError, match="at least 3"):
        ol.rout_column([1.0, 2.0])
    with pytest.raises(StatsError):
        ol.rout_column([1.0, 2.0, np.nan, np.inf])
    with pytest.raises(StatsError):
        ol.rout_column(None)
    for bad_q in (0, 1, -0.1, 5, "x", None):
        with pytest.raises(StatsError):
            ol.rout_column(NIST, q=bad_q)
    const = ol.rout_column([2.0] * 6)
    assert const["outlier_indices"] == [] and const["rsdr"] == 0.0 and "identical" in const["warnings"][0]
    mostly = ol.rout_column([5.0] * 8 + [9.0])
    assert mostly["outlier_indices"] == [] and "robust SD is zero" in mostly["warnings"][0]
    assert any("very little power" in w for w in ol.rout_column([1.0, 2.0, 4.0])["warnings"])
    gaps = ol.rout_column([np.nan, *NIST[:4], np.nan, *NIST[4:]])
    assert gaps["outlier_indices"] == [9] and gaps["residuals"].size == 10 and np.isnan(gaps["residuals"][0])
    assert gaps["outlier_mask"].sum() == 1 and gaps["outlier_mask"][9]
    data = NIST.copy()
    ol.rout_column(data)
    assert np.array_equal(data, NIST)

    def line(xv, a, b):
        return a + b * xv

    x = np.arange(10.0)
    with pytest.raises(StatsError, match="same number"):
        ol.rout(x, x[:5], line, [0, 1])
    with pytest.raises(StatsError, match="at least 4 points"):
        ol.rout(x[:3], x[:3], line, [0, 1])
    with pytest.raises(StatsError):
        ol.rout(x, x, line, [])
    with pytest.raises(StatsError):
        ol.rout(x, x, line, [np.nan, 1.0])
    with pytest.raises(StatsError):
        ol.rout(None, x, line, [0, 1])
    with pytest.raises(StatsError, match="could not be fitted"):
        ol.rout(x, x + 1.0, lambda xv, a, b: np.log(-np.abs(xv) - 1.0) * a + b, [1.0, 1.0])
    exact = ol.rout(x, 2.0 * x + 1.0, line, [0.0, 1.0])  # perfect fit: no scatter at all
    assert exact["outlier_indices"] == [] and exact["rsdr"] == 0.0 and "robust SD" in exact["warnings"][0]
    few = ol.rout(x[:5], np.array([1.0, 3.1, 4.9, 7.2, 9.0]), line, [0.0, 1.0], refit=True)
    assert any("very few points" in w for w in few["warnings"])
    not_converged = ol.rout(x, 2.0 * x + np.sin(x * 7.0), line, [0.0, 1.0], max_iter=1)
    assert not_converged["iterations"] == 1


def test_rout_refit_needs_enough_points():
    x = np.arange(6.0)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 500.0])
    res = ol.rout(x, y, lambda xv, a, b, c, d: a + b * xv + c * xv**2 + d * xv**3, [0.0, 1.0, 0.0, 0.0],
                  refit=True, q=0.5)
    assert res["ols_refit"] is None or res["ols_refit"]["n_used"] > 4


def test_rout_outliers_result():
    groups = {"A": [np.nan, *NIST], "B": MASKED, "C": [1.0, 2.0], "D": [5.0] * 8 + [9.0]}
    res = ol.rout_outliers(groups).finalize()
    assert res.test_name == "ROUT outlier detection (Q = 1%)" and res.analysis == "outliers_rout"
    assert res.extra["flagged"] == [{"group": "A", "row": 8, "value": 245.57},
                                    {"group": "B", "row": 7, "value": 30.0},
                                    {"group": "B", "row": 8, "value": 31.0}]
    assert res.extra["q"] == 0.01 and res.extra["method"] == "rout" and res.extra["removed"] is False
    assert "flagged 3 possible outliers" in res.interpretation
    assert "30 (row 8), 31 (row 9)" in res.interpretation
    assert "not removed" in res.interpretation and "your explicit decision" in res.interpretation
    assert res.n == {"A": 8, "B": 9, "C": 2, "D": 9} and res.p_value is None
    meaning = res.assumptions[-1]
    assert meaning.name == "Meaning of Q" and "false discovery rate" in meaning.detail
    assert res.assumptions[1].target == "B" and res.assumptions[1].p_value == pytest.approx(
        stats.shapiro(MASKED[:7]).pvalue)
    assert [p.name for p in res.parameters] == ["Robust centre of A", "Robust centre of B", "Robust centre of D"]
    assert res.parameters[0].value == pytest.approx(ol.rout_column(NIST)["location"])
    assert any("Not tested" in w and "C" in w for w in res.warnings)
    assert any(w.startswith("D:") for w in res.warnings)
    assert res.extra["table"][1]["flagged_values"] == [30.0, 31.0]
    assert "Motulsky" in res.citations[0] and "Q = 1%" in res.methods_text
    json.loads(res.model_dump_json())
    strict = ol.rout_outliers({"B": MASKED}, q=0.001)
    assert strict.test_name == "ROUT outlier detection (Q = 0.1%)"
    clean = ol.rout_outliers({"x": np.random.default_rng(6).normal(0, 1, 30)})
    assert clean.extra["flagged"] == [] and "did not flag any outlier" in clean.interpretation
    with pytest.raises(StatsError):
        ol.rout_outliers({"A": NIST}, q=1.5)
    with pytest.raises(StatsError):
        ol.rout_outliers({"A": [1.0]})


# ----------------------------------------------------------------------- registry


def _table():
    return ColumnTable.from_dict("plate", {
        "Control": [np.nan, *NIST],
        "Drug": [*MASKED],
        "Short": [1.0, 2.0],
    })


@pytest.mark.parametrize("key", ["outliers_grubbs", "outliers_rout"])
def test_registry_defaults(key):
    info = registry.REGISTRY[key]
    assert info.category == "Outliers" and info.tables == ("column",) and all(p.help for p in info.params)
    table = _table()
    before = table.data.copy()
    res = run_analysis(table, key)
    assert res.analysis == key and res.table == "plate"
    assert res.interpretation and res.why_chosen and len(res.assumptions) >= 1
    assert {"group": "Control", "row": 8, "value": 245.57} in res.extra["flagged"]
    for f in res.extra["flagged"]:  # row = zero-based row of the table column
        assert table.data[f["group"]].iloc[f["row"]] == f["value"]
        assert set(f) == {"group", "row", "value"}
    assert table.data.equals(before) and table.excluded == []  # nothing removed or excluded
    json.loads(res.model_dump_json())


def test_registry_options_and_row_mapping_with_exclusions():
    table = _table()
    assert run_analysis(table, "outliers_rout").extra["n_flagged"] == 3
    assert run_analysis(table, "outliers_grubbs").extra["n_flagged"] == 1
    assert "false discovery rate" in next(p for p in registry.REGISTRY["outliers_rout"].params if p.name == "q").help
    assert registry.REGISTRY["outliers_rout"].defaults()["q"] == 0.01
    only = run_analysis(table, "outliers_rout", groups=["Drug"], q=0.05)
    assert list(only.n) == ["Drug"] and only.params["q"] == 0.05
    assert run_analysis(table, "outliers_grubbs", groups="Control", iterative=True, alpha=0.01).alpha == 0.01
    for key in ("outliers_grubbs", "outliers_rout"):
        with pytest.raises(StatsError, match="Unknown group"):
            run_analysis(table, key, groups=["Nope"])
    # the user explicitly excludes the flagged value: it is then ignored and logged
    table.exclude([(8, "Control")])
    after = run_analysis(table, "outliers_grubbs", groups=["Control"])
    assert after.extra["flagged"] == [] and after.n["Control"] == 7
    assert any("explicitly excluded" in n for n in after.data_notes)
    # whole-row exclusions shift nothing: rows still refer to the original table
    table2 = ColumnTable.from_dict("t", {"X": [50.0, *NIST], "Y": np.arange(9.0)})
    table2.exclude([(0, "*")])
    res = run_analysis(table2, "outliers_rout", groups=["X"])
    assert res.extra["flagged"] == [{"group": "X", "row": 8, "value": 245.57}]
    assert table2.data["X"].iloc[8] == 245.57


# ------------------------------------------------------------------ property tests

_sample = st.lists(
    st.one_of(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
              st.integers(min_value=-3, max_value=3).map(float), st.just(float("nan"))),
    min_size=0, max_size=30,
)


def _n_finite(values):
    return sum(1 for v in values if not math.isnan(v))


@settings(max_examples=60, deadline=None)
@given(_sample, st.booleans(), st.sampled_from([0.01, 0.05, 0.2]))
def test_grubbs_never_crashes(values, iterative, alpha):
    if _n_finite(values) < 3:
        with pytest.raises(StatsError):
            ol.grubbs_test(values, alpha=alpha, iterative=iterative)
        return
    res = ol.grubbs_test(values, alpha=alpha, iterative=iterative)
    assert res["p_value"] is None or 0.0 <= res["p_value"] <= 1.0
    assert (res["p_value"] is None) == (res["g"] is None)
    if res["g"] is not None:
        n = res["n"]
        assert 0.0 <= res["g"] <= (n - 1) / math.sqrt(n) * (1 + 1e-9)
    assert len(res["flagged"]) <= (max(res["n"] - 2, 0) if iterative else 1)
    assert bool(res["flagged"]) == res["is_outlier"]
    seen = set()
    for f in res["flagged"]:
        assert values[f["index"]] == f["value"] and f["index"] not in seen
        seen.add(f["index"])


@settings(max_examples=60, deadline=None)
@given(_sample, st.sampled_from([0.001, 0.01, 0.1]))
def test_rout_column_never_crashes(values, q):
    if _n_finite(values) < 3:
        with pytest.raises(StatsError):
            ol.rout_column(values, q=q)
        return
    res = ol.rout_column(values, q=q)
    assert res["rsdr"] >= 0.0 and math.isfinite(res["location"])
    assert len(res["outlier_indices"]) <= math.ceil(0.3 * res["n"]) + 1  # only the top 30% are examined
    assert res["residuals"].size == len(values) and res["outlier_mask"].size == len(values)
    assert sorted(np.nonzero(res["outlier_mask"])[0].tolist()) == res["outlier_indices"]
    for f in res["flagged"]:
        assert values[f["index"]] == f["value"] and 0.0 <= f["p_value"] <= 1.0
    if res["rsdr"] == 0.0:
        assert res["outlier_indices"] == [] and res["warnings"]


_coordinate = st.one_of(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
                        st.integers(min_value=-2, max_value=2).map(float), st.just(float("nan")))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(_coordinate, _coordinate), min_size=0, max_size=18),
       st.sampled_from([0.01, 0.1]), st.booleans())
def test_rout_curve_fit_never_crashes(points, q, refit):
    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]
    complete = sum(1 for a, b in points if not (math.isnan(a) or math.isnan(b)))

    def line(xv, a, b):
        return a + b * xv

    if complete < 4:
        with pytest.raises(StatsError):
            ol.rout(x, y, line, [0.0, 1.0], q=q, refit=refit)
        return
    res = ol.rout(x, y, line, [0.0, 1.0], q=q, refit=refit)  # includes vertical / constant data
    assert res["n"] == complete and res["rsdr"] >= 0.0
    assert all(math.isfinite(v) for v in res["params_robust"] + res["params_ols"])
    assert len(res["outlier_indices"]) <= math.ceil(0.3 * complete) + 1
    assert sorted(np.nonzero(res["outlier_mask"])[0].tolist()) == res["outlier_indices"]
    for f in res["flagged"]:
        assert y[f["index"]] == f["value"] and x[f["index"]] == f["x"] and 0.0 <= f["p_value"] <= 1.0
    if res["ols_refit"] is not None:
        assert res["ols_refit"]["n_used"] == complete - len(res["outlier_indices"])
    if res["rsdr"] == 0.0:
        assert res["outlier_indices"] == [] and res["warnings"]


@settings(max_examples=40, deadline=None)
@given(st.dictionaries(st.sampled_from(["a", "b", "c"]), _sample, min_size=0, max_size=3), st.booleans())
def test_outlier_results_never_crash_and_only_flag(groups, iterative):
    testable = any(_n_finite(v) >= 3 for v in groups.values())
    snapshot = {k: list(v) for k, v in groups.items()}
    for call in (lambda: ol.grubbs_outliers(groups, iterative=iterative), lambda: ol.rout_outliers(groups)):
        if not testable:
            with pytest.raises(StatsError):
                call()
            continue
        res = call().finalize()
        assert res.n == {k: _n_finite(v) for k, v in groups.items()}
        for f in res.extra["flagged"]:
            assert groups[f["group"]][f["row"]] == f["value"]
        assert res.extra["n_flagged"] == len(res.extra["flagged"])
        assert "explicit decision" in res.interpretation
        json.loads(res.model_dump_json())
    assert all(str(snapshot[k]) == str(list(groups[k])) for k in groups)  # inputs untouched


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=5, max_size=25),
       st.integers(min_value=-50, max_value=50), st.sampled_from([0.25, 0.5, 2.0, 8.0]),
       st.randoms(use_true_random=False))
def test_flagging_is_invariant_to_units_and_order(values, shift, scale, rnd):
    x = np.array(values, dtype=float)
    y = x * scale + shift
    g1, g2 = ol.grubbs_test(x), ol.grubbs_test(y)
    if g1["g"] is not None:
        assert g2["g"] == pytest.approx(g1["g"], rel=1e-9)
        assert g2["p_value"] == pytest.approx(g1["p_value"], rel=1e-6, abs=1e-12)
    r1, r2 = ol.rout_column(x), ol.rout_column(y)
    assert r2["rsdr"] == pytest.approx(r1["rsdr"] * scale, rel=1e-6, abs=1e-9)
    assert r2["location"] == pytest.approx(r1["location"] * scale + shift, rel=1e-6, abs=1e-6)
    order = list(range(len(values)))
    rnd.shuffle(order)
    r3 = ol.rout_column(x[order])
    assert r3["rsdr"] == pytest.approx(r1["rsdr"], rel=1e-6, abs=1e-9)
    borderline = any(abs(t["p_value"] - t["threshold"]) < 1e-6 * t["threshold"] for t in r1["tests"])
    if not borderline:
        assert sorted(x[r1["outlier_indices"]]) == sorted(y[r2["outlier_indices"]] / scale - shift / scale)
        assert sorted(x[r1["outlier_indices"]]) == sorted(x[order][r3["outlier_indices"]])
