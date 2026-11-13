"""Survival analysis: Kaplan-Meier, log-rank / Gehan-Breslow-Wilcoxon, hazard ratios.

Validation layers (sources in tests/fixtures/sources/survival_contingency_nested.md):
* published worked example: Freireich et al. (1963) 6-MP leukaemia trial;
* cross-implementation: TorusPlot's own estimators vs lifelines;
* from-first-principles computations written out in this file;
* property-based edge cases (hypothesis).
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test as ll_logrank
from lifelines.statistics import multivariate_logrank_test as ll_multi
from lifelines.statistics import pairwise_logrank_test as ll_pairwise
from lifelines.utils import median_survival_times
from scipy import stats

from torusplot.data.tables import MultipleVariablesTable
from torusplot.stats import StatsError, registry, run_analysis
from torusplot.stats import survival as sv

pytestmark = pytest.mark.filterwarnings("ignore:Starting with pandas version 4.0")

# Freireich et al. (1963) Blood 21:699-716; remission times in weeks (1 = relapse, 0 = censored)
T_6MP = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
E_6MP = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
T_PLA = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
E_PLA = [1] * 21


@pytest.fixture(autouse=True)
def _registry_ready(monkeypatch):
    """The module under test registered itself on import; do not require sibling analysis
    modules (written by other people, possibly not importable yet) to load."""
    monkeypatch.setattr(registry, "_loaded", True)


def _random_groups(seed, k=3, n=25, censor=0.3, round_to=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(k):
        t = np.round(rng.exponential(8 + 4 * i, n) + 0.5, round_to)  # rounding creates ties
        e = (rng.random(n) > censor).astype(int)
        out[f"g{i + 1}"] = (t, e)
    return out


def _km_first_principles(time, event):
    """Textbook product-limit table written as an explicit loop (independent of the module)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    rows = []
    s, gw = 1.0, 0.0
    for t in sorted(set(time.tolist())):
        n = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        c = int(((time == t) & (event == 0)).sum())
        s *= 1 - d / n
        gw += d / (n * (n - d)) if n > d else math.inf
        rows.append((t, n, d, c, s, gw))
    return rows


def _logrank_first_principles(groups, gehan=False):
    """Observed, expected and the hypergeometric covariance, looping over event times."""
    names = list(groups)
    k = len(names)
    t_all = np.concatenate([np.asarray(groups[g][0], float) for g in names])
    e_all = np.concatenate([np.asarray(groups[g][1], int) for g in names])
    u = np.zeros(k)
    v = np.zeros((k, k))
    obs = np.zeros(k)
    exp = np.zeros(k)
    for t in sorted(set(t_all[e_all == 1].tolist())):
        n_i = np.array([(np.asarray(groups[g][0], float) >= t).sum() for g in names], float)
        d_i = np.array([((np.asarray(groups[g][0], float) == t) & (np.asarray(groups[g][1], int) == 1)).sum()
                        for g in names], float)
        n, d = n_i.sum(), d_i.sum()
        w = n if gehan else 1.0
        e_i = d * n_i / n
        obs += d_i
        exp += e_i
        u += w * (d_i - e_i)
        if n > 1:
            for a in range(k):
                for b in range(k):
                    delta = 1.0 if a == b else 0.0
                    v[a, b] += w * w * d * (n - d) / (n - 1) * (n_i[a] / n) * (delta - n_i[b] / n)
    return u, v, obs, exp


# ====================================================== published worked example
def test_freireich_kaplan_meier_published_values():
    km = sv.kaplan_meier(T_6MP, E_6MP)
    surv = {row["time"]: row["survival"] for row in km["table"]}
    # Kaplan-Meier estimates for the 6-MP arm as printed in survival textbooks (3 decimals)
    published = {6: 0.857, 7: 0.807, 10: 0.753, 13: 0.690, 16: 0.627, 22: 0.538, 23: 0.448}
    for t, s in published.items():
        assert round(surv[float(t)], 3) == s
    assert km["median"] == 23.0
    assert (km["n"], km["events"], km["censored"]) == (21, 9, 12)
    # first row of the life table: 21 at risk at week 6, 3 relapses, 1 censored
    first = km["table"][0]
    assert (first["time"], first["at_risk"], first["events"], first["censored"]) == (6.0, 21, 3, 1)

    placebo = sv.kaplan_meier(T_PLA, E_PLA)
    assert placebo["median"] == 8.0
    # no censoring in the placebo arm: KM equals the fraction still in remission
    for row in placebo["table"]:
        assert row["survival"] == pytest.approx(np.mean(np.array(T_PLA) > row["time"]), abs=1e-12)
    assert placebo["table"][-1]["survival"] == 0.0
    assert placebo["table"][-1]["ci_low"] is None and placebo["table"][-1]["se"] is None


def test_freireich_logrank_published_value():
    res = sv.logrank_test({"6-MP": (T_6MP, E_6MP), "Placebo": (T_PLA, E_PLA)})
    assert round(res["statistic"], 2) == 16.79  # published log-rank chi-square
    assert res["df"] == 1
    assert res["p_value"] < 0.0001
    assert res["observed"] == [9.0, 21.0]  # 9 relapses on 6-MP, 21 on placebo
    assert sum(res["expected"]) == pytest.approx(30.0, abs=1e-10)  # expected counts add up to all events


# ============================================================ cross-checks vs lifelines
@pytest.mark.parametrize("data", [(T_6MP, E_6MP), (T_PLA, E_PLA),
                                  _random_groups(1)["g1"], _random_groups(2, n=60, round_to=1)["g2"]])
def test_km_matches_lifelines_and_first_principles(data):
    t, e = data
    km = sv.kaplan_meier(t, e)
    kmf = KaplanMeierFitter().fit(t, e)
    z = stats.norm.ppf(0.975)
    for row, (tt, n, d, c, s, gw) in zip(km["table"], _km_first_principles(t, e)):
        assert (row["time"], row["at_risk"], row["events"], row["censored"]) == (tt, n, d, c)
        assert row["survival"] == pytest.approx(s, rel=1e-12, abs=1e-15)
        assert row["survival"] == pytest.approx(float(kmf.survival_function_at_times(tt).iloc[0]), rel=1e-9, abs=1e-12)
        if 0 < s < 1:
            assert row["se"] == pytest.approx(s * math.sqrt(gw), rel=1e-12)  # Greenwood
            lo, hi = kmf.confidence_interval_.loc[tt]
            assert row["ci_low"] == pytest.approx(lo, rel=1e-8)
            assert row["ci_high"] == pytest.approx(hi, rel=1e-8)
            # log-log limits from the defining formula
            theta = z * math.sqrt(gw) / abs(math.log(s))
            assert row["ci_low"] == pytest.approx(s ** math.exp(theta), rel=1e-12)
            assert row["ci_high"] == pytest.approx(s ** math.exp(-theta), rel=1e-12)
    # median and Brookmeyer-Crowley interval
    ll_median = kmf.median_survival_time_
    assert km["median"] == (None if math.isinf(ll_median) else ll_median)
    ci = median_survival_times(kmf.confidence_interval_).iloc[0]
    for mine, theirs in zip((km["median_ci_low"], km["median_ci_high"]), ci):
        assert mine == (None if math.isinf(theirs) else pytest.approx(theirs))


def test_km_linear_ci_and_ci_level():
    km = sv.kaplan_meier(T_6MP, E_6MP, ci_method="linear", ci_level=0.90)
    z = stats.norm.ppf(0.95)
    for row in km["table"]:
        assert row["ci_low"] == pytest.approx(max(0.0, row["survival"] - z * row["se"]), abs=1e-12)
        assert row["ci_high"] == pytest.approx(min(1.0, row["survival"] + z * row["se"]), abs=1e-12)
    wide = sv.kaplan_meier(T_6MP, E_6MP, ci_level=0.99)["table"][5]
    narrow = sv.kaplan_meier(T_6MP, E_6MP, ci_level=0.80)["table"][5]
    assert wide["ci_low"] < narrow["ci_low"] < narrow["ci_high"] < wide["ci_high"]
    with pytest.raises(StatsError):
        sv.kaplan_meier(T_6MP, E_6MP, ci_method="bootstrap")
    with pytest.raises(StatsError):
        sv.kaplan_meier(T_6MP, E_6MP, ci_level=1.5)


def test_km_curve_coordinates():
    km = sv.kaplan_meier(T_6MP, E_6MP)
    curve = km["curve"]
    assert curve["time"][0] == 0.0 and curve["survival"][0] == 1.0
    assert curve["ci_low"][0] == 1.0 and curve["ci_high"][0] == 1.0
    assert curve["at_risk"][0] == 21
    n = len(curve["time"])
    assert all(len(curve[key]) == n for key in ("survival", "ci_low", "ci_high", "at_risk"))
    assert curve["time"][1:] == [row["time"] for row in km["table"]]
    assert curve["at_risk"][1:] == [row["at_risk"] for row in km["table"]]
    # censoring ticks: distinct censoring times with the survival estimate there
    assert curve["censor_time"] == [6.0, 9.0, 10.0, 11.0, 17.0, 19.0, 20.0, 25.0, 32.0, 34.0, 35.0]
    assert curve["censor_survival"][0] == pytest.approx(0.857142857, abs=1e-8)
    assert len(curve["censor_time"]) == len(curve["censor_survival"])


def test_median_not_reached_and_exact_half():
    km = sv.kaplan_meier([1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0])
    assert km["median"] is None and km["median_ci_high"] is None
    assert "not reached" in km["median_note"]
    json.dumps(km)  # no infinities / NaN anywhere
    # S(t) hits exactly 0.5 at t = 2 and stays there until 5
    km = sv.kaplan_meier([1, 2, 5, 7], [1, 1, 1, 1])
    assert km["median"] == 2.0
    assert any("midpoint, 3.5" in note for note in km["notes"])
    assert KaplanMeierFitter().fit([1, 2, 5, 7], [1, 1, 1, 1]).median_survival_time_ == 2.0


@pytest.mark.parametrize("seed,k", [(3, 2), (4, 2), (5, 3), (6, 4)])
def test_logrank_and_gehan_match_lifelines(seed, k):
    groups = _random_groups(seed, k=k)
    names = list(groups)
    times = np.concatenate([groups[g][0] for g in names])
    events = np.concatenate([groups[g][1] for g in names])
    labels = np.concatenate([[g] * len(groups[g][0]) for g in names])
    for weighting, ll_weight in (("logrank", None), ("gehan", "wilcoxon")):
        mine = sv.logrank_test(groups, weighting=weighting)
        theirs = ll_multi(times, labels, events, weightings=ll_weight)
        assert mine["statistic"] == pytest.approx(theirs.test_statistic, rel=1e-8)
        assert mine["p_value"] == pytest.approx(theirs.p_value, rel=1e-6)
        assert mine["df"] == k - 1
        if k == 2:
            (ta, ea), (tb, eb) = groups[names[0]], groups[names[1]]
            two = ll_logrank(ta, tb, ea, eb, weightings=ll_weight)
            assert mine["statistic"] == pytest.approx(two.test_statistic, rel=1e-8)
        # independent loop implementation
        u, v, obs, exp = _logrank_first_principles(groups, gehan=weighting == "gehan")
        chi2 = float(u[:-1] @ np.linalg.solve(v[:-1, :-1], u[:-1]))
        assert mine["statistic"] == pytest.approx(chi2, rel=1e-9)
        assert mine["p_value"] == pytest.approx(stats.chi2.sf(chi2, k - 1), rel=1e-9)
        assert mine["observed"] == pytest.approx(obs)
        assert mine["expected"] == pytest.approx(exp)


def test_freireich_gehan_matches_lifelines():
    mine = sv.logrank_test({"6-MP": (T_6MP, E_6MP), "Placebo": (T_PLA, E_PLA)}, weighting="gehan")
    theirs = ll_logrank(T_6MP, T_PLA, E_6MP, E_PLA, weightings="wilcoxon")
    assert mine["statistic"] == pytest.approx(theirs.test_statistic, rel=1e-9)
    assert mine["p_value"] == pytest.approx(theirs.p_value, rel=1e-6)
    assert mine["test_name"] == "Gehan-Breslow-Wilcoxon test"


def test_logrank_two_groups_equals_simple_formula():
    # for two groups the statistic is (O1 - E1)^2 / V
    u, v, obs, exp = _logrank_first_principles({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)})
    res = sv.logrank_test({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)})
    assert res["statistic"] == pytest.approx((obs[0] - exp[0]) ** 2 / v[0, 0], rel=1e-12)


def test_trend_test():
    groups = _random_groups(11, k=4)
    res = sv.logrank_test(groups, trend=True)
    u, v, _, _ = _logrank_first_principles(groups)
    s = np.array([1.0, 2.0, 3.0, 4.0])
    chi2 = float(s @ u) ** 2 / float(s @ v @ s)
    assert res["trend"]["statistic"] == pytest.approx(chi2, rel=1e-9)
    assert res["trend"]["p_value"] == pytest.approx(stats.chi2.sf(chi2, 1), rel=1e-9)
    assert res["trend"]["statistic"] <= res["statistic"] + 1e-9  # a 1-df component of the omnibus test
    # shifting / rescaling the scores does not matter; custom scores are used
    alt = sv.logrank_test(groups, trend=True, scores=[10, 20, 30, 40])
    assert alt["trend"]["statistic"] == pytest.approx(chi2, rel=1e-9)
    other = sv.logrank_test(groups, trend=True, scores=[0, 1, 5, 6])
    assert other["trend"]["statistic"] != pytest.approx(chi2, rel=1e-6)
    # with two groups the trend test is the log-rank test itself
    two = sv.logrank_test({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)}, trend=True)
    assert two["trend"]["statistic"] == pytest.approx(two["statistic"], rel=1e-10)
    with pytest.raises(StatsError):
        sv.logrank_test(groups, trend=True, scores=[1, 2])


def test_trend_detects_ordered_doses():
    rng = np.random.default_rng(8)
    groups = {f"dose{i}": (rng.exponential(5 + 6 * i, 40), np.ones(40, int)) for i in range(4)}
    res = sv.logrank_test(groups, trend=True)
    assert res["trend"]["p_value"] < 0.001 and res["trend"]["z"] < 0  # fewer events than expected at high dose


# ================================================================== hazard ratios
def test_hazard_ratio_formulas_and_direction():
    hr = sv.hazard_ratio(T_6MP, E_6MP, T_PLA, E_PLA)
    u, v, obs, exp = _logrank_first_principles({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)})
    z = stats.norm.ppf(0.975)
    log_mh = (obs[0] - exp[0]) / v[0, 0]
    mh = hr["mantel_haenszel"]
    assert mh["hr"] == pytest.approx(math.exp(log_mh), rel=1e-12)
    assert mh["ci_low"] == pytest.approx(math.exp(log_mh - z / math.sqrt(v[0, 0])), rel=1e-12)
    assert mh["ci_high"] == pytest.approx(math.exp(log_mh + z / math.sqrt(v[0, 0])), rel=1e-12)
    lr = hr["logrank"]
    ratio = (obs[0] / exp[0]) / (obs[1] / exp[1])
    se = math.sqrt(1 / exp[0] + 1 / exp[1])
    assert lr["hr"] == pytest.approx(ratio, rel=1e-12)
    assert lr["ci_low"] == pytest.approx(math.exp(math.log(ratio) - z * se), rel=1e-12)
    assert lr["ci_high"] == pytest.approx(math.exp(math.log(ratio) + z * se), rel=1e-12)
    # 6-MP protects: hazard ratio (6-MP relative to placebo) well below 1, CI excludes 1
    assert mh["hr"] < 1 and mh["ci_high"] < 1 and lr["ci_high"] < 1
    # swapping the groups gives the reciprocal
    rev = sv.hazard_ratio(T_PLA, E_PLA, T_6MP, E_6MP)
    assert rev["mantel_haenszel"]["hr"] == pytest.approx(1 / mh["hr"], rel=1e-10)
    assert rev["mantel_haenszel"]["ci_low"] == pytest.approx(1 / mh["ci_high"], rel=1e-10)
    assert rev["logrank"]["hr"] == pytest.approx(1 / lr["hr"], rel=1e-10)
    # the MH confidence interval excludes 1 exactly when the log-rank test is significant
    res = sv.logrank_test({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)})
    assert res["statistic"] == pytest.approx((mh["log_hr"] / mh["se_log_hr"]) ** 2, rel=1e-10)


def test_hazard_ratio_degenerate():
    # no events in group A: the O/E ratio is undefined, the MH estimate still exists
    hr = sv.hazard_ratio([5, 6, 7], [0, 0, 0], [1, 2, 3, 8], [1, 1, 1, 1])
    assert hr["logrank"]["hr"] is None and hr["mantel_haenszel"]["hr"] is not None
    assert any("undefined" in n for n in hr["notes"])
    none = sv.hazard_ratio([1, 2], [0, 0], [3, 4], [0, 0])
    assert none["mantel_haenszel"]["hr"] is None and none["logrank"]["hr"] is None
    with pytest.raises(StatsError):
        sv.hazard_ratio([1, 2], [1, 1], [3, 4], [1, 1], ci_level=0)


# ============================================================== bundled analysis
def _freireich_result(**kwargs):
    return sv.survival_analysis(T_6MP + T_PLA, E_6MP + E_PLA, ["6-MP"] * 21 + ["Placebo"] * 21, **kwargs).finalize()


def test_survival_analysis_two_groups():
    res = _freireich_result(test="both")
    assert res.test_name == "Log-rank (Mantel-Cox) test"
    assert res.statistic_name == "chi2" and round(res.statistic, 2) == 16.79 and res.df == 1
    assert res.significant is True and res.p_value < 0.0001
    assert res.n == {"6-MP": 21, "Placebo": 21} and res.n_total == 42
    desc = {row["group"]: row for row in res.descriptives}
    assert (desc["6-MP"]["events"], desc["6-MP"]["censored"], desc["6-MP"]["median"]) == (9, 12, 23.0)
    assert desc["Placebo"]["median"] == 8.0 and desc["Placebo"]["median_ci_low"] == 4.0
    assert desc["6-MP"]["median_ci_high"] is None and "not reached" in desc["6-MP"]["median_note"]
    assert res.parameter("Median survival: 6-MP").value == 23.0
    hr = res.parameter("Hazard ratio (Mantel-Haenszel), 6-MP vs Placebo")
    assert hr.value == pytest.approx(0.1943, abs=5e-5) and hr.ci_low < hr.value < hr.ci_high < 1
    assert res.parameter("Hazard ratio (log-rank), 6-MP vs Placebo").value == pytest.approx(0.2393, abs=5e-5)
    assert res.effect_sizes[0].name == "Hazard ratio (Mantel-Haenszel)"
    assert res.effect_sizes[0].target == "6-MP vs Placebo"
    assert "6-MP relative to Placebo" in res.interpretation and "lower rate" in res.interpretation
    assert "Gehan-Breslow-Wilcoxon" in res.interpretation and "Kaplan-Meier" in res.methods_text
    # Gehan result available next to the headline log-rank test
    gehan = sv.logrank_test({"a": (T_6MP, E_6MP), "b": (T_PLA, E_PLA)}, weighting="gehan")
    assert res.extra["tests"]["gehan"]["statistic"] == pytest.approx(gehan["statistic"])
    assert set(res.extra["curves"]) == {"6-MP", "Placebo"} and set(res.extra["tables"]) == {"6-MP", "Placebo"}
    assert res.extra["curves"]["6-MP"]["time"][0] == 0.0
    statuses = {a.name: a.status for a in res.assumptions}
    assert statuses["Proportional hazards"] == "info"  # curves do not cross
    assert statuses["Enough events for the chi-square approximation"] == "pass"
    assert res.comparisons == []
    json.loads(res.model_dump_json())


def test_survival_analysis_gehan_as_primary():
    res = _freireich_result(test="gehan")
    theirs = ll_logrank(T_6MP, T_PLA, E_6MP, E_PLA, weightings="wilcoxon")
    assert res.test_name == "Gehan-Breslow-Wilcoxon test"
    assert res.statistic == pytest.approx(theirs.test_statistic, rel=1e-9)
    assert "logrank" not in res.extra["tests"]
    assert any("Gehan" in c for c in res.citations)


def test_survival_analysis_three_groups_pairwise():
    groups = _random_groups(21, k=3, n=30)
    names = list(groups)
    time = np.concatenate([groups[g][0] for g in names])
    event = np.concatenate([groups[g][1] for g in names])
    label = np.concatenate([[g] * 30 for g in names])
    res = sv.survival_analysis(time, event, label, trend=True).finalize()
    omnibus = ll_multi(time, label, event)
    assert res.statistic == pytest.approx(omnibus.test_statistic, rel=1e-8) and res.df == 2
    assert res.p_value == pytest.approx(omnibus.p_value, rel=1e-6)
    pairwise = ll_pairwise(time, label, event).summary
    assert len(res.comparisons) == 3 and res.correction == "Bonferroni"
    assert "3 pairwise comparisons among 3 groups" in res.comparison_family
    for comp in res.comparisons:
        row = pairwise.loc[(comp.group_a, comp.group_b)]
        assert comp.statistic == pytest.approx(row["test_statistic"], rel=1e-8)
        assert comp.p_value == pytest.approx(row["p"], rel=1e-6)
        assert comp.p_adjusted == pytest.approx(min(1.0, 3 * comp.p_value), rel=1e-12)
        assert comp.id.startswith(res.id) and comp.significant == (comp.p_adjusted < 0.05)
        assert comp.effect_size.name.startswith("Hazard ratio")
    assert res.extra["trend"]["df"] == 1 and 0 <= res.extra["trend"]["p_value"] <= 1
    assert "trend" in res.interpretation
    holm = sv.survival_analysis(time, event, label, adjust="holm").finalize()
    assert holm.correction == "Holm"
    assert all(h.p_adjusted <= b.p_adjusted + 1e-15 for h, b in zip(holm.comparisons, res.comparisons))
    with pytest.raises(StatsError):
        sv.survival_analysis(time, event, label, adjust="tukey")


def test_group_order_invariance_and_reordering():
    groups = _random_groups(31, k=3)
    names = list(groups)
    time = np.concatenate([groups[g][0] for g in names])
    event = np.concatenate([groups[g][1] for g in names])
    label = np.concatenate([[g] * 25 for g in names])
    base = sv.survival_analysis(time, event, label)
    reordered = sv.survival_analysis(time, event, label, group_order=["g3", "g1", "g2"])
    assert list(reordered.n) == ["g3", "g1", "g2"]
    assert reordered.statistic == pytest.approx(base.statistic, rel=1e-10)
    with pytest.raises(StatsError):
        sv.survival_analysis(time, event, label, group_order=["g1", "g2"])
    # multiplying all times by a constant changes nothing (rank-based tests)
    scaled = sv.survival_analysis(time * 7.0, event, label)
    assert scaled.statistic == pytest.approx(base.statistic, rel=1e-10)


def test_single_group_gives_curve_only():
    res = sv.survival_analysis(T_6MP, E_6MP).finalize()
    assert res.test_name == "Kaplan-Meier survival estimate"
    assert res.statistic is None and res.p_value is None and res.significant is None
    assert res.n == {"All subjects": 21}
    assert "only one group" in res.interpretation
    assert list(res.extra["curves"]) == ["All subjects"]
    json.loads(res.model_dump_json())
    never = sv.survival_analysis([1, 2, 3, 4], [1, 0, 0, 0])
    assert "not reached" in never.interpretation


def test_crossing_curves_flagged():
    early = ([1, 2, 3, 4, 20, 21, 22, 23], [1, 1, 1, 0, 0, 0, 0, 0])
    late = ([8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 1, 1, 1, 1, 1, 1])
    res = sv.survival_analysis(early[0] + late[0], early[1] + late[1], ["early"] * 8 + ["late"] * 8)
    check = next(a for a in res.assumptions if a.name == "Proportional hazards")
    assert check.status == "warn" and "cross" in check.detail
    events = next(a for a in res.assumptions if a.name.startswith("Enough events"))
    assert events.status == "warn" and "early" in events.detail


def test_degenerate_survival_inputs():
    # no events anywhere: curves are returned, the comparison is not
    res = sv.survival_analysis([1, 2, 3, 4, 5, 6], [0] * 6, ["a"] * 3 + ["b"] * 3).finalize()
    assert res.statistic is None and res.p_value is None
    assert any("No events were observed in any group" in w for w in res.warnings)
    assert res.extra["curves"]["a"]["survival"] == [1.0, 1.0, 1.0, 1.0]
    json.loads(res.model_dump_json())
    with pytest.raises(StatsError, match="No events"):
        sv.logrank_test({"a": ([1, 2], [0, 0]), "b": ([3, 4], [0, 0])})
    # one all-censored group is fine (it has zero observed events)
    res = sv.survival_analysis([5, 6, 7, 1, 2, 3], [0, 0, 0, 1, 1, 1], ["a"] * 3 + ["b"] * 3)
    assert res.p_value is not None and any("no events were observed" in w for w in res.warnings)
    # single-subject groups
    res = sv.survival_analysis([5, 1, 2, 3], [1, 1, 1, 0], ["a", "b", "b", "b"])
    assert any("single subject" in w for w in res.warnings) and 0 <= res.p_value <= 1
    # every subject fails at the same time: no information to compare the groups
    res = sv.survival_analysis([2, 2, 2, 2], [1, 1, 1, 1], ["a", "a", "b", "b"])
    assert res.statistic is None and any("undefined" in w for w in res.warnings)
    # a group that is never at risk when an event happens loses its degree of freedom
    res = sv.logrank_test({"a": ([1, 1], [0, 0]), "b": ([2, 3, 4], [1, 1, 1]), "c": ([2, 3, 5], [1, 0, 1])})
    assert res["df"] == 1 and any("degree" in n for n in res["notes"])
    three = sv.survival_analysis([1, 1, 2, 3, 4, 2, 3, 5], [0, 0, 1, 1, 1, 1, 0, 1], list("aabbbccc"))
    assert three.comparison("a", "b").p_value is None  # nothing to compare in that pair


def test_input_validation():
    with pytest.raises(StatsError, match="only 1 .* or 0"):
        sv.kaplan_meier([1, 2, 3], [1, 2, 0])
    with pytest.raises(StatsError, match="negative"):
        sv.kaplan_meier([1, -2, 3], [1, 1, 0])
    with pytest.raises(StatsError, match="same number"):
        sv.kaplan_meier([1, 2, 3], [1, 0])
    with pytest.raises(StatsError, match="coded with numbers"):
        sv.kaplan_meier([1, 2, 3], ["dead", "alive", "dead"])
    with pytest.raises(StatsError, match="must be numbers"):
        sv.kaplan_meier(["a", "b"], [1, 0])
    with pytest.raises(StatsError, match="No complete rows"):
        sv.kaplan_meier([np.nan, np.nan], [1, 0])
    with pytest.raises(StatsError, match="finite"):
        sv.kaplan_meier([1, np.inf], [1, 0])
    with pytest.raises(StatsError, match="No time"):
        sv.kaplan_meier(None, [1, 0])
    with pytest.raises(StatsError, match="group column has"):
        sv.survival_analysis([1, 2, 3], [1, 0, 1], ["a", "b"])
    with pytest.raises(StatsError, match="at least two groups"):
        sv.logrank_test({"a": ([1, 2], [1, 1])})
    with pytest.raises(StatsError, match="weighting"):
        sv.logrank_test({"a": ([1, 2], [1, 1]), "b": ([1, 3], [1, 1])}, weighting="peto")
    with pytest.raises(StatsError, match="test must be"):
        sv.survival_analysis([1, 2], [1, 1], ["a", "b"], test="cox")
    with pytest.raises(StatsError):
        sv.survival_analysis([1, 2], [1, 1], ["a", "b"], alpha=2)
    with pytest.raises(TypeError):
        sv.logrank_test([([1, 2], [1, 1])])
    with pytest.raises(TypeError):
        sv.logrank_test({"a": 5, "b": 6})


def test_booleans_missing_values_and_labels():
    time = [3.0, 5.0, np.nan, 7.0, 2.0, 9.0, 4.0]
    event = [True, False, True, True, None, True, False]
    group = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, None]
    res = sv.survival_analysis(time, event, group)
    assert res.n == {"1": 2, "2": 2}  # float labels shown without ".0"; 3 incomplete rows dropped
    assert any("3 row(s)" in note for note in res.data_notes)
    km = sv.kaplan_meier(np.array([1, 2, 3]), np.array([True, False, True]))
    assert km["events"] == 2
    labels = sv.survival_analysis([1, 2, 3, 4], [1, 1, 1, 1], [True, True, 2.5, 2.5])
    assert set(labels.n) == {"True", "2.5"}
    assert sv.survival_analysis([1, 2, 3, 4], ["1", "0", "1", "1"], ["x", "x", " ", "y"]).n == {"x": 2, "y": 1}


# ======================================================================= registry
def _table(time_name="Time (weeks)", event_name="Status", group_name="Arm"):
    frame = pd.DataFrame({
        "Patient": [f"p{i}" for i in range(42)],
        time_name: T_6MP + T_PLA,
        event_name: E_6MP + E_PLA,
        group_name: ["6-MP"] * 21 + ["Placebo"] * 21,
    })
    return MultipleVariablesTable("Freireich", frame)


def test_registry_survival_defaults():
    info = registry.REGISTRY["survival"]
    assert info.category == "Survival" and info.tables == ("multiple_variables",)
    assert all(p.help for p in info.params)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run_analysis(_table(), "survival")
    assert res.analysis == "survival" and res.table == "Freireich"
    assert round(res.statistic, 2) == 16.79 and res.significant is True
    assert res.interpretation and res.why_chosen and len(res.assumptions) >= 1
    assert res.extra["columns"] == {"time": "Time (weeks)", "event": "Status", "group": "Arm"}
    assert any("Time column chosen automatically: 'Time (weeks)'" in n for n in res.data_notes)
    assert any("Event column chosen automatically: 'Status'" in n for n in res.data_notes)
    assert any("Group column chosen automatically: 'Arm'" in n for n in res.data_notes)
    payload = json.loads(res.model_dump_json())
    assert payload["extra"]["curves"]["6-MP"]["survival"][0] == 1.0
    assert res.params["test"] == "logrank"


def test_registry_survival_column_fallbacks_and_errors():
    # no recognisable names: first numeric = time, second numeric = event, a categorical with repeats = group
    frame = pd.DataFrame({"id": [f"s{i}" for i in range(42)], "x1": T_6MP + T_PLA, "x2": E_6MP + E_PLA,
                          "label": ["A"] * 21 + ["B"] * 21})
    res = run_analysis(MultipleVariablesTable("t", frame), "survival", test="both")
    assert res.extra["columns"] == {"time": "x1", "event": "x2", "group": "label"}
    assert round(res.statistic, 2) == 16.79
    # no usable group column -> one curve
    res = run_analysis(MultipleVariablesTable("t", frame[["id", "x1", "x2"]]), "survival")
    assert res.extra["columns"]["group"] is None and res.statistic is None
    assert any("one curve" in n for n in res.data_notes)
    # explicit columns, numeric group column, and a 'censored' column name triggers a coding reminder
    frame2 = pd.DataFrame({"t": T_6MP + T_PLA, "censored": E_6MP + E_PLA, "dose": [0] * 21 + [1] * 21})
    res = run_analysis(MultipleVariablesTable("t", frame2), "survival", time="t", event="censored", group="dose")
    assert res.n == {"0": 21, "1": 21}
    assert any("Check the coding" in n for n in res.data_notes)
    with pytest.raises(StatsError, match="not in the table"):
        run_analysis(_table(), "survival", time="nope")
    with pytest.raises(StatsError, match="No numeric column"):
        run_analysis(MultipleVariablesTable("t", pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})), "survival")
    with pytest.raises(StatsError, match="event code"):
        run_analysis(MultipleVariablesTable("t", pd.DataFrame({"time": [1.0, 2.0], "b": ["u", "v"]})), "survival")
    # text event codes are refused with an explanation
    bad = pd.DataFrame({"time": [1.0, 2.0, 3.0], "outcome": ["dead", "alive", "dead"]})
    with pytest.raises(StatsError, match="coded with numbers"):
        run_analysis(MultipleVariablesTable("t", bad), "survival", event="outcome")


def test_registry_respects_user_exclusions():
    table = _table()
    table.exclude([(0, "*")])
    res = run_analysis(table, "survival")
    assert res.n["6-MP"] == 20
    assert any("excluded by the user" in n for n in res.data_notes)


# ================================================================ property-based
_subject = st.tuples(st.integers(0, 12), st.booleans())


@settings(max_examples=60, deadline=None)
@given(st.lists(_subject, min_size=1, max_size=40), st.sampled_from(["log-log", "linear"]))
def test_property_km_is_a_valid_survival_curve(subjects, ci_method):
    time = [float(t) for t, _ in subjects]
    event = [int(e) for _, e in subjects]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        km = sv.kaplan_meier(time, event, ci_method=ci_method)
    surv = km["curve"]["survival"]
    assert all(0.0 <= s <= 1.0 for s in surv)
    assert all(b <= a + 1e-12 for a, b in zip(surv, surv[1:]))  # non-increasing
    for row in km["table"]:
        if row["ci_low"] is not None:
            assert 0.0 <= row["ci_low"] <= row["survival"] + 1e-12
            assert row["survival"] - 1e-12 <= row["ci_high"] <= 1.0
    assert km["n"] == len(subjects) and km["events"] + km["censored"] == km["n"]
    assert sum(r["events"] + r["censored"] for r in km["table"]) == km["n"]
    json.dumps(km, allow_nan=False)
    # agreement with lifelines at every time point
    kmf = KaplanMeierFitter().fit(time, event)
    for row in km["table"]:
        assert row["survival"] == pytest.approx(float(kmf.survival_function_at_times(row["time"]).iloc[0]), abs=1e-9)
    if sum(event) == 0:
        assert km["median"] is None and all(s == 1.0 for s in surv)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 10), st.booleans(), st.integers(0, 2)), min_size=2, max_size=45),
    st.sampled_from(["logrank", "gehan", "both"]),
    st.randoms(use_true_random=False),
)
def test_property_survival_analysis_never_crashes(subjects, test, rnd):
    time = [float(t) for t, _, _ in subjects]
    event = [int(e) for _, e, _ in subjects]
    group = [f"g{g}" for _, _, g in subjects]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sv.survival_analysis(time, event, group, test=test).finalize()
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    assert res.statistic is None or res.statistic >= 0.0
    for comp in res.comparisons:
        assert comp.p_value is None or 0.0 <= comp.p_value <= comp.p_adjusted <= 1.0
    json.loads(res.model_dump_json())
    assert "Infinity" not in res.model_dump_json() and "NaN" not in res.model_dump_json()
    if len(set(group)) >= 2 and sum(event) == 0:
        assert res.statistic is None and res.warnings
    # shuffling the rows (which also permutes the group order) leaves the omnibus test unchanged
    order = list(range(len(subjects)))
    rnd.shuffle(order)
    shuffled = sv.survival_analysis([time[i] for i in order], [event[i] for i in order],
                                    [group[i] for i in order], test=test)
    if res.statistic is None:
        assert shuffled.statistic is None
    else:
        assert shuffled.statistic == pytest.approx(res.statistic, rel=1e-8, abs=1e-10)
        # and lifelines agrees whenever the full k - 1 degrees of freedom are available
        k = len(set(group))
        if res.df == k - 1 and test != "gehan":
            theirs = ll_multi(time, group, event)
            assert res.statistic == pytest.approx(theirs.test_statistic, rel=1e-6, abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.lists(_subject, min_size=1, max_size=20), st.lists(_subject, min_size=1, max_size=20))
def test_property_hazard_ratio(a, b):
    ta, ea = [float(t) for t, _ in a], [int(e) for _, e in a]
    tb, eb = [float(t) for t, _ in b], [int(e) for _, e in b]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hr = sv.hazard_ratio(ta, ea, tb, eb)
        rev = sv.hazard_ratio(tb, eb, ta, ea)
    json.dumps(hr, allow_nan=False)
    for key in ("mantel_haenszel", "logrank"):
        est = hr[key]
        if est["hr"] is not None and est["ci_low"] is not None and est["ci_high"] is not None:
            assert 0 <= est["ci_low"] <= est["hr"] <= est["ci_high"]
            if rev[key]["hr"] is not None and est["hr"] > 0:
                assert rev[key]["hr"] == pytest.approx(1.0 / est["hr"], rel=1e-8)
