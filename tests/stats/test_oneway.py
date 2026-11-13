"""One-way ANOVA family, Kruskal-Wallis and Friedman tests.

Reference values: published worked examples (sources in
``tests/fixtures/sources/oneway.md``), cross-checks against scipy / statsmodels, and
from-formula computations written out here. NIST StRD certified values are tested in
``test_nist_anova.py``.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway as sm_anova_oneway

from torusplot.data.tables import ColumnTable
from torusplot.stats import StatsError, get_analysis, registry, run_analysis
from torusplot.stats import oneway as ow

nan = float("nan")

# ------------------------------------------------------------------------- data

# Dobson (1983) Table 7.1: dried weight of plants under a control and two treatments.
PLANT = {
    "ctrl": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14],
    "trt1": [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69],
    "trt2": [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26],
}

# Hollander & Wolfe (1973), p. 116: mucociliary efficiency (rate of removal of dust).
MUCOCILIARY = {
    "normal": [2.9, 3.0, 2.5, 2.6, 3.2],
    "obstructive airway disease": [3.8, 2.7, 4.0, 2.4],
    "asbestosis": [2.8, 3.4, 3.7, 2.2, 2.0],
}

# New York daily ozone (ppb), May-September 1973 (Chambers, Cleveland, Kleiner & Tukey
# 1983); nan = no measurement that day.
OZONE = {
    "May": [41, 36, 12, 18, nan, 28, 23, 19, 8, nan, 7, 16, 11, 14, 18, 14, 34, 6, 30, 11, 1,
            11, 4, 32, nan, nan, nan, 23, 45, 115, 37],
    "June": [nan, nan, nan, nan, nan, nan, 29, nan, 71, 39, nan, nan, 23, nan, nan, 21, 37, 20,
             12, 13, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    "July": [135, 49, 32, nan, 64, 40, 77, 97, 97, 85, nan, 10, 27, nan, 7, 48, 35, 61, 79, 63,
             16, nan, nan, 80, 108, 20, 52, 82, 50, 64, 59],
    "August": [39, 9, 16, 78, 35, 66, 122, 89, 110, nan, nan, 44, 28, 65, nan, 22, 59, 23, 31,
               44, 21, 9, nan, 45, 168, 73, nan, 76, 118, 84, 85],
    "September": [96, 78, 73, 91, 47, 32, 20, 23, 21, 24, 44, 21, 28, 9, 13, 46, 18, 13, 24,
                  16, 13, 23, 36, 7, 14, 30, nan, 14, 18, 20],
}

# Hollander & Wolfe (1973), p. 140 ff.: times to round first base, 22 players x 3 methods.
ROUNDING_TIMES = np.array([
    [5.40, 5.50, 5.55], [5.85, 5.70, 5.75], [5.20, 5.60, 5.50], [5.55, 5.50, 5.40],
    [5.90, 5.85, 5.70], [5.45, 5.55, 5.60], [5.40, 5.40, 5.35], [5.45, 5.50, 5.35],
    [5.25, 5.15, 5.00], [5.85, 5.80, 5.70], [5.25, 5.20, 5.10], [5.65, 5.55, 5.45],
    [5.60, 5.35, 5.45], [5.05, 5.00, 4.95], [5.50, 5.50, 5.40], [5.45, 5.55, 5.50],
    [5.55, 5.55, 5.35], [5.45, 5.50, 5.55], [5.50, 5.45, 5.25], [5.65, 5.60, 5.40],
    [5.70, 5.65, 5.55], [6.30, 6.30, 6.25],
])
ROUNDING_NAMES = ["Round Out", "Narrow Angle", "Wide Angle"]

# Unequal sizes and spreads (made up for cross-checks).
UNEQUAL = {
    "A": [10.1, 12.3, 9.8, 11.4, 10.9, 12.0],
    "B": [13.2, 15.1, 14.8, 12.9, 16.3, 15.5, 14.1, 13.8],
    "C": [9.1, 8.4, 10.9, 9.9],
    "D": [11.0, 18.5, 14.2, 20.3, 9.7, 16.6, 12.3],
}
TIED = {
    "A": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
    "B": [2.0, 4.0, 5.0, 5.0, 6.0],
    "C": [4.0, 6.0, 7.0, 7.0, 7.0, 8.0, 9.0],
}


def _arrays(groups):
    return [np.asarray(v, float) for v in groups.values()]


def _long(groups):
    return pd.DataFrame({
        "y": np.concatenate(_arrays(groups)),
        "g": np.concatenate([[name] * len(v) for name, v in groups.items()]),
    })


# ============================================================ published examples


def test_anova_published_plantgrowth():
    """Dobson (1983) plant weights. R's documented example
    ``anova(lm(weight ~ group, data = PlantGrowth))`` prints
    group: Df 2, Sum Sq 3.7663, Mean Sq 1.8832, F 4.8461, Pr(>F) 0.01591;
    Residuals: Df 27, Sum Sq 10.4921, Mean Sq 0.3886."""
    res = ow.anova_oneway(PLANT)
    between, within, total = res.anova_table
    assert (res.df, res.df2) == (2, 27)
    assert round(between["ss"], 4) == 3.7663 and round(between["ms"], 4) == 1.8832
    assert round(within["ss"], 4) == 10.4921 and round(within["ms"], 4) == 0.3886
    assert round(res.statistic, 4) == 4.8461
    assert round(res.p_value, 5) == 0.01591
    assert total["df"] == 29 and total["ss"] == pytest.approx(between["ss"] + within["ss"])
    assert res.significant is True and res.statistic_name == "F"
    assert res.correction == "Tukey HSD" and len(res.comparisons) == 3
    assert res.comparison("trt1", "trt2").significant is True
    assert res.comparison("ctrl", "trt1").significant is False
    assert "trt1 vs trt2" in res.interpretation and "F(2, 27) = 4.846" in res.interpretation


def test_welch_published_plantgrowth():
    """Same data, Welch's test as printed by R's oneway.test(weight ~ group):
    F = 5.181, num df = 2.000, denom df = 17.128, p-value = 0.01739."""
    res = ow.welch_anova(PLANT)
    assert round(res.statistic, 3) == 5.181
    assert res.df == 2 and round(res.df2, 3) == 17.128
    assert round(res.p_value, 5) == 0.01739
    assert res.correction == "Games-Howell" and len(res.comparisons) == 3


def test_kruskal_published_examples():
    """R's documented kruskal.test examples: Hollander & Wolfe's mucociliary data
    (Kruskal-Wallis chi-squared = 0.77143, df = 2, p-value = 0.68) and the New York
    ozone data by month (chi-squared = 29.267, df = 4, p-value = 6.901e-06); plus
    Dobson's plant weights (chi-squared = 7.9882, df = 2, p-value = 0.01842)."""
    res = ow.kruskal_wallis(MUCOCILIARY)
    assert round(res.statistic, 5) == 0.77143 and res.df == 2
    assert round(res.p_value, 2) == 0.68
    assert res.significant is False
    assert any("5 or fewer" in w for w in res.warnings)

    res = ow.kruskal_wallis(OZONE)
    assert res.n == {"May": 26, "June": 9, "July": 26, "August": 26, "September": 29}
    assert round(res.statistic, 3) == 29.267 and res.df == 4
    assert float(f"{res.p_value:.4g}") == 6.901e-06
    assert len(res.data_notes) == 5 and "5 missing" in res.data_notes[0]
    assert len(res.comparisons) == 10 and res.correction == "Dunn's test, Bonferroni"

    res = ow.kruskal_wallis(PLANT)
    assert round(res.statistic, 4) == 7.9882 and round(res.p_value, 5) == 0.01842


def test_friedman_published_rounding_times():
    """Hollander & Wolfe (1973) rounding-first-base data; R's documented friedman.test
    example prints Friedman chi-squared = 11.143, df = 2, p-value = 0.003805."""
    res = ow.friedman_test(ROUNDING_TIMES, names=ROUNDING_NAMES)
    assert round(res.statistic, 3) == 11.143 and res.df == 2
    assert round(res.p_value, 6) == 0.003805
    assert res.statistic_name == "chi2" and res.significant is True
    assert res.n_total == 22 and res.n == dict.fromkeys(ROUNDING_NAMES, 22)
    assert res.extra["rank_sums"] == {"Round Out": 53.0, "Narrow Angle": 47.0, "Wide Angle": 32.0}
    # Kendall's W = chi2 / (n (k - 1))
    assert res.effect("Kendall's W").value == pytest.approx(res.statistic / 44)
    assert res.comparison("Round Out", "Wide Angle").significant is True
    assert res.comparison("Round Out", "Narrow Angle").significant is False


# ====================================================== cross-implementation: ANOVA


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
def test_anova_matches_scipy_and_statsmodels(groups):
    res = ow.anova_oneway(groups)
    arrays = _arrays(groups)
    ref = stats.f_oneway(*arrays)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    table = sm.stats.anova_lm(ols("y ~ C(g)", _long(groups)).fit())
    between, within, total = res.anova_table
    assert between["ss"] == pytest.approx(table.loc["C(g)", "sum_sq"], rel=1e-9)
    assert within["ss"] == pytest.approx(table.loc["Residual", "sum_sq"], rel=1e-9)
    assert between["ms"] == pytest.approx(table.loc["C(g)", "mean_sq"], rel=1e-9)
    assert within["ms"] == pytest.approx(table.loc["Residual", "mean_sq"], rel=1e-9)
    assert between["df"] == table.loc["C(g)", "df"] and within["df"] == table.loc["Residual", "df"]
    assert between["p"] == pytest.approx(table.loc["C(g)", "PR(>F)"], rel=1e-8)
    # textbook formulas
    pooled = np.concatenate(arrays)
    ssb = sum(a.size * (a.mean() - pooled.mean()) ** 2 for a in arrays)
    sst = ((pooled - pooled.mean()) ** 2).sum()
    assert between["ss"] == pytest.approx(ssb, rel=1e-10)
    assert total["ss"] == pytest.approx(sst, rel=1e-10)
    eta = res.effect("eta squared")
    omega = res.effect("omega squared")
    assert eta.value == pytest.approx(ssb / sst, rel=1e-10)
    msw = (sst - ssb) / (pooled.size - len(arrays))
    assert omega.value == pytest.approx((ssb - (len(arrays) - 1) * msw) / (sst + msw), rel=1e-9)
    assert eta.ci_low <= eta.value <= eta.ci_high and eta.ci_method.startswith("noncentral F")
    # the noncentral-F interval: observed F sits at the 97.5th / 2.5th percentile
    n_total = pooled.size
    for limit, target in ((eta.ci_low, 0.975), (eta.ci_high, 0.025)):
        if limit > 0:
            ncp = limit * n_total / (1 - limit)
            assert stats.ncf.cdf(res.statistic, res.df, res.df2, ncp) == pytest.approx(target, abs=1e-6)
    assert res.extra["r_squared"] == pytest.approx(ssb / sst, rel=1e-10)
    assert res.extra["root_mse"] == pytest.approx(math.sqrt(msw), rel=1e-10)
    assert res.extra["grand_mean"] == pytest.approx(pooled.mean(), rel=1e-12)
    resid = np.asarray(res.extra["residuals"]["residual"])
    assert resid == pytest.approx(np.concatenate([a - a.mean() for a in arrays]), abs=1e-12)
    assert len(res.extra["residuals"]["group"]) == n_total


def test_anova_descriptives_and_parameters():
    res = ow.anova_oneway(UNEQUAL)
    for row, (name, values) in zip(res.descriptives, UNEQUAL.items()):
        a = np.asarray(values)
        assert row["group"] == name and row["n"] == a.size
        assert row["mean"] == pytest.approx(a.mean()) and row["sd"] == pytest.approx(a.std(ddof=1))
        assert row["sem"] == pytest.approx(stats.sem(a)) and row["median"] == pytest.approx(np.median(a))
        lo, hi = stats.t.interval(0.95, a.size - 1, loc=a.mean(), scale=stats.sem(a))
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((lo, hi))
        par = res.parameter(f"Mean of {name}")
        assert par.value == pytest.approx(a.mean()) and par.se == pytest.approx(stats.sem(a))
    assert res.n == {"A": 6, "B": 8, "C": 4, "D": 7} and res.n_total == 25


def test_anova_assumption_checks_match_scipy():
    res = ow.anova_oneway(UNEQUAL)
    arrays = _arrays(UNEQUAL)
    by_name = {}
    for a in res.assumptions:
        by_name.setdefault(a.name, []).append(a)
    assert by_name["Independent observations"][0].status == "info"
    resid = np.concatenate([a - a.mean() for a in arrays])
    w, p = stats.shapiro(resid)
    check = by_name["Normality of residuals (Shapiro-Wilk)"][0]
    assert check.statistic == pytest.approx(w, rel=1e-6) and check.p_value == pytest.approx(p, rel=1e-4)
    assert check.status == ("fail" if p < 0.05 else "pass")
    per_group = by_name["Normality within group (Shapiro-Wilk)"]
    assert [c.target for c in per_group] == list(UNEQUAL)
    for c, a in zip(per_group, arrays):
        assert c.p_value == pytest.approx(stats.shapiro(a).pvalue, rel=1e-6)
    bf = by_name["Equal variances (Brown-Forsythe)"][0]
    lev = stats.levene(*arrays, center="median")
    assert bf.statistic == pytest.approx(lev.statistic, rel=1e-10)
    assert bf.p_value == pytest.approx(lev.pvalue, rel=1e-8)
    assert bf.status == "fail" and "Welch" in bf.detail  # spreads clearly differ here
    bart = by_name["Equal variances (Bartlett)"][0]
    ref = stats.bartlett(*arrays)
    assert bart.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert bart.p_value == pytest.approx(ref.pvalue, rel=1e-8) and bart.status == "warn"
    assert res.worst_assumption_status() == "fail"


def test_anova_two_groups_equals_t_test():
    two = {"A": UNEQUAL["A"], "B": UNEQUAL["B"]}
    res = ow.anova_oneway(two)
    ref = stats.ttest_ind(two["A"], two["B"])
    assert res.statistic == pytest.approx(ref.statistic ** 2, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert any("Only two groups" in w and "t test" in w for w in res.warnings)
    assert len(res.comparisons) == 1
    assert res.comparisons[0].p_adjusted == pytest.approx(ref.pvalue, rel=1e-8)


# ----------------------------------------------------------- post-hoc selection


def test_anova_posthoc_options():
    base = ow.anova_oneway(PLANT, posthoc="none")
    assert base.comparisons == [] and base.correction is None and base.comparison_family is None
    assert "followed by" not in base.methods_text
    labels = {
        "tukey": "Tukey HSD", "bonferroni": "Bonferroni", "sidak": "Šidák", "holm": "Holm",
        "holm-sidak": "Holm-Šidák", "fdr_bh": "Benjamini-Hochberg FDR",
        "lsd": "Fisher's LSD (unadjusted)", "games-howell": "Games-Howell",
    }
    for key, label in labels.items():
        res = ow.anova_oneway(PLANT, posthoc=key)
        assert res.correction == label and len(res.comparisons) == 3
        assert res.statistic == base.statistic
        assert res.comparison_family == "all 3 pairwise comparisons among 3 groups"
        assert all(c.id == f"{res.id}:c{i}" for i, c in enumerate(res.comparisons, start=1))
        assert "followed by" in res.methods_text
    # pooled-SD comparisons use the ANOVA error term (27 df)
    holm = ow.anova_oneway(PLANT, posthoc="holm")
    assert all(c.df == 27 for c in holm.comparisons)
    msw = holm.anova_table[1]["ms"]
    c = holm.comparison("ctrl", "trt1")
    assert c.se == pytest.approx(math.sqrt(msw * 2 / 10))
    lsd = ow.anova_oneway(PLANT, posthoc="lsd")
    assert "Fisher's LSD (unadjusted)" in lsd.interpretation
    assert "adjusted P" not in lsd.interpretation
    assert "adjusted P = 0.0134" in holm.interpretation
    with pytest.raises(StatsError, match="not available"):
        ow.anova_oneway(PLANT, posthoc="scheffe")
    assert ow.anova_oneway(PLANT, posthoc=None).comparisons == []


def test_anova_dunnett_and_control_handling():
    res = ow.anova_oneway(PLANT, posthoc="dunnett", control="ctrl")
    assert res.correction == "Dunnett" and len(res.comparisons) == 2
    assert all(c.group_b == "ctrl" for c in res.comparisons)
    assert "control group 'ctrl'" in res.comparison_family
    assert not any("first group" in w for w in res.warnings)
    default = ow.anova_oneway(PLANT, posthoc="dunnett")
    assert any("first group ('ctrl')" in w for w in default.warnings)
    assert [c.p_adjusted for c in default.comparisons] == [c.p_adjusted for c in res.comparisons]
    other = ow.anova_oneway(PLANT, posthoc="dunnett", control="trt2")
    assert all(c.group_b == "trt2" for c in other.comparisons)
    # many-to-one with a P value adjustment
    holm = ow.anova_oneway(PLANT, posthoc="holm", control="ctrl")
    assert len(holm.comparisons) == 2 and holm.correction == "Holm"
    # Tukey compares all pairs: the control setting is reported as ignored
    tukey = ow.anova_oneway(PLANT, posthoc="tukey", control="ctrl")
    assert len(tukey.comparisons) == 3 and any("ignored" in w for w in tukey.warnings)
    with pytest.raises(StatsError, match="control group 'placebo'"):
        ow.anova_oneway(PLANT, posthoc="dunnett", control="placebo")


def test_anova_lsd_warns_with_many_groups():
    res = ow.anova_oneway(UNEQUAL, posthoc="lsd")
    assert any("more than" in w and "3 groups" in w for w in res.warnings)
    assert any("not adjusted" in w for w in res.warnings)


# ------------------------------------------------------------------ ANOVA edges


def test_anova_single_value_group_warns():
    groups = {"A": [5.0], "B": [4.0, 6.0, 7.0], "C": [8.0, 9.5, 11.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ow.anova_oneway(groups)
    ref = stats.f_oneway(*_arrays(groups))
    assert res.statistic == pytest.approx(ref.statistic) and res.p_value == pytest.approx(ref.pvalue)
    assert any("single value" in w and "A" in w for w in res.warnings)
    assert res.descriptives[0]["sd"] is None and res.descriptives[0]["ci_low"] is None
    normal_a = [a for a in res.assumptions if a.target == "A"][0]
    assert normal_a.status == "warn"
    bf = [a for a in res.assumptions if a.name == "Equal variances (Brown-Forsythe)"][0]
    assert bf.status == "warn" and bf.p_value is None
    # Games-Howell needs n >= 2: reported as a warning, the ANOVA is still returned
    gh = ow.anova_oneway(groups, posthoc="games-howell")
    assert gh.comparisons == [] and gh.statistic == pytest.approx(ref.statistic)
    assert any("could not be computed" in w for w in gh.warnings)


def test_anova_impossible_input():
    with pytest.raises(StatsError, match="replicates"):
        ow.anova_oneway({"A": [1.0], "B": [2.0], "C": [3.0]})
    with pytest.raises(StatsError, match="at least 2 groups"):
        ow.anova_oneway({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(StatsError, match="no values"):
        ow.anova_oneway({"A": [1.0, 2.0], "B": [], "C": [3.0, 4.0]})
    with pytest.raises(StatsError, match="no values"):
        ow.anova_oneway({"A": [1.0, 2.0], "B": [nan, nan], "C": [3.0, 4.0]})
    with pytest.raises(StatsError, match="alpha"):
        ow.anova_oneway(PLANT, alpha=0)
    with pytest.raises(TypeError):
        ow.anova_oneway([[1.0, 2.0], [3.0, 4.0]])


def test_anova_zero_within_group_variance():
    groups = {"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0], "C": [3.0, 3.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ow.anova_oneway(groups)
    assert res.statistic == math.inf and res.p_value == 0.0 and res.significant is True
    assert any("within-group variability is zero" in w for w in res.warnings)
    assert res.anova_table[1]["ss"] == 0.0
    assert res.effect("eta squared").value == 1.0 and res.effect("eta squared").ci_low is None
    assert all(c.statistic == math.inf and c.p_adjusted == 0.0 for c in res.comparisons)
    assert [a.status for a in res.assumptions if "Normality of residuals" in a.name] == ["warn"]
    json.loads(res.model_dump_json())  # infinities serialise


def test_anova_all_values_identical():
    groups = {"A": [0.1, 0.1, 0.1], "B": [0.1, 0.1, 0.1, 0.1], "C": [0.1, 0.1]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ow.anova_oneway(groups)
    assert res.statistic is None and res.p_value is None and res.significant is None
    assert any("All values are identical" in w for w in res.warnings)
    assert res.effect("eta squared").value is None and res.effect("omega squared").value is None
    assert all(c.p_adjusted is None for c in res.comparisons)
    assert "no test could be carried out" in res.interpretation
    assert res.model_dump_json()


def test_anova_one_constant_group():
    groups = {"A": [2.0, 2.0, 2.0, 2.0], "B": [1.0, 3.0, 4.0], "C": [5.0, 6.5, 7.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ow.anova_oneway(groups)
    assert res.statistic == pytest.approx(stats.f_oneway(*_arrays(groups)).statistic)
    assert any("identical in group(s) A" in w for w in res.warnings)
    bart = [a for a in res.assumptions if a.name == "Equal variances (Bartlett)"][0]
    assert bart.status == "warn" and bart.p_value is None


def test_anova_equal_means_gives_f_zero():
    res = ow.anova_oneway({"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [0.0, 4.0]}, posthoc="none")
    assert res.statistic == 0.0 and res.p_value == 1.0
    assert res.effect("eta squared").value == 0.0 and res.effect("omega squared").value == 0.0


def test_missing_values_are_dropped_and_reported():
    groups = {"A": [1.0, nan, 2.5, 3.0], "B": [4.0, 5.5, float("inf"), 6.0], "C": [7.0, 8.0, 9.5]}
    res = ow.anova_oneway(groups)
    assert res.n == {"A": 3, "B": 3, "C": 3}
    assert any("A: 1 missing" in n for n in res.data_notes)
    assert any("B: 1 infinite" in n for n in res.data_notes)
    clean = {"A": [1.0, 2.5, 3.0], "B": [4.0, 5.5, 6.0], "C": [7.0, 8.0, 9.5]}
    assert res.statistic == pytest.approx(ow.anova_oneway(clean).statistic)


def test_shapiro_skipped_for_very_large_samples():
    rng = np.random.default_rng(0)
    groups = {f"g{i}": rng.normal(i * 0.01, 1, 5200 if i == 0 else 30) for i in range(3)}
    res = ow.anova_oneway(groups, posthoc="none")
    resid = [a for a in res.assumptions if "Normality of residuals" in a.name][0]
    assert resid.status == "warn" and "QQ plot" in resid.detail
    big = [a for a in res.assumptions if a.target == "g0"][0]
    assert big.status == "warn" and "too many" in big.detail


# ====================================================================== Welch


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
def test_welch_matches_statsmodels_and_formula(groups):
    res = ow.welch_anova(groups)
    arrays = _arrays(groups)
    ref = sm_anova_oneway(arrays, use_var="unequal", welch_correction=True)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.df == ref.df_num and res.df2 == pytest.approx(ref.df_denom, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    # Welch (1951) written out
    k = len(arrays)
    n = np.array([a.size for a in arrays])
    m = np.array([a.mean() for a in arrays])
    w = n / np.array([a.var(ddof=1) for a in arrays])
    mw = (w * m).sum() / w.sum()
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    assert res.statistic == pytest.approx(f, rel=1e-10)
    assert res.df2 == pytest.approx((k ** 2 - 1) / (3 * lam), rel=1e-10)
    assert res.p_value == pytest.approx(stats.f.sf(f, k - 1, (k ** 2 - 1) / (3 * lam)), rel=1e-9)
    # descriptive eta squared, no CI
    eta = res.effect("eta squared (descriptive)")
    pooled = np.concatenate(arrays)
    ssb = (n * (m - pooled.mean()) ** 2).sum()
    assert eta.value == pytest.approx(ssb / ((pooled - pooled.mean()) ** 2).sum(), rel=1e-10)
    assert eta.ci_low is None and any("describes this sample only" in x for x in res.warnings)
    assert res.anova_table[0]["f"] == res.statistic and res.anova_table[1]["df"] == res.df2
    variance = [a for a in res.assumptions if a.name == "Equal variances (Brown-Forsythe)"][0]
    assert variance.status == "info" and "Not required" in variance.detail
    assert variance.p_value == pytest.approx(stats.levene(*arrays, center="median").pvalue)


def test_welch_two_groups_equals_welch_t():
    two = {"A": UNEQUAL["A"], "D": UNEQUAL["D"]}
    res = ow.welch_anova(two)
    ref = stats.ttest_ind(two["A"], two["D"], equal_var=False)
    assert res.statistic == pytest.approx(ref.statistic ** 2, rel=1e-10)
    assert res.df2 == pytest.approx(ref.df, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert any("Welch's unpaired t test" in w for w in res.warnings)


def test_welch_posthoc_options_use_separate_variances():
    res = ow.welch_anova(UNEQUAL, posthoc="holm")
    assert res.correction == "Holm" and len(res.comparisons) == 6
    for c in res.comparisons:
        ref = stats.ttest_ind(UNEQUAL[c.group_a], UNEQUAL[c.group_b], equal_var=False)
        assert c.p_value == pytest.approx(ref.pvalue, rel=1e-9) and c.df == pytest.approx(ref.df)
    many = ow.welch_anova(UNEQUAL, posthoc="bonferroni", control="A")
    assert len(many.comparisons) == 3 and all(c.group_b == "A" for c in many.comparisons)
    ignored = ow.welch_anova(UNEQUAL, control="A")
    assert len(ignored.comparisons) == 6 and any("ignored" in w for w in ignored.warnings)
    assert ow.welch_anova(UNEQUAL, posthoc="none").comparisons == []
    lsd = ow.welch_anova(UNEQUAL, posthoc="lsd")
    assert lsd.correction == "Welch t tests (unadjusted)"
    assert any("LSD-type" in w for w in lsd.warnings)
    for bad in ("tukey", "dunnett"):
        with pytest.raises(StatsError, match="not available"):
            ow.welch_anova(UNEQUAL, posthoc=bad)


def test_welch_and_brown_forsythe_need_two_values_per_group():
    groups = {"A": [5.0], "B": [4.0, 6.0, 7.0], "C": [8.0, 9.5, 11.0]}
    for func in (ow.welch_anova, ow.brown_forsythe_anova):
        with pytest.raises(StatsError, match="at least 2 in every group"):
            func(groups)
        with pytest.raises(StatsError, match="at least 2 groups"):
            func({"A": [1.0, 2.0]})


def test_welch_with_constant_group_is_undefined():
    groups = {"A": [2.0, 2.0, 2.0, 2.0], "B": [1.0, 3.0, 4.0], "C": [5.0, 6.5, 7.0]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ow.welch_anova(groups)
    assert res.statistic is None and res.p_value is None and res.df2 is None
    assert any("zero variance" in w for w in res.warnings)
    assert "could not be computed" in res.interpretation
    assert len(res.comparisons) == 3 and all(c.p_adjusted is not None for c in res.comparisons)
    assert res.model_dump_json()


# ============================================================== Brown-Forsythe


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED])
def test_brown_forsythe_matches_statsmodels_and_formula(groups):
    res = ow.brown_forsythe_anova(groups)
    arrays = _arrays(groups)
    ref = sm_anova_oneway(arrays, use_var="bf")
    # statsmodels reports Mehrotra's corrected numerator df as its main result and the
    # original Brown-Forsythe test (df1 = k - 1) as df2 / pvalue2
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.df == ref.df_num2 == len(arrays) - 1
    assert res.df2 == pytest.approx(ref.df_denom, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue2, rel=1e-8)
    assert res.extra["mehrotra"]["df1"] == pytest.approx(ref.df_num, rel=1e-10)
    assert res.extra["mehrotra"]["p"] == pytest.approx(ref.pvalue, rel=1e-8)
    # Brown & Forsythe (1974) written out
    n = np.array([a.size for a in arrays])
    v = np.array([a.var(ddof=1) for a in arrays])
    m = np.array([a.mean() for a in arrays])
    pooled = np.concatenate(arrays)
    terms = (1 - n / n.sum()) * v
    f = (n * (m - pooled.mean()) ** 2).sum() / terms.sum()
    c = terms / terms.sum()
    df2 = 1 / (c ** 2 / (n - 1)).sum()
    assert res.statistic == pytest.approx(f, rel=1e-10) and res.df2 == pytest.approx(df2, rel=1e-10)
    assert res.p_value == pytest.approx(stats.f.sf(f, len(arrays) - 1, df2), rel=1e-9)
    assert res.statistic_name == "F*" and res.correction == "Games-Howell"
    assert res.interpretation.startswith("A Brown-Forsythe ANOVA compared")


def test_brown_forsythe_equals_classic_f_for_balanced_data():
    """With equal group sizes F* is identical to the classic F (only df2 differs)."""
    classic = ow.anova_oneway(PLANT, posthoc="none")
    res = ow.brown_forsythe_anova(PLANT, posthoc="none")
    assert res.statistic == pytest.approx(classic.statistic, rel=1e-12)
    assert res.df2 < classic.df2


def test_brown_forsythe_degenerate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        const = ow.brown_forsythe_anova({"A": [2.0, 2.0, 2.0], "B": [1.0, 3.0, 4.0], "C": [5.0, 6.5, 7.0]})
        zero = ow.brown_forsythe_anova({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]})
        same = ow.brown_forsythe_anova({"A": [1.0, 1.0], "B": [1.0, 1.0], "C": [1.0, 1.0]})
    assert math.isfinite(const.statistic) and 0 <= const.p_value <= 1
    assert any("contribute no variance" in w for w in const.warnings)
    assert zero.statistic == math.inf and zero.p_value == 0.0 and "mehrotra" not in zero.extra
    assert same.statistic is None and same.p_value is None
    assert any("All values are identical" in w for w in same.warnings)


# ============================================================== Kruskal-Wallis


@pytest.mark.parametrize("groups", [PLANT, UNEQUAL, TIED, MUCOCILIARY])
def test_kruskal_matches_scipy_and_formula(groups):
    res = ow.kruskal_wallis(groups)
    arrays = _arrays(groups)
    ref = stats.kruskal(*arrays)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
    # textbook form: 12 / (N (N + 1)) sum R_i^2 / n_i - 3 (N + 1), divided by C
    pooled = np.concatenate(arrays)
    ranks = stats.rankdata(pooled)
    n_total = pooled.size
    start, total = 0, 0.0
    for a, row in zip(arrays, res.descriptives):
        r = ranks[start:start + a.size]
        total += r.sum() ** 2 / a.size
        assert row["rank_sum"] == pytest.approx(r.sum()) and row["mean_rank"] == pytest.approx(r.mean())
        assert res.extra["mean_ranks"][row["group"]] == pytest.approx(r.mean())
        start += a.size
    _, counts = np.unique(pooled, return_counts=True)
    c = 1 - (counts ** 3 - counts).sum() / (n_total ** 3 - n_total)
    h = (12 / (n_total * (n_total + 1)) * total - 3 * (n_total + 1)) / c
    assert res.statistic == pytest.approx(h, rel=1e-10)
    assert res.extra["tie_correction"] == pytest.approx(c)
    eps = res.effect("epsilon squared (rank)")
    assert eps.value == pytest.approx(h / (n_total - 1), rel=1e-10)
    assert res.parameter(f"Median of {res.descriptives[0]['group']}").value == pytest.approx(
        np.median(arrays[0]))
    assert res.df == len(arrays) - 1 and res.df2 is None and res.statistic_name == "H"


def test_kruskal_options_and_edges():
    res = ow.kruskal_wallis(TIED, posthoc="none")
    assert res.comparisons == [] and "followed by" not in res.methods_text
    tie_note = [a for a in res.assumptions if a.name == "Ordinal or continuous measurements"][0]
    assert "tied" in tie_note.detail
    holm = ow.kruskal_wallis(TIED, adjust="holm", control="A")
    assert holm.correction == "Dunn's test, Holm" and len(holm.comparisons) == 2
    assert "Holm adjustment" in holm.methods_text
    raw = ow.kruskal_wallis(TIED, adjust="none")
    assert raw.correction == "Dunn's test (unadjusted)" and "without adjustment" in raw.methods_text
    assert "adjusted P" not in raw.interpretation
    with pytest.raises(StatsError, match="control group"):
        ow.kruskal_wallis(TIED, control="Z")
    with pytest.raises(StatsError, match="not available"):
        ow.kruskal_wallis(TIED, posthoc="tukey")
    with pytest.raises(StatsError, match="adjustment"):
        ow.kruskal_wallis(TIED, adjust="scheffe")
    with pytest.raises(StatsError, match="at least 3 values"):
        ow.kruskal_wallis({"A": [1.0], "B": [2.0]})
    # two groups: equivalent to Mann-Whitney without continuity correction
    two = ow.kruskal_wallis({"A": TIED["A"], "C": TIED["C"]})
    mwu = stats.mannwhitneyu(TIED["A"], TIED["C"], use_continuity=False, method="asymptotic")
    assert two.p_value == pytest.approx(mwu.pvalue, rel=1e-9)
    assert any("Mann-Whitney" in w for w in two.warnings)


def test_kruskal_degenerate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same = ow.kruskal_wallis({"A": [3.0, 3.0], "B": [3.0, 3.0, 3.0], "C": [3.0]})
        singles = ow.kruskal_wallis({"A": [1.0], "B": [2.0], "C": [3.0]})
    assert same.statistic is None and same.p_value is None
    assert any("identical" in w for w in same.warnings)
    assert same.effect("epsilon squared (rank)").value is None
    assert all(c.p_adjusted is None for c in same.comparisons)
    assert singles.statistic == pytest.approx(2.0) and 0 <= singles.p_value <= 1
    shape = [a for a in singles.assumptions if a.name.startswith("Similar distribution")][0]
    assert shape.status == "warn" and shape.p_value is None


def test_kruskal_spread_check():
    rng = np.random.default_rng(3)
    groups = {"narrow": rng.normal(0, 0.2, 25), "wide": rng.normal(0, 5, 25),
              "medium": rng.normal(0, 1, 25)}
    res = ow.kruskal_wallis(groups)
    shape = [a for a in res.assumptions if a.name.startswith("Similar distribution")][0]
    assert shape.status == "warn" and "differ in spread" in shape.detail
    ok = ow.kruskal_wallis(PLANT)
    assert [a for a in ok.assumptions if a.name.startswith("Similar")][0].status == "pass"


# ===================================================================== Friedman


def test_friedman_matches_scipy_with_and_without_ties():
    rng = np.random.default_rng(8)
    for data in (rng.normal(size=(12, 4)), np.round(rng.normal(size=(9, 5)), 0), ROUNDING_TIMES):
        res = ow.friedman_test(data)
        ref = stats.friedmanchisquare(*data.T)
        assert res.statistic == pytest.approx(ref.statistic, rel=1e-10)
        assert res.p_value == pytest.approx(ref.pvalue, rel=1e-8)
        n, k = data.shape
        assert res.effect("Kendall's W").value == pytest.approx(ref.statistic / (n * (k - 1)))
        # Iman & Davenport (1980): F = (n - 1) Q / (n (k - 1) - Q)
        iman = res.extra["iman_davenport"]
        f = (n - 1) * ref.statistic / (n * (k - 1) - ref.statistic)
        assert iman["f"] == pytest.approx(f, rel=1e-9)
        assert iman["p"] == pytest.approx(stats.f.sf(f, k - 1, (k - 1) * (n - 1)), rel=1e-8)
        assert [d["group"] for d in res.descriptives] == [f"Condition {j + 1}" for j in range(k)]
        ranks = np.array([stats.rankdata(row) for row in data])
        assert [d["mean_rank"] for d in res.descriptives] == pytest.approx(ranks.mean(axis=0))


def test_friedman_no_ties_textbook_formula():
    data = np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 4.5, 3.5], [3.0, 1.0, 2.0, 4.0],
                     [2.0, 1.0, 4.0, 3.0], [1.0, 3.0, 2.0, 4.0]])
    n, k = data.shape
    r = np.array([stats.rankdata(row) for row in data]).sum(axis=0)
    q = 12 / (n * k * (k + 1)) * (r ** 2).sum() - 3 * n * (k + 1)
    res = ow.friedman_test(data, posthoc="none")
    assert res.statistic == pytest.approx(q, rel=1e-12)
    assert any("chi-square approximation" in w for w in res.warnings)  # n = 5 is small


def test_friedman_mapping_input_and_missing_rows():
    as_dict = {name: ROUNDING_TIMES[:, j] for j, name in enumerate(ROUNDING_NAMES)}
    res = ow.friedman_test(as_dict)
    assert round(res.statistic, 3) == 11.143 and list(res.n) == ROUNDING_NAMES
    holes = ROUNDING_TIMES.copy()
    holes[3, 1] = np.nan
    holes[10, 2] = np.inf
    blank = np.vstack([holes, [np.nan, np.nan, np.nan]])
    dropped = ow.friedman_test(blank, names=ROUNDING_NAMES)
    keep = np.delete(ROUNDING_TIMES, [3, 10], axis=0)
    assert dropped.statistic == pytest.approx(stats.friedmanchisquare(*keep.T).statistic)
    assert dropped.n_total == 20
    assert any("2 subject(s)" in n and "left out" in n for n in dropped.data_notes)
    assert any("infinite" in n for n in dropped.data_notes)
    blocks = [a for a in dropped.assumptions if a.name == "Complete blocks"][0]
    assert blocks.status == "pass" and "2 incomplete" in blocks.detail


def test_friedman_posthoc_options():
    dunn = ow.friedman_test(ROUNDING_TIMES, names=ROUNDING_NAMES, adjust="holm")
    assert dunn.correction == "Dunn's test, Holm" and len(dunn.comparisons) == 3
    wil = ow.friedman_test(ROUNDING_TIMES, names=ROUNDING_NAMES, posthoc="wilcoxon")
    assert wil.correction == "Wilcoxon signed-rank tests, Bonferroni"
    assert wil.comparisons[0].statistic_name == "W"
    none = ow.friedman_test(ROUNDING_TIMES, names=ROUNDING_NAMES, posthoc="none")
    assert none.comparisons == [] and none.statistic == dunn.statistic
    with pytest.raises(StatsError, match="not available"):
        ow.friedman_test(ROUNDING_TIMES, posthoc="tukey")


def test_friedman_impossible_input():
    with pytest.raises(StatsError, match="at least 3 conditions"):
        ow.friedman_test(ROUNDING_TIMES[:, :2])
    with pytest.raises(StatsError, match="at least 2 subjects"):
        ow.friedman_test(ROUNDING_TIMES[:1])
    with pytest.raises(StatsError, match="at least 2 subjects"):
        ow.friedman_test(np.array([[1.0, 2.0, np.nan], [np.nan, 1.0, 2.0], [3.0, 2.0, 1.0]]))
    with pytest.raises(StatsError, match="different numbers of rows"):
        ow.friedman_test({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0], "c": [3.0, 1.0, 2.0]})
    with pytest.raises(StatsError, match="rows = subjects"):
        ow.friedman_test(np.arange(6.0))
    with pytest.raises(StatsError, match="No data"):
        ow.friedman_test({})
    with pytest.raises(StatsError, match="at least 2 subjects"):
        ow.friedman_test({"a": [], "b": [], "c": []})
    with pytest.raises(ValueError):
        ow.friedman_test(ROUNDING_TIMES, names=["a", "b"])


def test_friedman_degenerate():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flat = ow.friedman_test(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [5.0, 5.0, 5.0]]))
        perfect = ow.friedman_test(np.array([[1.0, 2.0, 3.0]] * 4) + np.arange(4)[:, None])
    assert flat.statistic is None and flat.p_value is None
    assert flat.effect("Kendall's W").value is None and "iman_davenport" not in flat.extra
    assert any("carry no information" in w for w in flat.warnings)
    assert all(c.p_value is not None for c in flat.comparisons)  # z = 0, P = 1
    # complete agreement: W = 1, chi2 = n (k - 1)
    assert perfect.statistic == pytest.approx(8.0)
    assert perfect.effect("Kendall's W").value == pytest.approx(1.0)
    assert perfect.extra["iman_davenport"]["f"] == math.inf
    assert perfect.model_dump_json()


# ===================================================================== registry

REGISTERED = ["anova_oneway", "welch_anova", "brown_forsythe_anova", "kruskal_wallis", "friedman"]


@pytest.fixture(autouse=True)
def _only_this_module_needed(monkeypatch):
    """Importing ``torusplot.stats.oneway`` (top of this file) registers its analyses.
    Mark the registry as loaded so these tests do not depend on every other analysis
    module being importable."""
    monkeypatch.setattr(registry, "_loaded", True)


def test_module_is_listed_and_registers_its_keys():
    assert "oneway" in registry.ANALYSIS_MODULES
    assert set(REGISTERED) <= set(registry.REGISTRY)
    for key in REGISTERED:
        assert registry.REGISTRY[key].category in registry.CATEGORIES


def _table():
    return ColumnTable.from_dict("Plant growth", PLANT)


@pytest.mark.parametrize("key", REGISTERED)
def test_registered_analysis_runs_with_defaults(key):
    info = get_analysis(key)
    assert info.tables == ("column",)
    assert info.category == ("Repeated measures" if key == "friedman"
                             else "Compare three or more groups")
    assert info.description and info.citation
    for spec in info.params:
        assert len(spec.help) > 30, spec.name
        if spec.kind == "choice":
            assert spec.default in spec.choices
    assert {"groups", "posthoc", "alpha"} <= {p.name for p in info.params}
    res = run_analysis(_table(), key)
    assert res.analysis == key and res.table == "Plant growth"
    assert res.interpretation and res.why_chosen and res.methods_text and res.citations
    assert len(res.assumptions) >= 1 and res.effect_sizes
    assert res.significant is not None and 0 <= res.p_value <= 1
    assert res.comparisons and all(c.id.startswith(res.id + ":c") for c in res.comparisons)
    assert all(c.significant is not None for c in res.comparisons)
    assert res.params["alpha"] == 0.05 and res.params["groups"] is None
    payload = json.loads(res.model_dump_json())
    assert payload["analysis"] == key and payload["descriptives"]
    assert all(isinstance(row["n"], int) for row in res.descriptives)


def test_registry_values_match_array_level():
    table = _table()
    assert run_analysis(table, "anova_oneway").statistic == pytest.approx(
        ow.anova_oneway(PLANT).statistic)
    assert run_analysis(table, "welch_anova").df2 == pytest.approx(ow.welch_anova(PLANT).df2)
    assert run_analysis(table, "brown_forsythe_anova").p_value == pytest.approx(
        ow.brown_forsythe_anova(PLANT).p_value)
    assert run_analysis(table, "kruskal_wallis", adjust="holm").correction == "Dunn's test, Holm"
    fried = run_analysis(table, "friedman", posthoc="wilcoxon", adjust="none")
    assert fried.statistic == pytest.approx(
        stats.friedmanchisquare(*[np.array(v) for v in PLANT.values()]).statistic)
    assert fried.correction == "Wilcoxon signed-rank tests (unadjusted)"


def test_registry_group_selection_and_control():
    table = ColumnTable.from_dict("t", UNEQUAL)
    res = run_analysis(table, "anova_oneway", groups=["A", "B", "D"], posthoc="dunnett", control="B")
    assert list(res.n) == ["A", "B", "D"] and len(res.comparisons) == 2
    assert all(c.group_b == "B" for c in res.comparisons)
    assert run_analysis(table, "anova_oneway", groups=[]).n_total == 25
    two = run_analysis(table, "kruskal_wallis", groups=["A", "C"])
    assert any("Only two groups" in w for w in two.warnings)
    with pytest.raises(StatsError, match="Unknown group"):
        run_analysis(table, "anova_oneway", groups=["A", "Z"])
    with pytest.raises(StatsError, match="only once"):
        run_analysis(table, "welch_anova", groups=["A", "A", "B"])
    with pytest.raises(StatsError, match="control group 'C'"):
        run_analysis(table, "anova_oneway", groups=["A", "B", "D"], control="C")
    with pytest.raises(StatsError, match="at least 2 groups"):
        run_analysis(table, "brown_forsythe_anova", groups="A")
    with pytest.raises(StatsError, match="Unknown option"):
        run_analysis(table, "friedman", control="A")
    assert run_analysis(table, "welch_anova", control="").correction == "Games-Howell"


def test_registry_exclusions_and_missing_rows():
    table = _table()
    table.exclude([(0, "ctrl")])
    res = run_analysis(table, "anova_oneway")
    assert res.n["ctrl"] == 9 and any("explicitly excluded" in n for n in res.data_notes)
    fried = run_analysis(table, "friedman")
    assert fried.n_total == 9
    assert any("1 subject(s)" in n and "left out" in n for n in fried.data_notes)
    blocks = [a for a in fried.assumptions if a.name == "Complete blocks"][0]
    assert "1 incomplete row(s) were left out" in blocks.detail
    # unequal column lengths: padding rows are not counted as incomplete subjects
    ragged = ColumnTable.from_dict("r", {"a": [1, 2, 3, 4, 5], "b": [2, 3, 1, 5, 4], "c": [3, 1, 2, 4]})
    out = run_analysis(ragged, "friedman", posthoc="none")
    assert out.n_total == 4 and any("1 subject(s)" in n for n in out.data_notes)
    with pytest.raises(StatsError, match="at least 3 conditions"):
        run_analysis(ragged, "friedman", groups=["a", "b"])


def test_registry_refuses_other_table_types():
    from torusplot.data.tables import XYTable

    xy = XYTable.from_arrays("xy", [1, 2, 3], {"s": [1.0, 2.0, 3.0]})
    with pytest.raises(StatsError, match="not available"):
        run_analysis(xy, "anova_oneway")


# =============================================================== property tests

_values = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False).map(
    lambda v: round(v, 1))  # one decimal: many ties, zeros and constant groups


@st.composite
def _groups(draw, min_groups=2, max_groups=5, min_n=1, max_n=7, with_nan=False):
    k = draw(st.integers(min_groups, max_groups))
    out = {}
    for i in range(k):
        values = draw(st.lists(_values, min_size=min_n, max_size=max_n))
        if with_nan and draw(st.booleans()):
            values = values + [nan]
        out[f"g{i}"] = values
    return out


def _same(a, b, rel=1e-6, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if math.isinf(a) or math.isinf(b):
        return a == b
    return a == pytest.approx(b, rel=rel, abs=tol)


OMNIBUS = {
    "anova": lambda g: ow.anova_oneway(g, posthoc="none"),
    "welch": lambda g: ow.welch_anova(g, posthoc="none"),
    "bf": lambda g: ow.brown_forsythe_anova(g, posthoc="none"),
    "kw": lambda g: ow.kruskal_wallis(g, posthoc="none"),
}


def _run_or_error(func, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no numpy / scipy warning may leak
        try:
            return func(groups)
        except StatsError:
            return None


@settings(max_examples=60, deadline=None)
@given(_groups(with_nan=True), st.sampled_from(sorted(OMNIBUS)), st.randoms(use_true_random=False))
def test_property_omnibus_never_crashes_and_ignores_group_order(groups, which, rnd):
    func = OMNIBUS[which]
    res = _run_or_error(func, groups)
    names = list(groups)
    rnd.shuffle(names)
    shuffled = _run_or_error(func, {name: groups[name] for name in names})
    if res is None:
        assert shuffled is None
        return
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    assert res.statistic is None or res.statistic >= 0
    assert (res.statistic is None) == (res.p_value is None)
    if res.statistic is None or math.isinf(res.statistic):
        assert res.warnings
    assert _same(res.statistic, shuffled.statistic) and _same(res.p_value, shuffled.p_value, tol=1e-12)
    for effect in res.effect_sizes:
        assert effect.value is None or -1e-12 <= effect.value <= 1 + 1e-12
    assert res.interpretation and res.assumptions
    json.loads(res.model_dump_json())


@settings(max_examples=50, deadline=None)
@given(_groups(min_n=2), st.sampled_from(["anova", "welch", "bf"]),
       st.floats(min_value=-100, max_value=100), st.floats(min_value=0.25, max_value=40))
def test_property_f_is_location_and_scale_invariant(groups, which, shift, scale):
    func = OMNIBUS[which]
    res = _run_or_error(func, groups)
    moved = _run_or_error(func, {k: [v * scale + shift for v in vals] for k, vals in groups.items()})
    assert res is not None and moved is not None
    assert _same(res.statistic, moved.statistic) and _same(res.p_value, moved.p_value, tol=1e-12)
    flipped = _run_or_error(func, {k: [-v for v in vals] for k, vals in groups.items()})
    assert _same(res.statistic, flipped.statistic)


@settings(max_examples=40, deadline=None)
@given(_groups(), st.floats(min_value=-100, max_value=100), st.floats(min_value=0.25, max_value=40))
def test_property_kruskal_is_invariant_to_monotone_transformations(groups, shift, scale):
    res = _run_or_error(OMNIBUS["kw"], groups)
    moved = _run_or_error(OMNIBUS["kw"], {
        k: [round(v * 10) * scale + shift for v in vals] for k, vals in groups.items()})
    if res is None:
        assert moved is None
        return
    assert _same(res.statistic, moved.statistic) and _same(res.p_value, moved.p_value, tol=1e-12)


@settings(max_examples=30, deadline=None)
@given(_groups(max_groups=4, max_n=5),
       st.sampled_from([("anova", "tukey"), ("anova", "dunnett"), ("anova", "holm"),
                        ("anova", "games-howell"), ("welch", "games-howell"), ("welch", "holm"),
                        ("bf", "bonferroni"), ("kw", "dunn")]))
def test_property_full_results_with_posthoc(groups, combo):
    which, posthoc = combo
    func = {"anova": ow.anova_oneway, "welch": ow.welch_anova, "bf": ow.brown_forsythe_anova,
            "kw": ow.kruskal_wallis}[which]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = func(groups, posthoc=posthoc)
        except StatsError:
            return
    for c in res.comparisons:
        for p in (c.p_value, c.p_adjusted):
            assert p is None or 0.0 <= p <= 1.0
        if c.p_value is not None and c.p_adjusted is not None:
            assert c.p_adjusted >= c.p_value - 1e-9
        assert c.significant is None or isinstance(c.significant, bool)
    assert res.interpretation and res.methods_text
    json.loads(res.model_dump_json())


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 8), st.integers(3, 5), st.data())
def test_property_friedman(n, k, data):
    rows = data.draw(st.lists(st.lists(st.one_of(_values, st.just(nan)), min_size=k, max_size=k),
                              min_size=n, max_size=n))
    matrix = np.array(rows, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = ow.friedman_test(matrix)
        except StatsError:
            assert np.all(np.isfinite(matrix), axis=1).sum() < 2
            return
        permuted = ow.friedman_test(matrix[:, ::-1][::-1], posthoc="none")
    assert res.p_value is None or 0.0 <= res.p_value <= 1.0
    assert _same(res.statistic, permuted.statistic)  # order of rows / columns is irrelevant
    w = res.effect("Kendall's W").value
    assert w is None or -1e-12 <= w <= 1 + 1e-12
    for c in res.comparisons:
        assert c.p_adjusted is None or c.p_adjusted >= c.p_value - 1e-12
    json.loads(res.model_dump_json())
