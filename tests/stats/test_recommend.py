"""recommend_test: the decision path must be right and explained in plain English."""

import numpy as np
import pandas as pd
import pytest

from torusplot.data import (
    ColumnTable,
    ContingencyTable,
    GroupedTable,
    MultipleVariablesTable,
    XYTable,
)
from torusplot.data.samples import SAMPLES, load_sample
from torusplot.stats._utils import StatsError
from torusplot.stats.recommend import Recommendation, recommend_test

RNG = np.random.default_rng(42)
NORMAL = {name: RNG.normal(mu, 1.0, 12) for name, mu in [("A", 10), ("B", 11), ("C", 12.5)]}
SKEWED = {name: np.exp(RNG.normal(mu, 1.2, 14)) for name, mu in [("A", 0), ("B", 0.8), ("C", 1.5)]}


def col(groups):
    return ColumnTable.from_dict("t", groups)


def test_two_independent_normal_groups_get_unpaired_t():
    rec = recommend_test(col({k: NORMAL[k] for k in "AB"}))
    assert rec.analysis == "ttest_unpaired" and rec.params == {"group_a": "A", "group_b": "B"}
    assert {a.analysis for a in rec.alternatives} == {"ttest_welch", "mann_whitney"}
    assert all(c.status == "pass" for c in rec.checks)
    assert "independent" in rec.reason and "paired" in " ".join(rec.cautions)
    text = rec.explain()
    assert text.startswith("Recommended: Unpaired t test") and "Alternatives:" in text


def test_unequal_variances_get_welch():
    rec = recommend_test(col({"A": RNG.normal(10, 0.3, 15), "B": RNG.normal(11, 4.0, 15)}))
    assert rec.analysis == "ttest_welch"
    assert [c.status for c in rec.checks if c.name.startswith("Equal")] == ["fail"]


def test_skewed_groups_get_mann_whitney():
    rec = recommend_test(col({k: SKEWED[k] for k in "AB"}))
    assert rec.analysis == "mann_whitney"
    assert any(c.status == "fail" for c in rec.checks)


def test_tiny_groups_are_flagged_honestly():
    rec = recommend_test(col({"A": [1.0, 2.0, 30.0], "B": [4.0, 5.0, 6.5]}))
    assert any("smallest group has n = 3" in c for c in rec.cautions)
    rec = recommend_test(col({"A": [1.0, 1.1, 50.0], "B": [2.0, 2.1, 90.0]}))
    if rec.analysis == "mann_whitney":
        assert any("cannot give P < 0.05" in c for c in rec.cautions)
    one = recommend_test(col({"A": [1.0, 2.0], "B": [3.0, 5.0]}))
    assert any(c.status == "warn" for c in one.checks)  # cannot test normality with n = 2


def test_paired_designs():
    before = RNG.normal(100, 10, 12)
    rec = recommend_test(col({"Before": before, "After": before + RNG.normal(5, 2, 12)}), paired=True)
    assert rec.analysis == "ttest_paired" and rec.design["pairs"] == 12
    assert rec.checks[0].target == "paired differences"
    diffs = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 25.0, 0.1, 0.3])
    rec = recommend_test(col({"Before": before[:10], "After": before[:10] + diffs}), paired=True)
    assert rec.analysis == "wilcoxon_signed_rank"
    few = recommend_test(col({"x": [1.0, 2, 3, 4, 50], "y": [1.1, 2.1, 3.3, 4.1, 5]}), paired=True)
    if few.analysis == "wilcoxon_signed_rank":
        assert any("at least 6 pairs" in c for c in few.cautions)
    with pytest.raises(StatsError):
        recommend_test(col({"x": [1.0, np.nan], "y": [np.nan, 2.0]}), paired=True)


def test_one_group():
    assert recommend_test(col({"A": NORMAL["A"]})).analysis == "ttest_one_sample"
    assert recommend_test(col({"A": np.exp(np.linspace(0, 6, 15))})).analysis == "wilcoxon_one_sample"


def test_three_or_more_groups():
    rec = recommend_test(col(NORMAL))
    assert rec.analysis == "anova_oneway" and "Tukey" in rec.label and rec.params == {}
    rec = recommend_test(col(NORMAL), control="A")
    assert rec.params == {"posthoc": "dunnett", "control": "A"} and "Dunnett" in rec.label
    with pytest.raises(StatsError):
        recommend_test(col(NORMAL), control="Zzz")
    assert recommend_test(col(SKEWED)).analysis == "kruskal_wallis"
    hetero = {"A": RNG.normal(10, 0.2, 20), "B": RNG.normal(11, 2.0, 20), "C": RNG.normal(12, 6.0, 20)}
    assert recommend_test(col(hetero)).analysis == "welch_anova"
    sub = recommend_test(col(NORMAL), groups=["A", "C"])
    assert sub.analysis in {"ttest_unpaired", "ttest_welch"} and sub.params["group_b"] == "C"
    with pytest.raises(StatsError):
        recommend_test(col(NORMAL), groups=["A", "Nope"])


def test_repeated_measures_three_conditions():
    base = RNG.normal(50, 8, 10)
    table = col({"T0": base, "T1": base + RNG.normal(3, 1, 10), "T2": base + RNG.normal(6, 1, 10)})
    assert recommend_test(table, paired=True).analysis == "anova_rm_oneway"
    noise = np.exp(RNG.normal(0, 1.8, (10, 3)))
    skew = col({"T0": base + noise[:, 0], "T1": base + noise[:, 1] * 9, "T2": base + noise[:, 2]})
    assert recommend_test(skew, paired=True).analysis in {"friedman", "anova_rm_oneway"}


def test_grouped_table():
    rec = recommend_test(load_sample("cell_uptake"))
    assert rec.analysis == "anova_twoway" and rec.params == {"posthoc": "tukey"}
    assert "interaction" in rec.reason
    assert recommend_test(load_sample("cell_uptake"), paired=True).analysis == "anova_mixed"
    single = GroupedTable.from_cells("g", {"r1": {"a": [1.0], "b": [2.0]}, "r2": {"a": [3.0], "b": [5.0]}})
    assert any("without an interaction" in c for c in recommend_test(single).cautions)
    unbalanced = GroupedTable.from_cells("g", {"r1": {"a": [1.0, 2, 3], "b": [2.0, 3]}, "r2": {"a": [3.0, 4, 6], "b": [5.0, 7, 8]}})
    assert any("unbalanced" in c for c in recommend_test(unbalanced).cautions)


def test_xy_tables():
    rec = recommend_test(load_sample("dose_response"))
    assert rec.analysis == "nonlinear_regression" and rec.params["model"] == "dose_response_4pl"
    x = np.arange(1.0, 11.0)
    lin = XYTable.from_arrays("lin", x, {"y": 2 * x + RNG.normal(0, 0.5, 10)})
    assert recommend_test(lin).analysis == "linear_regression"
    assert recommend_test(lin, goal="dose_response").analysis == "nonlinear_regression"
    logx = XYTable.from_arrays("d", [-9.0, -8, -7, -6, -5, -4], {"y": [100.0, 95, 80, 40, 10, 5]}, x_name="log[Drug] (M)")
    assert recommend_test(logx).params["model"] == "dose_response_4pl_logx"
    with pytest.raises(StatsError):
        recommend_test(XYTable.from_arrays("s", [1.0, 2.0], {"y": [1.0, 2.0]}))


def test_contingency_tables():
    rec = recommend_test(load_sample("contingency"))
    assert rec.analysis == "fisher_exact"
    big = ContingencyTable.from_counts("c", [[30, 20, 10], [10, 20, 30]], ["a", "b"], ["x", "y", "z"])
    assert recommend_test(big).analysis == "chi_square" and not recommend_test(big).cautions
    sparse = ContingencyTable.from_counts("c", [[1, 2, 1], [2, 1, 30]], ["a", "b"], ["x", "y", "z"])
    assert recommend_test(sparse).cautions
    with pytest.raises(StatsError):
        recommend_test(ContingencyTable.from_counts("c", [[0, 0], [0, 0]], ["a", "b"], ["x", "y"]))


def test_multiple_variables_tables():
    surv = recommend_test(load_sample("survival"))
    assert surv.analysis == "survival"
    assert surv.params == {"time": "Time (days)", "event": "Event", "group": "Group"}
    nested = recommend_test(load_sample("cell_uptake_cells"))
    assert nested.analysis == "nested_analysis"
    assert nested.params == {"value": "Uptake", "group": "Treatment", "replicate": "Experiment"}
    assert "pseudoreplication" in " ".join(nested.steps)
    df = pd.DataFrame({"y": RNG.normal(0, 1, 16), "sex": ["F", "M"] * 8, "drug": ["a"] * 8 + ["b"] * 8})
    assert recommend_test(MultipleVariablesTable("m", df)).analysis == "anova_twoway"
    df3 = df.assign(site=["s1", "s1", "s2", "s2"] * 4)
    assert recommend_test(MultipleVariablesTable("m", df3)).analysis == "anova_threeway"
    reg = pd.DataFrame(RNG.normal(0, 1, (20, 4)), columns=["y", "x1", "x2", "x3"])
    rec = recommend_test(MultipleVariablesTable("m", reg))
    assert rec.analysis == "multiple_regression" and rec.params["y"] == "y"
    assert recommend_test(MultipleVariablesTable("m", reg[["y", "x1"]])).analysis == "correlation"
    anc = reg[["y", "x1"]].assign(group=["a", "b"] * 10)
    assert recommend_test(MultipleVariablesTable("m", anc)).analysis == "ancova"
    assert recommend_test(MultipleVariablesTable("m", reg[["y"]])).analysis == "descriptives"
    with pytest.raises(StatsError):
        recommend_test(MultipleVariablesTable("m", pd.DataFrame({"g": ["a", "b"]})))
    with pytest.raises(StatsError):
        recommend_test(MultipleVariablesTable("m", reg), goal="survival")


def test_every_sample_gets_a_serialisable_recommendation():
    for name in SAMPLES:
        rec = recommend_test(load_sample(name))
        assert isinstance(rec, Recommendation) and rec.reason and rec.steps
        Recommendation.model_validate_json(rec.model_dump_json())


def test_empty_table_is_a_clear_error():
    with pytest.raises(StatsError, match="no data"):
        recommend_test(col({"A": [np.nan, np.nan]}))
