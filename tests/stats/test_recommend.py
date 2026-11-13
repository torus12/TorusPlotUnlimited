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


# =============================================================================
# Audit part 2: B2-B6 (and the reviewer's follow-ups)
# =============================================================================
from scipy import stats as _stats  # noqa: E402

from torusplot.stats import run_analysis  # noqa: E402
from torusplot.stats.recommend import (  # noqa: E402
    _is_unit_name,
    _mann_whitney_min_p,
    _name_tokens,
    _signed_rank_min_p,
)


def mv(frame):
    return MultipleVariablesTable("m", pd.DataFrame(frame))


def _runs(table, rec):
    """The recommendation must be executable exactly as given."""
    return run_analysis(table, rec.analysis, **rec.params)


def test_unit_columns_are_recognised_by_whole_words_only():
    for name in ("Experiment", "exp", "Exp1", "Mouse_ID", "MouseID", "subjectid", "Animal no.", "Run", "donor",
                 "Patient", "Bio rep", "Replicate #"):
        assert _is_unit_name(name), name
    for name in ("Exposure", "Expression", "Experience", "Runtime", "Running speed", "Acid", "Liquid",
                 "Day", "Holiday", "Treatment", "Repeat count", "Platelets", "Sex"):
        assert not _is_unit_name(name), name
    assert _name_tokens("Mouse_ID") == _name_tokens("MouseID") == {"mouse", "id"}


def test_a_factor_called_exposure_is_not_a_replicate_column():
    table = mv({"Exposure": np.tile(["smoker", "non-smoker"], 12), "Sex": np.repeat(["m", "f"], 12),
                "Value": RNG.normal(size=24)})
    rec = recommend_test(table)
    assert rec.analysis == "anova_twoway"
    assert rec.params == {"value": "Value", "factor_a": "Exposure", "factor_b": "Sex"}
    assert any(a.analysis == "nested_analysis" for a in rec.alternatives)
    assert _runs(table, rec).test_name == "Two-way ANOVA"


def test_a_replicate_name_is_only_a_hint_the_structure_decides():
    """'Experiment' with several cells per treatment and experiment: technical replicates."""
    rows = [(t, e, RNG.normal()) for t in ("ctl", "drug") for e in ("e1", "e2", "e3") for _ in range(6)]
    table = mv(pd.DataFrame(rows, columns=["Treatment", "Experiment", "Signal"]))
    rec = recommend_test(table)
    assert rec.analysis == "nested_analysis"
    assert rec.params == {"value": "Signal", "group": "Treatment", "replicate": "Experiment"}
    assert rec.design["unit_design"] == "nested" and "several rows" in " ".join(rec.steps)
    assert _runs(table, rec).n == {"ctl": 3, "drug": 3}
    # animals nested in treatment (different labels per group) with several cells each
    rows = [(t, f"{t}{i}", RNG.normal()) for t in ("ctl", "drug") for i in range(4) for _ in range(5)]
    table = mv(pd.DataFrame(rows, columns=["Treatment", "Animal", "Size"]))
    rec = recommend_test(table)
    assert rec.analysis == "nested_analysis" and rec.params["replicate"] == "Animal"
    assert _runs(table, rec).test_name.startswith("Welch")
    # one row per animal: the column is only a label, and never becomes a factor
    table = mv({"AnimalID": [f"a{i}" for i in range(24)], "Diet": np.tile(["hf", "chow"], 12),
                "Sex": np.repeat(["m", "f"], 12), "Glucose": RNG.normal(size=24)})
    rec = recommend_test(table)
    assert rec.analysis == "anova_twoway" and "AnimalID" not in rec.params.values()
    assert rec.design["id_column"] == "AnimalID"
    _runs(table, rec)
    # a numeric ID column is neither the outcome nor a predictor
    table = mv({"ID": np.arange(20), "y": RNG.normal(size=20), "x1": RNG.normal(size=20),
                "x2": RNG.normal(size=20), "x3": RNG.normal(size=20)})
    rec = recommend_test(table)
    assert rec.analysis == "multiple_regression" and rec.params == {"y": "y", "predictors": ["x1", "x2", "x3"]}


def test_long_repeated_measures_table_gets_a_mixed_anova_not_a_threeway_anova():
    table = mv({"Subject": np.repeat([f"S{i}" for i in range(8)], 3), "Treatment": np.repeat(["ctl", "drug"], 12),
                "Time": np.tile(["t0", "t1", "t2"], 8), "Value": RNG.normal(size=24)})
    rec = recommend_test(table)
    assert rec.analysis == "anova_mixed"
    assert rec.params == {"value": "Value", "subject": "Subject", "between": "Treatment", "within": "Time"}
    assert "must not be used as an ordinary factor" in rec.reason
    res = _runs(table, rec)
    assert res.extra["n_subjects"] == 8 and res.extra["between"] == "Treatment" and res.extra["within"] == "Time"
    # numeric subject IDs and numeric time points, plus a per-subject attribute that is not the outcome
    table = mv({"Mouse ID": np.repeat(np.arange(1, 9), 3), "Age": np.repeat([8.0, 9, 10, 11, 8, 9, 10, 11], 3),
                "Genotype": np.repeat(["wt", "ko"], 12), "Day": np.tile([0, 7, 14], 8),
                "Weight": RNG.normal(20, 2, 24)})
    rec = recommend_test(table)
    assert rec.analysis == "anova_mixed"
    assert rec.params == {"value": "Weight", "subject": "Mouse ID", "between": "Genotype", "within": "Day"}
    assert _runs(table, rec).extra["n_subjects"] == 8
    # no telling name: the split-plot layout itself identifies the subject column
    table = mv({"Clone": np.repeat([f"c{i}" for i in range(8)], 3), "Line": np.repeat(["wt", "ko"], 12),
                "Stage": np.tile(["a", "b", "c"], 8), "Expr": RNG.normal(size=24)})
    rec = recommend_test(table)
    assert rec.analysis == "anova_mixed" and rec.params["subject"] == "Clone"
    assert "layout of the data" in " ".join(rec.steps)
    _runs(table, rec)


def test_fully_within_designs_get_repeated_measures_analyses():
    rows = [(s, a, b, RNG.normal()) for s in range(6) for a in "xy" for b in "123"]
    table = mv(pd.DataFrame(rows, columns=["participant", "drug", "session", "score"]))
    rec = recommend_test(table)
    assert rec.analysis == "anova_rm_twoway"
    assert rec.params == {"value": "score", "subject": "participant", "within_a": "drug", "within_b": "session"}
    assert _runs(table, rec).extra["n_subjects"] == 6
    for visits, expected in ((["v1", "v2", "v3"], "Randomized-block"), (["pre", "post"], "Paired t test")):
        table = mv({"Patient": np.repeat([f"P{i}" for i in range(7)], len(visits)),
                    "Visit": np.tile(visits, 7), "BP": RNG.normal(120, 10, 7 * len(visits))})
        rec = recommend_test(table)
        assert rec.analysis == "nested_analysis"
        assert rec.params == {"value": "BP", "group": "Visit", "replicate": "Patient", "paired": "yes"}
        assert _runs(table, rec).test_name.startswith(expected)
    # two repeated factors plus a grouping factor: nothing fits, and the user is told not to use a factorial ANOVA
    rows = [(f"s{s}", "g1" if s < 3 else "g2", a, b, RNG.normal()) for s in range(6) for a in "xy" for b in "123"]
    table = mv(pd.DataFrame(rows, columns=["subject", "group", "drug", "session", "score"]))
    rec = recommend_test(table)
    assert rec.analysis == "descriptives" and "three-way ANOVA" in " ".join(rec.cautions)
    _runs(table, rec)


def test_genuine_factorial_tables_are_still_factorial():
    df = pd.DataFrame({"y": RNG.normal(0, 1, 16), "sex": ["F", "M"] * 8, "drug": ["a"] * 8 + ["b"] * 8,
                       "site": ["s1", "s1", "s2", "s2"] * 4})
    rec = recommend_test(mv(df))
    assert rec.analysis == "anova_threeway" and "different subject" in " ".join(rec.cautions)
    _runs(mv(df), rec)
    with pytest.raises(StatsError, match="replicate column"):
        recommend_test(mv(df), goal="nested")
    # an explicit goal still works with the looser name matching ('Day' as the replicate)
    rows = [(t, d, RNG.normal()) for t in ("ctl", "drug") for d in ("d1", "d2", "d3") for _ in range(5)]
    day = mv(pd.DataFrame(rows, columns=["Treatment", "Day", "Size"]))
    assert recommend_test(day).analysis == "anova_twoway"
    forced = recommend_test(day, goal="nested")
    assert forced.analysis == "nested_analysis" and forced.params["replicate"] == "Day"
    _runs(day, forced)


def test_sparse_r_by_c_table_gets_the_exact_test():
    sparse = ContingencyTable.from_counts("c", [[3, 1, 4], [1, 5, 2], [0, 2, 6]], ["r1", "r2", "r3"], ["c1", "c2", "c3"])
    rec = recommend_test(sparse)
    assert rec.analysis == "fisher_exact" and "Freeman-Halton" in rec.label
    assert rec.checks[0].status == "fail" and rec.cautions
    assert [a.analysis for a in rec.alternatives] == ["chi_square"]
    assert 0 < _runs(sparse, rec).p_value <= 1
    big = ContingencyTable.from_counts("c", [[30, 20, 10], [10, 20, 30]], ["a", "b"], ["x", "y", "z"])
    rec = recommend_test(big)
    assert rec.analysis == "chi_square"
    assert [a.analysis for a in rec.alternatives] == ["fisher_exact", "chi_square_trend"]
    # 2 x k sparse table: exact test first, the trend test is mentioned for ordered categories
    two_by_k = ContingencyTable.from_counts("c", [[1, 2, 1], [2, 1, 30]], ["a", "b"], ["x", "y", "z"])
    rec = recommend_test(two_by_k)
    assert rec.analysis == "fisher_exact" and "chi_square_trend" in [a.analysis for a in rec.alternatives]


def test_log_dose_explanation_states_the_real_span():
    logx = np.linspace(-9, -4, 6)
    y = np.column_stack([100 / (1 + 10 ** (-6 - logx)) + RNG.normal(0, 3, 6) for _ in range(3)])
    rec = recommend_test(XYTable.from_arrays("xy", logx, {"Y": y}, x_name="log[Agonist]"))
    assert rec.params["model"] == "dose_response_4pl_logx"
    assert rec.design["x_spans_decades"] == pytest.approx(5.0) and rec.design["x_is_log10"] is True
    assert "span about 5 decades" in rec.steps[0] and "already on a log10 scale" in rec.steps[0]
    assert "0 decades" not in " ".join(rec.steps)
    # linear doses: the ratio of the largest to the smallest positive dose
    dose = np.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    rec = recommend_test(XYTable.from_arrays("xy", dose, {"Y": 100 / (1 + 1e-7 / np.maximum(dose, 1e-12))}))
    assert rec.params["model"] == "dose_response_4pl" and rec.design["x_spans_decades"] == pytest.approx(4.0)
    assert "span about 4 decades" in rec.steps[0]
    # asked for explicitly on a narrow X range: no claim about orders of magnitude
    x = np.arange(1.0, 11.0)
    forced = recommend_test(XYTable.from_arrays("lin", x, {"y": 2 * x + 1}), goal="dose_response")
    assert "several orders of magnitude" not in forced.reason and "You asked" in forced.reason


@pytest.mark.parametrize("n1, n2", [(2, 2), (3, 3), (3, 4), (2, 5), (2, 20), (4, 4), (3, 5), (1, 30), (5, 5)])
def test_minimum_exact_p_of_the_rank_tests(n1, n2):
    """2 / C(n1 + n2, n1) is the exact two-sided P of completely separated groups (scipy)."""
    ref = _stats.mannwhitneyu(np.arange(n1), np.arange(n2) + 100.0, method="exact").pvalue
    assert _mann_whitney_min_p(n1, n2) == pytest.approx(ref, rel=1e-12)
    n = n1 + n2
    paired = _stats.wilcoxon(np.arange(1.0, n + 1), method="exact").pvalue
    assert _signed_rank_min_p(n) == pytest.approx(paired, rel=1e-12)


def test_rank_test_cautions_follow_the_minimum_attainable_p():
    def caution(a, b):
        rec = recommend_test(col({"A": a, "B": b}))
        assert rec.analysis == "mann_whitney"
        return [c for c in rec.cautions if "cannot give P < 0.05" in c]

    three_four = caution([1.0, 1.1, 30.0], [2.0, 2.1, 2.2, 90.0])
    assert three_four and "n = 3 and 4" in three_four[0] and "0.0571" in three_four[0]
    three_three = caution([1.0, 1.1, 50.0], [2.0, 2.1, 90.0])
    assert three_three and "n = 3 per group" in three_three[0] and "0.1" in three_three[0]
    assert caution([1.0, 1.1, 50.0], [2.0, 2.1, 2.2, 2.3, 90.0]) == []       # 3 vs 5: 2 / 56 = 0.036
    assert caution([1.0, 50.0], list(np.exp(np.linspace(0, 5, 20)))) == []     # 2 vs 20: 2 / 231
    # the Mann-Whitney alternative of a t test recommendation carries the same warning
    rec = recommend_test(col({"A": [4.1, 5.0, 5.9], "B": [5.2, 6.1, 6.6, 7.4]}))
    assert rec.analysis == "ttest_unpaired"
    alt = next(a for a in rec.alternatives if a.analysis == "mann_whitney")
    assert "cannot give P < 0.05" in alt.when and "0.0571" in alt.when
    # Wilcoxon signed-rank: 2 / 2^n, so five pairs (0.0625) are flagged and six (0.031) are not
    five = recommend_test(col({"x": [1.0, 2, 3, 4, 50], "y": [1.1, 2.1, 3.3, 4.1, 5]}), paired=True)
    assert five.analysis == "wilcoxon_signed_rank"
    assert any("at least 6 pairs" in c and "0.0625" in c for c in five.cautions)
    six = recommend_test(col({"x": [1.0, 2, 3, 4, 5, 90], "y": [1.1, 2.1, 3.3, 4.1, 5.2, 6]}), paired=True)
    assert not any("cannot give" in c for c in six.cautions)
    one = recommend_test(col({"A": [1.0, 1.1, 1.2, 1.3, 40.0]}))
    assert one.analysis == "wilcoxon_one_sample"
    assert any("cannot give a two-sided P below 0.05" in c for c in one.cautions)


def test_recommendation_checks_do_not_depend_on_the_unit():
    for scale in (1e-13, 1e9):
        base = recommend_test(col({k: NORMAL[k] for k in "AB"}))
        rec = recommend_test(col({k: NORMAL[k] * scale for k in "AB"}))
        assert rec.analysis == base.analysis
        assert [c.p_value for c in rec.checks] == pytest.approx([c.p_value for c in base.checks], rel=1e-9)
