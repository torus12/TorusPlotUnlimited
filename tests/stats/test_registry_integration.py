"""Every registered analysis: discoverable, documented for beginners, runnable, serialisable."""

import json

import numpy as np
import pandas as pd
import pytest

from torusplot.data import ColumnTable, MultipleVariablesTable, table_from_meta
from torusplot.data.samples import load_sample
from torusplot.stats import (
    CATEGORIES,
    ApproximateDataError,
    Comparison,
    StatResult,
    StatsError,
    analyses_for,
    format_p,
    get_analysis,
    load_all,
    p_to_stars,
    run_analysis,
)

REGISTRY = load_all()
RNG = np.random.default_rng(5)


def _mv_factorial() -> MultipleVariablesTable:
    rows = []
    for a in ("ctrl", "drug"):
        for b in ("F", "M"):
            for c in ("young", "old"):
                for _ in range(4):
                    rows.append((a, b, c, RNG.normal(10 + (a == "drug") * 2 + (b == "M"), 1.0), RNG.normal(50, 5)))
    return MultipleVariablesTable("mv", pd.DataFrame(rows, columns=["treatment", "sex", "age", "response", "weight"]))


def _mv_subjects() -> MultipleVariablesTable:
    rows = []
    for g in ("ctrl", "drug"):
        for s in range(6):
            base = RNG.normal(10, 1)
            for t, shift in (("t0", 0.0), ("t1", 1.0), ("t2", 2.0)):
                for cond in ("rest", "active"):
                    rows.append((f"{g}{s}", g, t, cond, base + shift * (1.5 if g == "drug" else 1.0) + (cond == "active") + RNG.normal(0, 0.4)))
    return MultipleVariablesTable("subj", pd.DataFrame(rows, columns=["subject", "group", "time", "condition", "value"]))


SIZE = load_sample("nanoparticle_size")
PAIRED = load_sample("paired")
DOSE = load_sample("dose_response")
UPTAKE = load_sample("cell_uptake")
CELLS = load_sample("cell_uptake_cells")
SURV = load_sample("survival")
CONT = load_sample("contingency")
FACT = _mv_factorial()
SUBJ = _mv_subjects()
TREND = table_from_meta(pd.DataFrame({"Group": ["Yes", "No"], "Low": [5, 30], "Mid": [12, 22], "High": [25, 9]}),
                        {"kind": "contingency", "name": "trend"})

#: analysis key -> (table, params) used to exercise every analysis
CASES = {
    "descriptives": (SIZE, {}),
    "normality": (SIZE, {}),
    "equal_variance": (SIZE, {}),
    "ttest_unpaired": (SIZE, {}),
    "ttest_welch": (SIZE, {"group_a": "Extrusion", "group_b": "Microfluidic"}),
    "ttest_paired": (PAIRED, {}),
    "ttest_one_sample": (PAIRED, {}),
    "mann_whitney": (SIZE, {}),
    "wilcoxon_signed_rank": (PAIRED, {}),
    "wilcoxon_one_sample": (PAIRED, {}),
    "anova_oneway": (SIZE, {}),
    "welch_anova": (SIZE, {}),
    "brown_forsythe_anova": (SIZE, {}),
    "kruskal_wallis": (SIZE, {}),
    "friedman": (SIZE, {}),
    "anova_rm_oneway": (SIZE, {}),
    "outliers_grubbs": (SIZE, {}),
    "outliers_rout": (SIZE, {}),
    "anova_twoway": (UPTAKE, {}),
    "anova_mixed": (UPTAKE, {}),
    "anova_threeway": (FACT, {"value": "response", "factor_a": "treatment", "factor_b": "sex", "factor_c": "age"}),
    "ancova": (FACT, {"value": "response", "factor": "treatment", "covariates": ["weight"]}),
    "anova_rm_twoway": (SUBJ, {"value": "value", "subject": "subject", "within_a": "time", "within_b": "condition"}),
    "correlation": (PAIRED, {}),
    "correlation_matrix": (FACT, {}),
    "linear_regression": (DOSE, {}),
    "multiple_regression": (FACT, {"y": "response", "predictors": ["weight", "treatment"]}),
    "nonlinear_regression": (DOSE, {"model": "dose_response_4pl"}),
    "compare_fits": (DOSE, {}),
    "compare_slopes": (DOSE, {}),
    "compare_curve_parameter": (DOSE, {}),
    "survival": (SURV, {}),
    "chi_square": (CONT, {}),
    "fisher_exact": (CONT, {}),
    "chi_square_trend": (TREND, {}),
    "nested_analysis": (CELLS, {}),
}


def test_every_registered_analysis_has_a_test_case():
    assert sorted(CASES) == sorted(REGISTRY), set(REGISTRY) ^ set(CASES)


def test_required_analyses_are_registered():
    required = {
        "descriptives", "normality", "equal_variance", "ttest_unpaired", "ttest_welch", "ttest_paired",
        "ttest_one_sample", "mann_whitney", "wilcoxon_signed_rank", "anova_oneway", "welch_anova",
        "brown_forsythe_anova", "kruskal_wallis", "anova_twoway", "anova_threeway", "anova_rm_oneway", "anova_mixed",
        "friedman", "ancova", "correlation", "linear_regression", "multiple_regression", "nonlinear_regression",
        "compare_fits", "survival", "chi_square", "fisher_exact", "outliers_grubbs", "outliers_rout", "nested_analysis",
    }
    assert required <= set(REGISTRY)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_beginner_documentation_on_every_analysis_and_option(key):
    info = REGISTRY[key]
    assert info.label and info.category in CATEGORIES and info.tables
    assert len(info.description) >= 30, "each analysis needs a plain-English description"
    assert info.citation, "each analysis cites its primary method reference"
    for spec in info.params:
        assert spec.label, (key, spec.name)
        assert len(spec.help) >= 20, f"{key}.{spec.name} needs a beginner-friendly tooltip"
        if spec.kind == "choice":
            assert spec.choices and spec.default in spec.choices, (key, spec.name)
    for banned in ("graphpad", "prism"):
        text = " ".join([info.label, info.description, *[p.help for p in info.params]]).lower()
        assert banned not in text


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_analysis_runs_and_returns_a_complete_result(key):
    table, params = CASES[key]
    result = run_analysis(table, key, **params)
    assert isinstance(result, StatResult)
    assert result.analysis == key and result.table == table.name
    assert result.id.startswith("sr_") and result.test_name
    assert result.why_chosen, "why_chosen must explain when the test applies"
    assert len(result.interpretation) > 40, "interpretation must be a plain-English reading"
    assert result.methods_text.strip(), "methods sentence required"
    assert result.assumptions, "at least one assumption entry (checked or stated)"
    assert result.citations
    for a in result.assumptions:
        assert a.status in {"pass", "warn", "fail", "info"} and a.name
    for p in [result.p_value, result.p_adjusted, *[c.p_value for c in result.comparisons], *[c.p_adjusted for c in result.comparisons]]:
        assert p is None or 0.0 <= p <= 1.0
    for i, comp in enumerate(result.comparisons, start=1):
        assert comp.id == f"{result.id}:c{i}"
    for text in (result.interpretation, result.why_chosen, result.methods_text):
        assert "graphpad" not in text.lower() and "prism" not in text.lower()
    # JSON round trip (what the .tplot file stores)
    again = StatResult.model_validate_json(result.model_dump_json())
    assert again.id == result.id and len(again.comparisons) == len(result.comparisons)
    json.dumps(result.extra, allow_nan=True)


def test_menu_is_filtered_by_table_type_and_ordered_by_category():
    for kind in ("column", "xy", "grouped", "multiple_variables", "contingency"):
        infos = analyses_for(kind)
        assert infos and all(kind in i.tables for i in infos)
        order = [CATEGORIES.index(i.category) for i in infos]
        assert order == sorted(order)
    keys = {i.key for i in analyses_for("contingency")}
    assert keys == {"chi_square", "fisher_exact", "chi_square_trend"}
    assert "nonlinear_regression" in {i.key for i in analyses_for("xy")}
    assert "nonlinear_regression" not in {i.key for i in analyses_for("column")}
    assert "survival" in {i.key for i in analyses_for("multiple_variables")}
    assert analyses_for("no_such_kind") == []


def test_run_analysis_guards():
    with pytest.raises(StatsError, match="Unknown analysis"):
        run_analysis(SIZE, "nope")
    with pytest.raises(StatsError, match="not available for a Column table"):
        run_analysis(SIZE, "fisher_exact")
    with pytest.raises(StatsError, match="Unknown option"):
        run_analysis(SIZE, "ttest_welch", magic=True)
    with pytest.raises(StatsError):
        get_analysis("")


def test_approximate_tables_are_blocked_unless_explicitly_overridden():
    table = ColumnTable.from_dict("digitized (APPROXIMATE)", {"A": [1.0, 2.0, 3.0, 4.0], "B": [2.0, 3.5, 4.1, 6.0]}, approximate=True)
    with pytest.raises(ApproximateDataError, match="APPROXIMATE"):
        run_analysis(table, "ttest_welch")
    result = run_analysis(table, "ttest_welch", allow_approximate=True)
    assert result.approximate_data and result.warnings[0].startswith("APPROXIMATE DATA")


def test_exclusions_are_reported_in_every_result():
    table = ColumnTable.from_dict("t", {"A": [1.0, 2.0, 3.0, 40.0], "B": [2.0, 3.0, 4.0, 5.0]})
    before = run_analysis(table, "ttest_welch")
    assert before.n == {"A": 4, "B": 4} and not before.data_notes
    table.exclude([(3, "A")])
    after = run_analysis(table, "ttest_welch")
    assert after.n == {"A": 3, "B": 4}
    assert any("explicitly excluded" in note and "A row 4" in note for note in after.data_notes)
    assert table.data["A"].tolist() == [1.0, 2.0, 3.0, 40.0]  # raw data are never altered


def test_outlier_analyses_flag_but_never_remove():
    values = [10.1, 9.8, 10.3, 9.9, 10.0, 10.2, 9.7, 10.4, 25.0]
    table = ColumnTable.from_dict("t", {"A": values, "B": [v + 0.1 for v in values[:-1]] + [10.0]})
    for key in ("outliers_grubbs", "outliers_rout"):
        result = run_analysis(table, key)
        flagged = result.extra["flagged"]
        assert {"group": "A", "row": 8, "value": 25.0} in [{k: f[k] for k in ("group", "row", "value")} for f in flagged], key
        assert table.excluded == [] and len(table.groups()["A"]) == 9
        assert "explicit" in result.interpretation.lower() or "your decision" in result.interpretation.lower() or "flag" in result.interpretation.lower()


def test_result_helpers():
    res = StatResult(
        id="sr_x", analysis="demo", test_name="Demo", p_value=0.03, n={"a": 3, "b": 4},
        comparisons=[Comparison(id="", group_a="a", group_b="b", p_value=0.04, p_adjusted=0.2, label="L1"),
                     Comparison(id="", group_a="a", group_b="c", p_value=0.001)],
    ).finalize()
    assert res.significant is True and res.n_total == 7
    assert [c.id for c in res.comparisons] == ["sr_x:c1", "sr_x:c2"]
    assert res.comparisons[0].significant is False and res.comparisons[1].significant is True
    assert res.comparison("b", "a").id == "sr_x:c1" and res.comparison("b", "a", label="zzz") is None
    assert res.comparison("x", "y") is None
    assert res.comparison_by_id("sr_x:c2").group_b == "c" and res.comparison_by_id("nope") is None
    assert res.effect("Cohen's d") is None and res.parameter("slope") is None
    assert res.worst_assumption_status() == "pass"
    assert format_p(None) == "n/a" and format_p(float("nan")) == "n/a"
    assert format_p(0.00001) == "<0.0001" and format_p(0.04321) == "0.0432" and format_p(0.5, digits=2) == "0.50"
    assert [p_to_stars(p) for p in (0.2, 0.04, 0.004, 0.0004, 0.00004, None)] == ["ns", "*", "**", "***", "****", "n/a"]
    assert p_to_stars(0.00004, style="apa") == "***"
    with pytest.raises(ValueError):
        StatResult(analysis="x", test_name="y", nonsense=1)
