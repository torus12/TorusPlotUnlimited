import json

import pandas as pd
import pytest

from tests.ai import (
    RESULT_ID,
    SECRET_NUMBERS,
    T_CELLS,
    T_CONT,
    T_DOSE,
    T_SIZE,
    T_SURV,
    T_UPTAKE,
    make_result,
    results_map,
    sample_tables,
)
from torusplot.ai import prompts
from torusplot.data import MultipleVariablesTable, XYTable
from torusplot.spec import PLOT_TYPES, FigureSpec, figure_spec_schema
from torusplot.spec.figure import DataBinding, Layer, StatAnnotation, StyleSpec


def test_normalize_tables_accepts_many_shapes():
    tables = sample_tables()
    one = tables[T_SIZE]
    assert prompts.normalize_tables(None) == {}
    assert prompts.normalize_tables(one) == {T_SIZE: one}
    assert prompts.normalize_tables([one, tables[T_DOSE]]) == {T_SIZE: one, T_DOSE: tables[T_DOSE]}
    assert prompts.normalize_tables(tables) == tables

    class Project:
        def __init__(self):
            self.tables = {T_SIZE: one}
            self.results = results_map()

    assert prompts.normalize_tables(Project()) == {T_SIZE: one}
    assert list(prompts.normalize_results(Project())) == [RESULT_ID]
    with pytest.raises(TypeError):
        prompts.normalize_tables(42)


def test_normalize_results_accepts_many_shapes():
    result = make_result()
    assert prompts.normalize_results(None) == {}
    assert prompts.normalize_results([result]) == {RESULT_ID: result}
    assert prompts.normalize_results({RESULT_ID: result}) == {RESULT_ID: result}
    assert prompts.normalize_results({RESULT_ID: result.model_dump()})[RESULT_ID].test_name == "One-way ANOVA"


def test_resolve_table():
    tables = sample_tables()
    assert prompts.resolve_table(T_SIZE, tables) is tables[T_SIZE]
    assert prompts.resolve_table(tables[T_DOSE], tables) is tables[T_DOSE]
    assert prompts.resolve_table("nope", tables) is None and prompts.resolve_table(None, tables) is None


def test_table_summaries_contain_names_and_counts_only():
    tables = sample_tables()
    column = prompts.table_summary(tables[T_SIZE])
    assert column["kind"] == "column" and column["groups"] == ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]
    assert column["n_per_group"]["Extrusion"] == 8
    xy = prompts.table_summary(tables[T_DOSE])
    assert xy["x"] == "Dose (uM)" and xy["series"] == ["Free drug", "Nanoformulation"] and xy["replicates"]["Free drug"] == 3
    grouped = prompts.table_summary(tables[T_UPTAKE])
    assert grouped["row_levels"] == ["1 h", "4 h", "24 h"] and grouped["groups"][0] == "Free dye"
    mv = prompts.table_summary(tables[T_CELLS])
    assert mv["numeric_columns"] == ["Uptake"]
    assert mv["categorical_columns"]["Treatment"] == {"n_levels": 2, "levels": ["Control", "Targeted NP"]}
    cont = prompts.table_summary(tables[T_CONT])
    assert cont["row_levels"] == ["Treated", "Control"] and cont["groups"] == ["Responder", "Non-responder"]

    # no data value of any sample table appears in any summary
    for table in (tables[T_SIZE], tables[T_DOSE], tables[T_UPTAKE], tables[T_CELLS]):
        text = json.dumps(prompts.table_summary(table))
        numeric = table.data.select_dtypes("number")
        values = {repr(float(v)) for v in numeric.to_numpy().ravel() if v == v and abs(v) > 30 and v != int(v)}
        assert values, table.name
        assert not [v for v in values if v in text], table.name


def test_long_category_lists_and_approximate_flag():
    frame = pd.DataFrame({"Subject": [f"S{i}" for i in range(40)], "Value": range(40)})
    summary = prompts.table_summary(MultipleVariablesTable("Many", frame))
    assert summary["categorical_columns"]["Subject"] == {"n_levels": 40}  # IDs are data: not listed
    approx = XYTable.from_arrays("Dig", [1, 2], {"Y": [3, 4]}, approximate=True)
    assert prompts.table_summary(approx)["approximate"] is True


def test_results_summary_has_ids_and_names_but_no_numbers():
    summary = prompts.results_summary(results_map())
    assert summary == [{
        "id": RESULT_ID,
        "test": "One-way ANOVA",
        "table": T_SIZE,
        "comparisons": [
            {"id": f"{RESULT_ID}:c1", "groups": ["Bulk mixing", "Sonication"]},
            {"id": f"{RESULT_ID}:c2", "groups": ["Bulk mixing", "Microfluidic"]},
            {"id": f"{RESULT_ID}:c3", "groups": ["Sonication", "Extrusion"]},
            {"id": f"{RESULT_ID}:c4", "groups": ["Extrusion", "Microfluidic"]},
        ],
    }]
    text = prompts.results_block(results_map())
    assert not [n for n in SECRET_NUMBERS if n in text]
    assert "significant" not in text and "p_value" not in text
    assert prompts.results_block(None).startswith("(none")
    labelled = make_result()
    labelled.comparisons[0].label = "24 h"
    assert prompts.results_summary([labelled])[0]["comparisons"][0]["within"] == "24 h"


def test_default_binding_for_each_kind():
    tables = sample_tables()
    expected = {T_SIZE: "bar", T_DOSE: "xy", T_UPTAKE: "bar", T_CELLS: "bar", T_SURV: "bar", T_CONT: "stacked_bar"}
    for name, plot_type in expected.items():
        binding = prompts.default_binding(prompts.table_summary(tables[name]))
        assert binding["plot_type"] == plot_type and binding["data"]["table"] == name
        FigureSpec.model_validate(binding)
    assert prompts.default_binding(prompts.table_summary(tables[T_CELLS]))["data"] == {
        "table": T_CELLS, "x": "Treatment", "y": "Uptake"}
    numbers_only = MultipleVariablesTable("N", pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    assert prompts.default_binding(prompts.table_summary(numbers_only)) == {
        "plot_type": "histogram", "data": {"table": "N", "y": "a"}}
    words_only = MultipleVariablesTable("W", pd.DataFrame({"a": ["x", "y"]}))
    assert prompts.default_binding(prompts.table_summary(words_only))["plot_type"] == "bar"


def test_field_reference_is_generated_from_the_schema():
    reference = prompts.field_reference()
    schema = figure_spec_schema()
    # every model and every field of the schema is present: nothing is hand-maintained
    for name, model in {"FigureSpec": schema, **schema["$defs"]}.items():
        if name != "FigureSpec" and f"{name}:" not in reference:
            raise AssertionError(f"{name} missing from the field reference")
        for field in model["properties"]:
            assert f"  {field}: " in reference
    for cls in (DataBinding, Layer, StatAnnotation, StyleSpec):
        assert f"{cls.__name__}:" in reference
    for plot_type in PLOT_TYPES:
        assert f'"{plot_type}"' in reference
    assert "plot_type:" in reference and "[REQUIRED]" in reference
    assert 'stat_result_id: string  [REQUIRED]' in reference
    assert "alpha: number (0..1)  [optional]" in reference
    assert "pairs: [[string, string], ...]  [optional]" in reference
    assert "colors: {name: string}" in reference
    assert '"all" | "significant" | [string, ...]' in reference


def test_describe_prompts():
    tables = sample_tables()
    system = prompts.describe_system_prompt()
    for needle in ("RULES", "PLOT TYPES", "STYLE PRESETS", "PALETTES", "FIELD REFERENCE", "Torus Signature",
                   "Okabe-Ito", "viridis", "never compute", "JSON only"):
        assert needle in system
    user = prompts.describe_user_prompt("  box plot  ", tables, results_map(), current_table=T_SIZE)
    assert user.startswith("TASK: describe_figure\nCURRENT_TABLE: " + T_SIZE)
    assert "REQUEST:\n<<<\nbox plot\n>>>" in user
    assert RESULT_ID in user and f"{RESULT_ID}:c2" in user
    assert "CURRENT FIGURE" not in user
    assert not [n for n in SECRET_NUMBERS if n in user + system]

    base = FigureSpec(plot_type="bar", data={"table": T_SIZE}, title="Sizes")
    edit = prompts.describe_user_prompt("blue bars", tables, None, current_table=T_SIZE, base_spec=base)
    assert "CURRENT FIGURE" in edit and '"title":"Sizes"' in edit and "(none" in edit
    assert "CURRENT_TABLE" not in prompts.describe_user_prompt("x", tables)
    assert "(no tables)" in prompts.describe_user_prompt("x", {})


def test_retry_prompt_contains_previous_answer_and_error():
    text = prompts.retry_prompt("ORIGINAL", "  {bad}  ", "data.table: Field required")
    assert text.startswith("ORIGINAL\n")
    assert "<<<\n{bad}\n>>>" in text
    assert "Your previous answer was invalid: data.table: Field required" in text
    assert text.endswith("Return the corrected JSON only.")
    long = prompts.retry_prompt("O", "x" * 10000, "e", max_chars=100)
    assert "(truncated)" in long and len(long) < 400


def test_vision_prompts_ask_for_style_only():
    system, user = prompts.style_system_prompt(), prompts.style_user_prompt()
    assert user.startswith("TASK: extract_style")
    assert "Do NOT read data values" in user and "STYLE only" in system
    assert "Do not read, estimate or report any data values" in system
    for key in ("plot_type", "colors", "font_category", "bar_edge", "error_bars", "points_shown", "legend_position"):
        assert f'"{key}"' in user

    spec = FigureSpec(plot_type="bar", data={"table": T_SIZE, "stat_result_id": "sr_x"},
                      annotations=[{"stat_result_id": "sr_x"}], title="My title")
    critique = prompts.critique_user_prompt(spec, ["style.grid", "layout.height_mm"])
    assert critique.startswith("TASK: critique_figure")
    assert "style.grid, layout.height_mm" in critique
    assert '"index": 0' in critique and "mean_bar" in critique
    # the critique model sees style only: no data binding, no annotations, no result IDs
    assert "sr_x" not in critique and "annotations" not in critique and "My title" not in critique
    assert "Never comment on, read or change data values or statistics" in prompts.critique_system_prompt()
