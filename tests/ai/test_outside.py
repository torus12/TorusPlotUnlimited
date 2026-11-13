""""Outside AI instructions": pasted chatbot output -> FigureSpec, with no provider for JSON."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ai import (
    RESULT_ID,
    SECRET_NUMBERS,
    T_DOSE,
    T_SIZE,
    all_prompt_text,
    results_map,
    sample_tables,
)
from torusplot.ai.describe import SpecCheckError
from torusplot.ai.outside import (
    NO_PROVIDER_MESSAGE,
    TEMPLATE_DOC,
    _nested_model,
    _normalize_plot_type,
    clean_outside_json,
    example_answer,
    import_outside_instructions,
    outside_prompt_template,
)
from torusplot.ai.providers import MockProvider
from torusplot.plot import palette_names, preset_names, render_figure
from torusplot.spec import PLOT_TYPES, FigureSpec, figure_spec_schema
from torusplot.spec.figure import AxisSpec, Layer, StatAnnotation

ROOT = Path(__file__).resolve().parents[2]
STATS_TEXT = "must be computed in TorusPlot and attached to a figure by result ID"


def J(obj) -> str:
    return json.dumps(obj)


class ExplodingProvider(MockProvider):
    """Fails the test if the JSON path ever tries to call a provider."""

    def complete(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the JSON path must not call a provider")


# ====================================================================== JSON path (no provider)
def test_clean_json_needs_no_provider():
    text = J({"plot_type": "box", "data": {"table": T_SIZE}, "style": {"palette": "Okabe-Ito"},
              "layout": {"width_mm": 89, "height_mm": 60}})
    outcome = import_outside_instructions(text, sample_tables())
    assert outcome.ok and outcome.attempts == 0 and outcome.errors == [] and outcome.warnings == []
    assert outcome.spec.plot_type == "box" and outcome.spec.layout.height_mm == 60
    assert json.loads(outcome.raw[0])["plot_type"] == "box"
    # a provider that is passed is not used when the JSON is fine
    assert import_outside_instructions(text, sample_tables(), provider=ExplodingProvider()).ok


def test_fenced_json_inside_prose():
    text = (
        "Great question! Here is a FigureSpec for your data:\n\n```json\n"
        + json.dumps({"plot_type": "violin", "data": {"table": T_SIZE}, "title": "Particle size"}, indent=2)
        + "\n```\n\nPaste it into TorusPlot. Let me know if you want {other} options."
    )
    outcome = import_outside_instructions(text, sample_tables(), current_table=T_SIZE)
    assert outcome.ok and outcome.spec.plot_type == "violin" and outcome.spec.title == "Particle size"
    render_figure(outcome.spec, sample_tables())


def test_unknown_keys_are_dropped_with_one_warning_each():
    text = J({
        "plot_type": "bar",
        "data": {"table": T_SIZE, "sheet": 2},
        "theme": "dark",
        "style": {"preset": "Minimal", "shadow": True},
        "layers": [{"type": "mean_bar", "glow": 1}, {"type": "sem_errorbar"}],
        "axes": {"y": {"label": "Size", "units": "nm"}, "z": {}},
        "legend": {"show": True, "bbox": [0, 1]},
    })
    outcome = import_outside_instructions(text, sample_tables())
    assert outcome.ok
    assert sorted(outcome.warnings) == sorted([
        "Unknown setting 'data.sheet' was ignored.",
        "Unknown setting 'theme' was ignored.",
        "Unknown setting 'style.shadow' was ignored.",
        "Unknown setting 'layers[0].glow' was ignored.",
        "Unknown setting 'axes.y.units' was ignored.",
        "Unknown setting 'axes.z' was ignored.",
        "Unknown setting 'legend.bbox' was ignored.",
    ])
    assert outcome.spec.style.preset == "Minimal" and outcome.spec.axes.y.label == "Size"
    assert [lay.type for lay in outcome.spec.layers] == ["mean_bar", "sem_errorbar"]


def test_aliases():
    text = J({
        "type": "Box plot", "dataset": T_SIZE, "palette": "Okabe-Ito", "preset": "Minimal", "font": "Inter",
        "width": 120, "height": "80 mm", "xlabel": "Method", "y_label": "Diameter (nm)", "font_size": 9,
    })
    outcome = import_outside_instructions(text, sample_tables())
    assert outcome.ok, outcome.errors
    spec = outcome.spec
    assert spec.plot_type == "box" and spec.data.table == T_SIZE
    assert (spec.style.palette, spec.style.preset, spec.style.font_family, spec.style.font_size) == (
        "Okabe-Ito", "Minimal", "Inter", 9)
    assert (spec.layout.width_mm, spec.layout.height_mm) == (120, 80)
    assert (spec.axes.x.label, spec.axes.y.label) == ("Method", "Diameter (nm)")
    for alias in ("'type' was read as 'plot_type'", "'dataset' was read as 'data.table'",
                  "'palette' was read as 'style.palette'", "'font' was read as 'style.font_family'",
                  "'width' was read as 'layout.width_mm' (millimetres)", "'xlabel' was read as 'axes.x.label'"):
        assert any(alias in w for w in outcome.warnings), alias


def test_more_alias_shapes():
    tables = sample_tables()
    wrapped = J({"figure": {"chart": "bar chart", "data": T_SIZE, "style": "Classic Lab"}})
    outcome = import_outside_instructions(wrapped, tables)
    assert outcome.ok and outcome.spec.style.preset == "Classic Lab" and outcome.spec.data.table == T_SIZE
    # explicit fields win over aliases; a bad size is ignored with a note
    text = J({"plot_type": "bar", "kind": "violin", "table": "Other", "data": {"table": T_SIZE},
              "style": {"palette": "Torus"}, "palette": "viridis", "width": "wide", "layout": {"height_mm": 70},
              "height": 99, "xlabel": 5})
    outcome = import_outside_instructions(text, tables)
    assert outcome.ok and outcome.spec.plot_type == "bar" and outcome.spec.data.table == T_SIZE
    assert outcome.spec.style.palette == "Torus" and outcome.spec.layout.height_mm == 70
    assert outcome.spec.layout.width_mm is None and outcome.spec.axes.x.label is None
    assert any("'width' was ignored" in w for w in outcome.warnings)


@pytest.mark.parametrize("value,expected", [("Box plot", "box"), ("bar chart", "bar"), ("Kaplan-Meier", "survival"),
                                            ("XY", "xy"), ("dose-response", "dose_response"), ("line graph", "xy"),
                                            ("heat map", "heatmap"), ("pie", "pie"), (3, 3)])
def test_normalize_plot_type(value, expected):
    assert _normalize_plot_type(value) == expected


def test_missing_or_unknown_table_becomes_the_current_table():
    tables = sample_tables()
    outcome = import_outside_instructions(J({"plot_type": "xy"}), tables, current_table=T_DOSE)
    assert outcome.ok and outcome.spec.data.table == T_DOSE
    assert outcome.warnings == [f"No table was named; the figure uses the current table '{T_DOSE}'."]
    outcome = import_outside_instructions(J({"plot_type": "xy", "data": {"table": "My table"}}), tables,
                                          current_table=T_DOSE)
    assert outcome.ok and outcome.spec.data.table == T_DOSE and "'My table' does not exist" in outcome.warnings[0]
    # no current table and several tables: a clear error instead of a guess
    outcome = import_outside_instructions(J({"plot_type": "xy"}), tables)
    assert not outcome.ok and "data.table is missing" in outcome.errors[0]
    with pytest.raises(SpecCheckError):
        clean_outside_json({"plot_type": "xy"}, tables)


def test_invalid_json_spec_gives_errors_without_provider():
    tables = sample_tables()
    outcome = import_outside_instructions(J({"plot_type": "pie", "data": {"table": T_SIZE}}), tables)
    assert not outcome.ok and outcome.attempts == 0
    assert "not a valid FigureSpec" in outcome.errors[0] and "plot_type" in outcome.errors[0]
    outcome = import_outside_instructions(J({"plot_type": "dose_response", "data": {"table": T_SIZE}}), tables)
    assert not outcome.ok and "cannot be drawn from table" in outcome.errors[0]
    outcome = import_outside_instructions(J({"hello": "world"}), tables, current_table=T_SIZE)
    assert not outcome.ok and "plot_type: Field required" in outcome.errors[0]


def test_empty_inputs():
    assert "Paste the instructions" in import_outside_instructions("  ", sample_tables()).errors[0]
    assert "Add a data table first" in import_outside_instructions("{}", {}).errors[0]


# ====================================================================== statistics smuggling
def test_statistics_cannot_be_pasted_in():
    text = J({
        "plot_type": "bar",
        "data": {"table": T_SIZE},
        "title": "Bulk vs microfluidic: p < 0.0001",
        "p_value": 0.00001,
        "significance": {"Bulk mixing vs Microfluidic": "****"},
        "stats": {"test": "ANOVA", "F": 31.4},
        "annotations": [
            {"type": "bracket", "groups": ["Bulk mixing", "Microfluidic"], "label": "****", "p": 0.00001},
            {"stat_result_id": "sr_invented", "comparisons": "all"},
            "*** between Bulk mixing and Extrusion",
            {"stat_result_id": RESULT_ID, "comparisons": "significant", "p_value": 0.01, "label": "**"},
        ],
        "texts": [{"text": "p = 0.03", "x": 0.5, "y": 0.9}, {"text": "n = 8 per group", "x": 0.1, "y": 0.9}],
        "layers": [{"type": "reference_line", "value": 100, "label": "ns"}],
        "axes": {"y": {"label": "Size (*** vs control)"},
                 "x": {"ticks": [0, 1], "tick_labels": ["A", "B (p<0.05)"]}},
        "legend": {"title": "significance: p<0.05"},
    })
    outcome = import_outside_instructions(text, sample_tables(), results_map())
    assert outcome.ok, outcome.errors
    spec = outcome.spec
    # only the annotation that points at a computed result survives, stripped of its typed numbers
    assert len(spec.annotations) == 1
    assert spec.annotations[0].stat_result_id == RESULT_ID and spec.annotations[0].comparisons == "significant"
    assert spec.title is None and spec.legend.title is None
    assert [t.text for t in spec.texts] == ["n = 8 per group"]
    assert spec.layers[0].label is None
    assert spec.axes.y.label is None and spec.axes.x.tick_labels is None and spec.axes.x.ticks is None
    assert spec.stat_result_ids() == [RESULT_ID]

    stats_warnings = [w for w in outcome.warnings if STATS_TEXT in w]
    text_all = " | ".join(outcome.warnings)
    assert len(stats_warnings) == len(outcome.warnings) == 14
    for needle in ("Annotation 1 was dropped: it has no result ID", "Annotation 2 was dropped: it refers to result "
                   "'sr_invented', which has not been computed", "Annotation 3 was dropped",
                   "'p_value' was dropped", "'significance' was dropped", "'stats' was dropped",
                   "'annotations[0].p_value' was dropped", "'annotations[0].label' was dropped",
                   "The title ('Bulk vs microfluidic: p < 0.0001') was removed", "A text annotation ('p = 0.03')",
                   "The label of layer 0 ('ns')", "The Y axis label", "A tick label of the X axis",
                   "The legend title"):
        assert needle in text_all, needle
    # nothing typed survives anywhere in the final spec
    dumped = spec.model_dump_json()
    for forbidden in ("0.00001", "1e-05", "****", "p <", "p=", "31.4"):
        assert forbidden not in dumped


def test_annotations_are_dropped_when_no_results_exist():
    text = J({"plot_type": "bar", "data": {"table": T_SIZE}, "annotations": {"stat_result_id": RESULT_ID}})
    outcome = import_outside_instructions(text, sample_tables())  # nothing computed
    assert outcome.ok and outcome.spec.annotations == []
    assert "has not been computed" in outcome.warnings[0] and STATS_TEXT in outcome.warnings[0]


def test_sanitiser_still_runs_after_the_clean_up():
    text = J({"plot_type": "bar", "data": {"table": T_SIZE, "groups": ["Extrusion", "Nonexistent"]},
              "annotations": [{"stat_result_id": RESULT_ID, "comparisons": [f"{RESULT_ID}:c2", "sr_x:c9"]}]})
    outcome = import_outside_instructions(text, sample_tables(), results_map())
    assert outcome.ok and outcome.spec.data.groups == ["Extrusion"]
    assert outcome.spec.annotations[0].comparisons == [f"{RESULT_ID}:c2"]
    assert len(outcome.warnings) == 2


# ====================================================================== prose path
def test_prose_without_provider_gives_a_clear_message():
    outcome = import_outside_instructions("Make a bar chart with blue bars and SD error bars.", sample_tables(),
                                          current_table=T_SIZE)
    assert not outcome.ok and outcome.attempts == 0
    assert outcome.errors == [NO_PROVIDER_MESSAGE] and outcome.message == NO_PROVIDER_MESSAGE
    assert "enable an AI provider in Settings" in NO_PROVIDER_MESSAGE and TEMPLATE_DOC in NO_PROVIDER_MESSAGE


def test_prose_with_mock_provider_is_delegated_to_describe():
    mock = MockProvider()
    outcome = import_outside_instructions("Please draw a box plot of all four methods.", sample_tables(),
                                          results_map(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.spec.plot_type == "box" and outcome.attempts == 1
    assert len(mock.calls) == 1 and mock.calls[0]["prompt"].startswith("TASK: describe_figure")
    assert "Please draw a box plot of all four methods." in mock.calls[0]["prompt"]
    assert not [n for n in SECRET_NUMBERS if n in all_prompt_text(mock)]


def test_unrepairable_json_falls_back_to_the_provider_when_there_is_one():
    good = J({"plot_type": "bar", "data": {"table": T_SIZE}})
    mock = MockProvider(responses=[good])
    outcome = import_outside_instructions(J({"plot_type": "pie", "data": {"table": T_SIZE}}), sample_tables(),
                                          current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 1 and len(mock.calls) == 1
    assert "not a valid FigureSpec" in outcome.errors[0]  # the first problem is still reported
    assert len(outcome.raw) == 2


# ====================================================================== helpers
def test_nested_model_discovery():
    fields = FigureSpec.model_fields
    assert _nested_model(fields["layers"].annotation) == (Layer, True)
    assert _nested_model(fields["annotations"].annotation) == (StatAnnotation, True)
    assert _nested_model(fields["axes"].annotation)[1] is False
    assert _nested_model(AxisSpec | None) == (AxisSpec, False)
    assert _nested_model(fields["title"].annotation) == (None, False)
    assert _nested_model(list[str]) == (None, False)


# ====================================================================== prompt template
def test_committed_prompt_template_is_up_to_date():
    doc = ROOT / TEMPLATE_DOC
    assert doc.is_file(), "run: python tools/build_prompt_template.py"
    assert doc.read_text(encoding="utf-8") == outside_prompt_template(), (
        "docs/PROMPT_TEMPLATE_FOR_OUTSIDE_AI.md is out of date: run python tools/build_prompt_template.py"
    )


def test_prompt_template_contents():
    text = outside_prompt_template()
    # the full schema, verbatim, in a fenced block
    start = text.index("## Full JSON schema of a FigureSpec")
    block = text[text.index("```json\n", start) + 8: text.index("\n```", text.index("```json\n", start))]
    assert json.loads(block) == figure_spec_schema()
    # rules, plot types, presets, palettes
    assert "You never compute, estimate, write or change statistics" in text
    assert "Never invent an ID" in text
    for plot_type in PLOT_TYPES:
        assert f"- {plot_type}: " in text
    for preset in preset_names():
        assert f"- {preset}: " in text
    for palette in palette_names():
        assert palette in text
    # the filled example is a valid spec
    example_block = text[text.index("## Example of a valid answer"):start]
    example = json.loads(example_block[example_block.index("{"): example_block.rindex("}") + 1])
    assert example == example_answer() and FigureSpec.model_validate(example).plot_type == "bar"
    # places for the user to fill in
    assert "[DESCRIBE YOUR TABLE HERE" in text and "[DESCRIBE THE FIGURE HERE" in text
    assert "GENERATED FILE" in text and text.endswith("Answer with the JSON object only.\n")
    for trademark in ("Graph" + "Pad", "Pri" + "sm"):
        assert trademark.lower() not in text.lower()


def test_prompt_template_with_tables_lists_names_only():
    tables = sample_tables()
    text = outside_prompt_template({T_SIZE: tables[T_SIZE]})
    assert "[DESCRIBE YOUR TABLE HERE" not in text
    assert '"groups": ["Bulk mixing", "Sonication", "Extrusion", "Microfluidic"]' in text
    assert f'"table": "{T_SIZE}"' in text  # the example uses the user's table
    values = tables[T_SIZE].data.to_numpy().ravel()
    assert not [v for v in values if repr(float(v)) in text]


def test_chatbot_answer_made_from_the_template_example_imports_cleanly():
    answer = "Here is the JSON:\n```json\n" + json.dumps(example_answer(T_SIZE), indent=1) + "\n```"
    outcome = import_outside_instructions(answer, sample_tables())
    assert outcome.ok and outcome.warnings == []
    render_figure(outcome.spec, sample_tables())


def test_build_tool_check_and_write(tmp_path):
    tool = ROOT / "tools" / "build_prompt_template.py"
    run = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr
    target = tmp_path / "out" / "template.md"
    run = subprocess.run([sys.executable, str(tool), "--output", str(target)], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0 and target.read_text(encoding="utf-8") == outside_prompt_template()
    target.write_text("stale", encoding="utf-8")
    run = subprocess.run([sys.executable, str(tool), "--check", "--output", str(target)], capture_output=True,
                         text=True, cwd=ROOT)
    assert run.returncode == 1 and "out of date" in run.stdout
