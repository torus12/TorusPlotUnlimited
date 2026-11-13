""""Describe it" tests, including THE HARD RULE: AI never touches statistics."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from tests.ai import (
    RESULT_ID,
    SECRET_NUMBERS,
    T_CELLS,
    T_CONT,
    T_DOSE,
    T_SIZE,
    T_UPTAKE,
    all_prompt_text,
    make_result,
    results_map,
    sample_tables,
)
from torusplot.ai.describe import (
    SpecCheckError,
    SpecOutcome,
    check_ai_text,
    check_plot_type,
    describe_to_spec,
    format_validation_error,
    free_text_fields,
    looks_like_statistic,
    resolve_data_table,
    sanitize_spec,
    validate_answer,
)
from torusplot.ai.jsonutil import JSONExtractError
from torusplot.ai.providers import MockProvider, ProviderError
from torusplot.data import MultipleVariablesTable
from torusplot.plot import figure_warnings, render_figure
from torusplot.spec import FigureSpec

T_PAIRED_NAME = "Tumour volume (mm3)"
GOOD = {"plot_type": "box", "data": {"table": T_SIZE}, "style": {"palette": "Okabe-Ito"}}


def J(obj) -> str:
    return json.dumps(obj)


# ====================================================================== happy path
def test_happy_path_one_call():
    mock = MockProvider(responses=[J(GOOD)])
    outcome = describe_to_spec("a box plot", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and isinstance(outcome.spec, FigureSpec)
    assert outcome.spec.plot_type == "box" and outcome.spec.data.table == T_SIZE
    assert outcome.attempts == 1 and outcome.errors == [] and outcome.warnings == []
    assert outcome.raw == [J(GOOD)]
    assert outcome.message == "Figure updated."
    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call["json_mode"] is True and call["n_images"] == 0
    assert call["prompt"].startswith("TASK: describe_figure")
    assert "a box plot" in call["prompt"] and "FIELD REFERENCE" in call["system"]


def test_recorded_mock_provider_end_to_end_renders():
    tables = sample_tables()
    outcome = describe_to_spec("please make a violin plot", tables, current_table=T_SIZE, provider=MockProvider())
    assert outcome.ok and outcome.spec.plot_type == "violin" and outcome.attempts == 1
    fig = render_figure(outcome.spec, tables)
    assert figure_warnings(fig) == []
    # generic recorded fallback for every kind of table
    for name in tables:
        result = describe_to_spec("draw it", tables, current_table=name, provider=MockProvider())
        assert result.ok, (name, result.errors)
        render_figure(result.spec, tables)


def test_answer_wrapped_in_prose_and_fences_is_accepted():
    raw = f"Here you go:\n```json\n{J(GOOD)}\n```\nEnjoy!"
    outcome = describe_to_spec("box", sample_tables(), current_table=T_SIZE, provider=MockProvider(responses=[raw]))
    assert outcome.ok and outcome.attempts == 1


def test_missing_table_name_uses_current_table_without_retry():
    mock = MockProvider(responses=[J({"plot_type": "bar", "data": {}})])
    outcome = describe_to_spec("bars", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.spec.data.table == T_SIZE and outcome.attempts == 1


def test_single_table_is_the_current_table():
    tables = {T_DOSE: sample_tables()[T_DOSE]}
    mock = MockProvider(responses=[J({"plot_type": "xy", "data": {"table": "Wrong name"}})])
    outcome = describe_to_spec("xy", tables, provider=mock)
    assert outcome.ok and outcome.spec.data.table == T_DOSE
    assert "CURRENT_TABLE: " + T_DOSE in mock.calls[0]["prompt"]
    assert outcome.warnings == [f"Table 'Wrong name' does not exist; the figure uses the current table '{T_DOSE}' instead."]
    assert outcome.message == "Figure updated with 1 note(s)."


def test_empty_inputs_need_no_provider_call():
    mock = MockProvider(responses=[])
    outcome = describe_to_spec("   ", sample_tables(), provider=mock)
    assert not outcome.ok and outcome.attempts == 0 and "Describe the figure" in outcome.message
    outcome = describe_to_spec("bars", {}, provider=mock)
    assert not outcome.ok and "Add a data table first" in outcome.errors[0]
    assert mock.calls == []
    assert SpecOutcome().message == "No figure could be produced."


# ====================================================================== retry loop
def test_retry_feeds_the_validation_error_back():
    bad = J({"plot_type": "pie", "data": {"table": T_SIZE}, "bogus": 1})
    mock = MockProvider(responses=[bad, J(GOOD)])
    outcome = describe_to_spec("a box plot", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 2 and len(mock.calls) == 2
    assert len(outcome.errors) == 1 and outcome.raw == [bad, J(GOOD)]
    error = outcome.errors[0]
    assert "plot_type" in error and "bogus: Extra inputs are not permitted" in error

    first, second = mock.calls[0]["prompt"], mock.calls[1]["prompt"]
    assert "previous answer" not in first.lower()
    assert second.startswith(first)  # the original request is repeated in full
    assert bad in second  # the model sees its own previous answer
    assert f"Your previous answer was invalid: {error}" in second  # ... and the exact error
    assert second.endswith("Return the corrected JSON only.")
    assert mock.calls[0]["system"] == mock.calls[1]["system"]


def test_retry_after_unparseable_answer():
    mock = MockProvider(responses=["I think a box plot would be lovely.", J(GOOD)])
    outcome = describe_to_spec("a box plot", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 2
    assert "No JSON object found" in outcome.errors[0]
    assert "No JSON object found" in mock.calls[1]["prompt"]


@pytest.mark.parametrize("max_retries,expected_calls", [(0, 1), (1, 2), (3, 4)])
def test_exhaustion_after_exactly_one_plus_max_retries_calls(max_retries, expected_calls):
    mock = MockProvider(responses=["nope"] * 10)
    outcome = describe_to_spec("x", sample_tables(), current_table=T_SIZE, provider=mock, max_retries=max_retries)
    assert not outcome.ok and outcome.spec is None
    assert outcome.attempts == expected_calls == len(mock.calls)
    assert len(outcome.errors) == expected_calls and len(outcome.raw) == expected_calls
    assert outcome.message == outcome.errors[-1]
    # every retry is built from the ORIGINAL prompt plus the latest answer (prompts do not snowball)
    assert all(call["prompt"].count("YOUR PREVIOUS ANSWER") <= 1 for call in mock.calls)


def test_default_is_three_retries():
    mock = MockProvider(responses=["nope"] * 10)
    outcome = describe_to_spec("x", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.attempts == 4


def test_provider_errors_are_not_retried():
    mock = MockProvider(responses=[])
    with pytest.raises(ProviderError):
        describe_to_spec("x", sample_tables(), current_table=T_SIZE, provider=mock)
    assert len(mock.calls) == 1


def test_invalid_plot_type_for_table_kind_is_fed_back():
    wrong = J({"plot_type": "dose_response", "data": {"table": T_SIZE}})
    mock = MockProvider(responses=[wrong, J(GOOD)])
    outcome = describe_to_spec("curve", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 2
    assert "plot_type 'dose_response' cannot be drawn from table" in outcome.errors[0]
    assert "Column table" in outcome.errors[0] and "bar, scatter, box" in outcome.errors[0]
    assert outcome.errors[0] in mock.calls[1]["prompt"]


def test_unknown_table_without_current_table_is_fed_back():
    wrong = J({"plot_type": "bar", "data": {"table": "Imaginary"}})
    mock = MockProvider(responses=[wrong, J(GOOD)])
    outcome = describe_to_spec("bars", sample_tables(), provider=mock)  # several tables, none current
    assert outcome.ok and outcome.attempts == 2
    assert "data.table 'Imaginary' is not one of the available tables" in outcome.errors[0]


# ====================================================================== edit mode
def test_edit_mode_sends_current_spec_and_validates_full_answer():
    tables = sample_tables()
    base = FigureSpec(name="Fig 7", plot_type="bar", data={"table": T_SIZE}, title="Particle size",
                      axes={"y": {"label": "Diameter (nm)"}})
    edited = base.model_dump(mode="json", exclude_defaults=True)
    edited["layers"] = [{"type": "mean_bar", "color": "#1F77B4"}, {"type": "sd_errorbar"}, {"type": "data_points_jitter"}]
    del edited["name"]
    mock = MockProvider(responses=[J(edited)])
    outcome = describe_to_spec("make the bars blue and switch to SD error bars", tables, provider=mock, base_spec=base)
    assert outcome.ok and outcome.attempts == 1
    prompt = mock.calls[0]["prompt"]
    assert "CURRENT FIGURE" in prompt and base.model_dump_json(exclude_defaults=True) in prompt
    assert "CURRENT_TABLE: " + T_SIZE in prompt  # taken from the base spec
    spec = outcome.spec
    assert spec.name == "Fig 7"  # kept from the base spec
    assert spec.title == "Particle size" and spec.axes.y.label == "Diameter (nm)"
    assert [lay.type for lay in spec.layers] == ["mean_bar", "sd_errorbar", "data_points_jitter"]
    assert spec.layers[0].color == "#1F77B4"
    assert base.layers == []  # the input spec is untouched


def test_edit_mode_still_rejects_an_invalid_answer():
    base = FigureSpec(plot_type="bar", data={"table": T_SIZE})
    mock = MockProvider(responses=[J({"style": {"palette": "Okabe-Ito"}})] * 2)  # a patch, not a full spec
    outcome = describe_to_spec("new palette", sample_tables(), provider=mock, base_spec=base, max_retries=1)
    assert not outcome.ok and outcome.attempts == 2
    assert "plot_type: Field required" in outcome.errors[0]


def test_edit_mode_keeps_the_users_own_text():
    base = FigureSpec(plot_type="bar", data={"table": T_SIZE}, title="No significant change after washing")
    answer = base.model_dump(mode="json", exclude_defaults=True) | {"legend": {"show": False}}
    outcome = describe_to_spec("hide the legend", sample_tables(), base_spec=base,
                               provider=MockProvider(responses=[J(answer)]))
    assert outcome.ok and outcome.spec.title == base.title  # unchanged user text is not re-judged


# ====================================================================== THE HARD RULE
@pytest.mark.parametrize("field,payload", [
    ("title", {"title": "Treated vs control, p < 0.001"}),
    ("title", {"title": "***"}),
    ("texts", {"texts": [{"text": "P = 0.0123", "x": 0.5, "y": 0.9}]}),
    ("texts", {"texts": [{"text": "n.s.", "x": 0.5, "y": 0.9}]}),
    ("layers", {"layers": [{"type": "reference_line", "value": 1, "label": "p adj < 0.05"}]}),
    ("axes.y.label", {"axes": {"y": {"label": "Size (p<0.05)"}}}),
    ("axes.x.tick_labels", {"axes": {"x": {"ticks": [0, 1], "tick_labels": ["Control", "Drug ***"]}}}),
    ("legend.title", {"legend": {"title": "significantly different"}}),
    ("title", {"title": "Sizes (p-value from ANOVA)"}),
    ("title", {"title": "Drug effect, P=1e-5"}),
])
def test_typed_statistics_are_rejected_and_retried(field, payload):
    smuggled = J({"plot_type": "bar", "data": {"table": T_SIZE}, **payload})
    mock = MockProvider(responses=[smuggled, J(GOOD)])
    outcome = describe_to_spec("bars with stars", sample_tables(), results_map(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 2, outcome.errors
    assert len(outcome.errors) == 1
    assert "looks like a statistical result" in outcome.errors[0]
    assert field.split(".")[0] in outcome.errors[0]
    assert "looks like a statistical result" in mock.calls[1]["prompt"]
    assert outcome.spec.title is None and outcome.spec.texts == []


def test_typed_statistics_never_survive_even_when_the_model_insists():
    smuggled = J({"plot_type": "bar", "data": {"table": T_SIZE}, "title": "p = 0.03"})
    mock = MockProvider(responses=[smuggled] * 4)
    outcome = describe_to_spec("bars", sample_tables(), current_table=T_SIZE, provider=mock)
    assert outcome.spec is None and outcome.attempts == 4


def test_annotation_cannot_carry_a_number():
    for extra in ({"p_value": 0.001}, {"label": "***"}, {"text": "p < 0.05"}, {"stars": "**"}):
        smuggled = J({"plot_type": "bar", "data": {"table": T_SIZE},
                      "annotations": [{"stat_result_id": RESULT_ID, **extra}]})
        mock = MockProvider(responses=[smuggled, J(GOOD)])
        outcome = describe_to_spec("stars", sample_tables(), results_map(), current_table=T_SIZE, provider=mock)
        assert outcome.attempts == 2
        assert "Extra inputs are not permitted" in outcome.errors[0]


def test_invented_result_id_is_removed_with_a_warning():
    answer = J({"plot_type": "bar", "data": {"table": T_SIZE},
                "annotations": [{"stat_result_id": "sr_invented"}, {"stat_result_id": RESULT_ID}]})
    mock = MockProvider(responses=[answer])
    outcome = describe_to_spec("show brackets", sample_tables(), results_map(), current_table=T_SIZE, provider=mock)
    assert outcome.ok and outcome.attempts == 1  # sanitised, not retried
    assert [a.stat_result_id for a in outcome.spec.annotations] == [RESULT_ID]
    assert len(outcome.warnings) == 1
    assert "sr_invented" in outcome.warnings[0] and "has not been computed" in outcome.warnings[0]
    assert "must be computed in TorusPlot" in outcome.warnings[0]
    assert outcome.spec.stat_result_ids() == [RESULT_ID]


def test_every_annotation_is_removed_when_nothing_was_computed():
    answer = J({"plot_type": "bar", "data": {"table": T_SIZE, "stat_result_id": "sr_made_up"},
                "layers": [{"type": "mean_bar"}, {"type": "effect_axis", "stat_result_id": "sr_made_up2"}],
                "annotations": [{"stat_result_id": "sr_made_up3", "kind": "summary_text"}]})
    outcome = describe_to_spec("stars", sample_tables(), None, current_table=T_SIZE,
                               provider=MockProvider(responses=[answer]))
    assert outcome.ok
    assert outcome.spec.annotations == [] and outcome.spec.stat_result_ids() == []
    assert outcome.spec.data.stat_result_id is None and outcome.spec.layers[1].stat_result_id is None
    assert len(outcome.warnings) == 3
    for made_up in ("sr_made_up'", "sr_made_up2'", "sr_made_up3'"):
        assert any(made_up in w for w in outcome.warnings)


def test_no_p_value_ever_reaches_the_provider():
    tables = sample_tables()
    results = results_map()
    # sanity: the numbers really are in the result object
    dumped = results[RESULT_ID].model_dump_json()
    assert "0.0123456" in dumped and "0.0456789" in dumped

    mock = MockProvider(responses=["not json", J({"plot_type": "pie"}), J(GOOD)])
    base = FigureSpec(plot_type="bar", data={"table": T_SIZE}, annotations=[{"stat_result_id": RESULT_ID}])
    outcome = describe_to_spec("show the significant comparisons", tables, results, current_table=T_SIZE,
                               provider=mock, base_spec=base)
    assert outcome.ok and len(mock.calls) == 3
    sent = all_prompt_text(mock)
    for number in SECRET_NUMBERS:
        assert number not in sent, number
    # what the model does get: IDs, the test name and group names
    first = mock.calls[0]["prompt"]
    assert RESULT_ID in first and "One-way ANOVA" in first and f"{RESULT_ID}:c3" in first
    assert '"groups": ["Bulk mixing", "Microfluidic"]' in first
    for forbidden in ("p_value", "p_adjusted", '"statistic"', '"estimate"', '"significant":'):
        assert forbidden not in sent


def test_model_can_choose_which_computed_comparisons_to_show_and_it_renders():
    tables = sample_tables()
    results = results_map()
    answer = J({"plot_type": "bar", "data": {"table": T_SIZE},
                "annotations": [{"stat_result_id": RESULT_ID, "comparisons": "significant", "format": "stars"}]})
    outcome = describe_to_spec("bars with significant brackets", tables, results, current_table=T_SIZE,
                               provider=MockProvider(responses=[answer]))
    assert outcome.ok and outcome.warnings == []
    fig = render_figure(outcome.spec, tables, results)
    drawn = fig._torusplot_context.brackets_drawn
    # labels come from the computed result (2 of the 4 comparisons are significant), never from the model
    assert sorted(b.comparison_id for b in drawn) == [f"{RESULT_ID}:c1", f"{RESULT_ID}:c2"]
    assert sorted(b.label for b in drawn) == ["*", "***"]


# ====================================================================== sanitiser
def test_sanitize_unknown_comparisons_and_pairs():
    spec = FigureSpec(plot_type="bar", data={"table": T_SIZE}, annotations=[
        {"stat_result_id": RESULT_ID, "comparisons": [f"{RESULT_ID}:c1", f"{RESULT_ID}:c99"]},
        {"stat_result_id": RESULT_ID, "comparisons": ["made:c1"]},
        {"stat_result_id": RESULT_ID, "pairs": [["Sonication", "Bulk mixing"], ["Bulk mixing", "Placebo"]]},
        {"stat_result_id": RESULT_ID, "pairs": [["A", "B"]]},
        {"stat_result_id": RESULT_ID, "comparisons": []},
    ])
    clean, warnings = sanitize_spec(spec, sample_tables(), results_map())
    assert len(clean.annotations) == 3
    assert clean.annotations[0].comparisons == [f"{RESULT_ID}:c1"]
    assert clean.annotations[1].pairs == [("Sonication", "Bulk mixing")]
    assert clean.annotations[2].comparisons == []
    text = " | ".join(warnings)
    assert f"Comparison '{RESULT_ID}:c99' is not part of result" in text
    assert "no comparison between 'Bulk mixing' and 'Placebo'" in text
    assert "Annotation 2 was removed" in text and "Annotation 4 was removed" in text
    assert len(spec.annotations) == 5  # input untouched


def test_sanitize_warns_about_results_from_another_table():
    spec = FigureSpec(plot_type="bar", data={"table": T_PAIRED_NAME}, annotations=[{"stat_result_id": RESULT_ID}])
    clean, warnings = sanitize_spec(spec, sample_tables(), results_map())
    assert len(clean.annotations) == 1
    assert "was computed on table" in warnings[0] and T_SIZE in warnings[0]



def test_sanitize_drops_unknown_groups_rows_colors_and_series():
    tables = sample_tables()
    spec = FigureSpec(
        plot_type="bar",
        data={"table": T_UPTAKE, "groups": ["Nanoparticle", "Placebo"], "rows": ["48 h"]},
        style={"colors": {"Free dye": "#111111", "24 h": "#222222", "Ghost": "#333333"}},
        layers=[{"type": "mean_bar"}, {"type": "mean_dash", "series": "Ghost"},
                {"type": "data_points", "series": "Nanoparticle"}],
    )
    clean, warnings = sanitize_spec(spec, tables)
    assert clean.data.groups == ["Nanoparticle"] and clean.data.rows is None
    assert clean.style.colors == {"Free dye": "#111111", "24 h": "#222222"}
    assert [lay.type for lay in clean.layers] == ["mean_bar", "data_points"]
    text = " | ".join(warnings)
    assert "Group 'Placebo' does not exist" in text and "Row '48 h' does not exist" in text
    assert "Colour override for 'Ghost' was dropped" in text and "series 'Ghost' does not exist" in text
    render_figure(clean, tables)

    xy = FigureSpec(plot_type="xy", data={"table": T_DOSE, "groups": ["Free drug", "Other"]})
    clean, warnings = sanitize_spec(xy, tables)
    assert clean.data.groups == ["Free drug"] and len(warnings) == 1

    cont = FigureSpec(plot_type="stacked_bar", data={"table": T_CONT, "rows": ["Treated", "X"], "groups": ["Y"]})
    clean, warnings = sanitize_spec(cont, tables)
    assert clean.data.rows == ["Treated"] and clean.data.groups is None and len(warnings) == 2


def test_sanitize_multiple_variables_columns_and_levels():
    tables = sample_tables()
    spec = FigureSpec(
        plot_type="bar",
        data={"table": T_CELLS, "x": "Treatment", "y": "Uptake", "hue": "Batch", "replicate": "Experiment",
              "columns": ["Uptake", "Missing"], "groups": ["Control", "Imaginary"]},
        style={"colors": {"Control": "#000000", "Exp 1": "#111111", "Nobody": "#222222"}},
    )
    clean, warnings = sanitize_spec(spec, tables)
    assert clean.data.hue is None and clean.data.replicate == "Experiment"
    assert clean.data.columns == ["Uptake"] and clean.data.groups == ["Control"]
    assert set(clean.style.colors) == {"Control", "Exp 1"}
    text = " | ".join(warnings)
    assert "Column 'Batch' (hue) is not in table" in text and "Column 'Missing' is not in table" in text
    assert "Group 'Imaginary' does not exist" in text
    render_figure(clean, tables)

    # groups cannot be checked without an x column: they are left alone
    loose = FigureSpec(plot_type="histogram", data={"table": T_CELLS, "y": "Uptake", "groups": ["whatever"]})
    clean, warnings = sanitize_spec(loose, tables)
    assert clean.data.groups == ["whatever"] and warnings == []

    frame = pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "signal": [2.0, 4.0, 6.0, 8.0], "arm": ["a", "a", "b", "b"]})
    xy = FigureSpec(plot_type="xy", data={"table": "MV", "x": "time", "y": "signal", "hue": "arm", "groups": ["a", "z"]})
    clean, warnings = sanitize_spec(xy, {"MV": MultipleVariablesTable("MV", frame)})
    assert clean.data.groups == ["a"] and "Group 'z'" in warnings[0]


def test_sanitize_table_resolution():
    tables = sample_tables()
    spec = FigureSpec(plot_type="bar", data={"table": "Old name"})
    clean, warnings = sanitize_spec(spec, tables, current_table=T_SIZE)
    assert clean.data.table == T_SIZE and "Old name" in warnings[0]
    with pytest.raises(SpecCheckError, match="is not one of the available tables"):
        sanitize_spec(spec, tables)
    with pytest.raises(SpecCheckError, match="available tables: \\(none\\)"):
        sanitize_spec(spec, {})
    assert resolve_data_table(T_SIZE, tables, None) == (T_SIZE, None)
    name, note = resolve_data_table(None, tables, T_DOSE)
    assert name == T_DOSE and "No table was named" in note
    with pytest.raises(SpecCheckError, match="data.table is missing"):
        resolve_data_table(None, tables, None)
    # a single table (also when passed as the table itself) is the obvious fallback
    clean, _ = sanitize_spec(spec, tables[T_SIZE])
    assert clean.data.table == T_SIZE


# ====================================================================== small helpers
@pytest.mark.parametrize("text", ["p < 0.05", "P=0.0123", "***", "ns", "n.s.", "p adj = 0.01", "p-value 0.2",
                                  "P value: 0.01", "significant difference", "Control vs drug ** ", "p<.001",
                                  "p = 1e-5", "non-significant"])
def test_looks_like_statistic_true(text):
    assert looks_like_statistic(text)


@pytest.mark.parametrize("text", ["Hydrodynamic diameter (nm)", "Time (ns)", "Dose * time", "pH = 7.4", "**bold** text",
                                  "Group A", "Viability (%)", "", None, 5, "Sample size", "Temperature (°C)"])
def test_looks_like_statistic_false(text):
    assert not looks_like_statistic(text)


def test_free_text_fields_and_check_ai_text():
    spec = FigureSpec(plot_type="bar", data={"table": T_SIZE}, title="T", texts=[{"text": "note", "x": 0, "y": 0}],
                      layers=[{"type": "reference_line", "value": 1, "label": "limit"}],
                      axes={"x": {"label": "X", "ticks": [0], "tick_labels": ["zero"]}, "y": {"label": "Y"}},
                      legend={"title": "L"})
    assert dict(free_text_fields(spec)) == {
        "title": "T", "texts.0.text": "note", "layers.0.label": "limit", "axes.x.label": "X",
        "axes.x.tick_labels.0": "zero", "axes.y.label": "Y", "legend.title": "L"}
    check_ai_text(spec)
    bad = spec.model_copy(update={"legend": spec.legend.model_copy(update={"title": "significant (p<0.01)"})})
    with pytest.raises(SpecCheckError, match="legend.title looks like a statistical result"):
        check_ai_text(bad)
    check_ai_text(bad, base_spec=bad)  # unchanged text of the user's own figure


def test_check_plot_type_and_format_validation_error():
    tables = sample_tables()
    check_plot_type(FigureSpec(plot_type="xy", data={"table": T_DOSE}), tables[T_DOSE])
    with pytest.raises(SpecCheckError, match="XY table"):
        check_plot_type(FigureSpec(plot_type="bar", data={"table": T_DOSE}), tables[T_DOSE])
    with pytest.raises(ValidationError) as err:
        FigureSpec.model_validate({"plot_type": "bar", "data": {}, **{f"k{i}": 1 for i in range(20)}})
    text = format_validation_error(err.value, limit=3)
    assert text.startswith("data.table: Field required") and "... and 18 more problem(s)" in text
    with pytest.raises(ValidationError) as err:
        FigureSpec.model_validate({"plot_type": "bar", "data": {"table": "T"}, "title": "p < 0.01"})
    assert format_validation_error(err.value).startswith("title: Figure title looks like a statistical result")


def test_validate_answer_accepts_dicts_and_does_not_mutate_them():
    tables = sample_tables()
    payload = {"plot_type": "bar", "data": {}}
    spec, warnings = validate_answer(payload, tables, current_table=T_SIZE)
    assert spec.data.table == T_SIZE and warnings == []
    assert payload == {"plot_type": "bar", "data": {}}
    with pytest.raises(JSONExtractError):
        validate_answer("no json", tables)
    with pytest.raises(ValidationError):
        validate_answer({"plot_type": "bar"}, tables)


def test_results_can_be_passed_as_a_list():
    answer = J({"plot_type": "bar", "data": {"table": T_SIZE}, "annotations": [{"stat_result_id": RESULT_ID}]})
    outcome = describe_to_spec("stars", sample_tables(), [make_result()], current_table=T_SIZE,
                               provider=MockProvider(responses=[answer]))
    assert outcome.ok and len(outcome.spec.annotations) == 1 and outcome.warnings == []
