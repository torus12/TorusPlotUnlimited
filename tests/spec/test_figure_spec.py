import json

import pytest
from pydantic import ValidationError

from torusplot.spec import (
    PLOT_TYPES,
    CompositionSpec,
    FigureSpec,
    Layer,
    StatAnnotation,
    TextAnnotation,
    default_layers,
    figure_spec_schema,
    figure_spec_schema_json,
    minimal_example,
    plot_type_guide,
    plot_types_for,
)


def test_minimal_example_validates_and_round_trips():
    spec = FigureSpec.model_validate(minimal_example())
    again = FigureSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert spec.style.preset == "Torus Signature"
    assert [lay.type for lay in spec.effective_layers()] == ["mean_bar", "sem_errorbar", "data_points_jitter"]


def test_every_plot_type_has_defaults_and_catalogue_entry():
    for plot_type, info in PLOT_TYPES.items():
        layers = default_layers(plot_type)
        assert layers and all(isinstance(lay, Layer) for lay in layers)
        assert info["label"] and info["description"] and info["tables"]
        FigureSpec(plot_type=plot_type, data={"table": "t"})
    with pytest.raises(ValueError):
        default_layers("pie")


@pytest.mark.parametrize("plot_type", ["bar", "scatter", "box", "violin", "box_violin", "beeswarm", "superplot", "grouped_bar"])
def test_raw_points_are_shown_by_default(plot_type):
    types = [lay.type for lay in default_layers(plot_type)]
    assert any(t.startswith("data_points") for t in types)


def test_plot_types_for_table_kind():
    assert "dose_response" in plot_types_for("xy") and "dose_response" not in plot_types_for("column")
    assert "survival" in plot_types_for("multiple_variables")
    assert "bar" in plot_types_for("column") and "stacked_bar" in plot_types_for("contingency")


def test_unknown_fields_and_bad_values_are_rejected():
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, colour="red")
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="pie", data={"table": "t"})
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, layers=[{"type": "sparkles"}])
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, axes={"y": {"min": 5, "max": 1}})
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, axes={"x": {"tick_labels": ["a"]}})
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, version=99)
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, layout={"width_mm": 2})


# ---- HARD RULE: statistics can only enter a figure through a StatResult ID ----------------
def test_stat_annotation_has_no_field_for_numbers():
    fields = set(StatAnnotation.model_fields)
    assert "stat_result_id" in fields
    assert not {"p", "p_value", "pvalue", "text", "label", "stars", "value"} & fields
    with pytest.raises(ValidationError):
        StatAnnotation(stat_result_id="sr_1", p_value=0.01)
    with pytest.raises(ValidationError):
        StatAnnotation(stat_result_id="sr_1", label="***")
    with pytest.raises(ValidationError):
        StatAnnotation()  # the result ID is mandatory


@pytest.mark.parametrize(
    "text", ["p < 0.05", "P = 0.003", "p<0.0001", "***", "ns", "*", "P value = 0.04", "p adj = 0.01", "p≤0.01"]
)
def test_typed_statistics_are_rejected_everywhere(text):
    with pytest.raises(ValidationError, match="StatResult"):
        TextAnnotation(text=text, x=0.5, y=0.5)
    with pytest.raises(ValidationError):
        FigureSpec(plot_type="bar", data={"table": "t"}, title=text)
    with pytest.raises(ValidationError):
        Layer(type="reference_line", label=text)


@pytest.mark.parametrize("text", ["Control vs treated", "Day 3", "n = 8 per group", "Particle size (nm)", "PEG*", "pH = 7.4"])
def test_ordinary_text_is_allowed(text):
    assert TextAnnotation(text=text, x=0.1, y=0.9).text == text
    assert FigureSpec(plot_type="bar", data={"table": "t"}, title=text).title == text


def test_stat_result_ids_are_collected():
    spec = FigureSpec(
        plot_type="dose_response",
        data={"table": "t", "stat_result_id": "sr_c"},
        layers=[{"type": "mean_dot"}, {"type": "curve_fit", "stat_result_id": "sr_a"}],
        annotations=[{"stat_result_id": "sr_b"}, {"stat_result_id": "sr_a", "kind": "summary_text"}],
    )
    assert spec.stat_result_ids() == ["sr_b", "sr_a", "sr_c"]


def test_error_bar_and_point_helpers():
    spec = FigureSpec(plot_type="bar", data={"table": "t"})
    sd = spec.with_error_bars("sd")
    assert [lay.type for lay in sd.layers] == ["mean_bar", "sd_errorbar", "data_points_jitter"]
    assert [lay.type for lay in spec.with_error_bars(None).layers] == ["mean_bar", "data_points_jitter"]
    assert [lay.type for lay in spec.with_points("beeswarm").layers][-1] == "data_points_beeswarm"
    assert not any(lay.type.startswith("data_points") for lay in spec.with_points(None).layers)
    assert spec.layers == []  # original untouched


def test_composition_grid_validation():
    comp = CompositionSpec(rows=2, cols=2, panels=[{"figure": "a", "row": 0, "col": 0}, {"figure": "b", "row": 0, "col": 1, "rowspan": 2}])
    assert len(comp.panels) == 2
    with pytest.raises(ValidationError, match="fit"):
        CompositionSpec(rows=1, cols=2, panels=[{"figure": "a", "row": 1, "col": 0}])
    with pytest.raises(ValidationError, match="overlaps"):
        CompositionSpec(rows=2, cols=2, panels=[{"figure": "a", "row": 0, "col": 0, "colspan": 2}, {"figure": "b", "row": 0, "col": 1}])


def test_json_schema_export():
    schema = figure_spec_schema()
    assert schema["title"] == "FigureSpec"
    assert {"plot_type", "data"} <= set(schema["required"])
    assert "StatAnnotation" in schema["$defs"]
    assert "stat_result_id" in schema["$defs"]["StatAnnotation"]["required"]
    text = figure_spec_schema_json()
    assert json.loads(text) == schema
    guide = plot_type_guide()
    assert all(key in guide for key in PLOT_TYPES)
