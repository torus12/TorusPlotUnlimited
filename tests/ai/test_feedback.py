"""Visual feedback: SSIM, critique parsing and the cosmetic-only patch whitelist."""

import json

import pytest
from PIL import Image

from tests.ai import (
    RESULT_ID,
    T_DOSE,
    T_SIZE,
    flat_figure,
    jpeg_of,
    png_of,
    results_map,
    sample_tables,
)
from torusplot.ai.feedback import (
    LAYER_COSMETIC_FIELDS,
    allowed_patch_paths,
    apply_style_patch,
    critique_patch,
    is_allowed_path,
    render_png,
    ssim_score,
)
from torusplot.ai.providers import MockProvider
from torusplot.plot import render_figure
from torusplot.spec import FigureSpec
from torusplot.spec.figure import StyleSpec

A = png_of(flat_figure())
B = png_of(flat_figure(("#D62728", "#9467BD", "#8C564B"), box=True, grid=True))


def base_spec(**extra) -> FigureSpec:
    return FigureSpec(plot_type="bar", data={"table": T_SIZE, "stat_result_id": None},
                      annotations=[{"stat_result_id": RESULT_ID}], title="Sizes", **extra)


# ====================================================================== SSIM
def test_ssim_identical_images_is_exactly_one():
    assert ssim_score(A, A) == 1.0
    assert ssim_score(B, B) == 1.0
    # same picture at another size is still (almost) the same
    bigger = png_of(flat_figure().resize((800, 600), Image.Resampling.NEAREST))
    assert ssim_score(A, bigger) > 0.9


def test_ssim_is_lower_for_different_images_and_deterministic():
    score = ssim_score(A, B)
    assert -1.0 <= score < 0.95
    assert ssim_score(A, B) == score
    blank = png_of(Image.new("RGB", (400, 300), "#FFFFFF"))
    assert ssim_score(A, blank) < 0.9
    # more different -> lower score
    slightly = flat_figure()
    slightly.putpixel((200, 10), (0, 0, 0))
    assert ssim_score(A, png_of(slightly)) > ssim_score(A, B)


def test_ssim_accepts_jpeg_and_transparency():
    assert ssim_score(A, jpeg_of(flat_figure())) > 0.9
    rgba = flat_figure().convert("RGBA")
    assert ssim_score(A, png_of(rgba)) == 1.0


def test_ssim_of_two_renders_of_the_same_spec():
    tables = sample_tables()
    spec = FigureSpec(plot_type="bar", data={"table": T_SIZE})
    one, two = render_png(spec, tables), render_png(spec, tables)
    assert one == two and ssim_score(one, two) == 1.0
    other = render_png(FigureSpec(plot_type="xy", data={"table": T_DOSE}), tables)
    assert ssim_score(one, other) < 0.9


# ====================================================================== whitelist
def test_allowed_paths_are_exactly_the_cosmetic_ones():
    paths = allowed_patch_paths()
    assert all(is_allowed_path(p.replace("<i>", "0")) for p in paths)
    assert {p.split(".")[0] for p in paths} == {"style", "layout", "legend", "axes", "layers"}
    assert {f"style.{f}" for f in StyleSpec.model_fields} <= set(paths)
    assert {"layout.width_mm", "layout.height_mm", "legend.position", "axes.x.scale", "axes.y.min", "axes.y.max",
            "axes.x.label", "axes.x.ticks", "axes.zero_line", "layers.<i>.color", "layers.<i>.alpha"} <= set(paths)
    assert set(LAYER_COSMETIC_FIELDS).isdisjoint({"type", "stat_result_id", "series", "label", "value", "bins"})


@pytest.mark.parametrize("path", [
    "data", "data.table", "data.groups", "data.stat_result_id", "data.x", "annotations", "annotations.0",
    "annotations.0.stat_result_id", "annotations.0.comparisons", "texts", "texts.0.text", "title", "name",
    "plot_type", "version", "export.dpi", "layers", "layers.0", "layers.0.type", "layers.0.stat_result_id",
    "layers.0.series", "layers.0.label", "layers.0.value", "layers.0.bins", "layers.x.color", "layers.-1.color",
    "style", "style.nonexistent", "style.colors.", "axes.z.label", "axes.x.nonexistent", "axes.x", "legend",
    "layout.depth", "", "a.b.c.d", "style.colors.a.b",
])
def test_non_cosmetic_paths_are_not_allowed(path):
    assert not is_allowed_path(path)


def test_patch_applies_whitelisted_changes():
    spec = base_spec()
    patch = {
        "style.palette": ["#111111", "#222222"],
        "style.grid": True,
        "style.spines": ["left", "bottom", "top", "right"],
        "style.bar_edge_color": "group",
        "style.colors.Extrusion": "#00FF00",
        "layout.height_mm": 55,
        "legend.position": "top",
        "legend.title": "Method",
        "axes.y.scale": "log10",
        "axes.y.label": "Diameter (nm)",
        "axes.x.tick_rotation": 30,
        "axes.zero_line": True,
        "layers.0.color": "#336699",
        "layers.1.cap_size": 4,
    }
    new, applied, rejected = apply_style_patch(spec, patch)
    assert rejected == [] and applied == list(patch)
    assert new.style.palette == ["#111111", "#222222"] and new.style.grid is True
    assert new.style.colors == {"Extrusion": "#00FF00"}
    assert new.layout.height_mm == 55 and new.legend.position == "top" and new.legend.title == "Method"
    assert new.axes.y.scale == "log10" and new.axes.y.label == "Diameter (nm)" and new.axes.zero_line is True
    # default layers were materialised so that they could be restyled; what is drawn did not change
    assert [lay.type for lay in new.layers] == ["mean_bar", "sem_errorbar", "data_points_jitter"]
    assert new.layers[0].color == "#336699" and new.layers[1].cap_size == 4
    # statistics, data and text are untouched
    assert new.data == spec.data and new.annotations == spec.annotations and new.title == "Sizes"
    assert new.plot_type == "bar"
    # the input is never modified
    assert spec.layers == [] and spec.style.palette is None
    render_figure(new, sample_tables(), results_map())


def test_patch_rejects_data_and_annotation_paths():
    spec = base_spec()
    patch = {
        "data.table": T_DOSE,
        "data.groups": ["Extrusion"],
        "data.stat_result_id": "sr_fake",
        "annotations": [],
        "annotations.0.stat_result_id": "sr_fake",
        "annotations.0.comparisons": ["sr_fake:c1"],
        "texts": [{"text": "p < 0.05", "x": 0.5, "y": 0.5}],
        "title": "Significant!",
        "plot_type": "xy",
        "layers.0.type": "curve_fit",
        "layers.0.stat_result_id": "sr_fake",
        "layers.3.color": "#000000",
        "layers": [{"index": 0, "series": "Extrusion"}, {"type": "km_curve"}, "curve_fit"],
        "export.dpi": 72,
    }
    new, applied, rejected = apply_style_patch(spec, patch)
    assert applied == [] and new is spec
    reasons = dict(r.split(": ", 1) for r in rejected)
    # a list of layer objects is read as per-layer settings; anything else as one (forbidden) value
    assert set(reasons) == (set(patch) - {"layers"}) | {"layers"}
    assert "not a style setting" in reasons["data.table"] and "not a style setting" in reasons["annotations"]
    assert "there is no layer 3 (layers cannot be added)" in reasons["layers.3.color"]
    assert "not a style setting" in reasons["layers"]
    _, applied, rejected = apply_style_patch(spec, {"layers": [{"index": 0, "series": "Extrusion"}, {"type": "x"}]})
    assert applied == [] and sorted(r.split(":")[0] for r in rejected) == ["layers.0.series", "layers.1.type"]


def test_mixed_patch_applies_only_the_cosmetic_part():
    spec = base_spec()
    new, applied, rejected = apply_style_patch(spec, {"style.grid": True, "data.table": T_DOSE,
                                                      "annotations.0.stat_result_id": "sr_fake"})
    assert applied == ["style.grid"] and len(rejected) == 2
    assert new.style.grid is True and new.data.table == T_SIZE
    assert [a.stat_result_id for a in new.annotations] == [RESULT_ID]


def test_patch_rejects_invalid_values_with_reasons():
    spec = base_spec()
    patch = {
        "style.bar_alpha": 5,
        "style.preset": "Vaporwave",
        "style.palette": "NoSuchPalette",
        "style.palette ": "x",
        "style.colormap": "not-a-colormap",
        "style.background": "not-a-colour",
        "style.colors": {"Extrusion": "nope"},
        "style.colors.Sonication": "nope",
        "layers.0.color": "nope",
        "layout.width_mm": 5,
        "axes.y.scale": "logarithmic",
        "legend.ncol": 99,
        "style.spines": ["diagonal"],
    }
    new, applied, rejected = apply_style_patch(spec, patch)
    assert applied == [] and new is spec and len(rejected) == len(patch)
    reasons = dict(r.split(": ", 1) for r in rejected)
    assert "unknown style preset 'Vaporwave'" in reasons["style.preset"]
    assert "Unknown palette" in reasons["style.palette"]
    assert "less than or equal to 1" in reasons["style.bar_alpha"]
    assert "not a valid colour" in reasons["style.background"]
    assert "not a valid colour" in reasons["layers.0.color"]
    assert "Unknown colormap" in reasons["style.colormap"]


def test_patch_valid_special_values():
    spec = base_spec(style={"colors": {"Sonication": "#111111"}})
    patch = {"style.preset": "Minimal", "style.palette": "viridis", "style.colormap": "magma",
             "style.colors": {"Extrusion": "#00FF00"}, "style.bar_alpha": None, "style.gradient": None,
             "style.background": "#FAFAFA", "axes.y.label": None}
    new, applied, rejected = apply_style_patch(spec, patch)
    assert rejected == [] and len(applied) == len(patch)
    assert new.style.preset == "Minimal" and new.style.palette == "viridis"
    assert new.style.colors == {"Sonication": "#111111", "Extrusion": "#00FF00"}  # merged, not replaced
    _, _, rejected = apply_style_patch(spec, {"style.colors": ["#000000"]})
    assert "must be an object" in rejected[0]


def test_patch_rejects_statistics_typed_into_cosmetic_text_fields():
    spec = base_spec()
    patch = {"axes.y.label": "Size (p < 0.001)", "legend.title": "*** significant",
             "axes.x.ticks": [0, 1], "axes.x.tick_labels": ["A", "B ****"]}
    new, applied, rejected = apply_style_patch(spec, patch)
    assert applied == ["axes.x.ticks"]
    assert len(rejected) == 3 and all("reads like a statistical result" in r for r in rejected)
    assert new.axes.y.label is None and new.legend.title is None and new.axes.x.tick_labels is None


def test_nested_patches_are_flattened():
    spec = base_spec()
    patch = {"style": {"grid": True, "colors": {"Extrusion": "#123456"}},
             "axes": {"y": {"min": 0, "max": 250}, "zero_line": True},
             "layers": [{"index": 1, "linewidth": 2.0}, {"alpha": 0.5}],
             "data": {"table": T_DOSE},
             "annotations": [{"stat_result_id": "sr_fake"}]}
    new, applied, rejected = apply_style_patch(spec, patch)
    assert applied == ["style.grid", "style.colors", "axes.y.min", "axes.y.max", "axes.zero_line",
                       "layers.1.linewidth", "layers.1.alpha"]
    assert sorted(r.split(":")[0] for r in rejected) == ["annotations", "data.table"]
    assert new.layers[1].linewidth == 2.0 and new.layers[1].alpha == 0.5
    assert new.axes.y.max == 250 and new.data.table == T_SIZE


def test_patch_order_dependent_validation_and_bad_input():
    spec = base_spec(axes={"y": {"min": 0, "max": 10}})
    _, applied, rejected = apply_style_patch(spec, {"axes.y.min": 50})
    assert applied == [] and "Axis min must be smaller than max" in rejected[0]
    assert apply_style_patch(spec, ["style.grid"]) == (spec, [], ["(patch): must be a JSON object of dot-paths"])
    assert apply_style_patch(spec, {}) == (spec, [], [])


def test_patch_on_explicit_layers_keeps_their_statistics_reference():
    spec = FigureSpec(plot_type="xy", data={"table": T_DOSE},
                      layers=[{"type": "mean_dot"}, {"type": "curve_fit", "stat_result_id": "sr_fit"}])
    new, applied, _ = apply_style_patch(spec, {"layers.1.color": "#FF0000", "layers.1.linestyle": "dashed"})
    assert applied == ["layers.1.color", "layers.1.linestyle"]
    assert new.layers[1].stat_result_id == "sr_fit" and new.layers[1].type == "curve_fit"
    assert new.layers[1].color == "#FF0000"


# ====================================================================== critique
def test_critique_patch_request_and_parsing():
    spec = base_spec()
    answer = json.dumps({"patch": {"style.grid": True}, "done": "false", "comment": "  Add a   grid.  "})
    mock = MockProvider(responses=[f"```json\n{answer}\n```"])
    critique = critique_patch(mock, A, B, spec)
    assert critique == {"patch": {"style.grid": True}, "done": False, "comment": "Add a grid."}
    call = mock.calls[0]
    assert call["n_images"] == 2 and call["json_mode"] is True
    assert call["prompt"].startswith("TASK: critique_figure")
    assert "layers.<i>.color" in call["prompt"] and "style.palette" in call["prompt"]
    # the model is shown the style only: no data binding, no statistics references
    assert RESULT_ID not in call["prompt"] and T_SIZE not in call["prompt"]
    assert "Image 1 is a reference figure" in call["system"]


@pytest.mark.parametrize("raw,done,patch", [
    ('{"patch": {}, "done": true}', True, {}),
    ('{"patch": [], "done": "TRUE", "comment": 5}', True, {}),
    ('{"done": 1}', False, {}),
    ('{"patch": {"style.grid": false}}', False, {"style.grid": False}),
])
def test_critique_patch_tolerates_odd_answers(raw, done, patch):
    critique = critique_patch(MockProvider(responses=[raw]), A, B, base_spec())
    assert critique["done"] is done and critique["patch"] == patch and critique["comment"] == ""


def test_critique_patch_errors_stop_the_loop_instead_of_raising():
    not_json = critique_patch(MockProvider(responses=["The bars should be bluer."]), A, B, base_spec())
    assert not_json["done"] is True and not_json["patch"] == {} and "No JSON object found" in not_json["error"]
    provider_error = critique_patch(MockProvider(responses=[]), A, B, base_spec())
    assert provider_error["done"] is True and "called 1 times" in provider_error["error"]


def test_critique_comment_cannot_carry_statistics():
    raw = json.dumps({"patch": {}, "done": True, "comment": "Looks significant, p < 0.05 for all groups"})
    assert critique_patch(MockProvider(responses=[raw]), A, B, base_spec())["comment"] == ""


def test_recorded_critique_says_done():
    critique = critique_patch(MockProvider(), A, B, base_spec())
    assert critique["done"] is True and critique["patch"] == {}
