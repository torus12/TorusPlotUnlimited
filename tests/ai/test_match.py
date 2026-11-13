""""Match this figure": deterministic style measurements, vision cross-check, bounded loop."""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tests.ai import (
    RESULT_ID,
    SECRET_NUMBERS,
    T_CELLS,
    T_CONT,
    T_DOSE,
    T_SIZE,
    T_UPTAKE,
    all_prompt_text,
    flat_figure,
    jpeg_of,
    png_of,
    results_map,
    sample_tables,
)
from torusplot.ai import feedback
from torusplot.ai.feedback import render_png
from torusplot.ai.match import (
    SSIM_NOTE,
    VISION_DELTA_E,
    MatchOutcome,
    StyleExtraction,
    analyze_image,
    cross_check_colors,
    data_colors,
    delta_e,
    dominant_colors,
    extract_style,
    find_axes,
    hex_to_rgb,
    load_rgb,
    match_figure,
    rgb_to_hex,
    style_to_spec,
    to_png_bytes,
)
from torusplot.ai.providers import MockProvider, Provider, ProviderError
from torusplot.data import MultipleVariablesTable
from torusplot.plot import figure_warnings, render_figure
from torusplot.spec import FigureSpec, plot_types_for

COLORS = ("#1F77B4", "#FF7F0E", "#2CA02C")
REFERENCE = png_of(flat_figure(COLORS))


def J(obj) -> str:
    return json.dumps(obj)


# ====================================================================== colour helpers
def test_hex_helpers_and_delta_e():
    assert hex_to_rgb("#1F77B4") == (31, 119, 180) and hex_to_rgb("abc") == (170, 187, 204)
    assert hex_to_rgb("#12345") is None and hex_to_rgb("#GGGGGG") is None and hex_to_rgb(5) is None
    assert rgb_to_hex((31, 119, 180)) == "#1F77B4"
    assert delta_e("#000000", "#000000") == 0.0
    assert delta_e("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.01)
    assert delta_e((31, 119, 180), "#1F78B5") < 2
    with pytest.raises(ValueError):
        delta_e("red", "#000000")


def test_load_rgb_accepts_bytes_images_and_arrays():
    rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 0))  # fully transparent -> white
    assert (load_rgb(png_of(rgba)) == 255).all()
    assert load_rgb(rgba).shape == (4, 4, 3)
    assert load_rgb(np.zeros((3, 5), dtype=np.uint8)).shape == (3, 5, 3)
    assert load_rgb(np.ones((3, 5, 3))).max() == 255  # floats in 0..1
    assert (load_rgb(np.dstack([np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 1), np.uint8)])) == 255).all()
    palette_image = flat_figure().convert("P")
    assert load_rgb(palette_image).shape == (300, 400, 3)
    with pytest.raises(ValueError, match="not an image"):
        load_rgb(b"definitely not an image")
    assert to_png_bytes(jpeg_of(flat_figure()))[:8] == b"\x89PNG\r\n\x1a\n"


# ====================================================================== palette
def test_dominant_colors_exact_hex_on_flat_image():
    colors = dominant_colors(REFERENCE, k=6)
    assert colors[0]["hex"] == "#FFFFFF" and colors[0]["role"] == "background"
    assert [c["hex"] for c in colors if c["role"] == "data"] == list(COLORS)  # ordered by area
    assert [c["hex"] for c in colors if c["role"] == "ink"] == ["#000000"]
    assert sum(c["fraction"] for c in colors) == pytest.approx(1.0, abs=1e-4)
    fractions = [c["fraction"] for c in colors]
    assert fractions == sorted(fractions, reverse=True)
    assert data_colors(REFERENCE) == list(COLORS)
    assert len(dominant_colors(REFERENCE, k=2)) == 2


def test_dominant_colors_is_deterministic():
    first = dominant_colors(REFERENCE, k=8)
    assert all(dominant_colors(REFERENCE, k=8) == first for _ in range(3))
    assert dominant_colors(flat_figure(COLORS), k=8) == first  # PIL image input, same answer


def test_dominant_colors_on_antialiased_render_recovers_exact_fills():
    palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]
    spec = FigureSpec(plot_type="bar", data={"table": T_SIZE}, style={"bar_alpha": 1.0, "palette": palette})
    png = render_png(spec, sample_tables())
    assert data_colors(png)[:4] == palette
    extraction = extract_style(png)
    assert extraction.palette == palette  # small marker/outline colours are not part of the palette
    assert extraction.background == "#FFFFFF" and extraction.source == "deterministic"


def test_dominant_colors_large_image_and_jpeg_are_near_exact():
    big = flat_figure(COLORS).resize((1600, 1200), Image.Resampling.BICUBIC)
    assert data_colors(png_of(big))[:3] == list(COLORS)
    measured = data_colors(jpeg_of(flat_figure(COLORS)))[:3]
    assert len(measured) == 3
    for truth, got in zip(COLORS, measured):
        assert delta_e(truth, got) < 3.0, (truth, got)


def test_dark_background_and_gray_fills():
    dark = dominant_colors(png_of(flat_figure(COLORS, background="#101418", ink="#F0F0F0")), k=6)
    assert dark[0] == {"hex": "#101418", "fraction": dark[0]["fraction"], "role": "background"}
    assert {c["hex"] for c in dark if c["role"] == "ink"} == {"#F0F0F0"}
    # large gray areas are fills (data), thin black lines are ink
    gray = dominant_colors(png_of(flat_figure(("#888888", "#BBBBBB", "#444444"))), k=6)
    assert [c["hex"] for c in gray if c["role"] == "data"] == ["#888888", "#BBBBBB", "#444444"]
    assert [c["hex"] for c in gray if c["role"] == "ink"] == ["#000000"]


# ====================================================================== geometry
def test_analyze_image_open_axes():
    info = analyze_image(REFERENCE)
    assert (info["width"], info["height"]) == (400, 300) and info["aspect_ratio"] == pytest.approx(1.3333)
    assert info["background"] == "#FFFFFF" and info["text_color"] == "#000000"
    left, top, right, bottom = info["plot_area"]
    assert abs(left - 50) <= 1 and abs(bottom - 250) <= 1 and abs(right - 380) <= 1 and abs(top - 20) <= 1
    assert info["has_top_spine"] is False and info["has_right_spine"] is False
    assert info["has_top_right_spines"] is False and info["grid"] is False
    assert [c["hex"] for c in info["colors"] if c["role"] == "data"] == list(COLORS)


def test_analyze_image_boxed_axes_and_grid():
    info = analyze_image(png_of(flat_figure(COLORS, box=True, grid=True)))
    assert info["has_top_spine"] and info["has_right_spine"] and info["has_top_right_spines"] is True
    assert info["grid"] is True
    assert analyze_image(png_of(flat_figure(COLORS, box=True)))["grid"] is False


def test_analyze_image_without_axes_returns_none():
    blank = Image.new("RGB", (200, 150), "#FFFFFF")
    info = analyze_image(png_of(blank))
    assert info["plot_area"] is None and info["has_top_right_spines"] is None and info["grid"] is None
    assert info["text_color"] is None and info["background"] == "#FFFFFF"
    assert find_axes(np.full((4, 4, 3), 255, dtype=np.uint8)) is None  # too small
    # one filled dark block is not a pair of axes
    block = Image.new("RGB", (200, 150), "#FFFFFF")
    block.paste("#000000", (40, 30, 160, 120))
    assert analyze_image(png_of(block))["plot_area"] is None


def test_analyze_real_renders():
    tables = sample_tables()
    open_axes = render_png(FigureSpec(plot_type="bar", data={"table": T_SIZE}), tables)
    info = analyze_image(open_axes)
    assert info["plot_area"] is not None and info["has_top_right_spines"] is False and info["grid"] is False
    boxed = render_png(FigureSpec(plot_type="xy", data={"table": T_DOSE},
                                  style={"spines": ["left", "bottom", "top", "right"], "grid": True}), tables)
    info = analyze_image(boxed)
    assert info["has_top_right_spines"] is True and info["grid"] is True


def test_find_axes_on_a_very_large_image():
    big = flat_figure(COLORS).resize((4000, 3000), Image.Resampling.NEAREST)
    axes = find_axes(load_rgb(big))
    assert axes is not None
    assert abs(axes["left"] - 500) <= 12 and abs(axes["bottom"] - 2500) <= 12


# ====================================================================== vision cross-check
def test_cross_check_keeps_near_colours_and_snaps_far_ones():
    measured = list(COLORS)
    colors, notes = cross_check_colors(["#1F78B5", "#FF00FF", "#2CA02C", "blue", "#1f77b4", 7], measured)
    # near colour kept as the model gave it; far colour replaced by the nearest measured colour
    assert colors[0] == "#1F78B5"
    nearest = min(measured, key=lambda c: delta_e(c, "#FF00FF"))
    assert colors[1] == nearest and colors[2] == "#2CA02C"
    assert delta_e("#1F78B5", "#1F77B4") <= VISION_DELTA_E < delta_e("#FF00FF", nearest)
    assert len(colors) == len(set(colors))
    text = " | ".join(notes)
    assert "#FF00FF is not in the image" in text and f"nearest measured colour {nearest}" in text
    assert "invalid colour ('blue')" in text and "invalid colour (7)" in text
    # with nothing measured, the model's colours are not trusted
    assert cross_check_colors(["#123456"], []) == ([], [
        "The vision model's colour #123456 could not be confirmed in the image; it was ignored."])
    assert cross_check_colors(["#102030"], ["#102030"], threshold=0.0) == (["#102030"], [])


def test_extract_style_with_vision_provider():
    answer = J({
        "plot_type": "Box", "colors": ["#2078B5", "#FF00FF", "#2CA02C"], "background": "#FEFEFE",
        "has_top_right_spines": True, "grid": True, "font_category": "Serif", "bar_edge": "Black",
        "error_bars": "SD", "points_shown": False, "legend_position": "upper right", "mystery": 1,
        "p_value": 0.01,
    })
    mock = MockProvider(responses=[answer])
    extraction = extract_style(REFERENCE, provider=mock)
    assert extraction.source == "deterministic+vision"
    assert extraction.plot_type == "box"
    # colours: near one kept, invented one snapped, measured colours not named by the model appended
    assert extraction.palette[0] == "#2078B5"
    assert extraction.palette[1] in COLORS and "#FF00FF" not in extraction.palette
    assert "#2CA02C" in extraction.palette and set(extraction.palette[1:]) <= set(COLORS)
    assert any("#FF00FF is not in the image" in n for n in extraction.notes)
    # measurements win over the model where pixels can tell
    assert extraction.has_top_right_spines is False and extraction.grid is False
    assert extraction.background == "#FFFFFF"
    assert extraction.font_category == "serif" and extraction.font_family == "Source Sans 3"
    assert any("serif font" in n for n in extraction.notes)
    assert extraction.bar_edge == "black" and extraction.error_bars == "sd"
    assert extraction.points_shown is False and extraction.legend_position == "upper right"
    # the request: one image, style only, JSON
    call = mock.calls[0]
    assert call["n_images"] == 1 and call["json_mode"] is True
    assert call["prompt"].startswith("TASK: extract_style") and "Do NOT read data values" in call["prompt"]


def test_extract_style_fills_gaps_from_vision_when_nothing_was_measured():
    blank = png_of(Image.new("RGB", (200, 150), "#FFFFFF"))
    answer = J({"plot_type": "pie chart", "colors": ["#FF0000"], "has_top_right_spines": True, "grid": False,
                "font_category": "sans", "bar_edge": "#aabbcc", "error_bars": "wiggly", "legend_position": "nowhere",
                "text_color": "#FFFFFF"})
    extraction = extract_style(blank, provider=MockProvider(responses=[answer]))
    assert extraction.plot_type is None and any("'pie chart'" in n for n in extraction.notes)
    assert extraction.palette == [] and any("could not be confirmed" in n for n in extraction.notes)
    assert extraction.has_top_right_spines is True and extraction.grid is False  # nothing measured: model's guess
    assert extraction.font_family == "Inter" and extraction.bar_edge == "#AABBCC"
    assert extraction.error_bars is None and extraction.legend_position is None
    assert extraction.text_color == "#FFFFFF"  # confirmed: white is in the image


def test_extract_style_survives_a_useless_vision_provider():
    garbage = extract_style(REFERENCE, provider=MockProvider(responses=["I cannot see images, sorry."]))
    assert garbage.source == "deterministic" and garbage.palette == list(COLORS)
    assert any("vision model could not be used" in n for n in garbage.notes)
    failing = extract_style(REFERENCE, provider=MockProvider(responses=[]))  # raises ProviderError inside
    assert failing.source == "deterministic" and any("could not be used" in n for n in failing.notes)

    class TextOnly(MockProvider):
        name = "text-only"
        supports_vision = False

    blind = TextOnly(responses=[])
    result = extract_style(REFERENCE, provider=blind)
    assert blind.calls == [] and any("cannot look at images" in n for n in result.notes)
    # the recorded mock answer adds nothing but is accepted
    recorded = extract_style(REFERENCE, provider=MockProvider())
    assert recorded.source == "deterministic+vision" and recorded.palette == list(COLORS)


def test_extract_style_deterministic_fields():
    extraction = extract_style(REFERENCE)
    assert isinstance(extraction, StyleExtraction)
    assert extraction.plot_type is None and extraction.font_category is None and extraction.error_bars is None
    assert extraction.palette == list(COLORS) and extraction.text_color == "#000000"
    assert extraction.aspect_ratio == pytest.approx(4 / 3, abs=1e-3)
    assert extraction.suggested_width_mm == 89 and extraction.suggested_height_mm == pytest.approx(66.8, abs=0.1)
    assert extraction.notes == []
    assert extract_style(REFERENCE) == extraction
    blank = extract_style(png_of(Image.new("RGB", (200, 150), "#FFFFFF")))
    assert len(blank.notes) == 2 and blank.palette == []


# ====================================================================== style -> spec
@pytest.mark.parametrize("name", list(sample_tables()))
def test_style_to_spec_is_valid_and_renders_for_every_table_kind(name):
    tables = sample_tables()
    table = tables[name]
    extraction = extract_style(png_of(flat_figure(COLORS, box=True, grid=True)))
    spec = style_to_spec(extraction, table, tables)
    assert isinstance(spec, FigureSpec) and FigureSpec.model_validate(spec.model_dump()) == spec
    assert spec.data.table == name and spec.plot_type in plot_types_for(table.kind)
    assert spec.annotations == [] and spec.texts == [] and spec.stat_result_ids() == []
    assert spec.style.palette == list(COLORS)
    assert spec.style.spines == ["left", "bottom", "top", "right"] and spec.style.grid is True
    assert spec.style.background == "#FFFFFF" and spec.style.text_color == "#000000"
    assert spec.layout.width_mm == 89 and spec.layout.height_mm == pytest.approx(66.8, abs=0.1)
    fig = render_figure(spec, tables)
    assert figure_warnings(fig) == []
    width, height = fig.get_size_inches()
    assert width * 25.4 == pytest.approx(89) and height * 25.4 == pytest.approx(spec.layout.height_mm)


def test_style_to_spec_default_plot_types():
    tables = sample_tables()
    extraction = StyleExtraction()
    kinds = {name: style_to_spec(extraction, name, tables).plot_type for name in tables}
    assert kinds[T_SIZE] == "bar" and kinds[T_UPTAKE] == "bar" and kinds[T_DOSE] == "xy"
    assert kinds[T_CELLS] == "bar" and kinds[T_CONT] == "stacked_bar"
    plain = style_to_spec(extraction, T_SIZE, tables)
    assert plain.style.palette is None and plain.layout.height_mm is None and plain.style.spines is None
    with pytest.raises(ValueError, match="was not found"):
        style_to_spec(extraction, "No such table", tables)


def test_style_to_spec_uses_a_valid_guess_and_explains_an_unusable_one():
    tables = sample_tables()
    assert style_to_spec(StyleExtraction(plot_type="violin"), T_SIZE, tables).plot_type == "violin"
    assert style_to_spec(StyleExtraction(plot_type="box"), T_CELLS, tables).plot_type == "box"
    assert style_to_spec(StyleExtraction(plot_type="histogram"), T_DOSE, tables).plot_type == "histogram"
    for guess, name, fallback in [("xy", T_SIZE, "bar"), ("bar", T_DOSE, "xy"), ("survival", T_CELLS, "bar"),
                                  ("estimation", T_SIZE, "bar"), ("dose_response", T_DOSE, "xy"),
                                  ("heatmap", T_CELLS, "bar")]:
        extraction = StyleExtraction(plot_type=guess)
        spec = style_to_spec(extraction, name, tables)
        assert spec.plot_type == fallback
        if guess != fallback:
            assert any(f"looks like a '{guess}' plot" in n for n in extraction.notes)
        render_figure(spec, tables)
    numbers = MultipleVariablesTable("N", pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 6.0]}))
    spec = style_to_spec(StyleExtraction(plot_type="box"), numbers, {"N": numbers})
    assert spec.plot_type == "histogram"


def test_style_to_spec_style_details_and_height_clamp():
    tables = sample_tables()
    wide = StyleExtraction(aspect_ratio=10.0, bar_edge="none", legend_position="none", has_top_right_spines=False,
                           font_family="Inter", error_bars="sd", points_shown=False, grid=False)
    spec = style_to_spec(wide, T_SIZE, tables)
    assert spec.layout.height_mm == 40.0  # clamped
    assert spec.style.bar_edge_width == 0.0 and spec.legend.show is False
    assert spec.style.spines == ["left", "bottom"] and spec.style.font_family == "Inter" and spec.style.grid is False
    # an error-bar guess is reported, never applied
    assert [lay.type for lay in spec.effective_layers()] == ["mean_bar", "sem_errorbar", "data_points_jitter"]
    assert any("did not change your error bars" in n for n in wide.notes)
    assert any("keeps showing yours" in n for n in wide.notes)

    tall = StyleExtraction(aspect_ratio=0.2, bar_edge="black", legend_position="top")
    spec = style_to_spec(tall, T_SIZE, tables)
    assert spec.layout.height_mm == 200.0 and spec.style.bar_edge_color == "#000000"
    assert spec.legend.position == "top"
    assert style_to_spec(StyleExtraction(bar_edge="#AABBCC"), T_SIZE, tables).style.bar_edge_color == "#AABBCC"
    assert style_to_spec(StyleExtraction(bar_edge="group"), T_SIZE, tables).style.bar_edge_color == "group"


def test_style_to_spec_with_base_spec_keeps_the_users_figure():
    tables = sample_tables()
    base = FigureSpec(name="Fig 2", plot_type="box", data={"table": T_SIZE, "groups": ["Extrusion", "Sonication"]},
                      annotations=[{"stat_result_id": RESULT_ID}], title="Mine",
                      style={"preset": "Journal 1.5-Column (120 mm)", "point_size": 2.0})
    extraction = extract_style(REFERENCE)
    extraction.plot_type = "bar"
    spec = style_to_spec(extraction, T_SIZE, tables, base_spec=base)
    # the user's own figure content is kept ...
    assert spec.plot_type == "box" and spec.data == base.data and spec.annotations == base.annotations
    assert spec.title == "Mine" and spec.name == "Fig 2"
    # ... and only its look changes
    assert spec.style.palette == list(COLORS) and spec.style.preset == "Journal 1.5-Column (120 mm)"
    assert spec.style.point_size == 2.0
    assert spec.layout.width_mm == 120 and spec.layout.height_mm == pytest.approx(90.0, abs=0.1)
    assert any("stays a 'box' plot" in n for n in extraction.notes)
    assert base.style.palette is None

    # a base spec of another table only lends its style
    other = style_to_spec(extract_style(REFERENCE), T_DOSE, tables, base_spec=base)
    assert other.plot_type == "xy" and other.data.table == T_DOSE and other.annotations == []
    assert other.style.preset == "Journal 1.5-Column (120 mm)"
    # an unknown preset name does not break the layout calculation
    odd = base.model_copy(update={"style": base.style.model_copy(update={"preset": "Lost preset"})})
    assert style_to_spec(extract_style(REFERENCE), T_SIZE, tables, base_spec=odd).layout.width_mm == 89


# ====================================================================== the loop
def test_match_figure_without_provider_is_one_deterministic_pass():
    tables = sample_tables()
    outcome = match_figure(REFERENCE, T_SIZE, tables)
    assert isinstance(outcome, MatchOutcome)
    assert len(outcome.history) == 1
    entry = outcome.history[0]
    assert entry["loop"] == 1 and entry["applied"] == [] and entry["rejected"] == [] and entry["critique"] is None
    assert 0.0 < entry["ssim"] < 1.0 and outcome.best_ssim == entry["ssim"]
    assert outcome.spec.data.table == T_SIZE and outcome.spec.style.palette == list(COLORS)
    assert outcome.spec.annotations == []
    assert outcome.extraction.source == "deterministic"
    assert outcome.notes[0] == SSIM_NOTE and "rough" in SSIM_NOTE
    assert outcome.reference_png[:4] == b"\x89PNG" and outcome.render_png[:4] == b"\x89PNG"
    # the outcome's image is the render of the outcome's spec, and the whole thing is reproducible
    assert outcome.render_png == render_png(outcome.spec, tables)
    again = match_figure(REFERENCE, T_SIZE, tables)
    assert again.spec == outcome.spec and again.best_ssim == outcome.best_ssim
    render_figure(outcome.spec, tables)


def test_match_figure_accepts_jpeg_and_a_table_object():
    tables = sample_tables()
    outcome = match_figure(jpeg_of(flat_figure(COLORS)), tables[T_DOSE], tables)
    assert outcome.spec.plot_type == "xy" and len(outcome.history) == 1
    assert Image.open(io.BytesIO(outcome.reference_png)).format == "PNG"


def _rising_ssim(monkeypatch):
    scores = iter([0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    monkeypatch.setattr(feedback, "ssim_score", lambda a, b: next(scores))


def test_feedback_loop_is_bounded_at_three(monkeypatch):
    _rising_ssim(monkeypatch)
    tables = sample_tables()
    style = J({"plot_type": "bar", "colors": list(COLORS)})
    patches = [J({"patch": {"style.bar_alpha": 0.5 + 0.1 * i}, "done": False, "comment": f"step {i}"})
               for i in range(10)]
    mock = MockProvider(responses=[style, *patches])
    outcome = match_figure(REFERENCE, T_SIZE, tables, provider=mock, max_loops=10)  # asks for 10, gets 3
    assert len(outcome.history) == 3
    assert [h["loop"] for h in outcome.history] == [1, 2, 3]
    assert [h["ssim"] for h in outcome.history] == [0.30, 0.40, 0.50]
    # 1 style extraction + a critique after loops 1 and 2 (none after the last render)
    assert len(mock.calls) == 3
    assert [c["n_images"] for c in mock.calls] == [1, 2, 2]
    assert [c["prompt"].split("\n")[0] for c in mock.calls] == [
        "TASK: extract_style", "TASK: critique_figure", "TASK: critique_figure"]
    assert outcome.history[0]["applied"] == ["style.bar_alpha"] and outcome.history[2]["critique"] is None
    assert outcome.history[1]["critique"]["comment"] == "step 1"
    assert outcome.best_ssim == 0.50 and outcome.spec.style.bar_alpha == pytest.approx(0.6)
    assert outcome.extraction.source == "deterministic+vision"
    render_figure(outcome.spec, tables)


@pytest.mark.parametrize("max_loops,expected", [(1, 1), (2, 2), (3, 3), (0, 1), (-5, 1)])
def test_feedback_loop_respects_smaller_limits(monkeypatch, max_loops, expected):
    _rising_ssim(monkeypatch)
    patches = [J({"patch": {"style.bar_alpha": 0.5 + 0.1 * i}, "done": False}) for i in range(5)]
    mock = MockProvider(responses=["{}", *patches])
    outcome = match_figure(REFERENCE, T_SIZE, sample_tables(), provider=mock, max_loops=max_loops)
    assert len(outcome.history) == expected and len(mock.calls) == expected


def test_feedback_loop_stops_when_the_model_is_done(monkeypatch):
    _rising_ssim(monkeypatch)
    mock = MockProvider(responses=["{}", J({"patch": {"style.grid": True}, "done": True})])
    outcome = match_figure(REFERENCE, T_SIZE, sample_tables(), provider=mock)
    assert len(outcome.history) == 1 and len(mock.calls) == 2
    assert outcome.history[0]["critique"]["done"] is True
    assert outcome.spec.style.grid is False  # the measured value: "done" means stop, nothing more is rendered


def test_feedback_loop_stops_when_ssim_stops_improving(monkeypatch):
    scores = iter([0.50, 0.40, 0.90])
    monkeypatch.setattr(feedback, "ssim_score", lambda a, b: next(scores))
    patches = [J({"patch": {"style.bar_alpha": 0.3}, "done": False}), J({"patch": {"style.bar_alpha": 0.2}})]
    mock = MockProvider(responses=["{}", *patches])
    outcome = match_figure(REFERENCE, T_SIZE, sample_tables(), provider=mock)
    assert [h["ssim"] for h in outcome.history] == [0.50, 0.40]  # stopped after the worse render
    assert outcome.best_ssim == 0.50 and outcome.spec.style.bar_alpha is None  # the best spec is kept
    assert len(mock.calls) == 2


def test_feedback_loop_rejects_non_style_patches_and_never_sends_statistics(monkeypatch):
    _rising_ssim(monkeypatch)
    tables, results = sample_tables(), results_map()
    evil = J({"patch": {"data.table": T_DOSE, "annotations": [{"stat_result_id": "sr_fake"}],
                        "title": "p < 0.001", "plot_type": "xy", "layers.0.type": "curve_fit",
                        "style.bar_alpha": 0.7}, "done": False})
    nothing = J({"patch": {"data.groups": ["Extrusion"]}, "done": False})
    mock = MockProvider(responses=["{}", evil, nothing])
    outcome = match_figure(REFERENCE, T_SIZE, tables, results, provider=mock)
    first = outcome.history[0]
    assert first["applied"] == ["style.bar_alpha"]
    assert sorted(r.split(":")[0] for r in first["rejected"]) == [
        "annotations", "data.table", "layers.0.type", "plot_type", "title"]
    # second critique proposes nothing usable -> the loop stops
    assert len(outcome.history) == 2 and outcome.history[1]["applied"] == []
    assert outcome.history[1]["rejected"][0].startswith("data.groups")
    spec = outcome.spec
    assert spec.data.table == T_SIZE and spec.data.groups is None and spec.plot_type == "bar"
    assert spec.annotations == [] and spec.title is None and spec.stat_result_ids() == []
    assert spec.style.bar_alpha == pytest.approx(0.7)
    sent = all_prompt_text(mock)
    assert not [n for n in SECRET_NUMBERS if n in sent] and RESULT_ID not in sent


def test_feedback_loop_survives_a_failing_critique_and_an_unrenderable_patch(monkeypatch):
    _rising_ssim(monkeypatch)
    tables = sample_tables()
    mock = MockProvider(responses=["{}"])  # the critique call raises ProviderError (no response queued)
    outcome = match_figure(REFERENCE, T_SIZE, tables, provider=mock)
    assert len(outcome.history) == 1 and "error" in outcome.history[0]["critique"]

    calls = {"n": 0}
    real = feedback.render_png

    def flaky(spec, tbls, results=None, dpi=100):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("cannot draw this")
        return real(spec, tbls, results, dpi)

    monkeypatch.setattr(feedback, "render_png", flaky)
    _rising_ssim(monkeypatch)
    mock = MockProvider(responses=["{}", J({"patch": {"style.bar_alpha": 0.4}})])
    outcome = match_figure(REFERENCE, T_SIZE, tables, provider=mock)
    assert len(outcome.history) == 1 and outcome.spec.style.bar_alpha is None
    assert any("could not be drawn (cannot draw this)" in n for n in outcome.notes)

    # if even the first render fails, the error is not hidden
    monkeypatch.setattr(feedback, "render_png", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        match_figure(REFERENCE, T_SIZE, tables)


def test_match_figure_with_recorded_mock_provider():
    outcome = match_figure(REFERENCE, T_SIZE, sample_tables(), provider=MockProvider())
    # recorded critique says "done": one render, two provider calls
    assert len(outcome.history) == 1 and outcome.history[0]["critique"]["done"] is True
    assert outcome.spec.style.palette == list(COLORS)


def test_text_only_provider_gives_one_deterministic_pass():
    class TextOnly(Provider):
        name = "text-only"
        supports_vision = False

        def available(self):
            return True, "ok"

        def complete(self, system, prompt, **kwargs):  # pragma: no cover - must never run
            raise ProviderError("should not be called")

    outcome = match_figure(REFERENCE, T_SIZE, sample_tables(), provider=TextOnly())
    assert len(outcome.history) == 1 and outcome.history[0]["critique"] is None
    assert any("cannot look at images" in n for n in outcome.notes)
