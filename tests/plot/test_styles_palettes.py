import numpy as np
import pytest

from torusplot.plot import (
    BUNDLED_FAMILIES,
    CVD_KINDS,
    PALETTES,
    PaletteError,
    StyleError,
    available_families,
    get_palette,
    get_preset,
    palette_report,
    preset_names,
    rc_params,
    register_bundled_fonts,
    resolve_colors,
    resolve_style,
    simulate_colors,
    simulate_image,
)
from torusplot.plot.colorblind import min_color_distance, simulate_rgb
from torusplot.plot.fonts import FONT_DIR, resolve_family
from torusplot.plot.palettes import (
    contrast_text_color,
    darken,
    lighten,
    normalize_color,
    sample_colormap,
)
from torusplot.plot.styles import STYLE_KEYS
from torusplot.spec import StyleSpec

REQUIRED_PRESETS = [
    "Classic Lab",
    "Journal Single-Column (89 mm)",
    "Journal 1.5-Column (120 mm)",
    "Journal Double-Column (183 mm)",
    "Minimal",
    "Presentation (large type)",
    "Torus Signature",
]


def test_all_required_presets_exist_and_are_complete():
    names = preset_names()
    assert sorted(names) == sorted(REQUIRED_PRESETS)
    for name in names:
        preset = get_preset(name)
        assert all(k in preset for k in STYLE_KEYS), name
        assert preset["font_family"] in BUNDLED_FAMILIES
        assert preset["width_mm"] > 0 and preset["height_mm"] > 0
    assert get_preset("Journal Single-Column (89 mm)")["width_mm"] == 89
    assert get_preset("Journal 1.5-Column (120 mm)")["width_mm"] == 120
    assert get_preset("Journal Double-Column (183 mm)")["width_mm"] == 183
    assert get_preset("torus signature")["name"] == "Torus Signature"
    with pytest.raises(StyleError):
        get_preset("Nope")


def test_no_trademarked_names_in_presets_or_palettes():
    banned = ["prism", "graphpad", "nature", "cell", "acs", "science", "lancet", "elsevier"]
    for name in preset_names() + list(PALETTES):
        words = name.lower().replace("-", " ").replace("(", " ").replace(")", " ").split()
        assert not set(words) & set(banned), name


def test_overrides_win_over_preset_and_rc_keeps_text_editable():
    style = resolve_style(StyleSpec(preset="Minimal", font_size=11, palette="Okabe-Ito", colors={"A": "#123456"}))
    assert style["font_size"] == 11 and style["palette"] == "Okabe-Ito" and style["colors"] == {"A": "#123456"}
    rc = rc_params(style)
    assert rc["svg.fonttype"] == "none" and rc["pdf.fonttype"] == 42
    assert rc["font.family"][0] == style["font_family"]


def test_palettes_and_custom_hex():
    assert get_palette("Okabe-Ito", 3) == PALETTES["Okabe-Ito"][:3]
    assert len(get_palette("Okabe-Ito", 20)) == 20  # cycles
    assert get_palette(["#ff0000", "blue"], 3) == ["#FF0000", "#0000FF", "#FF0000"]
    assert len(set(get_palette("viridis", 5))) == 5
    assert get_palette(None, 2) == PALETTES["Torus"][:2]
    assert get_palette("okabe-ito", 1) == PALETTES["Okabe-Ito"][:1]
    assert get_palette("Torus", 0) == []
    with pytest.raises(PaletteError):
        get_palette("nope", 2)
    with pytest.raises(PaletteError):
        get_palette(["not-a-colour"], 2)
    with pytest.raises(PaletteError):
        get_palette([], 2)
    with pytest.raises(PaletteError):
        sample_colormap("nope", 3)


def test_resolve_colors_overrides_and_gradient():
    colors = resolve_colors(["a", "b", "c"], "Okabe-Ito", {"b": "#000000", "zzz": "#111111"})
    assert colors["b"] == "#000000" and colors["a"] == PALETTES["Okabe-Ito"][0] and "zzz" not in colors
    grad = resolve_colors(["a", "b", "c"], "Okabe-Ito", gradient="viridis")
    assert list(grad.values()) == sample_colormap("viridis", 3)


def test_color_helpers():
    assert normalize_color("red") == "#FF0000"
    assert darken("#808080", 0.5) == "#404040"
    assert lighten("#000000", 0.5) == "#808080"
    assert contrast_text_color("#FFFFFF") == "#000000" and contrast_text_color("#000033") == "#FFFFFF"


def test_default_palettes_stay_distinguishable_for_colourblind_viewers():
    for name in ("Okabe-Ito", "Torus"):
        report = palette_report(PALETTES[name][:5])
        for kind in ("normal", "deuteranopia", "protanopia", "tritanopia"):
            assert report[kind]["ok"], (name, kind, report[kind])


def test_cvd_simulation_properties():
    grays = ["#000000", "#808080", "#FFFFFF"]
    for kind in CVD_KINDS:
        assert simulate_colors(grays, kind) == grays  # neutral colours are unchanged
    # red and green collapse towards each other for red-green deficiencies
    assert min_color_distance(["#FF0000", "#00AA00"], "deuteranopia") < 0.5 * min_color_distance(["#FF0000", "#00AA00"])
    sim = simulate_rgb(np.array([[0.2, 0.4, 0.6]]), "grayscale")
    assert np.allclose(sim[0], sim[0][0])
    with pytest.raises(ValueError):
        simulate_colors(grays, "nope")
    img = np.zeros((4, 5, 4), dtype=np.uint8)
    img[..., 0] = 255
    img[..., 3] = 200
    out = simulate_image(img, "protanopia")
    assert out.shape == img.shape and out.dtype == np.uint8 and (out[..., 3] == 200).all()
    assert simulate_image(img[..., :3] / 255.0, "tritanopia").shape == (4, 5, 3)
    assert min_color_distance(["#000000"]) == float("inf")


def test_bundled_fonts_are_ofl_and_registered():
    files = register_bundled_fonts()
    assert len(files) == 12
    for family in BUNDLED_FAMILIES:
        assert family in available_families()
    licenses = list(FONT_DIR.glob("*/OFL.*"))
    assert len(licenses) == 3
    for lic in licenses:
        assert "SIL Open Font License" in lic.read_text(encoding="utf-8")
    names = " ".join(p.name.lower() for p in FONT_DIR.rglob("*"))
    assert "arial" not in names and "helvetica" not in names
    assert resolve_family("inter") == ("Inter", None)
    assert resolve_family(None)[0] == "Inter"
    family, warning = resolve_family("Definitely Missing Font")
    assert family == "Inter" and "not installed" in warning


def test_palettes_declared_colourblind_safe_pass_at_every_size():
    from torusplot.plot.palettes import CVD_SAFE_PALETTES

    for name in CVD_SAFE_PALETTES:
        colors = PALETTES[name]
        assert len(colors) == 8
        for n in range(2, len(colors) + 1):
            report = palette_report(colors[:n])
            for kind in ("normal", "deuteranopia", "protanopia", "tritanopia"):
                assert report[kind]["ok"], (name, n, kind, report[kind])
