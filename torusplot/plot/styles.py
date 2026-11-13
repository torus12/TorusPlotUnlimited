"""Style presets: YAML files in ``styles/`` + per-figure overrides from the FigureSpec."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib as mpl
import yaml

from ..spec.figure import StyleSpec
from .fonts import resolve_family

STYLE_DIR = Path(__file__).resolve().parent / "styles"
DEFAULT_PRESET = "Torus Signature"

#: Every key a preset must define (also the documentation of what a style controls).
STYLE_KEYS = (
    "width_mm", "height_mm", "font_family", "font_size", "title_size", "label_size", "tick_size",
    "legend_size", "axis_linewidth", "line_width", "errorbar_linewidth", "cap_size", "point_size",
    "point_alpha", "point_edge_color", "point_edge_width", "point_color_mode", "bar_width",
    "bar_alpha", "bar_edge_color", "bar_edge_width", "bar_fill", "spines", "tick_direction",
    "tick_length", "grid", "grid_color", "background", "text_color", "axis_color", "palette",
    "colormap", "bracket_linewidth", "annotation_size", "title_weight", "label_weight",
)


class StyleError(ValueError):
    pass


@lru_cache(maxsize=1)
def _load_presets() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for path in sorted(STYLE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = [k for k in STYLE_KEYS if k not in data]
        if missing:
            raise StyleError(f"Style preset {path.name} is missing keys: {missing}")
        presets[data["name"]] = data
    return presets


def preset_names() -> list[str]:
    """Preset names in display order."""
    presets = _load_presets()
    return sorted(presets, key=lambda n: presets[n].get("order", 99))


def get_preset(name: str) -> dict[str, Any]:
    presets = _load_presets()
    if name in presets:
        return dict(presets[name])
    for key in presets:
        if key.lower() == str(name).lower():
            return dict(presets[key])
    raise StyleError(f"Unknown style preset '{name}'. Available: {', '.join(preset_names())}")


def resolve_style(style: StyleSpec | None = None) -> dict[str, Any]:
    """Merge a preset with the spec's overrides into one flat style dict.

    Adds ``colors`` (per-group overrides), ``gradient`` and ``warnings`` (e.g. a missing font).
    """
    style = style or StyleSpec()
    resolved = get_preset(style.preset)
    overrides = style.model_dump(exclude_none=True, exclude={"preset", "colors", "gradient"})
    resolved.update(overrides)
    resolved["colors"] = dict(style.colors)
    resolved["gradient"] = style.gradient
    family, warning = resolve_family(resolved.get("font_family"))
    resolved["font_family"] = family
    resolved["warnings"] = [warning] if warning else []
    return resolved


def rc_params(style: dict[str, Any]) -> dict[str, Any]:
    """Matplotlib rcParams implementing a resolved style."""
    text, axis = style["text_color"], style["axis_color"]
    spines = set(style["spines"])
    return {
        "font.family": [style["font_family"], "DejaVu Sans"],
        "font.size": style["font_size"],
        "axes.titlesize": style["title_size"],
        "axes.titleweight": style["title_weight"],
        "axes.labelsize": style["label_size"],
        "axes.labelweight": style["label_weight"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
        "legend.fontsize": style["legend_size"],
        "legend.title_fontsize": style["legend_size"],
        "axes.linewidth": style["axis_linewidth"],
        "axes.edgecolor": axis,
        "axes.labelcolor": text,
        "axes.titlecolor": text,
        "text.color": text,
        "xtick.color": axis,
        "ytick.color": axis,
        "xtick.labelcolor": text,
        "ytick.labelcolor": text,
        "xtick.direction": style["tick_direction"],
        "ytick.direction": style["tick_direction"],
        "xtick.major.size": style["tick_length"],
        "ytick.major.size": style["tick_length"],
        "xtick.minor.size": style["tick_length"] * 0.6,
        "ytick.minor.size": style["tick_length"] * 0.6,
        "xtick.major.width": style["axis_linewidth"],
        "ytick.major.width": style["axis_linewidth"],
        "xtick.minor.width": style["axis_linewidth"] * 0.75,
        "ytick.minor.width": style["axis_linewidth"] * 0.75,
        "axes.spines.left": "left" in spines,
        "axes.spines.bottom": "bottom" in spines,
        "axes.spines.top": "top" in spines,
        "axes.spines.right": "right" in spines,
        "axes.grid": bool(style["grid"]),
        "axes.axisbelow": True,
        "grid.color": style["grid_color"],
        "grid.linewidth": max(style["axis_linewidth"] * 0.6, 0.3),
        "axes.facecolor": style["background"],
        "figure.facecolor": style["background"],
        "savefig.facecolor": style["background"],
        "lines.linewidth": style["line_width"],
        "lines.solid_capstyle": "round",
        "patch.linewidth": style["bar_edge_width"],
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.4,
        "axes.unicode_minus": True,
        "mathtext.default": "regular",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        # exports: keep text as text (editable in vector editors), embed TrueType in PDF
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "torusplot",
    }


@contextmanager
def style_context(rc: dict[str, Any]) -> Iterator[None]:
    """Apply a TorusPlot style on top of matplotlib's *built-in* defaults.

    Any matplotlibrc file or style sheet configured on the computer is ignored inside this
    context, so the same FigureSpec renders identically on every machine (a system-wide
    matplotlibrc that, say, disables text hinting would otherwise change every figure).
    """
    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update(rc)
        yield
