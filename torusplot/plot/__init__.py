"""Plot layer: renders FigureSpecs with matplotlib (styles, palettes, brackets, panels)."""

from .brackets import comparison_label, summary_text
from .colorblind import CVD_KINDS, CVD_LABELS, palette_report, simulate_colors, simulate_image
from .fonts import BUNDLED_FAMILIES, available_families, register_bundled_fonts
from .palettes import PALETTES, PaletteError, get_palette, palette_names, resolve_colors
from .prepare import PlotError, summarize
from .render import ERROR_LABELS, PanelContext, figure_warnings, render_figure, render_panel
from .styles import StyleError, get_preset, preset_names, rc_params, resolve_style

__all__ = [
    "BUNDLED_FAMILIES",
    "CVD_KINDS",
    "CVD_LABELS",
    "ERROR_LABELS",
    "PALETTES",
    "PaletteError",
    "PanelContext",
    "PlotError",
    "StyleError",
    "available_families",
    "comparison_label",
    "figure_warnings",
    "get_palette",
    "get_preset",
    "palette_names",
    "palette_report",
    "preset_names",
    "rc_params",
    "register_bundled_fonts",
    "render_figure",
    "render_panel",
    "resolve_colors",
    "resolve_style",
    "simulate_colors",
    "simulate_image",
    "summarize",
    "summary_text",
]
