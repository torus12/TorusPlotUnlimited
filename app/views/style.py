"""Page 4: Style. Presets, palettes, fonts, sizes, colour-blind preview."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from torusplot.export import figure_bytes
from torusplot.plot import (
    CVD_LABELS,
    available_families,
    get_preset,
    palette_names,
    preset_names,
    resolve_colors,
    simulate_image,
)
from torusplot.plot.palettes import PaletteError, get_palette
from torusplot.spec import FigureSpec

from .. import state
from ..ui_helpers import preview_figure

LEGEND_POSITIONS = ["best", "upper right", "upper left", "lower left", "lower right", "right", "outside right", "top", "bottom"]


def _color_names(spec: FigureSpec, table) -> list[str]:
    if table is None:
        return []
    if table.kind == "column":
        return spec.data.groups or list(table.group_names)
    if table.kind == "xy":
        return spec.data.groups or list(table.series_names)
    if table.kind == "grouped":
        return spec.data.rows or table.row_levels if spec.data.transpose else (spec.data.groups or table.column_levels)
    if table.kind == "multiple_variables":
        col = spec.data.hue or spec.data.x
        if col and col in table.data.columns:
            return list(dict.fromkeys(table.data[col].dropna().astype(str)))
    if table.kind == "contingency":
        return list(table.column_levels)
    return []


def _opt_number(label: str, value: float | None, key: str, help_text: str, preset_value: float) -> float | None:
    raw = st.text_input(label, value="" if value is None else f"{value:g}", key=key, help=help_text,
                        placeholder=f"preset: {preset_value:g}")
    if not raw.strip():
        return None
    try:
        number = float(raw)
        return number if number > 0 else None
    except ValueError:
        st.warning(f"'{raw}' is not a number; using the preset value.")
        return None


def render() -> None:
    st.header("4 · Style")
    proj = state.project()
    if not proj.figures:
        st.info("Create a figure on **3 Graph** first.")
        return
    names = list(proj.figures)
    current = state.current_figure_name()
    name = st.selectbox("Figure", names, index=names.index(current) if current in names else 0, key="style_fig_pick")
    state.set_current_figure(name)
    spec: FigureSpec = proj.figures[name]
    table = proj.tables.get(spec.data.table)
    style = spec.style.model_dump()
    layout = spec.layout.model_dump()
    legend = spec.legend.model_dump()
    left, right = st.columns([1, 1.25])
    with left:
        presets = preset_names()
        style["preset"] = st.selectbox("Style preset", presets, index=presets.index(spec.style.preset) if spec.style.preset in presets else 0,
                                       key=f"preset_{name}", help="A preset sets fonts, line widths, colours and figure size in one click.")
        preset = get_preset(style["preset"])
        st.caption(preset["description"])
        with st.expander("Colours", expanded=True):
            pal_options = ["(preset default)", *palette_names(), "Custom colours"]
            cur = spec.style.palette
            cur_label = "Custom colours" if isinstance(cur, list) else cur if cur in pal_options else "(preset default)"
            choice = st.selectbox("Palette", pal_options, index=pal_options.index(cur_label), key=f"pal_{name}",
                                  help="Okabe-Ito and the viridis family are safe for colour-blind readers.")
            if choice == "(preset default)":
                style["palette"] = None
            elif choice == "Custom colours":
                default = ", ".join(cur) if isinstance(cur, list) else "#27318B, #E4572E, #17A398"
                raw = st.text_input("Hex colours, comma separated", value=default, key=f"pal_custom_{name}")
                colors = [c.strip() for c in raw.split(",") if c.strip()]
                try:
                    get_palette(colors, len(colors))
                    style["palette"] = colors
                except PaletteError as exc:
                    st.warning(str(exc))
            else:
                style["palette"] = choice
            grad_options = ["(off)", "viridis", "magma", "plasma", "inferno", "cividis"]
            grad = st.selectbox("Colour groups along a gradient", grad_options, key=f"grad_{name}",
                                index=grad_options.index(spec.style.gradient) if spec.style.gradient in grad_options else 0,
                                help="Useful for ordered groups such as doses or time points.")
            style["gradient"] = None if grad == "(off)" else grad
            groups = _color_names(spec, table)
            if groups and st.checkbox("Set a colour for individual groups", value=bool(spec.style.colors), key=f"percolor_{name}"):
                try:
                    base = resolve_colors(groups, style["palette"] or preset["palette"], None, style["gradient"])
                except PaletteError:
                    base = resolve_colors(groups, preset["palette"])
                overrides = {}
                cols = st.columns(min(len(groups), 4))
                for i, g in enumerate(groups):
                    picked = cols[i % len(cols)].color_picker(g, value=spec.style.colors.get(g, base[g]), key=f"col_{name}_{g}")
                    if picked.upper() != base[g].upper():
                        overrides[g] = picked.upper()
                style["colors"] = overrides
            else:
                style["colors"] = {}
        with st.expander("Text"):
            families = available_families()
            fam_options = ["(preset default)", *families]
            fam = st.selectbox("Font", fam_options, key=f"font_{name}",
                               index=fam_options.index(spec.style.font_family) if spec.style.font_family in fam_options else 0,
                               help="Inter, Source Sans 3 and Liberation Sans ship with TorusPlot. Liberation Sans has the same "
                                    "letter widths as Arial. Fonts installed on this computer are listed after them.")
            style["font_family"] = None if fam == "(preset default)" else fam
            style["font_size"] = _opt_number("Base font size (pt)", spec.style.font_size, f"fs_{name}", "Size of most text.", preset["font_size"])
            style["tick_size"] = _opt_number("Tick label size (pt)", spec.style.tick_size, f"ts_{name}", "Numbers along the axes.", preset["tick_size"])
            style["label_size"] = _opt_number("Axis title size (pt)", spec.style.label_size, f"ls_{name}", "Axis titles.", preset["label_size"])
        with st.expander("Size and lines"):
            layout["width_mm"] = _opt_number("Width (mm)", spec.layout.width_mm, f"w_{name}", "Exact figure width. Journals often ask for 89, 120 or 183 mm.", preset["width_mm"])
            layout["height_mm"] = _opt_number("Height (mm)", spec.layout.height_mm, f"h_{name}", "Exact figure height.", preset["height_mm"])
            style["axis_linewidth"] = _opt_number("Axis line width (pt)", spec.style.axis_linewidth, f"alw_{name}", "Thickness of the axes.", preset["axis_linewidth"])
            style["point_size"] = _opt_number("Point size (pt)", spec.style.point_size, f"ps_{name}", "Size of individual data points.", preset["point_size"])
            style["bar_width"] = _opt_number("Bar width (0-1)", spec.style.bar_width, f"bw_{name}", "Fraction of the slot a bar fills.", preset["bar_width"])
            if style["bar_width"] is not None and style["bar_width"] > 1:
                style["bar_width"] = 1.0
            style["grid"] = {"(preset default)": None, "On": True, "Off": False}[
                st.selectbox("Grid lines", ["(preset default)", "On", "Off"], key=f"grid_{name}",
                             index={None: 0, True: 1, False: 2}[spec.style.grid])]
        with st.expander("Legend"):
            show = st.selectbox("Show legend", ["Automatic", "Always", "Never"], key=f"leg_{name}",
                                index={None: 0, True: 1, False: 2}[spec.legend.show])
            legend["show"] = {"Automatic": None, "Always": True, "Never": False}[show]
            legend["position"] = st.selectbox("Position", LEGEND_POSITIONS, index=LEGEND_POSITIONS.index(spec.legend.position), key=f"legpos_{name}")
            legend["title"] = st.text_input("Legend title", value=spec.legend.title or "", key=f"legtitle_{name}") or None
    try:
        new = FigureSpec.model_validate({**spec.model_dump(exclude={"style", "layout", "legend"}),
                                         "style": style, "layout": layout, "legend": legend})
    except Exception as exc:  # noqa: BLE001
        st.error(f"That setting is not valid: {exc}")
        new = spec
    proj.figures[name] = new
    with right:
        st.markdown(f"**Live preview: {name}**")
        fig = preview_figure(new)
        if fig is not None:
            with st.expander("Colour-blind preview"):
                st.caption("How this figure looks to readers with the common forms of colour-vision deficiency, "
                           "and in grayscale print.")
                import io

                base = np.asarray(Image.open(io.BytesIO(figure_bytes(fig, "png", dpi=80))).convert("RGB"))
                cols = st.columns(2)
                for i, (kind, label) in enumerate(CVD_LABELS.items()):
                    cols[i % 2].image(simulate_image(base, kind), caption=label)
