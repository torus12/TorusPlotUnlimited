"""Page 5: Compose. Multi-panel figures with automatic letters and exact size in mm."""

from __future__ import annotations

import math

import streamlit as st

from torusplot.export import figure_bytes
from torusplot.plot import PlotError, preset_names
from torusplot.plot.compose import render_composition
from torusplot.spec import CompositionSpec

from .. import state

LABEL_STYLES = ["A", "a", "(A)", "(a)", "1", "none"]
SHARE = ["none", "row", "col", "all"]


def render() -> None:
    st.header("5 · Compose")
    st.caption("Arrange several figures into one multi-panel figure with panel letters.")
    proj = state.project()
    if not proj.figures:
        st.info("Create at least one figure on **3 Graph** first.")
        return
    existing = list(proj.compositions)
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Composition", ["(new)", *existing], key="comp_pick")
    base = proj.compositions.get(choice) if choice != "(new)" else None
    name = c2.text_input("Name", value=base.name if base else f"Composition {len(existing) + 1}", key=f"comp_name_{choice}")
    names = list(proj.figures)
    default = [p.figure for p in base.panels if p.figure in names] if base else names[:4]
    panels = st.multiselect("Panels (in reading order)", names, default=default, key=f"comp_panels_{choice}",
                            help="Panels are placed left to right, top to bottom, and lettered automatically.")
    if not panels:
        st.info("Choose one or more figures.")
        return
    left, right = st.columns([1, 1.6])
    with left:
        auto_cols = min(len(panels), 2 if len(panels) <= 4 else 3)
        cols = int(st.number_input("Columns", 1, 6, value=base.cols if base else auto_cols, key=f"comp_cols_{choice}"))
        rows = math.ceil(len(panels) / cols)
        width = st.number_input("Width (mm)", 40.0, 500.0, value=base.width_mm if base else 183.0, step=1.0, key=f"comp_w_{choice}",
                                help="Double-column journal figures are usually 183 mm wide.")
        height = st.number_input("Height (mm)", 30.0, 500.0, value=base.height_mm if base else float(min(70 * rows, 240)), step=1.0,
                                 key=f"comp_h_{choice}")
        labels = st.selectbox("Panel letters", LABEL_STYLES, index=LABEL_STYLES.index(base.panel_labels) if base else 0, key=f"comp_labels_{choice}")
        share_y = st.selectbox("Same Y range", SHARE, index=SHARE.index(base.share_y) if base else 0, key=f"comp_sharey_{choice}",
                               help="Give panels in the same row / column / all panels the same Y axis range.")
        share_x = st.selectbox("Same X range", SHARE, index=SHARE.index(base.share_x) if base else 0, key=f"comp_sharex_{choice}")
        shared_legend = st.checkbox("One shared legend", value=base.shared_legend if base else False, key=f"comp_legend_{choice}")
        preset_options = ["(keep each figure's own style)", *preset_names()]
        preset = st.selectbox("Apply one style to all panels", preset_options, key=f"comp_preset_{choice}",
                              index=preset_options.index(base.style_preset) if base and base.style_preset in preset_options else 0)
    try:
        comp = CompositionSpec(
            name=name.strip() or "Composition", rows=rows, cols=cols, width_mm=width, height_mm=height,
            panel_labels=labels, share_x=share_x, share_y=share_y, shared_legend=shared_legend,
            style_preset=None if preset.startswith("(") else preset,
            panels=[{"figure": fig, "row": i // cols, "col": i % cols} for i, fig in enumerate(panels)],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"That layout is not valid: {exc}")
        return
    with right:
        try:
            fig = render_composition(comp, proj)
            st.image(figure_bytes(fig, "png", dpi=120))
            for w in getattr(fig, "_torusplot_warnings", []):
                st.caption(f"Note: {w}")
        except PlotError as exc:
            st.warning(f"The composition cannot be drawn yet: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected drawing problem: {exc}")
    if st.button("Save composition", key="comp_save", type="primary"):
        if choice != "(new)" and choice != comp.name:
            proj.compositions.pop(choice, None)
        proj.compositions[comp.name] = comp
        state.log(f"Saved composition '{comp.name}' with {len(comp.panels)} panel(s).")
        st.success(f"Saved '{comp.name}'. Export it on **6 Export**.")
    if choice != "(new)" and st.button("Delete composition", key="comp_delete"):
        proj.compositions.pop(choice, None)
        st.rerun()
