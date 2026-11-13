"""Page 6: Export. Figure files, statistics report, methods paragraph, project file."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st

from torusplot.data import Project, ProjectError
from torusplot.export import (
    figure_bytes,
    methods_paragraph,
    significance_legend,
    stats_report_csv,
    stats_report_markdown,
    validate_svg,
)
from torusplot.plot import PlotError, render_figure
from torusplot.plot.compose import render_composition

from .. import state

MIME = {"svg": "image/svg+xml", "pdf": "application/pdf", "png": "image/png", "tiff": "image/tiff"}
FORMAT_HELP = {
    "svg": "Vector file with editable text. Best for touching up in Inkscape, Graphite or Illustrator.",
    "pdf": "Vector file accepted by nearly every journal.",
    "tiff": "High-resolution image with LZW compression (journal submission systems).",
    "png": "Image for slides, documents and the web.",
}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "figure"


def _figure_export() -> None:
    proj = state.project()
    items = [("figure", n) for n in proj.figures] + [("composition", n) for n in proj.compositions]
    if not items:
        st.info("Create a figure on **3 Graph** first.")
        return
    kind, name = st.selectbox("What to export", items, format_func=lambda it: f"{it[1]}  ({it[0]})", key="exp_pick")
    c1, c2, c3 = st.columns(3)
    formats = [f for f in ("svg", "pdf", "tiff", "png") if c1.checkbox(f.upper(), value=f != "tiff", key=f"exp_{f}", help=FORMAT_HELP[f])]
    dpi = c2.selectbox("Resolution (dpi) for TIFF/PNG", [300, 600, 150, 1200], key="exp_dpi",
                       help="300 dpi for colour/halftone figures; 600 dpi or more for line art.")
    transparent = c3.checkbox("Transparent background", key="exp_transparent")
    try:
        if kind == "figure":
            spec = proj.figures[name]
            fig = render_figure(spec, proj)
            specs = [spec]
        else:
            comp = proj.compositions[name]
            fig = render_composition(comp, proj)
            specs = [proj.figures[p.figure] for p in comp.panels if p.figure in proj.figures]
    except PlotError as exc:
        st.warning(f"Cannot export yet: {exc}")
        return
    st.image(figure_bytes(fig, "png", dpi=110))
    w, h = fig.get_size_inches()
    st.caption(f"Exact size: {w * 25.4:.1f} x {h * 25.4:.1f} mm")
    cols = st.columns(max(len(formats), 1))
    for col, fmt in zip(cols, formats):
        data = figure_bytes(fig, fmt, dpi=int(dpi), transparent=transparent)
        ext = "tif" if fmt == "tiff" else fmt
        col.download_button(f"Download {fmt.upper()}", data=data, file_name=f"{_safe(name)}.{ext}", mime=MIME[fmt],
                            key=f"dl_{fmt}", width="stretch")
        if fmt == "svg":
            info = validate_svg(data)
            col.caption("SVG check: editable text, named groups" if info["ok"] else f"SVG check: {info['problems']}")

    st.divider()
    st.subheader("Statistics report and methods paragraph")
    ids = list(dict.fromkeys(i for s in specs for i in s.stat_result_ids()))
    results = [proj.results[i] for i in ids if i in proj.results]
    extra = [r.id for r in proj.results.values() if r.id not in ids and any(r.table == s.data.table for s in specs)]
    more = st.multiselect("Also include these results for the same data", extra, key="exp_more",
                          format_func=lambda i: f"{proj.results[i].test_name} · {i}",
                          help="Results drawn on the figure are always included.")
    results += [proj.results[i] for i in more]
    spec0 = specs[0] if len(specs) == 1 else None
    if not results:
        st.caption("No statistics are attached to this figure. Add brackets or a curve on **3 Graph**, or pick results above.")
    else:
        st.markdown("**Methods paragraph** (copy into your manuscript)")
        st.text_area("Methods", value=methods_paragraph(results, spec0), height=140, key="methods_text", label_visibility="collapsed")
        st.caption("Figure legend text: " + significance_legend())
        md = stats_report_markdown(results, f"Statistics report: {name}", spec0)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (Markdown)", data=md.encode("utf-8"), file_name=f"{_safe(name)}_stats.md",
                           mime="text/markdown", key="dl_md", width="stretch")
        d2.download_button("Download report (CSV)", data=stats_report_csv(results).encode("utf-8"),
                           file_name=f"{_safe(name)}_stats.csv", mime="text/csv", key="dl_csv", width="stretch")
        with st.expander("Preview the report"):
            st.markdown(md)


def _project_io() -> None:
    proj = state.project()
    st.subheader("Project file (.tplot)")
    st.caption("A .tplot file holds your data, figures and statistics results. Opening it later re-creates every figure exactly.")
    proj.name = st.text_input("Project name", value=proj.name, key="proj_name")
    with tempfile.TemporaryDirectory() as tmp:
        path = proj.save(Path(tmp) / "project.tplot")
        data = path.read_bytes()
    st.download_button("Save project (.tplot)", data=data, file_name=f"{_safe(proj.name)}.tplot",
                       mime="application/zip", key="dl_project")
    upload = st.file_uploader("Open a project", type=["tplot"], key="open_project")
    if upload is not None and st.button("Open this project (replaces the current one)", key="open_project_btn"):
        try:
            state.set_project(Project.load(io.BytesIO(upload.getvalue())))
            st.success("Project opened.")
            st.rerun()
        except ProjectError as exc:
            st.error(str(exc))
    if proj.log:
        with st.expander(f"Activity log ({len(proj.log)})"):
            for line in proj.log:
                st.markdown(f"- {line}")


def render() -> None:
    st.header("6 · Export")
    tab_fig, tab_proj = st.tabs(["Figures and reports", "Project file"])
    with tab_fig:
        _figure_export()
    with tab_proj:
        _project_io()
