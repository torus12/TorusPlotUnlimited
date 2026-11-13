"""Multi-panel composer: grid layout, automatic panel letters, aligned axes, shared legend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..data.tables import DataTable
from ..spec.figure import CompositionSpec, FigureSpec
from ..stats.result import StatResult
from .fonts import register_bundled_fonts
from .prepare import PlotError
from .render import MM_PER_INCH, PanelContext, _unpack, render_panel
from .styles import rc_params, resolve_style, style_context


def panel_letter(index: int, style: str) -> str:
    """Panel label for the ``index``-th panel (0-based): A, B, ... Z, AA, AB ..."""
    if style == "none":
        return ""
    if style == "1":
        return str(index + 1)
    letters = ""
    n = index
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            break
    if style in {"a", "(a)"}:
        letters = letters.lower()
    return f"({letters})" if style.startswith("(") else letters


def render_composition(
    comp: CompositionSpec,
    figures: Mapping[str, FigureSpec] | Any,
    tables: Mapping[str, DataTable] | DataTable | None = None,
    results: Mapping[str, StatResult] | None = None,
) -> Figure:
    """Render a multi-panel figure at an exact size in millimetres.

    ``figures`` maps figure names to FigureSpecs (or pass a Project as ``figures`` and
    leave ``tables``/``results`` empty).
    """
    register_bundled_fonts()
    if hasattr(figures, "figures") and hasattr(figures, "tables"):  # Project
        project = figures
        figures, tables = project.figures, project.tables
        results = project.results if results is None else results
    tables, results = _unpack(tables or {}, results)
    if not comp.panels:
        raise PlotError("Add at least one panel to the composition.")
    missing = [p.figure for p in comp.panels if p.figure not in figures]
    if missing:
        raise PlotError(f"Figure(s) not found: {missing}. Available: {list(figures)}")

    specs: list[FigureSpec] = []
    for p in comp.panels:
        spec = figures[p.figure]
        if isinstance(spec, dict):
            spec = FigureSpec.model_validate(spec)
        if comp.style_preset:
            spec = spec.model_copy(update={"style": spec.style.model_copy(update={"preset": comp.style_preset})})
        specs.append(spec)

    base_style = resolve_style(specs[0].style)
    rc = rc_params(base_style)
    with style_context(rc):
        fig = Figure(figsize=(comp.width_mm / MM_PER_INCH, comp.height_mm / MM_PER_INCH), layout="constrained")
        FigureCanvasAgg(fig)
        fig.set_gid("torusplot-composition")
        engine = fig.get_layout_engine()
        engine.set(
            w_pad=1.5 / MM_PER_INCH, h_pad=1.5 / MM_PER_INCH,
            wspace=comp.wspace_mm / (comp.width_mm / comp.cols), hspace=comp.hspace_mm / (comp.height_mm / comp.rows),
        )
        grid = fig.add_gridspec(comp.rows, comp.cols)
        contexts: list[PanelContext] = []
        warnings: list[str] = list(base_style["warnings"])
        for i, (panel, spec) in enumerate(zip(comp.panels, specs)):
            letter = panel.label if panel.label is not None else panel_letter(i, comp.panel_labels)
            prefix = f"panel-{letter or i + 1}-".replace("(", "").replace(")", "")
            sub = grid[panel.row : panel.row + panel.rowspan, panel.col : panel.col + panel.colspan]
            style = resolve_style(spec.style)
            with style_context(rc_params(style)):
                ctx = render_panel(fig, sub, spec, tables, results, style=style, prefix=prefix)
            ctx._panel = panel
            ctx._letter = letter
            contexts.append(ctx)
            warnings.extend(f"Panel {letter or i + 1}: {w}" for w in ctx.warnings if w not in base_style["warnings"])

        _share_axes(comp, contexts)
        if comp.shared_legend:
            _shared_legend(fig, comp, contexts)
        fig.canvas.draw()  # run the layout so panel letters can be placed against final positions
        _panel_letters(fig, comp, contexts, base_style)
        fig.canvas.draw()
    fig._torusplot_rc = rc
    fig._torusplot_warnings = list(dict.fromkeys(warnings))
    fig._torusplot_contexts = contexts
    return fig


def _share_axes(comp: CompositionSpec, contexts: list[PanelContext]) -> None:
    """Give panels identical axis limits by row / column / all (aligned axes)."""
    for which, mode in (("x", comp.share_x), ("y", comp.share_y)):
        if mode == "none":
            continue
        groups: dict[Any, list[PanelContext]] = {}
        for ctx in contexts:
            key = 0 if mode == "all" else (ctx._panel.row if mode == "row" else ctx._panel.col)
            groups.setdefault(key, []).append(ctx)
        for members in groups.values():
            if len(members) < 2:
                continue
            getter = "get_xlim" if which == "x" else "get_ylim"
            setter = "set_xlim" if which == "x" else "set_ylim"
            scales = {(m.ax.get_xscale() if which == "x" else m.ax.get_yscale()) for m in members}
            if len(scales) > 1:
                continue  # never force a shared range across different scales
            lims = [getattr(m.ax, getter)() for m in members]
            lo, hi = min(min(pair) for pair in lims), max(max(pair) for pair in lims)
            for m in members:
                cur = getattr(m.ax, getter)()
                getattr(m.ax, setter)((lo, hi) if cur[0] <= cur[1] else (hi, lo))


def _shared_legend(fig: Figure, comp: CompositionSpec, contexts: list[PanelContext]) -> None:
    handles: dict[str, Any] = {}
    title = None
    for ctx in contexts:
        for handle, label in ctx.legend:
            handles.setdefault(label, handle)
        title = title or ctx.legend_title
        if ctx.legend_artist is not None:
            ctx.legend_artist.remove()
            ctx.legend_artist = None
    if not handles:
        return
    loc = {"bottom": "outside lower center", "top": "outside upper center", "right": "outside right center"}[
        comp.legend_position
    ]
    ncol = 1 if comp.legend_position == "right" else min(len(handles), 6)
    leg = fig.legend(list(handles.values()), list(handles), loc=loc, ncol=ncol, title=title, frameon=False)
    leg.set_gid("shared-legend")


def _panel_letters(fig: Figure, comp: CompositionSpec, contexts: list[PanelContext], style: dict[str, Any]) -> None:
    """Put the panel letter at the top-left corner of each panel, outside the tick labels.

    The letter is an annotation anchored to the axes, so the constrained layout reserves
    room for it. Its leftward offset equals the measured width of the y-axis decorations.
    """
    if comp.panel_labels == "none" and not any(c._panel.label for c in contexts):
        return
    renderer = fig.canvas.get_renderer()
    size = comp.panel_label_size or style["title_size"] * 1.25
    to_points = 72.0 / fig.dpi
    for ctx in contexts:
        if not ctx._letter:
            continue
        ax = ctx.ax
        frame = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer)
        left_pts = max((frame.x0 - tight.x0) * to_points, 0.0) if tight is not None else 0.0
        txt = ax.annotate(
            ctx._letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-left_pts, 5.0),
            textcoords="offset points", ha="left", va="bottom", fontsize=size,
            fontweight=comp.panel_label_weight, color=style["text_color"], annotation_clip=False,
        )
        txt.set_gid(f"panel-letter-{ctx._letter}".replace("(", "").replace(")", ""))
