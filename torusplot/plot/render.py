"""Render a validated FigureSpec with matplotlib.

``render_figure(spec, tables, results)`` returns a :class:`matplotlib.figure.Figure`
sized exactly as requested (millimetres). The same function draws single figures and
the panels of multi-panel compositions (see :mod:`torusplot.plot.compose`).

Every drawn element gets a stable SVG group id (``bar-Control``, ``points-Control``,
``bracket-1`` ...) so exported SVGs open in vector editors with named, editable parts.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..data.tables import ColumnTable, DataTable, MultipleVariablesTable
from ..spec.figure import AxisSpec, FigureSpec, Layer
from ..stats.result import StatResult
from . import brackets as _br
from .fonts import register_bundled_fonts
from .palettes import DIVERGING, darken, get_palette, lighten, resolve_colors
from .prepare import (
    CatData,
    Key,
    PlotError,
    XYData,
    prepare_categorical,
    prepare_matrix,
    prepare_value_sets,
    prepare_xy,
    summarize,
)
from .styles import rc_params, resolve_style

MM_PER_INCH = 25.4
CATEGORICAL = {
    "bar", "grouped_bar", "stacked_bar", "scatter", "box", "violin", "box_violin", "beeswarm",
    "superplot", "estimation", "paired",
}
_ERR = {
    "sem_errorbar": "sem", "sd_errorbar": "sd", "ci95_errorbar": "ci95", "range_errorbar": "range",
    "iqr_errorbar": "iqr",
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">"]
_LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
ERROR_LABELS = {"sem": "SEM", "sd": "SD", "ci95": "95% CI", "range": "range", "iqr": "IQR"}


@dataclass
class PanelContext:
    """Everything known about one rendered panel (returned for composition/legends/tests)."""

    fig: Figure
    ax: Any
    spec: FigureSpec
    style: dict[str, Any]
    prefix: str = ""
    warnings: list[str] = field(default_factory=list)
    legend: list[tuple[Any, str]] = field(default_factory=list)
    legend_title: str | None = None
    tops: dict[float, float] = field(default_factory=dict)
    positions: dict[Key, float] = field(default_factory=dict)
    cat: CatData | None = None
    axes: list[Any] = field(default_factory=list)
    brackets_drawn: list[_br.Bracket] = field(default_factory=list)
    legend_artist: Any = None

    def gid(self, kind: str, name: Any = "") -> str:
        slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
        return f"{self.prefix}{kind}" + (f"-{slug}" if slug else "")

    def bump(self, x: float, y: float) -> None:
        if y is None or not np.isfinite(y):
            return
        key = round(float(x), 6)
        if key not in self.tops or y > self.tops[key]:
            self.tops[key] = float(y)


# ====================================================================== public API
def render_figure(
    spec: FigureSpec,
    tables: Mapping[str, DataTable] | DataTable | Any,
    results: Mapping[str, StatResult] | None = None,
) -> Figure:
    """Render one figure. ``tables`` may be a mapping, a single table, or a Project."""
    register_bundled_fonts()
    tables, results = _unpack(tables, results)
    style = resolve_style(spec.style)
    rc = rc_params(style)
    with mpl.rc_context(rc):
        width = spec.layout.width_mm or style["width_mm"]
        height = spec.layout.height_mm or style["height_mm"]
        fig = Figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH), layout="constrained")
        FigureCanvasAgg(fig)
        fig.set_gid("torusplot-figure")
        ctx = render_panel(fig, fig.add_gridspec(1, 1)[0], spec, tables, results, style=style)
        fig.canvas.draw()
    fig._torusplot_rc = rc  # exports re-enter this style context
    fig._torusplot_warnings = list(dict.fromkeys(style["warnings"] + ctx.warnings))
    fig._torusplot_context = ctx
    return fig


def figure_warnings(fig: Figure) -> list[str]:
    """Non-fatal problems noticed while rendering (missing font, skipped layer...)."""
    return list(getattr(fig, "_torusplot_warnings", []))


def render_panel(
    fig: Figure,
    subplotspec: Any,
    spec: FigureSpec,
    tables: Mapping[str, DataTable],
    results: Mapping[str, StatResult] | None,
    *,
    style: dict[str, Any] | None = None,
    prefix: str = "",
) -> PanelContext:
    """Draw ``spec`` into a region of ``fig`` (must be called inside the style rc_context)."""
    style = style or resolve_style(spec.style)
    results = results or {}
    table = _table(spec, tables)
    layers = spec.effective_layers()
    if spec.plot_type == "estimation" and any(lay.type == "effect_axis" for lay in layers):
        sub = subplotspec.subgridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.05)
        ax = fig.add_subplot(sub[0])
        ax_effect = fig.add_subplot(sub[1])
    else:
        ax = fig.add_subplot(subplotspec)
        ax_effect = None
    ax.set_gid(f"{prefix}axes")
    ctx = PanelContext(fig=fig, ax=ax, spec=spec, style=style, prefix=prefix, axes=[ax])
    ctx.warnings.extend(style.get("warnings", []))

    ptype = spec.plot_type
    if ptype in CATEGORICAL:
        cat = prepare_categorical(spec, table)
        ctx.cat = cat
        _draw_categorical(ctx, cat, layers, results, table, ax_effect)
    elif ptype in {"xy", "dose_response"}:
        _draw_xy(ctx, prepare_xy(spec, table), layers, results)
    elif ptype == "survival":
        _draw_survival(ctx, table, layers, results)
    elif ptype == "histogram":
        _draw_histogram(ctx, prepare_value_sets(spec, table), layers)
    elif ptype == "heatmap":
        _draw_heatmap(ctx, table, layers, results)
    elif ptype == "qq":
        _draw_qq(ctx, table, layers, results)
    elif ptype == "residuals":
        _draw_residuals(ctx, layers, results)
    else:  # pragma: no cover - PlotType literal prevents this
        raise PlotError(f"Unknown plot type '{ptype}'.")

    _apply_axes(ctx)
    _draw_annotations(ctx, results)
    _draw_texts(ctx)
    _draw_legend(ctx)
    if spec.title:
        title = ax.set_title(spec.title)
        title.set_gid(ctx.gid("title"))
    return ctx


# ====================================================================== helpers
def _unpack(tables: Any, results: Any) -> tuple[Mapping[str, DataTable], Mapping[str, StatResult]]:
    if isinstance(tables, DataTable):
        tables = {tables.name: tables}
    elif hasattr(tables, "tables") and hasattr(tables, "results"):  # Project
        if results is None:
            results = tables.results
        tables = tables.tables
    return tables, (results or {})


def _table(spec: FigureSpec, tables: Mapping[str, DataTable]) -> DataTable:
    name = spec.data.table
    if name in tables:
        return tables[name]
    if len(tables) == 1 and name in {"", "My table"}:
        return next(iter(tables.values()))
    raise PlotError(f"Table '{name}' not found. Available tables: {list(tables)}")


def _result(results: Mapping[str, StatResult], rid: str | None, ctx: PanelContext, what: str) -> StatResult | None:
    if not rid:
        return None
    res = results.get(rid)
    if res is None:
        ctx.warnings.append(f"{what}: statistics result '{rid}' was not found, so it was not drawn.")
        return None
    if isinstance(res, dict):
        res = StatResult.model_validate(res)
    return res


def _jitter(n: int, half_width: float, *parts: Any) -> np.ndarray:
    """Deterministic, evenly spread horizontal offsets in [-half_width, half_width].

    Uses a golden-ratio low-discrepancy sequence seeded by a hash of the group name, so
    the same figure always renders identically on every platform and library version.
    """
    seed = zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) / 2**32
    u = (seed + np.arange(n) * 0.6180339887498949) % 1.0
    return (2.0 * u - 1.0) * half_width


def _ls(layer: Layer, default: str = "-") -> str:
    return _LINESTYLES.get(layer.linestyle or "", default)


def beeswarm_offsets(y: np.ndarray, dy: float, dx: float, max_half_width: float) -> np.ndarray:
    """Horizontal offsets that keep circles of diameter (dx, dy) from overlapping.

    Deterministic greedy placement: points are taken in order of y and put at the free
    position closest to the centre line. Offsets are compressed if the swarm would be
    wider than ``max_half_width``.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    x = np.zeros(n)
    if n < 2 or dy <= 0 or dx <= 0:
        return x
    order = np.argsort(y, kind="stable")
    placed_y: list[float] = []
    placed_x: list[float] = []
    start = 0
    for idx in order:
        yi = y[idx]
        while start < len(placed_y) and yi - placed_y[start] >= dy:
            start += 1
        ny = np.array(placed_y[start:])
        nx = np.array(placed_x[start:])
        if ny.size == 0:
            xi = 0.0
        else:
            need = dx * np.sqrt(np.clip(1.0 - ((yi - ny) / dy) ** 2, 0.0, 1.0))
            cands = np.concatenate([[0.0], nx + need, nx - need])
            ok = np.all(np.abs(cands[:, None] - nx[None, :]) >= need[None, :] - 1e-9, axis=1)
            valid = cands[ok]
            xi = float(valid[np.argmin(np.abs(valid))]) if valid.size else 0.0
        placed_y.append(yi)
        placed_x.append(xi)
        x[idx] = xi
    extent = np.max(np.abs(x))
    if extent > max_half_width > 0:
        x *= max_half_width / extent
    return x


def _error_bounds(values: np.ndarray, kind: str) -> tuple[float, float, float]:
    """(centre, low, high) for an error-bar kind."""
    s = summarize(values)
    if kind == "sem":
        return s["mean"], s["mean"] - s["sem"], s["mean"] + s["sem"]
    if kind == "sd":
        return s["mean"], s["mean"] - s["sd"], s["mean"] + s["sd"]
    if kind == "ci95":
        return s["mean"], s["ci_low"], s["ci_high"]
    if kind == "range":
        return s["mean"], s["min"], s["max"]
    return s["median"], s["q1"], s["q3"]  # iqr


# ====================================================================== categorical
def _draw_categorical(
    ctx: PanelContext,
    cat: CatData,
    layers: list[Layer],
    results: Mapping[str, StatResult],
    table: DataTable,
    ax_effect: Any,
) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    hues = cat.hues
    n_h = len(hues) if hues else 1
    group_width = 0.8 if hues else 1.0
    slot = group_width / n_h
    for i, c in enumerate(cat.categories):
        for j, h in enumerate(hues or [None]):
            ctx.positions[(c, h)] = i + (j - (n_h - 1) / 2.0) * slot if hues else float(i)
    color_names = hues if hues else cat.categories
    colors = resolve_colors(color_names, style.get("palette"), style.get("colors"), style.get("gradient"))

    def color_of(key: Key) -> str:
        return colors[key[1] if hues else key[0]]

    if not any(v.size for v in cat.values.values()):
        raise PlotError("There are no values to plot in the selected groups.")

    is_super = spec.plot_type == "superplot" and bool(cat.replicates)
    if spec.plot_type == "superplot" and not cat.replicates:
        ctx.warnings.append("SuperPlot needs a replicate column (data.replicate); drawing without replicate colours.")
    # values used for summaries: replicate means for SuperPlots, raw values otherwise
    summary_vals: dict[Key, np.ndarray] = {}
    rep_means: dict[Key, dict[str, float]] = {}
    for key, vals in cat.values.items():
        if is_super:
            reps = cat.replicates[key]
            rep_means[key] = {r: float(np.mean(vals[reps == r])) for r in dict.fromkeys(reps)}
            summary_vals[key] = np.array(list(rep_means[key].values()), dtype=float)
        else:
            summary_vals[key] = vals
    rep_levels = list(dict.fromkeys(r for key in cat.keys() for r in cat.replicates.get(key, []))) if is_super else []
    rep_colors = dict(zip(rep_levels, get_palette("Okabe-Ito", len(rep_levels)))) if is_super else {}

    all_vals = np.concatenate([v for v in cat.values.values() if v.size])
    data_lo, data_hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = (data_hi - min(data_lo, 0.0)) or 1.0
    fig_w, fig_h = ctx.fig.get_size_inches()
    box = ax.get_position()  # pre-layout estimate of the axes size, good enough for swarm spacing
    ax_w_in = max(fig_w * box.width * 0.93, 0.3)
    ax_h_in = max(fig_h * box.height * 0.93, 0.3)
    x_extent = len(cat.categories)
    has_fill = any(lay.type in {"mean_bar", "median_bar", "boxplot", "violin", "stack_bar"} for lay in layers)
    types = [lay.type for lay in layers]
    axis_color = style["axis_color"]
    log_y = spec.axes.y.scale in {"log10", "log2"}
    bar_floor = None
    if log_y:
        positive = all_vals[all_vals > 0]
        bar_floor = 10 ** math.floor(math.log10(positive.min())) if positive.size else 1.0

    for key in cat.keys():
        ctx.bump(ctx.positions[key], float(np.max(cat.values[key])) if cat.values[key].size else np.nan)

    stacked_bottom: dict[str, float] = dict.fromkeys(cat.categories, 0.0)

    for layer in layers:
        lt = layer.type
        lw_data = layer.linewidth if layer.linewidth is not None else style["line_width"]
        for key in cat.keys():
            vals, svals = cat.values[key], summary_vals[key]
            if layer.series and layer.series not in key:
                continue
            x = ctx.positions[key]
            col = layer.color or color_of(key)
            name = key[0] if key[1] is None else f"{key[0]}_{key[1]}"
            if svals.size == 0 and lt != "n_labels":
                continue

            if lt in {"mean_bar", "median_bar"}:
                stat = summarize(svals)["mean" if lt == "mean_bar" else "median"]
                width = (layer.width or style["bar_width"]) * slot
                fill = style["bar_fill"] if layer.fill is None else layer.fill
                alpha = layer.alpha if layer.alpha is not None else style["bar_alpha"]
                edge = layer.edge_color or (col if style["bar_edge_color"] == "group" else style["bar_edge_color"])
                bottom = bar_floor if log_y else 0.0
                face = mcolors.to_rgba(col, alpha) if fill else (0, 0, 0, 0)
                bars = ax.bar(x, stat - bottom, width=width, bottom=bottom, facecolor=face, edgecolor=edge,
                              linewidth=style["bar_edge_width"] if layer.linewidth is None else layer.linewidth,
                              zorder=2)
                bars[0].set_gid(ctx.gid("bar", name))
                ctx.bump(x, stat)

            elif lt == "stack_bar":
                stat = summarize(svals)["mean"]
                width = (layer.width or style["bar_width"]) * group_width
                alpha = layer.alpha if layer.alpha is not None else style["bar_alpha"]
                edge = layer.edge_color or (col if style["bar_edge_color"] == "group" else style["bar_edge_color"])
                xc = float(cat.categories.index(key[0]))
                ctx.positions[key] = xc
                bars = ax.bar(xc, stat, width=width, bottom=stacked_bottom[key[0]],
                              facecolor=mcolors.to_rgba(col, alpha), edgecolor=edge,
                              linewidth=style["bar_edge_width"], zorder=2)
                bars[0].set_gid(ctx.gid("bar", name))
                stacked_bottom[key[0]] += stat
                ctx.bump(xc, stacked_bottom[key[0]])

            elif lt in _ERR:
                if svals.size < 2:
                    continue
                centre, lo, hi = _error_bounds(svals, _ERR[lt])
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    continue
                direction = layer.direction or "both"
                lower = 0.0 if direction == "up" else centre - lo
                upper = 0.0 if direction == "down" else hi - centre
                ecolor = layer.color or (axis_color if not is_super else axis_color)
                cont = ax.errorbar(
                    x, centre, yerr=[[max(lower, 0.0)], [max(upper, 0.0)]], fmt="none", ecolor=ecolor,
                    elinewidth=layer.linewidth if layer.linewidth is not None else style["errorbar_linewidth"],
                    capsize=layer.cap_size if layer.cap_size is not None else style["cap_size"],
                    capthick=layer.linewidth if layer.linewidth is not None else style["errorbar_linewidth"],
                    zorder=4,
                )
                for part in cont.lines[2]:
                    part.set_gid(ctx.gid("errorbar", name))
                ctx.bump(x, hi if direction != "down" else centre)

            elif lt in {"mean_dash", "median_dash"}:
                stat = summarize(svals)["mean" if lt == "mean_dash" else "median"]
                half = (layer.width or 0.5) * slot / 2.0
                (ln,) = ax.plot([x - half, x + half], [stat, stat], color=layer.color or axis_color,
                                linewidth=lw_data * 1.15, solid_capstyle="butt", zorder=5)
                ln.set_gid(ctx.gid("mean", name))
                ctx.bump(x, stat)

            elif lt == "mean_dot":
                stat = summarize(svals)["mean"]
                (ln,) = ax.plot([x], [stat], marker=layer.marker or "o", linestyle="none",
                                markersize=(layer.size or style["point_size"]) * 1.6, color=col,
                                markeredgecolor=axis_color, markeredgewidth=style["point_edge_width"], zorder=5)
                ln.set_gid(ctx.gid("mean", name))
                ctx.bump(x, stat)

            elif lt == "boxplot":
                narrow = "violin" in types
                width = (layer.width or (0.16 if narrow else style["bar_width"])) * slot
                face = "#FFFFFF" if narrow else lighten(col, 0.55)
                bp = ax.boxplot(
                    [vals], positions=[x], widths=[width], patch_artist=True, showfliers=False,
                    manage_ticks=False, zorder=3,
                    boxprops={"facecolor": face, "edgecolor": darken(col, 0.15), "linewidth": style["bar_edge_width"] or 0.8},
                    medianprops={"color": darken(col, 0.45), "linewidth": lw_data},
                    whiskerprops={"color": darken(col, 0.15), "linewidth": style["errorbar_linewidth"]},
                    capprops={"color": darken(col, 0.15), "linewidth": style["errorbar_linewidth"]},
                    showcaps=not narrow,
                )
                bp["boxes"][0].set_gid(ctx.gid("box", name))
                ctx.bump(x, float(np.max(vals)))

            elif lt == "violin":
                _violin(ctx, x, vals, col, (layer.width or 0.8) * slot, layer, name)

            elif lt in {"data_points", "data_points_jitter", "data_points_beeswarm"}:
                size = layer.size if layer.size is not None else style["point_size"]
                if is_super:
                    size *= 0.7
                n = vals.size
                if lt == "data_points" or n == 1:
                    off = np.zeros(n)
                elif lt == "data_points_beeswarm" and n <= 1500:
                    dia_in = size / 72.0 * 1.12
                    y_t = np.log10(np.clip(vals, 1e-300, None)) if log_y else vals
                    span_t = (math.log10(max(data_hi, 1e-300)) - math.log10(max(bar_floor or data_lo, 1e-300))) if log_y else span
                    dy = dia_in * (span_t * 1.15 or 1.0) / ax_h_in
                    dx = dia_in * (x_extent / ax_w_in)
                    off = beeswarm_offsets(y_t, dy, dx, 0.46 * slot)
                else:
                    if lt == "data_points_beeswarm":
                        ctx.warnings.append("Beeswarm with more than 1500 points per group falls back to jitter.")
                    jit = (layer.jitter if layer.jitter is not None else 0.16) * slot
                    off = _jitter(n, jit, name)
                mode = style["point_color_mode"] if has_fill else "group"
                alpha = layer.alpha if layer.alpha is not None else style["point_alpha"]
                edge_c, edge_w = style["point_edge_color"], style["point_edge_width"]
                if is_super:
                    reps = cat.replicates[key]
                    face = [rep_colors[r] for r in reps]
                    alpha = min(alpha, 0.55)
                    edge_c, edge_w = "none", 0.0
                elif layer.color:
                    face = layer.color
                elif mode == "dark":
                    face = darken(col, 0.35)
                elif mode == "black":
                    face = "#1A1A1A"
                elif mode == "white":
                    face, edge_c, edge_w = "#FFFFFF", darken(col, 0.2) if col.upper() != "#FFFFFF" else "#000000", max(edge_w, 0.6)
                else:
                    face = col
                sc = ax.scatter(x + off, vals, s=size**2, c=face, alpha=alpha, marker=layer.marker or "o",
                                edgecolors=layer.edge_color or edge_c, linewidths=edge_w, zorder=3, clip_on=False)
                sc.set_gid(ctx.gid("points", name))

            elif lt == "replicate_means" and is_super:
                size = (layer.size if layer.size is not None else style["point_size"]) * 2.1
                n_rep = len(rep_means[key])
                offs = np.linspace(-0.18, 0.18, n_rep) * slot if n_rep > 1 else [0.0]
                for o, (r, m) in zip(offs, rep_means[key].items()):
                    marker = _MARKERS[rep_levels.index(r) % len(_MARKERS)]
                    (ln,) = ax.plot([x + o], [m], marker=marker, linestyle="none", markersize=size,
                                    color=rep_colors[r], markeredgecolor="#111111", markeredgewidth=0.8, zorder=6)
                    ln.set_gid(ctx.gid("replicate-mean", f"{name}_{r}"))

            elif lt == "value_labels":
                stat = summarize(svals)["mean"]
                top = ctx.tops.get(round(x, 6), stat)
                txt = ax.annotate(f"{stat:.3g}", (x, top), xytext=(0, 2), textcoords="offset points", ha="center",
                                  va="bottom", fontsize=layer.size or style["tick_size"], zorder=6)
                txt.set_gid(ctx.gid("value-label", name))
                ctx.bump(x, top + 0.07 * span)

            elif lt == "n_labels":
                txt = ax.annotate(f"n={svals.size}", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                                  textcoords="offset points", ha="center", va="bottom",
                                  fontsize=layer.size or style["tick_size"] * 0.9, color=style["text_color"], zorder=6)
                txt.set_gid(ctx.gid("n-label", name))

        if lt == "paired_lines":
            _paired_lines(ctx, cat, layer)
        elif lt == "reference_line":
            _reference_line(ctx, layer)
        elif lt == "mean_line":
            for h in hues or [None]:
                pts = [(ctx.positions[(c, h)], summarize(summary_vals[(c, h)])["mean"]) for c in cat.categories
                       if summary_vals[(c, h)].size]
                if len(pts) > 1:
                    col = layer.color or (colors[h] if hues else axis_color)
                    (ln,) = ax.plot(*zip(*pts), color=col, linewidth=lw_data, linestyle=_ls(layer), zorder=4)
                    ln.set_gid(ctx.gid("mean-line", h or "all"))

    # x axis
    ax.set_xticks(range(len(cat.categories)))
    ax.set_xticklabels(cat.categories)
    ax.set_xlim(-0.6, len(cat.categories) - 0.4)
    ax.tick_params(axis="x", length=0 if style["tick_length"] == 0 else style["tick_length"])
    label_chars = sum(len(c) for c in cat.categories) + 1.5 * len(cat.categories)
    if spec.axes.x.tick_rotation is None and label_chars * style["tick_size"] * 0.5 > ax_w_in * 72:
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
    if has_fill and not log_y and data_lo >= 0 and spec.axes.y.min is None and "boxplot" not in types and "violin" not in types:
        ax.set_ylim(bottom=0.0)
    ctx._auto_xlabel = cat.x_label
    ctx._auto_ylabel = cat.y_label

    # legend
    if hues and len(hues) >= 1 and (len(hues) > 1 or spec.legend.show):
        alpha = style["bar_alpha"] if has_fill else 1.0
        for h in hues:
            ctx.legend.append((Patch(facecolor=mcolors.to_rgba(colors[h], alpha), edgecolor=colors[h], linewidth=0.6), h))
        ctx.legend_title = cat.hue_label
    if is_super:
        for r in rep_levels:
            marker = _MARKERS[rep_levels.index(r) % len(_MARKERS)]
            ctx.legend.append((Line2D([], [], marker=marker, linestyle="none", color=rep_colors[r],
                                      markeredgecolor="#111111", markeredgewidth=0.6, markersize=style["point_size"] * 1.6), r))
        ctx.legend_title = spec.data.replicate

    if ax_effect is not None:
        _effect_axis(ctx, ax_effect, cat, layers, results, table)


def _violin(ctx: PanelContext, x: float, vals: np.ndarray, col: str, width: float, layer: Layer, name: str) -> None:
    from scipy.stats import gaussian_kde

    ax, style = ctx.ax, ctx.style
    if vals.size < 3 or np.ptp(vals) == 0:
        return
    try:
        kde = gaussian_kde(vals)
    except (np.linalg.LinAlgError, ValueError):
        return
    grid = np.linspace(vals.min(), vals.max(), 160)
    dens = kde(grid)
    dens = dens / dens.max() * (width / 2.0)
    alpha = layer.alpha if layer.alpha is not None else 0.35
    poly = ax.fill_betweenx(grid, x - dens, x + dens, facecolor=mcolors.to_rgba(col, alpha), edgecolor=col,
                            linewidth=style["bar_edge_width"] or 0.8, zorder=2)
    poly.set_gid(ctx.gid("violin", name))
    ctx.bump(x, float(vals.max()))


def _paired_lines(ctx: PanelContext, cat: CatData, layer: Layer) -> None:
    ax, style = ctx.ax, ctx.style
    if not cat.subjects:
        ctx.warnings.append("Paired lines need subject information (rows of a Column table, or data.subject).")
        return
    for h in cat.hues or [None]:
        by_subject: dict[Any, list[tuple[float, float]]] = {}
        for c in cat.categories:
            key = (c, h)
            for s, v in zip(cat.subjects.get(key, []), cat.values[key]):
                by_subject.setdefault(s, []).append((ctx.positions[key], float(v)))
        for n, (s, pts) in enumerate(by_subject.items()):
            if len(pts) > 1:
                (ln,) = ax.plot(*zip(*pts), color=layer.color or "#7A7F87", alpha=layer.alpha if layer.alpha is not None else 0.75,
                                linewidth=layer.linewidth if layer.linewidth is not None else style["line_width"] * 0.7,
                                linestyle=_ls(layer), zorder=2.5)
                ln.set_gid(ctx.gid("pair-line", f"{h or ''}{n + 1}_{s}"))


def _reference_line(ctx: PanelContext, layer: Layer) -> None:
    ax, style = ctx.ax, ctx.style
    value = 0.0 if layer.value is None else layer.value
    kw = {"color": layer.color or style["axis_color"], "linewidth": layer.linewidth if layer.linewidth is not None else style["axis_linewidth"],
          "linestyle": _ls(layer, "--"), "zorder": 1.5, "alpha": layer.alpha if layer.alpha is not None else 0.8}
    ln = ax.axvline(value, **kw) if layer.orientation == "vertical" else ax.axhline(value, **kw)
    ln.set_gid(ctx.gid("reference-line", f"{value:g}"))
    if layer.label:
        ctx.legend.append((ln, layer.label))


def _effect_axis(ctx: PanelContext, ax2: Any, cat: CatData, layers: list[Layer], results: Mapping[str, StatResult],
                 table: DataTable) -> None:
    """Right-hand axis of an estimation plot: mean difference(s) with CI from a StatResult."""
    style = ctx.style
    layer = next(lay for lay in layers if lay.type == "effect_axis")
    res = _result(results, layer.stat_result_id or ctx.spec.data.stat_result_id, ctx, "Estimation plot")
    if res is None and cat.hues is None and len(cat.categories) >= 2:
        # No result attached: compute the mean difference with the statistics engine.
        from ..stats import run_analysis

        a, b = cat.categories[0], cat.categories[1]
        tmp = ColumnTable.from_dict("estimation", {a: cat.values[(a, None)], b: cat.values[(b, None)]})
        try:
            res = run_analysis(tmp, "ttest_welch", group_a=b, group_b=a)
        except Exception as exc:  # noqa: BLE001
            ctx.warnings.append(f"Estimation plot: could not compute the mean difference ({exc}).")
    ctx.axes.append(ax2)
    ax2.set_gid(ctx.gid("effect-axes"))
    entries: list[tuple[str, float, float, float]] = []
    if res is not None:
        if res.comparisons:
            shown = set(cat.categories)
            for comp in res.comparisons:
                if not {comp.group_a, comp.group_b} <= shown:
                    continue
                if comp.estimate is not None and comp.ci_low is not None and comp.ci_high is not None:
                    entries.append((f"{comp.group_a}\n− {comp.group_b}", comp.estimate, comp.ci_low, comp.ci_high))
        else:
            for par in res.parameters:
                if "difference" in par.name.lower() and par.value is not None and par.ci_low is not None:
                    label = par.name
                    if res.n and len(res.n) == 2:
                        names = list(res.n)
                        label = f"{names[0]}\n− {names[1]}"
                    entries.append((label, par.value, par.ci_low, par.ci_high))
                    break
    if not entries:
        ctx.warnings.append("Estimation plot: no mean difference with a confidence interval was available.")
        ax2.set_axis_off()
        return
    for i, (_label, est, lo, hi) in enumerate(entries):
        cont = ax2.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", color=style["axis_color"],
                            markersize=style["point_size"] * 1.7, elinewidth=style["errorbar_linewidth"] * 1.4,
                            capsize=0, zorder=4)
        cont.lines[0].set_gid(ctx.gid("effect", i + 1))
    ax2.axhline(0.0, color=style["axis_color"], linewidth=style["axis_linewidth"], linestyle="--", alpha=0.7)
    ax2.set_xticks(range(len(entries)))
    ax2.set_xticklabels([e[0] for e in entries], fontsize=style["tick_size"] * 0.9)
    ax2.set_xlim(-0.7, len(entries) - 0.3)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    level = 95
    ax2.set_ylabel(f"Mean difference ({level}% CI)")
    ax2.tick_params(axis="x", length=0)


# ====================================================================== XY
def _draw_xy(ctx: PanelContext, data: XYData, layers: list[Layer], results: Mapping[str, StatResult]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    if not data.series or not any(s.x.size for s in data.series):
        raise PlotError("There are no XY points to plot.")
    names = [s.name for s in data.series]
    colors = resolve_colors(names, style.get("palette"), style.get("colors"), style.get("gradient"))
    multi = len(names) > 1
    curve_results: list[tuple[Layer, StatResult]] = []
    for layer in layers:
        if layer.type in {"curve_fit", "confidence_band", "prediction_band"}:
            if not layer.stat_result_id:
                if layer.type == "curve_fit":
                    ctx.warnings.append(
                        "No fit is attached to the curve layer. Run a regression in Analyze and attach its result."
                    )
                continue
            res = _result(results, layer.stat_result_id, ctx, "Fitted curve")
            if res is not None:
                curve_results.append((layer, res))
    x_is_log = any(bool(res.extra.get("x_is_log")) for _, res in curve_results)
    log_x = spec.axes.x.scale in {"log10", "log2"} or (
        spec.plot_type == "dose_response" and "scale" not in spec.axes.x.model_fields_set and not x_is_log
        and all(np.all(s.x[np.isfinite(s.x)] >= 0) for s in data.series)
    )
    ctx._force_log_x = log_x and spec.axes.x.scale == "linear"
    dropped = 0

    def xmask(x: np.ndarray) -> np.ndarray:
        return x > 0 if log_x else np.ones_like(x, dtype=bool)

    for layer in layers:
        lt = layer.type
        lw = layer.linewidth if layer.linewidth is not None else style["line_width"]
        if lt == "reference_line":
            _reference_line(ctx, layer)
            continue
        if lt in {"curve_fit", "confidence_band", "prediction_band"}:
            continue
        for i, s in enumerate(data.series):
            if layer.series and layer.series != s.name:
                continue
            col = layer.color or colors[s.name]
            m = xmask(s.x)
            dropped += int((~m).sum()) if lt in {"mean_dot", "data_points"} else 0
            if lt == "mean_line":
                (ln,) = ax.plot(s.x[m], s.mean[m], color=col, linewidth=lw, linestyle=_ls(layer), zorder=3)
                ln.set_gid(ctx.gid("line", s.name))
            elif lt == "mean_dot":
                marker = layer.marker or (_MARKERS[i % len(_MARKERS)] if multi else "o")
                (ln,) = ax.plot(s.x[m], s.mean[m], linestyle="none", marker=marker,
                                markersize=(layer.size or style["point_size"]) * 1.35, color=col,
                                markeredgecolor=style["point_edge_color"], markeredgewidth=style["point_edge_width"],
                                zorder=5, clip_on=False)
                ln.set_gid(ctx.gid("points", s.name))
            elif lt in {"data_points", "data_points_jitter", "data_points_beeswarm"}:
                mp = xmask(s.x_points)
                sc = ax.scatter(s.x_points[mp], s.y_points[mp], s=(layer.size or style["point_size"]) ** 2,
                                c=col, alpha=layer.alpha if layer.alpha is not None else min(style["point_alpha"], 0.75),
                                marker=layer.marker or "o", edgecolors=style["point_edge_color"],
                                linewidths=style["point_edge_width"], zorder=4)
                sc.set_gid(ctx.gid("raw-points", s.name))
            elif lt in _ERR:
                kind = _ERR[lt]
                half = {"sem": s.sem, "sd": s.sd, "ci95": s.ci_half}.get(kind)
                if half is None:
                    lows = np.array([_error_bounds(s.y_points[s.x_points == xv], kind)[1] for xv in s.x])
                    highs = np.array([_error_bounds(s.y_points[s.x_points == xv], kind)[2] for xv in s.x])
                    centre = np.array([_error_bounds(s.y_points[s.x_points == xv], kind)[0] for xv in s.x])
                else:
                    centre, lows, highs = s.mean, s.mean - half, s.mean + half
                ok = m & np.isfinite(lows) & np.isfinite(highs) & (s.n > 1)
                if ok.any():
                    cont = ax.errorbar(
                        s.x[ok], centre[ok], yerr=[np.clip(centre[ok] - lows[ok], 0, None), np.clip(highs[ok] - centre[ok], 0, None)],
                        fmt="none", ecolor=col,
                        elinewidth=layer.linewidth if layer.linewidth is not None else style["errorbar_linewidth"],
                        capsize=layer.cap_size if layer.cap_size is not None else style["cap_size"],
                        capthick=style["errorbar_linewidth"], zorder=4,
                    )
                    for part in cont.lines[2]:
                        part.set_gid(ctx.gid("errorbar", s.name))
    if dropped:
        ctx.warnings.append(f"{dropped} point(s) with X <= 0 cannot be shown on a logarithmic X axis and were skipped.")

    for layer, res in curve_results:
        curve = res.extra.get("curve") or {}
        cx = np.asarray(curve.get("x", []), dtype=float)
        cy = np.asarray(curve.get("y", []), dtype=float)
        if cx.size == 0 or cx.size != cy.size:
            ctx.warnings.append(f"Result '{res.id}' has no fitted curve to draw.")
            continue
        sname = layer.series or res.extra.get("series")
        col = layer.color or colors.get(str(sname), style["axis_color"] if not multi else None) or style["axis_color"]
        if not multi and not layer.color:
            col = darken(colors[names[0]], 0.1)
        m = xmask(cx)
        lw = layer.linewidth if layer.linewidth is not None else style["line_width"]
        if layer.type == "curve_fit":
            (ln,) = ax.plot(cx[m], cy[m], color=col, linewidth=lw, linestyle=_ls(layer), zorder=3.5)
            ln.set_gid(ctx.gid("curve", sname or res.id))
        else:
            lo_key, hi_key = ("ci_low", "ci_high") if layer.type == "confidence_band" else ("pred_low", "pred_high")
            lo, hi = np.asarray(curve.get(lo_key, []), dtype=float), np.asarray(curve.get(hi_key, []), dtype=float)
            if lo.size == cx.size and hi.size == cx.size:
                band = ax.fill_between(cx[m], lo[m], hi[m], color=col, linewidth=0,
                                       alpha=layer.alpha if layer.alpha is not None else (0.18 if layer.type == "confidence_band" else 0.09),
                                       zorder=1.8)
                band.set_gid(ctx.gid("band", f"{layer.type}_{sname or res.id}"))
            else:
                ctx.warnings.append(f"Result '{res.id}' has no {layer.type.replace('_', ' ')} to draw.")
    if multi or spec.legend.show:
        for i, s in enumerate(data.series):
            marker = _MARKERS[i % len(_MARKERS)] if multi else "o"
            ctx.legend.append((Line2D([], [], color=colors[s.name], marker=marker, markersize=style["point_size"] * 1.2,
                                      linewidth=style["line_width"]), s.name))
    ctx._auto_xlabel = data.x_label
    ctx._auto_ylabel = data.y_label


# ====================================================================== survival
def _draw_survival(ctx: PanelContext, table: DataTable, layers: list[Layer], results: Mapping[str, StatResult]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    res = _result(results, spec.data.stat_result_id, ctx, "Survival curve")
    if res is None:
        from ..stats import run_analysis

        if not isinstance(table, MultipleVariablesTable):
            raise PlotError("Survival curves need a Multiple-variables table with time and event columns.")
        params = {k: v for k, v in {"time": spec.data.time, "event": spec.data.event,
                                    "group": spec.data.hue or spec.data.x}.items() if v}
        try:
            res = run_analysis(table, "survival", **params)
        except Exception as exc:  # noqa: BLE001
            raise PlotError(f"Could not compute survival curves: {exc}") from exc
    curves = res.extra.get("curves") or {}
    if not curves:
        raise PlotError("The survival result contains no curves.")
    names = [n for n in curves if not spec.data.groups or n in spec.data.groups]
    colors = resolve_colors(names, style.get("palette"), style.get("colors"), style.get("gradient"))
    types = [lay.type for lay in layers]
    for name in names:
        c = curves[name]
        t = np.asarray(c.get("time", []), dtype=float)
        s = np.asarray(c.get("survival", []), dtype=float) * 100.0
        col = colors[name]
        for layer in layers:
            if layer.series and layer.series != name:
                continue
            if layer.type == "km_curve":
                (ln,) = ax.step(t, s, where="post", color=layer.color or col,
                                linewidth=layer.linewidth if layer.linewidth is not None else style["line_width"],
                                linestyle=_ls(layer), zorder=3)
                ln.set_gid(ctx.gid("km-curve", name))
            elif layer.type == "km_ci":
                lo = np.asarray(c.get("ci_low", []), dtype=float) * 100.0
                hi = np.asarray(c.get("ci_high", []), dtype=float) * 100.0
                if lo.size == t.size and hi.size == t.size:
                    band = ax.fill_between(t, lo, hi, step="post", color=layer.color or col, linewidth=0,
                                           alpha=layer.alpha if layer.alpha is not None else 0.15, zorder=1.8)
                    band.set_gid(ctx.gid("km-ci", name))
            elif layer.type == "censor_marks":
                ct = np.asarray(c.get("censor_time", []), dtype=float)
                cs = np.asarray(c.get("censor_survival", []), dtype=float) * 100.0
                if ct.size:
                    (ln,) = ax.plot(ct, cs, linestyle="none", marker="|", markersize=(layer.size or style["point_size"]) * 1.8,
                                    markeredgewidth=style["line_width"] * 0.8, color=layer.color or col, zorder=4)
                    ln.set_gid(ctx.gid("censor", name))
            elif layer.type == "reference_line" and name == names[0]:
                _reference_line(ctx, layer)
        ctx.legend.append((Line2D([], [], color=col, linewidth=style["line_width"]), name))
    if "km_curve" not in types:
        ctx.warnings.append("Survival plot has no km_curve layer.")
    if len(names) == 1 and not spec.legend.show:
        ctx.legend.clear()
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ctx._auto_xlabel = spec.data.time or "Time"
    ctx._auto_ylabel = "Percent survival"
    ctx._survival_result = res


# ====================================================================== histogram
def _draw_histogram(ctx: PanelContext, sets: dict[str, np.ndarray], layers: list[Layer]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    sets = {k: v[np.isfinite(v)] for k, v in sets.items()}
    sets = {k: v for k, v in sets.items() if v.size}
    if not sets:
        raise PlotError("There are no values for the histogram.")
    names = list(sets)
    colors = resolve_colors(names, style.get("palette"), style.get("colors"), style.get("gradient"))
    allv = np.concatenate(list(sets.values()))
    for layer in layers:
        if layer.type == "histogram":
            edges = np.histogram_bin_edges(allv, bins=layer.bins or "auto")
            if len(edges) > 61 and not layer.bins:
                edges = np.histogram_bin_edges(allv, bins=60)
            for name in names:
                alpha = layer.alpha if layer.alpha is not None else (0.85 if len(names) == 1 else 0.55)
                _, _, patches = ax.hist(sets[name], bins=edges, color=layer.color or colors[name], alpha=alpha,
                                        edgecolor=style["background"], linewidth=0.5, zorder=2)
                for patch in patches[:1]:
                    patch.set_gid(ctx.gid("hist", name))
        elif layer.type == "density":
            from scipy.stats import gaussian_kde

            ax2 = ax
            for name in names:
                v = sets[name]
                if v.size > 2 and np.ptp(v) > 0:
                    grid = np.linspace(allv.min(), allv.max(), 200)
                    edges = np.histogram_bin_edges(allv, bins="auto")
                    scale = v.size * (edges[1] - edges[0])
                    (ln,) = ax2.plot(grid, gaussian_kde(v)(grid) * scale, color=darken(colors[name], 0.2),
                                     linewidth=layer.linewidth if layer.linewidth is not None else style["line_width"], zorder=3)
                    ln.set_gid(ctx.gid("density", name))
        elif layer.type == "reference_line":
            _reference_line(ctx, layer)
    if len(names) > 1 or spec.legend.show:
        for name in names:
            ctx.legend.append((Patch(facecolor=mcolors.to_rgba(colors[name], 0.6), edgecolor=colors[name]), name))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    d = spec.data
    single_col = d.y or (d.columns[0] if d.columns and len(d.columns) == 1 else None)
    ctx._auto_xlabel = names[0] if len(names) == 1 else (single_col or "Value")
    ctx._auto_ylabel = "Count"


# ====================================================================== heat map
def _draw_heatmap(ctx: PanelContext, table: DataTable, layers: list[Layer], results: Mapping[str, StatResult]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    res = _result(results, spec.data.stat_result_id, ctx, "Heat map")
    diverging = False
    if res is not None and res.extra.get("matrix"):
        m = res.extra["matrix"]
        mat = np.array([[np.nan if v is None else v for v in row] for row in m["r"]], dtype=float)
        rows = cols = [str(v) for v in m["labels"]]
        label, diverging = "Correlation (r)", True
    else:
        mat, rows, cols, label = prepare_matrix(spec, table)
    if mat.size == 0 or np.all(np.isnan(mat)):
        raise PlotError("There are no values for the heat map.")
    cmap_name = style.get("colormap") or "viridis"
    if diverging and cmap_name not in DIVERGING:
        cmap_name = "RdBu_r"
    cmap = mpl.colormaps[cmap_name].with_extremes(bad="#DDDDDD")
    if diverging:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    mesh = ax.pcolormesh(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=style["background"],
                         linewidth=0.6 if mat.size <= 900 else 0)
    mesh.set_gid(ctx.gid("heatmap"))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols)
    if len(rows) <= 60:
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels(rows)
    else:
        ax.set_yticks([])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    if max((len(c) for c in cols), default=0) * len(cols) > 40:
        for lab in ax.get_xticklabels():
            lab.set_rotation(40)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
    want_labels = any(lay.type == "value_labels" for lay in layers) or mat.size <= 36
    if want_labels and mat.size <= 400:
        norm = mcolors.Normalize(vmin, vmax)
        for (i, j), v in np.ndenumerate(mat):
            if np.isfinite(v):
                rgb = cmap(norm(v))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                txt = f"{v:.2f}" if diverging else (f"{v:.0f}" if float(v).is_integer() else f"{v:.3g}")
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=style["tick_size"] * 0.9,
                        color="#000000" if lum > 0.5 else "#FFFFFF")
    cbar = ctx.fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2)
    ctx._auto_xlabel = None
    ctx._auto_ylabel = None
    ctx._is_heatmap = True


# ====================================================================== Q-Q / residuals
def _draw_qq(ctx: PanelContext, table: DataTable, layers: list[Layer], results: Mapping[str, StatResult]) -> None:
    from scipy import stats as _st

    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    res = _result(results, spec.data.stat_result_id, ctx, "Q-Q plot")
    qq: dict[str, dict[str, Any]] = {}
    if res is not None and res.extra.get("qq"):
        qq = {k: v for k, v in res.extra["qq"].items() if not spec.data.groups or k in spec.data.groups}
    else:
        for name, vals in prepare_value_sets(spec, table).items():
            vals = vals[np.isfinite(vals)]
            if vals.size >= 3 and np.ptp(vals) > 0:
                (osm, osr), (slope, intercept, _r) = _st.probplot(vals, dist="norm")
                qq[name] = {"theoretical": osm, "ordered": osr, "slope": slope, "intercept": intercept}
    if not qq:
        raise PlotError("Q-Q plots need at least 3 non-identical values per group.")
    names = list(qq)
    colors = resolve_colors(names, style.get("palette"), style.get("colors"), style.get("gradient"))
    for layer in layers:
        for name in names:
            d = qq[name]
            th, od = np.asarray(d["theoretical"], dtype=float), np.asarray(d["ordered"], dtype=float)
            col = layer.color or colors[name]
            if layer.type == "qq_points":
                sc = ax.scatter(th, od, s=(layer.size or style["point_size"]) ** 2, c=col, alpha=style["point_alpha"],
                                edgecolors=style["point_edge_color"], linewidths=style["point_edge_width"], zorder=3)
                sc.set_gid(ctx.gid("qq-points", name))
            elif layer.type == "qq_line" and d.get("slope") is not None:
                xs = np.array([th.min(), th.max()])
                (ln,) = ax.plot(xs, d["intercept"] + d["slope"] * xs, color=darken(col, 0.2), linestyle=_ls(layer, "--"),
                                linewidth=layer.linewidth if layer.linewidth is not None else style["line_width"] * 0.8, zorder=2)
                ln.set_gid(ctx.gid("qq-line", name))
    if len(names) > 1 or spec.legend.show:
        for name in names:
            ctx.legend.append((Line2D([], [], color=colors[name], marker="o", linestyle="none", markersize=style["point_size"]), name))
    ctx._auto_xlabel = "Theoretical normal quantiles"
    ctx._auto_ylabel = "Ordered data"


def _draw_residuals(ctx: PanelContext, layers: list[Layer], results: Mapping[str, StatResult]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    rid = spec.data.stat_result_id or next((lay.stat_result_id for lay in layers if lay.stat_result_id), None)
    if not rid:
        raise PlotError("A residual plot needs a fit result. Run a regression first and choose its result.")
    res = _result(results, rid, ctx, "Residual plot")
    if res is None or not res.extra.get("residuals"):
        raise PlotError(f"Result '{rid}' has no residuals to plot.")
    r = res.extra["residuals"]
    resid = np.asarray(r.get("residual", []), dtype=float)
    use_fitted = not r.get("x") or len(r.get("x", [])) != resid.size
    xs = np.asarray(r.get("fitted") if use_fitted else r.get("x"), dtype=float)
    if xs.size != resid.size or resid.size == 0:
        raise PlotError(f"Result '{rid}' has malformed residuals.")
    color = get_palette(style.get("palette"), 1)[0]
    for layer in layers:
        if layer.type == "residual_points":
            sc = ax.scatter(xs, resid, s=(layer.size or style["point_size"]) ** 2, c=layer.color or color,
                            alpha=style["point_alpha"], edgecolors=style["point_edge_color"],
                            linewidths=style["point_edge_width"], zorder=3)
            sc.set_gid(ctx.gid("residuals"))
        elif layer.type == "reference_line":
            _reference_line(ctx, layer)
    ctx._auto_xlabel = "Fitted value" if use_fitted else "X"
    ctx._auto_ylabel = "Residual"


# ====================================================================== axes, annotations, legend
def _format_axis(axis: Any, a: AxisSpec, forced_log: bool = False) -> None:
    scale = "log10" if forced_log else a.scale
    which = axis.axis_name
    ax = axis.axes
    setter = ax.set_xscale if which == "x" else ax.set_yscale
    if scale == "log10":
        setter("log", base=10)
    elif scale == "log2":
        setter("log", base=2)
    elif scale == "symlog":
        setter("symlog")
    if a.ticks is not None:
        axis.set_ticks(a.ticks)
        if a.tick_labels is not None:
            axis.set_ticklabels(a.tick_labels)
    fmt = a.tick_format
    if fmt == "percent":
        axis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    elif fmt == "sci":
        sf = ticker.ScalarFormatter(useMathText=True)
        sf.set_powerlimits((0, 0))
        axis.set_major_formatter(sf)
    elif fmt == "log_exp":
        axis.set_major_formatter(ticker.LogFormatterMathtext())
    elif fmt:
        try:
            fmt % 1.0
            axis.set_major_formatter(ticker.FormatStrFormatter(fmt))
        except (TypeError, ValueError):
            pass
    if a.tick_rotation is not None:
        for lab in axis.get_ticklabels():
            lab.set_rotation(a.tick_rotation)
            if which == "x" and a.tick_rotation not in (0, 90, -90):
                lab.set_ha("right" if a.tick_rotation > 0 else "left")
                lab.set_rotation_mode("anchor")
    if a.grid is not None:
        ax.grid(a.grid, axis=which)
    if not a.show:
        axis.set_visible(False)
        ax.spines["bottom" if which == "x" else "left"].set_visible(False)


def _apply_axes(ctx: PanelContext) -> None:
    ax, spec, style = ctx.ax, ctx.spec, ctx.style
    xs, ys = spec.axes.x, spec.axes.y
    _format_axis(ax.xaxis, xs, forced_log=getattr(ctx, "_force_log_x", False))
    _format_axis(ax.yaxis, ys)
    if xs.min is not None or xs.max is not None:
        ax.set_xlim(left=xs.min, right=xs.max)
    if ys.min is not None or ys.max is not None:
        ax.set_ylim(bottom=ys.min, top=ys.max)
    is_heat = getattr(ctx, "_is_heatmap", False)
    if xs.invert:
        ax.invert_xaxis()
    if ys.invert and not is_heat:
        ax.invert_yaxis()
    xlabel = xs.label if xs.label is not None else getattr(ctx, "_auto_xlabel", None)
    ylabel = ys.label if ys.label is not None else getattr(ctx, "_auto_ylabel", None)
    if xlabel:
        ax.set_xlabel(xlabel).set_gid(ctx.gid("x-label"))
    if ylabel:
        ax.set_ylabel(ylabel).set_gid(ctx.gid("y-label"))
    if spec.axes.zero_line:
        ax.axhline(0.0, color=style["axis_color"], linewidth=style["axis_linewidth"] * 0.8, zorder=1.5).set_gid(ctx.gid("zero-line"))
    if style["grid"] and not is_heat and ctx.spec.plot_type in CATEGORICAL | {"histogram"}:
        ax.grid(False, axis="x")
    ax.tick_params(axis="both", which="both", labelsize=style["tick_size"])
    if ctx.spec.plot_type in CATEGORICAL and style["tick_length"]:
        ax.tick_params(axis="x", which="both", length=0)
    # headroom so points/caps at the top are not clipped by the axis frame
    if ctx.spec.plot_type in CATEGORICAL and ys.max is None and ctx.tops and ax.get_yscale() == "linear":
        lo, hi = ax.get_ylim()
        top = max(ctx.tops.values())
        if top > hi - 0.04 * (hi - lo):
            ax.set_ylim(lo, top + 0.06 * (hi - lo))


def _bracket_positions(ctx: PanelContext, comp: Any) -> tuple[float, float] | None:
    cat = ctx.cat
    if cat is None:
        return None
    pos = ctx.positions

    def locate(cell: dict[str, str] | None, name: str) -> float | None:
        if cell:
            c = cell.get(cat.cat_key)
            h = cell.get(cat.hue_key) if cat.hue_key else None
            if c is not None and (c, h) in pos:
                return pos[(c, h)]
            if c is not None and h is None and c in cat.categories:
                return float(cat.categories.index(c))
            if c is None and h is not None and cat.hues and len(cat.categories) == 1:
                return pos.get((cat.categories[0], h))
            return None
        if cat.hues is None and (name, None) in pos:
            return pos[(name, None)]
        if cat.hues and len(cat.categories) == 1 and (cat.categories[0], name) in pos:
            return pos[(cat.categories[0], name)]
        if cat.hues and name in cat.categories:
            return float(cat.categories.index(name))
        return None

    xa, xb = locate(comp.cell_a, comp.group_a), locate(comp.cell_b, comp.group_b)
    if xa is None or xb is None or abs(xa - xb) < 1e-9:
        return None
    return xa, xb


def _draw_annotations(ctx: PanelContext, results: Mapping[str, StatResult]) -> None:
    ax, style, spec = ctx.ax, ctx.style, ctx.spec
    for ann in spec.annotations:
        res = _result(results, ann.stat_result_id, ctx, "Annotation")
        if res is None:
            continue
        size = ann.text_size or style["annotation_size"]
        if ann.kind == "summary_text":
            horiz, vert = ann.position.split()[1], ann.position.split()[0]
            if vert == "top" and ax.get_yscale() == "linear" and spec.axes.y.max is None and not getattr(ctx, "_is_heatmap", False):
                lo, hi = ax.get_ylim()  # headroom so the text does not sit on the data
                ax.set_ylim(lo, hi + 0.12 * (hi - lo))
            txt = ax.text(0.02 if horiz == "left" else 0.98, 0.98 if vert == "top" else 0.02, _br.summary_text(res),
                          transform=ax.transAxes, ha=horiz, va=vert, fontsize=size * 0.9, zorder=7)
            txt.set_gid(ctx.gid("stat-summary", res.id))
            continue
        if ctx.cat is None:
            ctx.warnings.append("Comparison brackets are only drawn on categorical plots; use kind='summary_text' here.")
            continue
        comps = _br.select_comparisons(res, ann)
        if not comps:
            if not res.comparisons:
                ctx.warnings.append(f"Result '{res.id}' ({res.test_name}) has no pairwise comparisons to draw.")
            continue
        brs: list[_br.Bracket] = []
        skipped = 0
        for comp in comps:
            xy = _bracket_positions(ctx, comp)
            if xy is None:
                skipped += 1
                continue
            brs.append(_br.Bracket(xy[0], xy[1], _br.comparison_label(comp, ann.format, ann.star_style),
                                   comp.significant, comp.id))
        if skipped:
            ctx.warnings.append(f"{skipped} comparison(s) of result '{res.id}' do not match groups on this plot and were not drawn.")
        lw = ann.line_width if ann.line_width is not None else style["bracket_linewidth"]
        done = False
        cat = ctx.cat
        if ann.engine == "statannotations" and cat.hues is None and brs:
            idx = {round(ctx.positions[(c, None)], 6): c for c in cat.categories}
            pairs = [(idx[round(b.x1, 6)], idx[round(b.x2, 6)]) for b in brs]
            done = _br.draw_with_statannotations(
                ax, cat.categories, {c: cat.values[(c, None)] for c in cat.categories}, pairs,
                [b.label for b in brs], line_width=lw, text_size=size, color=style["text_color"],
            )
            if not done:
                ctx.warnings.append("statannotations could not draw these brackets; the built-in engine was used.")
        if not done:
            _br.draw_brackets(ax, brs, ctx.tops, line_width=lw, text_size=size, color=style["text_color"],
                              tip=ann.tip_length if ann.tip_length is not None else 0.015,
                              step=ann.step or 0.075, gid_prefix=f"{ctx.prefix}{'' if len(spec.annotations) == 1 else res.id + '-'}")
        ctx.brackets_drawn.extend(brs)


def _draw_texts(ctx: PanelContext) -> None:
    for i, t in enumerate(ctx.spec.texts, start=1):
        txt = ctx.ax.text(t.x, t.y, t.text, transform=ctx.ax.transAxes if t.coords == "axes" else ctx.ax.transData,
                          ha=t.ha, va=t.va, fontsize=t.size or ctx.style["font_size"], color=t.color or ctx.style["text_color"],
                          fontweight=t.weight, fontstyle=t.style, zorder=7)
        txt.set_gid(ctx.gid("text", i))


def _draw_legend(ctx: PanelContext) -> None:
    spec = ctx.spec
    if spec.legend.show is False or not ctx.legend:
        return
    handles, labels = zip(*ctx.legend)
    pos = spec.legend.position
    kw: dict[str, Any] = {"title": spec.legend.title or ctx.legend_title, "frameon": spec.legend.frame,
                          "ncol": spec.legend.ncol}
    if pos == "outside right":
        kw.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
    elif pos == "top":
        kw.update(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=max(spec.legend.ncol, len(labels)))
    elif pos == "bottom":
        kw.update(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=max(spec.legend.ncol, len(labels)))
    elif pos == "best" and spec.plot_type == "qq":
        kw.update(loc="lower right")
    elif pos == "best" and spec.plot_type == "stacked_bar":
        kw.update(loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        kw.update(loc=pos)
    leg = ctx.ax.legend(handles, labels, **kw)
    leg.set_gid(ctx.gid("legend"))
    ctx.legend_artist = leg
