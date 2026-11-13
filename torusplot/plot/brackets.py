"""Significance brackets.

Brackets are always built from a computed :class:`~torusplot.stats.result.StatResult`:
the label text (stars or P value) is derived here from ``Comparison.p_adjusted`` /
``p_value``; nothing in a FigureSpec can supply its own number.

Two engines:

* ``torus`` (default): TorusPlot's own stacking algorithm. Works on any axes where the
  compared groups have x positions (single-factor, grouped/factorial, log scale).
* ``statannotations``: delegates drawing to the *statannotations* library (MIT) in its
  custom-annotation mode for simple one-factor categorical plots; the label strings are
  still produced here from the StatResult, so both engines show identical text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spec.figure import StatAnnotation
from ..stats.result import Comparison, StatResult, format_p, p_to_stars


@dataclass
class Bracket:
    x1: float
    x2: float
    label: str
    significant: bool | None = None
    comparison_id: str = ""


def comparison_label(comp: Comparison, fmt: str = "stars", star_style: str = "gp") -> str:
    """Text for a bracket, derived only from the comparison's computed p value."""
    p = comp.p_for_display
    stars = p_to_stars(p, style=star_style)
    if fmt == "stars":
        return stars
    ptxt = format_p(p)
    if fmt == "p_short":
        return ptxt
    full = f"P < {ptxt[1:]}" if ptxt.startswith("<") else f"P = {ptxt}"
    if fmt == "p":
        return full
    return f"{stars} ({full})"


def select_comparisons(result: StatResult, ann: StatAnnotation) -> list[Comparison]:
    """Comparisons of ``result`` selected by a StatAnnotation."""
    comps = list(result.comparisons)
    if isinstance(ann.comparisons, list):
        wanted = set(ann.comparisons)
        comps = [c for c in comps if c.id in wanted]
    elif ann.comparisons == "significant":
        comps = [c for c in comps if c.significant]
    if ann.pairs:
        wanted_pairs = [frozenset(p) for p in ann.pairs]
        comps = [c for c in comps if frozenset((c.group_a, c.group_b)) in wanted_pairs]
    if not ann.show_ns:
        comps = [c for c in comps if c.significant]
    return comps


def summary_text(result: StatResult) -> str:
    """One-line description of an omnibus result, e.g. 'One-way ANOVA: F(2, 27) = 4.85, P = 0.0159'."""
    parts = [result.test_name]
    if result.statistic is not None and result.statistic_name and not math.isnan(result.statistic):
        if result.df is not None and result.df2 is not None:
            dfs = f"({_num(result.df)}, {_num(result.df2)})"
        elif result.df is not None:
            dfs = f"({_num(result.df)})"
        else:
            dfs = ""
        parts.append(f"{result.statistic_name}{dfs} = {result.statistic:.4g}")
    p = result.p_adjusted if result.p_adjusted is not None else result.p_value
    ptxt = format_p(p)
    if ptxt != "n/a":
        parts.append(f"P < {ptxt[1:]}" if ptxt.startswith("<") else f"P = {ptxt}")
    return parts[0] + (": " + ", ".join(parts[1:]) if len(parts) > 1 else "")


def _num(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"


def draw_brackets(
    ax: Any,
    brackets: list[Bracket],
    tops: dict[float, float],
    *,
    line_width: float = 0.8,
    text_size: float = 8.0,
    color: str = "#000000",
    tip: float = 0.015,
    step: float = 0.075,
    gid_prefix: str = "",
) -> float | None:
    """Draw stacked brackets above the data; returns the new top (data units) or None.

    ``tops`` maps an x position to the highest drawn y at that position (bars, error
    bars, points). Narrow brackets are placed first, so wide ones stack above.
    """
    if not brackets:
        return None
    log = ax.get_yscale() == "log"

    def fwd(y: float) -> float:
        return math.log10(y) if log else y

    def inv(t: float) -> float:
        return 10.0**t if log else t

    y0, y1 = ax.get_ylim()
    lo, hi = fwd(min(y0, y1)), fwd(max(y0, y1))
    span = hi - lo if hi > lo else 1.0
    fig_h = ax.figure.get_figheight() * ax.get_position().height
    text_h = (text_size / 72.0) / max(fig_h, 0.1) * span
    xs = sorted(tops)
    level: dict[float, float] = {x: fwd(tops[x]) if (not log or tops[x] > 0) else lo for x in xs}
    order = sorted(range(len(brackets)), key=lambda i: (abs(brackets[i].x2 - brackets[i].x1), min(brackets[i].x1, brackets[i].x2)))
    top_overall = max(level.values()) if level else hi
    eps = 1e-9
    for n, i in enumerate(order, start=1):
        b = brackets[i]
        xa, xb = sorted((b.x1, b.x2))
        covered = [x for x in xs if xa - eps <= x <= xb + eps]
        base = max((level[x] for x in covered), default=top_overall)
        y = base + step * span * 0.55
        tip_len = tip * span
        yy = [inv(y - tip_len), inv(y), inv(y), inv(y - tip_len)]
        (line,) = ax.plot([xa, xa, xb, xb], yy, color=color, linewidth=line_width, solid_capstyle="butt",
                          solid_joinstyle="miter", clip_on=False, zorder=6)
        line.set_gid(f"{gid_prefix}bracket-{n}")
        stars_only = set(b.label) <= {"*"}
        # asterisks sit high in their line box: nudge them down so they hug the bracket
        ty = y - (0.32 * text_h if stars_only else -0.12 * text_h)
        txt = ax.text((xa + xb) / 2.0, inv(ty), b.label, ha="center", va="bottom", fontsize=text_size,
                      color=color, clip_on=False, zorder=6)
        txt.set_gid(f"{gid_prefix}bracket-label-{n}")
        new_top = y + (0.75 if stars_only else 1.25) * text_h
        for x in covered:
            level[x] = new_top
        top_overall = max(top_overall, new_top)
    upper = top_overall + 0.03 * span
    if upper > hi:
        if y0 <= y1:
            ax.set_ylim(y0, inv(upper))
        else:
            ax.set_ylim(inv(upper), y1)
    return inv(top_overall)


def draw_with_statannotations(
    ax: Any,
    categories: list[str],
    values: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    labels: list[str],
    *,
    line_width: float = 0.8,
    text_size: float = 8.0,
    color: str = "#000000",
) -> bool:
    """Draw brackets with the statannotations library (custom-annotation mode).

    Returns False (drawing nothing) if the library cannot handle the request, so the
    caller can fall back to the built-in engine.
    """
    if not pairs:
        return True
    try:
        import warnings

        import pandas as pd
        from statannotations.Annotator import Annotator

        frame = pd.DataFrame(
            {
                "group": np.concatenate([[c] * len(values[c]) for c in categories]),
                "value": np.concatenate([np.asarray(values[c], dtype=float) for c in categories]),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotator = Annotator(ax, pairs, data=frame, x="group", y="value", order=categories,
                                  plot="stripplot", verbose=False)
            annotator.configure(loc="inside", line_width=line_width, fontsize=text_size, color=color,
                                verbose=False)
            annotator.set_custom_annotations(labels)
            annotator.annotate()
        return True
    except Exception:  # noqa: BLE001 - any failure -> fall back to the built-in engine
        return False
