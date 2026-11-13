"""Reusable display pieces: figure previews, statistics results, badges, parameter widgets."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from torusplot.data import DataTable
from torusplot.export import figure_bytes
from torusplot.plot import PlotError, figure_warnings, render_figure
from torusplot.stats import ParamSpec, StatResult, format_p

from . import state

BADGE = {
    "pass": ("PASS", "#1B7F4B", "#E3F4EA"),
    "warn": ("CHECK", "#8A5A00", "#FFF1D0"),
    "fail": ("FAIL", "#A3261C", "#FBE3E0"),
    "info": ("INFO", "#2B4A8B", "#E4EAF8"),
}


def badge(status: str, text: str = "") -> str:
    label, fg, bg = BADGE.get(status, BADGE["info"])
    extra = f" {text}" if text else ""
    return (f"<span style='background:{bg};color:{fg};border-radius:6px;padding:1px 8px;"
            f"font-size:0.78rem;font-weight:700;letter-spacing:0.03em'>{label}</span>{extra}")


def preview_figure(spec: Any, dpi: int = 150, caption: str | None = None) -> Any:
    """Render a FigureSpec and show it; returns the Figure or None on error."""
    proj = state.project()
    try:
        fig = render_figure(spec, proj.tables, proj.results)
    except PlotError as exc:
        st.warning(f"This figure cannot be drawn yet: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - never crash the page on a drawing problem
        st.error(f"Unexpected drawing problem: {exc}")
        return None
    st.image(figure_bytes(fig, "png", dpi=dpi), caption=caption)
    for warning in figure_warnings(fig):
        st.caption(f"Note: {warning}")
    return fig


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(f"{value:.5g}")
    return value


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{k: _fmt(v) for k, v in row.items()} for row in rows])


def show_result(result: StatResult) -> None:
    """Results panel: numbers, plain-English reading, assumption badges, 'Why this test?'."""
    st.subheader(result.test_name)
    st.caption(f"Result ID `{result.id}`" + (f" · table: {result.table}" if result.table else ""))
    if result.approximate_data:
        st.error("APPROXIMATE DATA: for reference only. This analysis used digitized values.")
    if result.interpretation:
        st.success(result.interpretation) if result.significant else st.info(result.interpretation)

    cols = st.columns(4)
    if result.statistic is not None and not math.isnan(result.statistic):
        df = ""
        if result.df is not None and result.df2 is not None:
            df = f" ({result.df:.4g}, {result.df2:.4g})"
        elif result.df is not None:
            df = f" ({result.df:.4g})"
        cols[0].metric(f"{result.statistic_name or 'Statistic'}{df}", f"{result.statistic:.4g}")
    if result.p_value is not None:
        cols[1].metric("P value", format_p(result.p_value), help="Probability of data at least this extreme if there were truly no effect.")
    if result.p_adjusted is not None:
        cols[2].metric("Adjusted P", format_p(result.p_adjusted))
    if result.effect_sizes and result.effect_sizes[0].value is not None:
        es = result.effect_sizes[0]
        cols[3].metric(es.name, f"{es.value:.3g}", help="How big the effect is, independent of sample size.")

    if result.assumptions:
        st.markdown("**Assumption checks**")
        for a in result.assumptions:
            target = f" *({a.target})*" if a.target else ""
            st.markdown(badge(a.status) + f" **{a.name}**{target}: {a.detail}", unsafe_allow_html=True)
    if result.why_chosen:
        with st.expander("Why this test?"):
            st.write(result.why_chosen)
            if result.citations:
                st.caption("Method references: " + "; ".join(result.citations))
    for w in result.warnings:
        st.warning(w)
    if result.comparisons:
        title = "Multiple comparisons" + (f" ({result.correction})" if result.correction else "")
        st.markdown(f"**{title}**")
        if result.comparison_family:
            st.caption(f"Family: {result.comparison_family}")
        st.dataframe(_frame([
            {"ID": c.id, "Comparison": f"{c.group_a} vs {c.group_b}" + (f" [{c.label}]" if c.label else ""),
             "Difference": c.estimate, "CI low": c.ci_low, "CI high": c.ci_high,
             "P": format_p(c.p_value) if c.p_value is not None else "",
             "Adjusted P": format_p(c.p_adjusted) if c.p_adjusted is not None else "",
             "Significant": "yes" if c.significant else "no"} for c in result.comparisons]),
            hide_index=True, width="stretch")
    if result.anova_table:
        st.markdown("**ANOVA table**")
        st.dataframe(_frame(result.anova_table), hide_index=True, width="stretch")
    if result.parameters:
        st.markdown("**Estimates**")
        st.dataframe(_frame([p.model_dump(exclude={"ci_level", "unit_note"}) for p in result.parameters]),
                     hide_index=True, width="stretch")
    if result.effect_sizes:
        st.markdown("**Effect sizes**")
        st.dataframe(_frame([e.model_dump(exclude={"ci_level", "ci_method"}) for e in result.effect_sizes]),
                     hide_index=True, width="stretch")
    if result.descriptives:
        st.markdown("**Descriptive statistics**")
        st.dataframe(_frame(result.descriptives), hide_index=True, width="stretch")
    for note in result.data_notes:
        st.caption(f"Data note: {note}")
    if result.methods_text:
        with st.expander("Methods sentence"):
            st.code(result.methods_text, language=None)


def param_widget(spec: ParamSpec, table: DataTable, key: str) -> Any:
    """One Streamlit widget for one analysis option, with its beginner tooltip."""
    kind, help_text = spec.kind, spec.help or None
    auto = "(automatic)"
    if kind == "bool":
        return st.checkbox(spec.label, value=bool(spec.default), key=key, help=help_text)
    if kind == "int":
        return int(st.number_input(spec.label, value=int(spec.default or 0), step=1, key=key, help=help_text))
    if kind == "float":
        if spec.default is None:
            raw = st.text_input(spec.label, value="", key=key, help=help_text, placeholder="leave empty for automatic")
            try:
                return float(raw) if raw.strip() else None
            except ValueError:
                st.warning(f"'{raw}' is not a number.")
                return None
        return float(st.number_input(spec.label, value=float(spec.default), key=key, help=help_text, format="%g"))
    if kind == "choice":
        choices = list(spec.choices)
        index = choices.index(spec.default) if spec.default in choices else 0
        return st.selectbox(spec.label, choices, index=index, key=key, help=help_text)
    if kind in {"group", "column"}:
        names = state.group_names(table) if kind == "group" else [str(c) for c in table.data.columns]
        options = [auto, *names]
        index = options.index(spec.default) if spec.default in options else 0
        value = st.selectbox(spec.label, options, index=index, key=key, help=help_text)
        return None if value == auto else value
    if kind in {"groups", "columns"}:
        names = state.group_names(table) if kind == "groups" else [str(c) for c in table.data.columns]
        value = st.multiselect(spec.label, names, default=[], key=key, help=(help_text or "") + " Leave empty for automatic.")
        return value or None
    raw = st.text_input(spec.label, value="" if spec.default is None else str(spec.default), key=key, help=help_text)
    return raw if raw.strip() else spec.default
