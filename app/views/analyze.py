"""Page 2: Analyze. Test recommendation, analysis menu filtered by table type, results panel."""

from __future__ import annotations

import streamlit as st

from torusplot.stats import StatsError, analyses_for, run_analysis
from torusplot.stats.recommend import recommend_test

from .. import state
from ..ui_helpers import badge, param_widget, show_result


def _run(table, key: str, params: dict, allow_approximate: bool) -> None:
    proj = state.project()
    try:
        result = run_analysis(table, key, allow_approximate=allow_approximate, **params)
    except StatsError as exc:
        st.error(f"The analysis could not be run: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected problem in the analysis: {exc}")
        return
    proj.add_result(result)
    st.session_state["last_result"] = result.id
    state.log(f"Ran {result.test_name} on '{table.name}' (result {result.id}).")
    st.success(f"Done: {result.test_name}. See the results below.")


def _recommend(table, allow_approximate: bool) -> None:
    st.subheader("Not sure which test? Ask for a recommendation")
    c1, c2, c3 = st.columns(3)
    paired = None
    control = None
    if table.kind in {"column", "grouped"}:
        paired = c1.checkbox(
            "Same subjects measured repeatedly (paired)", key="rec_paired",
            help="Tick this if each row (Column table) or each replicate column (Grouped table) is the "
                 "same animal/sample/person measured under every condition.",
        )
    if table.kind == "column" and len(table.group_names) >= 3:
        choice = c2.selectbox("Control group (optional)", ["(none)", *table.group_names], key="rec_control",
                              help="If you only want to compare every group with one control, pick it here.")
        control = None if choice == "(none)" else choice
    if c3.button("Recommend a test", key="recommend", type="primary", width="stretch"):
        try:
            st.session_state["recommendation"] = recommend_test(table, paired=paired or None, control=control)
        except StatsError as exc:
            st.session_state.pop("recommendation", None)
            st.error(str(exc))
    rec = st.session_state.get("recommendation")
    if rec is None:
        return
    st.success(f"**Recommended: {rec.label}**")
    st.write(rec.reason)
    for i, step in enumerate(rec.steps, start=1):
        st.markdown(f"{i}. {step}")
    for check in rec.checks:
        target = f" *({check.target})*" if check.target else ""
        st.markdown(badge(check.status) + f" **{check.name}**{target}: {check.detail}", unsafe_allow_html=True)
    for caution in rec.cautions:
        st.warning(caution)
    if rec.alternatives:
        with st.expander("Alternatives"):
            for alt in rec.alternatives:
                st.markdown(f"- **{alt.label}**: {alt.when}")
    if st.button(f"Run: {rec.label}", key="run_recommended"):
        _run(table, rec.analysis, dict(rec.params), allow_approximate)


def _menu(table, allow_approximate: bool) -> None:
    st.subheader("Choose an analysis")
    infos = analyses_for(table.kind)
    if not infos:
        st.info("No analyses are available for this table type.")
        return
    categories = list(dict.fromkeys(i.category for i in infos))
    category = st.selectbox("Category", categories, key="an_category",
                            help="Only analyses that are valid for this table type are listed.")
    options = [i for i in infos if i.category == category]
    info = st.selectbox("Analysis", options, format_func=lambda i: i.label, key="an_choice")
    if info.description:
        st.caption(info.description)
    params: dict = {}
    if info.params:
        cols = st.columns(2)
        for n, spec in enumerate(info.params):
            with cols[n % 2]:
                params[spec.name] = param_widget(spec, table, key=f"p_{info.key}_{spec.name}")
    if info.citation:
        st.caption(f"Method reference: {info.citation}")
    if st.button(f"Run {info.label}", key="run_manual", type="primary"):
        _run(table, info.key, params, allow_approximate)


def _exclusion_offer(table, result) -> None:
    flagged = (result.extra or {}).get("flagged") or []
    if not flagged or result.table != table.name:
        return
    st.markdown("**Flagged values**")
    st.caption("Outliers are only flagged. Nothing is removed unless you decide to exclude it, and every "
               "exclusion is written into the statistics report and the methods paragraph.")
    labels = [f"{f.get('group')} · row {int(f.get('row', 0)) + 1} · value {f.get('value')}" for f in flagged]
    chosen = st.multiselect("Values to exclude from analysis", labels, key=f"excl_{result.id}")
    confirm = st.checkbox("I have a scientific reason to exclude these values and will report it.", key=f"excl_ok_{result.id}")
    if st.button("Exclude selected values", key=f"excl_btn_{result.id}", disabled=not (chosen and confirm)):
        cells = [(int(f["row"]), str(f["group"])) for f, lab in zip(flagged, labels) if lab in chosen]
        table.exclude(cells)
        state.log(f"User excluded {len(cells)} flagged value(s) from '{table.name}' after {result.test_name} ({result.id}).")
        st.success("Excluded. Re-run your analysis; the exclusion will be reported with every result.")


def _results(table) -> None:
    proj = state.project()
    results = [r for r in proj.results.values() if getattr(r, "table", None) == table.name]
    st.subheader(f"Results for this table ({len(results)})")
    if not results:
        st.caption("Run an analysis to see results here.")
        return
    ids = [r.id for r in reversed(results)]
    last = st.session_state.get("last_result")
    index = ids.index(last) if last in ids else 0
    rid = st.selectbox("Result", ids, index=index, key="result_pick",
                       format_func=lambda i: f"{proj.results[i].test_name}  ·  {i}")
    result = proj.results[rid]
    show_result(result)
    _exclusion_offer(table, result)
    if st.button("Delete this result", key="delete_result"):
        del proj.results[rid]
        st.rerun()


def _power() -> None:
    """Sample-size / power calculator (works without any data)."""
    from torusplot.stats.power import power_anova, power_ttest

    with st.expander("Power and sample size calculator"):
        st.caption("Plan an experiment: how many samples per group do you need to detect an effect of a given size? "
                   "Fill in two of the three quantities and leave the one to solve for empty.")
        design = st.selectbox(
            "Design", ["unpaired", "paired", "one-sample", "anova"], key="pw_design",
            format_func=lambda d: {"unpaired": "Two independent groups (t test)", "paired": "Paired measurements (t test)",
                                   "one-sample": "One group vs a fixed value (t test)", "anova": "Three or more groups (one-way ANOVA)"}[d],
        )
        c1, c2, c3, c4 = st.columns(4)
        label = "Effect size (Cohen's f)" if design == "anova" else "Effect size (Cohen's d)"
        tip = ("Cohen's f: 0.10 small, 0.25 medium, 0.40 large." if design == "anova"
               else "Cohen's d = expected difference / standard deviation. 0.2 small, 0.5 medium, 0.8 large.")
        raw = {
            "effect_size": c1.text_input(label, value="0.8", key="pw_es", help=tip),
            "n": c2.text_input("n per group", value="", key="pw_n", help="Leave empty to solve for the sample size."),
            "power": c3.text_input("Power", value="0.8", key="pw_power", help="Chance of detecting the effect if it is real. 0.8 is the usual target."),
        }
        alpha = c4.number_input("Alpha", value=0.05, min_value=0.0001, max_value=0.5, format="%g", key="pw_alpha",
                                help="Significance threshold, usually 0.05.")
        k = int(st.number_input("Number of groups", 3, 50, value=3, key="pw_k")) if design == "anova" else None
        if st.button("Calculate", key="pw_go"):
            try:
                values = {name: (float(text) if text.strip() else None) for name, text in raw.items()}
                if design == "anova":
                    result = power_anova(k_groups=k, alpha=float(alpha), **values)
                else:
                    result = power_ttest(design=design, alpha=float(alpha), **values)
            except ValueError as exc:  # StatsError or a number that could not be read
                st.error(f"Could not calculate: {exc}")
                return
            st.success(result.interpretation)
            for w in result.warnings:
                st.warning(w)


def render() -> None:
    st.header("2 · Analyze")
    st.caption("All statistics are computed by TorusPlot's own engine. AI features can never change these numbers.")
    table = state.table_picker(key="an_table_picker")
    if table is None:
        _power()
        return
    allow_approximate = False
    if table.approximate:
        st.error("APPROXIMATE: for reference only. This table holds values digitized from an image.")
        allow_approximate = st.checkbox(
            "I understand these are estimates read off an image. Allow statistics anyway.", key="allow_approx",
        )
    _recommend(table, allow_approximate)
    st.divider()
    _menu(table, allow_approximate)
    st.divider()
    _results(table)
    st.divider()
    _power()
