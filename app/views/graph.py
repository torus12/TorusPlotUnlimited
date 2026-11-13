"""Page 3: Graph. Point-and-click figure building with a live preview."""

from __future__ import annotations

import streamlit as st

from torusplot.spec import PLOT_TYPES, FigureSpec, Layer, StatAnnotation, plot_types_for

from .. import state
from ..ui_helpers import preview_figure

ERROR_CHOICES = {"SEM": "sem", "SD": "sd", "95% CI": "ci95", "Range": "range", "IQR": "iqr", "None": None}
POINT_CHOICES = {"Jittered": "jitter", "Beeswarm (no overlap)": "beeswarm", "In a line": "plain", "Hidden": None}
CATEGORICAL = {"bar", "grouped_bar", "scatter", "box", "violin", "box_violin", "beeswarm", "superplot", "estimation", "paired"}
SCALES = ["linear", "log10", "log2"]


def _unique_name(base: str) -> str:
    figures = state.project().figures
    name, i = base, 2
    while name in figures:
        name = f"{base} {i}"
        i += 1
    return name


def _new_figure(table) -> None:
    types = plot_types_for(table.kind)
    c1, c2, c3 = st.columns([2, 2, 1])
    ptype = c1.selectbox("Graph type", types, format_func=lambda t: PLOT_TYPES[t]["label"], key="new_plot_type",
                         help="Only graph types that suit this table type are listed.")
    c1.caption(PLOT_TYPES[ptype]["description"])
    name = c2.text_input("Figure name", value=_unique_name("Figure 1"), key="new_fig_name")
    if c3.button("Create figure", key="create_figure", type="primary", width="stretch"):
        data: dict = {"table": table.name}
        if table.kind == "multiple_variables":
            cats, nums = table.categorical_columns, table.numeric_columns
            if ptype == "survival":
                low = {c.lower(): c for c in nums}
                data["time"] = next((v for k, v in low.items() if "time" in k or "day" in k), nums[0] if nums else None)
                data["event"] = next((v for k, v in low.items() if "event" in k or "status" in k), nums[1] if len(nums) > 1 else None)
                data["hue"] = cats[0] if cats else None
            elif ptype == "xy":
                data["x"], data["y"] = (nums + [None, None])[:2]
                data["hue"] = cats[0] if cats else None
            else:
                data["x"] = cats[0] if cats else None
                data["y"] = nums[0] if nums else None
                if ptype == "superplot" and len(cats) > 1:
                    data["replicate"] = cats[1]
        spec = FigureSpec(name=name.strip() or _unique_name("Figure"), plot_type=ptype, data=data)
        spec = _attach_latest_results(spec, table)
        state.project().figures[spec.name] = spec
        state.set_current_figure(spec.name)
        state.log(f"Created figure '{spec.name}' ({ptype}) from table '{table.name}'.")
        st.rerun()


def _attach_latest_results(spec: FigureSpec, table) -> FigureSpec:
    """Convenience for new figures: attach the newest computed result that fits the graph.

    Only *existing* results from the statistics engine are attached (by ID); the user can
    change or remove them under "Statistics on the graph".
    """
    results = [r for r in state.project().results.values() if getattr(r, "table", None) == table.name]
    if not results:
        return spec
    if spec.plot_type in CATEGORICAL:
        with_comp = [r for r in results if r.comparisons]
        if with_comp:
            return spec.model_copy(update={"annotations": [StatAnnotation(stat_result_id=with_comp[-1].id)]})
    if spec.plot_type == "dose_response":
        latest: dict[str, str] = {}
        for r in results:
            if (r.extra or {}).get("curve"):
                latest[str((r.extra or {}).get("series"))] = r.id
        if latest:
            layers = [lay for lay in spec.effective_layers() if lay.type != "curve_fit"]
            layers += [Layer(type="curve_fit", stat_result_id=rid) for rid in latest.values()]
            return spec.model_copy(update={"layers": layers})
    if spec.plot_type == "survival":
        surv = [r for r in results if (r.extra or {}).get("curves")]
        if surv:
            return spec.model_copy(update={"data": spec.data.model_copy(update={"stat_result_id": surv[-1].id})})
    return spec


def _roles(spec: FigureSpec, table) -> dict:
    """Column-role pickers for Multiple-variables tables; group/row pickers otherwise."""
    d = spec.data.model_dump()
    none = "(none)"
    if table.kind == "multiple_variables":
        cols = [str(c) for c in table.data.columns]

        def pick(label: str, field: str, help_text: str) -> None:
            options = [none, *cols]
            current = d.get(field) if d.get(field) in cols else none
            value = st.selectbox(label, options, index=options.index(current), key=f"role_{spec.name}_{field}", help=help_text)
            d[field] = None if value == none else value

        if spec.plot_type == "survival":
            pick("Time column", "time", "How long each subject was followed.")
            pick("Event column", "event", "1 = the event happened (e.g. death), 0 = censored (still alive / lost).")
            pick("Group column", "hue", "Which treatment group each subject belongs to.")
        else:
            pick("X / categories", "x", "The grouping (or X) column.")
            pick("Y / values", "y", "The measured numbers to plot.")
            pick("Colour by", "hue", "Optional second grouping shown as colours.")
            if spec.plot_type == "superplot":
                pick("Biological replicate", "replicate", "Column identifying each independent experiment/animal.")
            if spec.plot_type == "paired":
                pick("Subject", "subject", "Column identifying the same subject across conditions.")
    else:
        names = state.group_names(table)
        label = {"column": "Groups to show (in this order)", "xy": "Series to show", "grouped": "Column groups to show"}.get(table.kind, "Groups")
        chosen = st.multiselect(label, names, default=[g for g in (d.get("groups") or []) if g in names],
                                key=f"groups_{spec.name}", help="Leave empty to show all, in table order.")
        d["groups"] = chosen or None
        if table.kind == "grouped":
            rows = st.multiselect("Rows to show", table.row_levels, default=[r for r in (d.get("rows") or []) if r in table.row_levels],
                                  key=f"rows_{spec.name}", help="Leave empty to show all rows.")
            d["rows"] = rows or None
            d["transpose"] = st.checkbox("Swap rows and columns", value=bool(d.get("transpose")), key=f"transpose_{spec.name}")
    return d


def _elements(spec: FigureSpec) -> list[Layer]:
    layers = spec.effective_layers()
    types = [lay.type for lay in layers]
    work = spec.model_copy(update={"layers": layers})
    if spec.plot_type in CATEGORICAL | {"xy", "dose_response"}:
        current_err = next((t.replace("_errorbar", "") for t in types if t.endswith("_errorbar")), None)
        labels = list(ERROR_CHOICES)
        idx = list(ERROR_CHOICES.values()).index(current_err) if current_err in ERROR_CHOICES.values() else labels.index("None")
        err = st.selectbox("Error bars", labels, index=idx, key=f"err_{spec.name}",
                           help="SEM: precision of the mean. SD: spread of the data. 95% CI: range likely to contain the true mean.")
        if ERROR_CHOICES[err] != current_err:
            work = work.with_error_bars(ERROR_CHOICES[err])
    if spec.plot_type in CATEGORICAL - {"paired"}:
        cur = "beeswarm" if "data_points_beeswarm" in types else "jitter" if "data_points_jitter" in types else "plain" if "data_points" in types else None
        labels = list(POINT_CHOICES)
        idx = list(POINT_CHOICES.values()).index(cur)
        pts = st.selectbox("Individual data points", labels, index=idx, key=f"pts_{spec.name}",
                           help="Showing every data point is recommended by most journals. Hide them only for very large datasets.")
        if POINT_CHOICES[pts] != cur:
            work = work.with_points(POINT_CHOICES[pts])
    return list(work.layers)


def _statistics(spec: FigureSpec, table, layers: list[Layer]) -> tuple[list[StatAnnotation], list[Layer]]:
    proj = state.project()
    results = [r for r in proj.results.values() if getattr(r, "table", None) == table.name]
    annotations = list(spec.annotations)
    none = "(none)"
    if not results:
        st.caption("Run an analysis on this table (page 2) to add significance brackets or fitted curves.")
        return annotations, layers
    label = {r.id: f"{r.test_name} · {r.id}" for r in results}
    with_comp = [r.id for r in results if r.comparisons]
    current = annotations[0] if annotations else None
    if spec.plot_type in CATEGORICAL:
        options = [none, *with_comp]
        cur = current.stat_result_id if current and current.kind == "brackets" and current.stat_result_id in with_comp else none
        rid = st.selectbox("Significance brackets from", options, index=options.index(cur), format_func=lambda i: label.get(i, i),
                           key=f"ann_{spec.name}",
                           help="Brackets and stars are read from this computed result. They cannot be typed in by hand.")
        if rid == none:
            annotations = [a for a in annotations if a.kind != "brackets"]
        else:
            which = st.radio("Show", ["all", "significant"], horizontal=True, key=f"ann_which_{spec.name}",
                             index=0 if not current or current.comparisons != "significant" else 1,
                             format_func=lambda v: "All comparisons" if v == "all" else "Only significant ones")
            fmt = st.selectbox("Label", ["stars", "p", "p_short", "stars_p"], key=f"ann_fmt_{spec.name}",
                               index=["stars", "p", "p_short", "stars_p"].index(current.format) if current else 0,
                               format_func=lambda v: {"stars": "Stars (ns, *, **, ...)", "p": "P = 0.0123", "p_short": "0.0123", "stars_p": "Stars and P value"}[v])
            bracket = StatAnnotation(stat_result_id=rid, comparisons=which, format=fmt)
            annotations = [a for a in annotations if a.kind != "brackets"] + [bracket]
    options = [none, *[r.id for r in results if r.p_value is not None]]
    cur_sum = next((a.stat_result_id for a in annotations if a.kind == "summary_text"), none)
    sid = st.selectbox("Test summary text in a corner", options, index=options.index(cur_sum) if cur_sum in options else 0,
                       format_func=lambda i: label.get(i, i), key=f"sum_{spec.name}",
                       help="Writes the test name and its P value (from the computed result) in a corner of the graph.")
    annotations = [a for a in annotations if a.kind != "summary_text"]
    if sid != none:
        annotations.append(StatAnnotation(stat_result_id=sid, kind="summary_text"))
    if spec.plot_type in {"xy", "dose_response"}:
        fits = [r.id for r in results if (r.extra or {}).get("curve")]
        current_fits = [lay.stat_result_id for lay in layers if lay.type == "curve_fit" and lay.stat_result_id in fits]
        chosen = st.multiselect("Fitted curves to draw", fits, default=current_fits, format_func=lambda i: label.get(i, i),
                                key=f"fits_{spec.name}", help="Curves come from regression results computed on page 2.")
        band = st.checkbox("Show 95% confidence band", key=f"band_{spec.name}",
                           value=any(lay.type == "confidence_band" for lay in layers))
        layers = [lay for lay in layers if lay.type not in {"curve_fit", "confidence_band", "prediction_band"}]
        for rid in chosen:
            if band:
                layers.insert(0, Layer(type="confidence_band", stat_result_id=rid))
            layers.append(Layer(type="curve_fit", stat_result_id=rid))
        if not chosen and spec.plot_type == "xy" and not any(lay.type == "mean_line" for lay in layers):
            layers.insert(0, Layer(type="mean_line"))
    if spec.plot_type in {"survival", "residuals", "heatmap", "qq"}:
        key_needed = {"survival": "curves", "residuals": "residuals", "heatmap": "matrix", "qq": "qq"}[spec.plot_type]
        usable = [r.id for r in results if (r.extra or {}).get(key_needed)]
        options = [none, *usable]
        cur = spec.data.stat_result_id if spec.data.stat_result_id in usable else none
        st.session_state.setdefault(f"src_{spec.name}", cur)
        st.selectbox("Draw from result", options, format_func=lambda i: label.get(i, i), key=f"src_{spec.name}",
                     help="Statistics-driven graphs are drawn from a computed result.")
    return annotations, layers


def _axes(spec: FigureSpec) -> tuple[dict, str | None]:
    axes = spec.axes.model_dump()
    title = st.text_input("Title", value=spec.title or "", key=f"title_{spec.name}") or None
    c1, c2 = st.columns(2)
    axes["x"]["label"] = c1.text_input("X axis title", value=spec.axes.x.label or "", key=f"xl_{spec.name}",
                                       placeholder="automatic") or None
    axes["y"]["label"] = c2.text_input("Y axis title", value=spec.axes.y.label or "", key=f"yl_{spec.name}",
                                       placeholder="automatic") or None
    if spec.plot_type not in {"heatmap"}:
        choices = ["auto", *SCALES]
        axes["y"]["scale"] = c2.selectbox("Y scale", choices, index=choices.index(spec.axes.y.scale) if spec.axes.y.scale in choices else 0,
                                          key=f"ys_{spec.name}", format_func=lambda v: "automatic" if v == "auto" else v,
                                          help="Use a log scale when values span several orders of magnitude.")
        if spec.plot_type in {"xy", "dose_response", "histogram", "residuals"}:
            axes["x"]["scale"] = c1.selectbox("X scale", choices, index=choices.index(spec.axes.x.scale) if spec.axes.x.scale in choices else 0,
                                              key=f"xs_{spec.name}", format_func=lambda v: "automatic" if v == "auto" else v,
                                              help="Dose-response graphs use a log X axis automatically.")
        lo = c2.text_input("Y minimum", value="" if spec.axes.y.min is None else f"{spec.axes.y.min:g}", key=f"ymin_{spec.name}", placeholder="automatic")
        hi = c2.text_input("Y maximum", value="" if spec.axes.y.max is None else f"{spec.axes.y.max:g}", key=f"ymax_{spec.name}", placeholder="automatic")
        for key, raw in (("min", lo), ("max", hi)):
            try:
                axes["y"][key] = float(raw) if raw.strip() else None
            except ValueError:
                st.warning(f"'{raw}' is not a number; using automatic.")
                axes["y"][key] = None
    return axes, title


def _editor(spec: FigureSpec, table) -> None:
    proj = state.project()
    left, right = st.columns([1, 1.25])
    with left:
        with st.expander("Data", expanded=True):
            data = _roles(spec, table)
        with st.expander("Elements", expanded=True):
            layers = _elements(spec)
        with st.expander("Statistics on the graph", expanded=True):
            annotations, layers = _statistics(spec, table, layers)
            src = st.session_state.get(f"src_{spec.name}")
            if src is not None:
                data["stat_result_id"] = None if src == "(none)" else src
        with st.expander("Titles and axes", expanded=False):
            axes, title = _axes(spec)
    try:
        new = FigureSpec.model_validate({
            **spec.model_dump(exclude={"data", "layers", "annotations", "axes", "title"}),
            "data": data, "layers": [lay.model_dump(exclude_none=True) for lay in layers],
            "annotations": [a.model_dump() for a in annotations], "axes": axes, "title": title,
        })
    except Exception as exc:  # noqa: BLE001 - show validation problems instead of crashing
        st.error(f"That setting is not valid: {exc}")
        new = spec
    proj.figures[spec.name] = new
    with right:
        st.markdown(f"**Live preview: {new.name}**")
        preview_figure(new)
        st.caption("Change the look on **4 Style**; combine figures on **5 Compose**; save files on **6 Export**.")


def render() -> None:
    st.header("3 · Graph")
    proj = state.project()
    table = state.table_picker(key="graph_table_picker")
    if table is None:
        return
    with st.expander("New figure", expanded=not proj.figures):
        _new_figure(table)
    mine = [n for n, f in proj.figures.items() if f.data.table == table.name]
    if not mine:
        st.info("Create a figure for this table above.")
        return
    current = state.current_figure_name()
    index = mine.index(current) if current in mine else 0
    c1, c2 = st.columns([4, 1])
    name = c1.selectbox("Figure", mine, index=index, key="graph_fig_pick")
    state.set_current_figure(name)
    if c2.button("Delete figure", key="delete_figure", width="stretch"):
        del proj.figures[name]
        state.set_current_figure(None)
        st.rerun()
    _editor(proj.figures[name], table)
