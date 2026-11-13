"""AI Assistant: Describe it, Match this figure, Digitize, Outside AI instructions.

AI is optional and off by default. Matching (basic), digitizing and JSON import work
without any AI provider. AI can change how a figure looks; it can never compute or edit
statistics (figures refer to computed results by ID only).
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from torusplot.ai import (
    DigitizeError,
    ProviderError,
    active_provider,
    calibration_from_plot_area,
    candidate_colors,
    describe_to_spec,
    detect_plot_area,
    digitize_image,
    import_outside_instructions,
    match_figure,
    outside_prompt_template,
    to_table,
)
from torusplot.spec import FigureSpec

from .. import state
from ..ui_helpers import preview_figure
from .settings import ai_settings


def _provider():
    try:
        return active_provider(ai_settings())
    except ProviderError as exc:
        st.warning(str(exc))
        return None


def _unique_figure_name(base: str) -> str:
    figures = state.project().figures
    name, i = base, 2
    while name in figures:
        name = f"{base} {i}"
        i += 1
    return name


def _offer_save(spec: FigureSpec, key: str, default_name: str) -> None:
    """Preview a proposed spec and let the user keep it as a project figure."""
    preview_figure(spec)
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Save as figure", value=_unique_figure_name(default_name), key=f"{key}_name")
    if c2.button("Save figure", key=f"{key}_save", type="primary", width="stretch"):
        saved = spec.model_copy(update={"name": name.strip() or _unique_figure_name(default_name)})
        state.project().figures[saved.name] = saved
        state.set_current_figure(saved.name)
        state.log(f"Saved AI-assisted figure '{saved.name}' ({key}).")
        st.success(f"Saved '{saved.name}'. Refine it on **3 Graph** and **4 Style**.")
    with st.expander("Figure description (FigureSpec JSON)"):
        st.json(spec.model_dump(mode="json", exclude_none=True))


def _show_outcome(outcome, key: str, default_name: str) -> None:
    for w in outcome.warnings:
        st.warning(w)
    if outcome.spec is None:
        st.error("No valid figure could be produced.")
        for e in outcome.errors[-3:]:
            st.caption(e)
        return
    if outcome.attempts > 1:
        st.caption(f"The model needed {outcome.attempts} attempts (earlier answers failed validation and were sent back).")
    _offer_save(outcome.spec, key, default_name)


def _describe(table) -> None:
    st.caption("Type what you want. Example: *bar chart with SD error bars, beeswarm points, journal single-column "
               "style, colour-blind safe colours, show only significant comparisons*.")
    provider = _provider()
    if provider is None:
        st.info("Turn on **Enable AI features** under **Settings** to use this. (Free option: Ollama, runs locally.)")
    text = st.text_area("Describe the figure", height=110, key="describe_text")
    proj = state.project()
    figures = [n for n, f in proj.figures.items() if f.data.table == table.name]
    edit = st.selectbox("Start from", ["(a new figure)", *figures], key="describe_base",
                        help="Pick an existing figure to change it, e.g. 'make the bars grey and use SD'.")
    if st.button("Create figure from description", key="describe_go", type="primary", disabled=provider is None or not text.strip()):
        base = proj.figures.get(edit) if edit != "(a new figure)" else None
        try:
            with st.spinner("Asking the model..."):
                outcome = describe_to_spec(text, proj.tables, proj.results, current_table=table.name, provider=provider,
                                           max_retries=ai_settings().max_retries, base_spec=base)
            st.session_state["describe_outcome"] = outcome
        except ProviderError as exc:
            st.session_state.pop("describe_outcome", None)
            st.error(f"The AI provider could not be reached: {exc}")
    outcome = st.session_state.get("describe_outcome")
    if outcome is not None:
        _show_outcome(outcome, "describe", "Described figure")


def _match(table) -> None:
    st.caption("Upload an image of a figure whose **look** you like. TorusPlot copies its style (colours, proportions, "
               "axes look) onto **your** data. It never copies the other figure's data or artwork.")
    upload = st.file_uploader("Reference figure (PNG or JPG)", type=["png", "jpg", "jpeg"], key="match_upload")
    proj = state.project()
    figures = [n for n, f in proj.figures.items() if f.data.table == table.name]
    base_name = st.selectbox("Apply the style to", ["(a new figure of this table)", *figures], key="match_base")
    provider = _provider()
    if provider is None:
        st.caption("AI is off: colours and proportions are measured directly from the image (no model). "
                   "With a vision model enabled, plot type and finer style details are also matched.")
    if upload is not None and st.button("Match this figure", key="match_go", type="primary"):
        base = proj.figures.get(base_name) if not base_name.startswith("(") else None
        try:
            with st.spinner("Measuring the reference and rendering..."):
                st.session_state["match_outcome"] = match_figure(
                    upload.getvalue(), table, proj.tables, proj.results, provider=provider,
                    max_loops=ai_settings().max_feedback_loops, base_spec=base)
        except ProviderError as exc:
            st.error(f"The AI provider could not be reached: {exc}")
        except Exception as exc:  # noqa: BLE001 - unreadable image etc.
            st.error(f"Could not match this image: {exc}")
    outcome = st.session_state.get("match_outcome")
    if outcome is None:
        return
    left, right = st.columns(2)
    left.image(outcome.reference_png, caption="Reference (style source)")
    right.image(outcome.render_png, caption="Your data in that style")
    ext = outcome.extraction
    swatches = " ".join(f"<span style='display:inline-block;width:22px;height:22px;border-radius:4px;border:1px solid #999;"
                        f"background:{c}' title='{c}'></span>" for c in ext.palette)
    st.markdown(f"Colours measured from the reference: {swatches}", unsafe_allow_html=True)
    st.caption(f"Similarity score (SSIM, rough guide only because the data differ): {outcome.best_ssim:.2f} "
               f"after {len(outcome.history)} round(s). Source: {ext.source}.")
    for note in [*outcome.notes, *ext.notes]:
        st.caption(f"Note: {note}")
    _offer_save(outcome.spec, "match", "Matched figure")


def _digitize() -> None:
    st.error("APPROXIMATE: for reference only. Digitized values are estimates read off an image. "
             "They are labelled approximate everywhere and statistical tests are blocked unless you explicitly override.")
    upload = st.file_uploader("Image of a plot (PNG or JPG)", type=["png", "jpg", "jpeg"], key="digi_upload")
    if upload is None:
        return
    data = upload.getvalue()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the image: {exc}")
        return
    st.image(image, caption=f"{image.width} x {image.height} pixels")
    area = detect_plot_area(data, inner=True)
    st.markdown("**1. Where is the plot area?** (pixels; detected automatically when the axes are visible)")
    c = st.columns(4)
    defaults = area or (int(image.width * 0.12), int(image.height * 0.08), int(image.width * 0.95), int(image.height * 0.85))
    box = tuple(int(c[i].number_input(lab, 0, max(image.width, image.height), value=int(defaults[i]), key=f"digi_{lab}"))
                for i, lab in enumerate(["left", "top", "right", "bottom"]))
    if area is None:
        st.caption("The axes could not be found automatically; adjust the four pixel positions above.")
    st.markdown("**2. What values do the axes show at the edges of that area?**")
    c = st.columns(4)
    x_min = c[0].number_input("X at left edge", value=0.0, format="%g", key="digi_xmin")
    x_max = c[1].number_input("X at right edge", value=10.0, format="%g", key="digi_xmax")
    y_min = c[2].number_input("Y at bottom edge", value=0.0, format="%g", key="digi_ymin")
    y_max = c[3].number_input("Y at top edge", value=100.0, format="%g", key="digi_ymax")
    c = st.columns(2)
    x_log = c[0].checkbox("X axis is logarithmic", key="digi_xlog")
    y_log = c[1].checkbox("Y axis is logarithmic", key="digi_ylog")
    st.markdown("**3. Which series?**")
    colors = candidate_colors(data)
    if not colors:
        st.warning("No distinct series colours were found in this image.")
        return
    c = st.columns(3)
    color = c[0].selectbox("Series colour", [col["hex"] for col in colors], key="digi_color",
                           format_func=lambda h: f"{h} ({next(col['role'] for col in colors if col['hex'] == h)})")
    c[0].markdown(f"<span style='display:inline-block;width:60px;height:18px;background:{color};border:1px solid #999'></span>",
                  unsafe_allow_html=True)
    mode = c[1].selectbox("What to extract", ["points", "curve", "bars"], key="digi_mode",
                          format_func=lambda m: {"points": "Separate markers", "curve": "A continuous line", "bars": "Bar heights"}[m])
    tolerance = c[2].slider("Colour tolerance", 5.0, 60.0, 20.0, key="digi_tol",
                            help="How different a pixel may be from the chosen colour and still count.")
    if st.button("Digitize", key="digi_go", type="primary"):
        try:
            x_axis, y_axis = calibration_from_plot_area(box, x_min, x_max, y_min, y_max, x_log=x_log, y_log=y_log)
            st.session_state["digi_result"] = digitize_image(data, x_axis, y_axis, color=color, tolerance=tolerance,
                                                             mode=mode, plot_area=box)
        except (DigitizeError, ValueError) as exc:
            st.session_state.pop("digi_result", None)
            st.error(f"Digitizing failed: {exc}")
    result = st.session_state.get("digi_result")
    if result is None:
        return
    st.success(f"Extracted {result.n} approximate value(s).")
    for note in result.notes:
        st.caption(f"Note: {note}")
    st.dataframe(result.points, hide_index=True, width="stretch")
    name = st.text_input("Table name", value="Digitized data", key="digi_name")
    if st.button("Add as an APPROXIMATE table", key="digi_add"):
        proj = state.project()
        table = to_table(result, proj.unique_table_name(name.strip() or "Digitized data"))
        table.name = proj.unique_table_name(table.name) if table.name in proj.tables else table.name
        proj.add_table(table)
        state.set_current_table(table.name)
        state.log(f"Added digitized (approximate) table '{table.name}' with {result.n} values.")
        st.success(f"Added '{table.name}'. It is marked APPROXIMATE on every page and in every report.")


def _outside(table) -> None:
    st.caption("Paste instructions or JSON written by any chatbot. JSON that follows TorusPlot's format is imported "
               "directly (no AI needed). Anything that looks like a typed statistic is dropped: statistics must be "
               "computed in TorusPlot.")
    with st.expander("Prompt to give your chatbot (copy all of it)"):
        template = outside_prompt_template(state.project().tables)
        st.download_button("Download the prompt as a text file", data=template.encode("utf-8"),
                           file_name="torusplot_prompt_for_chatbots.md", mime="text/markdown", key="outside_dl")
        st.code(template[:6000] + ("\n...\n(download for the full prompt)" if len(template) > 6000 else ""), language=None)
    text = st.text_area("Paste the chatbot's answer here", height=200, key="outside_text")
    if st.button("Turn into a figure", key="outside_go", type="primary", disabled=not text.strip()):
        proj = state.project()
        try:
            st.session_state["outside_outcome"] = import_outside_instructions(
                text, proj.tables, proj.results, current_table=table.name, provider=_provider())
        except ProviderError as exc:
            st.session_state.pop("outside_outcome", None)
            st.error(f"The AI provider could not be reached: {exc}")
    outcome = st.session_state.get("outside_outcome")
    if outcome is not None:
        _show_outcome(outcome, "outside", "Imported figure")


def render() -> None:
    st.header("AI Assistant")
    settings = ai_settings()
    status = f"on ({settings.provider})" if settings.enabled else "off"
    st.caption(f"AI features are **{status}**. Change this under **Settings**. AI can restyle and lay out figures; "
               "it can never compute, edit or invent statistics.")
    tabs = st.tabs(["Describe it", "Match this figure", "Digitize", "Outside AI instructions"])
    with tabs[2]:
        _digitize()
    table = state.current_table()
    for tab, func, key in ((tabs[0], _describe, "describe"), (tabs[1], _match, "match"), (tabs[3], _outside, "outside")):
        with tab:
            if table is None:
                st.info("Load or paste data on **1 Data** first.")
            else:
                if key == "describe":
                    table = state.table_picker("Table", key="ai_table_picker") or table
                else:
                    st.caption(f"Using table: **{table.name}** (change it on the first tab).")
                func(table)
