"""Page 1: Data. Load samples, paste from Excel, import files, edit the grid."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from torusplot.data import (
    KIND_DESCRIPTIONS,
    KIND_LABELS,
    TABLE_KINDS,
    TableError,
    build_table,
    guess_table_kind,
    list_sheets,
    parse_pasted_text,
    read_frame,
    table_from_meta,
    to_multiple_variables,
)
from torusplot.data.samples import SAMPLES, load_sample

from .. import state

HEADER_MODES = {"Automatic": None, "First row is column titles": True, "No titles (numbers only)": False}


def _kind_picker(key: str, suggestion: str | None = None) -> str:
    index = TABLE_KINDS.index(suggestion) if suggestion in TABLE_KINDS else 0
    kind = st.radio(
        "Table type", TABLE_KINDS, index=index, format_func=lambda k: KIND_LABELS[k], key=key, horizontal=True,
        help="The table type decides which analyses and graphs are offered. Pick the one that matches "
             "how your experiment is organised.",
    )
    st.caption(KIND_DESCRIPTIONS[kind])
    return kind


def _options(kind: str, key: str) -> dict:
    opts: dict = {}
    if kind in {"xy", "grouped"}:
        n = st.number_input(
            "Replicates side by side", min_value=0, max_value=50, value=0, step=1, key=f"{key}_rep",
            help="How many replicate columns each group has, placed next to each other. Leave 0 if the "
                 "columns are named like 'Drug:1, Drug:2' or repeat the same title (grouped automatically).",
        )
        if n and n > 1:
            opts["replicates"] = int(n)
    return opts


def _add_table(kind: str, frame: pd.DataFrame, name: str, **opts) -> None:
    proj = state.project()
    try:
        table = build_table(kind, frame, proj.unique_table_name(name.strip() or "Table"), **opts)
    except TableError as exc:
        st.error(f"Could not create the table: {exc}")
        return
    proj.add_table(table)
    state.set_current_table(table.name)
    state.log(f"Added {table.kind_label} table '{table.name}'.")
    st.success(f"Created {table.summary()}. Next: **2 Analyze** or **3 Graph**.")


def _add_data() -> None:
    proj = state.project()
    st.subheader("Load a sample dataset")
    c1, c2 = st.columns([3, 1])
    key = c1.selectbox("Sample", list(SAMPLES), format_func=lambda k: f"{SAMPLES[k]['title']}  ({KIND_LABELS[SAMPLES[k]['kind']]})",
                       key="sample_choice", help="Synthetic (simulated) datasets for learning the app. Not real measurements.")
    c1.caption(SAMPLES[key]["description"])
    if c2.button("Load sample", key="load_sample", width="stretch"):
        table = load_sample(key)
        table.name = proj.unique_table_name(table.name)
        proj.add_table(table)
        state.set_current_table(table.name)
        state.log(f"Loaded synthetic sample '{key}' as table '{table.name}'.")
        st.success(f"Loaded {table.summary()}.")

    st.divider()
    st.subheader("Paste from Excel")
    st.caption("Copy cells in Excel or Google Sheets (Ctrl+C) and paste them below (Ctrl+V). Include the column titles.")
    text = st.text_area("Pasted cells", height=160, key="paste_text", placeholder="Control\tDrug A\tDrug B\n1.2\t2.4\t3.1\n1.4\t2.2\t2.9")
    suggestion = None
    parsed = None
    header_mode = st.selectbox("Column titles", list(HEADER_MODES), key="paste_header",
                               help="Whether the first pasted row holds the column titles.")
    if text.strip():
        try:
            parsed = parse_pasted_text(text, header=HEADER_MODES[header_mode])
            suggestion = guess_table_kind(parsed)
            st.caption(f"Detected {parsed.shape[0]} rows x {parsed.shape[1]} columns"
                       + (" (decimal commas converted)" if parsed.attrs.get("decimal_comma") else "")
                       + f". Suggested table type: **{KIND_LABELS[suggestion]}**.")
            st.dataframe(parsed.head(8), hide_index=True, width="stretch")
        except TableError as exc:
            st.error(str(exc))
    kind = _kind_picker("paste_kind", suggestion)
    name = st.text_input("Table name", value="My data", key="paste_name",
                         help="A short name, e.g. 'Particle size (nm)'. It is used as the default axis title.")
    opts = _options(kind, "paste")
    if st.button("Create table from pasted cells", key="paste_create", type="primary", disabled=parsed is None):
        _add_table(kind, parsed, name, **opts)

    st.divider()
    st.subheader("Import a file")
    upload = st.file_uploader("CSV, TSV, TXT or Excel (.xlsx)", type=["csv", "tsv", "txt", "xlsx"], key="file_upload")
    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(upload.getvalue())
            sheet = None
            try:
                if suffix == ".xlsx":
                    sheet = st.selectbox("Worksheet", list_sheets(path), key="file_sheet")
                frame = read_frame(path, sheet=sheet)
            except (TableError, Exception) as exc:  # noqa: BLE001
                st.error(f"Could not read the file: {exc}")
                frame = None
        if frame is not None:
            st.dataframe(frame.head(8), hide_index=True, width="stretch")
            fkind = _kind_picker("file_kind", guess_table_kind(frame))
            fname = st.text_input("Table name", value=Path(upload.name).stem, key="file_name")
            fopts = _options(fkind, "file")
            if st.button("Create table from file", key="file_create", type="primary"):
                _add_table(fkind, frame, fname, **fopts)


def _tables() -> None:
    proj = state.project()
    table = state.table_picker("Table", key="data_table_picker")
    if table is None:
        return
    st.markdown(f"**{table.kind_label} table** · {table.data.shape[0]} rows x {table.data.shape[1]} columns")
    st.caption(KIND_DESCRIPTIONS[table.kind])
    if table.approximate:
        st.error("APPROXIMATE: for reference only. These values were digitized from an image. "
                 "Statistical tests are blocked on this table unless you explicitly override on the Analyze page.")
    if table.notes:
        st.caption(table.notes)
    edited = st.data_editor(table.data, num_rows="dynamic", key=f"grid_{table.name}", width="stretch",
                            hide_index=False)
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Apply edits", key="apply_edits", help="Save the changes you typed into the grid."):
        try:
            new = table_from_meta(pd.DataFrame(edited), table.meta())
            proj.tables[table.name] = new
            state.log(f"Edited table '{table.name}' by hand.")
            st.success("Edits saved.")
        except TableError as exc:
            st.error(f"Edits not saved: {exc}")
    c2.download_button("Download CSV", data=table.data.to_csv(index=False).encode("utf-8"),
                       file_name=f"{table.name}.csv", mime="text/csv", key="dl_table_csv")
    if c3.button("Make long-format copy", key="to_long",
                 help="Creates a 'Multiple variables' copy with one row per value (tidy format)."):
        long = to_multiple_variables(table, proj.unique_table_name(f"{table.name} (long)"))
        proj.add_table(long)
        st.success(f"Created '{long.name}'.")
    if c4.button("Delete table", key="delete_table"):
        del proj.tables[table.name]
        state.log(f"Deleted table '{table.name}'.")
        st.session_state.pop("table", None)
        st.rerun()
    if table.excluded:
        st.warning(" ".join(table.exclusion_notes()))
        if st.button("Restore excluded values", key="clear_excl"):
            table.clear_exclusions()
            state.log(f"Restored all excluded values in '{table.name}'.")
            st.rerun()
    with st.expander("Long-format view (what analyses see)"):
        buf = io.StringIO()
        long = table.to_long()
        long.head(200).to_csv(buf, index=False)
        st.dataframe(long.head(200), hide_index=True, width="stretch")


def render() -> None:
    st.header("1 · Data")
    st.caption("Bring your numbers in. Everything stays on this computer.")
    tab_add, tab_tables = st.tabs(["Add data", f"Tables ({len(state.project().tables)})"])
    with tab_add:
        _add_data()
    with tab_tables:
        _tables()
