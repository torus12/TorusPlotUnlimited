"""Session state helpers shared by all pages."""

from __future__ import annotations

from typing import Any

import streamlit as st

from torusplot.data import DataTable, Project

WORKFLOW = ["1 Data", "2 Analyze", "3 Graph", "4 Style", "5 Compose", "6 Export", "AI Assistant", "Settings"]


def project() -> Project:
    if "project" not in st.session_state:
        st.session_state["project"] = Project("My project")
    return st.session_state["project"]


def set_project(new: Project) -> None:
    st.session_state["project"] = new
    for key in ("table", "figure", "recommendation", "last_result"):
        st.session_state.pop(key, None)


def current_table() -> DataTable | None:
    proj = project()
    name = st.session_state.get("table")
    if name not in proj.tables:
        name = next(iter(proj.tables), None)
        st.session_state["table"] = name
    return proj.tables.get(name) if name else None


def set_current_table(name: str) -> None:
    st.session_state["table"] = name
    st.session_state.pop("recommendation", None)


def current_figure_name() -> str | None:
    proj = project()
    name = st.session_state.get("figure")
    if name not in proj.figures:
        name = next(iter(proj.figures), None)
        st.session_state["figure"] = name
    return name


def set_current_figure(name: str | None) -> None:
    st.session_state["figure"] = name


def log(message: str) -> None:
    project().log.append(message)


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def table_picker(label: str = "Table", key: str = "table_picker") -> DataTable | None:
    """Selectbox of the project's tables, kept in sync with the current table."""
    proj = project()
    names = list(proj.tables)
    if not names:
        st.info("No data yet. Go to **1 Data** and load a sample dataset or paste your own numbers.")
        return None
    current = current_table()
    index = names.index(current.name) if current and current.name in names else 0
    choice = st.selectbox(label, names, index=index, key=key,
                          help="The data table this page works on. Add tables on the Data page.")
    if current is None or choice != current.name:
        set_current_table(choice)
    return proj.tables[choice]


def group_names(table: DataTable) -> list[str]:
    """Names a 'group' parameter can take for this table."""
    kind = table.kind
    if kind == "column":
        return list(table.group_names)
    if kind == "xy":
        return list(table.series_names)
    if kind == "grouped":
        return list(table.column_levels)
    return [str(c) for c in table.data.columns]
