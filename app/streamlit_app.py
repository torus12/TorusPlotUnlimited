"""TorusPlot desktop app (Streamlit). Start it with run_torusplot.bat / run_torusplot.sh, or:

    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import state  # noqa: E402
from app.views import analyze, assistant, compose, data, export, graph, settings, style  # noqa: E402
from torusplot import __version__  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

PAGES = {
    "1 Data": data.render,
    "2 Analyze": analyze.render,
    "3 Graph": graph.render,
    "4 Style": style.render,
    "5 Compose": compose.render,
    "6 Export": export.render,
    "AI Assistant": assistant.render,
    "Settings": settings.render,
}

HINTS = {
    "1 Data": "Load or paste your numbers",
    "2 Analyze": "Pick and run a statistical test",
    "3 Graph": "Build the figure",
    "4 Style": "Make it publication-ready",
    "5 Compose": "Combine panels A, B, C",
    "6 Export": "Save files and the stats report",
    "AI Assistant": "Optional helpers (off by default)",
    "Settings": "AI provider and preferences",
}


def main() -> None:
    st.set_page_config(page_title="TorusPlot", page_icon=str(ASSETS / "icon.svg"), layout="wide")
    proj = state.project()
    with st.sidebar:
        st.image(str(ASSETS / "logo.svg"), width=210)
        st.caption(f"Scientific graphing and statistics · v{__version__}")
        page = st.radio("Workflow", state.WORKFLOW, key="page", captions=[HINTS[p] for p in state.WORKFLOW])
        st.divider()
        st.caption(f"Project: **{proj.name}**")
        st.caption(f"{len(proj.tables)} table(s) · {len(proj.results)} result(s) · {len(proj.figures)} figure(s)")
        st.caption("Runs locally. Your data never leaves this computer unless you turn on a cloud AI provider.")
    PAGES[page]()


main()
