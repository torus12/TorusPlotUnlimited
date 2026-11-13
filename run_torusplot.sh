#!/usr/bin/env bash
# TorusPlot launcher for macOS and Linux:   ./run_torusplot.sh
# First run creates a private environment in .venv and installs the requirements (once).
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "TorusPlot needs Python 3.11 or newer. Install it from https://www.python.org/downloads/ and run this script again."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "First run: creating TorusPlot's private Python environment..."
    "$PY" -m venv .venv
fi
if [ ! -f ".venv/torusplot_installed.txt" ]; then
    echo "Installing required packages from PyPI (once, a few minutes)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    echo installed > .venv/torusplot_installed.txt
fi

echo "Starting TorusPlot at http://localhost:8501 (press Ctrl+C here to stop)."
exec .venv/bin/python -m streamlit run app/streamlit_app.py --server.headless false --browser.gatherUsageStats false
