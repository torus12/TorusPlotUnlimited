@echo off
REM ============================================================================
REM  TorusPlot launcher for Windows. Double-click this file.
REM  First run: creates a private Python environment in the ".venv" folder and
REM  installs the required packages from PyPI (needs internet once, ~5 minutes).
REM  Later runs: starts the app straight away. Nothing is installed system-wide.
REM ============================================================================
setlocal
cd /d "%~dp0"
title TorusPlot

set "PY="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)
if not defined PY (
    echo.
    echo  TorusPlot needs Python 3.11 or newer, which was not found on this computer.
    echo.
    echo  1. Go to https://www.python.org/downloads/windows/
    echo  2. Download and run the 64-bit Windows installer.
    echo  3. On the first installer screen, TICK "Add python.exe to PATH".
    echo  4. Double-click run_torusplot.bat again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo  First run: creating TorusPlot's private Python environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  Could not create the environment. See the message above.
        pause
        exit /b 1
    )
)

if not exist ".venv\torusplot_installed.txt" (
    echo.
    echo  Installing required packages from PyPI. This happens once and takes a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  Installation failed. Check your internet connection and run this file again.
        pause
        exit /b 1
    )
    echo installed> ".venv\torusplot_installed.txt"
)

echo.
echo  Starting TorusPlot. Your web browser will open at http://localhost:8501
echo  Keep this window open while you work. Close it to stop TorusPlot.
echo.
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.headless false --browser.gatherUsageStats false
if errorlevel 1 pause
endlocal
