@echo off
REM One-shot launcher for Windows: sets up on first run, then starts the app.
cd /d "%~dp0"

if not exist ".venv" (
  echo First run - setting up ^(this takes a minute^)...
  python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\pip install --quiet -r requirements.txt
)

echo Starting on http://localhost:8501 - press Ctrl+C to stop.
.venv\Scripts\streamlit run app.py
