#!/usr/bin/env bash
# One-shot launcher: creates a virtual environment on first run, then starts the app.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "First run - setting up (this takes a minute)..."
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

echo "Starting on http://localhost:8501 - press Ctrl+C to stop."
exec ./.venv/bin/streamlit run app.py
