#!/usr/bin/env bash
# OnTime for macOS / Linux: sets up on first run, then starts the app and opens the browser.
#   ./start_ontime.sh          (port from .env, default 8000)
#   ./start_ontime.sh 8010     (another port)
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3 || true)
[ -z "$PY" ] && { echo "Python 3.11+ not found. Install it from python.org."; exit 1; }
if [ ! -f .venv/.ontime_ready ]; then
  echo "First run: creating .venv and installing packages..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip --quiet
  .venv/bin/python -m pip install -r requirements-app.txt
  echo requirements-app.txt > .venv/.ontime_ready
  [ -f .env ] || cp .env.example .env
fi
PORT=${1:-$(grep -E '^ONTIME_PORT=' .env 2>/dev/null | cut -d= -f2)}
export ONTIME_PORT=${PORT:-8000}
( for i in $(seq 1 60); do
    if curl -s "http://localhost:$ONTIME_PORT/api/health" >/dev/null; then
      (open "http://localhost:$ONTIME_PORT" 2>/dev/null || xdg-open "http://localhost:$ONTIME_PORT" 2>/dev/null) ; break
    fi; sleep 0.7; done ) &
exec .venv/bin/python run.py
