#!/usr/bin/env bash
# Start the Route Resilience API on http://localhost:8000  (docs at /docs)
set -e
cd "$(dirname "$0")"

# Create + activate a local virtualenv so deps don't pollute the system Python.
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -q -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
