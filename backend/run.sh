#!/usr/bin/env bash
# Start the Route Resilience API on http://localhost:8000  (docs at /docs)
set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
