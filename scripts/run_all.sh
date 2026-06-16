#!/usr/bin/env bash
#
# run_all.sh - Start both the FastAPI backend and the Vite/Bun frontend for the
# "Route Resilience" demo, then wait. Ctrl-C (or any exit) tears down both.
#
# Usage:
#   ./scripts/run_all.sh
#
# Prerequisites (set up once, NOT done here):
#   - Backend virtualenv at backend/.venv with deps installed:
#       python3.14 -m venv backend/.venv
#       backend/.venv/bin/pip install -r backend/requirements.txt
#   - Frontend deps installed in atlas-vision:
#       cd atlas-vision && bun install
#
set -euo pipefail

# Resolve the project root from this script's location so the script works no
# matter what directory it is invoked from. The repo path contains spaces, so
# every path below is quoted.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname -- "$SCRIPT_DIR")"

BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/atlas-vision"
VENV_UVICORN="$BACKEND_DIR/.venv/bin/uvicorn"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_PORT="5173"   # Vite default; bun run dev may pick another if taken.

# --- sanity checks ----------------------------------------------------------
if [[ ! -x "$VENV_UVICORN" ]]; then
  echo "ERROR: uvicorn not found at: $VENV_UVICORN" >&2
  echo "Create the venv and install deps first:" >&2
  echo "  python3.14 -m venv \"$BACKEND_DIR/.venv\"" >&2
  echo "  \"$BACKEND_DIR/.venv/bin/pip\" install -r \"$BACKEND_DIR/requirements.txt\"" >&2
  exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
  echo "ERROR: 'bun' is not on PATH. Install bun: https://bun.sh" >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "ERROR: frontend directory not found at: $FRONTEND_DIR" >&2
  exit 1
fi

# --- teardown ---------------------------------------------------------------
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  # Kill the whole process group of each child so uvicorn/vite workers also die.
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# --- start backend ----------------------------------------------------------
# Run uvicorn from inside the backend dir so its relative imports / data paths
# resolve. app.main:app is the conventional FastAPI entrypoint; adjust if yours
# differs (TODO: confirm module path matches backend/app/main.py).
echo "Starting backend (uvicorn) on http://$BACKEND_HOST:$BACKEND_PORT ..."
(
  cd -- "$BACKEND_DIR"
  exec "$VENV_UVICORN" app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
) &
BACKEND_PID=$!

# --- start frontend ---------------------------------------------------------
echo "Starting frontend (bun run dev) ..."
(
  cd -- "$FRONTEND_DIR"
  exec bun run dev --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

# --- report -----------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  Route Resilience demo is starting up"
echo "  Backend  API : http://$BACKEND_HOST:$BACKEND_PORT/api/health"
echo "  Backend docs : http://$BACKEND_HOST:$BACKEND_PORT/docs"
echo "  Frontend UI  : http://localhost:$FRONTEND_PORT"
echo ""
echo "  Press Ctrl-C to stop both."
echo "=================================================================="
echo ""

# Wait for either process to exit; if one dies, cleanup() takes the other down.
wait -n
