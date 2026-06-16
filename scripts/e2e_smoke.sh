#!/usr/bin/env bash
#
# e2e_smoke.sh - End-to-end smoke test for the Route Resilience backend.
#
# Verifies the core demo contract WITHOUT training, downloading datasets, or
# installing torch:
#   * every advertised /api endpoint responds,
#   * the network GeoJSON is non-empty,
#   * the DEMO STORY holds:
#       input=occluded & model=baseline  -> FRAGMENTED  (resilienceIndex == 79)
#       every other (input, model) combo -> INTACT      (resilienceIndex == 100)
#
# Two modes (auto-selected):
#   1. SERVER mode  - if a backend is already listening on $BASE_URL, curl it.
#   2. TESTCLIENT   - otherwise drive the app in-process via FastAPI TestClient
#                     (a single python one-liner; no network, no uvicorn).
#
# Usage:
#   ./scripts/e2e_smoke.sh                 # auto-detect
#   BASE_URL=http://localhost:8000 ./scripts/e2e_smoke.sh
#   FORCE_TESTCLIENT=1 ./scripts/e2e_smoke.sh   # skip server probe
#
# Exit code: 0 on success, non-zero on the first failed assertion.
#
# Prereqs for TESTCLIENT mode (set up once, NOT done here):
#   python3.14 -m venv backend/.venv
#   backend/.venv/bin/pip install -r backend/requirements.txt
#
set -euo pipefail

# --- paths (repo path contains spaces, so everything is quoted) -------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname -- "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"

BASE_URL="${BASE_URL:-http://localhost:8000}"
CITY="${CITY:-Bengaluru}"

# Expected resilience indices from the demo story.
FRAGMENTED_INDEX=79
INTACT_INDEX=100

# Pick a python interpreter: prefer the backend venv, fall back to python3.
if [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  PY="$BACKEND_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Decide mode: is a server already up?
# ---------------------------------------------------------------------------
server_up() {
  [[ "${FORCE_TESTCLIENT:-0}" == "1" ]] && return 1
  command -v curl >/dev/null 2>&1 || return 1
  curl -fsS --max-time 2 "$BASE_URL/api/health" >/dev/null 2>&1
}

# ===========================================================================
# MODE 1: live server via curl  (needs curl + python for JSON assertions)
# ===========================================================================
run_curl_mode() {
  echo "==> SERVER mode: probing $BASE_URL"

  # Fetch a URL and assert HTTP 200, returning the body on stdout.
  get() {
    local path="$1"
    curl -fsS --max-time 15 "$BASE_URL$path" \
      || fail "GET $path did not return HTTP 2xx"
  }

  # Extract a scalar from JSON on stdin using python (no jq dependency).
  # $1 = python expression over the parsed object `d`.
  jget() {
    "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"
  }

  # --- discovery ---
  get "/api/health" | jget 'd["status"]' | grep -q '^ok$' \
    && pass "health status=ok" || fail "health status != ok"
  get "/api/cities" | jget 'len(d)' | grep -q '^1$' \
    && pass "cities returned" || fail "cities empty"

  # --- per-combo network + metrics assertions ---
  for input in clean occluded; do
    for model in baseline robust; do
      local q="city=$CITY&input=$input&model=$model"

      # network must be a non-empty FeatureCollection with meta.edges>0
      local edges
      edges=$(get "/api/network?$q" | jget 'd["meta"]["edges"]')
      [[ "$edges" -gt 0 ]] \
        && pass "network($input,$model) has $edges edges" \
        || fail "network($input,$model) empty"

      # gatekeepers non-empty
      local gk
      gk=$(get "/api/gatekeepers?$q&top_k=5" | jget 'len(d)')
      [[ "$gk" -gt 0 ]] \
        && pass "gatekeepers($input,$model)=$gk" \
        || fail "gatekeepers($input,$model) empty"

      # resilience curve arrays present & aligned
      get "/api/resilience-curve?$q" \
        | jget 'len(d["removedFraction"])==len(d["efficiency"])==len(d["giantComponent"]) and len(d["efficiency"])>0' \
        | grep -q '^True$' \
        && pass "resilience-curve($input,$model) aligned" \
        || fail "resilience-curve($input,$model) malformed"

      # the headline assertion: resilienceIndex matches the demo story
      local ri
      ri=$(get "/api/metrics?$q" | jget 'd["resilienceIndex"]')
      local want="$INTACT_INDEX"
      [[ "$input" == "occluded" && "$model" == "baseline" ]] && want="$FRAGMENTED_INDEX"
      [[ "$ri" == "$want" ]] \
        && pass "metrics($input,$model) resilienceIndex=$ri (expected $want)" \
        || fail "metrics($input,$model) resilienceIndex=$ri, expected $want"
    done
  done

  # --- simulate (POST) ---
  curl -fsS --max-time 15 -H 'Content-Type: application/json' \
    -d "{\"city\":\"$CITY\",\"model\":\"baseline\",\"input\":\"occluded\",\"disabledEdgeIds\":[],\"disabledNodeIds\":[]}" \
    "$BASE_URL/api/simulate" \
    | jget '"resilienceIndexAfter" in d and "brokenRoutesSampled" in d' \
    | grep -q '^True$' \
    && pass "simulate returns contract fields" \
    || fail "simulate missing contract fields"

  echo "==> SERVER mode: all assertions passed"
}

# ===========================================================================
# MODE 2: in-process FastAPI TestClient  (no network / no uvicorn)
# ===========================================================================
run_testclient_mode() {
  echo "==> TESTCLIENT mode: driving app.main:app in-process via $PY"

  # The python program below is the real smoke test. It mirrors the curl-mode
  # assertions but talks to the ASGI app directly. It raises SystemExit(1) on
  # the first failed check, which propagates as this script's exit code.
  CITY="$CITY" FRAGMENTED_INDEX="$FRAGMENTED_INDEX" INTACT_INDEX="$INTACT_INDEX" \
  "$PY" - "$BACKEND_DIR" <<'PYEOF'
import os
import sys
from pathlib import Path

backend_dir = Path(sys.argv[1]).resolve()
# Make `import app.main` resolve no matter where we were invoked from.
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
# Run as if cwd == backend so any relative data paths inside the app resolve.
os.chdir(backend_dir)

CITY = os.environ.get("CITY", "Bengaluru")
FRAGMENTED = int(os.environ["FRAGMENTED_INDEX"])
INTACT = int(os.environ["INTACT_INDEX"])

try:
    from fastapi.testclient import TestClient
    from app.main import app
except Exception as exc:  # pragma: no cover - setup error
    print(f"  FAIL  could not import app/TestClient: {exc}", file=sys.stderr)
    print("        Did you create backend/.venv and install requirements.txt?",
          file=sys.stderr)
    raise SystemExit(2)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def ok(msg):
    print(f"  {GREEN}PASS{RESET} {msg}")


def die(msg):
    print(f"  {RED}FAIL{RESET} {msg}", file=sys.stderr)
    raise SystemExit(1)


def get_json(client, path):
    r = client.get(path)
    if r.status_code != 200:
        die(f"GET {path} -> HTTP {r.status_code}")
    return r.json()


with TestClient(app) as client:
    # --- discovery ---
    health = get_json(client, "/api/health")
    if health.get("status") != "ok":
        die(f"health status={health.get('status')!r}, expected 'ok'")
    if not health.get("baseEfficiency", 0) > 0:
        die("health.baseEfficiency must be > 0")
    ok("health status=ok, baseEfficiency>0")

    cities = get_json(client, "/api/cities")
    if not cities:
        die("cities is empty")
    if CITY not in cities:
        die(f"{CITY!r} not in cities {cities}")
    ok(f"cities includes {CITY}")

    # --- per-combo assertions ---
    for inp in ("clean", "occluded"):
        for model in ("baseline", "robust"):
            q = f"city={CITY}&input={inp}&model={model}"
            combo = f"({inp},{model})"

            net = get_json(client, f"/api/network?{q}")
            feats = net.get("features", [])
            meta = net.get("meta", {})
            edges = meta.get("edges", 0)
            if not feats or edges <= 0:
                die(f"network{combo} empty (features={len(feats)}, meta.edges={edges})")
            # spot-check the edge property contract on the first feature.
            props = feats[0].get("properties", {})
            required = {"id", "criticality", "travelTimeSec", "lengthM",
                        "roadClass", "isBridge"}
            missing = required - set(props)
            if missing:
                die(f"network{combo} edge missing props: {sorted(missing)}")
            geom = feats[0].get("geometry", {})
            if geom.get("type") not in ("LineString", "MultiLineString"):
                die(f"network{combo} unexpected geometry {geom.get('type')!r}")
            ok(f"network{combo}: {edges} edges, props+geometry OK")

            gks = get_json(client, f"/api/gatekeepers?{q}&top_k=5")
            if not gks:
                die(f"gatekeepers{combo} empty")
            gk0 = gks[0]
            gk_required = {"id", "lat", "lng", "betweenness",
                           "isArticulation", "label"}
            gk_missing = gk_required - set(gk0)
            if gk_missing:
                die(f"gatekeepers{combo} missing fields: {sorted(gk_missing)}")
            ok(f"gatekeepers{combo}: {len(gks)} nodes, fields OK")

            curve = get_json(client, f"/api/resilience-curve?{q}")
            rf = curve.get("removedFraction", [])
            eff = curve.get("efficiency", [])
            gc = curve.get("giantComponent", [])
            if not (len(rf) == len(eff) == len(gc)) or len(eff) == 0:
                die(f"resilience-curve{combo} arrays misaligned/empty "
                    f"({len(rf)},{len(eff)},{len(gc)})")
            ok(f"resilience-curve{combo}: {len(eff)} points aligned")

            metrics = get_json(client, f"/api/metrics?{q}")
            ri = metrics.get("resilienceIndex")
            want = FRAGMENTED if (inp == "occluded" and model == "baseline") else INTACT
            if ri != want:
                die(f"metrics{combo} resilienceIndex={ri}, expected {want}")
            ok(f"metrics{combo}: resilienceIndex={ri} (expected {want})")

    # --- simulate (POST) ---
    sim = client.post("/api/simulate", json={
        "city": CITY,
        "model": "baseline",
        "input": "occluded",
        "disabledEdgeIds": [],
        "disabledNodeIds": [],
    })
    if sim.status_code != 200:
        die(f"POST /api/simulate -> HTTP {sim.status_code}")
    sbody = sim.json()
    sim_required = {"disabledEdgeIds", "disabledNodeIds", "resilienceIndexAfter",
                    "avgTravelTimeIncreasePct", "newlyDisconnectedZones",
                    "brokenRoutesSampled", "sampledRoutes"}
    sim_missing = sim_required - set(sbody)
    if sim_missing:
        die(f"simulate missing contract fields: {sorted(sim_missing)}")
    ok("simulate returns full contract")

    # --- infer (POST) ---
    inf = client.post(f"/api/infer?city={CITY}")
    if inf.status_code != 200:
        die(f"POST /api/infer -> HTTP {inf.status_code}")
    ibody = inf.json()
    if "network" not in ibody or not ibody["network"].get("features"):
        die("infer returned empty network")
    ok("infer returns non-empty network")

print("==> TESTCLIENT mode: all assertions passed")
PYEOF
}

# ---------------------------------------------------------------------------
echo "Route Resilience :: e2e smoke test"
if server_up; then
  run_curl_mode
else
  run_testclient_mode
fi
echo "ALL SMOKE CHECKS PASSED"
