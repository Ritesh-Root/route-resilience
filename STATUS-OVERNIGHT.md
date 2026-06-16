# Overnight progress — Route Resilience

Worked the whole local plan while you slept. **The one thing only you can do is run the Kaggle training** (it needs your login + GPU toggle + dataset add). Everything around it is done, verified, and pushed.

---

## ✅ Done & verified

### 1. Backend P0 bug fixes (FLEET-REPORT §2) — **82/82 tests pass**
- **Resilience-curve chart bug fixed** — `resilience_curve` now normalizes against the intact `BASE_EFF`, so the fragmented view correctly starts at **efficiency 0.746** instead of a misleading 1.0. (Updated the test that had encoded the bug.)
- **`/api/simulate` DoS bounded** — `disabledEdgeIds`/`disabledNodeIds` capped at 500 items, each id ≤64 chars; `top_k` constrained to 1–100. Oversized payloads now return **422**.
- **`run.sh` fixed** — creates + activates `.venv` and uses `python -m uvicorn` (works on a fresh clone).
- **`requirements.txt` pinned** to the proven versions (fastapi 0.137.1, networkx 3.6.1, numpy 2.4.6, pydantic 2.13.4, uvicorn 0.49.0, httpx 0.28.1).

### 2. Real Bengaluru roads — **the grid is gone** 🎉
- New `scripts/fetch_bengaluru_osm.py` pulls a real central-Bengaluru drive network from the **OpenStreetMap Overpass API** (pure stdlib — no GDAL/osmnx), collapses it into an intersection graph, and writes `backend/app/data/real_network.geojson` (**367 nodes / 480 edges**).
- `network_factory` now auto-serves that file when present (synthetic fallback otherwise). The map now shows **actual Bengaluru arterials**, not graph paper.
- **The demo story still holds exactly on real roads:** intact combos **RI 100**, fragmented `occluded+baseline` **RI 79** (29 disconnected zones). `/api/health` and `/api/network` now report `networkSource: "real"`.
- Tests pinned to the synthetic grid via `conftest.py` so they stay deterministic regardless of the committed real graph.

### 3. Real-graph integration seam
- `graph_build.geojson_to_graph()` reverses `graph_to_geojson`, so **any** GeoJSON in the backend schema (OSM today, or your trained-model extraction later) drops straight in via `real_network.geojson`. Criticality / gatekeepers / simulate are unchanged.

### 4. Kaggle notebook extended
- Added a **visual-proof inference cell** (predicts roads on clean + occluded val tiles, saves orange-overlay grids for slides) and a **wiring cell** explaining how to feed real metrics + an optional model-extracted graph back into the backend.

### 5. Frontend polish (FLEET-REPORT §3 gaps)
- WhatIfCard loading/error handling + `.catch`, removed KpiGrid's fabricated "vs baseline" delta, defensive `[0,1]` clamp on the resilience chart. *(Build status confirmed in the final commit message.)*

---

## ⏳ The ONE step left for you — train on Kaggle (~$0, ~3–4h)

1. Kaggle → New Notebook → **File → Import Notebook → GitHub** →
   `https://github.com/Ritesh-Root/route-resilience/blob/main/ml/notebooks/kaggle_train_deepglobe.ipynb`
2. Right panel: **GPU on**, **Internet on**, **Add dataset `balraj98/deepglobe-road-extraction-dataset`**.
3. **Run All.** It clones the repo, preps chips, smoke-tests, trains baseline + robust, prints real IoU/Dice/**Occlusion-Recall**, and saves overlays + checkpoints.

### When it finishes
- Copy the printed IoU/Dice/Occlusion-Recall into `backend/app/main.py:_METRIC_TABLE` (replaces the placeholder constants — see notebook's wiring cell).
- (Optional) With a georeferenced tile, extract a model graph → `real_network.geojson` to show *your* extraction on the map instead of OSM.

---

## Run it locally right now
- Backend (real roads) is **running on http://127.0.0.1:8000** (`networkSource: real`).
- Frontend dev server on **http://localhost:8080**.
- Restart anytime: `cd backend && ./run.sh` then `cd atlas-vision && bun run dev`.
- Re-pull fresh OSM roads: `python scripts/fetch_bengaluru_osm.py` then restart the backend.

## Still honest about limits
- No model trained yet (that's your Kaggle run). The "robust vs baseline" fragmentation is still the synthetic `_fragment()` applied to the **real** OSM graph — so the topology is real, the occlusion effect is simulated until the trained model lands.
- `/api/metrics` IoU/Dice/etc. remain placeholders until you paste the Kaggle numbers in.
