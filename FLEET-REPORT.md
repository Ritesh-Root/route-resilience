# Route Resilience — Fleet Synthesis Report

**Project:** Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
**Date:** 2026-06-17
**Constraints honored:** Python 3.14 + 4 GB RTX 3050 laptop. No datasets downloaded, no `torch` installed, **no model training run**. All numbers below come from the synthetic NetworkX network and the committed test suite, executed in `backend/.venv` (Python 3.14, networkx 3.6.1, numpy 2.4.6).

---

## 1. Executive summary — what the fleet changed

The fleet delivered a **runnable, end-to-end demo skeleton** plus a complete (but un-trained) ML scaffold:

- **Backend (`backend/`)** — A self-contained FastAPI + NetworkX service implementing all 8 contract endpoints on a synthetic 9×9 Bengaluru grid (85 nodes / 148 edges + 4 suburb spurs). `resilienceIndex` is computed **live** from the real graph (not hardcoded). The demo invariant holds exactly: `occluded+baseline` fragments to **RI 79 / 6 components**; every other combo is **RI 100 / 1 component**. **82/82 backend tests pass.**
- **Frontend (`atlas-vision/`)** — The previously-mocked `src/lib/api.ts` was rewritten to **really `fetch()`** the backend, with a graceful mock fallback on any error. All contract gaps in `types.ts` were reconciled (`roadClass`, `isArticulation`, `disabledNodeIds`, `brokenRoutesSampled`, `sampledRoutes`, `meta`). Added `src/lib/config.ts` (`API_BASE`), `.env.example`, and client-side criticality histogram. Per the verify phase, **`bun run build` and `tsc --noEmit` both pass.**
- **ML scaffold (`ml/`)** — 13 new modules (~5,570 lines): occlusion aug, D-LinkNet/CoANet model, SpaceNet/OSMnx loaders + tiling, train loop + consistency loss, APLS + seg metrics, skeletonize → graph → MST healing vectorizer, and an **integration seam** (`ml/integration/segmenter_hook.py`) that produces a graph in the exact shape the backend serves. Every heavy import (torch/rasterio/skimage/osmnx) is **lazy-guarded** so modules import on the bare 4 GB box.
- **Docs/demo (`demo/`, `scripts/`)** — Demo script, slides outline, eval-metrics writeup, and smoke/run shell scripts.

**Important caveat on the audit digest:** The phase-1 audit digest describes many backend fixes (e.g. threading `BASE_EFF` into `resilience_curve`, pinning `requirements.txt`, activating the venv in `run.sh`, bounding `disabledEdgeIds`). **These were largely NOT applied** — see §2. The audit's proposed line numbers (and the `network.py`/`graph_metrics.py` filenames in some notes) do not match the shipped tree (`network_factory.py`/`criticality.py`). The frontend reconciliations, by contrast, **were** applied and verified.

---

## 2. Backend audit findings — status table

Verified against the actual shipped files. "Real" = genuinely affects correctness/security; "Benign" = harmless for the demo as-is.

| Sev | Finding | Location (actual) | Status / Real vs Benign | Suggested fix |
|---|---|---|---|---|
| **HIGH** | `resilience_curve` normalizes against a **local** `base_eff`, so the **fragmented** curve falsely starts at efficiency **1.0** | `criticality.py:82,96-97`; called `main.py:83` | **UNFIXED — REAL.** Confirmed: frag curve `efficiency[0]=1.0` (should be < 1.0). Cosmetically misleads the resilience chart for the headline view. | Add `base_eff: float` param; pass `network_factory.BASE_EFF` from `main.py`; normalize against it. |
| **HIGH** | `resilience_index` efficiency not comparable across node counts under **node** removal | `criticality.py:67-73,157` | **UNFIXED — partly real.** Demo only disables edges + degree-1 spur nodes, so it does not bite the scripted story; would matter for arbitrary node removal. | Normalize against a fixed node set (keep removed nodes isolated) or divide by `base_n*(base_n-1)`. |
| **HIGH** | `/api/metrics` `KeyError` if a new `(input,model)` combo lacks a table entry | `main.py:73` (`_METRIC_TABLE[(input,model)]`) | **UNFIXED — low real risk.** Pydantic `Literal` constrains the params to the 4 keyed combos, so it cannot trigger today; brittle on extension. | Use `.get(...)` with a default dict, or assert table covers the `Literal` product. |
| **HIGH** | `POST /api/simulate` accepts **unbounded** `disabledEdgeIds`/`disabledNodeIds` (DoS) | `schemas.py:40-41` | **UNFIXED — REAL** (open CPU-exhaustion vector for a public deploy; fine for localhost demo). | `Field(max_length=200)` + per-item `StringConstraints(max_length=64)`. |
| **HIGH** | `run.sh` never activates `.venv`; installs/run against the wrong interpreter | `run.sh:5-6` | **UNFIXED — REAL** for a fresh clone (works here only because `.venv` already exists and devs run `.venv/bin/...` directly). | Create+activate venv, use `python -m uvicorn`. A working `.venv` is already present. |
| MED | `simulate_removal` echoes back **non-existent** `disabledNodeIds` verbatim (asymmetric with edges) | `criticality.py:156` | **UNFIXED — REAL (minor).** Node-removal branch has no test coverage. | Filter to nodes actually present/removed, mirroring the edge logic; add a test. |
| MED | `sampledRoutes` is the **requested budget** (constant 60), not routes actually evaluated; not comparable to `brokenRoutesSampled` | `criticality.py:161` | **UNFIXED — REAL (semantic).** Frontend divides the two; ratio is incoherent. | Count actually-evaluated base-valid routes and return that. |
| MED | Wildcard CORS (`allow_origins=["*"]`) | `main.py:24-25` | **UNFIXED — benign for localhost demo, real for deploy.** `allow_credentials` is not set, so no cookie leak. | Restrict to `http://localhost:5173/3000`, methods `GET,POST`. |
| MED | `top_k` unbounded/unvalidated int | `main.py:63` | **UNFIXED — minor real.** Negative/huge values pass through. | `Query(8, ge=1, le=100)`. |
| MED | `requirements.txt` uses floor pins (`>=`) — non-reproducible; `numpy>=1.26` stale for py3.14 | `requirements.txt` | **UNFIXED — REAL (repro).** Installed/verified versions are fastapi 0.137.1, networkx 3.6.1, numpy 2.4.6, pydantic 2.13.4. | Pin `==` to the proven versions; bump numpy floor to `>=2.1`. |
| MED | `/api/network` & `InferResponse.network` have no typed `response_model`; edge-property contract unvalidated | `main.py:47-56`, `schemas.py:56` | **UNFIXED — benign.** Output is correct; just not schema-enforced. | Add a typed `FeatureCollection`/`Meta` model. |
| LOW | `graph_to_geojson` emits extra `source`/`target` props beyond the "EXACT" contract set | `graph_build.py:18-19` | **Benign / arguably good.** Harmless extra data; frontend `RoadFeatureProperties` can mark them optional. | Document them in the contract (recommended) or drop. |
| LOW | Empty/edgeless graph → `ZeroDivisionError` in `resilience_curve` | `criticality.py:86,95` | **UNFIXED — latent.** Cannot occur on the synthetic graph (148 edges); would bite the future real-inference path. | Early return when `m==0`. |
| LOW | `isArticulation` is wired but **never `True`** in any realistic gatekeeper response | `graph_build.py:43-49` | **REAL coverage gap.** The 4 articulation points (spur anchors) have low betweenness, so they never make `top_k`; only test is a trivial `isinstance(bool)`. | Add a unit test on a hand-built graph with a known cut vertex; consider surfacing articulation independent of betweenness rank. |
| LOW | `newlyDisconnectedZones` can count stranded single nodes; `_fragment` magic constants (seed=13, drop=0.30) undocumented and seed-fragile | `criticality.py:130-132`, `network_factory.py:99-117` | **Benign but fragile.** Produces the correct 79/6 today; a refactor of edge-iteration order could silently drift it. | Count only components of size ≥2; assert the RI band + giant<1 / comps>1 in a test; document the constants. |

**Bottom line:** The backend is **correct and demo-ready as a synthetic service**, and the criticality math that powers the *headline* RI 79-vs-100 is genuinely computed and tested. The real, worth-fixing items before any non-localhost use are: the `resilience_curve` base-efficiency bug (#1, visibly wrong chart), input bounding on `/simulate` (DoS), `run.sh` venv activation (fresh-clone breakage), and pinning `requirements.txt`.

---

## 3. Frontend wiring status

| Question | Answer |
|---|---|
| **Does it build?** | **Yes.** Verify phase: `bun run build` PASS (client+ssr, exit 0) and `bunx tsc --noEmit` PASS (no type errors). Only non-fatal warnings (vite-tsconfig-paths deprecation, "no Lovable context"). |
| **Does `api.ts` hit the real backend?** | **Yes.** `src/lib/api.ts` now uses `fetch()` against `${API_BASE}/api/...` for `/network`, `/metrics`, `/gatekeepers`, `/resilience-curve`, and `POST /simulate`. `API_BASE` comes from `src/lib/config.ts` (`VITE_API_BASE ?? "http://localhost:8000"`); `.env.example` is committed. Each call falls back to bundled mock data on any non-2xx/error so the demo never hard-crashes. |
| **Contract types reconciled?** | **Yes.** `types.ts` now has `RoadFeatureProperties.roadClass`, `GatekeeperNode.isArticulation`, `SimulationResult.{disabledNodeIds, brokenRoutesSampled, sampledRoutes}`, and `RoadNetwork.meta`. |
| **Criticality histogram?** | Computed **client-side** in `api.ts:fetchCriticalityHistogram` from `/api/network` `properties.criticality` (5 fixed buckets), matching the "no backend endpoint" guidance, with a mock fallback. |

**Remaining frontend gaps (non-blocking):**
- **Range ambiguity (highest live-demo risk):** the resilience chart multiplies `efficiency`/`giantComponent` by 100 assuming 0..1 fractions. The backend **does** return 0..1, so it works — but there is no defensive clamp. Combined with backend finding #1 (frag curve starts at 1.0), the fragmented curve will render as if starting at 100% efficiency.
- `brokenRoutesSampled` / `sampledRoutes` are typed as `number` and divided in `WhatIfCard`; backend returns integers (counts), so this is consistent — but `sampledRoutes` is the constant 60, making the ratio less meaningful (backend finding above).
- `simulateDisable(display, disabledEdgeIds, disabledNodeIds)` signature differs from the audit's proposed `(city,input,model,...)` but is functionally correct and matches its call site.
- No `fetchCities`/`fetchHealth`/`postInfer` wrappers (low priority; only Bengaluru exists). The mock `CITIES` list is broader than the backend's `["Bengaluru"]` — drive the selector from `/api/cities` if you add more.
- `WhatIfCard` `loading`/`error` props are not supplied by `routes/index.tsx`, and the simulate effect has no `.catch` → silent failure / stale numbers if `/simulate` rejects. The mock fallback masks this in practice.
- `KpiGrid` "vs baseline" badge is hardcoded to `ri - 44` (a stale mock anchor); against the real 79/100 it overstates uplift. Cosmetic.

---

## 4. ML scaffold inventory — runnable now vs blocked

13 modules, ~5,570 lines. All **import-safe without torch** (heavy libs lazy-imported inside functions).

| Module | Lines | Runnable **now** (no GPU/data)? | Blocked on |
|---|---|---|---|
| `ml/aug/occlusion.py` | 336 | **Yes** — numpy-only cloud/shadow/canopy + consistency-mask helper; import-OK. | — |
| `ml/vectorize/skeletonize.py` | 404 | **Yes** — import-OK; thinning helpers (skimage lazy). | skimage for the real skeletonize call. |
| `ml/vectorize/healing.py` | 330 | **Yes** — import-OK; MST gap-bridging is pure networkx+numpy. Verify phase ran its demo (bridges a ~25 m gap, raises connectivity). | networkx must be installed (see §5). |
| `ml/vectorize/skeleton_to_graph.py` | 774 | **Self-test** runs (`--self-test`) and produces criticality identical to backend `_annotate`. | hard `import networkx` at top (not lazy) → needs networkx. |
| `ml/integration/segmenter_hook.py` | 531 | **Torch-free self-test** (`python ml/integration/segmenter_hook.py`) exercises mask→skeleton→graph→heal→annotate on a synthetic mask. **This is the swap-in seam.** | skimage for `skeletonize`; networkx (top-level import). |
| `ml/metrics/apls.py` | 385 | `--demo` runs (APLS 0.987 with one link removed). | networkx (top-level import). |
| `ml/metrics/seg_metrics.py` | 190 | **Yes** — IoU/Dice/occlusionRecall, numpy-only; import-OK. | — |
| `ml/models/coanet.py` | 364 | Imports OK (torch lazy). **Cannot instantiate/run** without torch. | torch + 4 GB+ GPU (CoANet is OOM-prone on 4 GB — documented as aspirational). |
| `ml/train/losses.py` | 415 | Import-OK; consistency loss (MSE/KL) defined, torch lazy. | torch to actually compute. |
| `ml/train/train.py` | 891 | Imports OK; full AMP + grad-accum loop. **Not run.** | torch + GPU + dataset. |
| `ml/data/spacenet.py` | 388 | Import-OK (rasterio lazy). | SpaceNet download (multi-GB — **forbidden on this box**) + rasterio/GDAL. |
| `ml/data/osmnx_bengaluru.py` | 196 | Import-OK (osmnx lazy). | osmnx + GDAL/GEOS/PROJ; network access for OSM. |
| `ml/data/tiling.py` | 368 | **Yes** — import-OK; 512px chip tiling. | rasterio for GeoTIFF I/O. |

**Net:** the **CPU/numpy/networkx half of the pipeline is real and runnable** (occlusion, seg-metrics, vectorize, healing, APLS, the integration seam self-test). The **GPU half (model + training + GeoTIFF/dataset loading) is scaffolding only** — correct code, never executed, blocked on torch + a bigger GPU + datasets.

> **`ml/README.md` mismatch:** the README sketches a flat layout (`ml/train.py`, `ml/vectorize.py`, `ml/data/loader.py`) that does **not** match the shipped nested layout (`ml/train/train.py`, `ml/vectorize/skeleton_to_graph.py`, `ml/data/spacenet.py`). The run-order commands in the README (`python -m ml.train ...`) will not resolve as written. Treat the shipped modules as ground truth.

---

## 5. Test + demo-story verification (actual numbers)

**Backend test suite (`backend/.venv/bin/python -m pytest -q`): 82 passed, 0 failed, 1 warning** (Starlette/httpx deprecation — cosmetic).

**Demo invariant — measured live from the synthetic graph (`BASE_EFF = 0.001184`):**

| input | model | resilienceIndex | edges | nodes | components | giant fraction |
|---|---|---|---|---|---|---|
| clean | robust | **100** | 148 | 85 | 1 | 1.000 |
| clean | baseline | **100** | 148 | 85 | 1 | 1.000 |
| occluded | robust | **100** | 148 | 85 | 1 | 1.000 |
| **occluded** | **baseline** | **79** | **107** | 85 | **6** | **0.894** |

✅ Invariant satisfied: **only `occluded+baseline` is fragmented (79); the other three are intact (100).** The robust model "recovers" the 41 links and 5 zones the baseline loses under occlusion.

**`/api/metrics` seg-quality (table-driven constants, not measured — see §7):** occluded+baseline `iou=0.71, dice=0.77, occlusionRecall=0.58, connectivityRatio=0.64, apls=0.61`; clean+robust `iou=0.89 ... apls=0.87`.

**`/api/simulate` (disable a top gatekeeper, from the occluded+baseline view, RI 79):**
- Top **edge** `e_n4_4__n5_4` (criticality 1.0): RI **79 → 78**, travel time **+3.1%**.
- Top **node** `n4_4` (betweenness 1.0, GK-1): RI **79 → 76**, travel time **+7.2%**, `brokenRoutesSampled=2`.

**Known true-but-unfixed observations surfaced by verification:**
- `resilience-curve` for the fragmented view **starts at efficiency 1.0** (should be ~0.86) — backend finding #1, REAL, unfixed.
- `/api/gatekeepers` returns `isArticulation=False` for the top-k in every realistic call (articulation nodes never rank into top-k).
- `sampledRoutes` is the constant 60, not a count of routes actually evaluated.

**ML compile/import check (verify phase):** all 13 files compile (no `SyntaxError`); 10/13 import cleanly; 3/13 (`apls.py`, `segmenter_hook.py`, `skeleton_to_graph.py`) fail import **only** because they `import networkx` at top-level and networkx is not installed in the *system* env (it IS in `backend/.venv`). Not a code bug — install networkx.

---

## 6. Prioritized TODO for a human

**P0 — make the demo airtight (minutes, local):**
1. Fix the visible chart bug: thread `BASE_EFF` into `criticality.resilience_curve(G, base_eff)` and call it from `main.py:83`, so the fragmented curve starts below 1.0 (backend finding #1).
2. Fix `backend/run.sh` to create + activate `.venv` and use `python -m uvicorn`; pin `backend/requirements.txt` to the proven `==` versions (fastapi 0.137.1, uvicorn[standard] 0.49.0, networkx 3.6.1, numpy 2.4.6, pydantic 2.13.4).
3. Start both processes and click through: `cd backend && ./run.sh` (→ :8000), then `cd atlas-vision && cp .env.example .env && bun install && bun run dev`. Confirm the map shows the fragmented occluded+baseline view and `/simulate` updates the What-If card.

**P1 — harden if exposed beyond localhost:**
4. Bound `/simulate` lists (`Field(max_length=...)`) and `top_k` (`Query(ge=1,le=100)`); restrict CORS to the dev origins.
5. Filter non-existent `disabledNodeIds`; return `sampledRoutes` = routes actually evaluated.
6. Add the missing `WhatIfCard` loading/error plumbing and a defensive clamp in the resilience-chart adapter.

**P2 — wire the real model (requires a bigger GPU + datasets, OFF this laptop):**
7. On a machine with ≥8 GB VRAM (or the documented Python 3.12 + CUDA env): `pip install -r ml/requirements-ml.txt`, install GDAL/GEOS/PROJ via conda-forge.
8. Download **DeepGlobe or SpaceNet** there (NOT on the 4 GB laptop). Tile to 512px via `ml/data/tiling.py`.
9. Train baseline then robust (`ml/train/train.py`, AMP + grad-accum, batch 1–2). Train robust with occlusion aug + consistency loss. Copy back only the checkpoints + a few sample chips.
10. **Wire the real graph via `ml/integration/segmenter_hook.py`:** point `network_factory.get_network()` at `build_real_network(image, transform, input_mode, model)` (one-line swap; downstream criticality/gatekeepers/simulate/GeoJSON are unchanged). Heal occlusion gaps with MST bridging (`heal_graph`); keep healing OFF for `occluded+baseline` to preserve the demo story.
11. Replace the `_METRIC_TABLE` constants with **measured** IoU/Dice/occlusionRecall/connectivity/APLS from `ml/metrics/`.
12. Install `networkx` in whatever env runs `ml/metrics/apls.py` / `ml/vectorize/skeleton_to_graph.py` standalone.

**P3 — docs truthfulness:**
13. Reconcile `ml/README.md`'s sketched layout/run-order with the shipped nested modules. Soften "tested"/"6 disconnected zones" claims to match reality (5 isolated suburb zones + 1 main component = 6 components; seg metrics are placeholders).

---

## 7. Honest limitations

- **No training was performed.** No `torch` is installed and no dataset was downloaded, per the hard constraints. The "robust beats baseline" story is **simulated** by the synthetic `network_factory` (`_fragment` drops 30% of local edges + 2 arterials under `occluded+baseline`), **not** produced by a trained model. The model (`coanet.py`), training loop (`train.py`), and consistency loss are correct-looking scaffolding that has **never executed**.
- **All `/api/metrics` values except `resilienceIndex` are hardcoded constants** in `main.py:_METRIC_TABLE`. `iou=0.71`, `apls=0.61`, etc. are **plausible placeholders, not measured results.** Only `resilienceIndex` is computed from the real graph.
- **Python 3.14 is bleeding-edge for ML.** Several pinned ML deps (torch, opencv, rasterio) may lack cp314 wheels; `requirements-ml.txt` honestly recommends a separate Python 3.12 + conda-forge env for the training box. Do not attempt the ML install on the 4 GB laptop.
- **4 GB VRAM is the binding constraint.** Even with AMP + batch 1 + 512px tiles, the heavier CoANet variant is expected to OOM and is documented as aspirational; D-LinkNet34 is the intended fit. Inference on the real path would need tiled (≤512px, overlap) fp16 inference.
- **The synthetic 9×9 grid is topologically redundant**, so the fragmented view stays *measurably* degraded (RI 79, 6 components, 41 fewer edges, dropped gatekeepers) but is not catastrophically shattered — the fragmentation is real and deterministic, but it is a *demo* network, not extracted from imagery.
- **Several audit-recommended backend fixes were not applied** (see §2): the report reflects the code as shipped, not as the audit proposed. The frontend reconciliations *were* applied and independently verified to build.
- **The UI was not rendered in a browser in this environment.** Build + typecheck pass; the live-render UX findings in the verify phase are from source tracing, not a running app.
