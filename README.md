# Route Resilience

**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**
Bhartiya Antariksh Hackathon (ISRO) · demo city: **Bengaluru**

Extract a city's road network from satellite imagery **even where roads are hidden** by tree canopy, building shadows, and cloud cover; "heal" the broken masks into a **connected, routable graph**; and run **graph-theoretic criticality analysis** to find the roads/intersections ("Gatekeeper Nodes") whose failure would fragment urban mobility — then let a planner run **what-if disruption simulations** (disable a road → instant rerouting cost + Resilience Index).

---

## Repository layout

```
route-resilience/
├── backend/            FastAPI + NetworkX service (8 endpoints) — BUILT & tested (82/82)
│   └── app/
│       ├── main.py             API endpoints
│       ├── schemas.py          request/response models (the frontend contract)
│       ├── services/
│       │   ├── network_factory.py   synthetic Bengaluru graph + occluded/robust variants
│       │   ├── criticality.py       REAL analytics: betweenness, bridges, resilience, what-if
│       │   ├── graph_build.py       graph ↔ GeoJSON + Gatekeeper nodes
│       │   ├── occlusion.py         synthetic occlusion augmentation
│       │   ├── segmentation.py      model hook (stub until a trained model lands)
│       │   └── geo.py
│       └── tests/              pytest suite (per-endpoint + criticality units)
├── atlas-vision/       React + TanStack + Leaflet dashboard (wired to the live backend)
├── ml/                 Occlusion-robust extraction pipeline (scaffold; CPU half runs, GPU half un-trained)
│   ├── data/           DeepGlobe / SpaceNet loaders · OSMnx Bengaluru puller · tiling
│   ├── models/         D-LinkNet (smp) · CoANet (occlusion-aware)
│   ├── aug/            synthetic cloud/shadow/canopy occlusion
│   ├── train/          BCE+Dice + clean-vs-occluded consistency loss · training loop
│   ├── inference/      tile → mask prediction
│   ├── vectorize/      skeletonize → graph → MST gap-healing
│   ├── metrics/        APLS · IoU / Dice / Occlusion-Recall
│   └── integration/    segmenter_hook.py — swap-in seam into the backend
├── demo/               DEMO-SCRIPT.md · SLIDES-OUTLINE.md · EVAL-METRICS.md
├── scripts/            run_all.sh · e2e_smoke.sh
├── PROJECT-PLAN.md · route-resilience-architecture.md · ROUTE-RESILIENCE-CHECKPOINTS.md
└── FLEET-REPORT.md     audit findings, verification results, remaining TODO
```

---

## Quickstart (local)

**Backend** (Python 3.12+; pure-Python deps):
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000   # docs at /docs
```

**Frontend** (Bun):
```bash
cd atlas-vision
cp .env.example .env          # VITE_API_BASE defaults to http://localhost:8000
bun install
bun run dev                   # serves http://localhost:8080
```

Open **http://localhost:8080**. The dashboard fetches the live backend and falls back to bundled mock data if it's unreachable, so the demo never hard-crashes.

**Run the test suite:**
```bash
cd backend && .venv/bin/python -m pytest -q     # 82 passed
```

---

## API contract

| Method · Path | Query / Body | Returns |
|---|---|---|
| `GET /api/health` | — | status + base efficiency |
| `GET /api/cities` | — | `["Bengaluru"]` |
| `GET /api/network` | `city, input(clean\|occluded), model(baseline\|robust)` | GeoJSON; per-edge `{id, criticality 0..1, travelTimeSec, lengthM, roadClass, isBridge}` |
| `GET /api/gatekeepers` | `city, input, model, top_k` | `[{id, lat, lng, betweenness, isArticulation, label}]` |
| `GET /api/metrics` | `city, input, model` | `{iou, dice, occlusionRecall, connectivityRatio, apls, resilienceIndex}` |
| `GET /api/resilience-curve` | `city, input, model` | `{removedFraction[], efficiency[], giantComponent[]}` |
| `POST /api/simulate` | `{city, model, input, disabledEdgeIds[], disabledNodeIds[]}` | `{resilienceIndexAfter, avgTravelTimeIncreasePct, newlyDisconnectedZones, ...}` |
| `POST /api/infer` | `city` | stub: network as if freshly inferred |

**Demo story:** `input=occluded & model=baseline` → fragmented network (Resilience Index **79**, several disconnected zones). Every other combo → intact network (Index **100**) — the robust model recovers the critical links the baseline loses under occlusion.

---

## Current status & honest limitations

- The **backend criticality engine is real** — betweenness, weighted global efficiency, bridges, resilience curve, and what-if simulation are computed live in NetworkX. `resilienceIndex` is derived from the actual graph.
- The road network is a **synthetic Bengaluru graph** standing in for a model-extracted one; **no model has been trained** and **no datasets are bundled**. The segmentation-quality metrics (IoU/Dice/APLS/…) in `/api/metrics` are **placeholder constants**, not measured results.
- The `ml/` pipeline is **runnable scaffolding**: the CPU/NetworkX half (occlusion aug, vectorize, MST healing, APLS, the integration seam) runs today; the GPU half (D-LinkNet/CoANet + training + dataset loaders) is correct, import-safe code that has not been executed (needs ≥8 GB VRAM + the datasets).
- To go from synthetic to real: train on DeepGlobe/SpaceNet, then point `network_factory.get_network()` at `ml/integration/segmenter_hook.py` — everything downstream (criticality, gatekeepers, simulate, GeoJSON) is unchanged.

See **`FLEET-REPORT.md`** for the full audit, verification numbers, and prioritized TODO.

---

## Tech stack

FastAPI · NetworkX · NumPy · Pydantic (backend) · React 19 · TanStack Router/Query · Leaflet · Recharts · Tailwind (frontend) · PyTorch · segmentation_models_pytorch · albumentations · scikit-image · rasterio · OSMnx (ML, scaffold).
