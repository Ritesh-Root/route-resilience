# Route Resilience — Checkpoints & Progress Tracker

> Resumable build log for **Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility** (Bhartiya Antariksh Hackathon).
> Update the status boxes as you go. Read the **"RESUME HERE"** line first whenever you come back.

**Status legend:** ✅ done · 🔄 in progress · ⬜ not started · ⏭️ optional/stretch

---

## ▶ RESUME HERE
**Last completed:** Backend built + smoke-tested (FastAPI + real criticality engine, synthetic Bengaluru net).
**Do next:** (1) Front-end build from the design prompt and point it at `http://localhost:8000`. (2) CV team: download DeepGlobe + pull OSM Bengaluru graph, get baseline D-LinkNet inference running. See **Phase 2** and **Phase 3**.

---

## Phase 0 — Planning & scaffolding
- [✅] Pick problem statement (Route Resilience)
- [✅] Feasibility analysis vs other statements → `hackathon-feasibility-analysis.md`
- [✅] System architecture + repo tree + sprint plan → `route-resilience-architecture.md`
- [✅] Architecture diagram (pipeline)
- [✅] Confirm datasets + repos with live links (see PROJECT-PLAN §4–5)
- [✅] Master plan → `PROJECT-PLAN.md`

## Phase 1 — Backend (API + criticality engine)  ✅ DONE
- [✅] FastAPI app + CORS (`backend/app/main.py`)
- [✅] Schemas mirroring front-end contract (`backend/app/schemas.py`)
- [✅] Synthetic Bengaluru network + occluded/robust variants (`services/network_factory.py`)
- [✅] Real criticality engine: betweenness, bridges, global efficiency, resilience curve, what-if simulation (`services/criticality.py`)
- [✅] GeoJSON IO + Gatekeeper nodes (`services/graph_build.py`)
- [✅] Occlusion generators for training (`services/occlusion.py`)
- [✅] Segmentation interface stub (`services/segmentation.py`)
- [✅] Smoke-tested every endpoint (robust Resilience 100 vs baseline-occluded 79)
- [ ] ⬜ Wire real extracted graph into `/api/network` once the model exists
- [ ] ⏭️ Add OD-demand-weighted criticality (needs an OD matrix)

## Phase 2 — Front-end (dashboard)  🔄 (you)
- [ ] 🔄 Generate UI from the design prompt (v0 / Lovable / Bolt / Claude)
- [ ] ⬜ Map view with criticality-colored roads + Gatekeeper markers
- [ ] ⬜ Controls: city, pipeline-stage, clean↔occluded, baseline↔robust
- [ ] ⬜ KPI panel (IoU, Dice, Occlusion-Recall, Connectivity, APLS, Resilience Index)
- [ ] ⬜ What-if simulator (disable edge/node → call `/api/simulate`)
- [ ] ⬜ Resilience curve + criticality histogram (Recharts)
- [ ] ⬜ Before/after split slider (baseline vs robust)
- [ ] ⬜ Replace mock `lib/api.ts` with calls to `http://localhost:8000/api/*`

## Phase 3 — Data & ground truth  ⬜ (CV/GIS)
- [ ] ⬜ Download **DeepGlobe** road dataset (Kaggle) → `data/raw/`
- [ ] ⬜ Download **SpaceNet** SN3/SN5 from `s3://spacenet-dataset/` (centerlines + APLS)
- [ ] ⏭️ Pull **OpenSatMap** (HuggingFace `z-hb/OpenSatMap`) for higher-res fine-tuning
- [ ] ⬜ Pull **Bengaluru** drive graph via OSMnx (validation + live demo)
- [ ] ⬜ Tiling + normalize to 512×512; 80/10/10 split (no spatial leakage)

## Phase 4 — Model (occlusion-robust extraction)  ⬜ (CV/ML)
- [ ] ⬜ Baseline D-LinkNet inference on pretrained weights (end-to-end on one tile FIRST — de-risk)
- [ ] ⬜ Fine-tune D-LinkNet (`segmentation_models_pytorch`, ResNet34, BCE+Dice)
- [ ] ⬜ Implement synthetic occlusion aug (reuse `services/occlusion.py`)
- [ ] ⬜ Add consistency loss (clean vs occluded prediction) — the robustness claim
- [ ] ⏭️ Swap in CoANet for connectivity/occlusion if time allows
- [ ] ⬜ Build occluded test set; report F1/IoU + Occlusion-Recall retention vs baseline

## Phase 5 — Vectorization & graph healing  ⬜ (CV + Graph)
- [ ] ⬜ Mask → threshold → morphology → skeletonize (scikit-image / FilFinder)
- [ ] ⬜ Skeleton → graph via cresi; assign speed/travel-time
- [ ] ⬜ MST / disjoint-set gap-closing ("healing"); measure Connectivity Ratio
- [ ] ⬜ APLS vs OSM graph (CosmiQ/apls)
- [ ] ⬜ Feed real graph into the backend (replaces synthetic network)

## Phase 6 — Integration & demo  ⬜
- [ ] ⬜ Front-end ↔ backend end-to-end on a real Bengaluru tile
- [ ] ⬜ Before/after occlusion story validated (robust recovers critical links)
- [ ] ⬜ What-if disruption demo (disable a Gatekeeper → rerouting + Resilience drop)
- [ ] ⬜ Slides: problem, method, robustness table, criticality map, live demo
- [ ] ⬜ Record a backup demo video

---

## Quick reference

**Run the backend**
```bash
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000
```
**Key files:** `backend/app/services/criticality.py` (analytics), `network_factory.py` (graph), `occlusion.py` (training aug), `segmentation.py` (model hook).
**Companion docs:** `PROJECT-PLAN.md` (data, repos, API contract, eval) · `../route-resilience-architecture.md` (full architecture + sprint plan) · `../hackathon-feasibility-analysis.md`.

**Owner split:** CV/ML → Phase 4 · Graph → Phase 5 + criticality · GIS → Phase 3 + OSM/APLS · Full-stack → Phase 2 + integration.

## Decisions log
- 2026-06-16 — Chose Route Resilience over 14 other statements (best repo + data readiness, strong demo).
- 2026-06-16 — Backend = FastAPI; criticality = NetworkX; demo city = Bengaluru.
- 2026-06-16 — Extractor = D-LinkNet baseline, CoANet as occlusion-aware upgrade; vectorize via cresi; metric = APLS.

## Open questions
- Final demo city/AOI within Bengaluru? (drives the OSM pull)
- Do we have/want an OD demand matrix? (upgrades criticality to demand-weighted)
- GPU availability for the 30-hr training window?
