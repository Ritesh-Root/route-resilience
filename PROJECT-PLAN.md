# Route Resilience — Master Project Plan

**Problem statement:** Route Resilience — Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
**Event:** Bhartiya Antariksh Hackathon (ISRO) · default city **Bengaluru**
**Team:** strong across DL/CV, classical ML + signal, remote sensing/GIS, full-stack/systems · **sprint** timeline

---

## 1. What we are building (in one paragraph)

We extract a city's road network from satellite imagery **even where roads are hidden** by tree canopy, building shadows, and cloud cover; we "heal" the broken pixel masks into a **connected, routable graph**; and we run **graph-theoretic criticality analysis** to find the roads and intersections ("Gatekeeper Nodes") whose failure would fragment urban mobility — then let a planner run **what-if disruption simulations** (disable a road → instant rerouting cost + Resilience Index). Output = an automated, occlusion-robust road map plus predictive what-if simulations for disaster response and resilience planning.

This is the "dual challenge of urban spatial modelling": (1) **fragmentation** — spectral blindness produces broken masks that are useless for routing; (2) **stagnation** — legacy methods can't keep pace with fast-growing Indian metros. Our pipeline addresses both, and maximises downstream utility of indigenous EO satellites (Cartosat, Resourcesat LISS-IV), aligned with ISRO's NNRMS mandate.

## 2. Four core objectives

1. **Occlusion-Aware Extraction** — a deep model that infers road continuity under canopy/shadow/cloud and urban clutter; robust across illumination and seasons.
2. **Topological Reconstruction** — convert fragmented pixel masks into a unified, routable weighted graph via graph-theoretic "healing" (skeletonize → graph; MST / disjoint-set gap closing).
3. **Structural Intelligence** — quantify urban vulnerability by identifying Gatekeeper Nodes / bottlenecks via centrality metrics.
4. **Simulated Stress Testing** — predict the systemic impact of localised infrastructure failure (flooding, accidents) and produce a Resilience Index.

## 3. Expected deliverables

- **High-fidelity routable topology** — a mathematically closed, connected vector road network from high-res imagery (urban, rural, forested terrain).
- **Quantitative criticality map** — spatial heatmap of high-betweenness intersections that act as single points of failure (the "Gatekeeper Nodes").
- **Predictive impact assessment** — a simulation that disables nodes/edges and instantly shows rerouting effects, travel-time increase, and the Resilience Index for disaster scenarios.

---

## 4. Data — arranged and ready

### Primary imagery (the input we run on)
| Source | Resolution | Access |
|---|---|---|
| Sentinel-2 | 10 m | Free — Copernicus / Google Earth Engine |
| Resourcesat **LISS-IV** | 5.8 m | Free — ISRO Bhoonidhi (indigenous, NNRMS alignment) |
| **Cartosat-3** | sub-metre | Provided during the 30-hour hackathon for challenge-specific experimentation |

### Training / ground-truth datasets (to build & pre-train the extractor)
| Dataset | What it gives | Link |
|---|---|---|
| **SpaceNet Roads (SN3/SN5)** | 8,000 km road centerlines + attributes (type, lanes) + travel-time labels; 30 cm WorldView-3; ships the **APLS** metric | spacenet.ai/spacenet-roads-dataset · `s3://spacenet-dataset/spacenet/SN3_roads/` (AWS Open Data) |
| **DeepGlobe Road Extraction** | RGB + binary road masks; simplest to start | Kaggle: balraj98/deepglobe-road-extraction-dataset |
| **OpenSatMap** (NeurIPS 2024) | Largest, highest-res road dataset; instance-level annotations (L19 0.3 m / L20 0.15 m); aligns with nuScenes/Argoverse | huggingface.co/datasets/z-hb/OpenSatMap · github.com/OpenSatMap/OpenSatMap-offical · opensatmap.github.io |
| **Massachusetts Roads** | Large, easy aerial road masks | mnih thesis dataset (widely mirrored) |
| **OpenStreetMap via OSMnx** | Auto-generated ground-truth road graphs for any city — training masks, validation, and the live Bengaluru demo graph | github.com/gboeing/osmnx |

> **Ground-truth shortcut:** pair multi-resolution satellite feeds with OSM-derived vectors → near-zero manual annotation. Use OSMnx to pull the Bengaluru drive network for validation and the live demo.

## 5. Existing repos — arranged by pipeline stage (fork, don't rebuild)

**Segmentation / extraction**
- `zlckanata/DeepGlobe-Road-Extraction-Challenge` — **D-LinkNet** (ResNet34 + dilated center), 1st-place DeepGlobe. Baseline extractor.
- `mj129/CoANet` — **Connectivity Attention Network**; explicitly targets occluded roads → strong fit for our occlusion-aware objective and the "attention-based" framing.
- `qubvel/segmentation_models.pytorch` — LinkNet/Unet++ /DeepLabV3+ with pretrained encoders; fastest way to a D-LinkNet-equivalent in a few lines.
- `fudan-zvg/RoadNet` — transformer road-network extraction (advanced / stretch).
- *Occlusion-aware references:* OARENet, CAFormer, Seg-Road (papers; adopt ideas).

**Vectorization → routable graph**
- `avanetten/cresi` — satellite → **road-network graph with speed + travel time**; reuse its skeleton→graph stages. Backbone of topological reconstruction.
- `e-koch/FilFinder` — medial-axis **skeletonization** of masks into clean filaments.
- `gboeing/osmnx` — download/model/validate street-network graphs from OSM.

**Metrics**
- `CosmiQ/apls` — **APLS** (Average Path Length Similarity): topological-accuracy metric used to score SpaceNet roads.

**Meta-index**
- `satellite-image-deep-learning/techniques` — catalogue of alternatives / plan B.

## 6. Tech stack

| Layer | Choice |
|---|---|
| DL framework | PyTorch |
| Extractor | `segmentation_models_pytorch` (LinkNet/ResNet34) → optional CoANet for occlusion |
| Augmentation | `albumentations` + **custom occlusion** (clouds/shadows/canopy) ← innovation |
| Raster/geometry | rasterio, GDAL, OpenCV, scikit-image, shapely |
| Skeleton→graph | cresi + FilFinder + scikit-image |
| Graph logic | **NetworkX** (PyTorch-Geometric only if we add a GNN); MST/disjoint-set healing |
| Centrality | betweenness centrality (NetworkX) |
| Backend | **FastAPI** (built — see §8) |
| Visualization / demo | React + Leaflet/Folium, QGIS, Matplotlib, Streamlit |
| Compute | Graph + UI run on CPU; training the DL model needs a GPU within the 30-hr window |

## 7. Architecture (summary)

Pipeline: **imagery + masks → synthetic occlusion augmentation → D-LinkNet/CoANet segmentation → vectorize (skeleton→graph, MST healing) → weighted NetworkX graph → criticality (betweenness · bridges · edge-removal resilience curve) → interactive resilience map.**

Full component design, repo tree, and the day-by-day sprint plan live in **`route-resilience-architecture.md`** (outputs root). The robustness innovation is concentrated in two modules: `occlusion.py` (synthetic occluders + consistency loss) and `criticality.py` (graph analytics).

## 8. Backend — DONE and tested (this folder, `backend/`)

A working **FastAPI** service runs today on a synthetic Bengaluru network, so the front-end can integrate immediately. When the trained model is ready, swap `RoadSegmenter.predict` + vectorization into `services/segmentation.py`; everything downstream is unchanged. The criticality engine (`services/criticality.py`) is the **real** analysis — betweenness, bridges, weighted global efficiency, resilience curve, and what-if simulation.

**Endpoints (API contract the front-end builds against):**
| Method · Path | Query / Body | Returns |
|---|---|---|
| GET `/api/health` | — | status + base efficiency |
| GET `/api/cities` | — | `["Bengaluru"]` |
| GET `/api/network` | `city, input(clean\|occluded), model(baseline\|robust)` | GeoJSON FeatureCollection; per-edge `{id, criticality 0..1, travelTimeSec, lengthM, roadClass, isBridge}` |
| GET `/api/gatekeepers` | `city, input, model, top_k` | `[{id, lat, lng, betweenness, isArticulation, label}]` |
| GET `/api/metrics` | `city, input, model` | `{iou, dice, occlusionRecall, connectivityRatio, apls, resilienceIndex}` |
| GET `/api/resilience-curve` | `city, input, model` | `{removedFraction[], efficiency[], giantComponent[]}` |
| POST `/api/simulate` | `{city, model, input, disabledEdgeIds[], disabledNodeIds[]}` | `{resilienceIndexAfter, avgTravelTimeIncreasePct, newlyDisconnectedZones, ...}` |
| POST `/api/infer` | `city` | stub: returns network as if freshly inferred |

**The demo story is wired in:** `input=occluded & model=baseline` returns a **fragmented** network (Resilience Index ≈79, 6 disconnected zones); `model=robust` returns the intact network (Index 100) — proving the occlusion-robust model recovers critical links the baseline loses.

**Run it:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
```

## 9. Front-end

You are building this. The ready-to-paste design prompt (dark mission-control dashboard: pipeline strip, criticality map, Gatekeeper markers, what-if simulator, KPI panel, before/after slider) was delivered in chat and targets the contract in §8. Point its `lib/api.ts` at `http://localhost:8000`.

## 10. Evaluation parameters (how we're scored)

- **IoU & Dice** — segmentation accuracy, with special focus on **Occlusion-Recall** (recovery of roads under shadow/canopy/cloud).
- **Generalisation** — across dense urban, forested suburban, rural terrain.
- **Connectivity Ratio** — increase in the largest connected component after the MST "healing" phase.
- **Topological Accuracy (APLS)** — final graph vs OSM benchmark via Average Path Length Similarity.
- **Length-Complete / Relaxed IoU** — 3–5 px tolerance buffer so minor alignment shifts aren't over-penalised.

## 11. How the pieces connect

```
[Front-end React dashboard]  ←HTTP→  [FastAPI backend /api/*]  ←  [criticality engine]
        you build                         built (this folder)        real NetworkX
                                                  ↑
                                   [segmentation model + vectorize]  ← next big task (CV team)
                                   trains on SpaceNet/DeepGlobe/OpenSatMap,
                                   validates vs OSM (APLS)
```

See **`CHECKPOINTS.md`** for the resumable, phase-by-phase task tracker.
