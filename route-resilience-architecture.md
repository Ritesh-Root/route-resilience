# Route Resilience — Architecture & Build Plan

**Project:** Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
**Mode:** Hackathon sprint (days–weeks), cross-disciplinary team
**One-line pitch:** *Extract a city's road network from satellite imagery even when roads are hidden by clouds/shadows/trees, turn it into a graph, and rank which roads are the most critical single points of failure for urban mobility.*

---

## 1. The story (what the judges should remember)

Two well-solved problems, joined to make something new and useful:

1. **Robust extraction** — most road-extraction models break when roads are occluded (clouds, building/tree shadows, canopy). We train a model that stays accurate under occlusion. *(This is the technical novelty.)*
2. **Criticality** — we don't just draw roads; we compute *which roads matter most* — the bridges and high-betweenness links whose failure would fragment the city. *(This is the "so what.")*

The punchline demo: *"Here's the city's road graph. These 10 red segments are the most critical. Now watch — with a naive model, occlusion makes us miss 3 of them; our occlusion-robust model still finds all 10."* That single before/after is the win.

---

## 2. End-to-end pipeline

```
RGB tiles + road masks
        │
        ▼
[Synthetic occlusion augmentation]   ← innovation: clouds, shadows, canopy + consistency loss
        │
        ▼
[D-LinkNet segmentation]  ResNet34 encoder → road probability mask
        │
        ▼
[Vectorize]  threshold → morphology → skeletonize → graph (cresi)
        │
        ▼
[Build weighted graph]  NetworkX; nodes=intersections, edges=segments, weight=travel time
        │
        ▼
[Criticality analysis]  edge betweenness · bridges/articulation pts · sequential edge-removal resilience curve
        │
        ▼
[Interactive map demo]  Streamlit + Folium: roads colored by criticality, occlusion before/after, "what-if" edge removal
```

---

## 3. Repository structure

```
route-resilience/
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml              # dataset paths, tile size, splits
│   ├── model.yaml             # encoder, loss weights, occlusion params
│   └── train.yaml             # lr, epochs, batch size
├── data/
│   ├── raw/                   # downloaded DeepGlobe / SpaceNet
│   ├── processed/             # tiled + normalized chips
│   └── osm/                   # OSM graphs for validation
├── src/
│   ├── data/
│   │   ├── download.py        # fetch datasets
│   │   ├── tiling.py          # cut to 512×512, normalize, split
│   │   ├── dataset.py         # torch Dataset (image, mask)
│   │   └── occlusion.py       # ★ synthetic occlusion generators
│   ├── models/
│   │   ├── dlinknet.py        # or thin wrapper over segmentation_models_pytorch
│   │   ├── losses.py          # BCE + Dice + consistency
│   │   └── factory.py
│   ├── train/
│   │   ├── train.py           # training loop (+ occlusion consistency)
│   │   └── evaluate.py        # IoU, F1, APLS, occlusion-retention
│   ├── graph/
│   │   ├── vectorize.py       # mask → skeleton → graph (cresi-based)
│   │   ├── build_graph.py     # NetworkX graph + edge weights
│   │   └── criticality.py     # ★ betweenness, bridges, removal sim
│   ├── viz/
│   │   ├── map_render.py      # folium / leafmap layers
│   │   └── report.py          # metrics + figures
│   └── utils/  (io.py, geo.py, seed.py)
├── app/
│   └── streamlit_app.py       # the demo UI
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_train_baseline.ipynb
│   └── 03_graph_criticality.ipynb
└── scripts/
    ├── infer_image.py         # one image → roads → graph → criticality
    └── run_pipeline.sh
```

Two files carry the novelty: `src/data/occlusion.py` and `src/graph/criticality.py`. Everything else is plumbing you can fork.

---

## 4. Component design

### 4.1 Data
- **Primary training:** DeepGlobe Road Extraction (RGB + binary road masks, 50 cm/px) — simplest to start.
- **Alt / graph-native:** SpaceNet Roads (ships road centerlines as GeoJSON + travel-time labels — pairs perfectly with cresi).
- **Also useful:** Massachusetts Roads dataset (large, easy).
- **Validation / real-city demo:** pull ground-truth graphs from OpenStreetMap via `osmnx` for any city you choose (great for an Indian-city demo even without labels).
- **Tiling:** 512×512 chips, mean/std normalize, 80/10/10 split by tile (avoid leakage across adjacent tiles).

### 4.2 Segmentation model — D-LinkNet
- Encoder: **ResNet34** (ImageNet-pretrained) → dilated center block → LinkNet decoder → 1-channel sigmoid mask.
- Fastest path: use `segmentation_models_pytorch` (`Linknet`/`Unet` with `encoder_name="resnet34"`) and add a dilated bottleneck, rather than hand-rolling D-LinkNet. Same accuracy ballpark, far less code.
- **Loss:** BCE + Dice (handles thin, sparse roads). Add **focal** if recall on thin roads lags.
- Train at 512², batch 8–16, AdamW, cosine LR, mixed precision. A single mid-range GPU (or Colab/Kaggle T4) is enough.

### 4.3 ★ Occlusion robustness (the differentiator) — `occlusion.py`
Generate realistic occluders on-the-fly during training:
- **Clouds:** Perlin/simplex-noise blobs, semi-transparent white, soft edges.
- **Shadows:** darkened polygons; optionally oriented to a sun angle for building shadows.
- **Tree canopy:** green-textured irregular patches over roads.
- **Clutter:** small bright/dark specks (vehicles, rooftops).

Two ways to use them, pick based on time:
1. **Augmentation only (easy):** apply occlusion to inputs, keep the true mask as target → model learns to "see through" occlusion.
2. **Consistency training (stronger story):** for each tile, forward both the clean and occluded version; add a **consistency loss** (MSE/KL between the two predictions) so the model's output is stable under occlusion. This is the headline robustness claim.

**Evaluation that proves it:** build a held-out **occluded test set**, report (a) F1/IoU retention vs clean, and (b) downstream **graph-connectivity retention** (does the critical-link ranking survive occlusion?). Compare against a vanilla D-LinkNet baseline. That table is your slide.

### 4.4 Vectorization — `vectorize.py`
- Threshold probability mask → morphological close/open to bridge small gaps → `skimage.morphology.skeletonize` → trace skeleton into nodes/edges.
- Reuse **cresi's** skeleton→graph + speed/travel-time inference steps; it also provides **APLS** (Average Path Length Similarity) to score graph quality against ground truth — a credible metric judges respect.
- Snap/simplify with `shapely`; keep edge geometry for the map.

### 4.5 Weighted graph — `build_graph.py`
- `NetworkX` graph: nodes = intersections, edges = segments.
- Edge weight = **travel time** = length / inferred speed (speed from road class or cresi's speed head). Fall back to length if speed unavailable.
- Keep lat/lon on nodes so the graph renders on a real map.

### 4.6 ★ Criticality analysis — `criticality.py`
- **Edge betweenness centrality** — fraction of shortest paths using each edge → traffic-bearing importance.
- **Bridges & articulation points** — `nx.bridges`, `nx.articulation_points` → true single-points-of-failure (removal disconnects the network).
- **Sequential edge-removal / percolation** — remove edges in betweenness order; track **global efficiency** `E(G) = 1/(N(N-1)) Σ 1/d_ij` and **giant-component size** → a resilience curve.
- **Per-edge criticality score** = increase in total travel time (or drop in efficiency) when that edge is removed; optionally weight by origin-destination demand if you find/синтезize an OD matrix.
- **Robustness tie-in:** compute the criticality ranking from clean vs occluded extractions and show the robust model preserves the correct top-K.

### 4.7 Demo — `streamlit_app.py`
- Upload/select a tile or city → run inference → render roads on a Folium/leafmap map colored by criticality (green→red).
- Toggle: **clean vs occluded** input (shows missed roads on the naive path).
- Slider/click: **remove an edge** → live recompute of efficiency drop and re-route — the interactive "what-if" that lands with judges.

---

## 5. Tech stack

| Layer | Choice | Why |
|---|---|---|
| DL framework | PyTorch | Standard; matches all fork repos |
| Segmentation | `segmentation_models_pytorch` (LinkNet/ResNet34) | D-LinkNet-equivalent in a few lines, pretrained encoders |
| Augmentation | `albumentations` + custom `occlusion.py` | Fast, composable; custom occluders are the novelty |
| Geometry/raster | `rasterio`, `opencv`, `scikit-image`, `shapely` | Tiling, skeletonize, vectorize |
| Graph | `NetworkX` (swap to `igraph` only if too slow) | Betweenness, bridges, efficiency out of the box |
| Real-city graphs | `osmnx` | Pull OSM ground-truth/validation graphs |
| Maps/demo | `streamlit` + `folium`/`leafmap` | Fastest route to an interactive, judge-friendly demo |
| Metrics | IoU/F1 (seg), APLS (graph), efficiency-retention (robustness) | Credible, paper-backed |

---

## 6. Repos to fork (and exactly what to take)

- `avanetten/cresi` — take the **skeletonize → graph** and **speed/travel-time + APLS** stages; feed its graph into your `criticality.py`.
- `zlckanata/DeepGlobe-Road-Extraction-Challenge` — reference **D-LinkNet** implementation if you don't use `segmentation_models_pytorch`.
- `satellite-image-deep-learning/techniques` — meta-index for alternative extractors / datasets if you need a plan B.

---

## 7. Sprint plan (≈4 people, mapped to your strengths)

| Day | Milestone | Owner (strength) |
|---|---|---|
| 0–1 | Repo scaffold, env, download DeepGlobe; **baseline D-LinkNet inference with pretrained weights** (end-to-end on one tile) | All / CV |
| 1–3 | Fine-tune segmentation; build `vectorize.py` (cresi) → NetworkX graph; first betweenness map | CV + Graph |
| 3–5 | `occlusion.py` + consistency training; occluded eval set; **robustness comparison table** | CV/ML + Data/GIS |
| 3–5 | `criticality.py` complete (bridges + removal resilience curve); OSM validation on a real city | Graph + GIS |
| 5–7 | Streamlit demo (upload → roads → criticality → edge-removal what-if); polish, metrics, slides | Full-stack/UI |

**Parallel tracks:** (A) segmentation/robustness — CV+ML, (B) graph/criticality — algorithms, (C) data/GIS — remote sensing, (D) demo/UI — full-stack. Clean four-way split, no one blocked.

**De-risking:** Day-1 goal is a working end-to-end path on a *pretrained* model before any training — guarantees a demo exists even if training underperforms. Occlusion-consistency is the one "research" risk; the augmentation-only fallback still gives a robustness story.

---

## 8. What's still open (your side)
- Which **city/region** to feature in the live demo (drives the OSM pull + a compelling local narrative — an Indian metro is a strong choice).
- Whether you have/ want an **OD demand matrix** (optional; elevates criticality from topological to demand-weighted).
- Imagery source/resolution if you go beyond DeepGlobe/SpaceNet (affects the dataset loader only).

Drop the data in `data/raw/` when you have it — the loader is designed to be source-agnostic.
