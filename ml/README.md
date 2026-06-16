# `ml/` — Occlusion-Robust Road Extraction & Graph Healing Pipeline

This directory holds the **machine-learning pipeline** behind Route Resilience: it
turns satellite imagery into a *routable, occlusion-robust road graph* and the
metrics that prove the robust model recovers links the baseline loses under
occlusion. The FastAPI backend (`backend/app`) consumes the artifacts produced
here; nothing in this directory is required at API request time (inference
artifacts are pre-baked — see [Backend hook](#7-backend-hook)).

---

## ⚠️ Hard environment constraints (read first)

This project targets **one laptop**: a 4 GB RTX 3050 (mobile) + **Python 3.14**.
Every design choice below is shaped by that:

| Constraint | Consequence for this pipeline |
| --- | --- |
| **4 GB VRAM** | Train on **512×512 chips** with **batch size 1–2**, mixed precision (`torch.cuda.amp.autocast`), gradient accumulation for an effective larger batch, and a lightweight backbone (D-LinkNet34, **not** D-LinkNet101). Free the cache between stages. CoANet is provided as an *optional* heavier variant — expect OOM on 4 GB and treat it as documentation/aspiration unless you have a bigger GPU. |
| **Python 3.14** | Pin a torch build that publishes 3.14 wheels (see `requirements.txt`). Heavy libs (`torch`, `rasterio`, `gdal`, `scikit-image`) are **guard-imported** inside functions / `try`-`except` so every module stays *import-safe* even when the lib is absent — the file imports, prints a clear "install X" message, and exits cleanly. |
| **No big downloads on this machine** | Do **not** pull DeepGlobe / SpaceNet (multi-GB) here. The pipeline runs against a **tiny committed sample** (a few chips under `ml/data/sample/`) for smoke tests; full-dataset training is a documented TODO to run elsewhere. |

> Rule of thumb: if a step needs more than ~3.5 GB VRAM or a multi-GB download,
> it is gated behind a TODO and a smaller default, never the happy path.

---

## Pipeline overview

```
                    ┌──────────────────────────────────────────────┐
  raw imagery  ───► │ 1. data        load chips + masks (rasterio)  │
  (sample only)     └──────────────────────────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 2. occlusion aug   clouds / shadows / canopy  │  ◄─ reuses
                    │    + clean↔occluded consistency loss          │   backend/app/
                    └─────────────────┬────────────────────────────┘   services/occlusion.py
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 3. segmentation   D-LinkNet34 / (CoANet opt.) │
                    │    -> per-pixel road probability mask         │
                    └─────────────────┬────────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 4. vectorize   skeletonize -> node/edge graph │
                    └─────────────────┬────────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 5. MST healing   reconnect occlusion gaps     │
                    │    via min-spanning-tree over component gaps  │
                    └─────────────────┬────────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 6. metrics   IoU / Dice / occlusionRecall /   │
                    │    connectivityRatio / APLS / resilienceIndex │
                    └─────────────────┬────────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────────────┐
                    │ 7. backend hook  export GeoJSON + metrics     │
                    │    JSON the FastAPI service serves verbatim   │
                    └──────────────────────────────────────────────┘
```

Suggested module layout (create as you implement — only this README exists today):

```
ml/
├── README.md                ← this file (the only file created so far)
├── requirements.txt         ← TODO: pinned deps (torch 3.14 wheel, etc.)
├── data/
│   ├── sample/              ← TODO: a handful of committed 512×512 chips + masks
│   └── loader.py            ← step 1
├── augment/
│   └── occlusion.py         ← step 2 (import/extend backend occlusion generators)
├── models/
│   ├── dlinknet.py          ← step 3 (D-LinkNet34)
│   └── coanet.py            ← step 3 (optional, OOM-risky on 4 GB)
├── train.py                 ← steps 2+3 training loop (amp + grad-accum)
├── vectorize.py             ← step 4 (skeleton -> graph)
├── heal.py                  ← step 5 (MST gap healing)
├── metrics.py               ← step 6 (IoU/Dice/APLS/resilienceIndex)
└── export.py                ← step 7 (GeoJSON + metrics JSON for the API)
```

---

## 1. Data

- **Source (full):** DeepGlobe Road Extraction or SpaceNet Roads. **DO NOT
  download these on the target laptop** (multi-GB). Train on a bigger box, copy
  back only the checkpoint + a few chips.
- **Source (this repo):** `ml/data/sample/` — a few RGB chips + binary road
  masks for smoke-testing the *whole* pipeline end-to-end without a GPU dataset.
- **Format:** 512×512 GeoTIFF (RGB) + single-channel PNG/GeoTIFF mask
  (`1`=road, `0`=background). Geo-referencing (affine transform from rasterio)
  is carried through so step 7 can emit `[lng, lat]` coordinates.
- **Loading:** `rasterio` is **guard-imported** — if absent, the loader raises a
  clear `RuntimeError("rasterio not installed: pip install -r ml/requirements.txt")`
  instead of an `ImportError` at module import time.

## 2. Occlusion augmentation (the robustness innovation)

Reuse the **numpy-only** generators already battle-tested in
`backend/app/services/occlusion.py`:

- `add_clouds` — bright soft blobs (bright occlusion).
- `add_shadows` — dark soft blobs (luminance loss).
- `add_canopy` — green-tinted irregular patches (tree-cover occlusion).
- `random_occlude` — stochastically composes the above.

Training recipe that creates the **demo story** (`occluded + baseline` =
fragmented; everything else intact):

1. For each chip, forward a **clean** copy and an **occluded** copy
   (`random_occlude(img, seed=...)`); the **target mask is always the clean
   ground truth** — the network must "see through" the occlusion.
2. **Baseline model:** trained on clean chips only → fragments under occlusion.
3. **Robust model:** trained with occlusion aug **+ a consistency loss** between
   the clean-prediction and occluded-prediction logits (e.g. MSE/KL), so it
   recovers the links the baseline drops. This is exactly the gap the demo
   visualizes (`resilienceIndex` 79 → 100).

These generators are numpy-only by design (no heavy CV dep) and plug into
`albumentations` as a custom transform.

## 3. Segmentation — D-LinkNet34 (CoANet optional)

- **D-LinkNet34**: ResNet34 encoder + dilated center block + LinkNet decoder.
  Chosen because it fits 4 GB at 512² with batch 1–2 under AMP.
- **CoANet** (Connectivity-Attentive Network): stronger on connectivity but
  heavier — documented as **optional** and OOM-prone on 4 GB. Keep behind a flag.
- `torch` is **guard-imported**; modules import cleanly without it.
- **Memory levers (all on by default for 4 GB):** AMP autocast + `GradScaler`,
  `batch_size=1` with gradient accumulation, `torch.backends.cudnn.benchmark`,
  `optimizer.zero_grad(set_to_none=True)`, `torch.cuda.empty_cache()` between
  epochs, and **no** D-LinkNet101.
- **Output:** per-pixel road probability → threshold (default 0.5) → binary mask.

## 4. Vectorize (skeleton → graph)

- `skimage.morphology.skeletonize` (guard-imported) thins the binary mask to a
  1-px centerline.
- Walk the skeleton: skeleton pixels with ≠2 neighbours become **graph nodes**
  (endpoints / junctions); the runs between them become **edges**.
- Attach per-edge `properties` matching the backend contract:
  `lengthM` (pixels × ground-sample-distance), `travelTimeSec`
  (`lengthM / assumed_speed_by_roadClass`), `roadClass`, `isBridge`.
  `criticality` is filled in step 6.
- Pixel→geo conversion uses the rasterio affine transform from step 1, yielding
  `[lng, lat]` coordinate pairs (GeoJSON order).

## 5. MST healing (reconnect occlusion gaps)

Occlusion leaves a road as **multiple disconnected components**. Healing
reconnects them so the robust model's graph is whole:

1. Find connected components of the vectorized graph (e.g. `networkx`).
2. Build candidate "bridge" edges between **near** endpoints of *different*
   components (gap shorter than a max-jump threshold), weighted by gap distance.
3. Run a **minimum spanning tree** over the component-contraction graph and add
   only the MST bridge edges back — minimal, non-redundant reconnection.
4. Tag healed edges (`isBridge=True`) so the frontend can show recovered links.

This is what turns the robust model's fragmented raw prediction back into an
intact network (`resilienceIndex` → 100), while the baseline (no healing / worse
mask) stays fragmented (`resilienceIndex` ~79, several disconnected zones).

## 6. Metrics

Computed against the clean ground-truth mask/graph, emitted in the exact backend
shape (`GET /api/metrics`):

- **iou**, **dice** — pixel-mask overlap vs. ground truth.
- **occlusionRecall** — recall computed **only over occluded regions** (proves
  "seeing through" occlusion).
- **connectivityRatio** — fraction of GT routes still routable in the prediction.
- **apls** — Average Path Length Similarity (graph-topology metric: compares
  shortest-path lengths between matched node pairs).
- **resilienceIndex** — **int 0..100**; the headline number. 100 = intact,
  ~79 for `occluded+baseline`.

## 7. Backend hook

The pipeline does **not** run inside the API. Instead `export.py` writes static
artifacts the FastAPI service (`backend/app`) loads and serves verbatim:

- A **GeoJSON `FeatureCollection`** per `(city, input, model)` combo, each edge
  `feature.properties = {id, criticality, travelTimeSec, lengthM, roadClass,
  isBridge}`, geometry coords `[lng, lat]`, plus top-level
  `.meta = {city, input, model, edges, nodes}` — exactly what
  `GET /api/network` returns.
- A **metrics JSON** matching `GET /api/metrics`.

`POST /api/infer?city` can optionally invoke `export.py`'s entry function to
regenerate the network for a city; keep heavy imports lazy there too so the API
import path never touches torch.

---

## Exact run order

Run from the **repo root** (paths contain spaces — quote them). Everything below
defaults to the **tiny sample + 4 GB-safe settings**; full training is a TODO.

```bash
# 0. Install pinned deps (Python 3.14). TODO: author ml/requirements.txt.
python -m pip install -r "ml/requirements.txt"

# 1. Smoke-test data loading on the committed sample (no GPU needed)
python -m ml.data.loader --root "ml/data/sample"

# 2-3. Train baseline then robust model on the sample (AMP, batch=1, grad-accum)
#      DO NOT run on the full DeepGlobe/SpaceNet set on this laptop.
python -m ml.train --variant baseline --epochs 1 --batch-size 1 --amp \
    --data "ml/data/sample" --out "ml/checkpoints/baseline.pt"
python -m ml.train --variant robust   --epochs 1 --batch-size 1 --amp \
    --occlusion --consistency-loss \
    --data "ml/data/sample" --out "ml/checkpoints/robust.pt"

# 4. Vectorize predicted masks into a node/edge graph
python -m ml.vectorize --ckpt "ml/checkpoints/robust.pt" \
    --data "ml/data/sample" --out "ml/work/graph.json"

# 5. Heal occlusion gaps with MST bridging
python -m ml.heal --in "ml/work/graph.json" --out "ml/work/graph_healed.json"

# 6. Compute metrics (IoU/Dice/occlusionRecall/connectivity/APLS/resilience)
python -m ml.metrics --pred "ml/work/graph_healed.json" \
    --data "ml/data/sample" --out "ml/work/metrics.json"

# 7. Export GeoJSON + metrics for every (input, model) combo the backend serves
python -m ml.export --city Bengaluru \
    --graph "ml/work/graph_healed.json" --metrics "ml/work/metrics.json" \
    --out "backend/app/data"   # served verbatim by /api/network & /api/metrics
```

### TODOs before this is fully runnable

- [ ] Author `ml/requirements.txt` with **pinned** versions, incl. a `torch`
      build that ships **Python 3.14** wheels, `rasterio`, `scikit-image`,
      `networkx`, `albumentations`, `numpy`. (Do **not** `pip install torch`
      blindly on the target laptop — pick the CUDA build matching the 3050.)
- [ ] Commit a few 512×512 sample chips + masks under `ml/data/sample/`.
- [ ] Implement each `ml/*.py` module (signatures sketched above) with
      **guarded heavy imports** and a runnable `__main__` / argparse CLI.
- [ ] Wire `export.py` output into `backend/app` and `POST /api/infer`.

> Import-safety contract: importing any `ml/*` module **must not** require torch
> /rasterio/skimage to be installed. Guard those imports inside functions or
> `try/except ImportError` with a clear remediation message, mirroring the
> numpy-only style of `backend/app/services/occlusion.py`.
