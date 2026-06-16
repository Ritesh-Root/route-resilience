# Route Resilience — Slide Outline

**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**

> Presenter notes are in blockquotes. Numbers cited below mirror the live demo backend
> (`GET /api/metrics`, `GET /api/resilience-curve`, `POST /api/simulate`) so the deck and the
> running system stay in sync. City = **Bengaluru**.

---

## Slide 1 — Title

- Project: **Route Resilience**
- Subtitle: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility
- Team / Hackathon / Date
- One-line hook: *"From cloudy satellite tiles to a map of the roads a city cannot afford to lose."*

> Set the frame: this is two hard problems solved together — seeing roads through occlusion,
> and reasoning about which roads matter.

---

## Slide 2 — Problem: The Dual Challenge

- **Challenge A — Perception under occlusion.** Satellite/aerial imagery over Indian cities is
  routinely degraded by clouds, haze, tree canopy, building shadows, and flyovers. Naive road
  extractors *drop links* exactly where the image is occluded, producing a **fragmented graph**.
- **Challenge B — Reasoning about criticality.** Even a perfect map doesn't tell a planner
  *which* roads are structurally critical. A single bridge or arterial can hold an entire zone
  together; losing it disconnects neighborhoods.
- **The trap:** occlusion-induced gaps masquerade as real missing roads, so downstream
  resilience analysis is computed on a *broken* network and silently gives wrong answers.

> Emphasize the coupling: a perception error (Challenge A) corrupts the graph analysis
> (Challenge B). The contribution is solving them as one pipeline.

---

## Slide 3 — Method Overview (Pipeline)

```
Satellite tile ─► [1] Occlusion-Robust Extraction ─► raw road mask
                                                         │
                          [2] Graph Healing  ◄───────────┘  (vectorize + repair gaps)
                                                         │
                          [3] Criticality Analysis ◄──────┘  (betweenness, articulation,
                                                              criticality 0..1, resilience index)
```

1. **Occlusion-robust extraction** — segmentation model trained with occlusion augmentation so
   it *inpaints* road continuity under clouds/shadows instead of breaking it.
2. **Graph healing** — vectorize mask to a graph; reconnect short collinear gaps; recover
   bridges/articulation links the baseline lost.
3. **Graph-theoretic criticality** — per-edge `criticality (0..1)`, node `betweenness`,
   `isArticulation` flags, and a single `resilienceIndex (0..100)`.

> Map each stage to a backend endpoint: extraction/healing → `/api/network` & `/api/infer`;
> criticality → `/api/gatekeepers` & `/api/metrics`; what-if → `/api/simulate`.

---

## Slide 4 — Method Detail: Occlusion-Robust Extraction

- Backbone: encoder–decoder segmentation (U-Net / DeepLab-style head).
- **Key idea:** synthetic occlusion augmentation (random cloud/shadow/canopy masks) at train
  time forces the network to predict road continuity from context, not just visible pixels.
- Loss combines pixel segmentation with a **connectivity-aware term** (penalizes broken
  topology), so the optimizer is rewarded for keeping links intact through occlusion.
- Output: `baseline` model (no occlusion training) vs `robust` model (with it) — the demo
  switch (`model=baseline|robust`) shows the difference directly.

> Honesty note for judges: training is out of scope on the hackathon hardware (4GB GPU);
> the code/scaffolding and pinned deps are provided, results shown are the served demo graph.

---

## Slide 5 — Method Detail: Graph Healing & Criticality

- **Vectorize** mask → nodes/edges; each edge carries `lengthM`, `travelTimeSec`, `roadClass`,
  `isBridge`.
- **Heal** occlusion gaps: bridge short collinear breaks and restore articulation links so the
  analyzed graph reflects real connectivity, not image artifacts.
- **Criticality metrics:**
  - Edge `criticality (0..1)` — contribution to network-wide reachability.
  - Node `betweenness` + `isArticulation` — "gatekeeper" nodes whose removal splits the city.
  - `resilienceIndex (0..100)` — single headline number for the whole network.

> This is where Challenge A meets Challenge B: healing must happen *before* criticality, or the
> gatekeepers you report are occlusion ghosts.

---

## Slide 6 — Robustness Table (Baseline vs Robust)

Metrics from `GET /api/metrics?city=Bengaluru&input=...&model=...`.

| Scenario                    | IoU  | Dice | Occlusion-Recall | Connectivity Ratio | APLS | Resilience Index |
|-----------------------------|------|------|------------------|--------------------|------|------------------|
| Clean · Baseline            | high | high | n/a              | 1.00               | high | **100**          |
| **Occluded · Baseline**     | ↓    | ↓    | **low**          | **< 1.00**         | ↓    | **~79** (fragmented) |
| Occluded · Robust           | high | high | **high**         | ~1.00              | high | **100** (recovered) |
| Clean · Robust              | high | high | n/a              | 1.00               | high | **100**          |

- Headline: under occlusion, **baseline collapses** (Occlusion-Recall drops, network fragments,
  resilience ~79); the **robust model recovers** the critical links → resilience back to 100.

> Pull the exact IoU/Dice/APLS numbers live from `/api/metrics` the morning of the demo so the
> table matches the screen. The story to land: *Occlusion-Recall* is the metric that explains
> *why* the graph fragments.

---

## Slide 7 — Criticality Map (Visual)

- Map of Bengaluru with edges colored by `criticality (0..1)` (cool → hot).
- **Gatekeeper nodes** (`/api/gatekeepers`, `top_k`) highlighted; articulation points starred.
- Side-by-side: **Occluded·Baseline** (visible fragmentation, disconnected zones) vs
  **Occluded·Robust** (intact, critical bridges restored).
- Callout: the few hot edges / articulation nodes are where mitigation budget should go.

> Live-demo tip: toggle `input` and `model` to animate the fragment-then-heal moment on the map.

---

## Slide 8 — What-If Demo (Interactive)

- Driven by `POST /api/simulate` with `disabledEdgeIds[]` / `disabledNodeIds[]`.
- Flow on stage:
  1. Start Occluded·Baseline → show ~79 resilience + disconnected zones.
  2. Disable a top gatekeeper edge → `resilienceIndexAfter` drops further,
     `avgTravelTimeIncreasePct` spikes, `newlyDisconnectedZones` appear,
     `brokenRoutesSampled` / `sampledRoutes` list concrete severed trips.
  3. Switch to Robust → links restored, resilience returns to 100.
- Takeaway: planners can *stress-test* the city and see which single failures are catastrophic.

> This is the emotional peak — let the audience pick the edge to disable.

---

## Slide 9 — ISRO / NNRMS Alignment

- **NNRMS (National Natural Resources Management System)** & ISRO Earth-observation priorities:
  turning satellite imagery into actionable infrastructure intelligence.
- Fits **disaster management & urban resilience** themes: occlusion robustness directly serves
  monsoon/cloud-heavy Indian conditions where clean imagery is rare.
- Complements Bhuvan / national geospatial stack: outputs are standard **GeoJSON** (lng,lat),
  ready to ingest into existing GIS pipelines.
- Supports planning for **critical infrastructure protection** and rapid post-event reconnect
  assessment using degraded, real-world imagery.

> Tie explicitly to "operational under occlusion" — that is the differentiator for Indian
> EO conditions and the NNRMS mandate.

---

## Slide 10 — Roadmap

- **Now (hackathon):** end-to-end pipeline + interactive resilience/what-if demo on Bengaluru;
  runnable scaffolding with pinned deps (training-ready, not yet trained on this hardware).
- **Next:** train robust extractor on DeepGlobe/SpaceNet with occlusion augmentation on adequate
  GPU; validate Occlusion-Recall & APLS gains quantitatively.
- **Then:** multi-city scale-out; ingest Bhuvan/Sentinel tiles; temporal change detection.
- **Later:** real-time disaster mode (fresh imagery → updated resilience index); planner-facing
  mitigation recommendations (which links to harden first).

> Close on impact: *the same pipeline that heals a cloudy map tells a city which roads to
> protect first.*

---

## Appendix — Demo Run Notes

- Backend: FastAPI at `http://localhost:8000`, all routes under `/api`.
- Canonical demo states:
  - **Fragmented:** `input=occluded&model=baseline` → resilienceIndex ~79, several disconnected zones.
  - **Intact:** every other (input, model) combination → resilienceIndex 100.
- Sanity check before presenting:
  - `GET /api/health` → `{status, service, baseEfficiency}`
  - `GET /api/cities` → `["Bengaluru"]`
  - `GET /api/network?city=Bengaluru&input=occluded&model=baseline` → fragmented FeatureCollection.
- TODO (not run on this 4GB / Python 3.14 machine): dataset download, `torch` install, and model
  training are intentionally deferred — see project `requirements*.txt` and backend README.
