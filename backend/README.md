# Route Resilience — Backend (FastAPI)

Occlusion-robust road graph + graph-theoretic criticality API. Runs today on a
synthetic Bengaluru network so the front-end integrates immediately; the trained
segmentation model plugs into `services/segmentation.py` later with no downstream
changes.

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# docs: http://localhost:8000/docs
```

## Endpoints
| Method · Path | Purpose |
|---|---|
| GET `/api/health` | liveness + base efficiency |
| GET `/api/cities` | available cities |
| GET `/api/network?city=&input=&model=` | road network GeoJSON, edges colored by criticality |
| GET `/api/gatekeepers?city=&input=&model=&top_k=` | top intersections by betweenness |
| GET `/api/metrics?city=&input=&model=` | IoU, Dice, Occlusion-Recall, Connectivity, APLS, Resilience Index |
| GET `/api/resilience-curve?city=&input=&model=` | efficiency/giant-component vs edges removed |
| POST `/api/simulate` | disable edges/nodes → rerouting cost + Resilience Index after |
| POST `/api/infer` | stub: returns network as if freshly inferred |

`input ∈ {clean, occluded}`, `model ∈ {baseline, robust}`.
`occluded + baseline` = fragmented network (the "before"); `robust` = intact ("after").

## Demo check
```bash
curl "http://localhost:8000/api/metrics?input=occluded&model=baseline"   # resilienceIndex ~79
curl "http://localhost:8000/api/metrics?input=occluded&model=robust"     # resilienceIndex 100
```

## Plugging in the real model
1. Implement `RoadSegmenter.predict` in `services/segmentation.py` (D-LinkNet/CoANet).
2. Add a `mask → skeleton → graph` step (cresi/FilFinder) → NetworkX graph with
   `travel_time` + `id` edge attributes.
3. Return that graph from `network_factory.get_network`. Criticality, gatekeepers,
   simulation all work unchanged.
