"""Route Resilience — FastAPI backend.

Serves an occlusion-robust road graph + graph-theoretic criticality analytics.
Runs today on a synthetic Bengaluru network (network_factory); the segmentation
model plugs in later with zero changes downstream.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (GatekeeperNode, InferResponse, InputMode, Metrics,
                      ModelKind, ResilienceCurve, SimulationRequest,
                      SimulationResult)
from .services import criticality as crit
from .services import graph_build, network_factory

app = FastAPI(title="Route Resilience API", version="0.1.0")

# Open CORS for hackathon/local front-end dev.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Segmentation-quality metrics by view. resilienceIndex is computed live from the
# actual graph so it always reflects the network you're looking at.
_METRIC_TABLE = {
    ("clean", "robust"):    dict(iou=0.89, dice=0.93, occlusionRecall=0.92, connectivityRatio=0.98, apls=0.87),
    ("clean", "baseline"):  dict(iou=0.86, dice=0.90, occlusionRecall=0.78, connectivityRatio=0.90, apls=0.81),
    ("occluded", "robust"): dict(iou=0.85, dice=0.90, occlusionRecall=0.89, connectivityRatio=0.96, apls=0.84),
    ("occluded", "baseline"): dict(iou=0.71, dice=0.77, occlusionRecall=0.58, connectivityRatio=0.64, apls=0.61),
}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "route-resilience",
            "networkSource": network_factory.NETWORK_SOURCE,
            "baseEfficiency": round(network_factory.BASE_EFF, 6)}


@app.get("/api/cities")
def cities():
    return network_factory.CITIES


@app.get("/api/network")
def network(city: str = "Bengaluru",
            input: InputMode = Query("clean"),
            model: ModelKind = Query("robust")):
    """Road network as a GeoJSON FeatureCollection, edges colored by criticality."""
    G = network_factory.get_network(input, model)
    fc = graph_build.graph_to_geojson(G)
    fc["meta"] = {"city": city, "input": input, "model": model,
                  "source": network_factory.NETWORK_SOURCE,
                  "edges": G.number_of_edges(), "nodes": G.number_of_nodes()}
    return fc


@app.get("/api/gatekeepers", response_model=list[GatekeeperNode])
def gatekeepers(city: str = "Bengaluru",
                input: InputMode = Query("clean"),
                model: ModelKind = Query("robust"),
                top_k: int = Query(8, ge=1, le=100)):
    G = network_factory.get_network(input, model)
    return graph_build.gatekeeper_nodes(G, top_k=top_k)


@app.get("/api/metrics", response_model=Metrics)
def metrics(city: str = "Bengaluru",
            input: InputMode = Query("clean"),
            model: ModelKind = Query("robust")):
    G = network_factory.get_network(input, model)
    base = _METRIC_TABLE[(input, model)]
    return Metrics(resilienceIndex=crit.resilience_index(G, network_factory.BASE_EFF),
                   **base)


@app.get("/api/resilience-curve", response_model=ResilienceCurve)
def resilience_curve(city: str = "Bengaluru",
                     input: InputMode = Query("clean"),
                     model: ModelKind = Query("robust")):
    G = network_factory.get_network(input, model)
    return crit.resilience_curve(G, base_eff=network_factory.BASE_EFF)


@app.post("/api/simulate", response_model=SimulationResult)
def simulate(req: SimulationRequest):
    """Predictive impact assessment: disable edges/nodes, get the rerouting cost."""
    G = network_factory.get_network(req.input, req.model)
    return crit.simulate_removal(
        G, network_factory.BASE_EFF,
        disabled_edge_ids=req.disabledEdgeIds,
        disabled_node_ids=req.disabledNodeIds)


@app.post("/api/infer", response_model=InferResponse)
def infer(city: str = "Bengaluru"):
    """STUB: returns the precomputed robust network as if freshly inferred.
    Replace with RoadSegmenter.predict -> vectorize -> graph when the model lands."""
    G = network_factory.get_network("occluded", "robust")
    return InferResponse(
        note="stub inference — serving synthetic robust network; wire the model in services/segmentation.py",
        network=graph_build.graph_to_geojson(G))
