"""Convert a NetworkX road graph <-> GeoJSON (the format the front-end consumes)."""
from __future__ import annotations

import networkx as nx


def graph_to_geojson(G: nx.Graph) -> dict:
    """Edges -> GeoJSON LineString FeatureCollection, colored by criticality."""
    features = []
    for u, v, d in G.edges(data=True):
        nu, nv = G.nodes[u], G.nodes[v]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[nu["lng"], nu["lat"]], [nv["lng"], nv["lat"]]],
            },
            "properties": {
                "id": d["id"],
                "source": u,
                "target": v,
                "criticality": d.get("criticality", 0.0),
                "travelTimeSec": d.get("travel_time", 0.0),
                "lengthM": d.get("length_m", 0.0),
                "roadClass": d.get("road_class", "local"),
                "isBridge": d.get("is_bridge", False),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def gatekeeper_nodes(G: nx.Graph, top_k: int = 8) -> list[dict]:
    """Top-K intersections by betweenness — the city's 'Gatekeeper Nodes'."""
    ranked = sorted(G.nodes(data=True),
                    key=lambda nv: nv[1].get("betweenness", 0.0), reverse=True)
    out = []
    for rank, (n, d) in enumerate(ranked[:top_k], start=1):
        out.append({
            "id": n,
            "lat": d["lat"],
            "lng": d["lng"],
            "betweenness": d.get("betweenness", 0.0),
            "isArticulation": False,
            "label": f"GK-{rank}",
        })
    arts = set(nx.articulation_points(G)) if G.number_of_nodes() else set()
    for gk in out:
        gk["isArticulation"] = gk["id"] in arts
    return out
