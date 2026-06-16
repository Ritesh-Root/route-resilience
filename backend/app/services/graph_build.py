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


def geojson_to_graph(fc: dict) -> nx.Graph:
    """Rebuild a road graph from a GeoJSON FeatureCollection (reverse of
    ``graph_to_geojson``).

    Used to load a *real* extracted/OSM network from disk and run the exact same
    criticality pipeline on it. Each LineString feature becomes one edge; its two
    endpoints become nodes keyed by ``source``/``target`` when present, else by
    rounded coordinates. Edge attributes mirror what the criticality engine needs
    (``id``, ``length_m``, ``travel_time``, ``road_class``).
    """
    G = nx.Graph()
    for i, feat in enumerate(fc.get("features", [])):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") != "LineString" or len(coords) < 2:
            continue
        (lng_u, lat_u), (lng_v, lat_v) = coords[0], coords[-1]
        p = feat.get("properties") or {}
        u = p.get("source") or f"n_{round(lat_u, 6)}_{round(lng_u, 6)}"
        v = p.get("target") or f"n_{round(lat_v, 6)}_{round(lng_v, 6)}"
        if u not in G:
            G.add_node(u, lat=float(lat_u), lng=float(lng_u))
        if v not in G:
            G.add_node(v, lat=float(lat_v), lng=float(lng_v))
        if u == v or G.has_edge(u, v):
            continue
        G.add_edge(
            u, v,
            id=p.get("id", f"e_{i}"),
            length_m=float(p.get("lengthM", 0.0) or 0.0),
            travel_time=float(p.get("travelTimeSec", 0.0) or 0.0),
            road_class=p.get("roadClass", "local"),
        )
    return G


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
