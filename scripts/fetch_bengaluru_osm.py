#!/usr/bin/env python3
"""Fetch a REAL central-Bengaluru drive network from OpenStreetMap (Overpass API)
and write it as a GeoJSON the backend serves in place of the synthetic grid.

No GDAL / osmnx / geopandas — pure stdlib + the backend's own graph utilities.
The Overpass response (ways + nodes) is collapsed into an *intersection graph*
(edges run between junctions, intermediate geometry vertices are dropped), which
keeps the graph small enough that betweenness / resilience stay fast while still
looking like a real road map.

Run (from repo root):
    python scripts/fetch_bengaluru_osm.py
Then restart the backend — it auto-detects backend/app/data/real_network.geojson.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

# Make the backend package importable so we reuse its haversine + criticality.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.services import criticality as crit            # noqa: E402
from app.services import graph_build, network_factory    # noqa: E402
from app.services.geo import haversine_m                 # noqa: E402
import networkx as nx                                    # noqa: E402

# Tight central-Bengaluru box (~2-3 km) — keeps the intersection graph compact.
BBOX = (12.965, 77.585, 12.985, 77.610)  # (south, west, north, east)

# Only the arterial skeleton — gives a clean, real-looking map without thousands
# of residential stubs (which would also blow up betweenness runtime).
HIGHWAYS = "motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link"

SPEED_KMH = {"arterial": 50.0, "local": 30.0}
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _road_class(highway: str) -> str:
    return "arterial" if highway.split("_")[0] in ("motorway", "trunk", "primary") else "local"


def fetch_overpass() -> dict:
    s, w, n, e = BBOX
    query = (
        f"[out:json][timeout:60];"
        f'way["highway"~"^({HIGHWAYS})$"]({s},{w},{n},{e});'
        f"(._;>;);out body;"
    )
    last_err = None
    for url in ENDPOINTS:
        try:
            print(f"[osm] querying {url} ...")
            req = urllib.request.Request(
                url, data=("data=" + urllib.parse.quote(query)).encode(),
                headers={"User-Agent": "route-resilience/1.0 (hackathon)"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # try the next mirror
            print(f"[osm]   {url} failed: {exc}")
            last_err = exc
    raise RuntimeError(f"all Overpass endpoints failed: {last_err}")


def build_graph(osm: dict) -> nx.Graph:
    """Collapse OSM ways into an intersection graph in the backend's schema."""
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    ways = [el for el in osm["elements"] if el["type"] == "way" and len(el.get("nodes", [])) >= 2]

    # A node is a junction if it's shared by >1 way or is a way endpoint.
    usage: dict[int, int] = {}
    for way in ways:
        for nid in way["nodes"]:
            usage[nid] = usage.get(nid, 0) + 1
    junction = set()
    for way in ways:
        seq = way["nodes"]
        junction.add(seq[0]); junction.add(seq[-1])
        for nid in seq:
            if usage.get(nid, 0) > 1:
                junction.add(nid)

    G = nx.Graph()
    eid = 0
    for way in ways:
        rc = _road_class(way.get("tags", {}).get("highway", ""))
        seq = way["nodes"]
        seg_start = seq[0]
        seg_len = 0.0
        prev = seq[0]
        for nid in seq[1:]:
            if nid not in nodes or prev not in nodes:
                prev = nid
                continue
            la1, lo1 = nodes[prev]; la2, lo2 = nodes[nid]
            seg_len += haversine_m(la1, lo1, la2, lo2)
            prev = nid
            if nid in junction:                      # close a segment at the next junction
                if seg_start != nid and seg_len > 0 and seg_start in nodes and nid in nodes:
                    u, v = f"n{seg_start}", f"n{nid}"
                    for end, oid in ((u, seg_start), (v, nid)):
                        if end not in G:
                            la, lo = nodes[oid]
                            G.add_node(end, lat=round(la, 6), lng=round(lo, 6))
                    if not G.has_edge(u, v):
                        speed = SPEED_KMH[rc] * 1000 / 3600
                        G.add_edge(u, v, id=f"e_osm_{eid}", length_m=round(seg_len, 1),
                                   road_class=rc, travel_time=round(seg_len / speed, 1))
                        eid += 1
                seg_start = nid
                seg_len = 0.0
    return G


def main() -> int:
    osm = fetch_overpass()
    G = build_graph(osm)
    if G.number_of_edges() == 0:
        print("[osm] no edges built — aborting (keeping synthetic network).")
        return 1
    # Keep the largest connected component so resilience metrics are meaningful.
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
    network_factory._annotate(G)
    fc = graph_build.graph_to_geojson(G)
    fc["meta"] = {"city": "Bengaluru", "source": "osm", "bbox": BBOX,
                  "edges": G.number_of_edges(), "nodes": G.number_of_nodes()}

    out_dir = os.path.join(ROOT, "backend", "app", "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "real_network.geojson")
    with open(out, "w") as fh:
        json.dump(fc, fh)
    eff = crit.global_efficiency_weighted(G)
    print(f"[osm] wrote {out}")
    print(f"[osm] nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"efficiency={eff:.6g} — restart the backend to serve it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
