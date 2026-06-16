"""Synthetic road network for the demo (Bengaluru-shaped grid + arterials + spurs).

This is a stand-in for the real extracted graph so the API + front-end work
end-to-end before the model exists. Swap `build_real_network()` in once the
segmentation -> vectorization pipeline produces a graph; everything downstream
(criticality, gatekeepers, simulation) is identical.

Four "views" are exposed via (input, model):
    clean    + robust   -> full network (ideal)
    clean    + baseline -> full network
    occluded + robust   -> full network (model "saw through" the occlusion)
    occluded + baseline -> FRAGMENTED network (spectral blindness: broken roads)
"""
from __future__ import annotations

import random

import networkx as nx

from . import criticality as crit
from .geo import haversine_m

# Bengaluru bounding box (approx central area)
BBOX = {"lat_min": 12.90, "lat_max": 13.04, "lng_min": 77.52, "lng_max": 77.66}
ROWS, COLS = 9, 9
SPEED_KMH = {"arterial": 50.0, "local": 30.0}
CITIES = ["Bengaluru"]


def _nid(r: int, c: int) -> str:
    return f"n{r}_{c}"


def _add_edge(G: nx.Graph, u: str, v: str, road_class: str) -> None:
    lu, lv = G.nodes[u], G.nodes[v]
    length = haversine_m(lu["lat"], lu["lng"], lv["lat"], lv["lng"])
    speed_ms = SPEED_KMH[road_class] * 1000 / 3600
    G.add_edge(
        u, v,
        id=f"e_{u}__{v}",
        length_m=round(length, 1),
        road_class=road_class,
        travel_time=round(length / speed_ms, 1),
    )


def build_base_network() -> nx.Graph:
    """Full, intact network: grid + central arterials + peripheral spur 'suburbs'."""
    G = nx.Graph()
    lat_step = (BBOX["lat_max"] - BBOX["lat_min"]) / (ROWS - 1)
    lng_step = (BBOX["lng_max"] - BBOX["lng_min"]) / (COLS - 1)
    for r in range(ROWS):
        for c in range(COLS):
            G.add_node(_nid(r, c),
                       lat=round(BBOX["lat_min"] + r * lat_step, 6),
                       lng=round(BBOX["lng_min"] + c * lng_step, 6))

    arterial_r, arterial_c = ROWS // 2, COLS // 2
    for r in range(ROWS):
        for c in range(COLS):
            if c + 1 < COLS:
                _add_edge(G, _nid(r, c), _nid(r, c + 1),
                          "arterial" if r == arterial_r else "local")
            if r + 1 < ROWS:
                _add_edge(G, _nid(r, c), _nid(r + 1, c),
                          "arterial" if c == arterial_c else "local")

    # Peripheral "suburb" spurs — single-edge attachments => genuine bridges /
    # articulation points. Disabling one disconnects a zone (great what-if demo).
    spurs = {"s_north": _nid(0, 2), "s_east": _nid(4, COLS - 1),
             "s_south": _nid(ROWS - 1, 6), "s_west": _nid(6, 0)}
    off = 0.012
    for sid, anchor in spurs.items():
        an = G.nodes[anchor]
        dlat = off if "north" in sid else (-off if "south" in sid else 0.0)
        dlng = off if "east" in sid else (-off if "west" in sid else 0.0)
        G.add_node(sid, lat=round(an["lat"] + dlat, 6), lng=round(an["lng"] + dlng, 6))
        _add_edge(G, anchor, sid, "local")

    _annotate(G)
    return G


def _annotate(G: nx.Graph) -> None:
    """Attach criticality (edges) + betweenness (nodes) + isBridge flags."""
    eb = crit.edge_betweenness(G)
    nb = crit.node_betweenness(G)
    bridge_set = crit.bridges(G)
    max_eb = max(eb.values()) if eb else 1.0
    for u, v, d in G.edges(data=True):
        d["criticality"] = round((eb.get((u, v), eb.get((v, u), 0.0)) / max_eb)
                                 if max_eb > 0 else 0.0, 4)
        d["is_bridge"] = frozenset((u, v)) in bridge_set
    max_nb = max(nb.values()) if nb else 1.0
    for n, val in nb.items():
        G.nodes[n]["betweenness"] = round((val / max_nb) if max_nb > 0 else 0.0, 4)


def _fragment(G: nx.Graph, drop_fraction: float = 0.30, seed: int = 13) -> nx.Graph:
    """Simulate occluded + baseline extraction: broken road masks.

    Drops a large share of local edges (incl. some spur bridges, isolating
    suburbs) plus a couple of arterial segments, so the network genuinely
    fragments — sharply lower connectivity + resilience than the robust view.
    """
    H = G.copy()
    rng = random.Random(seed)
    local_edges = [(u, v) for u, v, d in H.edges(data=True) if d["road_class"] == "local"]
    arterial_edges = [(u, v) for u, v, d in H.edges(data=True) if d["road_class"] == "arterial"]
    rng.shuffle(local_edges)
    rng.shuffle(arterial_edges)
    for u, v in local_edges[: int(len(local_edges) * drop_fraction)]:
        H.remove_edge(u, v)
    for u, v in arterial_edges[:2]:  # occlusion also breaks a couple of arterials
        H.remove_edge(u, v)
    _annotate(H)
    return H


# --------------------------------------------------------------------------- #
# Real-network override: if a GeoJSON file exists (produced by the OSM fetcher
# `scripts/fetch_bengaluru_osm.py` or by trained-model inference via
# `ml/integration/segmenter_hook.py`), serve THAT as the base network instead of
# the synthetic grid. Everything downstream (criticality, gatekeepers, simulate)
# is identical. Set REAL_NETWORK_PATH to override the default location.
# --------------------------------------------------------------------------- #
import json
import os

from . import graph_build

_DEFAULT_REAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_network.geojson")


def _load_base() -> tuple[nx.Graph, str]:
    """Return ``(base_graph, source)`` where source is 'real' or 'synthetic'."""
    path = os.environ.get("REAL_NETWORK_PATH", _DEFAULT_REAL_PATH)
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                fc = json.load(fh)
            G = graph_build.geojson_to_graph(fc)
            if G.number_of_edges() > 0:
                _annotate(G)
                return G, "real"
        except Exception as exc:  # never let a bad file break the demo
            print(f"[network_factory] failed to load real network {path!r}: {exc}; using synthetic")
    return build_base_network(), "synthetic"


# --------------------------------------------------------------------------- #
# Cache + public accessors
# --------------------------------------------------------------------------- #
_BASE, NETWORK_SOURCE = _load_base()
BASE_EFF = crit.global_efficiency_weighted(_BASE)
_CACHE: dict[tuple[str, str], nx.Graph] = {}


def get_network(input_mode: str = "clean", model: str = "robust") -> nx.Graph:
    key = (input_mode, model)
    if key not in _CACHE:
        if input_mode == "occluded" and model == "baseline":
            _CACHE[key] = _fragment(_BASE)
        else:
            _CACHE[key] = _BASE.copy()
    return _CACHE[key]
