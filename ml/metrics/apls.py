"""APLS (Average Path Length Similarity) — graph-topology metric.

APLS measures how well a *proposed* road graph reproduces the shortest-path
structure of an OSM *ground-truth* graph. Unlike pixel metrics (IoU/Dice) it is
sensitive to connectivity: a tiny gap that splits a road costs many broken
routes, so APLS is the right complement to the segmentation scores reported by
``GET /api/metrics`` (``apls`` field, 0..1).

Algorithm (after CosmiQ/apls, https://github.com/CosmiQ/apls):

  1. Sample node pairs (s, t) from the ground-truth graph G.
  2. For each pair compute the shortest-path length L_gt(s, t) in G.
  3. Snap s and t onto their nearest nodes in the proposed graph H, compute the
     shortest-path length L_prop(s', t') in H.
  4. Score each pair by  max(0, 1 - |L_gt - L_prop| / L_gt).  If the pair is
     reachable in G but NOT in H (a broken route), the score is 0 — this is the
     penalty that makes APLS connectivity-aware.
  5. APLS = mean of the per-pair scores. The full metric symmetrises by also
     running G<->H swapped and averaging the two means.

Pure ``networkx`` + ``numpy``; no heavy/geospatial deps at import time. OSM
loading via ``osmnx`` is guarded so this module imports even when osmnx (and
its GDAL/rasterio stack) is absent — see :func:`load_osm_graph`.

Edge weight: by default ``length`` (metres). The backend's edge properties
expose ``lengthM`` / ``travelTimeSec``; pass ``weight="travelTimeSec"`` to score
travel-time similarity instead of geometric length.

Run:
    python apls.py --demo          # self-contained synthetic example
    python apls.py --help          # options
TODO (real data): wire ``load_osm_graph(place=...)`` to fetch the Bengaluru
ground truth, and build the proposed graph from the inference network returned
by ``POST /api/infer`` (GeoJSON -> networkx via :func:`graph_from_geojson`).
"""
from __future__ import annotations

import math
from typing import Any, Hashable, Iterable, Sequence

import networkx as nx
import numpy as np

# Type aliases for clarity.
Node = Hashable
Coord = tuple[float, float]  # (lng, lat) to match the GeoJSON contract


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _node_xy(graph: nx.Graph, node: Node) -> Coord:
    """Return a node's (x, y) == (lng, lat).

    Accepts the common attribute spellings: ``x``/``y`` (osmnx),
    ``lng``/``lat`` or ``lon``/``lat`` (the backend GeoJSON contract).
    """
    d = graph.nodes[node]
    if "x" in d and "y" in d:
        return float(d["x"]), float(d["y"])
    if "lng" in d and "lat" in d:
        return float(d["lng"]), float(d["lat"])
    if "lon" in d and "lat" in d:
        return float(d["lon"]), float(d["lat"])
    raise KeyError(
        f"node {node!r} has no usable coordinates "
        "(expected x/y, lng/lat or lon/lat attributes)"
    )


def _haversine_m(a: Coord, b: Coord) -> float:
    """Great-circle distance in metres between two (lng, lat) points."""
    lng1, lat1 = a
    lng2, lat2 = b
    r = 6_371_000.0  # Earth radius, metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _nearest_node(graph: nx.Graph, point: Coord) -> tuple[Node, float]:
    """Nearest node in ``graph`` to ``point`` (haversine). Returns (node, dist_m)."""
    best: Node | None = None
    best_d = math.inf
    for n in graph.nodes:
        d = _haversine_m(point, _node_xy(graph, n))
        if d < best_d:
            best_d, best = d, n
    if best is None:
        raise ValueError("graph has no nodes to snap onto")
    return best, best_d


# --------------------------------------------------------------------------- #
# Core APLS
# --------------------------------------------------------------------------- #
def _pair_score(l_gt: float, l_prop: float | None) -> float:
    """Per-pair similarity in [0, 1].

    ``l_prop is None`` means the pair is unreachable in the proposed graph
    (broken route) -> score 0. Identical lengths -> score 1.
    """
    if l_prop is None:
        return 0.0
    if l_gt <= 0:
        # Degenerate (same snapped node); treat as a perfect match.
        return 1.0
    return max(0.0, 1.0 - abs(l_gt - l_prop) / l_gt)


def _directional_apls(
    g_truth: nx.Graph,
    g_prop: nx.Graph,
    *,
    weight: str,
    max_snap_m: float,
    pairs: Sequence[tuple[Node, Node]],
) -> tuple[float, list[float]]:
    """APLS for ground-truth ``g_truth`` scored against proposal ``g_prop``.

    Snaps each ground-truth endpoint onto the nearest proposal node (rejecting
    snaps farther than ``max_snap_m``, which then count as broken routes).
    Returns (mean_score, per_pair_scores).
    """
    scores: list[float] = []
    # Cache snaps so repeated endpoints are only resolved once.
    snap_cache: dict[Node, Node | None] = {}

    def snap(n: Node) -> Node | None:
        if n not in snap_cache:
            try:
                cand, dist = _nearest_node(g_prop, _node_xy(g_truth, n))
            except (KeyError, ValueError):
                snap_cache[n] = None
            else:
                snap_cache[n] = cand if dist <= max_snap_m else None
        return snap_cache[n]

    for s, t in pairs:
        try:
            l_gt = nx.shortest_path_length(g_truth, s, t, weight=weight)
        except nx.NetworkXNoPath:
            # Unreachable even in ground truth: not a meaningful pair, skip it.
            continue

        s2, t2 = snap(s), snap(t)
        if s2 is None or t2 is None:
            scores.append(0.0)
            continue
        try:
            l_prop: float | None = nx.shortest_path_length(g_prop, s2, t2, weight=weight)
        except nx.NetworkXNoPath:
            l_prop = None
        scores.append(_pair_score(float(l_gt), None if l_prop is None else float(l_prop)))

    mean = float(np.mean(scores)) if scores else 0.0
    return mean, scores


def _sample_pairs(
    graph: nx.Graph,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[Node, Node]]:
    """Sample ``n_pairs`` distinct unordered node pairs from ``graph``."""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return []
    max_pairs = len(nodes) * (len(nodes) - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    seen: set[frozenset[Node]] = set()
    out: list[tuple[Node, Node]] = []
    # Rejection sampling; cheap because n_pairs << max_pairs in practice.
    attempts = 0
    while len(out) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        i, j = rng.integers(0, len(nodes), size=2)
        if i == j:
            continue
        key = frozenset((nodes[i], nodes[j]))
        if key in seen:
            continue
        seen.add(key)
        out.append((nodes[i], nodes[j]))
    return out


def apls(
    g_truth: nx.Graph,
    g_prop: nx.Graph,
    *,
    n_pairs: int = 500,
    weight: str = "length",
    max_snap_m: float = 50.0,
    symmetric: bool = True,
    seed: int | None = 0,
) -> float:
    """Average Path Length Similarity between two road graphs, in [0, 1].

    Args:
        g_truth: ground-truth graph (e.g. OSM). Nodes need coordinates and
            edges a numeric ``weight`` attribute (default ``length``).
        g_prop: proposed/predicted graph, same conventions.
        n_pairs: node pairs sampled from each graph.
        weight: edge attribute used as path cost (``length`` or
            ``travelTimeSec`` to match the backend contract).
        max_snap_m: max distance (metres) an endpoint may be snapped onto the
            other graph before the pair is treated as a broken route.
        symmetric: if True, average the truth->prop and prop->truth directions
            (the standard CosmiQ definition); otherwise truth->prop only.
        seed: RNG seed for reproducible pair sampling.

    Returns:
        Scalar APLS score. 1.0 == identical path structure, 0.0 == no usable
        correspondence.
    """
    rng = np.random.default_rng(seed)
    fwd, _ = _directional_apls(
        g_truth, g_prop, weight=weight, max_snap_m=max_snap_m,
        pairs=_sample_pairs(g_truth, n_pairs, rng),
    )
    if not symmetric:
        return fwd
    rev, _ = _directional_apls(
        g_prop, g_truth, weight=weight, max_snap_m=max_snap_m,
        pairs=_sample_pairs(g_prop, n_pairs, rng),
    )
    # CosmiQ averages the two directional scores.
    return float((fwd + rev) / 2.0)


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #
def graph_from_geojson(fc: dict[str, Any], *, weight: str = "lengthM") -> nx.Graph:
    """Build a networkx graph from the backend's ``/api/network`` GeoJSON.

    Each LineString feature becomes a chain of edges between nodes keyed by
    rounded (lng, lat) so shared endpoints coalesce. Edge ``length`` is summed
    haversine; the requested ``weight`` property (e.g. ``travelTimeSec``,
    ``lengthM``) is copied through when present.
    """
    g = nx.Graph()

    def key(c: Sequence[float]) -> Coord:
        return (round(float(c[0]), 7), round(float(c[1]), 7))

    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        props = feat.get("properties", {}) or {}
        coords = geom.get("coordinates", [])
        prev_k: Coord | None = None
        for c in coords:
            k = key(c)
            if k not in g:
                g.add_node(k, lng=k[0], lat=k[1])
            if prev_k is not None and prev_k != k:
                seg = _haversine_m(prev_k, k)
                attrs: dict[str, Any] = {"length": seg}
                if weight in props:
                    attrs[weight] = props[weight]
                g.add_edge(prev_k, k, **attrs)
            prev_k = k
    return g


def load_osm_graph(
    place: str | None = None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    network_type: str = "drive",
) -> nx.Graph:
    """Fetch an OSM ground-truth graph via osmnx (guarded heavy import).

    Provide either a ``place`` query (e.g. ``"Bengaluru, India"``) or a
    ``bbox`` = (north, south, east, west). Returns an undirected graph whose
    edges carry a ``length`` (metres) attribute, matching :func:`apls`.

    osmnx pulls in GDAL/rasterio, so it is imported lazily here — the rest of
    this module works without it.
    """
    try:
        import osmnx as ox  # noqa: WPS433  (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "osmnx is required for load_osm_graph(); install it with "
            "`pip install osmnx` (brings GDAL). It is NOT needed for the core "
            "apls() computation on already-built graphs."
        ) from exc

    if place is not None:
        mg = ox.graph_from_place(place, network_type=network_type)
    elif bbox is not None:
        north, south, east, west = bbox
        mg = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
    else:
        raise ValueError("load_osm_graph requires either `place` or `bbox`")

    g = nx.Graph(mg)  # collapse MultiDiGraph; keeps node x/y and edge length
    return g


# --------------------------------------------------------------------------- #
# Demo / CLI
# --------------------------------------------------------------------------- #
def _demo_graphs(seed: int = 0) -> tuple[nx.Graph, nx.Graph]:
    """Synthetic ground-truth grid + a degraded proposal (one edge removed).

    Mimics the demo story: an occluded/baseline prediction drops a critical
    link, fragmenting routes and lowering APLS below 1.0.
    """
    rng = np.random.default_rng(seed)
    rows, cols = 5, 5
    g_truth = nx.Graph()
    pos: dict[tuple[int, int], Coord] = {}
    # Lay a grid over a small patch of Bengaluru (~ 77.59E, 12.97N) so the
    # haversine lengths are realistic city-block scales.
    base_lng, base_lat, step = 77.590, 12.970, 0.002
    for r in range(rows):
        for c in range(cols):
            lng = base_lng + c * step
            lat = base_lat + r * step
            pos[(r, c)] = (lng, lat)
            g_truth.add_node((r, c), lng=lng, lat=lat)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr < rows and nc < cols:
                    a, b = (r, c), (nr, nc)
                    g_truth.add_edge(a, b, length=_haversine_m(pos[a], pos[b]))

    # Proposal: copy truth, then sever a central edge to simulate the broken
    # link the baseline model loses under occlusion.
    g_prop = g_truth.copy()
    g_prop.remove_edge((2, 2), (2, 3))
    # Add a little geometric noise to the proposal node coords.
    for n in g_prop.nodes:
        lng, lat = _node_xy(g_prop, n)
        g_prop.nodes[n]["lng"] = lng + rng.normal(0, 1e-5)
        g_prop.nodes[n]["lat"] = lat + rng.normal(0, 1e-5)
    return g_truth, g_prop


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Compute APLS between two road graphs.")
    p.add_argument("--demo", action="store_true", help="run on synthetic graphs")
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--weight", default="length")
    p.add_argument("--max-snap-m", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.demo:
        p.print_help()
        print(
            "\nTODO: load real graphs, e.g.\n"
            "  gt = load_osm_graph(place='Bengaluru, India')\n"
            "  pred = graph_from_geojson(<POST /api/infer network>)\n"
            "  print(apls(gt, pred, weight='length'))"
        )
        return 0

    g_truth, g_prop = _demo_graphs(seed=args.seed)
    score = apls(
        g_truth, g_prop,
        n_pairs=args.n_pairs, weight=args.weight,
        max_snap_m=args.max_snap_m, seed=args.seed,
    )
    print(f"truth: {g_truth.number_of_nodes()} nodes, "
          f"{g_truth.number_of_edges()} edges")
    print(f"prop : {g_prop.number_of_nodes()} nodes, "
          f"{g_prop.number_of_edges()} edges (1 critical link removed)")
    print(f"APLS = {score:.4f}  (1.0 == identical path structure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
