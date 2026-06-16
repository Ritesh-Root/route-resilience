"""Graph 'healing' — close small gaps between near endpoints to repair connectivity.

WHY THIS FILE EXISTS
--------------------
After a segmentation mask is vectorised into a road graph (skeletonise -> nodes +
edges), occlusion-induced holes in the mask leave the graph *fragmented*: a road
that runs under a cloud / shadow / tree canopy becomes two dangling endpoints with
a small spatial gap between them. That is exactly the demo's failure mode
(``input=occluded & model=baseline`` -> fragmented network, low Connectivity
Ratio / resilienceIndex ~79). The robust path "recovers the critical links the
baseline loses under occlusion" — partly via the connectivity-aware model
(``ml/models/coanet.py``) and partly via this purely geometric post-process that
bridges the residual short gaps the model still drops.

The occlusion that *creates* these gaps is modelled in
``backend/app/services/occlusion.py`` (numpy-only blob masks for clouds / shadows /
canopy). Healing is the inverse intuition at the graph level: where two road
endpoints are closer than a tolerance (the typical width of such a blob's effect
on the vectorised line), assume the occluder hid a real connection and add a
bridge edge — but only when it *helps* connectivity, chosen greedily so the
largest connected component grows fastest.

APPROACH (pure networkx / numpy — no heavy / geo deps)
------------------------------------------------------
1. Find candidate endpoints: degree-1 nodes (dangling road tips). These are where
   occlusion most plausibly severed a road. Isolated (degree-0) nodes are included
   too so they can be reattached.
2. Generate candidate bridges between endpoints in *different* connected components
   whose great-circle distance is <= ``gap_tolerance_m``.
3. Add bridges greedily by ascending length using a disjoint-set (union-find), i.e.
   a Kruskal-style MST over the components: each accepted bridge must merge two
   currently-separate components (no intra-component shortcuts), so every added
   edge strictly reduces the component count and grows the giant component. This
   maximises the Connectivity Ratio for the fewest / shortest added edges.

The added bridge edges carry the same schema as the rest of the graph
(``id`` / ``length_m`` / ``road_class`` / ``travel_time``; node attrs ``lat`` /
``lng``) so everything downstream — ``backend/app/services/graph_build.py``,
``criticality.py``, ``gatekeeper`` logic — works unchanged. Bridges are tagged
``healed=True`` so the UI / metrics can highlight what robustness recovered.

Schema reused (see backend/app/services/network_factory.py + graph_build.py):
    node attrs : lat (float), lng (float)
    edge attrs : id (str), length_m (float), road_class ("arterial"|"local"),
                 travel_time (float, seconds)  [+ healed (bool) on bridges]

This module does NO model training, downloads nothing, and needs no GPU. networkx
is imported lazily inside functions (same defensive posture as
``ml/data/osmnx_bengaluru.py`` / ``backend/app/services/occlusion.py``) so the file
imports cleanly on a bare interpreter; numpy is used only for the vectorised
distance math.

Run a self-contained synthetic demo (no install beyond numpy + networkx)::

    python ml/vectorize/healing.py

Pinned deps (already in ml/requirements-ml.txt — do NOT pip install here):
    networkx==3.4.2
    numpy==2.1.3
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # import only for type-checkers; never required at runtime
    import networkx as nx

# Match SPEED_KMH in backend/app/services/network_factory.py so healed bridges get
# travel times consistent with the rest of the graph.
SPEED_KMH: dict[str, float] = {"arterial": 50.0, "local": 30.0}

# Mean Earth radius (metres) for the haversine distance between [lat, lng] nodes.
_EARTH_RADIUS_M: float = 6_371_000.0


def _require_networkx() -> Any:
    """Import networkx lazily with a clear, actionable error if it is missing."""
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "networkx is required for ml/vectorize/healing.py. "
            "Install the pinned version (do NOT pip install ad-hoc here):\n"
            "    pip install networkx==3.4.2\n"
            "It is already listed in ml/requirements-ml.txt."
        ) from exc
    return nx


# ---------------------------------------------------------------------------
# Disjoint-set (union-find) — the core of the Kruskal-style component merge.
# ---------------------------------------------------------------------------
class _DisjointSet:
    """Union-find with path compression + union by size.

    Tracks which connected component each node belongs to as bridges are added,
    so we can accept a candidate bridge only when it merges two *different*
    components (guaranteeing every accepted edge grows the giant component).
    """

    def __init__(self, items: Iterable[Any]) -> None:
        self._parent: dict[Any, Any] = {x: x for x in items}
        self._size: dict[Any, int] = {x: 1 for x in self._parent}

    def find(self, x: Any) -> Any:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Any, b: Any) -> bool:
        """Merge the sets of ``a`` and ``b``; return False if already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        return True

    def largest_size(self) -> int:
        return max(self._size[self.find(r)] for r in self._parent) if self._parent else 0


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy / math; coords are degrees lat / lng).
# ---------------------------------------------------------------------------
def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres between two [lat, lng] points (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _node_xy(G: "nx.Graph", n: Any) -> tuple[float, float]:
    """Return (lat, lng) for a node, raising a clear error if missing."""
    d = G.nodes[n]
    try:
        return float(d["lat"]), float(d["lng"])
    except KeyError as exc:  # pragma: no cover - bad input graph
        raise KeyError(
            f"node {n!r} lacks 'lat'/'lng' attributes required for healing "
            "(see backend/app/services/network_factory.py schema)."
        ) from exc


# ---------------------------------------------------------------------------
# Connectivity Ratio — the metric healing is optimising.
# ---------------------------------------------------------------------------
def measure_connectivity_ratio(G: "nx.Graph") -> float:
    """Fraction of nodes in the largest connected component (0..1).

    This is the project's "Connectivity Ratio": 1.0 means a single intact network,
    lower means the graph is split into islands. Mirrors the giant-component logic
    in ``backend/app/services/criticality.py`` (``max(len(c) for c in
    nx.connected_components(G))``). Returns 0.0 for an empty graph.
    """
    nx = _require_networkx()
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    return largest / n


# ---------------------------------------------------------------------------
# Candidate generation.
# ---------------------------------------------------------------------------
def _endpoints(G: "nx.Graph") -> list[Any]:
    """Dangling road tips (degree<=1) — where occlusion most plausibly cut a road."""
    return [n for n, deg in G.degree() if deg <= 1]


def _candidate_bridges(
    G: "nx.Graph",
    endpoints: list[Any],
    gap_tolerance_m: float,
) -> list[tuple[float, Any, Any]]:
    """All (distance, u, v) endpoint pairs within ``gap_tolerance_m``, sorted short-first.

    O(E^2) over endpoints, where E is usually small relative to |V| (only dangling
    tips qualify). For very large endpoint sets, swap this for a scipy.spatial
    cKDTree query — kept dependency-free here on purpose.
    """
    coords = {n: _node_xy(G, n) for n in endpoints}
    out: list[tuple[float, Any, Any]] = []
    for i in range(len(endpoints)):
        ui = endpoints[i]
        lat1, lng1 = coords[ui]
        for j in range(i + 1, len(endpoints)):
            vj = endpoints[j]
            lat2, lng2 = coords[vj]
            # Cheap bounding pre-filter (~1 deg lat ~= 111 km) before haversine.
            if abs(lat1 - lat2) * 111_000.0 > gap_tolerance_m:
                continue
            dist = _haversine_m(lat1, lng1, lat2, lng2)
            if dist <= gap_tolerance_m:
                out.append((dist, ui, vj))
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def heal_graph(
    G: "nx.Graph",
    gap_tolerance_m: float = 35.0,
    road_class: str = "local",
    in_place: bool = False,
) -> tuple["nx.Graph", list[str]]:
    """Bridge small gaps between near endpoints to maximise the giant component.

    Args:
        G: undirected road graph with node attrs ``lat``/``lng`` and edge attrs
            matching the backend schema (``id``/``length_m``/``road_class``/
            ``travel_time``).
        gap_tolerance_m: max great-circle gap (metres) to consider bridging. Tune
            to roughly the spatial footprint an occluder leaves on the vectorised
            line; too large invents roads that don't exist.
        road_class: class assigned to healed bridges ("local" | "arterial"); drives
            the travel-time estimate via ``SPEED_KMH``.
        in_place: mutate ``G`` directly when True, else operate on a copy.

    Returns:
        (healed_graph, added_edge_ids) — ``added_edge_ids`` are the ``id`` values of
        the bridge edges added (each also tagged ``healed=True``).

    Each accepted bridge merges two previously-separate components (Kruskal-style
    over a disjoint-set), so the component count strictly decreases and the
    Connectivity Ratio is non-decreasing with every edge added.
    """
    nx = _require_networkx()
    if road_class not in SPEED_KMH:
        raise ValueError(f"road_class must be one of {sorted(SPEED_KMH)}; got {road_class!r}")

    H: "nx.Graph" = G if in_place else G.copy()

    # Disjoint-set seeded with the graph's *current* components so candidate
    # bridges are only accepted when they join two different islands.
    ds = _DisjointSet(H.nodes())
    for u, v in H.edges():
        ds.union(u, v)

    endpoints = _endpoints(H)
    candidates = _candidate_bridges(H, endpoints, gap_tolerance_m)

    speed_ms = SPEED_KMH[road_class] * 1000.0 / 3600.0
    added_ids: list[str] = []
    for dist, u, v in candidates:
        # Skip if an endpoint already got consumed into u's side by an earlier
        # bridge, or if u and v are already connected (no shortcut edges).
        if not ds.union(u, v):
            continue
        edge_id = f"heal-{u}-{v}"
        H.add_edge(
            u,
            v,
            id=edge_id,
            length_m=round(dist, 1),
            road_class=road_class,
            travel_time=round(dist / speed_ms, 1) if speed_ms > 0 else 0.0,
            healed=True,
        )
        added_ids.append(edge_id)

    return H, added_ids


# ---------------------------------------------------------------------------
# Self-contained synthetic demo (no datasets, no GPU, no training).
# ---------------------------------------------------------------------------
def _demo_graph() -> "nx.Graph":
    """Build a tiny fragmented graph: two road stubs split by a ~25 m occlusion gap.

    Coordinates are around central Bengaluru to match the demo city; the gap is
    small enough that ``heal_graph`` should bridge it and restore a single
    connected component.
    """
    nx = _require_networkx()
    G: "nx.Graph" = nx.Graph()
    # Left stub: a -> b (an arterial running east).
    G.add_node("a", lat=12.97160, lng=77.59450)
    G.add_node("b", lat=12.97160, lng=77.59550)  # ~108 m east of a
    G.add_edge("a", "b", id="e-ab", length_m=108.0, road_class="arterial", travel_time=7.8)
    # Right stub: c -> d, starting ~25 m east of b (the occluded gap b..c).
    G.add_node("c", lat=12.97160, lng=77.59573)  # ~25 m east of b
    G.add_node("d", lat=12.97160, lng=77.59673)
    G.add_edge("c", "d", id="e-cd", length_m=108.0, road_class="arterial", travel_time=7.8)
    # A far-away isolated stub that must NOT be bridged (gap > tolerance).
    G.add_node("x", lat=12.98000, lng=77.61000)
    G.add_node("y", lat=12.98000, lng=77.61100)
    G.add_edge("x", "y", id="e-xy", length_m=108.0, road_class="local", travel_time=13.0)
    return G


def _main() -> None:
    nx = _require_networkx()
    G = _demo_graph()
    before = measure_connectivity_ratio(G)
    components_before = nx.number_connected_components(G)
    print("=== Graph healing demo (synthetic Bengaluru stubs) ===")
    print(f"before : nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"components={components_before} connectivity_ratio={before:.3f}")

    healed, added = heal_graph(G, gap_tolerance_m=35.0, road_class="arterial")
    after = measure_connectivity_ratio(healed)
    components_after = nx.number_connected_components(healed)
    print(f"healed : nodes={healed.number_of_nodes()} edges={healed.number_of_edges()} "
          f"components={components_after} connectivity_ratio={after:.3f}")
    print(f"bridges added ({len(added)}): {added}")
    # The b..c gap (~25 m) should be bridged; the x/y stub (>900 m away) should not.
    print("expectation: 1 bridge added, connectivity_ratio rises, x/y stub untouched.")


if __name__ == "__main__":
    # TODO(integrate): call heal_graph(...) inside the vectorise pipeline right
    # after skeleton->graph conversion and before criticality/gatekeeper analysis,
    # then expose measure_connectivity_ratio(...) as the /api/metrics
    # connectivityRatio field for the robust model path.
    _main()
