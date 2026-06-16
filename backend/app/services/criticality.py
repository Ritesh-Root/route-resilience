"""Graph-theoretic criticality engine.

Pure NetworkX functions — these are the REAL analytics that power the
"Quantitative Criticality Map", "Gatekeeper Nodes" and "Predictive Impact
Assessment" deliverables. They operate on any weighted graph whose edges carry
a ``travel_time`` attribute (seconds) and an ``id`` attribute.

When the segmentation + vectorization pipeline is ready, the only change needed
is to feed the *real* extracted graph in place of the synthetic one — every
function here works unchanged.
"""
from __future__ import annotations

import random
from typing import Iterable

import networkx as nx

WEIGHT = "travel_time"


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def edge_betweenness(G: nx.Graph) -> dict:
    """Travel-time-weighted edge betweenness centrality."""
    return nx.edge_betweenness_centrality(G, weight=WEIGHT, normalized=True)


def node_betweenness(G: nx.Graph) -> dict:
    return nx.betweenness_centrality(G, weight=WEIGHT, normalized=True)


def bridges(G: nx.Graph) -> set:
    """Edges whose removal disconnects the graph (true single points of failure)."""
    if G.number_of_edges() == 0:
        return set()
    return {frozenset(e) for e in nx.bridges(G)}


def global_efficiency_weighted(G: nx.Graph) -> float:
    """Mean of 1/shortest-path-time over all reachable ordered node pairs.

    Robust to disconnection (unreachable pairs contribute 0). Lower = more
    fragile / fragmented network.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 2:
        return 0.0
    total = 0.0
    for u in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, u, weight=WEIGHT)
        for v, d in lengths.items():
            if v != u and d > 0:
                total += 1.0 / d
    return total / (n * (n - 1))


def giant_component_fraction(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    return largest / G.number_of_nodes()


def resilience_index(G: nx.Graph, base_eff: float) -> int:
    """0-100 index blending efficiency retention and giant-component size."""
    if base_eff <= 0:
        return 0
    eff_ratio = min(global_efficiency_weighted(G) / base_eff, 1.0)
    giant = giant_component_fraction(G)
    return round(100 * (0.6 * eff_ratio + 0.4 * giant))


# --------------------------------------------------------------------------- #
# Resilience curve (sequential attack)
# --------------------------------------------------------------------------- #
def resilience_curve(G: nx.Graph, base_eff: float | None = None,
                     steps: int = 10, max_fraction: float = 0.5) -> dict:
    """Remove edges in descending-betweenness order; track network decay.

    ``base_eff`` is the reference (intact-network) efficiency the curve is
    normalized against. Pass the intact ``BASE_EFF`` so a *fragmented* input
    correctly starts below 1.0 instead of falsely renormalizing to 1.0. If
    omitted, falls back to this graph's own efficiency (legacy behavior).
    """
    H = G.copy()
    m = H.number_of_edges()
    if m == 0:
        return {"removedFraction": [0.0], "efficiency": [0.0],
                "giantComponent": [giant_component_fraction(H)]}
    if base_eff is None or base_eff <= 0:
        base_eff = global_efficiency_weighted(H)

    def _ratio() -> float:
        return round((global_efficiency_weighted(H) / base_eff) if base_eff > 0 else 0.0, 3)

    ranked = [e for e, _ in sorted(
        edge_betweenness(H).items(), key=lambda kv: kv[1], reverse=True)]
    per_step = max(1, int((m * max_fraction) / steps))

    removed_fraction, efficiency, giant = [0.0], [_ratio()], [giant_component_fraction(H)]
    removed = 0
    for s in range(steps):
        for e in ranked[removed: removed + per_step]:
            if H.has_edge(*e):
                H.remove_edge(*e)
        removed += per_step
        removed_fraction.append(round(removed / m, 3))
        efficiency.append(round(
            (global_efficiency_weighted(H) / base_eff) if base_eff > 0 else 0.0, 3))
        giant.append(round(giant_component_fraction(H), 3))
        if removed >= m:
            break
    return {"removedFraction": removed_fraction, "efficiency": efficiency,
            "giantComponent": giant}


# --------------------------------------------------------------------------- #
# What-if simulation (predictive impact assessment)
# --------------------------------------------------------------------------- #
def simulate_removal(
    G: nx.Graph,
    base_eff: float,
    disabled_edge_ids: Iterable[str] | None = None,
    disabled_node_ids: Iterable[str] | None = None,
    n_samples: int = 60,
    seed: int = 7,
) -> dict:
    """Disable edges/nodes, then quantify the systemic impact."""
    H = G.copy()
    id2edge = {d["id"]: (u, v) for u, v, d in G.edges(data=True)}

    removed_edges: list[str] = []
    for eid in (disabled_edge_ids or []):
        uv = id2edge.get(eid)
        if uv and H.has_edge(*uv):
            H.remove_edge(*uv)
            removed_edges.append(eid)
    for nid in (disabled_node_ids or []):
        if H.has_node(nid):
            H.remove_node(nid)

    base_components = nx.number_connected_components(G)
    after_components = nx.number_connected_components(H) if H.number_of_nodes() else 0
    newly_disconnected = max(0, after_components - base_components)

    rng = random.Random(seed)
    nodes = list(G.nodes())
    increases: list[float] = []
    broken = 0
    for _ in range(n_samples):
        s, t = rng.choice(nodes), rng.choice(nodes)
        if s == t:
            continue
        try:
            base_d = nx.shortest_path_length(G, s, t, weight=WEIGHT)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        try:
            new_d = nx.shortest_path_length(H, s, t, weight=WEIGHT)
            if base_d > 0:
                increases.append((new_d - base_d) / base_d * 100.0)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            broken += 1

    avg_increase = round(sum(increases) / len(increases), 1) if increases else 0.0
    return {
        "disabledEdgeIds": removed_edges,
        "disabledNodeIds": list(disabled_node_ids or []),
        "resilienceIndexAfter": resilience_index(H, base_eff),
        "avgTravelTimeIncreasePct": avg_increase,
        "newlyDisconnectedZones": newly_disconnected,
        "brokenRoutesSampled": broken,
        "sampledRoutes": n_samples,
    }
