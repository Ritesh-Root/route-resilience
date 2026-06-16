"""Unit tests for :mod:`app.services.criticality`.

These exercise the pure graph-theoretic core on tiny, hand-built NetworkX
graphs whose expected values can be computed by hand — no segmentation
pipeline, no synthetic-network generation, no GPU. Run with::

    cd backend && python -m pytest tests/test_criticality_unit.py

(``backend/pytest.ini`` makes ``backend/`` the rootdir so the ``app`` package
imports resolve.)
"""
from __future__ import annotations

import networkx as nx
import pytest

from app.services import criticality as crit

# Every edge in these graphs carries a unit travel time so shortest-path
# distances equal hop counts and the expected metrics stay hand-checkable.
WEIGHT = crit.WEIGHT


def _path_graph_unit_weights() -> nx.Graph:
    """Path 0 -- 1 -- 2 with ``travel_time == 1`` on each edge."""
    G = nx.Graph()
    G.add_edge(0, 1, **{WEIGHT: 1.0, "id": "e01"})
    G.add_edge(1, 2, **{WEIGHT: 1.0, "id": "e12"})
    return G


# --------------------------------------------------------------------------- #
# global_efficiency_weighted
# --------------------------------------------------------------------------- #
def test_global_efficiency_path_graph():
    """On P3 the weighted global efficiency is exactly 5/6.

    Ordered reachable pairs (n*(n-1) = 6 denominator):
      distance 1: (0,1)(1,0)(1,2)(2,1) -> 1/1 each  = 4.0
      distance 2: (0,2)(2,0)           -> 1/2 each  = 1.0
    sum = 5.0; 5.0 / 6 = 0.8333...
    """
    G = _path_graph_unit_weights()
    assert crit.global_efficiency_weighted(G) == pytest.approx(5.0 / 6.0)


def test_global_efficiency_single_node_is_zero():
    """Fewer than two nodes => no pairs => efficiency 0.0 (no ZeroDivision)."""
    G = nx.Graph()
    G.add_node(0)
    assert crit.global_efficiency_weighted(G) == 0.0


def test_global_efficiency_disconnected_pairs_contribute_zero():
    """Unreachable pairs add nothing: two isolated unit edges.

    Graph: 0--1 and 2--3 (two components). Reachable ordered pairs are the
    intra-component ones: (0,1)(1,0)(2,3)(3,2) = 4 pairs * (1/1) = 4.0,
    over n*(n-1) = 12.
    """
    G = nx.Graph()
    G.add_edge(0, 1, **{WEIGHT: 1.0, "id": "a"})
    G.add_edge(2, 3, **{WEIGHT: 1.0, "id": "b"})
    assert crit.global_efficiency_weighted(G) == pytest.approx(4.0 / 12.0)


# --------------------------------------------------------------------------- #
# bridges
# --------------------------------------------------------------------------- #
def test_bridges_identifies_known_bridge():
    """Two triangles joined by a single link: only that link is a bridge.

    Triangle A: 0-1-2-0 ; Triangle B: 3-4-5-3 ; bridge edge: 2-3.
    Edges inside a cycle are never bridges; the connecting edge is.
    """
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        G.add_edge(u, v, **{WEIGHT: 1.0, "id": f"{u}{v}"})
    G.add_edge(2, 3, **{WEIGHT: 1.0, "id": "bridge"})

    result = crit.bridges(G)
    assert result == {frozenset((2, 3))}


def test_bridges_path_graph_all_edges_are_bridges():
    """In a tree (P3) every edge is a single point of failure."""
    G = _path_graph_unit_weights()
    assert crit.bridges(G) == {frozenset((0, 1)), frozenset((1, 2))}


def test_bridges_empty_graph_is_empty_set():
    assert crit.bridges(nx.Graph()) == set()


def test_bridges_cycle_has_no_bridges():
    """A pure cycle has redundancy everywhere -> no bridges."""
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        G.add_edge(u, v, **{WEIGHT: 1.0, "id": f"{u}{v}"})
    assert crit.bridges(G) == set()


# --------------------------------------------------------------------------- #
# resilience_index
# --------------------------------------------------------------------------- #
def test_resilience_index_within_bounds():
    """Index stays in [0, 100] for intact, degraded and over-budget base_eff."""
    G = _path_graph_unit_weights()
    base = crit.global_efficiency_weighted(G)
    for be in (base, base * 0.5, base * 2.0, 1e-9):
        idx = crit.resilience_index(G, be)
        assert isinstance(idx, int)
        assert 0 <= idx <= 100


def test_resilience_index_full_when_base_matches_current():
    """When the graph is its own baseline: eff_ratio=1, giant=1 -> 100."""
    G = _path_graph_unit_weights()
    base = crit.global_efficiency_weighted(G)
    assert crit.resilience_index(G, base) == 100


def test_resilience_index_nonpositive_base_is_zero():
    """Guard clause: base_eff <= 0 returns 0 rather than dividing by zero."""
    G = _path_graph_unit_weights()
    assert crit.resilience_index(G, 0.0) == 0
    assert crit.resilience_index(G, -1.0) == 0


def test_resilience_index_eff_ratio_is_capped_at_one():
    """A tiny baseline would push eff_ratio above 1; min() caps it, so the
    score cannot exceed 100 (0.6*1 + 0.4*1)."""
    G = _path_graph_unit_weights()
    idx = crit.resilience_index(G, base_eff=1e-6)
    assert idx == 100


# --------------------------------------------------------------------------- #
# Empty-graph safety across the surface
# --------------------------------------------------------------------------- #
def test_empty_graph_safety():
    """No criticality primitive should raise on an empty graph."""
    G = nx.Graph()
    assert crit.global_efficiency_weighted(G) == 0.0
    assert crit.bridges(G) == set()
    assert crit.giant_component_fraction(G) == 0.0
    assert crit.resilience_index(G, base_eff=1.0) == 0
    assert crit.edge_betweenness(G) == {}
    assert crit.node_betweenness(G) == {}


def test_giant_component_fraction_basic():
    """0--1 plus an isolated node 2: largest component = 2 of 3 nodes."""
    G = nx.Graph()
    G.add_edge(0, 1, **{WEIGHT: 1.0, "id": "a"})
    G.add_node(2)
    assert crit.giant_component_fraction(G) == pytest.approx(2.0 / 3.0)


if __name__ == "__main__":  # pragma: no cover - allow plain `python` execution
    raise SystemExit(pytest.main([__file__, "-v"]))
