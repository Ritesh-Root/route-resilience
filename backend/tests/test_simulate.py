"""Tests for POST /api/simulate (predictive impact assessment).

Verifies the two demo-critical behaviours of the what-if engine:

  1. Disabling a HIGH-CRITICALITY edge (a busy arterial that carries lots of
     shortest paths) lowers ``resilienceIndexAfter`` versus the untouched base,
     and the broken-route count stays within the sampled-route budget
     (brokenRoutesSampled <= sampledRoutes).

  2. Disabling a SPUR edge (a single-edge attachment that is a graph bridge,
     e.g. the suburb 'e_n0_2__s_north') strands its zone, so
     ``newlyDisconnectedZones`` increases above the base of 0.

These exercise the real NetworkX analytics in app.services.criticality through
the live FastAPI route, on the synthetic Bengaluru network.

Run (pytest preferred):
    cd backend
    .venv/bin/python -m pip install pytest        # one-time, tiny pure-py dep
    .venv/bin/python -m pytest tests/test_simulate.py -v

If you'd rather not install pytest, this file also runs standalone via its
__main__ fallback (no third-party test runner needed):
    .venv/bin/python tests/test_simulate.py
"""
from __future__ import annotations

import os
import sys

# Make the backend package importable when run from anywhere (tests/ -> backend/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import network_factory  # noqa: E402

client = TestClient(app)

# The fragmented demo view (occluded + baseline) starts already disconnected,
# so use an intact view as the baseline for these single-removal experiments.
CITY = "Bengaluru"
INTACT = {"city": CITY, "input": "clean", "model": "robust"}


def _base_resilience() -> int:
    """resilienceIndex of the intact network with nothing disabled."""
    r = client.post("/api/simulate", json={**INTACT, "disabledEdgeIds": [], "disabledNodeIds": []})
    assert r.status_code == 200, r.text
    return r.json()["resilienceIndexAfter"]


def _highest_criticality_edge_id() -> str:
    """ID of the single most critical edge from the live /api/network response."""
    r = client.get("/api/network", params={"city": CITY, "input": "clean", "model": "robust"})
    assert r.status_code == 200, r.text
    feats = r.json()["features"]
    top = max(feats, key=lambda f: f["properties"]["criticality"])
    return top["properties"]["id"]


def test_disable_high_criticality_edge_lowers_resilience():
    """Removing the most critical edge must reduce resilience vs the base."""
    base = _base_resilience()
    edge_id = _highest_criticality_edge_id()

    r = client.post("/api/simulate", json={
        **INTACT, "disabledEdgeIds": [edge_id], "disabledNodeIds": []})
    assert r.status_code == 200, r.text
    out = r.json()

    # The edge we asked to disable was actually present and removed.
    assert out["disabledEdgeIds"] == [edge_id]
    # Systemic impact: resilience drops (rerouting/efficiency loss).
    assert out["resilienceIndexAfter"] < base, (
        f"expected resilience < {base}, got {out['resilienceIndexAfter']}")
    # Sampling invariant always holds.
    assert 0 <= out["brokenRoutesSampled"] <= out["sampledRoutes"]


def test_disable_spur_edge_increases_disconnected_zones():
    """Removing a spur bridge strands its zone -> +1 disconnected zone."""
    # Spur edges are the single-edge suburb attachments; they are graph bridges
    # whose removal isolates a node. network_factory wires up s_north et al.
    spur_edge_id = "e_n0_2__s_north"

    # Sanity: the spur edge really exists in the intact network we're testing.
    net = client.get("/api/network", params={"city": CITY, "input": "clean", "model": "robust"})
    edge_ids = {f["properties"]["id"] for f in net.json()["features"]}
    assert spur_edge_id in edge_ids, f"{spur_edge_id} not in network; spurs changed?"

    base = client.post("/api/simulate", json={
        **INTACT, "disabledEdgeIds": [], "disabledNodeIds": []})
    assert base.json()["newlyDisconnectedZones"] == 0

    r = client.post("/api/simulate", json={
        **INTACT, "disabledEdgeIds": [spur_edge_id], "disabledNodeIds": []})
    assert r.status_code == 200, r.text
    out = r.json()

    assert out["disabledEdgeIds"] == [spur_edge_id]
    assert out["newlyDisconnectedZones"] >= 1, (
        f"expected a stranded zone, got {out['newlyDisconnectedZones']}")
    assert 0 <= out["brokenRoutesSampled"] <= out["sampledRoutes"]


def test_simulate_response_shape():
    """Contract guard: all SimulationResult fields present with sane types."""
    r = client.post("/api/simulate", json={
        **INTACT, "disabledEdgeIds": [], "disabledNodeIds": []})
    assert r.status_code == 200, r.text
    out = r.json()
    for key in ("disabledEdgeIds", "disabledNodeIds", "resilienceIndexAfter",
                "avgTravelTimeIncreasePct", "newlyDisconnectedZones",
                "brokenRoutesSampled", "sampledRoutes"):
        assert key in out, f"missing field: {key}"
    assert isinstance(out["resilienceIndexAfter"], int)
    assert isinstance(out["disabledEdgeIds"], list)
    # base_eff is precomputed and positive, so the intact base scores 100.
    assert network_factory.BASE_EFF > 0


if __name__ == "__main__":
    # Standalone fallback so the file is runnable without pytest installed.
    failures = 0
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
