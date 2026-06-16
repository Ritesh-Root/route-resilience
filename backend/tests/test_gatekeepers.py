"""Tests for GET /api/gatekeepers.

Verifies each returned node carries the full contract
(id, lat, lng, betweenness, isArticulation, label) and that top_k is respected.

Run from the backend/ directory:
    # preferred (install pytest into the existing venv first):
    .venv/bin/pip install pytest         # TODO: add `pytest` to requirements.txt
    .venv/bin/pytest tests/test_gatekeepers.py -q

    # or, with no pytest installed, run it as a plain script (uses the same asserts):
    .venv/bin/python tests/test_gatekeepers.py

Only stdlib + already-installed fastapi/httpx are used, so no extra downloads
or GPU are required.
"""
from __future__ import annotations

import os
import sys

# Make `import app...` work whether pytest is launched from backend/ or elsewhere.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

# Field name -> expected python type for every GatekeeperNode (see app/schemas.py).
_EXPECTED_FIELDS = {
    "id": str,
    "lat": float,
    "lng": float,
    "betweenness": float,
    "isArticulation": bool,
    "label": str,
}


def _get(top_k: int | None = None, **params):
    """Call /api/gatekeepers and return the parsed JSON list, asserting 200."""
    query = {"city": "Bengaluru", **params}
    if top_k is not None:
        query["top_k"] = top_k
    resp = client.get("/api/gatekeepers", params=query)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list), f"expected a JSON array, got {type(body)}"
    return body


def test_gatekeepers_returns_full_node_contract():
    """Every node must expose exactly the contract fields with correct types."""
    nodes = _get(top_k=8)
    assert nodes, "expected at least one gatekeeper node"
    for node in nodes:
        assert isinstance(node, dict)
        for field, py_type in _EXPECTED_FIELDS.items():
            assert field in node, f"missing field {field!r} in {node}"
            # bool is a subclass of int, so check it before any int/float check.
            assert isinstance(node[field], py_type), (
                f"field {field!r} = {node[field]!r} is {type(node[field]).__name__}, "
                f"expected {py_type.__name__}"
            )


def test_gatekeepers_betweenness_and_coords_in_range():
    """Sanity-check value ranges: betweenness in [0,1], lat/lng finite."""
    for node in _get(top_k=8):
        assert 0.0 <= node["betweenness"] <= 1.0, node
        # Bengaluru is roughly lat ~12.9, lng ~77.6; just assert plausible bounds.
        assert -90.0 <= node["lat"] <= 90.0, node
        assert -180.0 <= node["lng"] <= 180.0, node


def test_gatekeepers_ids_are_unique():
    ids = [n["id"] for n in _get(top_k=8)]
    assert len(ids) == len(set(ids)), f"duplicate node ids returned: {ids}"


def test_top_k_limits_result_count():
    """The endpoint must never return more than top_k nodes."""
    for k in (1, 3, 5):
        nodes = _get(top_k=k)
        assert len(nodes) <= k, f"top_k={k} returned {len(nodes)} nodes"


def test_top_k_one_returns_single_node():
    nodes = _get(top_k=1)
    assert len(nodes) == 1, f"expected exactly 1 node, got {len(nodes)}"


def test_top_k_orders_by_betweenness_desc():
    """Top gatekeepers should be ranked by descending betweenness."""
    bw = [n["betweenness"] for n in _get(top_k=8)]
    assert bw == sorted(bw, reverse=True), f"not sorted by betweenness desc: {bw}"


def test_top_k_is_a_prefix_of_larger_top_k():
    """top_k=k should be the first k of a larger ranked list (stable ranking)."""
    small = [n["id"] for n in _get(top_k=3)]
    large = [n["id"] for n in _get(top_k=8)]
    assert large[: len(small)] == small, (small, large)


def test_all_input_model_combos_return_nodes():
    """Endpoint works for every (input, model) view used by the demo."""
    for input_mode in ("clean", "occluded"):
        for model in ("baseline", "robust"):
            nodes = _get(top_k=5, input=input_mode, model=model)
            assert nodes, f"no gatekeepers for input={input_mode} model={model}"


if __name__ == "__main__":
    # Allow running without pytest: execute every test_* function and report.
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{'OK' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
