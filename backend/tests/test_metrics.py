"""Tests for GET /api/metrics across all (input, model) combinations.

Run from the backend/ directory:  cd backend && python -m pytest

These tests assert the demo invariant that drives the whole story:
(occluded, baseline) yields a FRAGMENTED network (resilienceIndex ~79) while
every other combo yields an intact network (resilienceIndex == 100). They also
verify the metrics contract: all segmentation-quality keys present and in [0, 1].
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Segmentation-quality keys that must be present and normalized to [0, 1].
# resilienceIndex is an int 0..100 and is checked separately.
UNIT_INTERVAL_KEYS = ("iou", "dice", "occlusionRecall", "connectivityRatio", "apls")

ALL_COMBOS = [
    ("clean", "baseline"),
    ("clean", "robust"),
    ("occluded", "baseline"),
    ("occluded", "robust"),
]


def _get_metrics(input_mode: str, model: str) -> dict:
    resp = client.get("/api/metrics",
                      params={"city": "Bengaluru", "input": input_mode, "model": model})
    assert resp.status_code == 200, f"/api/metrics failed for ({input_mode}, {model}): {resp.text}"
    return resp.json()


@pytest.mark.parametrize("input_mode, model", ALL_COMBOS)
def test_all_metric_keys_present(input_mode, model):
    data = _get_metrics(input_mode, model)
    for key in UNIT_INTERVAL_KEYS + ("resilienceIndex",):
        assert key in data, f"missing key {key!r} for ({input_mode}, {model})"


@pytest.mark.parametrize("input_mode, model", ALL_COMBOS)
def test_unit_interval_metrics_in_range(input_mode, model):
    data = _get_metrics(input_mode, model)
    for key in UNIT_INTERVAL_KEYS:
        val = data[key]
        assert isinstance(val, (int, float)), f"{key} should be numeric, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] for ({input_mode}, {model})"


@pytest.mark.parametrize("input_mode, model", ALL_COMBOS)
def test_resilience_index_is_int_in_range(input_mode, model):
    data = _get_metrics(input_mode, model)
    ri = data["resilienceIndex"]
    assert isinstance(ri, int), f"resilienceIndex should be int, got {type(ri)}"
    assert 0 <= ri <= 100, f"resilienceIndex={ri} out of [0,100]"


def test_demo_invariant_occluded_baseline_is_fragmented():
    """The one fragmented combo: resilienceIndex should sit around 79."""
    ri = _get_metrics("occluded", "baseline")["resilienceIndex"]
    # "79-ish" — allow a small band so live graph computation isn't brittle,
    # while still clearly distinguishing it from the intact (100) case.
    assert 75 <= ri <= 85, f"expected fragmented ~79 for (occluded, baseline), got {ri}"
    assert ri < 100, "fragmented network must not score a perfect 100"


@pytest.mark.parametrize("input_mode, model", [
    ("clean", "baseline"),
    ("clean", "robust"),
    ("occluded", "robust"),
])
def test_demo_invariant_other_combos_are_intact(input_mode, model):
    """Every combo except (occluded, baseline) must be intact (resilienceIndex == 100)."""
    ri = _get_metrics(input_mode, model)["resilienceIndex"]
    assert ri == 100, f"expected intact 100 for ({input_mode}, {model}), got {ri}"
