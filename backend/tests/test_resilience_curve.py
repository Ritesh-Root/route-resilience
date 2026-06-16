"""Tests for GET /api/resilience-curve.

Run from the backend/ directory:  cd backend && python -m pytest

Validates the shape and ordering invariants of the resilience curve across all
(input, model) combinations:
  * the three arrays are non-empty and equal length,
  * removedFraction is monotonic non-decreasing and bounded to [0, 1],
  * efficiency starts at 1.0 (full network) and is "non-increasing-ish"
    (removing links never *helps* global efficiency, modulo float noise),
  * giantComponent is a fraction in [0, 1].
"""
from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# All view combinations the backend serves; the curve invariants hold for every one.
COMBOS = list(itertools.product(("clean", "occluded"), ("baseline", "robust")))

# Allow a small tolerance so floating-point jitter doesn't trip the ordering checks.
EPS = 1e-9


def _get_curve(input_mode: str, model: str) -> dict:
    resp = client.get(
        "/api/resilience-curve",
        params={"city": "Bengaluru", "input": input_mode, "model": model},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("input_mode,model", COMBOS)
def test_arrays_present_and_equal_length(input_mode: str, model: str):
    curve = _get_curve(input_mode, model)
    removed = curve["removedFraction"]
    efficiency = curve["efficiency"]
    giant = curve["giantComponent"]

    assert len(removed) > 0
    assert len(removed) == len(efficiency) == len(giant)


@pytest.mark.parametrize("input_mode,model", COMBOS)
def test_removed_fraction_monotonic_in_unit_interval(input_mode: str, model: str):
    removed = _get_curve(input_mode, model)["removedFraction"]

    assert all(0.0 - EPS <= x <= 1.0 + EPS for x in removed), removed
    assert all(b >= a - EPS for a, b in zip(removed, removed[1:])), removed


@pytest.mark.parametrize("input_mode,model", COMBOS)
def test_efficiency_starts_at_one_and_is_non_increasing_ish(input_mode: str, model: str):
    efficiency = _get_curve(input_mode, model)["efficiency"]

    # The curve is normalized: the intact network is efficiency 1.0.
    assert efficiency[0] == pytest.approx(1.0, abs=1e-6)
    # Removing edges/nodes should never increase global efficiency (small slack
    # for float noise and any tie-breaking in the removal order).
    assert all(b <= a + 1e-6 for a, b in zip(efficiency, efficiency[1:])), efficiency


@pytest.mark.parametrize("input_mode,model", COMBOS)
def test_giant_component_is_unit_fraction(input_mode: str, model: str):
    giant = _get_curve(input_mode, model)["giantComponent"]

    assert all(0.0 - EPS <= x <= 1.0 + EPS for x in giant), giant
