"""Smoke tests for the basic discovery endpoints.

Run from the ``backend`` directory:
    pytest tests/test_health_cities.py

Requires the deps in ``requirements.txt`` (fastapi, httpx for TestClient, pytest).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["baseEfficiency"] > 0


def test_cities(client):
    resp = client.get("/api/cities")
    assert resp.status_code == 200
    assert resp.json() == ["Bengaluru"]
