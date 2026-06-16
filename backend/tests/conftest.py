"""Pytest fixtures for the Route Resilience backend tests.

Run from the backend/ directory:  cd backend && python -m pytest
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the backend/ root (parent of this tests/ dir) to sys.path so that
# `import app.main` resolves regardless of the pytest invocation directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402  (import after sys.path setup)


@pytest.fixture()
def client():
    """A FastAPI TestClient bound to the application instance."""
    with TestClient(app) as test_client:
        yield test_client
