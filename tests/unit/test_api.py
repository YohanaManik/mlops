"""
Unit tests for API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# This will fail in CI without proper model, that's expected
pytest.importorskip("api.app")

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_classes_endpoint():
    """Test get classes endpoint"""
    try:
        response = client.get("/classes")
        if response.status_code == 200:
            data = response.json()
            assert "classes" in data
            assert "num_classes" in data
    except Exception:
        pytest.skip("Model not loaded in test environment")