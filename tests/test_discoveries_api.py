"""Tests for /api/discoveries routes.

All Supabase calls are mocked so the suite is deterministic and offline.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_client() -> TestClient:
    """Build a TestClient with the discoveries router."""
    from api.routes.discoveries import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


_SAMPLE_ROWS = [
    {
        "id": "1", "series_a": "SPY", "series_b": "GLD",
        "lag_days": 5, "pearson_r": 0.72, "granger_p": 0.03,
        "mutual_info": 0.1, "regime": "all", "strength": "strong",
        "relationship_type": "discovered", "run_id": "run-001",
        "computed_at": "2026-03-22T12:00:00+00:00",
    },
    {
        "id": "2", "series_a": "TLT", "series_b": "DFF",
        "lag_days": 3, "pearson_r": -0.55, "granger_p": 0.08,
        "mutual_info": 0.05, "regime": "all", "strength": "moderate",
        "relationship_type": "rates_bonds", "run_id": "run-001",
        "computed_at": "2026-03-22T12:00:00+00:00",
    },
    {
        "id": "3", "series_a": "SPY", "series_b": "VIXCLS",
        "lag_days": 1, "pearson_r": -0.80, "granger_p": 0.001,
        "mutual_info": 0.15, "regime": "bear", "strength": "strong",
        "relationship_type": "volatility_equity", "run_id": "run-002",
        "computed_at": "2026-03-23T12:00:00+00:00",
    },
]


# ── GET /api/discoveries ─────────────────────────────────────────────────────

class TestListDiscoveries:
    def test_returns_200_with_all(self):
        """Default call returns all discoveries."""
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            r = client.get("/api/discoveries")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        assert len(data["discoveries"]) == 3

    def test_returns_empty(self):
        """Empty discoveries table returns total=0."""
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[]):
            r = client.get("/api/discoveries")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_passes_series_filter(self):
        """series query param is forwarded to get_discoveries."""
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[_SAMPLE_ROWS[0]]) as mock_get:
            r = client.get("/api/discoveries", params={"series": "SPY"})
        assert r.status_code == 200
        assert r.json()["filters"]["series"] == "SPY"
        mock_get.assert_called_once_with(series="SPY", strength=None, run_id=None, min_abs_r=None, limit=50)

    def test_passes_strength_filter(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[_SAMPLE_ROWS[0]]) as mock_get:
            r = client.get("/api/discoveries", params={"strength": "strong"})
        assert r.status_code == 200
        assert r.json()["filters"]["strength"] == "strong"
        mock_get.assert_called_once_with(series=None, strength="strong", run_id=None, min_abs_r=None, limit=50)

    def test_passes_min_abs_r_filter(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[_SAMPLE_ROWS[0]]):
            r = client.get("/api/discoveries", params={"min_abs_r": 0.7})
        assert r.status_code == 200
        assert r.json()["filters"]["min_abs_r"] == 0.7

    def test_passes_run_id_filter(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[_SAMPLE_ROWS[0]]):
            r = client.get("/api/discoveries", params={"run_id": "run-001"})
        assert r.status_code == 200
        assert r.json()["filters"]["run_id"] == "run-001"

    def test_passes_limit(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[_SAMPLE_ROWS[0]]):
            r = client.get("/api/discoveries", params={"limit": 1})
        assert r.status_code == 200

    def test_limit_validation_min(self):
        """limit < 1 returns 422."""
        client = _make_client()
        r = client.get("/api/discoveries", params={"limit": 0})
        assert r.status_code == 422

    def test_limit_validation_max(self):
        """limit > 500 returns 422."""
        client = _make_client()
        r = client.get("/api/discoveries", params={"limit": 501})
        assert r.status_code == 422

    def test_min_abs_r_validation(self):
        """min_abs_r > 1 returns 422."""
        client = _make_client()
        r = client.get("/api/discoveries", params={"min_abs_r": 1.5})
        assert r.status_code == 422

    def test_filters_omit_none_values(self):
        """filters dict should only include explicitly set params."""
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[]):
            r = client.get("/api/discoveries")
        data = r.json()
        # Only limit should be present (it has a default of 50)
        assert "series" not in data["filters"]
        assert "strength" not in data["filters"]
        assert "run_id" not in data["filters"]

    def test_db_error_returns_500(self):
        """Supabase failure returns 500 with error detail."""
        client = _make_client()
        with patch("db.supabase.client.get_client", side_effect=Exception("connection refused")):
            r = client.get("/api/discoveries")
        assert r.status_code == 500
        assert "connection refused" in r.json()["detail"]


# ── GET /api/discoveries/summary ─────────────────────────────────────────────

class TestDiscoverySummary:
    def test_returns_200(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            r = client.get("/api/discoveries/summary")
        assert r.status_code == 200

    def test_counts_total(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            data = client.get("/api/discoveries/summary").json()
        assert data["total_discoveries"] == 3

    def test_breakdown_by_strength(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            data = client.get("/api/discoveries/summary").json()
        assert data["by_strength"]["strong"] == 2
        assert data["by_strength"]["moderate"] == 1

    def test_unique_series(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            data = client.get("/api/discoveries/summary").json()
        # SPY, GLD, TLT, DFF, VIXCLS = 5 unique
        assert data["unique_series"] == 5

    def test_run_count(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            data = client.get("/api/discoveries/summary").json()
        assert data["run_count"] == 2

    def test_latest_run_id(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=_SAMPLE_ROWS):
            data = client.get("/api/discoveries/summary").json()
        assert data["latest_run_id"] == "run-001"

    def test_empty_state(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", return_value=[]):
            data = client.get("/api/discoveries/summary").json()
        assert data["total_discoveries"] == 0
        assert data["by_strength"] == {}
        assert data["unique_series"] == 0
        assert data["run_count"] == 0
        assert data["latest_run_id"] is None

    def test_db_error_returns_500(self):
        client = _make_client()
        with patch("db.supabase.client.get_discoveries", side_effect=Exception("timeout")):
            r = client.get("/api/discoveries/summary")
        assert r.status_code == 500
