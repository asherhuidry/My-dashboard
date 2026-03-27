"""Tests for /api/sources routes.

All source registry access is mocked so the suite is deterministic.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    """Build a TestClient with the sources router."""
    from api.routes.sources import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _make_record(**overrides):
    """Build a mock SourceRecord-like object with .to_dict()."""
    from data.registry.source_registry import SourceRecord, SourceStatus
    defaults = dict(
        source_id="test_src",
        name="Test Source",
        category="macro",
        url="https://example.com/api",
        acquisition_method="api",
        auth_required=False,
        free_tier=True,
        rate_limit_notes="",
        update_frequency="daily",
        asset_classes=["macro"],
        data_types=["rates"],
        reliability_score=0.75,
        status=SourceStatus.VALIDATED,
        notes="",
        discovered_at="2026-03-01T00:00:00+00:00",
        last_checked_at="2026-03-20T00:00:00+00:00",
    )
    defaults.update(overrides)
    return SourceRecord(**defaults)


_SAMPLE_RECORDS = [
    _make_record(
        source_id="fred_api", name="FRED", category="macro",
        reliability_score=0.85, status=_make_record().status.__class__("validated"),
    ),
    _make_record(
        source_id="ecb_api", name="ECB", category="macro",
        reliability_score=0.78, status=_make_record().status.__class__("approved"),
    ),
    _make_record(
        source_id="yfinance", name="Yahoo Finance", category="equity",
        reliability_score=0.60, status=_make_record().status.__class__("sampled"),
    ),
    _make_record(
        source_id="sketchy_api", name="Sketchy Data", category="alternative",
        reliability_score=0.20, status=_make_record().status.__class__("rejected"),
    ),
]


def _mock_registry(records=None):
    """Create a mock SourceRegistry returning the given records."""
    if records is None:
        records = _SAMPLE_RECORDS
    mock_reg = MagicMock()
    mock_reg.all.return_value = list(records)

    def mock_filter(status=None, category=None, min_score=0.0, **_kw):
        results = list(records)
        if status is not None:
            results = [r for r in results if r.status == status]
        if category is not None:
            results = [r for r in results if r.category == category]
        if min_score > 0:
            results = [r for r in results if r.reliability_score >= min_score]
        return results

    mock_reg.filter.side_effect = mock_filter

    def mock_search(query):
        q = query.lower()
        return [r for r in records if q in r.name.lower() or q in r.source_id.lower()]

    mock_reg.search.side_effect = mock_search
    return mock_reg


# ── GET /api/sources ──────────────────────────────────────────────────────────

class TestListSources:
    def test_returns_all_sources(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 4

    def test_filter_by_status(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"status": "approved"})
        assert r.status_code == 200
        data = r.json()
        assert all(s["status"] == "approved" for s in data["sources"])

    def test_filter_by_category(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"category": "macro"})
        assert r.status_code == 200
        data = r.json()
        assert all(s["category"] == "macro" for s in data["sources"])

    def test_search(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"search": "fred"})
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        assert any("FRED" in s["name"] for s in data["sources"])

    def test_sort_by_score(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"sort": "score"})
        assert r.status_code == 200
        scores = [s["reliability_score"] for s in r.json()["sources"]]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_name(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"sort": "name"})
        assert r.status_code == 200
        names = [s["name"].lower() for s in r.json()["sources"]]
        assert names == sorted(names)

    def test_limit(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"limit": 2})
        assert r.status_code == 200
        assert len(r.json()["sources"]) <= 2

    def test_empty_registry(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry([])):
            r = client.get("/api/sources")
        assert r.status_code == 200
        assert r.json()["total"] == 0
        assert r.json()["sources"] == []

    def test_registry_error_returns_empty(self):
        """If registry can't load, return empty state not 500."""
        client = _make_client()
        with patch("api.routes.sources._get_registry", side_effect=Exception("file not found")):
            r = client.get("/api/sources")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["registry_missing"] is True

    def test_filters_in_response(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources", params={"status": "validated", "category": "macro"})
        data = r.json()
        assert data["filters"]["status"] == "validated"
        assert data["filters"]["category"] == "macro"


# ── GET /api/sources/summary ──────────────────────────────────────────────────

class TestSourceSummary:
    def test_returns_summary(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            r = client.get("/api/sources/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 4

    def test_by_status_counts(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            data = client.get("/api/sources/summary").json()
        assert data["by_status"]["validated"] == 1
        assert data["by_status"]["approved"] == 1
        assert data["by_status"]["sampled"] == 1
        assert data["by_status"]["rejected"] == 1

    def test_by_category_counts(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            data = client.get("/api/sources/summary").json()
        assert data["by_category"]["macro"] == 2
        assert data["by_category"]["equity"] == 1

    def test_top_scored(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            data = client.get("/api/sources/summary").json()
        assert len(data["top_scored"]) > 0
        scores = [s["score"] for s in data["top_scored"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_validated(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            data = client.get("/api/sources/summary").json()
        # Only validated + approved sources
        for s in data["top_validated"]:
            assert s["status"] in ("validated", "approved")

    def test_free_and_auth_counts(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry()):
            data = client.get("/api/sources/summary").json()
        assert data["free_count"] == 4  # all test records have free_tier=True
        assert data["auth_required_count"] == 0

    def test_empty_registry_summary(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", return_value=_mock_registry([])):
            data = client.get("/api/sources/summary").json()
        assert data["total"] == 0
        assert data["by_status"] == {}
        assert data["top_scored"] == []
        assert data["top_validated"] == []

    def test_summary_error_returns_empty(self):
        client = _make_client()
        with patch("api.routes.sources._get_registry", side_effect=Exception("boom")):
            r = client.get("/api/sources/summary")
        assert r.status_code == 200
        assert r.json()["total"] == 0
