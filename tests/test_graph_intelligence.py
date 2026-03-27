"""Tests for the graph intelligence integration layer.

Covers: source provenance, edge-level diffs, insight ranking,
combined report assembly, and the API endpoint.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_api_client() -> TestClient:
    from api.routes.graph_analysis import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── Sample analysis outputs for testing ──────────────────────────────────────

_ANALYSIS_V1: dict[str, Any] = {
    "exposure_profiles": {
        "similar_pairs": [
            {"asset_a": "AAPL", "asset_b": "MSFT", "cosine_similarity": 0.96,
             "shared_factors": 3, "class_a": "equity", "class_b": "equity"},
            {"asset_a": "GLD", "asset_b": "AAPL", "cosine_similarity": -0.40,
             "shared_factors": 2, "class_a": "commodity", "class_b": "equity"},
        ],
        "asset_count": 3, "factor_count": 4,
    },
    "regime_divergence": {
        "divergences": [
            {"asset": "AAPL", "factor": "rate", "regime": "stress",
             "beta_all": -0.30, "beta_regime": -0.60, "abs_change": 0.30,
             "rel_change": 1.0, "direction": "amplified"},
        ],
        "pairs_analyzed": 2,
    },
    "bridge_factors": {
        "bridges": [
            {"factor_id": "VIXCLS", "factor_name": "VIX",
             "asset_classes": ["equity", "commodity"], "class_count": 2,
             "assets": ["AAPL", "GLD"], "asset_count": 2,
             "edge_count": 4, "is_bridge": True},
        ],
        "total_factors": 1, "bridge_count": 1,
    },
    "centrality": {
        "ranking": [
            {"node_id": "AAPL", "label": "Asset", "degree": 12},
            {"node_id": "VIXCLS", "label": "MacroIndicator", "degree": 8},
        ],
        "total_ranked": 2,
    },
}

_ANALYSIS_V2: dict[str, Any] = {
    "exposure_profiles": {
        "similar_pairs": [
            {"asset_a": "AAPL", "asset_b": "MSFT", "cosine_similarity": 0.88,
             "shared_factors": 3, "class_a": "equity", "class_b": "equity"},
            {"asset_a": "NVDA", "asset_b": "MSFT", "cosine_similarity": 0.82,
             "shared_factors": 2, "class_a": "equity", "class_b": "equity"},
        ],
        "asset_count": 4, "factor_count": 4,
    },
    "regime_divergence": {
        "divergences": [
            {"asset": "AAPL", "factor": "rate", "regime": "stress",
             "beta_all": -0.30, "beta_regime": -0.65, "abs_change": 0.35,
             "rel_change": 1.17, "direction": "amplified"},
            {"asset": "NVDA", "factor": "oil", "regime": "bear",
             "beta_all": 0.10, "beta_regime": -0.15, "abs_change": 0.25,
             "rel_change": 2.5, "direction": "dampened"},
        ],
        "pairs_analyzed": 3,
    },
    "bridge_factors": {
        "bridges": [
            {"factor_id": "VIXCLS", "factor_name": "VIX",
             "asset_classes": ["equity", "commodity", "crypto"], "class_count": 3,
             "assets": ["AAPL", "GLD", "BTC-USD"], "asset_count": 3,
             "edge_count": 6, "is_bridge": True},
            {"factor_id": "GS10", "factor_name": "10Y Yield",
             "asset_classes": ["equity", "commodity"], "class_count": 2,
             "assets": ["AAPL", "GLD"], "asset_count": 2,
             "edge_count": 3, "is_bridge": True},
        ],
        "total_factors": 2, "bridge_count": 2,
    },
    "centrality": {
        "ranking": [
            {"node_id": "VIXCLS", "label": "MacroIndicator", "degree": 14},
            {"node_id": "AAPL", "label": "Asset", "degree": 12},
        ],
        "total_ranked": 2,
    },
}


# ── Test: Source provenance ──────────────────────────────────────────────────

class TestResolveSeriesSource:
    """Tests for resolve_series_source()."""

    def test_equity_resolves_to_yfinance(self) -> None:
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("AAPL")
        assert result["source_id"] == "yfinance"
        assert "Yahoo" in result["source_name"]

    def test_macro_resolves_to_fred(self) -> None:
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("VIXCLS")
        assert result["source_id"] == "fred_api"
        assert "FRED" in result["source_name"]

    def test_etf_resolves_to_yfinance(self) -> None:
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("SPY")
        assert result["source_id"] == "yfinance"

    def test_crypto_resolves_to_yfinance(self) -> None:
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("BTC-USD")
        assert result["source_id"] == "yfinance"

    def test_unknown_series_returns_unknown(self) -> None:
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("ZZZZZZZ")
        assert result["source_id"] in ("unknown", "yfinance")

    def test_heuristic_fallback_for_dash(self) -> None:
        """Series with special chars inferred as yfinance."""
        from data.agents.graph_intelligence import resolve_series_source
        result = resolve_series_source("SOME-THING")
        assert result["source_id"] == "yfinance"


# ── Test: Edge-level changes ────────────────────────────────────────────────

class TestComputeEdgeChanges:
    """Tests for compute_edge_changes()."""

    def test_new_exposure_pair_detected(self) -> None:
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        new = changes["exposure_profiles"]["new"]
        new_keys = {(p["asset_a"], p["asset_b"]) for p in new}
        assert ("NVDA", "MSFT") in new_keys

    def test_lost_exposure_pair_detected(self) -> None:
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        lost = changes["exposure_profiles"]["lost"]
        lost_keys = {(p["asset_a"], p["asset_b"]) for p in lost}
        assert ("GLD", "AAPL") in lost_keys

    def test_shifted_exposure_pair_detected(self) -> None:
        """AAPL/MSFT went from 0.96 to 0.88 — should flag as weakened."""
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        shifted = changes["exposure_profiles"]["shifted"]
        assert len(shifted) >= 1
        aapl_msft = [s for s in shifted if s["asset_a"] == "AAPL" and s["asset_b"] == "MSFT"][0]
        assert aapl_msft["direction"] == "weakened"
        assert aapl_msft["delta"] == pytest.approx(-0.08, abs=0.001)

    def test_new_regime_divergence_detected(self) -> None:
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        new_divs = changes["regime_divergence"]["new"]
        assert any(d["asset"] == "NVDA" for d in new_divs)

    def test_shifted_regime_divergence_detected(self) -> None:
        """AAPL rate stress went from abs_change=0.30 to 0.35."""
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        shifted = changes["regime_divergence"]["shifted"]
        aapl = [s for s in shifted if s["asset"] == "AAPL"][0]
        assert aapl["trend"] == "amplifying"

    def test_bridge_gained_detected(self) -> None:
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        gained = changes["bridge_factors"]["gained"]
        assert any(b["factor_id"] == "GS10" for b in gained)

    def test_bridge_expanded_detected(self) -> None:
        """VIXCLS went from 2 classes to 3."""
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        expanded = changes["bridge_factors"]["expanded"]
        assert any(b["factor_id"] == "VIXCLS" for b in expanded)

    def test_centrality_mover_detected(self) -> None:
        """VIXCLS moved from rank 2 to rank 1, AAPL from 1 to 2."""
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        movers = changes["centrality"]["movers"]
        vix = [m for m in movers if m["node_id"] == "VIXCLS"][0]
        assert vix["direction"] == "up"
        aapl = [m for m in movers if m["node_id"] == "AAPL"][0]
        assert aapl["direction"] == "down"

    def test_no_changes_produces_empty_lists(self) -> None:
        from data.agents.graph_intelligence import compute_edge_changes
        changes = compute_edge_changes(_ANALYSIS_V1, _ANALYSIS_V1)
        assert changes["exposure_profiles"]["new"] == []
        assert changes["exposure_profiles"]["lost"] == []
        assert changes["regime_divergence"]["new"] == []
        assert changes["bridge_factors"]["gained"] == []


# ── Test: Insight ranking ────────────────────────────────────────────────────

class TestRankInsights:
    """Tests for rank_insights()."""

    def test_returns_sorted_by_priority(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import rank_insights
        metrics = compute_summary_metrics(_ANALYSIS_V2)
        insights = rank_insights(_ANALYSIS_V2, metrics)
        priorities = [i["priority"] for i in insights]
        assert priorities == sorted(priorities)

    def test_crisis_exposure_insight_generated(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import rank_insights
        metrics = compute_summary_metrics(_ANALYSIS_V2)
        insights = rank_insights(_ANALYSIS_V2, metrics)
        crisis = [i for i in insights if i["type"] == "crisis_exposure"]
        assert len(crisis) >= 1

    def test_systemic_bridge_insight_for_3plus_classes(self) -> None:
        """VIX spans 3 classes in V2 → should generate systemic_bridge."""
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import rank_insights
        metrics = compute_summary_metrics(_ANALYSIS_V2)
        insights = rank_insights(_ANALYSIS_V2, metrics)
        systemic = [i for i in insights if i["type"] == "systemic_bridge"]
        assert len(systemic) >= 1
        assert systemic[0]["priority"] == 1

    def test_edge_change_insights_added(self) -> None:
        """When edge_changes is provided, new regime insights appear."""
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import rank_insights, compute_edge_changes
        metrics = compute_summary_metrics(_ANALYSIS_V2)
        edge_changes = compute_edge_changes(_ANALYSIS_V2, _ANALYSIS_V1)
        insights = rank_insights(_ANALYSIS_V2, metrics, edge_changes)
        new_types = {i["type"] for i in insights}
        assert "new_regime_exposure" in new_types or "new_bridge" in new_types

    def test_empty_analysis_produces_no_insights(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        empty = {
            "exposure_profiles": {"similar_pairs": [], "asset_count": 0, "factor_count": 0},
            "regime_divergence": {"divergences": [], "pairs_analyzed": 0},
            "bridge_factors": {"bridges": [], "total_factors": 0, "bridge_count": 0},
            "centrality": {"ranking": [], "total_ranked": 0},
        }
        insights = rank_insights(empty, {})
        assert insights == []


# ── Test: Combined report ────────────────────────────────────────────────────

class TestBuildIntelligenceReport:
    """Tests for build_intelligence_report()."""

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_report_contains_all_sections(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_intelligence import build_intelligence_report
        report = build_intelligence_report(run_id="run-1")
        assert "insights" in report
        assert "metrics" in report
        assert "provenance" in report
        assert "analysis" in report
        assert report["run_id"] == "run-1"

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_first_run_has_no_diff(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_intelligence import build_intelligence_report
        report = build_intelligence_report()
        assert "metric_diff" not in report
        assert "edge_changes" not in report

    @patch("db.supabase.client.get_structural_snapshots")
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V2)
    def test_report_includes_diff_when_previous_exists(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import build_intelligence_report
        prev_metrics = compute_summary_metrics(_ANALYSIS_V1)
        mock_prev.return_value = [{
            "after_state": {"metrics": prev_metrics, "analysis": _ANALYSIS_V1},
            "created_at": "2026-03-20T08:00:00Z",
        }]
        report = build_intelligence_report()
        assert "metric_diff" in report
        assert "edge_changes" in report
        assert report["previous_timestamp"] == "2026-03-20T08:00:00Z"

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_provenance_includes_known_series(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_intelligence import build_intelligence_report
        report = build_intelligence_report()
        prov = report["provenance"]
        assert "AAPL" in prov
        assert prov["AAPL"]["source_id"] == "yfinance"
        assert "VIXCLS" in prov
        assert prov["VIXCLS"]["source_id"] == "fred_api"

    @patch("db.supabase.client.get_structural_snapshots",
           side_effect=RuntimeError("Supabase down"))
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_supabase_failure_still_returns_report(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_intelligence import build_intelligence_report
        report = build_intelligence_report()
        assert "insights" in report
        assert "provenance" in report


# ── Test: API endpoint ───────────────────────────────────────────────────────

class TestGraphIntelligenceAPI:
    """Tests for the /graph/intelligence endpoint."""

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_endpoint_returns_200(self, mock_analyze, mock_prev) -> None:
        client = _make_api_client()
        resp = client.get("/api/graph/intelligence")
        assert resp.status_code == 200
        data = resp.json()
        assert "insights" in data
        assert "provenance" in data

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_run_id_passed_through(self, mock_analyze, mock_prev) -> None:
        client = _make_api_client()
        resp = client.get("/api/graph/intelligence?run_id=test-run-42")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "test-run-42"
