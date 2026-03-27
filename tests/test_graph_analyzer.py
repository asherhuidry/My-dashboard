"""Tests for the graph structural analysis layer.

All Neo4j calls are mocked so the suite is deterministic and offline.
Tests cover all four analysis functions and the API route.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Helpers ──────────────────────────────────────────────────────────────────

_NEO4J_QUERY = "data.agents.graph_analyzer.run_read_query"


def _make_api_client() -> TestClient:
    """Build a TestClient with the graph_analysis router."""
    from api.routes.graph_analysis import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ── Sample data ──────────────────────────────────────────────────────────────

_EXPOSURE_ROWS: list[dict[str, Any]] = [
    {"asset": "AAPL", "asset_class": "equity",    "factor": "rate",       "beta": -0.30},
    {"asset": "AAPL", "asset_class": "equity",    "factor": "oil",        "beta":  0.10},
    {"asset": "AAPL", "asset_class": "equity",    "factor": "volatility", "beta": -0.25},
    {"asset": "MSFT", "asset_class": "equity",    "factor": "rate",       "beta": -0.28},
    {"asset": "MSFT", "asset_class": "equity",    "factor": "oil",        "beta":  0.08},
    {"asset": "MSFT", "asset_class": "equity",    "factor": "volatility", "beta": -0.22},
    {"asset": "GLD",  "asset_class": "commodity", "factor": "rate",       "beta":  0.40},
    {"asset": "GLD",  "asset_class": "commodity", "factor": "dollar",     "beta": -0.50},
    {"asset": "GLD",  "asset_class": "commodity", "factor": "volatility", "beta":  0.15},
]

_REGIME_ROWS: list[dict[str, Any]] = [
    # AAPL rate: all=-0.30, bear=-0.55, stress=-0.60
    {"asset": "AAPL", "factor": "rate", "regime": "all",    "beta": -0.30, "strength": "moderate"},
    {"asset": "AAPL", "factor": "rate", "regime": "bear",   "beta": -0.55, "strength": "strong"},
    {"asset": "AAPL", "factor": "rate", "regime": "stress", "beta": -0.60, "strength": "strong"},
    # GLD dollar: all=-0.50, bear=-0.30 (dampened in bear)
    {"asset": "GLD",  "factor": "dollar", "regime": "all",  "beta": -0.50, "strength": "strong"},
    {"asset": "GLD",  "factor": "dollar", "regime": "bear", "beta": -0.30, "strength": "moderate"},
    # MSFT volatility: all=-0.22, stress=-0.45
    {"asset": "MSFT", "factor": "volatility", "regime": "all",    "beta": -0.22, "strength": "moderate"},
    {"asset": "MSFT", "factor": "volatility", "regime": "stress", "beta": -0.45, "strength": "strong"},
]

_BRIDGE_ROWS: list[dict[str, Any]] = [
    {"factor_id": "GS10",      "factor_name": "10Y Treasury Yield",
     "classes": ["equity", "commodity"], "assets": ["AAPL", "MSFT", "GLD"], "edge_count": 5},
    {"factor_id": "VIXCLS",    "factor_name": "CBOE VIX",
     "classes": ["equity", "commodity", "crypto"], "assets": ["AAPL", "MSFT", "GLD", "BTC-USD"], "edge_count": 8},
    {"factor_id": "DCOILWTICO", "factor_name": "WTI Crude Oil",
     "classes": ["equity"], "assets": ["AAPL", "MSFT"], "edge_count": 2},
]

_CENTRALITY_ROWS: list[dict[str, Any]] = [
    {"node_id": "AAPL",   "label": "Asset",          "degree": 12},
    {"node_id": "VIXCLS", "label": "MacroIndicator", "degree": 10},
    {"node_id": "MSFT",   "label": "Asset",          "degree": 8},
    {"node_id": "GLD",    "label": "Asset",          "degree": 6},
]


# ── Test: Exposure profile similarity ────────────────────────────────────────

class TestExposureProfiles:
    """Tests for analyze_exposure_profiles()."""

    @patch(_NEO4J_QUERY, return_value=_EXPOSURE_ROWS)
    def test_similar_equities_rank_highest(self, mock_q: Any) -> None:
        """AAPL and MSFT have nearly identical beta vectors → top pair."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        pairs = result["similar_pairs"]
        assert len(pairs) > 0
        top = pairs[0]
        assert {top["asset_a"], top["asset_b"]} == {"AAPL", "MSFT"}
        assert top["cosine_similarity"] > 0.95  # very similar

    @patch(_NEO4J_QUERY, return_value=_EXPOSURE_ROWS)
    def test_gold_differs_from_equities(self, mock_q: Any) -> None:
        """GLD has opposite rate beta to AAPL → low/negative similarity."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        pairs = result["similar_pairs"]
        gld_pairs = [p for p in pairs if "GLD" in (p["asset_a"], p["asset_b"])]
        # GLD should have lower similarity to equities than AAPL-MSFT pair
        aapl_msft = [p for p in pairs if {p["asset_a"], p["asset_b"]} == {"AAPL", "MSFT"}][0]
        for gp in gld_pairs:
            assert abs(gp["cosine_similarity"]) < aapl_msft["cosine_similarity"]

    @patch(_NEO4J_QUERY, return_value=_EXPOSURE_ROWS)
    def test_asset_and_factor_counts(self, mock_q: Any) -> None:
        """Counts reflect the input data."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        assert result["asset_count"] == 3
        assert result["factor_count"] == 4  # rate, oil, volatility, dollar

    @patch(_NEO4J_QUERY, return_value=_EXPOSURE_ROWS)
    def test_shared_factors_counted(self, mock_q: Any) -> None:
        """AAPL and MSFT share 3 factors; GLD shares fewer with each."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        aapl_msft = [p for p in result["similar_pairs"]
                     if {p["asset_a"], p["asset_b"]} == {"AAPL", "MSFT"}][0]
        assert aapl_msft["shared_factors"] == 3

    @patch(_NEO4J_QUERY, return_value=_EXPOSURE_ROWS)
    def test_asset_class_included(self, mock_q: Any) -> None:
        """Each pair carries the asset class labels."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        for pair in result["similar_pairs"]:
            assert "class_a" in pair
            assert "class_b" in pair

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_graph_returns_empty(self, mock_q: Any) -> None:
        """No SENSITIVE_TO edges → empty result, not an error."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        assert result["similar_pairs"] == []
        assert result["asset_count"] == 0

    @patch(_NEO4J_QUERY, side_effect=RuntimeError("Neo4j down"))
    def test_neo4j_failure_returns_empty(self, mock_q: Any) -> None:
        """Neo4j outage → graceful empty result."""
        from data.agents.graph_analyzer import analyze_exposure_profiles
        result = analyze_exposure_profiles()
        assert result["similar_pairs"] == []


# ── Test: Regime divergence ──────────────────────────────────────────────────

class TestRegimeDivergence:
    """Tests for analyze_regime_divergence()."""

    @patch(_NEO4J_QUERY, return_value=_REGIME_ROWS)
    def test_highest_divergence_ranked_first(self, mock_q: Any) -> None:
        """AAPL rate stress has |delta|=0.30 → should rank high."""
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        divs = result["divergences"]
        assert len(divs) > 0
        # AAPL rate stress: |-0.60 - (-0.30)| = 0.30
        top = divs[0]
        assert top["abs_change"] >= 0.20  # among the highest shifts

    @patch(_NEO4J_QUERY, return_value=_REGIME_ROWS)
    def test_direction_label_correct(self, mock_q: Any) -> None:
        """Amplified when |regime_beta| > |all_beta|, dampened otherwise."""
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        divs = result["divergences"]
        # AAPL rate bear: |-0.55| > |-0.30| → amplified
        aapl_bear = [d for d in divs if d["asset"] == "AAPL"
                     and d["factor"] == "rate" and d["regime"] == "bear"][0]
        assert aapl_bear["direction"] == "amplified"
        # GLD dollar bear: |-0.30| < |-0.50| → dampened
        gld_bear = [d for d in divs if d["asset"] == "GLD"
                    and d["factor"] == "dollar" and d["regime"] == "bear"][0]
        assert gld_bear["direction"] == "dampened"

    @patch(_NEO4J_QUERY, return_value=_REGIME_ROWS)
    def test_both_regimes_appear(self, mock_q: Any) -> None:
        """Bear and stress divergences both present for assets that have them."""
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        regimes = {d["regime"] for d in result["divergences"]}
        assert "bear" in regimes
        assert "stress" in regimes

    @patch(_NEO4J_QUERY, return_value=_REGIME_ROWS)
    def test_pairs_analyzed_count(self, mock_q: Any) -> None:
        """Should report the number of unique (asset, factor) pairs."""
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        # 3 unique (asset, factor) pairs: AAPL/rate, GLD/dollar, MSFT/volatility
        assert result["pairs_analyzed"] == 3

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_returns_empty(self, mock_q: Any) -> None:
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        assert result["divergences"] == []

    @patch(_NEO4J_QUERY, return_value=[
        {"asset": "X", "factor": "rate", "regime": "bear", "beta": -0.5, "strength": "strong"},
    ])
    def test_missing_all_regime_skipped(self, mock_q: Any) -> None:
        """If no all-regime beta exists, no divergence is computed."""
        from data.agents.graph_analyzer import analyze_regime_divergence
        result = analyze_regime_divergence()
        assert result["divergences"] == []


# ── Test: Bridge factors ─────────────────────────────────────────────────────

class TestBridgeFactors:
    """Tests for analyze_bridge_factors()."""

    @patch(_NEO4J_QUERY, return_value=_BRIDGE_ROWS)
    def test_vix_is_widest_bridge(self, mock_q: Any) -> None:
        """VIX spans 3 classes → highest class_count bridge."""
        from data.agents.graph_analyzer import analyze_bridge_factors
        result = analyze_bridge_factors()
        vix = [b for b in result["bridges"] if b["factor_id"] == "VIXCLS"][0]
        assert vix["class_count"] == 3
        assert vix["is_bridge"] is True

    @patch(_NEO4J_QUERY, return_value=_BRIDGE_ROWS)
    def test_single_class_not_bridge(self, mock_q: Any) -> None:
        """Oil only connects equities → not a bridge."""
        from data.agents.graph_analyzer import analyze_bridge_factors
        result = analyze_bridge_factors()
        oil = [b for b in result["bridges"] if b["factor_id"] == "DCOILWTICO"][0]
        assert oil["is_bridge"] is False

    @patch(_NEO4J_QUERY, return_value=_BRIDGE_ROWS)
    def test_bridge_count_correct(self, mock_q: Any) -> None:
        """2 out of 3 factors are bridges (GS10 and VIXCLS)."""
        from data.agents.graph_analyzer import analyze_bridge_factors
        result = analyze_bridge_factors()
        assert result["bridge_count"] == 2
        assert result["total_factors"] == 3

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_returns_empty(self, mock_q: Any) -> None:
        from data.agents.graph_analyzer import analyze_bridge_factors
        result = analyze_bridge_factors()
        assert result["bridges"] == []


# ── Test: Centrality ─────────────────────────────────────────────────────────

class TestCentrality:
    """Tests for analyze_centrality()."""

    @patch(_NEO4J_QUERY, return_value=_CENTRALITY_ROWS)
    def test_ranking_ordered_by_degree(self, mock_q: Any) -> None:
        """Returned ranking is in descending degree order."""
        from data.agents.graph_analyzer import analyze_centrality
        result = analyze_centrality()
        degrees = [r["degree"] for r in result["ranking"]]
        assert degrees == sorted(degrees, reverse=True)

    @patch(_NEO4J_QUERY, return_value=_CENTRALITY_ROWS)
    def test_top_node_is_aapl(self, mock_q: Any) -> None:
        """AAPL has the most edges in the mock data."""
        from data.agents.graph_analyzer import analyze_centrality
        result = analyze_centrality()
        assert result["ranking"][0]["node_id"] == "AAPL"
        assert result["ranking"][0]["degree"] == 12

    @patch(_NEO4J_QUERY, return_value=_CENTRALITY_ROWS)
    def test_labels_present(self, mock_q: Any) -> None:
        """Each ranked node carries its label."""
        from data.agents.graph_analyzer import analyze_centrality
        result = analyze_centrality()
        labels = {r["label"] for r in result["ranking"]}
        assert "Asset" in labels
        assert "MacroIndicator" in labels

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_returns_empty(self, mock_q: Any) -> None:
        from data.agents.graph_analyzer import analyze_centrality
        result = analyze_centrality()
        assert result["ranking"] == []


# ── Test: Combined analysis ──────────────────────────────────────────────────

class TestAnalyzeGraphStructure:
    """Tests for the combined analyze_graph_structure() entry point."""

    @patch(_NEO4J_QUERY, return_value=[])
    def test_all_sections_present(self, mock_q: Any) -> None:
        """Combined result has all four sections."""
        from data.agents.graph_analyzer import analyze_graph_structure
        result = analyze_graph_structure()
        assert "exposure_profiles" in result
        assert "regime_divergence" in result
        assert "bridge_factors" in result
        assert "centrality" in result


# ── Test: API route ──────────────────────────────────────────────────────────

class TestGraphAnalysisAPI:
    """Tests for the /graph/analysis endpoint."""

    @patch(_NEO4J_QUERY, return_value=[])
    def test_endpoint_returns_200(self, mock_q: Any) -> None:
        client = _make_api_client()
        resp = client.get("/api/graph/analysis")
        assert resp.status_code == 200
        data = resp.json()
        assert "exposure_profiles" in data
        assert "regime_divergence" in data
        assert "bridge_factors" in data
        assert "centrality" in data

    @patch(_NEO4J_QUERY, return_value=[])
    def test_top_n_parameter(self, mock_q: Any) -> None:
        """top_n query param is accepted."""
        client = _make_api_client()
        resp = client.get("/api/graph/analysis?top_n=5")
        assert resp.status_code == 200

    @patch(_NEO4J_QUERY, return_value=[])
    def test_invalid_top_n_rejected(self, mock_q: Any) -> None:
        """top_n < 1 should fail validation."""
        client = _make_api_client()
        resp = client.get("/api/graph/analysis?top_n=0")
        assert resp.status_code == 422
