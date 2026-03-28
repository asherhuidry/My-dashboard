"""Tests for graph-state reasoning summaries."""
from __future__ import annotations

import pytest

from data.agents.graph_intelligence import compute_graph_reasoning_summary


def _make_analysis(
    bridges: list | None = None,
    divergences: list | None = None,
    similar_pairs: list | None = None,
    centrality: list | None = None,
) -> dict:
    return {
        "bridge_factors": {"bridges": bridges or []},
        "regime_divergence": {"divergences": divergences or []},
        "exposure_profiles": {"similar_pairs": similar_pairs or []},
        "centrality": {"ranking": centrality or []},
    }


class TestMostInfluentialFactor:
    """Tests for most_influential_factor reasoning."""

    def test_picks_bridge_with_most_classes(self) -> None:
        analysis = _make_analysis(bridges=[
            {"factor_id": "FEDFUNDS", "is_bridge": True, "class_count": 4,
             "asset_count": 12, "asset_classes": ["equity", "crypto", "fx", "commodity"]},
            {"factor_id": "CPIAUCSL", "is_bridge": True, "class_count": 2,
             "asset_count": 8, "asset_classes": ["equity", "commodity"]},
        ])
        result = compute_graph_reasoning_summary(analysis, {})
        assert result["most_influential_factor"]["factor_id"] == "FEDFUNDS"
        assert "narrative" in result["most_influential_factor"]

    def test_no_bridges_no_factor(self) -> None:
        analysis = _make_analysis()
        result = compute_graph_reasoning_summary(analysis, {})
        assert "most_influential_factor" not in result

    def test_non_bridge_excluded(self) -> None:
        analysis = _make_analysis(bridges=[
            {"factor_id": "X", "is_bridge": False, "class_count": 1, "asset_count": 2},
        ])
        result = compute_graph_reasoning_summary(analysis, {})
        assert "most_influential_factor" not in result


class TestMostExposedAsset:
    """Tests for most_exposed_asset reasoning."""

    def test_picks_highest_total_divergence(self) -> None:
        analysis = _make_analysis(divergences=[
            {"asset": "AAPL", "factor": "DFF", "abs_change": 0.15, "direction": "amplifies"},
            {"asset": "AAPL", "factor": "VIX", "abs_change": 0.20, "direction": "amplifies"},
            {"asset": "MSFT", "factor": "DFF", "abs_change": 0.10, "direction": "dampens"},
        ])
        result = compute_graph_reasoning_summary(analysis, {})
        assert result["most_exposed_asset"]["asset"] == "AAPL"
        assert result["most_exposed_asset"]["divergence_count"] == 2

    def test_no_divergences_no_exposure(self) -> None:
        result = compute_graph_reasoning_summary(_make_analysis(), {})
        assert "most_exposed_asset" not in result


class TestStructuralHealth:
    """Tests for the structural health score."""

    def test_computes_score_with_confidence(self) -> None:
        analysis = _make_analysis(
            bridges=[{"factor_id": "X", "is_bridge": True, "class_count": 3, "asset_count": 10}],
            centrality=[{"node_id": f"N{i}"} for i in range(30)],
        )
        conf = {
            "scored_edges": 100,
            "mean_confidence": 0.65,
            "reconfirmed": 40,
            "stale_edges": 5,
            "expired_edges": 2,
        }
        result = compute_graph_reasoning_summary(analysis, {}, confidence_stats=conf)
        health = result["structural_health"]
        assert 0.0 <= health["score"] <= 1.0
        assert health["grade"] in ("excellent", "good", "fair", "poor")
        assert "narrative" in health
        assert "confidence" in health["components"]

    def test_no_confidence_still_works(self) -> None:
        analysis = _make_analysis(
            centrality=[{"node_id": f"N{i}"} for i in range(10)],
        )
        result = compute_graph_reasoning_summary(analysis, {})
        # May not have health if insufficient data, but shouldn't crash
        if "structural_health" in result:
            assert 0.0 <= result["structural_health"]["score"] <= 1.0


class TestDominantPattern:
    """Tests for dominant pattern detection."""

    def test_convergence_pattern(self) -> None:
        analysis = _make_analysis(similar_pairs=[
            {"asset_a": "AAPL", "asset_b": "MSFT", "cosine_similarity": 0.92, "shared_factors": 5},
            {"asset_a": "GOOGL", "asset_b": "META", "cosine_similarity": 0.88, "shared_factors": 4},
        ])
        result = compute_graph_reasoning_summary(analysis, {})
        assert result["dominant_pattern"]["type"] == "convergence"
        assert result["dominant_pattern"]["pair_count"] == 2

    def test_no_patterns_when_empty(self) -> None:
        result = compute_graph_reasoning_summary(_make_analysis(), {})
        assert "dominant_pattern" not in result
