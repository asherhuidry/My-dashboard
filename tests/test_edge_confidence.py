"""Tests for the edge confidence scoring engine.

Covers: per-edge-type scorers, auto-detect dispatch, boundary
conditions, and score monotonicity (stronger evidence → higher score).
"""
from __future__ import annotations

import pytest

from data.agents.edge_confidence import (
    score_correlation_edge,
    score_sensitivity_edge,
    score_causal_edge,
    score_edge,
)


# ── Correlation edge scoring ─────────────────────────────────────────────────

class TestScoreCorrelationEdge:
    """Tests for CORRELATED_WITH edge scoring."""

    def test_returns_float_in_unit_range(self) -> None:
        score = score_correlation_edge(pearson_r=0.5)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_strong_evidence_scores_high(self) -> None:
        """Strong r + significant granger + high MI + strong strength."""
        score = score_correlation_edge(
            pearson_r=0.85,
            granger_p=0.001,
            mutual_info=0.35,
            strength="strong",
        )
        assert score >= 0.85

    def test_weak_evidence_scores_low(self) -> None:
        """Weak r, no granger, no MI, weak strength."""
        score = score_correlation_edge(
            pearson_r=0.10,
            granger_p=None,
            mutual_info=None,
            strength="weak",
        )
        assert score < 0.35

    def test_monotonic_in_pearson_r(self) -> None:
        """Higher |r| → higher score, all else equal."""
        low = score_correlation_edge(pearson_r=0.2, strength="moderate")
        mid = score_correlation_edge(pearson_r=0.5, strength="moderate")
        high = score_correlation_edge(pearson_r=0.8, strength="moderate")
        assert low < mid < high

    def test_monotonic_in_granger_p(self) -> None:
        """Lower p → higher score (more significant)."""
        insignificant = score_correlation_edge(pearson_r=0.5, granger_p=0.09)
        significant = score_correlation_edge(pearson_r=0.5, granger_p=0.01)
        very_sig = score_correlation_edge(pearson_r=0.5, granger_p=0.001)
        assert insignificant < significant <= very_sig

    def test_mutual_info_bonus(self) -> None:
        """Non-zero MI adds to score."""
        without = score_correlation_edge(pearson_r=0.5, mutual_info=0.0)
        with_mi = score_correlation_edge(pearson_r=0.5, mutual_info=0.2)
        assert with_mi > without

    def test_strength_matters(self) -> None:
        weak = score_correlation_edge(pearson_r=0.5, strength="weak")
        strong = score_correlation_edge(pearson_r=0.5, strength="strong")
        assert strong > weak

    def test_all_none_returns_positive(self) -> None:
        """Even with no evidence, returns a small positive score."""
        score = score_correlation_edge()
        assert 0.0 < score < 0.5

    def test_clamped_to_1(self) -> None:
        """Even with perfect evidence, does not exceed 1.0."""
        score = score_correlation_edge(
            pearson_r=2.0, granger_p=0.0, mutual_info=1.0, strength="strong",
        )
        assert score == 1.0


# ── Sensitivity edge scoring ─────────────────────────────────────────────────

class TestScoreSensitivityEdge:
    """Tests for SENSITIVE_TO edge scoring."""

    def test_returns_float_in_unit_range(self) -> None:
        score = score_sensitivity_edge(beta=0.3)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_strong_evidence_scores_high(self) -> None:
        score = score_sensitivity_edge(
            beta=-0.8, t_stat=5.5, r_squared=0.4, strength="strong",
        )
        assert score >= 0.80

    def test_weak_evidence_scores_low(self) -> None:
        score = score_sensitivity_edge(
            beta=0.05, t_stat=1.2, r_squared=0.01, strength="weak",
        )
        assert score < 0.35

    def test_monotonic_in_t_stat(self) -> None:
        low = score_sensitivity_edge(beta=0.3, t_stat=1.5)
        high = score_sensitivity_edge(beta=0.3, t_stat=4.5)
        assert low < high

    def test_p_value_fallback(self) -> None:
        """When t_stat is None, uses p_value instead."""
        from_p = score_sensitivity_edge(beta=0.3, p_value=0.005, strength="moderate")
        from_t = score_sensitivity_edge(beta=0.3, t_stat=3.5, strength="moderate")
        # Both should be in the high range
        assert from_p > 0.4
        assert from_t > 0.4

    def test_r_squared_matters(self) -> None:
        low_fit = score_sensitivity_edge(beta=0.5, t_stat=3.0, r_squared=0.02)
        high_fit = score_sensitivity_edge(beta=0.5, t_stat=3.0, r_squared=0.5)
        assert high_fit > low_fit

    def test_all_none_returns_positive(self) -> None:
        score = score_sensitivity_edge()
        assert 0.0 < score < 0.5


# ── Causal edge scoring ──────────────────────────────────────────────────────

class TestScoreCausalEdge:
    """Tests for CAUSES edge scoring."""

    def test_returns_float_in_unit_range(self) -> None:
        score = score_causal_edge(pearson_r=0.4, granger_p=0.02)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_granger_dominates(self) -> None:
        """Granger significance is weighted 45% for causal edges."""
        no_granger = score_causal_edge(pearson_r=0.7, granger_p=0.09)
        with_granger = score_causal_edge(pearson_r=0.7, granger_p=0.001)
        assert with_granger > no_granger
        # The difference should be large (45% weight on Granger)
        assert with_granger - no_granger > 0.15

    def test_strong_causal_evidence(self) -> None:
        score = score_causal_edge(
            pearson_r=0.7, granger_p=0.001, mutual_info=0.25, strength="strong",
        )
        assert score >= 0.80

    def test_weak_causal_evidence(self) -> None:
        score = score_causal_edge(
            pearson_r=0.2, granger_p=None, strength="weak",
        )
        assert score < 0.35


# ── Auto-detect dispatch ─────────────────────────────────────────────────────

class TestScoreEdge:
    """Tests for the auto-detect score_edge() dispatcher."""

    def test_dispatches_to_correlation(self) -> None:
        props = {"rel_type": "CORRELATED_WITH", "pearson_r": 0.6, "strength": "moderate"}
        score = score_edge(props)
        expected = score_correlation_edge(pearson_r=0.6, strength="moderate")
        assert score == expected

    def test_dispatches_to_sensitivity(self) -> None:
        props = {"rel_type": "SENSITIVE_TO", "beta": -0.5, "strength": "strong"}
        score = score_edge(props)
        expected = score_sensitivity_edge(beta=-0.5, strength="strong")
        assert score == expected

    def test_dispatches_to_causal(self) -> None:
        props = {"rel_type": "CAUSES", "pearson_r": 0.5, "granger_p": 0.02}
        score = score_edge(props)
        expected = score_causal_edge(pearson_r=0.5, granger_p=0.02)
        assert score == expected

    def test_unknown_rel_type_uses_correlation(self) -> None:
        props = {"rel_type": "UNKNOWN_TYPE", "pearson_r": 0.4}
        score = score_edge(props)
        expected = score_correlation_edge(pearson_r=0.4)
        assert score == expected

    def test_missing_rel_type_uses_correlation(self) -> None:
        props = {"pearson_r": 0.4}
        score = score_edge(props)
        assert score > 0

    def test_empty_props(self) -> None:
        """Empty dict still returns a valid score."""
        score = score_edge({})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_edge_types_produce_different_scores(self) -> None:
        """Same statistical evidence produces different scores per type."""
        base = {"pearson_r": 0.6, "granger_p": 0.02, "strength": "moderate"}
        corr = score_edge({**base, "rel_type": "CORRELATED_WITH"})
        cause = score_edge({**base, "rel_type": "CAUSES"})
        # Different weights on granger_p → different scores
        assert corr != cause


# ── Materializer integration ─────────────────────────────────────────────────

class TestMaterializerEdgeBuilding:
    """Test that materializer edge builders produce confidence + provenance."""

    def test_correlation_edge_has_confidence(self) -> None:
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL", "series_b": "MSFT",
            "pearson_r": 0.65, "lag_days": 0, "granger_p": 0.03,
            "mutual_info": 0.1, "strength": "moderate",
            "regime": "all", "relationship_type": "discovered",
            "id": "disc-123", "run_id": "run-abc", "computed_at": "2026-01-01T00:00:00Z",
        }
        asset_lookup = {"AAPL": "equity", "MSFT": "equity"}
        macro_lookup = {}

        edges = _build_edge(row, asset_lookup, macro_lookup)
        assert len(edges) >= 1

        corr_edge = edges[0]
        assert "confidence" in corr_edge
        assert 0.0 < corr_edge["confidence"] <= 1.0
        assert corr_edge["discovery_id"] == "disc-123"
        assert corr_edge["run_id"] == "run-abc"
        assert corr_edge["discovered_at"] == "2026-01-01T00:00:00Z"

    def test_causal_edge_has_confidence(self) -> None:
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL", "series_b": "MSFT",
            "pearson_r": 0.65, "lag_days": 5, "granger_p": 0.01,
            "mutual_info": 0.15, "strength": "strong",
            "regime": "all", "relationship_type": "discovered",
            "id": "disc-456", "run_id": "run-def", "computed_at": "2026-01-02T00:00:00Z",
        }
        asset_lookup = {"AAPL": "equity", "MSFT": "equity"}
        macro_lookup = {}

        edges = _build_edge(row, asset_lookup, macro_lookup)
        # Should have both CORRELATED_WITH and CAUSES (granger_p < 0.05)
        assert len(edges) == 2
        causal = [e for e in edges if e["rel_type"] == "CAUSES"][0]
        assert "confidence" in causal
        assert 0.0 < causal["confidence"] <= 1.0
        assert causal["run_id"] == "run-def"

    def test_sensitivity_edge_has_confidence(self) -> None:
        from data.agents.graph_materializer import _build_sensitivity_edge

        row = {
            "series_a": "AAPL", "series_b": "VIXCLS",
            "pearson_r": -0.3, "lag_days": 0, "granger_p": None,
            "mutual_info": -0.45, "strength": "strong",
            "regime": "all", "relationship_type": "volatility_sensitive",
            "id": "disc-789", "run_id": "run-ghi", "computed_at": "2026-01-03T00:00:00Z",
        }
        asset_lookup = {"AAPL": "equity"}
        macro_lookup = {"VIXCLS": ("VIX", "daily")}

        edges = _build_sensitivity_edge(row, asset_lookup, macro_lookup)
        assert len(edges) == 1
        edge = edges[0]
        assert edge["rel_type"] == "SENSITIVE_TO"
        assert "confidence" in edge
        assert 0.0 < edge["confidence"] <= 1.0
        assert edge["discovery_id"] == "disc-789"
        assert edge["factor_group"] == "volatility"

    def test_missing_provenance_fields_are_none(self) -> None:
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL", "series_b": "MSFT",
            "pearson_r": 0.5, "lag_days": 0,
            "strength": "moderate", "regime": "all",
            "relationship_type": "discovered",
        }
        asset_lookup = {"AAPL": "equity", "MSFT": "equity"}
        edges = _build_edge(row, asset_lookup, {})

        edge = edges[0]
        assert edge["discovery_id"] is None
        assert edge["run_id"] is None
        assert edge["discovered_at"] is None
        # Confidence should still be computed
        assert "confidence" in edge
        assert edge["confidence"] > 0


# ── Intelligence integration ─────────────────────────────────────────────────

class TestConfidenceInsights:
    """Test that confidence stats produce insights in rank_insights."""

    def test_weak_evidence_base_insight(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        analysis = {"regime_divergence": {}, "bridge_factors": {}, "exposure_profiles": {}}
        metrics = {}
        conf_stats = {
            "scored_edges": 100,
            "mean_confidence": 0.35,
            "low_confidence": 60,
            "reconfirmed": 5,
            "mean_evidence_count": 1.1,
        }
        insights = rank_insights(analysis, metrics, confidence_stats=conf_stats)
        types = [i["type"] for i in insights]
        assert "weak_evidence_base" in types

    def test_many_weak_edges_insight(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        analysis = {"regime_divergence": {}, "bridge_factors": {}, "exposure_profiles": {}}
        metrics = {}
        conf_stats = {
            "scored_edges": 100,
            "mean_confidence": 0.55,
            "low_confidence": 40,
            "reconfirmed": 10,
            "mean_evidence_count": 1.2,
        }
        insights = rank_insights(analysis, metrics, confidence_stats=conf_stats)
        types = [i["type"] for i in insights]
        assert "many_weak_edges" in types

    def test_strong_reconfirmation_insight(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        analysis = {"regime_divergence": {}, "bridge_factors": {}, "exposure_profiles": {}}
        metrics = {}
        conf_stats = {
            "scored_edges": 100,
            "mean_confidence": 0.72,
            "low_confidence": 5,
            "reconfirmed": 65,
            "mean_evidence_count": 3.2,
        }
        insights = rank_insights(analysis, metrics, confidence_stats=conf_stats)
        types = [i["type"] for i in insights]
        assert "strong_reconfirmation" in types

    def test_no_confidence_stats_no_crash(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        analysis = {"regime_divergence": {}, "bridge_factors": {}, "exposure_profiles": {}}
        insights = rank_insights(analysis, {}, confidence_stats=None)
        # Should not crash; confidence insights just absent
        types = [i["type"] for i in insights]
        assert "weak_evidence_base" not in types

    def test_zero_edges_no_insights(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        analysis = {"regime_divergence": {}, "bridge_factors": {}, "exposure_profiles": {}}
        conf_stats = {
            "scored_edges": 0,
            "mean_confidence": 0.0,
            "low_confidence": 0,
            "reconfirmed": 0,
            "mean_evidence_count": 0,
        }
        insights = rank_insights(analysis, {}, confidence_stats=conf_stats)
        types = [i["type"] for i in insights]
        assert "weak_evidence_base" not in types
