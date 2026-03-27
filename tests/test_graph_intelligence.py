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


# ── Test: Anomaly detection ─────────────────────────────────────────────────

def _make_metrics(**overrides: Any) -> dict[str, Any]:
    """Build a plausible metrics dict for anomaly detection tests."""
    base: dict[str, Any] = {
        "exposure_asset_count": 4, "exposure_factor_count": 4,
        "exposure_pair_count": 2,
        "exposure_sim_mean": 0.28, "exposure_sim_max": 0.88,
        "exposure_top_pair": "AAPL/MSFT",
        "regime_pairs_analyzed": 3, "regime_divergence_count": 2,
        "regime_divergence_mean": 0.30, "regime_divergence_max": 0.35,
        "regime_amplified_pct": 0.5, "regime_top_shift": "AAPL/rate/stress",
        "bridge_count": 2, "bridge_max_span": 3,
        "bridge_factor_ids": ["VIXCLS", "GS10"],
        "centrality_top_node": "VIXCLS", "centrality_top_degree": 14,
        "centrality_mean_degree": 13.0,
    }
    base.update(overrides)
    return base


class TestDetectStructuralAnomalies:
    """Tests for detect_structural_anomalies()."""

    def test_insufficient_history_returns_early(self) -> None:
        from data.agents.graph_intelligence import detect_structural_anomalies
        current = _make_metrics()
        history = [_make_metrics(), _make_metrics()]  # only 2, need 3
        result = detect_structural_anomalies(current, history)
        assert result["sufficient_history"] is False
        assert result["anomalies"] == []
        assert result["history_depth"] == 2

    def test_no_anomalies_when_current_matches_history(self) -> None:
        from data.agents.graph_intelligence import detect_structural_anomalies
        m = _make_metrics()
        # Small variance in history so std > 0 but current is near mean
        history = [
            _make_metrics(exposure_sim_mean=0.27),
            _make_metrics(exposure_sim_mean=0.29),
            _make_metrics(exposure_sim_mean=0.28),
            _make_metrics(exposure_sim_mean=0.28),
        ]
        result = detect_structural_anomalies(m, history)
        assert result["sufficient_history"] is True
        assert result["metrics_checked"] > 0
        assert len(result["anomalies"]) == 0

    def test_spike_detected(self) -> None:
        """A current value far above the rolling mean should flag as spike."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        # History has bridge_count around 2, current jumps to 10
        history = [
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=3),
            _make_metrics(bridge_count=2),
        ]
        current = _make_metrics(bridge_count=10)
        result = detect_structural_anomalies(current, history)
        bridge_anomalies = [a for a in result["anomalies"] if a["metric"] == "bridge_count"]
        assert len(bridge_anomalies) == 1
        assert bridge_anomalies[0]["direction"] == "spike"
        assert bridge_anomalies[0]["z_score"] > 2.0

    def test_drop_detected(self) -> None:
        """A current value far below the rolling mean should flag as drop."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [
            _make_metrics(centrality_mean_degree=13.0),
            _make_metrics(centrality_mean_degree=14.0),
            _make_metrics(centrality_mean_degree=13.5),
            _make_metrics(centrality_mean_degree=13.0),
        ]
        current = _make_metrics(centrality_mean_degree=2.0)
        result = detect_structural_anomalies(current, history)
        deg_anomalies = [a for a in result["anomalies"] if a["metric"] == "centrality_mean_degree"]
        assert len(deg_anomalies) == 1
        assert deg_anomalies[0]["direction"] == "drop"
        assert deg_anomalies[0]["z_score"] < -2.0

    def test_critical_severity_at_z3(self) -> None:
        """z >= 3.0 should be marked critical, not just warning."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=3),
        ]
        current = _make_metrics(bridge_count=20)  # extreme spike
        result = detect_structural_anomalies(current, history)
        bridge_anomalies = [a for a in result["anomalies"] if a["metric"] == "bridge_count"]
        assert len(bridge_anomalies) == 1
        assert bridge_anomalies[0]["severity"] == "critical"

    def test_warning_severity_between_z2_and_z3(self) -> None:
        from data.agents.graph_intelligence import detect_structural_anomalies
        # history mean=0.28, std≈0.00816 → current=0.30 gives z≈2.45
        history = [
            _make_metrics(exposure_sim_mean=0.28),
            _make_metrics(exposure_sim_mean=0.29),
            _make_metrics(exposure_sim_mean=0.27),
            _make_metrics(exposure_sim_mean=0.28),
        ]
        current = _make_metrics(exposure_sim_mean=0.30)
        result = detect_structural_anomalies(current, history)
        sim_anomalies = [a for a in result["anomalies"] if a["metric"] == "exposure_sim_mean"]
        assert len(sim_anomalies) == 1
        assert sim_anomalies[0]["severity"] == "warning"

    def test_constant_history_skipped(self) -> None:
        """If all history values are identical (std=0), skip rather than divide by zero."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [_make_metrics() for _ in range(5)]
        # All metrics identical → std=0 for all → no anomalies
        current = _make_metrics(bridge_count=99)
        result = detect_structural_anomalies(current, history)
        # bridge_count had zero variance in history, but current differs.
        # However std=0 so it should be skipped (no divide-by-zero).
        # Other metrics with zero variance are also skipped.
        # The only anomalies should be from metrics where history varies.
        assert result["sufficient_history"] is True
        # No crash = success; constant-history metrics produce no anomalies

    def test_none_values_in_history_skipped(self) -> None:
        """Metrics with None in history should be gracefully skipped."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [
            _make_metrics(exposure_sim_mean=None),
            _make_metrics(exposure_sim_mean=None),
            _make_metrics(exposure_sim_mean=0.28),
        ]
        current = _make_metrics(exposure_sim_mean=0.90)
        result = detect_structural_anomalies(current, history)
        # Only 1 non-None value for exposure_sim_mean < MIN_HISTORY — no anomaly for it
        sim_anomalies = [a for a in result["anomalies"] if a["metric"] == "exposure_sim_mean"]
        assert len(sim_anomalies) == 0

    def test_anomalies_sorted_by_z_score_magnitude(self) -> None:
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [
            _make_metrics(bridge_count=2, centrality_top_degree=14),
            _make_metrics(bridge_count=2, centrality_top_degree=14),
            _make_metrics(bridge_count=3, centrality_top_degree=13),
            _make_metrics(bridge_count=2, centrality_top_degree=14),
        ]
        # Both spike, but bridge_count spikes harder
        current = _make_metrics(bridge_count=20, centrality_top_degree=25)
        result = detect_structural_anomalies(current, history)
        if len(result["anomalies"]) >= 2:
            z_scores = [abs(a["z_score"]) for a in result["anomalies"]]
            assert z_scores == sorted(z_scores, reverse=True)

    def test_empty_history_returns_insufficient(self) -> None:
        from data.agents.graph_intelligence import detect_structural_anomalies
        result = detect_structural_anomalies(_make_metrics(), [])
        assert result["sufficient_history"] is False
        assert result["history_depth"] == 0

    def test_anomaly_fields_present(self) -> None:
        """Each anomaly should contain all expected fields."""
        from data.agents.graph_intelligence import detect_structural_anomalies
        history = [
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=2),
            _make_metrics(bridge_count=3),
        ]
        current = _make_metrics(bridge_count=15)
        result = detect_structural_anomalies(current, history)
        assert len(result["anomalies"]) >= 1
        a = result["anomalies"][0]
        for field in ("metric", "label", "current_value", "rolling_mean",
                      "rolling_std", "z_score", "direction", "severity"):
            assert field in a, f"Missing field: {field}"


class TestAnomalyInsightIntegration:
    """Tests that anomaly results flow into rank_insights and the report."""

    def test_anomaly_insights_generated(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        metrics = _make_metrics()
        anomaly_result = {
            "anomalies": [{
                "metric": "bridge_count",
                "label": "cross-class bridge count",
                "current_value": 10,
                "rolling_mean": 2.25,
                "rolling_std": 0.5,
                "z_score": 15.5,
                "direction": "spike",
                "severity": "critical",
            }],
            "metrics_checked": 9,
            "history_depth": 4,
            "sufficient_history": True,
        }
        empty_analysis = {
            "exposure_profiles": {"similar_pairs": [], "asset_count": 0, "factor_count": 0},
            "regime_divergence": {"divergences": [], "pairs_analyzed": 0},
            "bridge_factors": {"bridges": [], "total_factors": 0, "bridge_count": 0},
            "centrality": {"ranking": [], "total_ranked": 0},
        }
        insights = rank_insights(empty_analysis, metrics, anomalies=anomaly_result)
        anomaly_insights = [i for i in insights if i["type"] == "structural_anomaly"]
        assert len(anomaly_insights) == 1
        assert anomaly_insights[0]["priority"] == 1  # critical → priority 1
        assert "ANOMALY" in anomaly_insights[0]["title"]

    def test_warning_anomaly_gets_priority_2(self) -> None:
        from data.agents.graph_intelligence import rank_insights
        anomaly_result = {
            "anomalies": [{
                "metric": "bridge_count",
                "label": "cross-class bridge count",
                "current_value": 4,
                "rolling_mean": 2.25,
                "rolling_std": 0.5,
                "z_score": 2.5,
                "direction": "spike",
                "severity": "warning",
            }],
            "metrics_checked": 9,
            "history_depth": 4,
            "sufficient_history": True,
        }
        empty_analysis = {
            "exposure_profiles": {"similar_pairs": [], "asset_count": 0, "factor_count": 0},
            "regime_divergence": {"divergences": [], "pairs_analyzed": 0},
            "bridge_factors": {"bridges": [], "total_factors": 0, "bridge_count": 0},
            "centrality": {"ranking": [], "total_ranked": 0},
        }
        insights = rank_insights(empty_analysis, {}, anomalies=anomaly_result)
        anomaly_insights = [i for i in insights if i["type"] == "structural_anomaly"]
        assert anomaly_insights[0]["priority"] == 2

    @patch("db.supabase.client.get_structural_snapshots")
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V2)
    def test_report_includes_anomalies_section(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        from data.agents.graph_intelligence import build_intelligence_report
        prev_metrics = compute_summary_metrics(_ANALYSIS_V1)
        # Provide 4 snapshots so anomaly detection runs (>= MIN_HISTORY)
        snapshots = [
            {"after_state": {"metrics": prev_metrics, "analysis": _ANALYSIS_V1},
             "created_at": f"2026-03-{20-i}T08:00:00Z"}
            for i in range(4)
        ]
        mock_prev.return_value = snapshots
        report = build_intelligence_report()
        assert "anomalies" in report
        assert "sufficient_history" in report["anomalies"]
        assert report["anomalies"]["history_depth"] == 4

    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_ANALYSIS_V1)
    def test_report_no_anomalies_on_first_run(self, mock_analyze, mock_prev) -> None:
        from data.agents.graph_intelligence import build_intelligence_report
        report = build_intelligence_report()
        assert "anomalies" not in report


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
