"""Tests for sector stress analysis and earnings exposure intelligence.

Tests cover:
- analyze_sector_stress() aggregation logic
- analyze_earnings_exposure() structural importance scoring
- New insight types (sector_stress, earnings_catalyst, earnings_surprise)
- Reasoning summary (most_stressed_sector)
- Summary metrics and snapshot diff extensions
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

_NEO4J_QUERY = "data.agents.graph_analyzer.run_read_query"


# ── Sample analysis data (reusable across tests) ────────────────────────────

def _sample_analysis() -> dict[str, Any]:
    """Build a realistic analysis dict with regime divergence and bridges."""
    return {
        "regime_divergence": {
            "divergences": [
                {"asset": "XOM",  "factor": "oil",  "regime": "stress",
                 "beta_all": 0.30, "beta_regime": 0.65, "abs_change": 0.35, "direction": "amplified"},
                {"asset": "CVX",  "factor": "oil",  "regime": "stress",
                 "beta_all": 0.28, "beta_regime": 0.55, "abs_change": 0.27, "direction": "amplified"},
                {"asset": "AAPL", "factor": "rate", "regime": "bear",
                 "beta_all": -0.30, "beta_regime": -0.55, "abs_change": 0.25, "direction": "amplified"},
                {"asset": "MSFT", "factor": "rate", "regime": "bear",
                 "beta_all": -0.28, "beta_regime": -0.45, "abs_change": 0.17, "direction": "amplified"},
                {"asset": "JPM",  "factor": "rate", "regime": "stress",
                 "beta_all": 0.40, "beta_regime": 0.60, "abs_change": 0.20, "direction": "amplified"},
                {"asset": "GLD",  "factor": "dollar","regime": "bear",
                 "beta_all": -0.50, "beta_regime": -0.30, "abs_change": 0.20, "direction": "dampened"},
            ],
            "pairs_analyzed": 20,
        },
        "bridge_factors": {
            "bridges": [
                {"factor_id": "DCOILWTICO", "factor_name": "WTI Crude",
                 "asset_classes": ["equity", "commodity"], "class_count": 2,
                 "assets": ["XOM", "CVX", "COP"], "asset_count": 3, "edge_count": 5, "is_bridge": True},
                {"factor_id": "VIXCLS", "factor_name": "VIX",
                 "asset_classes": ["equity", "commodity", "crypto"], "class_count": 3,
                 "assets": ["AAPL", "MSFT", "JPM", "GLD"], "asset_count": 4, "edge_count": 8, "is_bridge": True},
            ],
            "total_factors": 5,
            "bridge_count": 2,
        },
        "exposure_profiles": {
            "similar_pairs": [
                {"asset_a": "XOM", "asset_b": "CVX", "class_a": "equity", "class_b": "equity",
                 "cosine_similarity": 0.95, "shared_factors": 3},
            ],
            "asset_count": 6,
            "factor_count": 4,
        },
        "centrality": {
            "ranking": [
                {"node_id": "AAPL",   "label": "Asset", "degree": 12},
                {"node_id": "VIXCLS", "label": "MacroIndicator", "degree": 10},
                {"node_id": "MSFT",   "label": "Asset", "degree": 8},
                {"node_id": "JPM",    "label": "Asset", "degree": 6},
                {"node_id": "XOM",    "label": "Asset", "degree": 5},
            ],
            "total_ranked": 5,
        },
    }


# Sector membership rows returned by Neo4j
_SECTOR_ROWS = [
    {"asset": "XOM",  "sector": "Energy"},
    {"asset": "CVX",  "sector": "Energy"},
    {"asset": "COP",  "sector": "Energy"},
    {"asset": "AAPL", "sector": "Information Technology"},
    {"asset": "MSFT", "sector": "Information Technology"},
    {"asset": "JPM",  "sector": "Financials"},
    {"asset": "BAC",  "sector": "Financials"},
]

# Earnings event rows returned by Neo4j
_EARNINGS_ROWS = [
    {"asset": "AAPL", "event_id": "earnings_AAPL_2026-04-01",
     "event_date": "2099-04-01", "eps_estimate": 1.45, "eps_actual": None,
     "surprise_pct": None, "hour": "amc"},
    {"asset": "XOM",  "event_id": "earnings_XOM_2026-04-02",
     "event_date": "2099-04-02", "eps_estimate": 2.10, "eps_actual": None,
     "surprise_pct": None, "hour": "bmo"},
    {"asset": "JPM",  "event_id": "earnings_JPM_2026-03-20",
     "event_date": "2020-03-20", "eps_estimate": 3.00, "eps_actual": 3.25,
     "surprise_pct": 8.33, "hour": "bmo"},
]


# ── Sector stress analysis tests ────────────────────────────────────────────

class TestSectorStress:
    """Tests for analyze_sector_stress()."""

    @patch(_NEO4J_QUERY, return_value=_SECTOR_ROWS)
    def test_energy_sector_has_high_stress(self, mock_q: Any) -> None:
        """Energy sector has high stress due to XOM/CVX regime divergence."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())

        assert result["sector_count"] >= 3
        sectors = result["sectors"]
        energy = next(s for s in sectors if s["sector"] == "Energy")
        assert energy["stress_score"] > 0.3
        assert energy["mean_regime_divergence"] > 0
        # All sectors should be ranked by stress score descending
        scores = [s["stress_score"] for s in sectors]
        assert scores == sorted(scores, reverse=True)

    @patch(_NEO4J_QUERY, return_value=_SECTOR_ROWS)
    def test_sector_members_populated(self, mock_q: Any) -> None:
        """Each sector has its member list."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())

        energy = next(s for s in result["sectors"] if s["sector"] == "Energy")
        assert set(energy["members"]) == {"XOM", "CVX", "COP"}
        assert energy["member_count"] == 3

    @patch(_NEO4J_QUERY, return_value=_SECTOR_ROWS)
    def test_divergence_breadth_calculated(self, mock_q: Any) -> None:
        """Divergence breadth = fraction of members with ANY regime divergence."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())

        energy = next(s for s in result["sectors"] if s["sector"] == "Energy")
        # XOM and CVX have divergences, COP does not → 2/3
        assert energy["divergence_breadth"] == pytest.approx(2.0 / 3.0, abs=0.01)

    @patch(_NEO4J_QUERY, return_value=_SECTOR_ROWS)
    def test_bridge_exposure_aggregated(self, mock_q: Any) -> None:
        """Bridge exposure counts how many bridge factors touch sector members."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())

        tech = next(s for s in result["sectors"] if s["sector"] == "Information Technology")
        # AAPL and MSFT each appear in VIXCLS bridge → mean bridge exposure > 0
        assert tech["mean_bridge_exposure"] > 0

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_graph_returns_empty(self, mock_q: Any) -> None:
        """Returns empty sectors list if no BELONGS_TO edges exist."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())
        assert result == {"sectors": [], "sector_count": 0}

    @patch(_NEO4J_QUERY, side_effect=Exception("neo4j down"))
    def test_handles_neo4j_failure(self, mock_q: Any) -> None:
        """Returns empty result on Neo4j failure."""
        from data.agents.graph_analyzer import analyze_sector_stress
        result = analyze_sector_stress(_sample_analysis())
        assert result == {"sectors": [], "sector_count": 0}


# ── Earnings exposure analysis tests ────────────────────────────────────────

class TestEarningsExposure:
    """Tests for analyze_earnings_exposure()."""

    @patch(_NEO4J_QUERY, return_value=_EARNINGS_ROWS)
    def test_structural_importance_scored(self, mock_q: Any) -> None:
        """Each event gets a structural importance score."""
        from data.agents.graph_analyzer import analyze_earnings_exposure
        result = analyze_earnings_exposure(_sample_analysis())

        assert result["event_count"] == 3
        for e in result["events"]:
            assert "structural_importance" in e
            assert 0 <= e["structural_importance"] <= 1.0

    @patch(_NEO4J_QUERY, return_value=_EARNINGS_ROWS)
    def test_aapl_highest_importance(self, mock_q: Any) -> None:
        """AAPL has highest centrality (12) → should rank first."""
        from data.agents.graph_analyzer import analyze_earnings_exposure
        result = analyze_earnings_exposure(_sample_analysis())

        top = result["events"][0]
        assert top["asset"] == "AAPL"
        assert top["degree_centrality"] == 12

    @patch(_NEO4J_QUERY, return_value=_EARNINGS_ROWS)
    def test_upcoming_vs_recent_counted(self, mock_q: Any) -> None:
        """Correctly separates upcoming from recent events."""
        from data.agents.graph_analyzer import analyze_earnings_exposure
        result = analyze_earnings_exposure(_sample_analysis())

        # 2 events in 2099 (upcoming), 1 in 2020 (past with actuals)
        assert result["upcoming_count"] == 2
        assert result["recent_with_actuals"] == 1

    @patch(_NEO4J_QUERY, return_value=[])
    def test_empty_calendar(self, mock_q: Any) -> None:
        """Returns empty when no earnings events exist."""
        from data.agents.graph_analyzer import analyze_earnings_exposure
        result = analyze_earnings_exposure(_sample_analysis())
        assert result == {"events": [], "event_count": 0}


# ── Insight ranking tests ───────────────────────────────────────────────────

class TestSectorInsights:
    """Test that sector stress and earnings generate proper insights."""

    def _make_analysis_with_sectors(self) -> dict[str, Any]:
        """Analysis dict that includes sector_stress and earnings_exposure."""
        analysis = _sample_analysis()
        analysis["sector_stress"] = {
            "sectors": [
                {
                    "sector": "Energy",
                    "stress_score": 0.72,
                    "member_count": 3,
                    "mean_regime_divergence": 0.207,
                    "max_regime_divergence": 0.35,
                    "most_divergent_member": "XOM",
                    "mean_bridge_exposure": 1.0,
                    "divergence_breadth": 0.667,
                    "members": ["COP", "CVX", "XOM"],
                },
                {
                    "sector": "Information Technology",
                    "stress_score": 0.38,
                    "member_count": 2,
                    "mean_regime_divergence": 0.21,
                    "max_regime_divergence": 0.25,
                    "most_divergent_member": "AAPL",
                    "mean_bridge_exposure": 1.0,
                    "divergence_breadth": 1.0,
                    "members": ["AAPL", "MSFT"],
                },
            ],
            "sector_count": 3,
        }
        analysis["earnings_exposure"] = {
            "events": [
                {
                    "asset": "AAPL", "event_id": "e1",
                    "event_date": "2099-04-01",
                    "is_upcoming": True, "has_actuals": False,
                    "structural_importance": 0.65,
                    "degree_centrality": 12, "regime_divergence": 0.25,
                    "bridge_exposure": 1,
                    "eps_estimate": 1.45, "eps_actual": None,
                    "surprise_pct": None, "hour": "amc",
                },
            ],
            "event_count": 1,
            "upcoming_count": 1,
            "recent_with_actuals": 0,
        }
        return analysis

    def test_sector_stress_insight_generated(self) -> None:
        """High-stress sector (>0.60) generates priority-1 insight."""
        from data.agents.graph_intelligence import rank_insights
        from data.agents.graph_analyzer import compute_summary_metrics

        analysis = self._make_analysis_with_sectors()
        metrics = compute_summary_metrics(analysis)
        insights = rank_insights(analysis, metrics)

        sector_insights = [i for i in insights if i["type"] == "sector_stress"]
        assert len(sector_insights) >= 1
        top = sector_insights[0]
        assert "Energy" in top["title"]
        assert top["priority"] == 1

    def test_earnings_catalyst_insight_generated(self) -> None:
        """Upcoming earnings for high-importance asset generates insight."""
        from data.agents.graph_intelligence import rank_insights
        from data.agents.graph_analyzer import compute_summary_metrics

        analysis = self._make_analysis_with_sectors()
        metrics = compute_summary_metrics(analysis)
        insights = rank_insights(analysis, metrics)

        earnings_insights = [i for i in insights if i["type"] == "earnings_catalyst"]
        assert len(earnings_insights) >= 1
        assert "AAPL" in earnings_insights[0]["title"]

    def test_moderate_stress_gets_priority_2(self) -> None:
        """Sector with stress 0.35-0.60 gets priority 2."""
        from data.agents.graph_intelligence import rank_insights
        from data.agents.graph_analyzer import compute_summary_metrics

        analysis = self._make_analysis_with_sectors()
        metrics = compute_summary_metrics(analysis)
        insights = rank_insights(analysis, metrics)

        tech_insights = [
            i for i in insights
            if i["type"] == "sector_stress" and "Information Technology" in i["title"]
        ]
        assert len(tech_insights) == 1
        assert tech_insights[0]["priority"] == 2


# ── Reasoning summary tests ─────────────────────────────────────────────────

class TestSectorReasoning:
    """Test most_stressed_sector in reasoning summary."""

    def test_most_stressed_sector_in_reasoning(self) -> None:
        """compute_graph_reasoning_summary includes most_stressed_sector."""
        from data.agents.graph_intelligence import compute_graph_reasoning_summary

        analysis = _sample_analysis()
        analysis["sector_stress"] = {
            "sectors": [
                {
                    "sector": "Energy",
                    "stress_score": 0.72,
                    "member_count": 3,
                    "mean_regime_divergence": 0.207,
                    "most_divergent_member": "XOM",
                    "divergence_breadth": 0.667,
                },
            ],
            "sector_count": 1,
        }
        reasoning = compute_graph_reasoning_summary(analysis, {})

        assert "most_stressed_sector" in reasoning
        mss = reasoning["most_stressed_sector"]
        assert mss["sector"] == "Energy"
        assert mss["stress_score"] == 0.72
        assert "Energy" in mss["narrative"]
        assert "XOM" in mss["narrative"]

    def test_no_sectors_no_crash(self) -> None:
        """Reasoning doesn't crash when sector_stress is empty."""
        from data.agents.graph_intelligence import compute_graph_reasoning_summary

        analysis = _sample_analysis()
        analysis["sector_stress"] = {"sectors": [], "sector_count": 0}
        reasoning = compute_graph_reasoning_summary(analysis, {})
        assert "most_stressed_sector" not in reasoning


# ── Summary metrics tests ───────────────────────────────────────────────────

class TestSectorMetrics:
    """Test that summary metrics include sector and earnings fields."""

    def test_sector_metrics_present(self) -> None:
        """compute_summary_metrics includes sector stress metrics."""
        from data.agents.graph_analyzer import compute_summary_metrics

        analysis = _sample_analysis()
        analysis["sector_stress"] = {
            "sectors": [
                {"sector": "Energy", "stress_score": 0.72, "member_count": 3,
                 "mean_regime_divergence": 0.2, "max_regime_divergence": 0.35,
                 "most_divergent_member": "XOM", "mean_bridge_exposure": 1.0,
                 "divergence_breadth": 0.667, "members": ["XOM", "CVX", "COP"]},
            ],
            "sector_count": 1,
        }
        analysis["earnings_exposure"] = {
            "events": [
                {"structural_importance": 0.65, "is_upcoming": True, "has_actuals": False},
            ],
            "event_count": 1, "upcoming_count": 1, "recent_with_actuals": 0,
        }
        metrics = compute_summary_metrics(analysis)

        assert metrics["sector_count"] == 1
        assert metrics["sector_stress_max"] == 0.72
        assert metrics["sector_most_stressed"] == "Energy"
        assert metrics["earnings_upcoming"] == 1
        assert metrics["earnings_high_impact"] == 1

    def test_snapshot_diff_tracks_sector_changes(self) -> None:
        """compute_snapshot_diff includes sector_most_stressed_changed."""
        from data.agents.graph_analyzer import compute_snapshot_diff

        current = {"sector_most_stressed": "Energy", "sector_stress_max": 0.72}
        previous = {"sector_most_stressed": "Financials", "sector_stress_max": 0.55}
        diff = compute_snapshot_diff(current, previous)

        assert diff["sector_most_stressed_changed"] is True
        assert diff["metric_deltas"]["sector_stress_max"] == pytest.approx(0.17, abs=0.001)
