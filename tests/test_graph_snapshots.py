"""Tests for structural snapshot persistence and week-over-week diff.

Covers: summary metrics computation, diff logic, snapshot persistence
to evolution_log, pipeline wiring via _snapshot_graph_structure, and
the Supabase query helper.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Sample analysis outputs ──────────────────────────────────────────────────

_SAMPLE_ANALYSIS: dict[str, Any] = {
    "exposure_profiles": {
        "similar_pairs": [
            {"asset_a": "AAPL", "asset_b": "MSFT", "cosine_similarity": 0.96,
             "shared_factors": 3, "class_a": "equity", "class_b": "equity"},
            {"asset_a": "GLD", "asset_b": "AAPL", "cosine_similarity": -0.40,
             "shared_factors": 2, "class_a": "commodity", "class_b": "equity"},
        ],
        "asset_count": 3,
        "factor_count": 4,
    },
    "regime_divergence": {
        "divergences": [
            {"asset": "AAPL", "factor": "rate", "regime": "stress",
             "beta_all": -0.30, "beta_regime": -0.60, "abs_change": 0.30,
             "rel_change": 1.0, "direction": "amplified"},
            {"asset": "GLD", "factor": "dollar", "regime": "bear",
             "beta_all": -0.50, "beta_regime": -0.30, "abs_change": 0.20,
             "rel_change": 0.4, "direction": "dampened"},
        ],
        "pairs_analyzed": 3,
    },
    "bridge_factors": {
        "bridges": [
            {"factor_id": "VIXCLS", "factor_name": "VIX", "asset_classes": ["equity", "commodity"],
             "class_count": 2, "assets": ["AAPL", "GLD"], "asset_count": 2,
             "edge_count": 4, "is_bridge": True},
            {"factor_id": "DCOILWTICO", "factor_name": "Oil", "asset_classes": ["equity"],
             "class_count": 1, "assets": ["AAPL"], "asset_count": 1,
             "edge_count": 1, "is_bridge": False},
        ],
        "total_factors": 2,
        "bridge_count": 1,
    },
    "centrality": {
        "ranking": [
            {"node_id": "AAPL", "label": "Asset", "degree": 12},
            {"node_id": "VIXCLS", "label": "MacroIndicator", "degree": 8},
        ],
        "total_ranked": 2,
    },
}


# ── Test: Summary metrics computation ────────────────────────────────────────

class TestComputeSummaryMetrics:
    """Tests for compute_summary_metrics()."""

    def test_exposure_metrics(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        m = compute_summary_metrics(_SAMPLE_ANALYSIS)
        assert m["exposure_asset_count"] == 3
        assert m["exposure_factor_count"] == 4
        assert m["exposure_pair_count"] == 2
        assert m["exposure_sim_max"] == 0.96
        assert m["exposure_top_pair"] == "AAPL/MSFT"

    def test_regime_divergence_metrics(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        m = compute_summary_metrics(_SAMPLE_ANALYSIS)
        assert m["regime_pairs_analyzed"] == 3
        assert m["regime_divergence_count"] == 2
        assert m["regime_divergence_max"] == 0.30
        assert m["regime_amplified_pct"] == 0.5  # 1 amplified out of 2
        assert m["regime_top_shift"] == "AAPL/rate/stress"

    def test_bridge_metrics(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        m = compute_summary_metrics(_SAMPLE_ANALYSIS)
        assert m["bridge_count"] == 1
        assert m["bridge_max_span"] == 2
        assert m["bridge_factor_ids"] == ["VIXCLS"]

    def test_centrality_metrics(self) -> None:
        from data.agents.graph_analyzer import compute_summary_metrics
        m = compute_summary_metrics(_SAMPLE_ANALYSIS)
        assert m["centrality_top_node"] == "AAPL"
        assert m["centrality_top_degree"] == 12
        assert m["centrality_mean_degree"] == 10.0

    def test_empty_analysis(self) -> None:
        """Empty analysis produces safe None values, no crash."""
        from data.agents.graph_analyzer import compute_summary_metrics
        empty = {
            "exposure_profiles": {"similar_pairs": [], "asset_count": 0, "factor_count": 0},
            "regime_divergence": {"divergences": [], "pairs_analyzed": 0},
            "bridge_factors": {"bridges": [], "total_factors": 0, "bridge_count": 0},
            "centrality": {"ranking": [], "total_ranked": 0},
        }
        m = compute_summary_metrics(empty)
        assert m["exposure_sim_mean"] is None
        assert m["regime_divergence_max"] is None
        assert m["bridge_count"] == 0
        assert m["centrality_top_node"] is None


# ── Test: Snapshot diff ──────────────────────────────────────────────────────

class TestComputeSnapshotDiff:
    """Tests for compute_snapshot_diff()."""

    def _make_metrics(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "exposure_asset_count": 3, "exposure_factor_count": 4,
            "exposure_sim_mean": 0.28, "exposure_sim_max": 0.96,
            "exposure_top_pair": "AAPL/MSFT",
            "regime_pairs_analyzed": 3, "regime_divergence_count": 2,
            "regime_divergence_mean": 0.25, "regime_divergence_max": 0.30,
            "regime_amplified_pct": 0.5, "regime_top_shift": "AAPL/rate/stress",
            "bridge_count": 1, "bridge_max_span": 2,
            "bridge_factor_ids": ["VIXCLS"],
            "centrality_top_node": "AAPL", "centrality_top_degree": 12,
            "centrality_mean_degree": 10.0,
        }
        base.update(overrides)
        return base

    def test_no_change_produces_zero_deltas(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        m = self._make_metrics()
        diff = compute_snapshot_diff(m, m)
        for key, val in diff["metric_deltas"].items():
            if val is not None:
                assert val == 0.0, f"{key} should be zero"

    def test_metric_deltas_computed(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        prev = self._make_metrics(bridge_count=1, centrality_top_degree=10)
        curr = self._make_metrics(bridge_count=3, centrality_top_degree=14)
        diff = compute_snapshot_diff(curr, prev)
        assert diff["metric_deltas"]["bridge_count"] == 2
        assert diff["metric_deltas"]["centrality_top_degree"] == 4

    def test_bridge_gained(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        prev = self._make_metrics(bridge_factor_ids=["VIXCLS"])
        curr = self._make_metrics(bridge_factor_ids=["VIXCLS", "GS10"])
        diff = compute_snapshot_diff(curr, prev)
        assert diff["bridges_gained"] == ["GS10"]
        assert diff["bridges_lost"] == []

    def test_bridge_lost(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        prev = self._make_metrics(bridge_factor_ids=["VIXCLS", "GS10"])
        curr = self._make_metrics(bridge_factor_ids=["VIXCLS"])
        diff = compute_snapshot_diff(curr, prev)
        assert diff["bridges_gained"] == []
        assert diff["bridges_lost"] == ["GS10"]

    def test_top_node_changed(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        prev = self._make_metrics(centrality_top_node="AAPL")
        curr = self._make_metrics(centrality_top_node="MSFT")
        diff = compute_snapshot_diff(curr, prev)
        assert diff["centrality_top_node_changed"] is True

    def test_top_node_unchanged(self) -> None:
        from data.agents.graph_analyzer import compute_snapshot_diff
        m = self._make_metrics()
        diff = compute_snapshot_diff(m, m)
        assert diff["centrality_top_node_changed"] is False
        assert diff["exposure_top_pair_changed"] is False
        assert diff["regime_top_shift_changed"] is False


# ── Test: Snapshot persistence ───────────────────────────────────────────────

class TestSnapshotGraphStructure:
    """Tests for snapshot_graph_structure()."""

    @patch("db.supabase.client.log_evolution")
    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_persists_to_evolution_log(self, mock_analyze, mock_prev, mock_log) -> None:
        """snapshot_graph_structure should call log_evolution."""
        from data.agents.graph_analyzer import snapshot_graph_structure
        result = snapshot_graph_structure(run_id="run-42")
        mock_log.assert_called_once()
        entry = mock_log.call_args[0][0]
        assert entry.agent_id == "graph_analyzer"
        assert entry.action == "structural_snapshot"

    @patch("db.supabase.client.log_evolution")
    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_snapshot_contains_metrics(self, mock_analyze, mock_prev, mock_log) -> None:
        """Snapshot should include computed summary metrics."""
        from data.agents.graph_analyzer import snapshot_graph_structure
        result = snapshot_graph_structure(run_id="run-42")
        assert "metrics" in result
        assert result["metrics"]["exposure_asset_count"] == 3
        assert result["metrics"]["centrality_top_node"] == "AAPL"

    @patch("db.supabase.client.log_evolution")
    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_no_diff_on_first_snapshot(self, mock_analyze, mock_prev, mock_log) -> None:
        """First snapshot has no previous → no diff in result."""
        from data.agents.graph_analyzer import snapshot_graph_structure
        result = snapshot_graph_structure(run_id="run-42")
        assert "diff" not in result

    @patch("db.supabase.client.log_evolution")
    @patch("db.supabase.client.get_structural_snapshots")
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_diff_computed_against_previous(self, mock_analyze, mock_prev, mock_log) -> None:
        """When a previous snapshot exists, diff should be computed."""
        from data.agents.graph_analyzer import snapshot_graph_structure, compute_summary_metrics
        prev_metrics = compute_summary_metrics(_SAMPLE_ANALYSIS)
        prev_metrics["bridge_count"] = 0  # simulate a change
        mock_prev.return_value = [{
            "after_state": {"metrics": prev_metrics},
            "created_at": "2026-03-20T08:00:00Z",
        }]
        result = snapshot_graph_structure(run_id="run-42")
        assert "diff" in result
        assert result["diff"]["metric_deltas"]["bridge_count"] == 1

    @patch("db.supabase.client.log_evolution")
    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_run_id_stored(self, mock_analyze, mock_prev, mock_log) -> None:
        """Snapshot should carry the discovery run_id."""
        from data.agents.graph_analyzer import snapshot_graph_structure
        result = snapshot_graph_structure(run_id="run-42")
        assert result["run_id"] == "run-42"

    @patch("db.supabase.client.log_evolution", side_effect=RuntimeError("Supabase down"))
    @patch("db.supabase.client.get_structural_snapshots", return_value=[])
    @patch("data.agents.graph_analyzer.analyze_graph_structure", return_value=_SAMPLE_ANALYSIS)
    def test_persistence_failure_does_not_raise(self, mock_analyze, mock_prev, mock_log) -> None:
        """Supabase failure should log a warning, not crash."""
        from data.agents.graph_analyzer import snapshot_graph_structure
        result = snapshot_graph_structure(run_id="run-42")
        assert "metrics" in result  # result still returned


# ── Test: Pipeline wiring ────────────────────────────────────────────────────

class TestPipelineWiring:
    """Tests for _snapshot_graph_structure in correlation_hunter."""

    def test_skips_on_none_run_id(self) -> None:
        """Should return None when run_id is None."""
        from data.agents.correlation_hunter import _snapshot_graph_structure
        assert _snapshot_graph_structure(None) is None

    @patch("data.agents.graph_analyzer.snapshot_graph_structure")
    def test_calls_snapshot(self, mock_snap) -> None:
        """Should call snapshot_graph_structure with the run_id."""
        from data.agents.correlation_hunter import _snapshot_graph_structure
        mock_snap.return_value = {"metrics": {}, "timestamp": "now"}
        result = _snapshot_graph_structure("run-99")
        mock_snap.assert_called_once_with(run_id="run-99")
        assert result is not None

    @patch("data.agents.graph_analyzer.snapshot_graph_structure",
           side_effect=RuntimeError("Neo4j down"))
    def test_failure_returns_none(self, mock_snap) -> None:
        """Should return None on failure, not raise."""
        from data.agents.correlation_hunter import _snapshot_graph_structure
        result = _snapshot_graph_structure("run-99")
        assert result is None


# ── Test: Supabase query helper ──────────────────────────────────────────────

class TestGetStructuralSnapshots:
    """Tests for get_structural_snapshots()."""

    @patch("db.supabase.client.get_client")
    def test_queries_evolution_log(self, mock_get_client) -> None:
        """Should query evolution_log with correct filters."""
        mock_result = MagicMock()
        mock_result.data = [{"id": "1", "after_state": {"metrics": {}}}]

        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query

        mock_table = MagicMock()
        mock_table.select.return_value = mock_query

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        from db.supabase.client import get_structural_snapshots
        rows = get_structural_snapshots(limit=5)

        mock_client.table.assert_called_with("evolution_log")
        assert len(rows) == 1

    @patch("db.supabase.client.get_client")
    def test_returns_empty_list_when_no_snapshots(self, mock_get_client) -> None:
        mock_result = MagicMock()
        mock_result.data = []

        mock_query = MagicMock()
        mock_query.execute.return_value = mock_result
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query

        mock_table = MagicMock()
        mock_table.select.return_value = mock_query

        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        from db.supabase.client import get_structural_snapshots
        rows = get_structural_snapshots()
        assert rows == []
