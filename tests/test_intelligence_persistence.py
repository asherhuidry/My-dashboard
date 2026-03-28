"""Tests for intelligence report persistence — full report stored, summary in metadata."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from db.supabase.client import persist_intelligence_report


def _sample_report() -> dict:
    """Build a minimal but representative intelligence report dict."""
    return {
        "run_id": "run-abc-123",
        "previous_timestamp": "2026-03-20T12:00:00Z",
        "insights": [
            {"type": "bridge_factor", "message": "FEDFUNDS bridges 4 classes", "priority": 1},
            {"type": "confidence", "message": "Mean confidence is high", "priority": 2},
        ],
        "reasoning": {
            "most_influential_factor": {"factor_id": "FEDFUNDS"},
            "structural_health": {"score": 0.72, "grade": "good"},
        },
        "metrics": {"total_nodes": 42, "total_edges": 108},
        "provenance": {"AAPL": {"source": "yfinance"}},
        "analysis": {"centrality": {"ranking": []}},
        "confidence": {"mean_effective": 0.64, "scored_edges": 80},
        "anomalies": {"anomalies": [{"metric": "density", "z": 2.5}]},
    }


@patch("db.supabase.client.get_client")
class TestPersistIntelligenceReport:
    """Verify that the full report is stored, not just a summary."""

    def test_full_report_in_after_state(self, mock_get_client: MagicMock) -> None:
        """after_state should contain the entire report dict."""
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]
        mock_get_client.return_value = client

        report = _sample_report()
        persist_intelligence_report(report, run_id="run-abc-123")

        call_args = client.table.return_value.insert.call_args[0][0]
        after = call_args["after_state"]

        # Full report keys must be present
        assert "insights" in after
        assert "analysis" in after
        assert "provenance" in after
        assert "reasoning" in after
        assert "confidence" in after
        assert "anomalies" in after
        assert len(after["insights"]) == 2

    def test_summary_in_metadata(self, mock_get_client: MagicMock) -> None:
        """metadata should contain quick-access summary fields."""
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]
        mock_get_client.return_value = client

        report = _sample_report()
        persist_intelligence_report(report, run_id="run-abc-123")

        call_args = client.table.return_value.insert.call_args[0][0]
        meta = call_args["metadata"]

        assert meta["run_id"] == "run-abc-123"
        assert meta["insight_count"] == 2
        assert meta["node_count"] == 42
        assert meta["edge_count"] == 108
        assert meta["anomaly_count"] == 1
        assert meta["mean_effective_confidence"] == 0.64

    def test_reasoning_summary_extracted(self, mock_get_client: MagicMock) -> None:
        """metadata.reasoning_summary should pull structural_health from reasoning."""
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]
        mock_get_client.return_value = client

        report = _sample_report()
        persist_intelligence_report(report)

        call_args = client.table.return_value.insert.call_args[0][0]
        meta = call_args["metadata"]
        assert meta["reasoning_summary"] == {"score": 0.72, "grade": "good"}

    def test_agent_id_and_action(self, mock_get_client: MagicMock) -> None:
        """Should write with agent_id=graph_intelligence and action=intelligence_report."""
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.return_value.data = [{"id": 1}]
        mock_get_client.return_value = client

        persist_intelligence_report(_sample_report())

        call_args = client.table.return_value.insert.call_args[0][0]
        assert call_args["agent_id"] == "graph_intelligence"
        assert call_args["action"] == "intelligence_report"
