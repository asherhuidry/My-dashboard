"""Tests for discovery persistence — DiscoveryRecord, save/query, and correlation_hunter wiring.

All Supabase calls are mocked so the suite is deterministic and offline.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _make_mock_client(return_data: list[dict[str, Any]]) -> MagicMock:
    """Return a mock Supabase client with fluent API support."""
    mock_result = MagicMock()
    mock_result.data = return_data

    mock_query = MagicMock()
    mock_query.execute.return_value = mock_result
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query

    mock_table = MagicMock()
    mock_table.insert.return_value = mock_query
    mock_table.select.return_value = mock_query

    mock_client = MagicMock()
    mock_client.table.return_value = mock_table
    return mock_client


# ── DiscoveryRecord dataclass ────────────────────────────────────────────────

class TestDiscoveryRecord:
    def test_defaults(self):
        from db.supabase.client import DiscoveryRecord
        rec = DiscoveryRecord(series_a="SPY", series_b="GLD", lag_days=5, pearson_r=0.72)
        assert rec.regime == "all"
        assert rec.strength == "moderate"
        assert rec.relationship_type == "discovered"
        assert rec.granger_p is None
        assert rec.mutual_info is None
        assert isinstance(rec.run_id, str)
        assert isinstance(rec.computed_at, datetime)

    def test_to_payload_required_fields(self):
        from db.supabase.client import DiscoveryRecord
        rec = DiscoveryRecord(
            series_a="T10Y2Y", series_b="SPY", lag_days=21,
            pearson_r=-0.456789, strength="moderate",
        )
        payload = rec.to_payload()
        assert payload["series_a"] == "T10Y2Y"
        assert payload["series_b"] == "SPY"
        assert payload["lag_days"] == 21
        assert payload["pearson_r"] == -0.456789
        assert payload["regime"] == "all"
        assert payload["strength"] == "moderate"
        assert payload["relationship_type"] == "discovered"
        assert "run_id" in payload
        assert "computed_at" in payload

    def test_to_payload_optional_fields(self):
        from db.supabase.client import DiscoveryRecord
        rec = DiscoveryRecord(
            series_a="VIXCLS", series_b="SPY", lag_days=1,
            pearson_r=-0.65, granger_p=0.01234, mutual_info=0.087654,
        )
        payload = rec.to_payload()
        assert payload["granger_p"] == 0.012340
        assert payload["mutual_info"] == 0.087654

    def test_to_payload_omits_none_optionals(self):
        from db.supabase.client import DiscoveryRecord
        rec = DiscoveryRecord(
            series_a="A", series_b="B", lag_days=0, pearson_r=0.5,
        )
        payload = rec.to_payload()
        assert "granger_p" not in payload
        assert "mutual_info" not in payload

    def test_run_id_shared_across_batch(self):
        from db.supabase.client import DiscoveryRecord
        shared_id = str(uuid.uuid4())
        recs = [
            DiscoveryRecord(series_a="A", series_b="B", lag_days=0, pearson_r=0.5, run_id=shared_id),
            DiscoveryRecord(series_a="C", series_b="D", lag_days=1, pearson_r=0.6, run_id=shared_id),
        ]
        assert recs[0].to_payload()["run_id"] == recs[1].to_payload()["run_id"]


# ── save_discoveries ─────────────────────────────────────────────────────────

class TestSaveDiscoveries:
    def test_bulk_inserts_to_discoveries_table(self):
        from db.supabase.client import DiscoveryRecord, save_discoveries

        inserted = [{"id": "1"}, {"id": "2"}]
        mock_client = _make_mock_client(inserted)

        recs = [
            DiscoveryRecord(series_a="SPY", series_b="GLD", lag_days=5, pearson_r=0.72, run_id="run-1"),
            DiscoveryRecord(series_a="TLT", series_b="DFF", lag_days=3, pearson_r=-0.55, run_id="run-1"),
        ]

        with patch("db.supabase.client.get_client", return_value=mock_client):
            result = save_discoveries(recs)

        assert len(result) == 2
        mock_client.table.assert_called_with("discoveries")
        payloads = mock_client.table().insert.call_args[0][0]
        assert len(payloads) == 2
        assert payloads[0]["series_a"] == "SPY"
        assert payloads[1]["series_a"] == "TLT"

    def test_empty_list_returns_empty(self):
        from db.supabase.client import save_discoveries
        result = save_discoveries([])
        assert result == []

    def test_payload_roundtrip(self):
        """Payloads passed to insert match to_payload() output."""
        from db.supabase.client import DiscoveryRecord, save_discoveries

        mock_client = _make_mock_client([{"id": "x"}])
        rec = DiscoveryRecord(
            series_a="A", series_b="B", lag_days=10, pearson_r=0.88,
            granger_p=0.001, mutual_info=0.12, strength="strong",
            relationship_type="yield_curve_equity",
        )

        with patch("db.supabase.client.get_client", return_value=mock_client):
            save_discoveries([rec])

        payloads = mock_client.table().insert.call_args[0][0]
        assert payloads[0]["strength"] == "strong"
        assert payloads[0]["relationship_type"] == "yield_curve_equity"
        assert payloads[0]["granger_p"] == 0.001
        assert payloads[0]["mutual_info"] == 0.12


# ── get_discoveries ──────────────────────────────────────────────────────────

class TestGetDiscoveries:
    def _make_rows(self) -> list[dict[str, Any]]:
        return [
            {"series_a": "SPY", "series_b": "GLD", "pearson_r": 0.72, "strength": "strong"},
            {"series_a": "TLT", "series_b": "DFF", "pearson_r": -0.55, "strength": "moderate"},
            {"series_a": "SPY", "series_b": "VIX", "pearson_r": -0.80, "strength": "strong"},
        ]

    def test_returns_all_by_default(self):
        from db.supabase.client import get_discoveries
        mock_client = _make_mock_client(self._make_rows())
        with patch("db.supabase.client.get_client", return_value=mock_client):
            rows = get_discoveries()
        assert len(rows) == 3

    def test_filter_by_series(self):
        from db.supabase.client import get_discoveries
        mock_client = _make_mock_client(self._make_rows())
        with patch("db.supabase.client.get_client", return_value=mock_client):
            rows = get_discoveries(series="SPY")
        assert len(rows) == 2
        assert all(r["series_a"] == "SPY" or r["series_b"] == "SPY" for r in rows)

    def test_filter_by_strength(self):
        from db.supabase.client import get_discoveries
        mock_client = _make_mock_client(self._make_rows())
        with patch("db.supabase.client.get_client", return_value=mock_client):
            rows = get_discoveries(strength="strong")
        mock_client.table().select().eq.assert_called_with("strength", "strong")

    def test_filter_by_run_id(self):
        from db.supabase.client import get_discoveries
        mock_client = _make_mock_client([])
        with patch("db.supabase.client.get_client", return_value=mock_client):
            get_discoveries(run_id="abc-123")
        mock_client.table().select().eq.assert_called_with("run_id", "abc-123")

    def test_filter_by_min_abs_r(self):
        from db.supabase.client import get_discoveries
        mock_client = _make_mock_client(self._make_rows())
        with patch("db.supabase.client.get_client", return_value=mock_client):
            rows = get_discoveries(min_abs_r=0.7)
        assert len(rows) == 2
        assert all(abs(r["pearson_r"]) >= 0.7 for r in rows)


# ── Correlation hunter integration ───────────────────────────────────────────

class TestCorrelationHunterPersistence:
    """Test that correlation_hunter.run() persists discoveries."""

    def _make_finding(self, **kwargs):
        from data.agents.correlation_hunter import CorrelationFinding
        defaults = dict(
            series_a="SPY", series_b="GLD", lag_days=5,
            pearson_r=0.72, granger_p=0.03, mutual_info=0.1,
            strength="strong", relationship_type="discovered",
        )
        defaults.update(kwargs)
        return CorrelationFinding(**defaults)

    @patch("data.agents.correlation_hunter._snapshot_graph_structure", return_value=None)
    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass", return_value=[])
    @patch("data.agents.correlation_hunter.hunt_correlations")
    def test_run_calls_persist(self, mock_hunt, mock_sens, mock_persist, mock_mat, mock_snap):
        """run() should call _persist_discoveries with the findings."""
        from data.agents.correlation_hunter import run

        findings = [self._make_finding(), self._make_finding(series_b="TLT")]
        mock_hunt.return_value = findings
        mock_persist.return_value = "run-id-123"
        mock_mat.return_value = None

        with patch("data.agents.correlation_hunter.log_evolution", create=True):
            with patch("db.supabase.client.get_client", return_value=_make_mock_client([{}])):
                result = run(symbols=["SPY", "GLD", "TLT"])

        mock_persist.assert_called_once_with(findings)
        assert len(result) == 2

    @patch("db.supabase.client.get_client")
    def test_persist_discoveries_converts_findings(self, mock_get_client):
        """_persist_discoveries should convert CorrelationFindings to DiscoveryRecords."""
        from data.agents.correlation_hunter import _persist_discoveries

        mock_client = _make_mock_client([{"id": "1"}, {"id": "2"}])
        mock_get_client.return_value = mock_client

        findings = [
            self._make_finding(series_a="A", series_b="B", pearson_r=0.8),
            self._make_finding(series_a="C", series_b="D", pearson_r=-0.6),
        ]
        run_id = _persist_discoveries(findings)

        assert run_id is not None
        payloads = mock_client.table().insert.call_args[0][0]
        assert len(payloads) == 2
        assert payloads[0]["series_a"] == "A"
        assert payloads[1]["series_a"] == "C"
        # All should share same run_id
        assert payloads[0]["run_id"] == payloads[1]["run_id"]

    def test_persist_empty_returns_none(self):
        """_persist_discoveries with empty list returns None without DB call."""
        from data.agents.correlation_hunter import _persist_discoveries
        assert _persist_discoveries([]) is None

    @patch("db.supabase.client.get_client")
    def test_persist_failure_returns_none(self, mock_get_client):
        """_persist_discoveries returns None on DB error without crashing."""
        from data.agents.correlation_hunter import _persist_discoveries

        mock_get_client.side_effect = Exception("connection refused")
        findings = [self._make_finding()]
        result = _persist_discoveries(findings)
        assert result is None

    @patch("db.supabase.client.get_client")
    def test_persist_preserves_timestamp(self, mock_get_client):
        """_persist_discoveries preserves the original finding timestamp."""
        from data.agents.correlation_hunter import _persist_discoveries

        mock_client = _make_mock_client([{"id": "1"}])
        mock_get_client.return_value = mock_client

        finding = self._make_finding()
        ts = finding.timestamp
        _persist_discoveries([finding])

        payloads = mock_client.table().insert.call_args[0][0]
        assert payloads[0]["computed_at"] == ts.isoformat()


# ── Discovery → Graph materialization integration ────────────────────────────

class TestDiscoveryGraphLoop:
    """Test that run() automatically materializes the graph after persistence."""

    def _make_finding(self, **kwargs):
        from data.agents.correlation_hunter import CorrelationFinding
        defaults = dict(
            series_a="SPY", series_b="GLD", lag_days=5,
            pearson_r=0.72, granger_p=0.03, mutual_info=0.1,
            strength="strong", relationship_type="discovered",
        )
        defaults.update(kwargs)
        return CorrelationFinding(**defaults)

    @patch("data.agents.correlation_hunter._snapshot_graph_structure", return_value=None)
    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass", return_value=[])
    @patch("data.agents.correlation_hunter.hunt_correlations")
    def test_run_calls_materialize_after_persist(self, mock_hunt, mock_sens, mock_persist, mock_mat, mock_snap):
        """run() should call _materialize_graph after successful persistence."""
        from data.agents.correlation_hunter import run

        findings = [self._make_finding()]
        mock_hunt.return_value = findings
        mock_persist.return_value = "run-id-abc"
        mock_mat.return_value = {"edges_merged": 5}

        with patch("db.supabase.client.log_evolution"):
            with patch("db.supabase.client.get_client", return_value=_make_mock_client([{}])):
                run(symbols=["SPY", "GLD"])

        mock_persist.assert_called_once_with(findings)
        mock_mat.assert_called_once_with("run-id-abc")

    @patch("data.agents.correlation_hunter._snapshot_graph_structure", return_value=None)
    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass", return_value=[])
    @patch("data.agents.correlation_hunter.hunt_correlations")
    def test_run_skips_materialize_when_persist_fails(self, mock_hunt, mock_sens, mock_persist, mock_mat, mock_snap):
        """run() should pass None to _materialize_graph when persistence fails."""
        from data.agents.correlation_hunter import run

        mock_hunt.return_value = [self._make_finding()]
        mock_persist.return_value = None  # persistence failed

        with patch("db.supabase.client.log_evolution"):
            with patch("db.supabase.client.get_client", return_value=_make_mock_client([{}])):
                run(symbols=["SPY", "GLD"])

        mock_mat.assert_called_once_with(None)

    def test_materialize_graph_skips_on_none_run_id(self):
        """_materialize_graph should return None when run_id is None."""
        from data.agents.correlation_hunter import _materialize_graph
        assert _materialize_graph(None) is None

    @patch("data.agents.graph_materializer.materialize")
    def test_materialize_graph_calls_materializer(self, mock_mat):
        """_materialize_graph should call materialize(min_strength='moderate')."""
        from data.agents.correlation_hunter import _materialize_graph

        mock_mat.return_value = {
            "asset_nodes_merged": 10,
            "macro_nodes_merged": 5,
            "edges_merged": 20,
            "correlated_with_edges": 18,
            "causes_edges": 2,
        }
        result = _materialize_graph("run-id-xyz")

        mock_mat.assert_called_once_with(min_strength="moderate")
        assert result["edges_merged"] == 20

    @patch("data.agents.graph_materializer.materialize")
    def test_materialize_graph_handles_failure_safely(self, mock_mat):
        """_materialize_graph should return None on failure, not raise."""
        from data.agents.correlation_hunter import _materialize_graph

        mock_mat.side_effect = Exception("Neo4j connection refused")
        result = _materialize_graph("run-id-xyz")
        assert result is None

    @patch("data.agents.correlation_hunter._snapshot_graph_structure", return_value=None)
    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass", return_value=[])
    @patch("data.agents.correlation_hunter.hunt_correlations")
    def test_graph_stats_included_in_evolution_log(self, mock_hunt, mock_sens, mock_persist, mock_mat, mock_snap):
        """run() should include graph stats in the evolution log when materialization succeeds."""
        from data.agents.correlation_hunter import run

        mock_hunt.return_value = [self._make_finding()]
        mock_persist.return_value = "run-id-log"
        mock_mat.return_value = {"edges_merged": 42, "asset_nodes_merged": 10}

        mock_log_evo = MagicMock()
        with patch("db.supabase.client.log_evolution", mock_log_evo):
            with patch("db.supabase.client.get_client", return_value=_make_mock_client([{}])):
                run(symbols=["SPY", "GLD"])

        # Check that the evolution log entry includes graph info
        call_args = mock_log_evo.call_args
        entry = call_args[0][0]
        assert "graph" in entry.after_state
        assert entry.after_state["graph"]["edges_merged"] == 42
