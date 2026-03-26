"""Tests for the Supabase client module.

All tests mock the Supabase network calls so they run without live credentials.
Integration against a real Supabase project requires SUPABASE_URL and SUPABASE_KEY
to be set in the environment.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(return_data: list[dict[str, Any]]) -> MagicMock:
    """Return a mock Supabase client whose table().op().execute() returns data.

    Args:
        return_data: The list of row dicts to return from .execute().

    Returns:
        A configured MagicMock that mimics the Supabase fluent API.
    """
    mock_result = MagicMock()
    mock_result.data = return_data

    mock_query = MagicMock()
    mock_query.execute.return_value = mock_result
    mock_query.eq.return_value = mock_query  # support chained .eq()

    mock_table = MagicMock()
    mock_table.insert.return_value = mock_query
    mock_table.upsert.return_value = mock_query
    mock_table.update.return_value = mock_query
    mock_table.select.return_value = mock_query

    mock_client = MagicMock()
    mock_client.table.return_value = mock_table
    return mock_client


# ---------------------------------------------------------------------------
# evolution_log
# ---------------------------------------------------------------------------

class TestEvolutionLog:
    """Tests for log_evolution()."""

    def test_writes_required_fields(self) -> None:
        """log_evolution inserts agent_id and action into evolution_log."""
        from db.supabase.client import EvolutionLogEntry, log_evolution

        row_id = str(uuid.uuid4())
        mock_client = _make_mock_client([{"id": row_id, "agent_id": "test_agent", "action": "ran"}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            result = log_evolution(EvolutionLogEntry(agent_id="test_agent", action="ran"))

        assert result["agent_id"] == "test_agent"
        assert result["action"] == "ran"
        mock_client.table.assert_called_once_with("evolution_log")

    def test_includes_optional_states(self) -> None:
        """log_evolution includes before_state and after_state when provided."""
        from db.supabase.client import EvolutionLogEntry, log_evolution

        mock_client = _make_mock_client([{"id": str(uuid.uuid4())}])
        before = {"accuracy": 0.54}
        after = {"accuracy": 0.61}

        with patch("db.supabase.client.get_client", return_value=mock_client):
            log_evolution(EvolutionLogEntry(
                agent_id="agent_x", action="retrain",
                before_state=before, after_state=after
            ))

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["before_state"] == before
        assert call_args["after_state"] == after


# ---------------------------------------------------------------------------
# roadmap
# ---------------------------------------------------------------------------

class TestRoadmap:
    """Tests for file_roadmap_task()."""

    def test_files_task_with_defaults(self) -> None:
        """file_roadmap_task inserts a task with default priority 3."""
        from db.supabase.client import RoadmapTask, file_roadmap_task

        mock_client = _make_mock_client([{"id": str(uuid.uuid4()), "priority": 3}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            result = file_roadmap_task(RoadmapTask(filed_by="master_architect", title="Retrain LSTM"))

        assert result["priority"] == 3
        mock_client.table.assert_called_once_with("roadmap")

    def test_backtest_gate_flag(self) -> None:
        """file_roadmap_task sets backtest_gate correctly."""
        from db.supabase.client import RoadmapTask, file_roadmap_task

        mock_client = _make_mock_client([{"id": str(uuid.uuid4())}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            file_roadmap_task(RoadmapTask(filed_by="agent", title="Deploy model", backtest_gate=True))

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["backtest_gate"] is True


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

class TestSignals:
    """Tests for write_signal()."""

    def test_writes_signal_fields(self) -> None:
        """write_signal persists asset, direction, confidence, asset_class."""
        from db.supabase.client import Signal, write_signal

        mock_client = _make_mock_client([{
            "id": str(uuid.uuid4()), "asset": "AAPL",
            "direction": "long", "confidence": 0.72
        }])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            result = write_signal(Signal(
                asset="AAPL", asset_class="equity",
                direction="long", confidence=0.72
            ))

        assert result["asset"] == "AAPL"
        assert result["confidence"] == 0.72

    def test_optional_model_id(self) -> None:
        """write_signal includes model_id in payload only when provided."""
        from db.supabase.client import Signal, write_signal

        model_id = str(uuid.uuid4())
        mock_client = _make_mock_client([{"id": str(uuid.uuid4())}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            write_signal(Signal(asset="BTC", asset_class="crypto",
                                direction="long", confidence=0.65, model_id=model_id))

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["model_id"] == model_id


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Tests for register_model()."""

    def test_upserts_model(self) -> None:
        """register_model calls upsert (not insert) on model_registry."""
        from db.supabase.client import ModelRegistryEntry, register_model

        mock_client = _make_mock_client([{"id": str(uuid.uuid4()), "name": "lstm_v1"}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            register_model(ModelRegistryEntry(
                name="lstm_v1", version="1.0.0",
                model_type="lstm", accuracy=0.61
            ))

        mock_client.table().upsert.assert_called_once()

    def test_last_trained_at_serialised(self) -> None:
        """register_model converts last_trained_at datetime to ISO string."""
        from db.supabase.client import ModelRegistryEntry, register_model

        now = datetime.now(timezone.utc)
        mock_client = _make_mock_client([{"id": str(uuid.uuid4())}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            register_model(ModelRegistryEntry(
                name="lstm_v1", version="1.0.1",
                model_type="lstm", last_trained_at=now
            ))

        call_args = mock_client.table().upsert.call_args[0][0]
        assert call_args["last_trained_at"] == now.isoformat()


# ---------------------------------------------------------------------------
# agent_runs
# ---------------------------------------------------------------------------

class TestAgentRuns:
    """Tests for start_agent_run() and end_agent_run()."""

    def test_start_inserts_running_status(self) -> None:
        """start_agent_run inserts a row with status='running'."""
        from db.supabase.client import AgentRun, start_agent_run

        mock_client = _make_mock_client([{"id": "abc", "status": "running"}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            run_id = start_agent_run(AgentRun(agent_name="noise_filter"))

        assert isinstance(run_id, str)
        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["status"] == "running"

    def test_end_sets_completed_status(self) -> None:
        """end_agent_run updates the row with status='completed' on success."""
        from db.supabase.client import AgentRun, end_agent_run

        mock_client = _make_mock_client([{"id": "abc"}])
        run = AgentRun(agent_name="noise_filter")

        with patch("db.supabase.client.get_client", return_value=mock_client):
            end_agent_run(run, success=True, summary="filtered 3 records")

        update_args = mock_client.table().update.call_args[0][0]
        assert update_args["status"] == "completed"
        assert update_args["result_summary"] == "filtered 3 records"

    def test_end_sets_failed_status(self) -> None:
        """end_agent_run sets status='failed' and records error message."""
        from db.supabase.client import AgentRun, end_agent_run

        mock_client = _make_mock_client([{"id": "abc"}])
        run = AgentRun(agent_name="noise_filter")

        with patch("db.supabase.client.get_client", return_value=mock_client):
            end_agent_run(run, success=False, error="connection refused")

        update_args = mock_client.table().update.call_args[0][0]
        assert update_args["status"] == "failed"
        assert update_args["error_message"] == "connection refused"


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------

class TestQuarantine:
    """Tests for quarantine_record()."""

    def test_quarantines_record(self) -> None:
        """quarantine_record inserts into quarantine table with reason."""
        from db.supabase.client import QuarantineRecord, quarantine_record

        mock_client = _make_mock_client([{"id": str(uuid.uuid4())}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            quarantine_record(QuarantineRecord(
                original_table="prices",
                data={"close": -999.0, "asset": "AAPL"},
                reason="negative_price",
                quarantined_by="noise_filter"
            ))

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["original_table"] == "prices"
        assert call_args["reason"] == "negative_price"
        assert call_args["quarantined_by"] == "noise_filter"


# ---------------------------------------------------------------------------
# system_health
# ---------------------------------------------------------------------------

class TestSystemHealth:
    """Tests for write_system_health()."""

    def test_flags_when_threshold_exceeded(self) -> None:
        """write_system_health sets flagged=True when value exceeds threshold."""
        from db.supabase.client import write_system_health

        mock_client = _make_mock_client([{"id": str(uuid.uuid4()), "flagged": True}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            write_system_health("query_latency_ms", 620.0, threshold=500.0, source="timescale")

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["flagged"] is True
        assert call_args["threshold"] == 500.0

    def test_not_flagged_when_below_threshold(self) -> None:
        """write_system_health sets flagged=False when value is within threshold."""
        from db.supabase.client import write_system_health

        mock_client = _make_mock_client([{"id": str(uuid.uuid4()), "flagged": False}])

        with patch("db.supabase.client.get_client", return_value=mock_client):
            write_system_health("query_latency_ms", 120.0, threshold=500.0)

        call_args = mock_client.table().insert.call_args[0][0]
        assert call_args["flagged"] is False


# ---------------------------------------------------------------------------
# client singleton
# ---------------------------------------------------------------------------

class TestClientSingleton:
    """Tests for get_client() lazy initialisation."""

    def test_raises_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_client raises RuntimeError when SUPABASE_URL is not set."""
        import sys
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)

        # Reset cached client and env module
        import db.supabase.client as mod
        mod._client = None
        for key in list(sys.modules.keys()):
            if key in ("skills.env",):
                del sys.modules[key]

        with pytest.raises(RuntimeError):
            mod.get_client()
