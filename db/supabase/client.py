"""Typed Supabase client wrapper for FinBrain.

All database interactions go through this module.
Never import supabase directly in agents — use get_client() from here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from skills.env import get_supabase_key, get_supabase_url
from skills.logger import get_logger

logger = get_logger(__name__)

_client: Client | None = None


def get_client() -> Client:
    """Return a cached Supabase client instance.

    Lazily initialises the client on first call and reuses it
    across all subsequent calls within the same process.

    Returns:
        An authenticated Supabase Client.
    """
    global _client
    if _client is None:
        url = get_supabase_url()
        key = get_supabase_key()
        _client = create_client(url, key)
        logger.info("supabase_client_initialised", url=url[:40] + "...")
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses matching each table's insert shape
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvolutionLogEntry:
    """Represents one row written to the evolution_log table."""
    agent_id: str
    action: str
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RoadmapTask:
    """Represents one row written to the roadmap table."""
    filed_by: str
    title: str
    description: str = ""
    priority: int = 3
    backtest_gate: bool = False


@dataclass
class Signal:
    """Represents one trading signal row in the signals table."""
    asset: str
    asset_class: str
    direction: str
    confidence: float
    model_id: str | None = None
    features: dict[str, Any] | None = None


@dataclass
class ModelRegistryEntry:
    """Represents one model version in model_registry."""
    name: str
    version: str
    model_type: str
    asset_class: str | None = None
    accuracy: float | None = None
    val_loss: float | None = None
    train_samples: int | None = None
    hyperparams: dict[str, Any] | None = None
    artifact_path: str | None = None
    status: str = "staging"
    last_trained_at: datetime | None = None


@dataclass
class AgentRun:
    """Tracks the lifecycle of one agent execution."""
    agent_name: str
    trigger: str = "manual"
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _start_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_start_payload(self) -> dict[str, Any]:
        """Return the INSERT payload for starting a run."""
        return {
            "id": self._id,
            "agent_name": self.agent_name,
            "trigger": self.trigger,
            "status": "running",
        }

    def to_end_payload(self, *, success: bool, summary: str = "", error: str = "") -> dict[str, Any]:
        """Return the UPDATE payload for completing or failing a run.

        Args:
            success: True if the agent completed without error.
            summary: Human-readable summary of what was done.
            error: Error message if the run failed.

        Returns:
            Dict suitable for a Supabase update call.
        """
        end_ms = int(time.time() * 1000)
        return {
            "status": "completed" if success else "failed",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": end_ms - self._start_ms,
            "result_summary": summary,
            "error_message": error or None,
        }


@dataclass
class QuarantineRecord:
    """Represents one quarantined data record."""
    original_table: str
    data: dict[str, Any]
    reason: str
    quarantined_by: str
    original_id: str | None = None


@dataclass
class DiscoveryRecord:
    """One discovered statistical relationship to persist in the discoveries table.

    Attributes:
        series_a:          First series (potential cause or co-mover).
        series_b:          Second series (potential effect).
        lag_days:          Lead/lag in trading days (positive = A leads B).
        pearson_r:         Pearson correlation at best lag.
        granger_p:         p-value of Granger causality test (A causes B), or None.
        mutual_info:       Mutual information in bits, or None.
        regime:            Market regime: 'all', 'bull', or 'bear'.
        strength:          'strong', 'moderate', or 'weak'.
        relationship_type: Known type from universe or 'discovered'.
        run_id:            Groups findings from the same discovery run.
        computed_at:       When the discovery was computed.
    """
    series_a:          str
    series_b:          str
    lag_days:          int
    pearson_r:         float
    granger_p:         float | None       = None
    mutual_info:       float | None       = None
    regime:            str                = "all"
    strength:          str                = "moderate"
    relationship_type: str                = "discovered"
    run_id:            str                = field(default_factory=lambda: str(uuid.uuid4()))
    computed_at:       datetime           = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_payload(self) -> dict[str, Any]:
        """Convert to a Supabase insert payload.

        Returns:
            Dict suitable for a Supabase table insert.
        """
        payload: dict[str, Any] = {
            "series_a":          self.series_a,
            "series_b":          self.series_b,
            "lag_days":          self.lag_days,
            "pearson_r":         round(self.pearson_r, 6),
            "regime":            self.regime,
            "strength":          self.strength,
            "relationship_type": self.relationship_type,
            "run_id":            self.run_id,
            "computed_at":       self.computed_at.isoformat(),
        }
        if self.granger_p is not None:
            payload["granger_p"] = round(self.granger_p, 6)
        if self.mutual_info is not None:
            payload["mutual_info"] = round(self.mutual_info, 6)
        return payload


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers — one per table
# ─────────────────────────────────────────────────────────────────────────────

def log_evolution(entry: EvolutionLogEntry) -> dict[str, Any]:
    """Insert one row into evolution_log and return the inserted record.

    Args:
        entry: The EvolutionLogEntry to persist.

    Returns:
        The row as returned by Supabase after insert.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "agent_id": entry.agent_id,
        "action": entry.action,
    }
    if entry.before_state is not None:
        payload["before_state"] = entry.before_state
    if entry.after_state is not None:
        payload["after_state"] = entry.after_state
    if entry.metadata is not None:
        payload["metadata"] = entry.metadata

    result = client.table("evolution_log").insert(payload).execute()
    logger.info("evolution_log_written", agent_id=entry.agent_id, action=entry.action)
    return result.data[0]


def file_roadmap_task(task: RoadmapTask) -> dict[str, Any]:
    """Insert one task into the roadmap table and return the inserted record.

    Args:
        task: The RoadmapTask to file.

    Returns:
        The row as returned by Supabase after insert.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "filed_by": task.filed_by,
        "title": task.title,
        "description": task.description,
        "priority": task.priority,
        "backtest_gate": task.backtest_gate,
    }
    result = client.table("roadmap").insert(payload).execute()
    logger.info("roadmap_task_filed", filed_by=task.filed_by, title=task.title)
    return result.data[0]


def write_signal(signal: Signal) -> dict[str, Any]:
    """Insert one signal into the signals table and return the inserted record.

    Args:
        signal: The Signal to persist.

    Returns:
        The row as returned by Supabase after insert.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "asset": signal.asset,
        "asset_class": signal.asset_class,
        "direction": signal.direction,
        "confidence": signal.confidence,
    }
    if signal.model_id is not None:
        payload["model_id"] = signal.model_id
    if signal.features is not None:
        payload["features"] = signal.features

    result = client.table("signals").insert(payload).execute()
    logger.info("signal_written", asset=signal.asset, direction=signal.direction)
    return result.data[0]


def register_model(entry: ModelRegistryEntry) -> dict[str, Any]:
    """Upsert a model version into model_registry and return the record.

    Uses upsert so re-runs with the same (name, version) update in place.

    Args:
        entry: The ModelRegistryEntry to persist.

    Returns:
        The row as returned by Supabase after upsert.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "name": entry.name,
        "version": entry.version,
        "model_type": entry.model_type,
        "status": entry.status,
    }
    for attr in ("asset_class", "accuracy", "val_loss", "train_samples",
                 "hyperparams", "artifact_path"):
        val = getattr(entry, attr)
        if val is not None:
            payload[attr] = val
    if entry.last_trained_at is not None:
        payload["last_trained_at"] = entry.last_trained_at.isoformat()

    result = client.table("model_registry").upsert(payload, on_conflict="name,version").execute()
    logger.info("model_registered", name=entry.name, version=entry.version)
    return result.data[0]


def start_agent_run(run: AgentRun) -> str:
    """Insert the start record for an agent run and return the run ID.

    Args:
        run: The AgentRun being started.

    Returns:
        The UUID string of the inserted run row.
    """
    client = get_client()
    client.table("agent_runs").insert(run.to_start_payload()).execute()
    logger.info("agent_run_started", agent=run.agent_name, run_id=run._id)
    return run._id


def end_agent_run(run: AgentRun, *, success: bool, summary: str = "", error: str = "") -> None:
    """Update the agent_runs row to mark the run as completed or failed.

    Args:
        run: The AgentRun that just finished.
        success: Whether the run completed without error.
        summary: Human-readable result summary.
        error: Error message if failed.
    """
    client = get_client()
    client.table("agent_runs").update(
        run.to_end_payload(success=success, summary=summary, error=error)
    ).eq("id", run._id).execute()
    logger.info("agent_run_ended", agent=run.agent_name, success=success)


def quarantine_record(record: QuarantineRecord) -> dict[str, Any]:
    """Move a bad record into the quarantine table. Never deletes source data.

    Args:
        record: The QuarantineRecord to persist.

    Returns:
        The row as returned by Supabase after insert.
    """
    client = get_client()
    payload: dict[str, Any] = {
        "original_table": record.original_table,
        "data": record.data,
        "reason": record.reason,
        "quarantined_by": record.quarantined_by,
    }
    if record.original_id is not None:
        payload["original_id"] = record.original_id

    result = client.table("quarantine").insert(payload).execute()
    logger.info("record_quarantined", table=record.original_table, reason=record.reason)
    return result.data[0]


def write_system_health(metric: str, value: float, threshold: float | None = None,
                        source: str | None = None) -> dict[str, Any]:
    """Insert one system health metric reading into system_health.

    Args:
        metric: The metric name (e.g. 'query_latency_ms').
        value: The measured numeric value.
        threshold: Optional threshold to compare against; flags if value exceeds it.
        source: Optional label for the component being measured.

    Returns:
        The row as returned by Supabase after insert.
    """
    client = get_client()
    flagged = (threshold is not None and value > threshold)
    payload: dict[str, Any] = {
        "metric": metric,
        "value": value,
        "flagged": flagged,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    if source is not None:
        payload["source"] = source

    result = client.table("system_health").insert(payload).execute()
    if flagged:
        logger.warning("system_health_threshold_exceeded", metric=metric, value=value,
                       threshold=threshold)
    return result.data[0]


# ─────────────────────────────────────────────────────────────────────────────
# Discovery persistence — correlation findings
# ─────────────────────────────────────────────────────────────────────────────

def save_discoveries(records: list[DiscoveryRecord]) -> list[dict[str, Any]]:
    """Bulk-insert discovery records into the discoveries table.

    Args:
        records: List of DiscoveryRecord objects to persist.

    Returns:
        List of inserted rows as returned by Supabase.
    """
    if not records:
        return []
    client = get_client()
    payloads = [r.to_payload() for r in records]
    result = client.table("discoveries").insert(payloads).execute()
    logger.info("discoveries_saved", count=len(records), run_id=records[0].run_id)
    return result.data


def get_structural_snapshots(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent structural analysis snapshots from evolution_log.

    Queries for entries with agent_id='graph_analyzer' and
    action='structural_snapshot', ordered most-recent first.

    Args:
        limit: Maximum number of snapshots to return.

    Returns:
        List of evolution_log row dicts, newest first.
    """
    client = get_client()
    result = (
        client.table("evolution_log")
        .select("*")
        .eq("agent_id", "graph_analyzer")
        .eq("action", "structural_snapshot")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def get_discoveries(
    *,
    series: str | None = None,
    strength: str | None = None,
    run_id: str | None = None,
    min_abs_r: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query persisted discoveries with optional filters.

    Args:
        series:    Filter to findings involving this series (as A or B).
        strength:  Filter by strength level ('strong', 'moderate', 'weak').
        run_id:    Filter to a specific discovery run.
        min_abs_r: Minimum |pearson_r| to include.
        limit:     Maximum rows to return (default 100).

    Returns:
        List of discovery row dicts, ordered by |pearson_r| descending.
    """
    client = get_client()
    query = client.table("discoveries").select("*")

    if strength is not None:
        query = query.eq("strength", strength)
    if run_id is not None:
        query = query.eq("run_id", run_id)

    query = query.order("pearson_r", desc=True).limit(limit)
    result = query.execute()
    rows = result.data

    # Client-side filters that Supabase PostgREST doesn't handle neatly
    if series is not None:
        rows = [r for r in rows if r.get("series_a") == series or r.get("series_b") == series]
    if min_abs_r is not None:
        rows = [r for r in rows if abs(r.get("pearson_r", 0)) >= min_abs_r]

    return rows


def persist_intelligence_report(report: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Persist an intelligence report to the evolution_log.

    Stores the full report as after_state so it can be retrieved later
    for history, comparison, and audit purposes.

    Args:
        report: The intelligence report dict from build_intelligence_report().
        run_id: Optional discovery run_id to link this report to.

    Returns:
        The created evolution_log row.
    """
    entry = EvolutionLogEntry(
        agent_id="graph_intelligence",
        action="intelligence_report",
        after_state={
            "run_id": run_id or report.get("run_id"),
            "insight_count": len(report.get("insights", [])),
            "reasoning": report.get("reasoning", {}),
            "metrics": report.get("metrics", {}),
            "confidence": report.get("confidence"),
            "anomaly_count": len(report.get("anomalies", {}).get("anomalies", [])),
        },
    )
    return log_evolution(entry)


def get_intelligence_reports(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent intelligence reports from evolution_log.

    Args:
        limit: Maximum number of reports to return.

    Returns:
        List of evolution_log row dicts, newest first.
    """
    client = get_client()
    result = (
        client.table("evolution_log")
        .select("*")
        .eq("agent_id", "graph_intelligence")
        .eq("action", "intelligence_report")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data
