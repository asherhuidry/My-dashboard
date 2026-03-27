"""Experiment registry for tracking ML training runs and their results.

Every model training run is recorded here with its hyperparameters, dataset
info, evaluation metrics, and optional backtest results.  The registry is
local-first (JSON file) so it works without a database connection.

Usage::

    from ml.registry import ExperimentRegistry, ExperimentRecord

    reg = ExperimentRegistry()
    exp = reg.create(
        name        = "mlp_baseline_v1",
        model_type  = "mlp",
        hyperparams = {"hidden": [128, 64], "lr": 0.001, "epochs": 50},
        dataset_info= {"symbols": ["AAPL", "MSFT"], "start": "2020-01-01"},
        notes       = "First baseline MLP run",
    )
    # ... train model ...
    reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.62, "val_loss": 0.58})
    reg.attach_backtest(exp.experiment_id, result)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "registry" / "experiments.json"
REGISTRY_PATH = Path(os.getenv("FINBRAIN_EXPERIMENT_REGISTRY_PATH", str(_DEFAULT_PATH)))


# ── Status ────────────────────────────────────────────────────────────────────

class ExperimentStatus(str, Enum):
    """Lifecycle of an experiment."""
    RUNNING   = "running"    # Training is in progress
    COMPLETED = "completed"  # Training finished; metrics recorded
    FAILED    = "failed"     # Training raised an unhandled exception
    PROMOTED  = "promoted"   # Passed backtest gate; promoted to production
    ARCHIVED  = "archived"   # No longer active; kept for reference


# ── BacktestResult ────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Summary statistics from a backtest run.

    Attributes:
        cumulative_return: Total return over the test period (e.g. 0.18 = 18%).
        annualised_return: Annualised equivalent return.
        hit_rate:          Fraction of trades with correct direction (0–1).
        max_drawdown:      Maximum peak-to-trough loss (negative, e.g. -0.12).
        sharpe:            Sharpe-like ratio (return / volatility of daily returns).
        trade_count:       Number of trades executed.
        benchmark_return:  Buy-and-hold benchmark return over same period.
        period_start:      ISO-8601 start date of the backtest window.
        period_end:        ISO-8601 end date of the backtest window.
        extra:             Any additional metrics (e.g. per-symbol breakdown).
    """
    cumulative_return: float
    annualised_return: float
    hit_rate:          float
    max_drawdown:      float
    sharpe:            float
    trade_count:       int
    benchmark_return:  float
    period_start:      str
    period_end:        str
    extra:             dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestResult":
        """Reconstruct from a dictionary."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── ExperimentRecord ──────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    """One ML training run.

    Attributes:
        experiment_id: Unique UUID for this run.
        name:          Human-readable name (e.g. 'mlp_baseline_v1').
        model_type:    Short model family tag (e.g. 'mlp', 'lstm', 'gnn').
        status:        Current lifecycle status.
        hyperparams:   Dict of hyperparameter name → value.
        dataset_info:  Dict describing the training dataset.
        metrics:       Dict of evaluation metric name → value (set on finish).
        backtest:      Attached BacktestResult, or None.
        notes:         Free-text notes.
        checkpoint_path: Path to the saved model checkpoint (empty if none).
        started_at:    ISO-8601 timestamp when the run began.
        finished_at:   ISO-8601 timestamp when the run ended (None if running).
        tags:          Optional list of string tags for filtering.
    """
    experiment_id:   str
    name:            str
    model_type:      str
    status:          ExperimentStatus
    hyperparams:     dict[str, Any]     = field(default_factory=dict)
    dataset_info:    dict[str, Any]     = field(default_factory=dict)
    metrics:         dict[str, float]   = field(default_factory=dict)
    backtest:        BacktestResult | None = None
    notes:           str                = ""
    checkpoint_path: str                = ""
    started_at:      str                = ""
    finished_at:     str | None         = None
    tags:            list[str]          = field(default_factory=list)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Return True if the experiment is currently in progress."""
        return self.status == ExperimentStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        """Return wall-clock duration in seconds, or None if not finished."""
        if not self.finished_at or not self.started_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end   = datetime.fromisoformat(self.finished_at)
        return (end - start).total_seconds()

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        d = asdict(self)
        d["status"]   = self.status.value
        d["backtest"] = self.backtest.to_dict() if self.backtest else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentRecord":
        """Reconstruct an ExperimentRecord from a dictionary."""
        d = dict(d)
        d["status"] = ExperimentStatus(d["status"])
        if d.get("backtest"):
            d["backtest"] = BacktestResult.from_dict(d["backtest"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ── ExperimentRegistry ────────────────────────────────────────────────────────

class ExperimentRegistry:
    """Local-first experiment registry backed by a single JSON file.

    Args:
        path: Path to the JSON registry file.  Defaults to
              ``data/registry/experiments.json`` (or ``FINBRAIN_EXPERIMENT_REGISTRY_PATH``).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = Path(path) if path else REGISTRY_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ExperimentRecord] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, ExperimentRecord]:
        """Load all records from disk."""
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                e["experiment_id"]: ExperimentRecord.from_dict(e)
                for e in raw.get("experiments", [])
            }
        except Exception as exc:
            log.warning("Could not load experiment registry: %s", exc)
            return {}

    def _save(self) -> None:
        """Persist all records to disk."""
        payload = {
            "version":    "1",
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "experiments": [r.to_dict() for r in self._records.values()],
        }
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def create(
        self,
        name:         str,
        model_type:   str,
        hyperparams:  dict[str, Any]   | None = None,
        dataset_info: dict[str, Any]   | None = None,
        notes:        str              = "",
        tags:         list[str]        | None = None,
    ) -> ExperimentRecord:
        """Create a new experiment record in RUNNING state.

        Args:
            name:         Human-readable name for this run.
            model_type:   Short model family tag (e.g. 'mlp', 'lstm').
            hyperparams:  Dict of hyperparameter values.
            dataset_info: Dict describing training data (symbols, date range, etc.).
            notes:        Free-text notes.
            tags:         Optional list of string tags.

        Returns:
            The newly created ExperimentRecord.
        """
        exp_id = str(uuid.uuid4())
        rec = ExperimentRecord(
            experiment_id = exp_id,
            name          = name,
            model_type    = model_type,
            status        = ExperimentStatus.RUNNING,
            hyperparams   = hyperparams or {},
            dataset_info  = dataset_info or {},
            notes         = notes,
            tags          = tags or [],
            started_at    = datetime.now(tz=timezone.utc).isoformat(),
        )
        self._records[exp_id] = rec
        self._save()
        log.info("Experiment started: %s (%s)", name, exp_id[:8])
        return rec

    def get(self, experiment_id: str) -> ExperimentRecord:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: Full UUID string.

        Raises:
            KeyError: If the experiment does not exist.
        """
        if experiment_id not in self._records:
            raise KeyError(f"Experiment not found: {experiment_id!r}")
        return self._records[experiment_id]

    def finish(
        self,
        experiment_id:   str,
        metrics:         dict[str, float] | None = None,
        checkpoint_path: str = "",
        failed:          bool = False,
        failure_reason:  str = "",
    ) -> ExperimentRecord:
        """Mark an experiment as COMPLETED or FAILED and record metrics.

        Args:
            experiment_id:   Experiment UUID.
            metrics:         Dict of metric name → value (e.g. ``{'val_accuracy': 0.62}``).
            checkpoint_path: File path to the saved model checkpoint.
            failed:          If True, mark as FAILED instead of COMPLETED.
            failure_reason:  Human-readable reason for failure (appended to notes).

        Returns:
            Updated ExperimentRecord.

        Raises:
            KeyError: If the experiment does not exist.
        """
        rec = self.get(experiment_id)
        rec.status        = ExperimentStatus.FAILED if failed else ExperimentStatus.COMPLETED
        rec.finished_at   = datetime.now(tz=timezone.utc).isoformat()
        rec.metrics       = metrics or {}
        rec.checkpoint_path = checkpoint_path
        if failure_reason:
            rec.notes = f"{rec.notes}\nFAILURE: {failure_reason}".strip()
        self._save()
        log.info(
            "Experiment %s: %s (metrics=%s)",
            rec.name, rec.status.value, metrics,
        )
        return rec

    def attach_backtest(
        self,
        experiment_id: str,
        result:        BacktestResult,
    ) -> ExperimentRecord:
        """Attach a BacktestResult to an experiment.

        Args:
            experiment_id: Experiment UUID.
            result:        BacktestResult to attach.

        Returns:
            Updated ExperimentRecord.
        """
        rec = self.get(experiment_id)
        rec.backtest = result
        self._save()
        log.info(
            "Backtest attached to %s: return=%.2f%% hit_rate=%.2f",
            rec.name, result.cumulative_return * 100, result.hit_rate,
        )
        return rec

    def promote(
        self,
        experiment_id: str,
        notes:         str = "",
    ) -> ExperimentRecord:
        """Promote a COMPLETED experiment to PROMOTED status.

        Only experiments with an attached BacktestResult can be promoted.

        Args:
            experiment_id: Experiment UUID.
            notes:         Promotion notes (optional).

        Raises:
            ValueError: If the experiment is not COMPLETED or has no backtest.
        """
        rec = self.get(experiment_id)
        if rec.status != ExperimentStatus.COMPLETED:
            raise ValueError(
                f"Can only promote COMPLETED experiments; current status: {rec.status.value}"
            )
        if rec.backtest is None:
            raise ValueError(
                "Experiment must have an attached BacktestResult before promotion. "
                "Call attach_backtest() first."
            )
        rec.status = ExperimentStatus.PROMOTED
        if notes:
            rec.notes = f"{rec.notes}\nPROMOTED: {notes}".strip()
        self._save()
        log.info("Experiment promoted: %s", rec.name)
        return rec

    def archive(self, experiment_id: str) -> ExperimentRecord:
        """Archive an experiment (mark as ARCHIVED).

        Args:
            experiment_id: Experiment UUID.

        Returns:
            Updated ExperimentRecord.
        """
        rec = self.get(experiment_id)
        rec.status = ExperimentStatus.ARCHIVED
        self._save()
        return rec

    def update_notes(self, experiment_id: str, notes: str) -> ExperimentRecord:
        """Append text to an experiment's notes field.

        Args:
            experiment_id: Experiment UUID.
            notes:         Text to append.

        Returns:
            Updated ExperimentRecord.
        """
        rec = self.get(experiment_id)
        rec.notes = f"{rec.notes}\n{notes}".strip()
        self._save()
        return rec

    # ── Query ─────────────────────────────────────────────────────────────────

    def all(self) -> list[ExperimentRecord]:
        """Return all experiments, most recent first."""
        return sorted(
            self._records.values(),
            key=lambda r: r.started_at,
            reverse=True,
        )

    def filter(
        self,
        model_type:  str | None              = None,
        status:      ExperimentStatus | None = None,
        tag:         str | None              = None,
        has_backtest: bool | None            = None,
    ) -> list[ExperimentRecord]:
        """Filter experiments by one or more criteria.

        Args:
            model_type:   Filter by model family.
            status:       Filter by lifecycle status.
            tag:          Filter to experiments that have this tag.
            has_backtest: True = only with backtest; False = only without.

        Returns:
            Matching records, most recent first.
        """
        results = list(self._records.values())
        if model_type is not None:
            results = [r for r in results if r.model_type == model_type]
        if status is not None:
            results = [r for r in results if r.status == status]
        if tag is not None:
            results = [r for r in results if tag in r.tags]
        if has_backtest is not None:
            results = [r for r in results if (r.backtest is not None) == has_backtest]
        return sorted(results, key=lambda r: r.started_at, reverse=True)

    def best(
        self,
        metric:     str,
        model_type: str | None = None,
        higher_is_better: bool = True,
    ) -> ExperimentRecord | None:
        """Return the experiment with the best value for a given metric.

        Args:
            metric:           Metric key to compare (must exist in record.metrics).
            model_type:       Optional filter by model family.
            higher_is_better: If True, return the record with the highest value.

        Returns:
            Best ExperimentRecord, or None if no records have the metric.
        """
        candidates = self.filter(
            model_type=model_type,
            status=ExperimentStatus.COMPLETED,
        ) + self.filter(
            model_type=model_type,
            status=ExperimentStatus.PROMOTED,
        )
        candidates = [r for r in candidates if metric in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics[metric]) \
            if higher_is_better else \
            min(candidates, key=lambda r: r.metrics[metric])

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of the registry."""
        records = list(self._records.values())
        by_status: dict[str, int] = {}
        for r in records:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        by_model: dict[str, int] = {}
        for r in records:
            by_model[r.model_type] = by_model.get(r.model_type, 0) + 1
        return {
            "total":      len(records),
            "by_status":  by_status,
            "by_model":   by_model,
            "promoted":   by_status.get("promoted", 0),
            "with_backtest": sum(1 for r in records if r.backtest is not None),
        }

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, experiment_id: str) -> bool:
        return experiment_id in self._records
