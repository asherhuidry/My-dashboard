"""Experiment registry API routes.

Exposes read-only visibility into training experiment history, metrics,
and backtest results stored in the ExperimentRegistry.

Routes
------
GET  /api/experiments              — list all experiments (filterable)
GET  /api/experiments/summary      — aggregate counts by status/model
GET  /api/experiments/best         — best experiment for a given metric
GET  /api/experiments/{id}         — full detail for one experiment
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ml.registry.experiment_registry import ExperimentRegistry, ExperimentStatus

router = APIRouter()
log    = logging.getLogger(__name__)

# Shared registry instance — loaded once per process from the default path.
# Tests can override by patching `api.routes.experiments._registry`.
_registry = ExperimentRegistry()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialise_experiment(rec) -> dict[str, Any]:
    """Convert an ExperimentRecord to a JSON-safe dict."""
    d = rec.to_dict()
    # Ensure backtest sub-dict is included cleanly
    if d.get("backtest") is None:
        d["backtest"] = None
    return d


def _get_registry() -> ExperimentRegistry:
    """Return the module-level registry (injectable for tests)."""
    return _registry


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/experiments")
def list_experiments(
    model_type:  str | None = Query(None, description="Filter by model family, e.g. 'mlp'"),
    status:      str | None = Query(None, description="Filter by status, e.g. 'completed'"),
    tag:         str | None = Query(None, description="Filter by tag"),
    has_backtest: bool | None = Query(None, description="True = only with backtest attached"),
    limit:       int         = Query(50, ge=1, le=500, description="Max records to return"),
) -> dict[str, Any]:
    """List experiments with optional filtering.

    Returns the most recent experiments first, up to ``limit`` records.
    """
    reg = _get_registry()
    status_enum = None
    if status is not None:
        try:
            status_enum = ExperimentStatus(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown status {status!r}. "
                       f"Valid values: {[s.value for s in ExperimentStatus]}",
            )

    records = reg.filter(
        model_type   = model_type,
        status       = status_enum,
        tag          = tag,
        has_backtest = has_backtest,
    )[:limit]

    return {
        "total":       len(records),
        "experiments": [_serialise_experiment(r) for r in records],
    }


@router.get("/experiments/summary")
def get_experiments_summary() -> dict[str, Any]:
    """Return aggregate statistics about the experiment registry."""
    reg = _get_registry()
    return reg.summary()


@router.get("/experiments/best")
def get_best_experiment(
    metric:           str  = Query("accuracy", description="Metric to rank by"),
    model_type:       str | None = Query(None, description="Filter by model family"),
    higher_is_better: bool = Query(True, description="True = higher metric is better"),
) -> dict[str, Any]:
    """Return the single best experiment for a given metric.

    Only COMPLETED and PROMOTED experiments are considered.
    Returns 404 if no experiment with the requested metric exists.
    """
    reg  = _get_registry()
    best = reg.best(metric, model_type=model_type, higher_is_better=higher_is_better)
    if best is None:
        raise HTTPException(
            status_code=404,
            detail=f"No completed/promoted experiment found with metric {metric!r}.",
        )
    return _serialise_experiment(best)


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Return full detail for a single experiment by ID.

    Returns 404 if the experiment does not exist.
    """
    reg = _get_registry()
    try:
        rec = reg.get(experiment_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id!r} not found.",
        )
    return _serialise_experiment(rec)
