"""Tests for /api/experiments routes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ml.registry.experiment_registry import (
    BacktestResult,
    ExperimentRegistry,
    ExperimentStatus,
)


# ── Setup helpers ──────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(path=tmp_path / "experiments.json")


def _make_backtest(**overrides) -> BacktestResult:
    defaults = dict(
        cumulative_return = 0.15,
        annualised_return = 0.10,
        hit_rate          = 0.58,
        max_drawdown      = -0.08,
        sharpe            = 1.1,
        trade_count       = 80,
        benchmark_return  = 0.12,
        period_start      = "2024-01-01",
        period_end        = "2024-12-31",
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _populated_registry(tmp_path: Path) -> ExperimentRegistry:
    """Return a registry with 4 experiments in various states."""
    reg = _make_registry(tmp_path)

    e1 = reg.create("mlp_v1", "mlp", tags=["baseline"])
    reg.finish(e1.experiment_id, metrics={"accuracy": 0.60, "f1": 0.59})
    reg.attach_backtest(e1.experiment_id, _make_backtest(cumulative_return=0.15))
    reg.promote(e1.experiment_id)

    e2 = reg.create("mlp_v2", "mlp", tags=["baseline", "wider"])
    reg.finish(e2.experiment_id, metrics={"accuracy": 0.63, "f1": 0.62})

    e3 = reg.create("lstm_v1", "lstm")
    reg.finish(e3.experiment_id, failed=True)

    e4 = reg.create("mlp_v3", "mlp")
    # still running

    return reg


def _make_client(registry: ExperimentRegistry) -> TestClient:
    """Build a TestClient with the given registry injected."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import api.routes.experiments as mod

    # Patch the module-level registry
    mod._registry = registry

    from api.routes.experiments import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/experiments
# ══════════════════════════════════════════════════════════════════════════════

class TestListExperiments:
    def test_returns_200(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments")
        assert r.status_code == 200

    def test_returns_all(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments")
        assert r.json()["total"] == 4

    def test_filter_by_model_type(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"model_type": "mlp"})
        data = r.json()
        assert data["total"] == 3
        for exp in data["experiments"]:
            assert exp["model_type"] == "mlp"

    def test_filter_by_status(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"status": "completed"})
        data = r.json()
        assert data["total"] == 1
        assert data["experiments"][0]["status"] == "completed"

    def test_filter_by_tag(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"tag": "baseline"})
        assert r.json()["total"] == 2

    def test_filter_has_backtest_true(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"has_backtest": True})
        data = r.json()
        assert data["total"] == 1
        assert data["experiments"][0]["backtest"] is not None

    def test_filter_has_backtest_false(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"has_backtest": False})
        assert r.json()["total"] == 3

    def test_invalid_status_returns_422(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"status": "invalid_status"})
        assert r.status_code == 422

    def test_limit_respected(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"limit": 2})
        assert len(r.json()["experiments"]) == 2

    def test_empty_registry_returns_zero(self, tmp_path):
        reg    = _make_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments")
        assert r.json()["total"] == 0
        assert r.json()["experiments"] == []

    def test_backtest_field_included(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments", params={"status": "promoted"})
        exp = r.json()["experiments"][0]
        assert "backtest" in exp
        assert exp["backtest"]["hit_rate"] == 0.58


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/experiments/summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSummary:
    def test_returns_200(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        assert r.status_code == 200

    def test_total_correct(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        assert r.json()["total"] == 4

    def test_promoted_count(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        assert r.json()["promoted"] == 1

    def test_with_backtest_count(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        assert r.json()["with_backtest"] == 1

    def test_by_status_breakdown(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        by_status = r.json()["by_status"]
        assert "promoted"  in by_status
        assert "completed" in by_status
        assert "failed"    in by_status
        assert "running"   in by_status

    def test_empty_registry(self, tmp_path):
        reg    = _make_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/summary")
        assert r.json()["total"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/experiments/best
# ══════════════════════════════════════════════════════════════════════════════

class TestGetBest:
    def test_returns_200_with_completed(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/best", params={"metric": "accuracy"})
        assert r.status_code == 200

    def test_best_accuracy(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/best", params={"metric": "accuracy"})
        exp = r.json()
        # mlp_v2 has accuracy=0.63, mlp_v1 has 0.60 (but is promoted)
        assert exp["metrics"]["accuracy"] >= 0.60

    def test_best_lower_is_better(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/best",
                       params={"metric": "accuracy", "higher_is_better": False})
        # Should return the one with lowest accuracy
        assert r.status_code == 200

    def test_filter_by_model_type(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/best",
                       params={"metric": "accuracy", "model_type": "mlp"})
        assert r.json()["model_type"] == "mlp"

    def test_no_metric_returns_404(self, tmp_path):
        reg    = _make_registry(tmp_path)  # empty
        client = _make_client(reg)
        r = client.get("/api/experiments/best", params={"metric": "nonexistent_metric"})
        assert r.status_code == 404

    def test_missing_metric_in_all_returns_404(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/best", params={"metric": "sharpe_ratio_xyz"})
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/experiments/{id}
# ══════════════════════════════════════════════════════════════════════════════

class TestGetExperiment:
    def test_returns_200_for_existing(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        exp_id = reg.all()[0].experiment_id
        r = client.get(f"/api/experiments/{exp_id}")
        assert r.status_code == 200

    def test_returns_correct_experiment(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        target = reg.all()[0]
        r = client.get(f"/api/experiments/{target.experiment_id}")
        assert r.json()["experiment_id"] == target.experiment_id

    def test_returns_404_for_unknown(self, tmp_path):
        reg    = _make_registry(tmp_path)
        client = _make_client(reg)
        r = client.get("/api/experiments/nonexistent-id")
        assert r.status_code == 404

    def test_backtest_serialised_in_detail(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        # Find the promoted one (has a backtest)
        promoted = reg.filter(status=ExperimentStatus.PROMOTED)[0]
        r = client.get(f"/api/experiments/{promoted.experiment_id}")
        exp = r.json()
        assert exp["backtest"] is not None
        assert "cumulative_return" in exp["backtest"]

    def test_no_backtest_field_is_null(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        # mlp_v2 has no backtest
        completed = reg.filter(status=ExperimentStatus.COMPLETED)[0]
        r = client.get(f"/api/experiments/{completed.experiment_id}")
        assert r.json()["backtest"] is None

    def test_hyperparams_present(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        exp_id = reg.all()[0].experiment_id
        r = client.get(f"/api/experiments/{exp_id}")
        assert "hyperparams" in r.json()

    def test_metrics_present_for_finished(self, tmp_path):
        reg    = _populated_registry(tmp_path)
        client = _make_client(reg)
        completed = reg.filter(status=ExperimentStatus.COMPLETED)[0]
        r = client.get(f"/api/experiments/{completed.experiment_id}")
        assert "metrics" in r.json()
        assert r.json()["metrics"]["accuracy"] > 0
