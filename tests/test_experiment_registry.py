"""Tests for ml.registry.experiment_registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.registry.experiment_registry import (
    BacktestResult,
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStatus,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(path=tmp_path / "experiments.json")


def _make_backtest(**overrides) -> BacktestResult:
    defaults = dict(
        cumulative_return = 0.18,
        annualised_return = 0.11,
        hit_rate          = 0.58,
        max_drawdown      = -0.09,
        sharpe            = 1.2,
        trade_count       = 100,
        benchmark_return  = 0.14,
        period_start      = "2023-01-01",
        period_end        = "2024-12-31",
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


# ── BacktestResult ────────────────────────────────────────────────────────────

class TestBacktestResult:
    def test_to_dict_round_trip(self):
        bt = _make_backtest()
        d  = bt.to_dict()
        bt2 = BacktestResult.from_dict(d)
        assert bt2.cumulative_return == bt.cumulative_return
        assert bt2.hit_rate == bt.hit_rate

    def test_json_serializable(self):
        assert json.dumps(_make_backtest().to_dict())

    def test_extra_field_preserved(self):
        bt = _make_backtest(extra={"AAPL": 0.22})
        assert bt.to_dict()["extra"]["AAPL"] == 0.22


# ── ExperimentRecord ──────────────────────────────────────────────────────────

class TestExperimentRecord:
    def _make(self) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id = "test-uuid",
            name          = "test_run",
            model_type    = "mlp",
            status        = ExperimentStatus.RUNNING,
            started_at    = "2024-01-01T00:00:00+00:00",
        )

    def test_is_running_true(self):
        assert self._make().is_running

    def test_is_running_false(self):
        r = self._make()
        r.status = ExperimentStatus.COMPLETED
        assert not r.is_running

    def test_duration_none_when_not_finished(self):
        assert self._make().duration_seconds is None

    def test_duration_computed(self):
        r = self._make()
        r.started_at  = "2024-01-01T00:00:00+00:00"
        r.finished_at = "2024-01-01T00:01:00+00:00"
        assert r.duration_seconds == 60.0

    def test_to_dict_status_is_string(self):
        d = self._make().to_dict()
        assert isinstance(d["status"], str)

    def test_to_dict_backtest_none(self):
        d = self._make().to_dict()
        assert d["backtest"] is None

    def test_round_trip_with_backtest(self):
        r = self._make()
        r.status  = ExperimentStatus.COMPLETED
        r.backtest = _make_backtest()
        d  = r.to_dict()
        r2 = ExperimentRecord.from_dict(d)
        assert r2.backtest is not None
        assert r2.backtest.hit_rate == r.backtest.hit_rate

    def test_round_trip_status_enum(self):
        r  = self._make()
        d  = r.to_dict()
        r2 = ExperimentRecord.from_dict(d)
        assert r2.status == ExperimentStatus.RUNNING


# ── Registry CRUD ─────────────────────────────────────────────────────────────

class TestRegistryCreate:
    def test_create_returns_record(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("run1", "mlp")
        assert exp.name == "run1"
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.experiment_id != ""

    def test_create_persists(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("run1", "mlp")
        reg2 = _make_registry(tmp_path)
        assert exp.experiment_id in reg2

    def test_create_with_hyperparams(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp", hyperparams={"lr": 0.001})
        assert reg.get(exp.experiment_id).hyperparams["lr"] == 0.001

    def test_create_with_tags(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp", tags=["baseline", "ohlcv"])
        assert "baseline" in reg.get(exp.experiment_id).tags

    def test_len_increments(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert len(reg) == 0
        reg.create("a", "mlp")
        reg.create("b", "lstm")
        assert len(reg) == 2


class TestRegistryGet:
    def test_get_existing(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("run1", "mlp")
        assert reg.get(exp.experiment_id).name == "run1"

    def test_get_missing_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(KeyError):
            reg.get("nonexistent-uuid")

    def test_contains(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        assert exp.experiment_id in reg
        assert "nope" not in reg


class TestRegistryFinish:
    def test_finish_sets_completed(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.62})
        assert reg.get(exp.experiment_id).status == ExperimentStatus.COMPLETED

    def test_finish_records_metrics(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.62, "val_loss": 0.48})
        r = reg.get(exp.experiment_id)
        assert r.metrics["val_accuracy"] == 0.62

    def test_finish_failed(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, failed=True, failure_reason="OOM")
        r = reg.get(exp.experiment_id)
        assert r.status == ExperimentStatus.FAILED
        assert "OOM" in r.notes

    def test_finish_sets_checkpoint_path(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, checkpoint_path="/checkpoints/model.pt")
        assert reg.get(exp.experiment_id).checkpoint_path == "/checkpoints/model.pt"

    def test_finish_sets_finished_at(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id)
        assert reg.get(exp.experiment_id).finished_at is not None


class TestRegistryBacktestAndPromote:
    def test_attach_backtest(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.62})
        reg.attach_backtest(exp.experiment_id, _make_backtest())
        r = reg.get(exp.experiment_id)
        assert r.backtest is not None
        assert r.backtest.cumulative_return == 0.18

    def test_promote_succeeds(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, metrics={"val_accuracy": 0.62})
        reg.attach_backtest(exp.experiment_id, _make_backtest())
        reg.promote(exp.experiment_id)
        assert reg.get(exp.experiment_id).status == ExperimentStatus.PROMOTED

    def test_promote_without_backtest_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id)
        with pytest.raises(ValueError, match="BacktestResult"):
            reg.promote(exp.experiment_id)

    def test_promote_non_completed_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        # Still RUNNING
        with pytest.raises(ValueError, match="COMPLETED"):
            reg.promote(exp.experiment_id)

    def test_promote_with_notes(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id)
        reg.attach_backtest(exp.experiment_id, _make_backtest())
        reg.promote(exp.experiment_id, notes="Beats benchmark")
        assert "Beats benchmark" in reg.get(exp.experiment_id).notes


class TestRegistryArchive:
    def test_archive(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.archive(exp.experiment_id)
        assert reg.get(exp.experiment_id).status == ExperimentStatus.ARCHIVED


class TestRegistryUpdateNotes:
    def test_notes_appended(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp", notes="first note")
        reg.update_notes(exp.experiment_id, "second note")
        assert "first note" in reg.get(exp.experiment_id).notes
        assert "second note" in reg.get(exp.experiment_id).notes


# ── Query ─────────────────────────────────────────────────────────────────────

class TestRegistryFilter:
    def _populate(self, reg: ExperimentRegistry) -> list[str]:
        ids = []
        e1 = reg.create("mlp_v1", "mlp", tags=["baseline"])
        reg.finish(e1.experiment_id, metrics={"val_accuracy": 0.60})
        ids.append(e1.experiment_id)

        e2 = reg.create("mlp_v2", "mlp", tags=["baseline"])
        reg.finish(e2.experiment_id, metrics={"val_accuracy": 0.63})
        reg.attach_backtest(e2.experiment_id, _make_backtest())
        reg.promote(e2.experiment_id)
        ids.append(e2.experiment_id)

        e3 = reg.create("lstm_v1", "lstm")
        reg.finish(e3.experiment_id, failed=True)
        ids.append(e3.experiment_id)

        return ids

    def test_filter_by_model_type(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.filter(model_type="mlp")) == 2

    def test_filter_by_status(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.filter(status=ExperimentStatus.PROMOTED)) == 1

    def test_filter_by_tag(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.filter(tag="baseline")) == 2

    def test_filter_has_backtest_true(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.filter(has_backtest=True)) == 1

    def test_filter_has_backtest_false(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.filter(has_backtest=False)) == 2

    def test_filter_combined(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        result = reg.filter(model_type="mlp", status=ExperimentStatus.COMPLETED)
        assert len(result) == 1

    def test_all_returns_all(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        assert len(reg.all()) == 3

    def test_all_sorted_most_recent_first(self, tmp_path):
        reg = _make_registry(tmp_path)
        self._populate(reg)
        times = [r.started_at for r in reg.all()]
        assert times == sorted(times, reverse=True)


class TestRegistryBest:
    def test_best_higher_is_better(self, tmp_path):
        reg = _make_registry(tmp_path)
        e1 = reg.create("a", "mlp")
        reg.finish(e1.experiment_id, metrics={"val_accuracy": 0.60})
        e2 = reg.create("b", "mlp")
        reg.finish(e2.experiment_id, metrics={"val_accuracy": 0.65})
        best = reg.best("val_accuracy", model_type="mlp")
        assert best.experiment_id == e2.experiment_id

    def test_best_lower_is_better(self, tmp_path):
        reg = _make_registry(tmp_path)
        e1 = reg.create("a", "mlp")
        reg.finish(e1.experiment_id, metrics={"val_loss": 0.60})
        e2 = reg.create("b", "mlp")
        reg.finish(e2.experiment_id, metrics={"val_loss": 0.45})
        best = reg.best("val_loss", model_type="mlp", higher_is_better=False)
        assert best.experiment_id == e2.experiment_id

    def test_best_returns_none_when_no_metric(self, tmp_path):
        reg = _make_registry(tmp_path)
        exp = reg.create("r", "mlp")
        reg.finish(exp.experiment_id, metrics={})
        assert reg.best("val_accuracy") is None

    def test_best_filters_by_model_type(self, tmp_path):
        reg = _make_registry(tmp_path)
        e1 = reg.create("mlp", "mlp")
        reg.finish(e1.experiment_id, metrics={"val_accuracy": 0.70})
        e2 = reg.create("lstm", "lstm")
        reg.finish(e2.experiment_id, metrics={"val_accuracy": 0.75})
        best = reg.best("val_accuracy", model_type="mlp")
        assert best.experiment_id == e1.experiment_id


# ── Summary ───────────────────────────────────────────────────────────────────

class TestRegistrySummary:
    def test_summary_counts(self, tmp_path):
        reg = _make_registry(tmp_path)
        e1 = reg.create("a", "mlp")
        reg.finish(e1.experiment_id, metrics={"val_accuracy": 0.62})
        reg.attach_backtest(e1.experiment_id, _make_backtest())
        reg.promote(e1.experiment_id)

        e2 = reg.create("b", "lstm")
        reg.finish(e2.experiment_id, failed=True)

        s = reg.summary()
        assert s["total"] == 2
        assert s["promoted"] == 1
        assert s["with_backtest"] == 1
        assert s["by_status"]["promoted"] == 1
        assert s["by_status"]["failed"] == 1

    def test_summary_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        s = reg.summary()
        assert s["total"] == 0
        assert s["promoted"] == 0


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_survives_reload(self, tmp_path):
        path = tmp_path / "experiments.json"
        reg1 = ExperimentRegistry(path=path)
        exp  = reg1.create("run1", "mlp", hyperparams={"lr": 0.001})
        reg1.finish(exp.experiment_id, metrics={"val_accuracy": 0.62})

        reg2 = ExperimentRegistry(path=path)
        r    = reg2.get(exp.experiment_id)
        assert r.name == "run1"
        assert r.status == ExperimentStatus.COMPLETED
        assert r.metrics["val_accuracy"] == 0.62
        assert r.hyperparams["lr"] == 0.001

    def test_json_format(self, tmp_path):
        path = tmp_path / "experiments.json"
        reg  = ExperimentRegistry(path=path)
        reg.create("r", "mlp")
        payload = json.loads(path.read_text())
        assert payload["version"] == "1"
        assert len(payload["experiments"]) == 1
