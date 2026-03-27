"""Tests for ml.patterns.baseline — logistic regression baseline model."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.patterns.baseline import (
    BaselineConfig,
    LogisticBaseline,
    run_baseline_pipeline,
)
from ml.data.dataset_builder import assemble_dataset
from ml.registry.experiment_registry import ExperimentRegistry, ExperimentStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    end   = datetime.now(tz=timezone.utc)
    idx   = pd.bdate_range(end=end, periods=n, tz="UTC")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high":   close * (1 + np.random.uniform(0.001, 0.015, n)),
            "low":    close * (1 - np.random.uniform(0.001, 0.015, n)),
            "close":  close,
            "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture()
def dataset(ohlcv):
    data, meta = assemble_dataset(ohlcv, symbol="AAPL")
    return data, meta


@pytest.fixture()
def fitted_model(dataset):
    data, _ = dataset
    model = LogisticBaseline()
    model.fit(data["X_train"], data["y_train"])
    return model, data


# ── BaselineConfig ─────────────────────────────────────────────────────────────

class TestBaselineConfig:
    def test_defaults(self):
        cfg = BaselineConfig()
        assert cfg.C == 1.0
        assert cfg.max_iter == 1000
        assert cfg.class_weight == "balanced"
        assert cfg.solver == "lbfgs"

    def test_to_dict_round_trip(self):
        cfg = BaselineConfig(C=0.5, max_iter=500)
        assert BaselineConfig.from_dict(cfg.to_dict()) == cfg

    def test_from_dict_ignores_unknown(self):
        d = BaselineConfig().to_dict()
        d["unknown_key"] = "value"
        cfg = BaselineConfig.from_dict(d)
        assert cfg.C == 1.0


# ── LogisticBaseline ──────────────────────────────────────────────────────────

class TestLogisticBaselineFit:
    def test_fit_returns_train_accuracy(self, dataset):
        data, _ = dataset
        model = LogisticBaseline()
        result = model.fit(data["X_train"], data["y_train"])
        assert "train_accuracy" in result
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_predict_proba_before_fit_raises(self):
        model = LogisticBaseline()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 10)))

    def test_predict_proba_after_fit_returns_array(self, fitted_model):
        model, data = fitted_model
        probs = model.predict_proba(data["X_test"])
        assert probs.shape == (len(data["X_test"]),)

    def test_predict_proba_in_unit_interval(self, fitted_model):
        model, data = fitted_model
        probs = model.predict_proba(data["X_test"])
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_predict_proba_dtype_float32(self, fitted_model):
        model, data = fitted_model
        probs = model.predict_proba(data["X_test"])
        assert probs.dtype == np.float32


class TestLogisticBaselineEvaluate:
    def test_returns_all_metric_keys(self, fitted_model):
        model, data = fitted_model
        metrics = model.evaluate(data["X_test"], data["y_test"])
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in metrics

    def test_all_metrics_in_unit_interval(self, fitted_model):
        model, data = fitted_model
        metrics = model.evaluate(data["X_test"], data["y_test"])
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} not in [0, 1]"

    def test_custom_threshold(self, fitted_model):
        model, data = fitted_model
        # High threshold → low recall
        m_high = model.evaluate(data["X_test"], data["y_test"], threshold=0.9)
        m_low  = model.evaluate(data["X_test"], data["y_test"], threshold=0.1)
        # recall should be lower with high threshold (some positives missed)
        assert m_high["recall"] <= m_low["recall"] + 0.01  # small tolerance


class TestLogisticBaselinePersistence:
    def test_save_creates_pkl(self, tmp_path, fitted_model):
        model, _ = fitted_model
        path = tmp_path / "model.pkl"
        model.save(path)
        assert path.exists()

    def test_save_creates_json_sidecar(self, tmp_path, fitted_model):
        model, _ = fitted_model
        path = tmp_path / "model.pkl"
        model.save(path)
        assert path.with_suffix(".json").exists()

    def test_json_sidecar_has_config(self, tmp_path, fitted_model):
        model, _ = fitted_model
        path = tmp_path / "model.pkl"
        model.save(path)
        d = json.loads(path.with_suffix(".json").read_text())
        assert "config" in d

    def test_load_round_trip(self, tmp_path, fitted_model):
        model, data = fitted_model
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = LogisticBaseline.load(path)
        probs_original = model.predict_proba(data["X_test"])
        probs_loaded   = loaded.predict_proba(data["X_test"])
        np.testing.assert_array_almost_equal(probs_original, probs_loaded)


# ── run_baseline_pipeline ─────────────────────────────────────────────────────

class TestRunBaselinePipeline:
    def _run(self, tmp_path: Path, **kwargs):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        return run_baseline_pipeline(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            **kwargs,
        ), reg

    def test_returns_pipeline_result(self, tmp_path):
        from ml.patterns.train_mlp import PipelineResult
        result, _ = self._run(tmp_path)
        assert isinstance(result, PipelineResult)

    def test_experiment_id_set(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert result.experiment_id != ""

    def test_metrics_present(self, tmp_path):
        result, _ = self._run(tmp_path)
        for key in ("accuracy", "f1", "auc"):
            assert key in result.metrics

    def test_metrics_in_range(self, tmp_path):
        result, _ = self._run(tmp_path)
        for v in result.metrics.values():
            assert 0.0 <= v <= 1.0

    def test_checkpoint_file_created(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert Path(result.checkpoint_path).exists()

    def test_scaler_file_created(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert Path(result.scaler_path).exists()

    def test_scaler_json_has_expected_keys(self, tmp_path):
        result, _ = self._run(tmp_path)
        d = json.loads(Path(result.scaler_path).read_text())
        assert "mean_" in d and "std_" in d

    def test_backtest_summary_present(self, tmp_path):
        result, _ = self._run(tmp_path)
        bt = result.backtest_summary
        assert "cumulative_return" in bt
        assert "hit_rate" in bt
        assert "trade_count" in bt

    def test_promotion_recommendation_is_bool(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert isinstance(result.promotion_recommended, bool)

    def test_promotion_reasons_populated(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert len(result.promotion_reasons) == 3  # one per threshold

    def test_experiment_in_registry(self, tmp_path):
        result, reg = self._run(tmp_path)
        exp = reg.get(result.experiment_id)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.model_type == "logistic"
        assert exp.backtest is not None

    def test_accepts_pre_assembled_dataset(self, tmp_path):
        """Pipeline can share a pre-assembled dataset dict."""
        ohlcv = _make_ohlcv()
        data, meta = assemble_dataset(ohlcv, symbol="AAPL")
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_baseline_pipeline(
            symbol         = "AAPL",
            dataset        = data,
            dataset_meta   = meta,
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
        )
        exp = reg.get(result.experiment_id)
        # dataset_version should be stored in the registry
        assert exp.dataset_info.get("dataset_version") == meta.dataset_version

    def test_failed_pipeline_marks_experiment_failed(self, tmp_path):
        from unittest.mock import patch
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        with patch("ml.patterns.baseline.LogisticBaseline.fit",
                   side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                run_baseline_pipeline(
                    symbol         = "AAPL",
                    df             = _make_ohlcv(),
                    registry       = reg,
                    checkpoint_dir = tmp_path / "ckpts",
                )
        failed = reg.filter(status=ExperimentStatus.FAILED)
        assert len(failed) == 1

    def test_tags_stored_in_registry(self, tmp_path):
        result, reg = self._run(tmp_path, tags=["tag_a", "baseline"])
        exp = reg.get(result.experiment_id)
        assert "tag_a" in exp.tags

    def test_custom_config(self, tmp_path):
        from ml.patterns.baseline import BaselineConfig
        cfg = BaselineConfig(C=0.1, max_iter=200)
        result, reg = self._run(tmp_path, config=cfg)
        exp = reg.get(result.experiment_id)
        assert exp.hyperparams["C"] == 0.1
