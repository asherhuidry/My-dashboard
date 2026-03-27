"""Tests for ml.patterns.train_lstm — registry-integrated LSTM pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.patterns.train_lstm import LSTMConfig, _evaluate_lstm, _train_lstm, run_lstm_pipeline
from ml.data.dataset_builder import assemble_sequence_dataset
from ml.registry.experiment_registry import ExperimentRegistry, ExperimentStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(7)
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


# ── LSTMConfig ────────────────────────────────────────────────────────────────

class TestLSTMConfig:
    def test_defaults(self):
        cfg = LSTMConfig()
        assert cfg.seq_len    == 20
        assert cfg.hidden_size == 128
        assert cfg.num_layers  == 2
        assert cfg.dropout     == 0.3

    def test_to_dict_round_trip(self):
        cfg = LSTMConfig(seq_len=10, hidden_size=32)
        assert LSTMConfig.from_dict(cfg.to_dict()) == cfg

    def test_from_dict_ignores_unknown(self):
        d = LSTMConfig().to_dict()
        d["future_key"] = "value"
        cfg = LSTMConfig.from_dict(d)
        assert cfg.seq_len == 20


# ── _train_lstm and _evaluate_lstm ────────────────────────────────────────────

class TestLSTMInternals:
    @pytest.fixture()
    def small_datasets(self):
        ohlcv = _make_ohlcv(n=300)
        cfg   = LSTMConfig(seq_len=5, hidden_size=8, num_layers=1,
                           epochs=2, patience=2, batch_size=16)
        train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(
            ohlcv, symbol="TEST", seq_len=cfg.seq_len, train_frac=0.70, val_frac=0.15
        )
        return train_ds, val_ds, test_ds, close_test, cfg

    def test_train_returns_history(self, small_datasets):
        from ml.patterns.lstm import FinBrainLSTM
        train_ds, val_ds, _, _, cfg = small_datasets
        model = FinBrainLSTM(
            input_size  = train_ds.n_features,
            hidden_size = cfg.hidden_size,
            num_layers  = cfg.num_layers,
        )
        history = _train_lstm(model, train_ds, val_ds, cfg)
        for key in ("train_losses", "val_losses", "val_accs", "best_epoch"):
            assert key in history

    def test_train_produces_non_empty_loss_list(self, small_datasets):
        from ml.patterns.lstm import FinBrainLSTM
        train_ds, val_ds, _, _, cfg = small_datasets
        model = FinBrainLSTM(input_size=train_ds.n_features,
                             hidden_size=cfg.hidden_size, num_layers=1)
        history = _train_lstm(model, train_ds, val_ds, cfg)
        assert len(history["train_losses"]) >= 1

    def test_evaluate_returns_metrics_and_probs(self, small_datasets):
        from ml.patterns.lstm import FinBrainLSTM
        train_ds, val_ds, test_ds, _, cfg = small_datasets
        model = FinBrainLSTM(input_size=train_ds.n_features,
                             hidden_size=cfg.hidden_size, num_layers=1)
        model.eval()
        metrics, probs = _evaluate_lstm(model, test_ds)
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in metrics
        assert len(probs) == len(test_ds)

    def test_metrics_in_unit_interval(self, small_datasets):
        from ml.patterns.lstm import FinBrainLSTM
        train_ds, _, test_ds, _, cfg = small_datasets
        model = FinBrainLSTM(input_size=train_ds.n_features,
                             hidden_size=cfg.hidden_size, num_layers=1)
        model.eval()
        metrics, _ = _evaluate_lstm(model, test_ds)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v}"


# ── run_lstm_pipeline ─────────────────────────────────────────────────────────

class TestRunLstmPipeline:
    """Integration tests using tiny model config for speed."""

    def _cfg(self) -> LSTMConfig:
        return LSTMConfig(
            seq_len     = 5,
            hidden_size = 8,
            num_layers  = 1,
            dropout     = 0.0,
            epochs      = 2,
            patience    = 2,
            batch_size  = 32,
            train_frac  = 0.70,
            val_frac    = 0.15,
        )

    def _run(self, tmp_path: Path, **kwargs):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        return run_lstm_pipeline(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            config         = self._cfg(),
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

    def test_scaler_json_created(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert Path(result.scaler_path).exists()

    def test_scaler_json_has_expected_keys(self, tmp_path):
        result, _ = self._run(tmp_path)
        d = json.loads(Path(result.scaler_path).read_text())
        assert "scaler_mean" in d and "scaler_std" in d and "feature_cols" in d

    def test_backtest_summary_present(self, tmp_path):
        result, _ = self._run(tmp_path)
        for key in ("cumulative_return", "hit_rate", "trade_count"):
            assert key in result.backtest_summary

    def test_promotion_recommendation_is_bool(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert isinstance(result.promotion_recommended, bool)

    def test_promotion_reasons_populated(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert len(result.promotion_reasons) == 3

    def test_experiment_in_registry_completed(self, tmp_path):
        result, reg = self._run(tmp_path)
        exp = reg.get(result.experiment_id)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.model_type == "lstm"

    def test_backtest_attached_to_registry(self, tmp_path):
        result, reg = self._run(tmp_path)
        exp = reg.get(result.experiment_id)
        assert exp.backtest is not None

    def test_dataset_version_in_registry(self, tmp_path):
        result, reg = self._run(tmp_path)
        exp = reg.get(result.experiment_id)
        assert "dataset_version" in exp.dataset_info

    def test_failed_pipeline_marks_failed(self, tmp_path):
        from unittest.mock import patch
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        with patch("ml.patterns.train_lstm._train_lstm",
                   side_effect=RuntimeError("bang")):
            with pytest.raises(RuntimeError, match="bang"):
                run_lstm_pipeline(
                    symbol         = "AAPL",
                    df             = _make_ohlcv(),
                    registry       = reg,
                    checkpoint_dir = tmp_path / "ckpts",
                    config         = self._cfg(),
                )
        failed = reg.filter(status=ExperimentStatus.FAILED)
        assert len(failed) == 1

    def test_tags_stored_in_registry(self, tmp_path):
        result, reg = self._run(tmp_path, tags=["lstm_run"])
        exp = reg.get(result.experiment_id)
        assert "lstm_run" in exp.tags

    def test_flat_dataset_version_crosslinked(self, tmp_path):
        ohlcv = _make_ohlcv()
        from ml.data.dataset_builder import assemble_dataset
        _, flat_meta = assemble_dataset(ohlcv, symbol="AAPL")
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_lstm_pipeline(
            symbol         = "AAPL",
            df             = ohlcv,
            dataset_meta   = flat_meta,
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            config         = self._cfg(),
        )
        exp = reg.get(result.experiment_id)
        assert exp.dataset_info.get("flat_dataset_version") == flat_meta.dataset_version
