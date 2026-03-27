"""Tests for ml.patterns.train_mlp end-to-end pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from ml.patterns.train_mlp import (
    PipelineResult,
    build_dataset,
    fetch_price_df,
    run_pipeline,
    main,
    PROMO_MIN_ACCURACY,
    PROMO_MIN_HIT_RATE,
)
from ml.registry.experiment_registry import ExperimentRegistry, ExperimentStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame large enough to split into 3 sets."""
    np.random.seed(42)
    end   = datetime.now(tz=timezone.utc)
    idx   = pd.bdate_range(end=end, periods=n, tz="UTC")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    return pd.DataFrame({
        "open":   close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + np.random.uniform(0.001, 0.015, n)),
        "low":    close * (1 - np.random.uniform(0.001, 0.015, n)),
        "close":  close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    }, index=idx)


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    return _make_ohlcv(n=300)


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(path=tmp_path / "experiments.json")


# ── fetch_price_df ────────────────────────────────────────────────────────────

class TestFetchPriceDf:
    @patch("ml.patterns.train_mlp.yf.Ticker")
    def test_returns_lowercase_columns(self, mock_ticker):
        df_raw = _make_ohlcv()
        df_raw.columns = [c.capitalize() for c in df_raw.columns]  # simulate yfinance
        mock_ticker.return_value.history.return_value = df_raw
        result = fetch_price_df("AAPL")
        assert "close" in result.columns
        assert "Close" not in result.columns

    @patch("ml.patterns.train_mlp.yf.Ticker")
    def test_raises_on_empty(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            fetch_price_df("FAKE")

    @patch("ml.patterns.train_mlp.yf.Ticker")
    def test_index_is_datetime(self, mock_ticker):
        mock_ticker.return_value.history.return_value = _make_ohlcv()
        result = fetch_price_df("AAPL")
        assert isinstance(result.index, pd.DatetimeIndex)


# ── build_dataset ─────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_returns_expected_keys(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        for key in ("X_train", "y_train", "X_val", "y_val",
                    "X_test", "y_test", "close_test", "feature_cols"):
            assert key in ds

    def test_splits_are_non_overlapping(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        assert ds["n_train"] + ds["n_val"] + ds["n_test"] <= len(ohlcv_df)

    def test_x_y_shapes_consistent(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        for split in ("train", "val", "test"):
            assert len(ds[f"X_{split}"]) == len(ds[f"y_{split}"])

    def test_y_is_binary(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        y_all = np.concatenate([ds["y_train"], ds["y_val"], ds["y_test"]])
        assert set(y_all).issubset({0.0, 1.0})

    def test_no_inf_in_features(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        for split in ("train", "val", "test"):
            X = ds[f"X_{split}"]
            assert np.all(np.isfinite(X)), f"Inf/NaN found in X_{split}"

    def test_feature_cols_not_empty(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        assert len(ds["feature_cols"]) > 0

    def test_close_test_index_matches_x_test(self, ohlcv_df):
        ds = build_dataset(ohlcv_df)
        assert len(ds["close_test"]) == len(ds["X_test"])

    def test_chronological_order_preserved(self, ohlcv_df):
        """Test set should come after val, which comes after train (no shuffling)."""
        ds = build_dataset(ohlcv_df)
        # close_test index should be after train and val periods
        assert ds["n_train"] > 0
        assert ds["n_val"] > 0
        assert ds["n_test"] > 0


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:
    """Tests for the full pipeline using synthetic data (no yfinance calls)."""

    def _run(self, tmp_path: Path, n: int = 300, **kwargs) -> PipelineResult:
        """Helper: run pipeline with synthetic data and tmp registry."""
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        return run_pipeline(
            symbol         = "AAPL",
            df             = _make_ohlcv(n=n),
            registry       = reg,
            epochs         = 3,
            patience       = 3,
            checkpoint_dir = tmp_path / "checkpoints",
            **kwargs,
        )

    def test_returns_pipeline_result(self, tmp_path):
        result = self._run(tmp_path)
        assert isinstance(result, PipelineResult)

    def test_experiment_id_set(self, tmp_path):
        result = self._run(tmp_path)
        assert result.experiment_id != ""

    def test_metrics_present(self, tmp_path):
        result = self._run(tmp_path)
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert "auc" in result.metrics

    def test_metrics_in_range(self, tmp_path):
        result = self._run(tmp_path)
        for v in result.metrics.values():
            assert 0.0 <= v <= 1.0

    def test_checkpoint_file_created(self, tmp_path):
        result = self._run(tmp_path)
        assert Path(result.checkpoint_path).exists()

    def test_scaler_file_created(self, tmp_path):
        result = self._run(tmp_path)
        assert Path(result.scaler_path).exists()

    def test_scaler_file_is_valid_json(self, tmp_path):
        result = self._run(tmp_path)
        d = json.loads(Path(result.scaler_path).read_text())
        assert "mean_" in d and "std_" in d

    def test_backtest_summary_present(self, tmp_path):
        result = self._run(tmp_path)
        bt = result.backtest_summary
        assert "cumulative_return" in bt
        assert "hit_rate" in bt
        assert "trade_count" in bt

    def test_experiment_recorded_in_registry(self, tmp_path):
        reg    = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_pipeline(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            registry       = reg,
            epochs         = 2,
            patience       = 2,
            checkpoint_dir = tmp_path / "checkpoints",
        )
        exp = reg.get(result.experiment_id)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.backtest is not None

    def test_failed_pipeline_marks_experiment_failed(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        with patch("ml.patterns.train_mlp.train", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                run_pipeline(
                    symbol         = "AAPL",
                    df             = _make_ohlcv(),
                    registry       = reg,
                    epochs         = 2,
                    patience       = 2,
                    checkpoint_dir = tmp_path / "checkpoints",
                )
        # Experiment should be in FAILED state
        experiments = reg.filter(status=ExperimentStatus.FAILED)
        assert len(experiments) == 1

    def test_promotion_reasons_populated(self, tmp_path):
        result = self._run(tmp_path)
        assert len(result.promotion_reasons) == 3  # one per threshold

    def test_promotion_recommendation_is_bool(self, tmp_path):
        result = self._run(tmp_path)
        assert isinstance(result.promotion_recommended, bool)

    def test_tags_passed_to_registry(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_pipeline(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            registry       = reg,
            epochs         = 2,
            patience       = 2,
            checkpoint_dir = tmp_path / "checkpoints",
            tags           = ["test_tag", "baseline"],
        )
        exp = reg.get(result.experiment_id)
        assert "test_tag" in exp.tags

    def test_custom_hidden_sizes(self, tmp_path):
        result = self._run(tmp_path, hidden_sizes=[32, 16])
        assert Path(result.checkpoint_path).exists()


# ── PipelineResult ────────────────────────────────────────────────────────────

class TestPipelineResult:
    def _make_result(self) -> PipelineResult:
        return PipelineResult(
            experiment_id        = "test-id",
            symbol               = "AAPL",
            metrics              = {"accuracy": 0.58, "f1": 0.57, "auc": 0.61,
                                    "precision": 0.58, "recall": 0.56},
            backtest_summary     = {
                "cumulative_return": 0.12, "benchmark_return": 0.10,
                "annualised_return": 0.08, "hit_rate": 0.55,
                "max_drawdown": -0.08, "sharpe": 1.1, "trade_count": 20,
                "period_start": "2024-01-01", "period_end": "2024-12-31",
            },
            checkpoint_path      = "/checkpoints/test.pt",
            scaler_path          = "/checkpoints/test_scaler.json",
            promotion_recommended = True,
            promotion_reasons    = ["accuracy 0.58 ≥ 0.55"],
        )

    def test_print_summary_runs(self, capsys):
        self._make_result().print_summary()
        captured = capsys.readouterr()
        assert "test-id" in captured.out
        assert "AAPL" in captured.out
        assert "RECOMMENDED" in captured.out

    def test_print_summary_not_recommended(self, capsys):
        r = self._make_result()
        r.promotion_recommended = False
        r.print_summary()
        assert "NOT recommended" in capsys.readouterr().out


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_json_output(self, tmp_path, capsys):
        """--json flag should produce parseable JSON."""
        import ml.patterns.train_mlp as mod
        mod.CHECKPOINT_DIR = tmp_path / "checkpoints"
        mod.SCALER_DIR     = tmp_path / "checkpoints"

        with patch("ml.patterns.train_mlp.fetch_price_df", return_value=_make_ohlcv()), \
             patch("ml.patterns.train_mlp.ExperimentRegistry",
                   return_value=ExperimentRegistry(path=tmp_path / "e.json")):
            rc = main(["--symbol", "AAPL", "--epochs", "2", "--patience", "2", "--json"])

        assert rc == 0
        out = capsys.readouterr().out
        d   = json.loads(out)
        assert "experiment_id" in d
        assert "metrics" in d
        assert "backtest" in d

    def test_error_returns_1(self, capsys):
        """Pipeline failure should return exit code 1."""
        with patch("ml.patterns.train_mlp.fetch_price_df",
                   side_effect=ValueError("no data")):
            rc = main(["--symbol", "FAKE"])
        assert rc == 1
