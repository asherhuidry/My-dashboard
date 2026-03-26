"""Tests for LSTM model, dataset, and training utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from datetime import datetime, timezone, timedelta

from ml.patterns.lstm import FinBrainLSTM, save_checkpoint, load_model, predict
from ml.patterns.dataset import PriceSequenceDataset, chronological_split


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 200, n_feat: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    rng   = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    data  = rng.standard_normal((n, n_feat)).astype(np.float32)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    df    = pd.DataFrame(data, index=dates, columns=[f"f{i}" for i in range(n_feat)])
    return df, close


# ─────────────────────────────────────────────────────────────────────────────
# FinBrainLSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestFinBrainLSTM:
    def test_output_shape(self) -> None:
        model = FinBrainLSTM(input_size=10)
        x     = torch.randn(4, 20, 10)
        out   = model(x)
        assert out.shape == (4, 1)

    def test_output_bounded(self) -> None:
        model = FinBrainLSTM(input_size=8)
        x     = torch.randn(8, 15, 8)
        out   = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_different_batch_sizes(self) -> None:
        model = FinBrainLSTM(input_size=5)
        for bs in [1, 8, 32]:
            out = model(torch.randn(bs, 20, 5))
            assert out.shape == (bs, 1)

    def test_single_layer_no_dropout_crash(self) -> None:
        model = FinBrainLSTM(input_size=4, num_layers=1, dropout=0.5)
        out   = model(torch.randn(2, 10, 4))
        assert out.shape == (2, 1)

    def test_eval_mode_deterministic(self) -> None:
        model = FinBrainLSTM(input_size=6)
        model.eval()
        x    = torch.randn(1, 20, 6)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_gradients_flow(self) -> None:
        model = FinBrainLSTM(input_size=4)
        x     = torch.randn(4, 10, 4, requires_grad=False)
        y     = torch.rand(4, 1)
        out   = model(x)
        loss  = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


# ─────────────────────────────────────────────────────────────────────────────
# PriceSequenceDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceSequenceDataset:
    def test_length(self) -> None:
        df, close = _make_feature_df(100, 5)
        ds = PriceSequenceDataset(df, close, seq_len=20, target_horizon=1)
        # Expect n - seq_len - horizon samples (approx)
        assert len(ds) > 0

    def test_sample_shape(self) -> None:
        df, close = _make_feature_df(100, 8)
        ds = PriceSequenceDataset(df, close, seq_len=15)
        X, y = ds[0]
        assert X.shape == (15, 8)
        assert y.shape == (1,)

    def test_targets_binary(self) -> None:
        df, close = _make_feature_df(100, 4)
        ds = PriceSequenceDataset(df, close)
        for i in range(min(20, len(ds))):
            _, y = ds[i]
            assert y.item() in {0.0, 1.0}

    def test_n_features_property(self) -> None:
        df, close = _make_feature_df(100, 12)
        ds = PriceSequenceDataset(df, close)
        assert ds.n_features == 12

    def test_scaler_fitted(self) -> None:
        df, close = _make_feature_df(100, 5)
        ds = PriceSequenceDataset(df, close)
        assert ds.scaler_mean.shape == (5,)
        assert ds.scaler_std.shape  == (5,)

    def test_precomputed_scaler_applied(self) -> None:
        df, close = _make_feature_df(100, 5)
        ds1  = PriceSequenceDataset(df, close)
        mean = ds1.scaler_mean
        std  = ds1.scaler_std
        ds2  = PriceSequenceDataset(df, close, scaler_mean=mean, scaler_std=std)
        # Same scaler should give same data
        X1, _ = ds1[0]
        X2, _ = ds2[0]
        assert torch.allclose(X1, X2)


# ─────────────────────────────────────────────────────────────────────────────
# chronological_split
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalSplit:
    def test_sizes_sum_to_total(self) -> None:
        df, close = _make_feature_df(200, 6)
        ds        = PriceSequenceDataset(df, close, seq_len=20)
        tr, vl    = chronological_split(ds, 0.8)
        assert len(tr) + len(vl) == len(ds)

    def test_correct_split_ratio(self) -> None:
        df, close = _make_feature_df(300, 6)
        ds        = PriceSequenceDataset(df, close, seq_len=20)
        tr, vl    = chronological_split(ds, 0.8)
        ratio = len(tr) / len(ds)
        assert abs(ratio - 0.8) < 0.02

    def test_val_inherits_scaler(self) -> None:
        df, close = _make_feature_df(200, 5)
        ds        = PriceSequenceDataset(df, close)
        _, vl     = chronological_split(ds)
        np.testing.assert_array_equal(vl.scaler_mean, ds.scaler_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        model    = FinBrainLSTM(input_size=8)
        optim    = torch.optim.Adam(model.parameters())
        ckpt     = tmp_path / "test.pt"
        mean     = np.zeros(8, dtype=np.float32)
        std      = np.ones(8,  dtype=np.float32)

        save_checkpoint(ckpt, model, optim, epoch=5,
                        metrics={"val_acc": 0.55},
                        scaler_mean=mean, scaler_std=std,
                        feature_cols=[f"f{i}" for i in range(8)],
                        config={"input_size": 8, "hidden_size": 128, "num_layers": 2, "dropout": 0.3})

        loaded, ckpt_data = load_model(ckpt)
        assert ckpt_data["epoch"] == 5
        assert ckpt_data["metrics"]["val_acc"] == 0.55

    def test_loaded_model_produces_same_output(self, tmp_path) -> None:
        model = FinBrainLSTM(input_size=6)
        model.eval()
        x     = torch.randn(1, 10, 6)
        out1  = model(x).item()

        optim = torch.optim.Adam(model.parameters())
        ckpt  = tmp_path / "m.pt"
        save_checkpoint(ckpt, model, optim, 1, {},
                        np.zeros(6), np.ones(6), [], {"input_size":6,"hidden_size":128,"num_layers":2,"dropout":0.3})

        loaded, _ = load_model(ckpt)
        out2  = loaded(x).item()
        assert abs(out1 - out2) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# predict helper
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_float_in_01(self) -> None:
        model = FinBrainLSTM(input_size=5)
        model.eval()
        x     = np.random.randn(20, 5).astype(np.float32)
        mean  = np.zeros(5, dtype=np.float32)
        std   = np.ones(5,  dtype=np.float32)
        prob  = predict(model, x, mean, std)
        assert 0.0 <= prob <= 1.0

    def test_predict_consistent_eval(self) -> None:
        model = FinBrainLSTM(input_size=4)
        model.eval()
        x    = np.ones((15, 4), dtype=np.float32)
        mean = np.zeros(4, dtype=np.float32)
        std  = np.ones(4,  dtype=np.float32)
        p1   = predict(model, x, mean, std)
        p2   = predict(model, x, mean, std)
        assert abs(p1 - p2) < 1e-9
