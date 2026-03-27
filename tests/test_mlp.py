"""Tests for ml.patterns.mlp."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.patterns.mlp import (
    MLP,
    MLPConfig,
    FeatureScaler,
    TrainingHistory,
    evaluate,
    load_checkpoint,
    predict,
    save_checkpoint,
    train,
    _roc_auc,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_cfg() -> MLPConfig:
    """Minimal config for fast tests."""
    return MLPConfig(
        input_size   = 10,
        hidden_sizes = [16, 8],
        dropout      = 0.0,
        batch_norm   = False,  # BN needs batch size > 1
        lr           = 1e-2,
        epochs       = 5,
        patience     = 5,
        batch_size   = 16,
    )


@pytest.fixture()
def synthetic_data(small_cfg: MLPConfig):
    """Return (X_train, y_train, X_val, y_val) tensors."""
    torch.manual_seed(42)
    n_train, n_val = 80, 20
    X_train = torch.randn(n_train, small_cfg.input_size)
    y_train = torch.randint(0, 2, (n_train,)).float()
    X_val   = torch.randn(n_val, small_cfg.input_size)
    y_val   = torch.randint(0, 2, (n_val,)).float()
    return X_train, y_train, X_val, y_val


# ── MLPConfig ─────────────────────────────────────────────────────────────────

class TestMLPConfig:
    def test_defaults(self):
        cfg = MLPConfig()
        assert cfg.input_size == 60
        assert cfg.hidden_sizes == [128, 64]

    def test_to_dict_round_trip(self):
        cfg  = MLPConfig(input_size=20, hidden_sizes=[32, 16], lr=0.01)
        d    = cfg.to_dict()
        cfg2 = MLPConfig.from_dict(d)
        assert cfg2.input_size    == 20
        assert cfg2.hidden_sizes  == [32, 16]
        assert cfg2.lr            == 0.01

    def test_to_dict_json_serializable(self):
        assert json.dumps(MLPConfig().to_dict())

    def test_from_dict_ignores_unknown_keys(self):
        d = MLPConfig().to_dict()
        d["unknown_future_key"] = "ignored"
        cfg = MLPConfig.from_dict(d)
        assert cfg.input_size == 60


# ── MLP construction ──────────────────────────────────────────────────────────

class TestMLPConstruction:
    def test_forward_shape(self, small_cfg):
        model = MLP(small_cfg)
        x = torch.randn(4, small_cfg.input_size)
        out = model(x)
        assert out.shape == (4, 1)

    def test_forward_single_sample(self, small_cfg):
        model = MLP(small_cfg)
        x = torch.randn(1, small_cfg.input_size)
        out = model(x)
        assert out.shape == (1, 1)

    def test_predict_proba_range(self, small_cfg):
        model = MLP(small_cfg)
        x = torch.randn(8, small_cfg.input_size)
        probs = model.predict_proba(x)
        assert probs.shape == (8,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_proba_single_vec(self, small_cfg):
        model = MLP(small_cfg)
        x = torch.randn(small_cfg.input_size)  # 1-D
        probs = model.predict_proba(x)
        assert probs.shape == (1,)

    def test_activation_gelu(self):
        cfg = MLPConfig(input_size=5, hidden_sizes=[8], activation="gelu",
                        batch_norm=False, dropout=0.0)
        model = MLP(cfg)
        out = model(torch.randn(2, 5))
        assert out.shape == (2, 1)

    def test_activation_tanh(self):
        cfg = MLPConfig(input_size=5, hidden_sizes=[8], activation="tanh",
                        batch_norm=False, dropout=0.0)
        model = MLP(cfg)
        out = model(torch.randn(2, 5))
        assert out.shape == (2, 1)

    def test_unknown_activation_raises(self):
        cfg = MLPConfig(input_size=5, hidden_sizes=[8], activation="sigmoid",
                        batch_norm=False, dropout=0.0)
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(cfg)

    def test_batch_norm_layers_present(self):
        cfg   = MLPConfig(input_size=5, hidden_sizes=[8, 4], batch_norm=True, dropout=0.0)
        model = MLP(cfg)
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 2

    def test_dropout_layers_present(self):
        cfg   = MLPConfig(input_size=5, hidden_sizes=[8], dropout=0.3, batch_norm=False)
        model = MLP(cfg)
        drop_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(drop_layers) == 1

    def test_no_dropout_when_zero(self):
        cfg   = MLPConfig(input_size=5, hidden_sizes=[8], dropout=0.0, batch_norm=False)
        model = MLP(cfg)
        drop_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(drop_layers) == 0


# ── Training ──────────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_history(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)
        assert isinstance(history, TrainingHistory)

    def test_history_lengths(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)
        n = len(history.train_loss)
        assert n == len(history.val_loss) == len(history.val_acc)
        assert n <= small_cfg.epochs

    def test_best_epoch_in_range(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)
        assert 0 <= history.best_epoch < len(history.val_loss)

    def test_weights_change_after_training(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        before = [p.clone() for p in model.parameters()]
        train(model, X_train, y_train, X_val, y_val, small_cfg)
        after  = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed

    def test_val_loss_finite(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)
        assert all(np.isfinite(v) for v in history.val_loss)

    def test_training_history_to_dict(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)
        d = history.to_dict()
        assert "train_loss" in d and "val_loss" in d and "best_epoch" in d
        assert json.dumps(d)  # must be JSON-serializable


# ── Evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_all_metrics(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        train(model, X_train, y_train, X_val, y_val, small_cfg)
        metrics = evaluate(model, X_val, y_val)
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in metrics

    def test_metrics_in_range(self, small_cfg, synthetic_data):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        train(model, X_train, y_train, X_val, y_val, small_cfg)
        metrics = evaluate(model, X_val, y_val)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


class TestRocAuc:
    def test_perfect_auc(self):
        probs  = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc    = _roc_auc(probs, labels)
        assert auc >= 0.95

    def test_worst_auc(self):
        probs  = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        auc    = _roc_auc(probs, labels)
        assert auc <= 0.1  # inverted predictions ≈ 0

    def test_all_same_class_returns_half(self):
        probs  = np.array([0.7, 0.8, 0.9])
        labels = np.array([1.0, 1.0, 1.0])  # all positive
        auc    = _roc_auc(probs, labels)
        assert auc == 0.5


# ── Predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_float_in_range(self, small_cfg):
        model = MLP(small_cfg)
        x = torch.randn(small_cfg.input_size)
        p = predict(model, x)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_accepts_numpy(self, small_cfg):
        model = MLP(small_cfg)
        x = np.random.randn(small_cfg.input_size).astype(np.float32)
        p = predict(model, x)
        assert 0.0 <= p <= 1.0


# ── Checkpoints ───────────────────────────────────────────────────────────────

class TestCheckpoints:
    def test_save_and_load(self, small_cfg, synthetic_data, tmp_path):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        history = train(model, X_train, y_train, X_val, y_val, small_cfg)

        path = tmp_path / "model.pt"
        save_checkpoint(model, path, history=history, extra={"symbols": ["AAPL"]})

        model2, meta = load_checkpoint(path)
        assert "config"  in meta
        assert "history" in meta
        assert meta["extra"]["symbols"] == ["AAPL"]

    def test_loaded_model_produces_same_output(self, small_cfg, synthetic_data, tmp_path):
        model = MLP(small_cfg)
        X_train, y_train, X_val, y_val = synthetic_data
        train(model, X_train, y_train, X_val, y_val, small_cfg)

        path = tmp_path / "model.pt"
        save_checkpoint(model, path)
        model2, _ = load_checkpoint(path)

        x = torch.randn(3, small_cfg.input_size)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)

    def test_config_preserved(self, small_cfg, tmp_path):
        model = MLP(small_cfg)
        path  = tmp_path / "model.pt"
        save_checkpoint(model, path)
        _, meta = load_checkpoint(path)
        cfg2 = MLPConfig.from_dict(meta["config"])
        assert cfg2.input_size    == small_cfg.input_size
        assert cfg2.hidden_sizes  == small_cfg.hidden_sizes


# ── FeatureScaler ─────────────────────────────────────────────────────────────

class TestFeatureScaler:
    def test_fit_transform(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = FeatureScaler()
        Xn = scaler.fit_transform(X)
        # After z-score, mean ≈ 0
        assert np.allclose(Xn.mean(axis=0), 0.0, atol=1e-6)

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError, match="fitted"):
            scaler.transform(np.array([[1.0, 2.0]]))

    def test_zero_variance_column_safe(self):
        X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
        scaler = FeatureScaler()
        Xn = scaler.fit_transform(X)
        # First column all-same → std=0 → std replaced by 1 → column stays 0
        assert np.allclose(Xn[:, 0], 0.0, atol=1e-6)

    def test_to_dict_round_trip(self):
        X = np.random.randn(50, 10).astype(np.float32)
        scaler = FeatureScaler().fit(X)
        d = scaler.to_dict()
        scaler2 = FeatureScaler.from_dict(d)
        Xn1 = scaler.transform(X)
        Xn2 = scaler2.transform(X)
        assert np.allclose(Xn1, Xn2)

    def test_to_dict_json_serializable(self):
        X = np.random.randn(10, 5).astype(np.float32)
        scaler = FeatureScaler().fit(X)
        assert json.dumps(scaler.to_dict())

    def test_fit_transform_chain(self):
        X = np.random.randn(30, 8).astype(np.float32)
        scaler = FeatureScaler()
        Xn1 = scaler.fit_transform(X)
        Xn2 = scaler.transform(X)
        assert np.allclose(Xn1, Xn2)
