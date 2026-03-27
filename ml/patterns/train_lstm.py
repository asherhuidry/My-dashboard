"""Registry-integrated LSTM training pipeline.

Thin adapter that connects the existing ``FinBrainLSTM`` model and
``PriceSequenceDataset`` to the ``ExperimentRegistry`` and backtest engine,
producing the same ``PipelineResult`` as the MLP pipeline.

The existing ``ml/patterns/train.py`` trains the LSTM for multi-asset
production use and logs to Supabase.  This module is for *research*
comparisons: single-symbol, registry-tracked, with a proper held-out test
set to match the MLP evaluation setup.

Split strategy (default 70 / 15 / 15):
- The val set is used exclusively for early stopping.
- The test set is the held-out evaluation window, matching the MLP pipeline.

Usage::

    from ml.patterns.train_lstm import run_lstm_pipeline, LSTMConfig

    result = run_lstm_pipeline(
        symbol         = "AAPL",
        df             = ohlcv_df,
        registry       = reg,
        checkpoint_dir = Path("ml/checkpoints"),
        config         = LSTMConfig(seq_len=10, hidden_size=32, epochs=5),
    )
    print(result.metrics)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ml.backtest.engine import BacktestConfig, run_backtest
from ml.data.dataset_builder import DatasetMeta, assemble_sequence_dataset
from ml.patterns.lstm import FinBrainLSTM
from ml.patterns.lstm import save_checkpoint as _lstm_save_checkpoint
from ml.patterns.train_mlp import (
    PROMO_MIN_ACCURACY,
    PROMO_MIN_BEAT_BM,
    PROMO_MIN_HIT_RATE,
    PipelineResult,
    fetch_price_df,
)
from ml.registry.experiment_registry import ExperimentRegistry

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "ml" / "checkpoints"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """Hyperparameters for the registry-integrated LSTM pipeline.

    Attributes:
        seq_len:     Rolling window length in trading days (default 20).
        hidden_size: LSTM hidden state dimension (default 128).
        num_layers:  Number of stacked LSTM layers (default 2).
        dropout:     Dropout probability applied between layers and in head.
        lr:          AdamW learning rate.
        epochs:      Maximum training epochs.
        patience:    Early-stopping patience (epochs without val loss improvement).
        batch_size:  Mini-batch size for training and evaluation.
        train_frac:  Fraction of sequences for training (default 0.70).
        val_frac:    Fraction of sequences for validation / early stopping
                     (default 0.15).  Remaining sequences go to the test set.
    """
    seq_len:    int   = 20
    hidden_size: int  = 128
    num_layers:  int  = 2
    dropout:     float = 0.3
    lr:          float = 1e-3
    epochs:      int   = 50
    patience:    int   = 10
    batch_size:  int   = 64
    train_frac:  float = 0.70
    val_frac:    float = 0.15

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LSTMConfig":
        """Deserialise from a dict."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_lstm(
    model:    FinBrainLSTM,
    train_ds: Any,
    val_ds:   Any,
    cfg:      LSTMConfig,
) -> dict[str, Any]:
    """Train a FinBrainLSTM with early stopping on validation loss.

    ``FinBrainLSTM`` applies ``nn.Sigmoid`` in its head, so this function
    uses ``nn.BCELoss`` (not ``BCEWithLogitsLoss``).

    Args:
        model:    Initialised (untrained) FinBrainLSTM.
        train_ds: Training dataset (``PriceSequenceDataset`` or slice).
        val_ds:   Validation dataset.
        cfg:      Training configuration.

    Returns:
        History dict with ``train_losses``, ``val_losses``, ``val_accs``,
        and ``best_epoch``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    criterion    = nn.BCELoss()
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    best_val_loss   = float("inf")
    best_state_dict: dict | None = None
    patience_count  = 0
    history: dict[str, Any] = {
        "train_losses": [], "val_losses": [], "val_accs": [], "best_epoch": 0
    }

    for epoch in range(cfg.epochs):
        # ── training pass ──────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item() * len(X_batch)
        train_loss = ep_loss / max(len(train_ds), 1)

        # ── validation pass ────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds   = model(X_batch)
                val_loss   += criterion(preds, y_batch).item() * len(X_batch)
                val_correct += ((preds >= 0.5).float() == y_batch).sum().item()
                val_total   += y_batch.numel()
        val_loss /= max(len(val_ds), 1)
        val_acc   = val_correct / max(val_total, 1)

        history["train_losses"].append(round(train_loss, 6))
        history["val_losses"].append(round(val_loss,   6))
        history["val_accs"].append(round(val_acc,      4))

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                log.info("LSTM early stop at epoch %d (patience=%d)", epoch, cfg.patience)
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    log.info(
        "LSTM training done: best_epoch=%d  best_val_loss=%.4f",
        history["best_epoch"], best_val_loss,
    )
    return history


def _evaluate_lstm(
    model:     FinBrainLSTM,
    dataset:   Any,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate a trained LSTM on a dataset.

    Args:
        model:     Trained FinBrainLSTM in eval mode.
        dataset:   Any Dataset with ``__len__`` and ``__getitem__``.
        threshold: Decision threshold for binary predictions.

    Returns:
        ``(metrics, proba_array)`` — metrics dict matching the MLP interface
        and raw float32 probability scores (used for backtesting).
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    device = next(model.parameters()).device
    all_probs:  list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs  = model(X_batch.to(device)).squeeze().cpu().numpy()
            labels = y_batch.squeeze().numpy()
            all_probs.append(np.atleast_1d(probs.astype(np.float32)))
            all_labels.append(np.atleast_1d(labels.astype(np.float32)))

    probs_arr  = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds_arr  = (probs_arr >= threshold).astype(int)
    labels_int = labels_arr.astype(int)

    auc = (
        float(roc_auc_score(labels_int, probs_arr))
        if len(np.unique(labels_int)) > 1
        else 0.5
    )
    metrics = {
        "accuracy":  float(accuracy_score(labels_int, preds_arr)),
        "precision": float(precision_score(labels_int, preds_arr, zero_division=0)),
        "recall":    float(recall_score(labels_int, preds_arr, zero_division=0)),
        "f1":        float(f1_score(labels_int, preds_arr, zero_division=0)),
        "auc":       auc,
    }
    return metrics, probs_arr


# ── End-to-end pipeline ───────────────────────────────────────────────────────

def run_lstm_pipeline(
    symbol:             str,
    period:             str                       = "3y",
    target_horizon:     int                       = 1,
    backtest_threshold: float                     = 0.55,
    tags:               list[str] | None          = None,
    notes:              str                       = "",
    df:                 pd.DataFrame | None       = None,
    dataset_meta:       DatasetMeta | None        = None,
    registry:           ExperimentRegistry | None = None,
    checkpoint_dir:     Path | None               = None,
    config:             LSTMConfig | None         = None,
) -> PipelineResult:
    """Run the full LSTM training → evaluate → backtest pipeline.

    Mirrors ``ml.patterns.train_mlp.run_pipeline`` in structure and returns
    the same ``PipelineResult``, so LSTM experiments can be directly compared
    with MLP experiments in the ranking and comparison flow.

    Evaluation note: the LSTM uses a held-out test set (default 15%) that
    is never seen during training or early stopping.  The test set close
    prices are used for the backtest, matching the MLP evaluation setup.

    Args:
        symbol:             Ticker symbol.
        period:             yfinance period string (ignored if ``df`` is given).
        target_horizon:     Bars ahead to predict direction.
        backtest_threshold: Signal threshold for long entry in the backtest.
        tags:               Registry tags applied to the experiment record.
        notes:              Free-text notes.
        df:                 Pre-built OHLCV DataFrame; fetched if ``None``.
        dataset_meta:       Flat ``DatasetMeta`` from ``assemble_dataset``
                            (for cross-model version linkage in the registry).
                            The sequence meta is always recorded; this is an
                            optional cross-reference.
        registry:           ``ExperimentRegistry``; created locally if ``None``.
        checkpoint_dir:     Directory for ``.pt`` checkpoints.
        config:             ``LSTMConfig`` hyperparameters.

    Returns:
        ``PipelineResult`` with metrics, backtest, checkpoint paths, and
        promotion recommendation.

    Raises:
        ValueError: If the test set has fewer than 20 sequences.
    """
    reg      = registry if registry is not None else ExperimentRegistry()
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    cfg      = config or LSTMConfig()

    # ── 1. Data ───────────────────────────────────────────────────────────
    if df is None:
        df = fetch_price_df(symbol, period=period)

    train_ds, val_ds, test_ds, close_test, seq_meta = assemble_sequence_dataset(
        df             = df,
        symbol         = symbol,
        seq_len        = cfg.seq_len,
        target_horizon = target_horizon,
        train_frac     = cfg.train_frac,
        val_frac       = cfg.val_frac,
    )

    if len(test_ds) < 20:
        raise ValueError(
            f"LSTM test set has only {len(test_ds)} sequences — need at least 20. "
            "Try a longer period or a shorter seq_len."
        )

    # dataset_info: prefer cross-referencing the flat dataset version so all
    # models sharing the same OHLCV window are linked in the registry
    ds_info = seq_meta.to_dataset_info()
    if dataset_meta is not None:
        ds_info["comparison_group_version"] = dataset_meta.dataset_version

    # ── 2. Create experiment record ───────────────────────────────────────
    exp = reg.create(
        name=(
            f"lstm_{symbol.lower()}_"
            f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M')}"
        ),
        model_type   = "lstm",
        hyperparams  = cfg.to_dict(),
        dataset_info = ds_info,
        notes        = notes,
        tags         = tags or [],
    )
    log.info("LSTM experiment started: %s", exp.experiment_id)

    try:
        # ── 3. Train ──────────────────────────────────────────────────────
        n_features = train_ds.n_features
        model = FinBrainLSTM(
            input_size  = n_features,
            hidden_size = cfg.hidden_size,
            num_layers  = cfg.num_layers,
            dropout     = cfg.dropout,
        )
        history = _train_lstm(model, train_ds, val_ds, cfg)

        # ── 4. Evaluate on held-out test set ──────────────────────────────
        metrics, probs_arr = _evaluate_lstm(model, test_ds)
        log.info("LSTM evaluation (test): %s", metrics)

        # ── 5. Save checkpoint ────────────────────────────────────────────
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path   = ckpt_dir / f"{exp.experiment_id}_lstm.pt"
        scaler_path = ckpt_dir / f"{exp.experiment_id}_lstm_scaler.json"

        # Save full training state (reuse lstm.save_checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        _lstm_save_checkpoint(
            path         = ckpt_path,
            model        = model,
            optimizer    = optimizer,
            epoch        = history["best_epoch"],
            metrics      = metrics,
            scaler_mean  = train_ds.scaler_mean,
            scaler_std   = train_ds.scaler_std,
            feature_cols = train_ds.feature_cols,
            config       = {"input_size": n_features, **cfg.to_dict()},
        )
        scaler_path.write_text(
            json.dumps({
                "scaler_mean": train_ds.scaler_mean.tolist(),
                "scaler_std":  train_ds.scaler_std.tolist(),
                "feature_cols": train_ds.feature_cols,
            }),
            encoding="utf-8",
        )

        # ── 6. Finish experiment ──────────────────────────────────────────
        reg.finish(
            exp.experiment_id,
            metrics         = metrics,
            checkpoint_path = str(ckpt_path),
        )

        # ── 7. Backtest on test set ───────────────────────────────────────
        # Align probs_arr to close_test length (guards against NaN-skip edge cases)
        n_bt = min(len(probs_arr), len(close_test))
        signals = pd.Series(
            probs_arr[:n_bt],
            index=close_test.index[:n_bt],
        )
        bt_report = run_backtest(
            prices  = close_test.iloc[:n_bt],
            signals = signals,
            config  = BacktestConfig(threshold=backtest_threshold),
        )
        reg.attach_backtest(exp.experiment_id, bt_report.to_backtest_result())

        # ── 8. Promotion recommendation ───────────────────────────────────
        reasons: list[str] = []
        ok_acc  = metrics.get("accuracy", 0) >= PROMO_MIN_ACCURACY
        ok_hit  = bt_report.hit_rate          >= PROMO_MIN_HIT_RATE
        ok_beat = bt_report.cumulative_return > bt_report.benchmark_return + PROMO_MIN_BEAT_BM

        reasons.append(
            f"accuracy {metrics['accuracy']:.3f} "
            f"{'≥' if ok_acc else '<'} {PROMO_MIN_ACCURACY}"
        )
        reasons.append(
            f"hit rate {bt_report.hit_rate:.3f} "
            f"{'≥' if ok_hit else '<'} {PROMO_MIN_HIT_RATE}"
        )
        reasons.append(
            f"return {bt_report.cumulative_return*100:+.1f}% "
            f"{'beats' if ok_beat else 'does NOT beat'} benchmark "
            f"{bt_report.benchmark_return*100:+.1f}%"
        )

        return PipelineResult(
            experiment_id         = exp.experiment_id,
            symbol                = symbol,
            metrics               = metrics,
            backtest_summary      = bt_report.to_dict(),
            checkpoint_path       = str(ckpt_path),
            scaler_path           = str(scaler_path),
            promotion_recommended = ok_acc and ok_hit and ok_beat,
            promotion_reasons     = reasons,
        )

    except Exception as exc:
        reg.finish(exp.experiment_id, failed=True, failure_reason=str(exc))
        log.error("LSTM pipeline failed for %s: %s", symbol, exc, exc_info=True)
        raise
