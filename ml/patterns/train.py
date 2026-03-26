"""FinBrain LSTM training script.

Usage:
    python -m ml.patterns.train
    python -m ml.patterns.train --assets AAPL NVDA BTC-USD --epochs 100

Trains a FinBrainLSTM on rolling feature sequences, evaluates on a
held-out chronological split, saves the best checkpoint, and logs
results to the model_registry table in Supabase.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import DataLoader, ConcatDataset

from ml.patterns.dataset import PriceSequenceDataset, chronological_split
from ml.patterns.features import build_features
from ml.patterns.lstm import FinBrainLSTM, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "ml" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ASSETS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "SPY",
    "BTC-USD", "ETH-USD", "SOL-USD",
]

DEFAULT_CONFIG = {
    "seq_len":       20,
    "hidden_size":   128,
    "num_layers":    2,
    "dropout":       0.3,
    "lr":            1e-3,
    "batch_size":    64,
    "epochs":        80,
    "patience":      12,
    "train_frac":    0.80,
    "target_horizon":1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_asset(symbol: str, period: str = "2y") -> tuple[pd.DataFrame, pd.Series] | None:
    """Fetch OHLCV from yfinance and compute features.

    Returns:
        (feature_df, close_series) or None if data unavailable.
    """
    try:
        df = yf.Ticker(symbol).history(period=period)
        if len(df) < 100:
            log.warning("Too few rows for %s (%d)", symbol, len(df))
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
        feats = build_features(ohlcv).dropna(how="all")
        close = ohlcv["close"].loc[feats.index]
        log.info("  %s: %d rows → %d feature rows", symbol, len(df), len(feats))
        return feats, close
    except Exception as e:
        log.warning("Failed to load %s: %s", symbol, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits >= 0.5).float()
    return (preds == targets).float().mean().item()


def train(
    assets:  list[str] = DEFAULT_ASSETS,
    config:  dict      = DEFAULT_CONFIG,
    version: str       = "v1",
) -> dict:
    """Run the full training pipeline.

    Args:
        assets:  List of ticker symbols to train on.
        config:  Hyperparameter dict.
        version: Model version string (used in checkpoint filename).

    Returns:
        Dict with final metrics and checkpoint path.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load & build datasets ─────────────────────────────────────────────
    log.info("Loading %d assets…", len(assets))
    train_sets, val_sets = [], []
    shared_mean = shared_std = None
    feature_cols: list[str] = []

    raw_data = []
    for sym in assets:
        result = _load_asset(sym)
        if result:
            raw_data.append(result)

    if not raw_data:
        raise RuntimeError("No assets loaded successfully")

    # Fit scaler on combined training portion first
    all_feats = pd.concat([fd for fd, _ in raw_data], axis=0)
    all_feats_num = all_feats.select_dtypes(include=[np.number]).ffill().bfill().fillna(0.0)
    all_feats_num = all_feats_num.replace([np.inf, -np.inf], 0.0)
    shared_mean  = all_feats_num.values.astype(np.float32).mean(axis=0)
    shared_std   = all_feats_num.values.astype(np.float32).std(axis=0)
    feature_cols = list(all_feats_num.columns)

    for feat_df, close in raw_data:
        ds = PriceSequenceDataset(
            feat_df, close,
            seq_len        = config["seq_len"],
            target_horizon = config["target_horizon"],
            scaler_mean    = shared_mean,
            scaler_std     = shared_std,
        )
        if len(ds) < 50:
            continue
        tr, vl = chronological_split(ds, config["train_frac"])
        train_sets.append(tr)
        val_sets.append(vl)

    if not train_sets:
        raise RuntimeError("All datasets too small after splitting")

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=config["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(ConcatDataset(val_sets),   batch_size=config["batch_size"], shuffle=False)

    n_features = train_sets[0].n_features
    log.info("Features: %d | Train batches: %d | Val batches: %d",
             n_features, len(train_loader), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = FinBrainLSTM(
        input_size  = n_features,
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_left = config["patience"]
    ckpt_path     = CHECKPOINT_DIR / f"finbrain_lstm_{version}.pt"
    history: list[dict] = []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_acc  += _accuracy(out, y)
        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        # Validate
        model.eval()
        vl_loss, vl_acc = 0.0, 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out   = model(X)
                vl_loss += criterion(out, y).item()
                vl_acc  += _accuracy(out, y)
        vl_loss /= len(val_loader)
        vl_acc  /= len(val_loader)

        scheduler.step(vl_loss)

        row = {"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
               "val_loss": round(vl_loss, 4), "val_acc": round(vl_acc, 4)}
        history.append(row)

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d | train loss %.4f acc %.3f | val loss %.4f acc %.3f",
                     epoch, tr_loss, tr_acc, vl_loss, vl_acc)

        # Best model
        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_epoch    = epoch
            patience_left = config["patience"]
            save_checkpoint(
                path         = ckpt_path,
                model        = model,
                optimizer    = optimizer,
                epoch        = epoch,
                metrics      = row,
                scaler_mean  = shared_mean,
                scaler_std   = shared_std,
                feature_cols = feature_cols,
                config       = {**config, "input_size": n_features, "version": version},
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping at epoch %d (best val acc %.3f at epoch %d)",
                         epoch, best_val_acc, best_epoch)
                break

    log.info("Training complete. Best val accuracy: %.3f (epoch %d)", best_val_acc, best_epoch)

    # ── Log to Supabase model_registry ────────────────────────────────────
    metrics = {
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch":        best_epoch,
        "final_train_acc":   round(tr_acc, 4),
        "epochs_run":        epoch,
        "n_assets":          len(raw_data),
        "n_features":        n_features,
        "history_last_10":   history[-10:],
    }
    _log_to_registry(version, assets, config, metrics, str(ckpt_path))

    return {"checkpoint": str(ckpt_path), "metrics": metrics}


def _log_to_registry(version: str, assets: list[str], config: dict,
                     metrics: dict, checkpoint_path: str) -> None:
    """Write model metadata to Supabase model_registry."""
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            log.warning("Supabase credentials not set — skipping registry log")
            return
        client = create_client(url, key)
        client.table("model_registry").upsert({
            "model_id":        f"lstm_{version}",
            "model_type":      "lstm",
            "version":         version,
            "assets":          json.dumps(assets),
            "config":          json.dumps(config),
            "metrics":         json.dumps(metrics),
            "checkpoint_path": checkpoint_path,
            "trained_at":      datetime.now(timezone.utc).isoformat(),
            "status":          "active",
        }, on_conflict="model_id").execute()
        log.info("Logged to model_registry as lstm_%s", version)
    except Exception as e:
        log.warning("Failed to log to model_registry: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets",  nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--epochs",  type=int,  default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--version", type=str,  default="v1")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, "epochs": args.epochs}
    result = train(assets=args.assets, config=cfg, version=args.version)
    print(f"\nCheckpoint: {result['checkpoint']}")
    print(f"Metrics: {result['metrics']}")
