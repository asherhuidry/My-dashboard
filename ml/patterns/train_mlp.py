"""End-to-end MLP training CLI for FinBrain.

Downloads price history via yfinance, engineers features, trains the MLP
baseline model, evaluates it, runs a backtest, and logs everything to the
experiment registry.

Usage (CLI)::

    python -m ml.patterns.train_mlp --symbol AAPL --period 3y

Usage (programmatic)::

    from ml.patterns.train_mlp import run_pipeline
    result = run_pipeline(symbol="AAPL", period="3y")
    print(result.experiment_id)

Assumptions
-----------
- yfinance is available and has network access (or you pass a pre-built
  DataFrame via the ``df`` parameter in programmatic mode).
- The experiment registry persists to ``data/registry/experiments.json``
  (or FINBRAIN_EXPERIMENT_REGISTRY_PATH env var).
- No database connection is required — this pipeline runs fully local.
- Target variable: next-day binary direction (1 = close[t+1] > close[t]).
- Train / validation / test split is chronological: 70 / 15 / 15 percent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from ml.backtest.engine import BacktestConfig, run_backtest
from ml.patterns.features import build_features
from ml.patterns.mlp import (
    MLP,
    MLPConfig,
    FeatureScaler,
    evaluate,
    save_checkpoint,
    train,
)
from ml.registry.experiment_registry import ExperimentRegistry

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "ml" / "checkpoints"
SCALER_DIR     = CHECKPOINT_DIR  # scalers stored alongside checkpoints

# Promotion thresholds — a model must beat ALL of these to receive a
# PROMOTION RECOMMENDED flag.  We do NOT auto-promote; the user decides.
PROMO_MIN_ACCURACY    = 0.55   # above random for direction
PROMO_MIN_HIT_RATE    = 0.52   # backtest hit rate
PROMO_MIN_BEAT_BM     = 0.0    # cumulative return > benchmark return


# ── Data assembly ─────────────────────────────────────────────────────────────

def fetch_price_df(symbol: str, period: str = "3y") -> pd.DataFrame:
    """Download OHLCV history from yfinance and normalise column names.

    Args:
        symbol: Ticker symbol, e.g. 'AAPL'.
        period: yfinance period string, e.g. '3y', '2y', '1y'.

    Returns:
        DataFrame with lowercase columns (open, high, low, close, volume)
        and a DatetimeIndex (UTC-aware).

    Raises:
        ValueError: If yfinance returns an empty DataFrame.
    """
    df = yf.Ticker(symbol).history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"yfinance returned empty data for {symbol!r}")
    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    log.info("Fetched %d bars for %s", len(df), symbol)
    return df


def build_dataset(
    df:             pd.DataFrame,
    target_horizon: int = 1,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
) -> dict[str, Any]:
    """Build flat feature matrices and binary targets from an OHLCV DataFrame.

    Feature engineering uses ``ml.patterns.features.build_features``.
    Splits are strictly chronological (no shuffling).

    Args:
        df:             OHLCV DataFrame with lowercase column names.
        target_horizon: How many bars ahead to predict direction.
        train_frac:     Fraction of rows for training.
        val_frac:       Fraction of rows for validation.
                        Remaining rows go to the test set.

    Returns:
        Dict with keys:
            X_train, y_train, X_val, y_val, X_test, y_test  — numpy arrays
            close_test       — close price Series for backtest
            signal_index     — DatetimeIndex for the test window
            feature_cols     — list[str] feature column names
            n_train, n_val, n_test
    """
    feat_df = build_features(df)

    # Drop rows that are all-NaN (rolling-window warm-up period)
    feat_df = feat_df.dropna(how="all")
    # Align df to feat_df index
    df_aligned = df.reindex(feat_df.index)

    # Target: 1 if close rises over next target_horizon bars, else 0
    close    = df_aligned["close"]
    returns  = close.pct_change(target_horizon).shift(-target_horizon)
    target   = (returns > 0).astype(float)

    # Drop rows where target is NaN (end of series)
    valid_mask = target.notna() & feat_df.notna().all(axis=1)
    feat_df    = feat_df.loc[valid_mask]
    target     = target.loc[valid_mask]
    close      = close.loc[valid_mask]

    X_raw = feat_df.values.astype(np.float32)
    y_raw = target.values.astype(np.float32)

    # Replace inf
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)

    n       = len(X_raw)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    return {
        "X_train":      X_raw[:n_train],
        "y_train":      y_raw[:n_train],
        "X_val":        X_raw[n_train : n_train + n_val],
        "y_val":        y_raw[n_train : n_train + n_val],
        "X_test":       X_raw[n_train + n_val :],
        "y_test":       y_raw[n_train + n_val :],
        "close_test":   close.iloc[n_train + n_val :],
        "signal_index": feat_df.index[n_train + n_val :],
        "feature_cols": list(feat_df.columns),
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       n_test,
    }


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Output of a complete train → evaluate → backtest pipeline run.

    Attributes:
        experiment_id:       ID in the experiment registry.
        symbol:              Ticker symbol trained on.
        metrics:             Evaluation metrics dict (accuracy, f1, auc, …).
        backtest_summary:    Serialised BacktestReport dict.
        checkpoint_path:     Path to the saved .pt checkpoint.
        scaler_path:         Path to the saved scaler JSON.
        promotion_recommended: True if all promotion thresholds are met.
        promotion_reasons:   List explaining why promotion was or was not recommended.
    """
    experiment_id:        str
    symbol:               str
    metrics:              dict[str, float]
    backtest_summary:     dict[str, Any]
    checkpoint_path:      str
    scaler_path:          str
    promotion_recommended: bool
    promotion_reasons:    list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        sep = "─" * 60
        print(sep)
        print(f"  Experiment ID : {self.experiment_id}")
        print(f"  Symbol        : {self.symbol}")
        print(f"  Checkpoint    : {self.checkpoint_path}")
        print(f"  Scaler        : {self.scaler_path}")
        print(sep)
        print("  Evaluation metrics:")
        for k, v in self.metrics.items():
            print(f"    {k:18s}: {v:.4f}")
        print(sep)
        print("  Backtest summary:")
        bt = self.backtest_summary
        print(f"    Period        : {bt.get('period_start','?')[:10]} → {bt.get('period_end','?')[:10]}")
        print(f"    Return        : {bt.get('cumulative_return', 0)*100:+.1f}%"
              f"  (benchmark {bt.get('benchmark_return', 0)*100:+.1f}%)")
        print(f"    Annualised    : {bt.get('annualised_return', 0)*100:+.1f}%")
        print(f"    Hit rate      : {bt.get('hit_rate', 0)*100:.1f}%")
        print(f"    Max drawdown  : {bt.get('max_drawdown', 0)*100:.1f}%")
        print(f"    Sharpe        : {bt.get('sharpe', 0):.2f}")
        print(f"    Trades        : {bt.get('trade_count', 0)}")
        print(sep)
        status = "RECOMMENDED" if self.promotion_recommended else "NOT recommended"
        print(f"  Promotion      : {status}")
        for r in self.promotion_reasons:
            print(f"    • {r}")
        print(sep)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    symbol:         str,
    period:         str     = "3y",
    target_horizon: int     = 1,
    hidden_sizes:   list[int] | None = None,
    dropout:        float   = 0.2,
    lr:             float   = 1e-3,
    epochs:         int     = 100,
    patience:       int     = 10,
    batch_size:     int     = 64,
    backtest_threshold: float = 0.55,
    tags:           list[str] | None = None,
    notes:          str     = "",
    df:             pd.DataFrame | None = None,
    registry:       ExperimentRegistry | None = None,
    checkpoint_dir: Path | None = None,
) -> PipelineResult:
    """Run the full MLP training → evaluation → backtest pipeline.

    Args:
        symbol:             Ticker symbol (used for data download and labelling).
        period:             yfinance period string for historical data.
        target_horizon:     Bars ahead to predict direction (default 1 = next day).
        hidden_sizes:       MLP hidden layer widths.  Default: [128, 64].
        dropout:            Dropout rate.
        lr:                 AdamW learning rate.
        epochs:             Max training epochs.
        patience:           Early-stopping patience.
        batch_size:         Mini-batch size.
        backtest_threshold: Signal score threshold for the backtest (go long ≥ this).
        tags:               Optional string tags to attach to the experiment record.
        notes:              Optional free-text notes.
        df:                 Pre-built OHLCV DataFrame (skips yfinance download).
                            Must have lowercase columns + DatetimeIndex.
        registry:           Optional pre-built ExperimentRegistry (for testing).

    Returns:
        PipelineResult with all artifacts and a promotion recommendation.
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]

    reg      = registry if registry is not None else ExperimentRegistry()
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR

    # ── 1. Download / prepare data ────────────────────────────────────────
    if df is None:
        df = fetch_price_df(symbol, period=period)

    dataset = build_dataset(df, target_horizon=target_horizon)

    if dataset["n_test"] < 20:
        raise ValueError(
            f"Test set has only {dataset['n_test']} samples — need at least 20. "
            "Try a longer period or shorter target_horizon."
        )

    n_features = dataset["X_train"].shape[1]

    # ── 2. Normalise ──────────────────────────────────────────────────────
    scaler  = FeatureScaler()
    X_train = scaler.fit_transform(dataset["X_train"]).astype(np.float32)
    X_val   = scaler.transform(dataset["X_val"]).astype(np.float32)
    X_test  = scaler.transform(dataset["X_test"]).astype(np.float32)

    # ── 3. Create experiment record ───────────────────────────────────────
    cfg = MLPConfig(
        input_size   = n_features,
        hidden_sizes = hidden_sizes,
        dropout      = dropout,
        lr           = lr,
        epochs       = epochs,
        patience     = patience,
        batch_size   = batch_size,
    )
    exp = reg.create(
        name         = f"mlp_{symbol.lower()}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M')}",
        model_type   = "mlp",
        hyperparams  = cfg.to_dict(),
        dataset_info = {
            "symbol":         symbol,
            "period":         period,
            "n_bars":         len(df),
            "n_features":     n_features,
            "n_train":        dataset["n_train"],
            "n_val":          dataset["n_val"],
            "n_test":         dataset["n_test"],
            "target_horizon": target_horizon,
            "feature_cols":   dataset["feature_cols"][:10],  # first 10 for brevity
        },
        notes = notes,
        tags  = tags or [],
    )
    log.info("Experiment started: %s", exp.experiment_id)

    try:
        # ── 4. Train ──────────────────────────────────────────────────────
        model = MLP(cfg)
        Xt = torch.from_numpy(X_train)
        yt = torch.from_numpy(dataset["y_train"])
        Xv = torch.from_numpy(X_val)
        yv = torch.from_numpy(dataset["y_val"])

        history = train(model, Xt, yt, Xv, yv, cfg)

        # ── 5. Evaluate on held-out test set ──────────────────────────────
        X_test_t = torch.from_numpy(X_test)
        y_test_t = torch.from_numpy(dataset["y_test"])
        metrics  = evaluate(model, X_test_t, y_test_t)
        log.info("Evaluation: %s", metrics)

        # ── 6. Save checkpoint + scaler ───────────────────────────────────
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path   = ckpt_dir / f"{exp.experiment_id}.pt"
        scaler_path = ckpt_dir / f"{exp.experiment_id}_scaler.json"

        save_checkpoint(model, ckpt_path, history=history,
                        extra={"symbol": symbol, "feature_cols": dataset["feature_cols"]})
        scaler_path.write_text(json.dumps(scaler.to_dict()), encoding="utf-8")

        # ── 7. Finish experiment ──────────────────────────────────────────
        reg.finish(
            exp.experiment_id,
            metrics         = metrics,
            checkpoint_path = str(ckpt_path),
        )

        # ── 8. Backtest on test set ───────────────────────────────────────
        # Generate probability scores for each test bar, then use them as signals.
        model.eval()
        with torch.no_grad():
            scores = model.predict_proba(X_test_t).numpy()

        close_test = dataset["close_test"]
        signals    = pd.Series(scores, index=close_test.index)

        bt_report = run_backtest(
            prices  = close_test,
            signals = signals,
            config  = BacktestConfig(threshold=backtest_threshold),
        )
        reg.attach_backtest(exp.experiment_id, bt_report.to_backtest_result())

        # ── 9. Promotion recommendation ───────────────────────────────────
        reasons: list[str] = []
        ok_acc    = metrics.get("accuracy", 0) >= PROMO_MIN_ACCURACY
        ok_hit    = bt_report.hit_rate          >= PROMO_MIN_HIT_RATE
        ok_beat   = bt_report.cumulative_return >  bt_report.benchmark_return + PROMO_MIN_BEAT_BM

        if ok_acc:
            reasons.append(f"accuracy {metrics['accuracy']:.3f} ≥ {PROMO_MIN_ACCURACY}")
        else:
            reasons.append(f"accuracy {metrics['accuracy']:.3f} < {PROMO_MIN_ACCURACY} (threshold)")

        if ok_hit:
            reasons.append(f"hit rate {bt_report.hit_rate:.3f} ≥ {PROMO_MIN_HIT_RATE}")
        else:
            reasons.append(f"hit rate {bt_report.hit_rate:.3f} < {PROMO_MIN_HIT_RATE} (threshold)")

        if ok_beat:
            reasons.append(
                f"return {bt_report.cumulative_return*100:+.1f}% beats benchmark "
                f"{bt_report.benchmark_return*100:+.1f}%"
            )
        else:
            reasons.append(
                f"return {bt_report.cumulative_return*100:+.1f}% does NOT beat benchmark "
                f"{bt_report.benchmark_return*100:+.1f}%"
            )

        promote = ok_acc and ok_hit and ok_beat

        return PipelineResult(
            experiment_id         = exp.experiment_id,
            symbol                = symbol,
            metrics               = metrics,
            backtest_summary      = bt_report.to_dict(),
            checkpoint_path       = str(ckpt_path),
            scaler_path           = str(scaler_path),
            promotion_recommended = promote,
            promotion_reasons     = reasons,
        )

    except Exception as exc:
        reg.finish(exp.experiment_id, failed=True, failure_reason=str(exc))
        log.error("Pipeline failed for %s: %s", symbol, exc, exc_info=True)
        raise


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "train_mlp",
        description = "Train the MLP baseline model on OHLCV data.",
    )
    p.add_argument("--symbol",    default="AAPL",   help="Ticker symbol (default: AAPL)")
    p.add_argument("--period",    default="3y",     help="yfinance period string (default: 3y)")
    p.add_argument("--horizon",   type=int, default=1,
                   help="Target horizon in bars (default: 1 = next-day direction)")
    p.add_argument("--hidden",    nargs="+", type=int, default=[128, 64],
                   help="Hidden layer widths (default: 128 64)")
    p.add_argument("--dropout",   type=float, default=0.2, help="Dropout rate (default: 0.2)")
    p.add_argument("--lr",        type=float, default=1e-3, help="Learning rate (default: 0.001)")
    p.add_argument("--epochs",    type=int,   default=100,  help="Max epochs (default: 100)")
    p.add_argument("--patience",  type=int,   default=10,   help="Early stopping patience (default: 10)")
    p.add_argument("--batch",     type=int,   default=64,   help="Batch size (default: 64)")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Backtest signal threshold (default: 0.55)")
    p.add_argument("--tags",      nargs="*",  default=[],   help="Experiment tags")
    p.add_argument("--notes",     default="",               help="Free-text notes")
    p.add_argument("--json",      action="store_true",      help="Output result as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success, 1 on failure.
    """
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    try:
        result = run_pipeline(
            symbol             = args.symbol,
            period             = args.period,
            target_horizon     = args.horizon,
            hidden_sizes       = args.hidden,
            dropout            = args.dropout,
            lr                 = args.lr,
            epochs             = args.epochs,
            patience           = args.patience,
            batch_size         = args.batch,
            backtest_threshold = args.threshold,
            tags               = args.tags or [],
            notes              = args.notes,
        )
        if args.json:
            print(json.dumps({
                "experiment_id":        result.experiment_id,
                "symbol":               result.symbol,
                "metrics":              result.metrics,
                "backtest":             result.backtest_summary,
                "checkpoint_path":      result.checkpoint_path,
                "scaler_path":          result.scaler_path,
                "promotion_recommended": result.promotion_recommended,
                "promotion_reasons":    result.promotion_reasons,
            }, indent=2))
        else:
            result.print_summary()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
