"""Walk-forward model evaluation runner.

Evaluates baseline, MLP, and (optionally) LSTM across multiple chronological
folds and collects structured per-fold results.

Each fold uses a scaler fit **only** on that fold's training rows so no
look-ahead bias is introduced via normalisation.

LSTM support
------------
LSTM walk-forward is supported on a best-effort basis.  If a fold does not
contain enough sequences (≥ 20 per split) the fold is skipped with a warning
and the ``WalkForwardResult`` reflects fewer completed folds.  Baseline and
MLP folds are always attempted for every fold.

Usage::

    from ml.validation.walk_forward import WalkForwardConfig
    from ml.validation.wf_runner import run_walk_forward, run_walk_forward_model

    cfg = WalkForwardConfig(n_folds=5, min_train_size=120)

    # Single model
    wf = run_walk_forward_model("baseline", symbol="AAPL", df=ohlcv_df, config=cfg)
    print(wf.n_folds_run, wf.fold_results[0].metrics)

    # Multiple models
    results = run_walk_forward(
        symbol  = "AAPL",
        df      = ohlcv_df,
        models  = ("baseline", "mlp"),
        config  = cfg,
        epochs  = 20,
        patience= 5,
    )
    # results is dict[model_type, WalkForwardResult]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.backtest.engine import BacktestConfig, run_backtest
from ml.patterns.mlp import FeatureScaler
from ml.patterns.train_mlp import (
    PROMO_MIN_ACCURACY,
    PROMO_MIN_BEAT_BM,
    PROMO_MIN_HIT_RATE,
    fetch_price_df,
)
from ml.registry.experiment_registry import ExperimentRegistry
from ml.validation.walk_forward import (
    FoldSpec,
    WalkForwardConfig,
    assemble_walk_forward_dataset,
    make_folds,
    slice_fold,
)

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Evaluation result for one model on one walk-forward fold.

    Attributes:
        fold_idx:              Zero-based fold number.
        fold_spec:             The ``FoldSpec`` that generated this fold.
        model_type:            ``'logistic'``, ``'mlp'``, or ``'lstm'``.
        metrics:               Evaluation metrics dict (accuracy, auc, f1, …).
        backtest_summary:      Serialised backtest dict.
        promotion_recommended: True when all single-fold promotion gates pass.
        promotion_reasons:     Per-gate pass/fail messages.
        experiment_id:         Registry experiment ID, or empty string when
                               ``register_folds=False``.
    """
    fold_idx:              int
    fold_spec:             FoldSpec
    model_type:            str
    metrics:               dict[str, float]
    backtest_summary:      dict[str, Any]
    promotion_recommended: bool
    promotion_reasons:     list[str]
    experiment_id:         str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "fold_idx":              self.fold_idx,
            "fold_spec":             self.fold_spec.to_dict(),
            "model_type":            self.model_type,
            "metrics":               self.metrics,
            "backtest_summary":      self.backtest_summary,
            "promotion_recommended": self.promotion_recommended,
            "promotion_reasons":     self.promotion_reasons,
            "experiment_id":         self.experiment_id,
        }


@dataclass
class WalkForwardResult:
    """All fold results for one model in a walk-forward evaluation run.

    Attributes:
        symbol:        Ticker symbol.
        model_type:    ``'logistic'``, ``'mlp'``, or ``'lstm'``.
        n_folds_total: Number of folds that were requested / available.
        n_folds_run:   Number of folds that completed successfully.
        fold_results:  Per-fold results in chronological order.
        config:        ``WalkForwardConfig`` used for this run.
        generated_at:  ISO-8601 timestamp.
    """
    symbol:        str
    model_type:    str
    n_folds_total: int
    n_folds_run:   int
    fold_results:  list[FoldResult]
    config:        WalkForwardConfig
    generated_at:  str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "symbol":        self.symbol,
            "model_type":    self.model_type,
            "n_folds_total": self.n_folds_total,
            "n_folds_run":   self.n_folds_run,
            "fold_results":  [fr.to_dict() for fr in self.fold_results],
            "config":        self.config.to_dict(),
            "generated_at":  self.generated_at,
        }


# ── Per-fold training helpers ─────────────────────────────────────────────────

def _promotion_reasons(
    metrics:   dict[str, float],
    bt_report: Any,
) -> tuple[bool, list[str]]:
    """Compute single-fold promotion recommendation and reason strings."""
    ok_acc  = metrics.get("accuracy", 0.0) >= PROMO_MIN_ACCURACY
    ok_hit  = bt_report.hit_rate >= PROMO_MIN_HIT_RATE
    ok_beat = bt_report.cumulative_return > bt_report.benchmark_return + PROMO_MIN_BEAT_BM
    reasons = [
        f"accuracy {metrics.get('accuracy', 0):.3f} "
        f"{'≥' if ok_acc else '<'} {PROMO_MIN_ACCURACY}",
        f"hit rate {bt_report.hit_rate:.3f} "
        f"{'≥' if ok_hit else '<'} {PROMO_MIN_HIT_RATE}",
        f"return {bt_report.cumulative_return*100:+.1f}% "
        f"{'beats' if ok_beat else 'does NOT beat'} benchmark "
        f"{bt_report.benchmark_return*100:+.1f}%",
    ]
    return ok_acc and ok_hit and ok_beat, reasons


def _scaled_arrays(
    fold_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit scaler on training rows; return (X_train, X_val, X_test) scaled."""
    scaler  = FeatureScaler()
    X_train = scaler.fit_transform(fold_data["X_train"]).astype(np.float32)
    X_val   = (
        scaler.transform(fold_data["X_val"]).astype(np.float32)
        if len(fold_data["X_val"]) > 0
        else fold_data["X_val"]
    )
    X_test  = scaler.transform(fold_data["X_test"]).astype(np.float32)
    return X_train, X_val, X_test


def _eval_baseline_fold(
    fold_data:          dict[str, Any],
    baseline_config:    Any | None        = None,
    backtest_threshold: float             = 0.55,
) -> tuple[dict[str, float], dict[str, Any], bool, list[str]]:
    """Train and evaluate a logistic baseline on one fold.

    Args:
        fold_data:          Output of ``slice_fold``.
        baseline_config:    ``BaselineConfig``; defaults used if ``None``.
        backtest_threshold: Signal threshold for the backtest engine.

    Returns:
        ``(metrics, backtest_summary_dict, promotion_recommended, reasons)``
    """
    from ml.patterns.baseline import BaselineConfig, LogisticBaseline

    cfg   = baseline_config or BaselineConfig()
    X_tr, _, X_te = _scaled_arrays(fold_data)

    model = LogisticBaseline(cfg)
    model.fit(X_tr, fold_data["y_train"])
    metrics = model.evaluate(X_te, fold_data["y_test"])

    probs      = model.predict_proba(X_te)
    close_test = fold_data["close_test"]
    signals    = pd.Series(probs, index=close_test.index)
    bt         = run_backtest(close_test, signals, BacktestConfig(threshold=backtest_threshold))

    promo, reasons = _promotion_reasons(metrics, bt)
    return metrics, bt.to_dict(), promo, reasons


def _eval_mlp_fold(
    fold_data:          dict[str, Any],
    epochs:             int   = 20,
    patience:           int   = 5,
    backtest_threshold: float = 0.55,
) -> tuple[dict[str, float], dict[str, Any], bool, list[str]]:
    """Train and evaluate an MLP on one fold.

    Uses a smaller architecture than the full pipeline (hidden [64, 32]) for
    efficient fold-level training.  Early stopping uses the fold's validation
    rows; if validation is empty, the last 15 % of training rows are used
    internally.

    Args:
        fold_data:          Output of ``slice_fold``.
        epochs:             Maximum training epochs per fold.
        patience:           Early-stopping patience.
        backtest_threshold: Signal threshold for the backtest engine.

    Returns:
        ``(metrics, backtest_summary_dict, promotion_recommended, reasons)``
    """
    import torch
    from ml.patterns.mlp import MLP, MLPConfig, evaluate, train

    X_tr, X_va, X_te = _scaled_arrays(fold_data)
    y_tr = fold_data["y_train"]
    y_va = fold_data["y_val"]

    # If no validation data, carve off the last 15 % of training rows
    if len(X_va) == 0:
        split = max(1, int(len(X_tr) * 0.85))
        X_va  = X_tr[split:]
        y_va  = y_tr[split:]
        X_tr  = X_tr[:split]
        y_tr  = y_tr[:split]

    n_features = X_tr.shape[1]
    cfg = MLPConfig(
        input_size   = n_features,
        hidden_sizes = [64, 32],
        dropout      = 0.2,
        lr           = 1e-3,
        epochs       = epochs,
        patience     = patience,
        batch_size   = 64,
    )

    model = MLP(cfg)
    train(
        model,
        torch.from_numpy(X_tr), torch.from_numpy(y_tr),
        torch.from_numpy(X_va), torch.from_numpy(y_va),
        cfg,
    )

    X_te_t = torch.from_numpy(X_te)
    y_te_t = torch.from_numpy(fold_data["y_test"])
    metrics = evaluate(model, X_te_t, y_te_t)

    model.eval()
    with torch.no_grad():
        scores = model.predict_proba(X_te_t).numpy()

    close_test = fold_data["close_test"]
    signals    = pd.Series(scores, index=close_test.index)
    bt         = run_backtest(close_test, signals, BacktestConfig(threshold=backtest_threshold))

    promo, reasons = _promotion_reasons(metrics, bt)
    return metrics, bt.to_dict(), promo, reasons


def _eval_lstm_fold(
    fold_data:          dict[str, Any],
    full_data:          dict[str, Any],
    fold:               FoldSpec,
    lstm_config:        Any | None  = None,
    backtest_threshold: float       = 0.55,
) -> tuple[dict[str, float], dict[str, Any], bool, list[str]] | None:
    """Train and evaluate an LSTM on one fold.

    Uses the combined feature_df window ``[train_start, test_end)`` to build
    rolling sequences, then splits chronologically into train / val / test.
    Returns ``None`` when the fold contains too few sequences to train
    reliably (< 20 per split).

    **Note:** The LSTM test sequences start at the boundary of the fold's
    training window plus ``seq_len`` look-back bars.  This means the first
    few test sequences borrow data from the training period, which is the
    standard behaviour for sequence models and mirrors the existing
    ``run_lstm_pipeline`` evaluation setup.

    Args:
        fold_data:          Output of ``slice_fold``.
        full_data:          Output of ``assemble_walk_forward_dataset``
                            (provides full feature_df and close_full).
        fold:               The ``FoldSpec`` for index bounds.
        lstm_config:        ``LSTMConfig``; defaults used if ``None``.
        backtest_threshold: Signal threshold for the backtest engine.

    Returns:
        ``(metrics, backtest_summary_dict, promotion_recommended, reasons)``
        or ``None`` if the fold is skipped.
    """
    from ml.patterns.dataset import PriceSequenceDataset, chronological_split
    from ml.patterns.lstm import FinBrainLSTM
    from ml.patterns.train_lstm import LSTMConfig, _evaluate_lstm, _train_lstm

    cfg     = lstm_config or LSTMConfig(hidden_size=64, num_layers=1, epochs=20, patience=5)
    seq_len = cfg.seq_len

    # Use the full combined fold window (train + gap + test)
    fdf_fold   = fold_data["feature_df_fold"]
    close_fold = fold_data["close_fold"]

    # Build sequence dataset over the full fold window
    full_ds = PriceSequenceDataset(
        feature_df     = fdf_fold,
        close_series   = close_fold,
        seq_len        = seq_len,
        target_horizon = full_data.get("target_horizon", 1),
    )

    total_seq = len(full_ds)
    if total_seq < 40:
        log.warning(
            "LSTM fold %d skipped: only %d sequences in combined window "
            "(need ≥ 40).", fold.fold_idx, total_seq,
        )
        return None

    # Approximate fold proportions from row counts
    fold_total_rows   = fold.test_end - fold.train_start
    train_window_rows = fold.train_end - fold.train_start
    train_frac_fold   = max(0.4, min(0.85, train_window_rows / fold_total_rows))
    val_frac_of_train = max(0.1, min(0.4, fold.n_val / max(1, fold.n_train + fold.n_val)))

    # Split: train+val vs test
    tv_ds, test_ds = chronological_split(full_ds, train_frac=train_frac_fold)
    if len(test_ds) < 10 or len(tv_ds) < 20:
        log.warning(
            "LSTM fold %d skipped: test_ds=%d, tv_ds=%d (too small).",
            fold.fold_idx, len(test_ds), len(tv_ds),
        )
        return None

    # Split train+val into train and val
    train_ds, val_ds = chronological_split(tv_ds, train_frac=1.0 - val_frac_of_train)
    if len(train_ds) < 10:
        log.warning(
            "LSTM fold %d skipped: train_ds=%d (too small).", fold.fold_idx, len(train_ds)
        )
        return None

    # Train
    model = FinBrainLSTM(
        input_size  = train_ds.n_features,
        hidden_size = cfg.hidden_size,
        num_layers  = cfg.num_layers,
        dropout     = cfg.dropout,
    )
    _train_lstm(model, train_ds, val_ds, cfg)

    # Evaluate on test sequences
    metrics, probs_arr = _evaluate_lstm(model, test_ds)

    # Align close prices to test sequences
    n_test_seq  = len(test_ds)
    n_tv_seq    = len(tv_ds)
    # Offset in the fold window: seq_len warm-up rows + train+val sequences
    test_offset_in_fold = seq_len + n_tv_seq
    close_test_seq = close_fold.iloc[test_offset_in_fold : test_offset_in_fold + n_test_seq]
    close_test_seq = close_test_seq.iloc[:len(probs_arr)]

    if len(close_test_seq) == 0:
        log.warning("LSTM fold %d: empty close_test after alignment.", fold.fold_idx)
        return None

    signals = pd.Series(probs_arr[:len(close_test_seq)], index=close_test_seq.index)
    bt      = run_backtest(
        close_test_seq, signals, BacktestConfig(threshold=backtest_threshold)
    )

    promo, reasons = _promotion_reasons(metrics, bt)
    return metrics, bt.to_dict(), promo, reasons


# ── Main runner ───────────────────────────────────────────────────────────────

def run_walk_forward_model(
    model_key:          str,
    symbol:             str,
    df:                 pd.DataFrame | None        = None,
    full_data:          dict[str, Any] | None      = None,
    config:             WalkForwardConfig | None   = None,
    model_config:       Any                        = None,
    registry:           ExperimentRegistry | None  = None,
    checkpoint_dir:     Path | None                = None,
    tags:               list[str] | None           = None,
    backtest_threshold: float                      = 0.55,
    target_horizon:     int                        = 1,
    epochs:             int                        = 20,
    patience:           int                        = 5,
    register_folds:     bool                       = False,
) -> WalkForwardResult:
    """Run walk-forward evaluation for a single model over all folds.

    Assembles the full feature matrix once (or accepts a pre-assembled
    ``full_data`` dict), generates fold specs, and evaluates the model on each
    fold independently.  The scaler is fit only on each fold's training rows.

    Args:
        model_key:          ``'baseline'``, ``'mlp'``, or ``'lstm'``.
        symbol:             Ticker symbol.
        df:                 OHLCV DataFrame; fetched from yfinance if both
                            ``df`` and ``full_data`` are ``None``.
        full_data:          Pre-assembled output of
                            ``assemble_walk_forward_dataset``.  Passing this
                            avoids repeated feature engineering when comparing
                            multiple models.
        config:             Walk-forward configuration; defaults applied if
                            ``None``.
        model_config:       Model-specific config (``BaselineConfig``,
                            ``LSTMConfig``); ``None`` uses defaults.
        registry:           ``ExperimentRegistry`` for fold logging.
        checkpoint_dir:     Checkpoint directory (used only when
                            ``register_folds=True``).
        tags:               Extra tags applied to every fold's registry record.
        backtest_threshold: Signal threshold for long entry in the backtest.
        target_horizon:     Bars ahead to predict.
        epochs:             Training epochs for MLP / LSTM folds.
        patience:           Early-stopping patience for MLP / LSTM folds.
        register_folds:     When ``True``, log each fold as an experiment in
                            the registry tagged ``wf_fold_{k}``.

    Returns:
        ``WalkForwardResult`` with all completed fold results.

    Raises:
        ValueError: If ``model_key`` is not one of the supported values.
    """
    supported = ("baseline", "mlp", "lstm")
    if model_key not in supported:
        raise ValueError(f"model_key must be one of {supported}, got {model_key!r}")

    cfg = config or WalkForwardConfig()
    reg = registry if registry is not None else ExperimentRegistry()

    # ── 1. Assemble full dataset once ────────────────────────────────────
    if full_data is None:
        if df is None:
            df = fetch_price_df(symbol)
        full_data = assemble_walk_forward_dataset(df, symbol, target_horizon)

    # ── 2. Generate fold specs ────────────────────────────────────────────
    folds = make_folds(full_data["n_samples"], cfg, full_data["date_index"])

    # ── 3. Evaluate per fold ──────────────────────────────────────────────
    fold_results: list[FoldResult] = []
    wf_tags = list(tags or []) + ["walk_forward", f"wf_{model_key}"]

    for fold in folds:
        fd = slice_fold(full_data, fold)

        log.info(
            "Walk-forward fold %d/%d  model=%s  train=%d  val=%d  test=%d",
            fold.fold_idx + 1, len(folds), model_key,
            fold.n_train, fold.n_val, fold.n_test,
        )

        try:
            if model_key == "baseline":
                metrics, bt, promo, reasons = _eval_baseline_fold(
                    fd, model_config, backtest_threshold
                )
            elif model_key == "mlp":
                metrics, bt, promo, reasons = _eval_mlp_fold(
                    fd, epochs, patience, backtest_threshold
                )
            else:  # lstm
                result = _eval_lstm_fold(
                    fd, full_data, fold, model_config, backtest_threshold
                )
                if result is None:
                    continue
                metrics, bt, promo, reasons = result

        except Exception as exc:
            log.warning(
                "Walk-forward fold %d failed for model=%s: %s",
                fold.fold_idx, model_key, exc, exc_info=True,
            )
            continue

        # ── Optional registry logging ────────────────────────────────────
        exp_id = ""
        if register_folds:
            fold_tags = wf_tags + [f"wf_fold_{fold.fold_idx}"]
            try:
                exp = reg.create(
                    name       = (
                        f"wf_{model_key}_{symbol.lower()}_"
                        f"fold{fold.fold_idx}_"
                        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M')}"
                    ),
                    model_type  = model_key if model_key != "baseline" else "logistic",
                    hyperparams = {},
                    dataset_info = {
                        "symbol":      symbol.upper(),
                        "fold_idx":    fold.fold_idx,
                        "n_train":     fold.n_train,
                        "n_val":       fold.n_val,
                        "n_test":      fold.n_test,
                        "test_start":  fold.test_date_start,
                        "test_end":    fold.test_date_end,
                        "walk_forward": True,
                    },
                    tags  = fold_tags,
                    notes = f"Walk-forward fold {fold.fold_idx}",
                )
                from ml.registry.experiment_registry import BacktestResult
                reg.finish(exp.experiment_id, metrics=metrics)
                # Attach backtest summary (reconstruct BacktestResult from dict)
                bt_for_reg = BacktestResult(
                    cumulative_return = bt.get("cumulative_return", 0.0),
                    annualised_return = bt.get("annualised_return", 0.0),
                    hit_rate          = bt.get("hit_rate", 0.0),
                    max_drawdown      = bt.get("max_drawdown", 0.0),
                    sharpe            = bt.get("sharpe", 0.0),
                    trade_count       = bt.get("trade_count", 0),
                    benchmark_return  = bt.get("benchmark_return", 0.0),
                    period_start      = bt.get("period_start", ""),
                    period_end        = bt.get("period_end", ""),
                )
                reg.attach_backtest(exp.experiment_id, bt_for_reg)
                exp_id = exp.experiment_id
            except Exception as reg_exc:
                log.warning("Failed to register fold %d: %s", fold.fold_idx, reg_exc)

        fold_results.append(FoldResult(
            fold_idx              = fold.fold_idx,
            fold_spec             = fold,
            model_type            = model_key,
            metrics               = metrics,
            backtest_summary      = bt,
            promotion_recommended = promo,
            promotion_reasons     = reasons,
            experiment_id         = exp_id,
        ))

        log.info(
            "  fold %d done: acc=%.3f  auc=%.3f  hit=%.3f  promo=%s",
            fold.fold_idx,
            metrics.get("accuracy", 0),
            metrics.get("auc", 0),
            bt.get("hit_rate", 0),
            promo,
        )

    return WalkForwardResult(
        symbol        = full_data["symbol"],
        model_type    = model_key,
        n_folds_total = len(folds),
        n_folds_run   = len(fold_results),
        fold_results  = fold_results,
        config        = cfg,
        generated_at  = datetime.now(tz=timezone.utc).isoformat(),
    )


def run_walk_forward(
    symbol:             str,
    df:                 pd.DataFrame | None       = None,
    models:             tuple[str, ...] | list[str] = ("baseline", "mlp"),
    config:             WalkForwardConfig | None  = None,
    registry:           ExperimentRegistry | None = None,
    checkpoint_dir:     Path | None               = None,
    epochs:             int                       = 20,
    patience:           int                       = 5,
    tags:               list[str] | None          = None,
    target_horizon:     int                       = 1,
    backtest_threshold: float                     = 0.55,
    register_folds:     bool                      = False,
    period:             str                       = "3y",
) -> dict[str, WalkForwardResult]:
    """Run walk-forward evaluation for multiple models on the same dataset.

    Assembles the full feature matrix **once** and shares it across all
    requested models so that every model is evaluated on identical data.

    Failed models are logged as warnings and excluded from the result dict.

    Args:
        symbol:             Ticker symbol.
        df:                 OHLCV DataFrame; fetched from yfinance if ``None``.
        models:             Model keys to evaluate.  Supported:
                            ``'baseline'``, ``'mlp'``, ``'lstm'``.
        config:             Walk-forward configuration.
        registry:           Shared ``ExperimentRegistry``.
        checkpoint_dir:     Checkpoint directory (for fold registration).
        epochs:             Training epochs for MLP / LSTM folds.
        patience:           Early-stopping patience for MLP / LSTM folds.
        tags:               Tags applied to every fold record.
        target_horizon:     Bars ahead to predict.
        backtest_threshold: Signal threshold for the backtest engine.
        register_folds:     Whether to log each fold in the registry.
        period:             yfinance period string (ignored when ``df`` given).

    Returns:
        ``dict[model_type, WalkForwardResult]`` — one entry per model that
        completed at least one fold successfully.
    """
    cfg = config or WalkForwardConfig()
    reg = registry if registry is not None else ExperimentRegistry()

    if df is None:
        df = fetch_price_df(symbol, period=period)

    # Assemble once, share across models
    full_data = assemble_walk_forward_dataset(df, symbol, target_horizon)

    results: dict[str, WalkForwardResult] = {}
    for model_key in models:
        model_key = model_key.lower()
        if model_key not in ("baseline", "mlp", "lstm"):
            log.warning("Unknown model %r — skipping.", model_key)
            continue
        log.info("Walk-forward: starting model=%s", model_key)
        try:
            wf = run_walk_forward_model(
                model_key          = model_key,
                symbol             = symbol,
                full_data          = full_data,
                config             = cfg,
                registry           = reg,
                checkpoint_dir     = checkpoint_dir,
                tags               = tags,
                backtest_threshold = backtest_threshold,
                target_horizon     = target_horizon,
                epochs             = epochs,
                patience           = patience,
                register_folds     = register_folds,
            )
            results[model_key] = wf
            log.info(
                "Walk-forward model=%s done: %d/%d folds completed.",
                model_key, wf.n_folds_run, wf.n_folds_total,
            )
        except Exception as exc:
            log.warning("Walk-forward model=%s failed: %s", model_key, exc, exc_info=True)

    return results
