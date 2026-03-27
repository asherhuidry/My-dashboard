"""Walk-forward split utilities for chronological supervised ML.

Provides deterministic, leakage-free fold specifications for evaluating
models across multiple non-overlapping time windows.  No model is ever
trained on data that post-dates its own test window.

Two expansion strategies
------------------------
expanding (default):
    Fold k trains on rows [0, T + k*W) and tests on [T + k*W, T + (k+1)*W).
    The training window grows with each fold — the standard approach in finance.

rolling:
    Fold k trains on rows [k*W, T + k*W) and tests on the same test window.
    The training window is a fixed-size sliding view.

Usage::

    from ml.validation.walk_forward import (
        WalkForwardConfig, FoldSpec,
        make_folds,
        assemble_walk_forward_dataset,
        slice_fold,
    )

    cfg  = WalkForwardConfig(n_folds=5, min_train_size=120)
    full = assemble_walk_forward_dataset(ohlcv_df, symbol="AAPL")
    folds = make_folds(full["n_samples"], cfg, full["date_index"])

    for fold in folds:
        fd = slice_fold(full, fold)
        # fd keys: X_train, y_train, X_val, y_val, X_test, y_test, close_test, ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml.patterns.features import build_features

log = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward fold generation.

    Attributes:
        n_folds:        Number of train/test folds to produce.  Fewer folds
                        may be returned if data runs out before ``n_folds``
                        complete windows can be formed.
        min_train_size: Minimum rows in the first training window.  Must be
                        large enough to cover the feature-engineering warm-up
                        period and produce stable statistics (≥ 100 recommended
                        for 60-day rolling indicators).
        test_size:      Fixed test window size in rows per fold.  When ``None``,
                        auto-computed as
                        ``(n_samples - min_train_size) // n_folds``.
        val_frac:       Fraction of the *current* raw training window carved out
                        at the tail for validation / early stopping.  Each fold
                        uses a different absolute validation window (it tracks
                        the growing train window).  Pass ``0.0`` to disable
                        per-fold validation splits.
        gap:            Rows to skip between the last training bar and the first
                        test bar.  A gap of 1 simulates next-bar execution with
                        no look-ahead; 0 is fine for daily close-to-close data
                        where the signal is generated at today's close.
        window:         ``'expanding'`` (default) — anchored training window
                        that grows each fold; or ``'rolling'`` — fixed-size
                        training window that slides forward.
    """
    n_folds:        int        = 5
    min_train_size: int        = 120
    test_size:      int | None = None
    val_frac:       float      = 0.15
    gap:            int        = 0
    window:         str        = "expanding"

    def __post_init__(self) -> None:
        if self.window not in ("expanding", "rolling"):
            raise ValueError(
                f"window must be 'expanding' or 'rolling', got {self.window!r}"
            )
        if not 0.0 <= self.val_frac < 1.0:
            raise ValueError(
                f"val_frac must be in [0, 1), got {self.val_frac}"
            )
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {self.n_folds}")
        if self.min_train_size < 2:
            raise ValueError(
                f"min_train_size must be >= 2, got {self.min_train_size}"
            )
        if self.gap < 0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "n_folds":        self.n_folds,
            "min_train_size": self.min_train_size,
            "test_size":      self.test_size,
            "val_frac":       self.val_frac,
            "gap":            self.gap,
            "window":         self.window,
        }


# ── Fold specification ─────────────────────────────────────────────────────────

@dataclass
class FoldSpec:
    """Explicit, auditable description of one walk-forward fold.

    All ``*_start`` / ``*_end`` fields are integer *row positions* into the
    full assembled array.  End indices are **exclusive** (Python slice
    convention: ``array[start:end]``).

    The raw training window ``[train_start, train_end)`` includes the
    optional validation slice at its tail.  ``n_train`` reflects only the
    *pure* training rows (i.e. ``raw_train_rows - n_val``).

    Date fields are ISO-8601 strings populated when a ``DatetimeIndex`` was
    supplied to ``make_folds``; otherwise empty strings.

    Attributes:
        fold_idx:         Zero-based fold number (0 = earliest).
        train_start:      First row index of the raw training window.
        train_end:        Exclusive end of the raw training window.
        val_start:        First row index of the validation slice inside the
                          training window.  ``None`` when ``val_frac == 0``.
        val_end:          Exclusive end of the validation slice.  ``None``
                          when ``val_frac == 0``.
        test_start:       First row index of the held-out test window.
        test_end:         Exclusive end of the test window.
        n_train:          Pure training rows (``train_end - train_start - n_val``).
        n_val:            Validation rows; 0 when validation is disabled.
        n_test:           Test rows.
        *_date_*:         ISO-8601 date strings (empty when no DatetimeIndex
                          was provided).
    """
    fold_idx:          int
    train_start:       int
    train_end:         int
    val_start:         int | None
    val_end:           int | None
    test_start:        int
    test_end:          int
    n_train:           int
    n_val:             int
    n_test:            int
    train_date_start:  str = ""
    train_date_end:    str = ""
    val_date_start:    str = ""
    val_date_end:      str = ""
    test_date_start:   str = ""
    test_date_end:     str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "fold_idx":         self.fold_idx,
            "train_start":      self.train_start,
            "train_end":        self.train_end,
            "val_start":        self.val_start,
            "val_end":          self.val_end,
            "test_start":       self.test_start,
            "test_end":         self.test_end,
            "n_train":          self.n_train,
            "n_val":            self.n_val,
            "n_test":           self.n_test,
            "train_date_start": self.train_date_start,
            "train_date_end":   self.train_date_end,
            "val_date_start":   self.val_date_start,
            "val_date_end":     self.val_date_end,
            "test_date_start":  self.test_date_start,
            "test_date_end":    self.test_date_end,
        }


# ── Fold generation ────────────────────────────────────────────────────────────

def make_folds(
    n_samples:  int,
    config:     WalkForwardConfig,
    date_index: pd.DatetimeIndex | None = None,
) -> list[FoldSpec]:
    """Generate walk-forward fold specifications for ``n_samples`` data rows.

    Returns index specifications only — no arrays are sliced here.  Use
    ``slice_fold`` to extract the actual train/val/test arrays for each fold.

    The function produces *at most* ``config.n_folds`` folds; it may produce
    fewer if data runs out before all requested folds can be formed.

    Args:
        n_samples:   Total valid rows in the assembled dataset (after NaN
                     removal and target alignment).
        config:      Walk-forward configuration.
        date_index:  Optional ``DatetimeIndex`` of length ``n_samples``.
                     When provided, each ``FoldSpec`` is annotated with
                     ISO-8601 date strings for easy inspection.

    Returns:
        List of ``FoldSpec`` in chronological order (fold 0 is earliest).

    Raises:
        ValueError: If ``n_samples`` is too small to form even one valid fold.
    """
    cfg = config

    # ── Compute test window size ───────────────────────────────────────────
    if cfg.test_size is not None:
        test_size = cfg.test_size
    else:
        available = n_samples - cfg.min_train_size
        if available <= 0:
            raise ValueError(
                f"n_samples ({n_samples}) is not larger than "
                f"min_train_size ({cfg.min_train_size})."
            )
        test_size = max(1, available // cfg.n_folds)

    def _date(pos: int) -> str:
        if date_index is None or pos < 0 or pos >= len(date_index):
            return ""
        return date_index[pos].isoformat()

    folds: list[FoldSpec] = []

    for k in range(cfg.n_folds):
        # ── Raw training window bounds ─────────────────────────────────────
        if cfg.window == "expanding":
            raw_start = 0
            raw_end   = cfg.min_train_size + k * test_size
        else:  # rolling
            raw_start = k * test_size
            raw_end   = cfg.min_train_size + k * test_size

        # ── Test window ───────────────────────────────────────────────────
        test_start = raw_end + cfg.gap
        test_end   = test_start + test_size

        # Stop if we'd exceed the data
        if test_end > n_samples:
            break
        if raw_end <= raw_start:
            continue

        # ── Carve validation slice off the tail of the training window ─────
        raw_n = raw_end - raw_start
        if cfg.val_frac > 0.0:
            n_val        = max(1, int(raw_n * cfg.val_frac))
            n_pure_train = raw_n - n_val
            val_start: int | None = raw_start + n_pure_train
            val_end:   int | None = raw_end
        else:
            n_val        = 0
            n_pure_train = raw_n
            val_start    = None
            val_end      = None

        fold = FoldSpec(
            fold_idx         = k,
            train_start      = raw_start,
            train_end        = raw_end,
            val_start        = val_start,
            val_end          = val_end,
            test_start       = test_start,
            test_end         = test_end,
            n_train          = n_pure_train,
            n_val            = n_val,
            n_test           = test_size,
            train_date_start = _date(raw_start),
            train_date_end   = _date(raw_end - 1),
            val_date_start   = _date(val_start) if val_start is not None else "",
            val_date_end     = _date(val_end - 1) if val_end is not None else "",
            test_date_start  = _date(test_start),
            test_date_end    = _date(test_end - 1),
        )
        folds.append(fold)

    if not folds:
        raise ValueError(
            f"No valid folds could be formed from {n_samples} samples.  "
            f"Config: n_folds={cfg.n_folds}, min_train_size={cfg.min_train_size}, "
            f"computed test_size={test_size}, gap={cfg.gap}.  "
            f"Reduce n_folds, reduce min_train_size, or provide more data."
        )

    if len(folds) < cfg.n_folds:
        log.warning(
            "Only %d of %d requested folds could be formed from %d samples.",
            len(folds), cfg.n_folds, n_samples,
        )

    return folds


# ── Full-dataset assembly (walk-forward variant) ───────────────────────────────

def assemble_walk_forward_dataset(
    df:             pd.DataFrame,
    symbol:         str,
    target_horizon: int = 1,
) -> dict[str, Any]:
    """Assemble a full, unsplit feature matrix for walk-forward evaluation.

    Runs the same feature-engineering pipeline as ``assemble_dataset`` but
    returns the *complete* arrays without any pre-determined train/val/test
    split.  The walk-forward runner slices per fold and fits the feature
    scaler **only on each fold's training rows** to prevent look-ahead bias.

    Args:
        df:             OHLCV DataFrame with lowercase columns and DatetimeIndex.
        symbol:         Ticker or dataset identifier.
        target_horizon: Bars ahead to predict direction (default 1 = next bar).

    Returns:
        Dict with keys:

        ``X_full``        – float32 array ``(n_samples, n_features)``
        ``y_full``        – float32 array ``(n_samples,)``
        ``close_full``    – ``pd.Series`` of close prices with DatetimeIndex
        ``feature_df``    – raw unscaled feature ``pd.DataFrame`` (needed by
                           LSTM sequence construction)
        ``feature_cols``  – ordered list of feature column names
        ``date_index``    – ``pd.DatetimeIndex`` of length ``n_samples``
        ``n_samples``     – int, total valid rows
        ``symbol``        – uppercased ticker string
        ``target_horizon``– int
    """
    feat_df    = build_features(df)
    feat_df    = feat_df.dropna(how="all")
    df_aligned = df.reindex(feat_df.index)

    close   = df_aligned["close"]
    returns = close.pct_change(target_horizon).shift(-target_horizon)
    target  = (returns > 0).astype(float)

    valid_mask = target.notna() & feat_df.notna().all(axis=1)
    feat_df    = feat_df.loc[valid_mask]
    target     = target.loc[valid_mask]
    close      = close.loc[valid_mask]

    X_raw = feat_df.values.astype(np.float32)
    y_raw = target.values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)

    log.info(
        "Walk-forward dataset: %s  n_samples=%d  n_features=%d",
        symbol.upper(), len(X_raw), X_raw.shape[1],
    )

    return {
        "X_full":         X_raw,
        "y_full":         y_raw,
        "close_full":     close,
        "feature_df":     feat_df,
        "feature_cols":   list(feat_df.columns),
        "date_index":     feat_df.index,
        "n_samples":      len(X_raw),
        "symbol":         symbol.upper(),
        "target_horizon": target_horizon,
    }


# ── Fold slicing ───────────────────────────────────────────────────────────────

def slice_fold(
    full_data: dict[str, Any],
    fold:      FoldSpec,
) -> dict[str, Any]:
    """Slice the full dataset into one fold's train / val / test windows.

    The feature scaler is intentionally NOT applied here — callers must fit
    the scaler on the fold's training rows only, then transform val and test
    independently.  This ensures no information from future rows leaks into
    normalisation.

    Args:
        full_data: Output of ``assemble_walk_forward_dataset``.
        fold:      ``FoldSpec`` with row-index positions.

    Returns:
        Dict with keys:

        ``X_train``         – unscaled float32, shape ``(n_train, n_features)``
        ``y_train``         – float32 labels, shape ``(n_train,)``
        ``X_val``           – unscaled float32, shape ``(n_val, n_features)``
                             (empty array when val_frac == 0)
        ``y_val``           – float32 labels (empty when val_frac == 0)
        ``X_test``          – unscaled float32, shape ``(n_test, n_features)``
        ``y_test``          – float32 labels, shape ``(n_test,)``
        ``close_test``      – ``pd.Series`` aligned to test rows
        ``feature_df_fold`` – raw feature ``pd.DataFrame`` for the combined
                             train+test window (used by LSTM)
        ``close_fold``      – ``pd.Series`` for the combined train+test window
        ``feature_cols``    – list of feature column names
        ``n_train``         – int
        ``n_val``           – int
        ``n_test``          – int
    """
    X     = full_data["X_full"]
    y     = full_data["y_full"]
    close = full_data["close_full"]
    fdf   = full_data["feature_df"]

    # ── Training window (includes val slice at tail) ───────────────────────
    X_raw_train = X[fold.train_start : fold.train_end]
    y_raw_train = y[fold.train_start : fold.train_end]

    if fold.val_start is not None:
        X_train = X_raw_train[:fold.n_train]
        y_train = y_raw_train[:fold.n_train]
        X_val   = X_raw_train[fold.n_train:]
        y_val   = y_raw_train[fold.n_train:]
    else:
        X_train = X_raw_train
        y_train = y_raw_train
        n_feat  = X.shape[1]
        X_val   = np.empty((0, n_feat), dtype=np.float32)
        y_val   = np.empty((0,),        dtype=np.float32)

    # ── Test window ───────────────────────────────────────────────────────
    X_test     = X[fold.test_start : fold.test_end]
    y_test     = y[fold.test_start : fold.test_end]
    close_test = close.iloc[fold.test_start : fold.test_end]

    # ── Combined fold window for LSTM sequence construction ────────────────
    # Includes gap rows so sequence continuity is preserved across the fold.
    fdf_fold   = fdf.iloc[fold.train_start : fold.test_end]
    close_fold = close.iloc[fold.train_start : fold.test_end]

    return {
        "X_train":        X_train,
        "y_train":        y_train,
        "X_val":          X_val,
        "y_val":          y_val,
        "X_test":         X_test,
        "y_test":         y_test,
        "close_test":     close_test,
        "feature_df_fold": fdf_fold,
        "close_fold":      close_fold,
        "feature_cols":   full_data["feature_cols"],
        "n_train":        fold.n_train,
        "n_val":          fold.n_val,
        "n_test":         fold.n_test,
    }
