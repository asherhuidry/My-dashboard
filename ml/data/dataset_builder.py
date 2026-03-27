"""Dataset assembly with version identity for supervised ML experiments.

Wraps the existing feature-engineering pipeline with a ``DatasetMeta``
dataclass that captures what the dataset is and produces a stable version
hash.  Use this module when you need to compare multiple models on the same
data — the version hash guarantees that two experiments sharing a version
used identical input, target, and split logic.

Flat datasets (MLP, logistic baseline) and sequence datasets (LSTM) are both
supported.  Both carry a ``DatasetMeta`` so cross-model comparisons are
traceable in the experiment registry.

Usage::

    from ml.data.dataset_builder import assemble_dataset, assemble_sequence_dataset

    data, meta = assemble_dataset(df, symbol="AAPL")
    # data keys: X_train, y_train, X_val, y_val, X_test, y_test,
    #            close_test, signal_index, feature_cols, n_train, n_val, n_test
    # meta.dataset_version — stable 12-char hash

    train_ds, val_ds, test_ds, close_test, meta = assemble_sequence_dataset(df, "AAPL")
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.patterns.dataset import PriceSequenceDataset, chronological_split
from ml.patterns.features import build_features

log = logging.getLogger(__name__)


# ── Dataset metadata ──────────────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """Metadata and version identity for an assembled dataset.

    The ``dataset_version`` field is a stable 12-character SHA-256 prefix
    computed from the key parameters that define the dataset.  Two assembly
    runs producing the same hash used identical data, features, and split
    logic, so their experiment results are directly comparable.

    Attributes:
        dataset_version:   12-char hex hash uniquely identifying this dataset.
        symbol:            Ticker or dataset identifier.
        feature_cols:      Ordered list of feature column names.
        target_definition: Human-readable description of the target variable.
        n_rows:            Total usable rows after dropping NaNs.
        n_train:           Training rows / sequences.
        n_val:             Validation rows / sequences.
        n_test:            Test rows / sequences.
        time_range_start:  ISO-8601 start of the data window.
        time_range_end:    ISO-8601 end of the data window.
        train_frac:        Training fraction.
        val_frac:          Validation fraction.
        target_horizon:    Bars ahead to predict.
        seq_len:           Rolling window for sequence models; None for flat.
        generated_at:      ISO-8601 assembly timestamp.
        notes:             Optional free-text notes.
    """
    dataset_version:   str
    symbol:            str
    feature_cols:      list[str]
    target_definition: str
    n_rows:            int
    n_train:           int
    n_val:             int
    n_test:            int
    time_range_start:  str
    time_range_end:    str
    train_frac:        float
    val_frac:          float
    target_horizon:    int
    seq_len:           int | None
    generated_at:      str
    notes:             str = ""

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetMeta":
        """Deserialise from a dict produced by ``to_dict()``."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dataset_info(self) -> dict[str, Any]:
        """Return a registry-friendly subset for ``ExperimentRecord.dataset_info``.

        Stores the version hash and key counts so that experiments can be
        grouped and compared by dataset version later.

        Returns:
            Dict suitable for the ``dataset_info`` field of ExperimentRegistry.
        """
        return {
            "dataset_version":   self.dataset_version,
            "symbol":            self.symbol,
            "n_rows":            self.n_rows,
            "n_train":           self.n_train,
            "n_val":             self.n_val,
            "n_test":            self.n_test,
            "time_range_start":  self.time_range_start,
            "time_range_end":    self.time_range_end,
            "target_horizon":    self.target_horizon,
            "seq_len":           self.seq_len,
            "target_definition": self.target_definition,
            "feature_cols":      self.feature_cols[:10],   # first 10 for brevity
        }


# ── Version hashing ───────────────────────────────────────────────────────────

def _compute_version(
    symbol:           str,
    feature_cols:     list[str],
    target_horizon:   int,
    n_rows:           int,
    time_range_start: str,
    time_range_end:   str,
    train_frac:       float,
    val_frac:         float,
    seq_len:          int | None,
) -> str:
    """Compute a stable version hash from dataset parameters.

    Two datasets sharing the same hash were assembled from equivalent
    inputs using the same feature set, target, and split fractions.

    Args:
        All parameters that uniquely identify a dataset assembly.

    Returns:
        12-character lowercase hex prefix of the SHA-256 hash.
    """
    canonical = json.dumps(
        {
            "symbol":           symbol.upper(),
            "feature_cols":     list(feature_cols),
            "target_horizon":   target_horizon,
            "n_rows":           n_rows,
            "time_range_start": time_range_start[:19],   # drop sub-second precision
            "time_range_end":   time_range_end[:19],
            "train_frac":       round(train_frac, 4),
            "val_frac":         round(val_frac, 4),
            "seq_len":          seq_len,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ── Flat dataset assembly (MLP / logistic baseline) ───────────────────────────

def assemble_dataset(
    df:             pd.DataFrame,
    symbol:         str,
    target_horizon: int   = 1,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
    notes:          str   = "",
) -> tuple[dict[str, Any], DatasetMeta]:
    """Assemble flat feature matrices and binary targets from an OHLCV DataFrame.

    Runs the standard feature-engineering pipeline (``ml.patterns.features``),
    constructs a binary next-bar direction target, applies a strict
    chronological split, and returns both the NumPy arrays and a
    ``DatasetMeta`` with a stable version hash.

    The returned data dict is directly usable by both the MLP and logistic
    baseline pipelines — it is identical to the dict produced by
    ``ml.patterns.train_mlp.build_dataset`` but additionally produces
    metadata for experiment comparisons.

    Args:
        df:             OHLCV DataFrame with lowercase columns and DatetimeIndex.
        symbol:         Ticker or dataset identifier stored in metadata.
        target_horizon: How many bars ahead to predict direction (default 1).
        train_frac:     Fraction of rows for training (default 0.70).
        val_frac:       Fraction of rows for validation (default 0.15).
                        Remaining rows form the test set.
        notes:          Optional free-text notes stored in metadata.

    Returns:
        ``(data, meta)`` where ``data`` is a dict with keys:
        ``X_train, y_train, X_val, y_val, X_test, y_test, close_test,
        signal_index, feature_cols, n_train, n_val, n_test``
        and ``meta`` is a ``DatasetMeta``.
    """
    feat_df    = build_features(df)
    feat_df    = feat_df.dropna(how="all")
    df_aligned = df.reindex(feat_df.index)

    close   = df_aligned["close"]
    returns = close.pct_change(target_horizon).shift(-target_horizon)
    target  = (returns > 0).astype(float)

    valid_mask = target.notna() & feat_df.notna().all(axis=1)
    feat_df = feat_df.loc[valid_mask]
    target  = target.loc[valid_mask]
    close   = close.loc[valid_mask]

    X_raw = feat_df.values.astype(np.float32)
    y_raw = target.values.astype(np.float32)
    X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)

    n       = len(X_raw)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    feature_cols     = list(feat_df.columns)
    time_range_start = feat_df.index[0].isoformat()
    time_range_end   = feat_df.index[-1].isoformat()

    version = _compute_version(
        symbol, feature_cols, target_horizon, n,
        time_range_start, time_range_end,
        train_frac, val_frac, seq_len=None,
    )

    data: dict[str, Any] = {
        "X_train":      X_raw[:n_train],
        "y_train":      y_raw[:n_train],
        "X_val":        X_raw[n_train : n_train + n_val],
        "y_val":        y_raw[n_train : n_train + n_val],
        "X_test":       X_raw[n_train + n_val :],
        "y_test":       y_raw[n_train + n_val :],
        "close_test":   close.iloc[n_train + n_val :],
        "signal_index": feat_df.index[n_train + n_val :],
        "feature_cols": feature_cols,
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       n_test,
    }

    meta = DatasetMeta(
        dataset_version   = version,
        symbol            = symbol.upper(),
        feature_cols      = feature_cols,
        target_definition = f"binary_direction_{target_horizon}bar",
        n_rows            = n,
        n_train           = n_train,
        n_val             = n_val,
        n_test            = n_test,
        time_range_start  = time_range_start,
        time_range_end    = time_range_end,
        train_frac        = train_frac,
        val_frac          = val_frac,
        target_horizon    = target_horizon,
        seq_len           = None,
        generated_at      = datetime.now(tz=timezone.utc).isoformat(),
        notes             = notes,
    )

    log.info(
        "Dataset assembled: %s v=%s  rows=%d  (train=%d / val=%d / test=%d)",
        symbol, version, n, n_train, n_val, n_test,
    )
    return data, meta


# ── Sequence dataset assembly (LSTM) ─────────────────────────────────────────

def assemble_sequence_dataset(
    df:             pd.DataFrame,
    symbol:         str,
    seq_len:        int   = 20,
    target_horizon: int   = 1,
    train_frac:     float = 0.70,
    val_frac:       float = 0.15,
    notes:          str   = "",
) -> tuple[Any, Any, Any, pd.Series, DatasetMeta]:
    """Assemble rolling-window sequence datasets for LSTM training.

    Uses the same feature-engineering pipeline as ``assemble_dataset`` so
    both flat and sequence experiments can be logged against the same
    dataset identity (the version hash differs only in ``seq_len``).

    The chronological split mirrors the flat case — 70 / 15 / 15 by
    default — so LSTM and MLP test windows cover the same calendar period
    and backtest results are comparable.

    Args:
        df:             OHLCV DataFrame with lowercase columns and DatetimeIndex.
        symbol:         Ticker or dataset identifier stored in metadata.
        seq_len:        Rolling window length in trading days.
        target_horizon: How many bars ahead to predict direction.
        train_frac:     Fraction of sequences for training.
        val_frac:       Fraction of sequences for validation (early stopping).
                        Remaining sequences form the held-out test set.
        notes:          Optional free-text notes stored in metadata.

    Returns:
        ``(train_ds, val_ds, test_ds, close_test, meta)`` where the first
        three are ``PriceSequenceDataset`` (or ``_SlicedDataset``) instances,
        ``close_test`` is a ``pd.Series`` of close prices aligned to the test
        sequences (for backtesting), and ``meta`` is a ``DatasetMeta``.
    """
    feat_df    = build_features(df)
    feat_df    = feat_df.dropna(how="all")
    df_aligned = df.reindex(feat_df.index)
    close      = df_aligned["close"]

    full_ds = PriceSequenceDataset(
        feature_df     = feat_df,
        close_series   = close,
        seq_len        = seq_len,
        target_horizon = target_horizon,
    )

    total_seq = len(full_ds)
    n_train   = int(total_seq * train_frac)
    n_valtest = total_seq - n_train
    # Split valtest 50/50 to approximate val_frac ≈ test_frac
    val_split_frac = val_frac / (1.0 - train_frac) if (1.0 - train_frac) > 0 else 0.5
    val_split_frac = max(0.1, min(0.9, val_split_frac))   # clamp for safety

    train_ds, valtest_ds = chronological_split(full_ds, train_frac=train_frac)
    val_ds,   test_ds   = chronological_split(valtest_ds, train_frac=val_split_frac)

    n_val  = len(val_ds)
    n_test = len(test_ds)

    # Reconstruct the close prices aligned to each test sequence.
    # In PriceSequenceDataset, the j-th sequence (0-indexed) corresponds to
    # feat_df index position seq_len + j (the bar whose target is predicted).
    # Test sequences start at j = n_train + n_val in the original full_ds.
    test_start_idx = seq_len + n_train + n_val
    test_end_idx   = test_start_idx + n_test
    test_end_idx   = min(test_end_idx, len(close))
    close_test     = close.iloc[test_start_idx:test_end_idx]
    # Align length (guards against rare NaN-skip edge cases)
    close_test = close_test.iloc[:n_test]

    feature_cols     = full_ds.feature_cols
    time_range_start = feat_df.index[0].isoformat()
    time_range_end   = feat_df.index[-1].isoformat()

    version = _compute_version(
        symbol, feature_cols, target_horizon, len(feat_df),
        time_range_start, time_range_end,
        train_frac, val_frac, seq_len=seq_len,
    )

    meta = DatasetMeta(
        dataset_version   = version,
        symbol            = symbol.upper(),
        feature_cols      = feature_cols,
        target_definition = f"binary_direction_{target_horizon}bar_seq{seq_len}",
        n_rows            = len(feat_df),
        n_train           = n_train,
        n_val             = n_val,
        n_test            = n_test,
        time_range_start  = time_range_start,
        time_range_end    = time_range_end,
        train_frac        = train_frac,
        val_frac          = val_frac,
        target_horizon    = target_horizon,
        seq_len           = seq_len,
        generated_at      = datetime.now(tz=timezone.utc).isoformat(),
        notes             = notes,
    )

    log.info(
        "Sequence dataset assembled: %s v=%s seq_len=%d  "
        "rows=%d  (train=%d / val=%d / test=%d)",
        symbol, version, seq_len, len(feat_df), n_train, n_val, n_test,
    )
    return train_ds, val_ds, test_ds, close_test, meta
