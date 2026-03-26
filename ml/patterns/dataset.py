"""PyTorch Dataset for LSTM sequence training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PriceSequenceDataset(Dataset):
    """Rolling-window sequence dataset built from a feature DataFrame.

    Each sample is a (seq_len × n_features) input window X and a
    binary target y = 1 if close[t + horizon] > close[t] else 0.

    Args:
        feature_df:     DataFrame of pre-computed features (numeric cols).
        close_series:   Raw close price series (same index as feature_df).
        seq_len:        Look-back window length in trading days.
        target_horizon: How many days ahead to predict direction.
        scaler_mean:    Optional precomputed mean for normalisation.
        scaler_std:     Optional precomputed std for normalisation.
    """

    def __init__(
        self,
        feature_df:      pd.DataFrame,
        close_series:    pd.Series,
        seq_len:         int = 20,
        target_horizon:  int = 1,
        scaler_mean:     np.ndarray | None = None,
        scaler_std:      np.ndarray | None = None,
    ) -> None:
        num_df = feature_df.select_dtypes(include=[np.number]).copy()
        num_df = num_df.ffill().bfill().fillna(0.0)

        # Replace inf
        num_df = num_df.replace([np.inf, -np.inf], 0.0)

        self.feature_cols = list(num_df.columns)
        X_raw = num_df.values.astype(np.float32)

        # Fit or apply normalisation
        if scaler_mean is None:
            self.scaler_mean = X_raw.mean(axis=0)
            self.scaler_std  = X_raw.std(axis=0)
        else:
            self.scaler_mean = scaler_mean
            self.scaler_std  = scaler_std

        std_safe = np.where(self.scaler_std == 0, 1.0, self.scaler_std)
        X_norm   = (X_raw - self.scaler_mean) / std_safe

        # Binary targets: 1 = price higher in `target_horizon` days
        returns = close_series.pct_change(target_horizon).shift(-target_horizon)
        y_all   = (returns > 0).astype(np.float32).values

        sequences: list[np.ndarray] = []
        targets:   list[float]      = []

        for i in range(seq_len, len(X_norm) - target_horizon):
            y_val = y_all[i]
            if np.isnan(y_val):
                continue
            sequences.append(X_norm[i - seq_len:i])
            targets.append(y_val)

        self._X = torch.from_numpy(np.array(sequences, dtype=np.float32))
        self._y = torch.from_numpy(np.array(targets,   dtype=np.float32)).unsqueeze(1)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]

    @property
    def n_features(self) -> int:
        """Number of feature dimensions."""
        return self._X.shape[2]


def chronological_split(
    dataset: PriceSequenceDataset,
    train_frac: float = 0.80,
) -> tuple[PriceSequenceDataset, PriceSequenceDataset]:
    """Split a dataset into train/val preserving temporal order.

    Args:
        dataset:    Source dataset.
        train_frac: Fraction of samples for training.

    Returns:
        (train_dataset, val_dataset) — shallow-copy subsets.
    """
    n_train = int(len(dataset) * train_frac)
    train   = _SlicedDataset(dataset, slice(None, n_train))
    val     = _SlicedDataset(dataset, slice(n_train, None))
    return train, val  # type: ignore[return-value]


class _SlicedDataset(Dataset):
    """View of a subset of another Dataset without copying tensors."""

    def __init__(self, parent: PriceSequenceDataset, slc: slice) -> None:
        self._X = parent._X[slc]
        self._y = parent._y[slc]
        self.feature_cols = parent.feature_cols
        self.scaler_mean  = parent.scaler_mean
        self.scaler_std   = parent.scaler_std

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]

    @property
    def n_features(self) -> int:
        return self._X.shape[2]
