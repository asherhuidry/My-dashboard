"""FinBrain LSTM model for directional price prediction.

Architecture:
  - 2-layer LSTM with dropout
  - Dense head: hidden → 64 → 1 (sigmoid for direction probability)

Input:  (batch, seq_len, n_features) — sequence of feature vectors
Output: (batch, 1)                   — probability that next-day return > 0
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class FinBrainLSTM(nn.Module):
    """Two-layer LSTM with a classification head."""

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ) -> None:
        """Initialise the LSTM.

        Args:
            input_size:  Number of input features per timestep.
            hidden_size: LSTM hidden state dimension.
            num_layers:  Number of stacked LSTM layers.
            dropout:     Dropout probability (applied between layers + head).
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Probability tensor of shape (batch, 1).
        """
        out, _ = self.lstm(x)
        last    = self.dropout(out[:, -1, :])   # take last timestep
        return self.head(last)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:       str | Path,
    model:      FinBrainLSTM,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    dict[str, Any],
    scaler_mean: np.ndarray,
    scaler_std:  np.ndarray,
    feature_cols: list[str],
    config:     dict[str, Any],
) -> None:
    """Save full training state to disk.

    Args:
        path:         Checkpoint file path (.pt).
        model:        Trained LSTM model.
        optimizer:    Optimizer state.
        epoch:        Last completed epoch.
        metrics:      Dict of train/val metrics.
        scaler_mean:  Feature mean array for normalisation.
        scaler_std:   Feature std array for normalisation.
        feature_cols: Ordered list of feature column names.
        config:       Hyperparameter dict.
    """
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "metrics":      metrics,
        "scaler_mean":  scaler_mean,
        "scaler_std":   scaler_std,
        "feature_cols": feature_cols,
        "config":       config,
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        path:   Checkpoint file path.
        device: Torch device string.

    Returns:
        Checkpoint dict with all saved state.
    """
    return torch.load(path, map_location=device, weights_only=False)


def load_model(path: str | Path, device: str = "cpu") -> tuple[FinBrainLSTM, dict]:
    """Rebuild a FinBrainLSTM from a saved checkpoint.

    Args:
        path:   Checkpoint file path.
        device: Torch device string.

    Returns:
        (model, checkpoint_dict) ready for inference.
    """
    ckpt   = load_checkpoint(path, device)
    config = ckpt["config"]
    model  = FinBrainLSTM(
        input_size  = config["input_size"],
        hidden_size = config.get("hidden_size", 128),
        num_layers  = config.get("num_layers", 2),
        dropout     = config.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(
    model:       FinBrainLSTM,
    x:           np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std:  np.ndarray,
    device:      str = "cpu",
) -> float:
    """Run inference on a single feature sequence.

    Args:
        model:        Loaded FinBrainLSTM (eval mode).
        x:            Array of shape (seq_len, n_features).
        scaler_mean:  Feature mean for normalisation.
        scaler_std:   Feature std for normalisation.
        device:       Torch device.

    Returns:
        Probability in [0, 1] that next-day return is positive.
    """
    std_safe = np.where(scaler_std == 0, 1.0, scaler_std)
    x_norm   = (x - scaler_mean) / std_safe
    tensor   = torch.FloatTensor(x_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(tensor).squeeze())
