"""MLP baseline model for binary direction prediction.

A shallow multi-layer perceptron trained on engineered features to predict
next-day price direction (up/flat vs. down).  Designed to be simple,
interpretable, and fast to train — a baseline that more complex models must
beat to be worth the extra complexity.

Features expected
-----------------
The model accepts a flat feature vector per sample.  The caller is responsible
for feature engineering (see ``ml/patterns/features_expanded.py``) and for
normalizing inputs before passing them here.

Usage::

    from ml.patterns.mlp import MLP, MLPConfig, train, predict, evaluate

    cfg   = MLPConfig(input_size=60, hidden_sizes=[128, 64], dropout=0.2)
    model = MLP(cfg)

    # train
    history = train(model, X_train, y_train, X_val, y_val, cfg)

    # evaluate
    metrics = evaluate(model, X_val, y_val)

    # single prediction
    prob = predict(model, x)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MLPConfig:
    """Hyperparameters for the MLP baseline model.

    Attributes:
        input_size:   Number of input features.
        hidden_sizes: Sizes of hidden layers (e.g. [128, 64]).
        dropout:      Dropout rate applied after each hidden layer (0 = off).
        activation:   Activation function name ('relu', 'gelu', 'tanh').
        batch_norm:   Whether to apply BatchNorm after each hidden layer.
        lr:           Learning rate for AdamW.
        weight_decay: L2 regularisation coefficient.
        epochs:       Maximum training epochs.
        batch_size:   Mini-batch size.
        patience:     Early stopping patience (epochs without val improvement).
        threshold:    Decision threshold for converting probability to label.
    """
    input_size:   int         = 60
    hidden_sizes: list[int]   = field(default_factory=lambda: [128, 64])
    dropout:      float       = 0.2
    activation:   str         = "relu"
    batch_norm:   bool        = True
    lr:           float       = 1e-3
    weight_decay: float       = 1e-4
    epochs:       int         = 100
    batch_size:   int         = 64
    patience:     int         = 10
    threshold:    float       = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "input_size":   self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "dropout":      self.dropout,
            "activation":   self.activation,
            "batch_norm":   self.batch_norm,
            "lr":           self.lr,
            "weight_decay": self.weight_decay,
            "epochs":       self.epochs,
            "batch_size":   self.batch_size,
            "patience":     self.patience,
            "threshold":    self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLPConfig":
        """Reconstruct an MLPConfig from a dictionary."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Model ─────────────────────────────────────────────────────────────────────

def _activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    mapping = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(mapping)}")
    return mapping[name]


class MLP(nn.Module):
    """Multi-layer perceptron for binary direction prediction.

    Args:
        config: MLPConfig with all hyperparameter values.

    Inputs:
        x: Float tensor of shape ``(batch, input_size)``.

    Outputs:
        Logit tensor of shape ``(batch, 1)``.  Use ``torch.sigmoid`` to
        convert to a probability.
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_size = config.input_size
        act = _activation(config.activation)

        for out_size in config.hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(act)
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_size = out_size

        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialisation to all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(batch, input_size)``.

        Returns:
            Logit tensor ``(batch, 1)``.
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability of the positive class (up).

        Args:
            x: Input tensor ``(batch, input_size)`` or ``(input_size,)``.

        Returns:
            Probability tensor ``(batch,)`` in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return torch.sigmoid(self.forward(x)).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    """Record of per-epoch training metrics.

    Attributes:
        train_loss: Training loss per epoch.
        val_loss:   Validation loss per epoch.
        val_acc:    Validation accuracy per epoch.
        best_epoch: Epoch with the best validation loss (0-indexed).
    """
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)
    best_epoch: int         = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss":   self.val_loss,
            "val_acc":    self.val_acc,
            "best_epoch": self.best_epoch,
        }


def train(
    model:   MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    config:  MLPConfig | None = None,
) -> TrainingHistory:
    """Train the MLP with early stopping.

    Args:
        model:   MLP instance (modified in-place).
        X_train: Training features ``(n_train, input_size)``.
        y_train: Training labels ``(n_train,)`` — float binary (0 or 1).
        X_val:   Validation features ``(n_val, input_size)``.
        y_val:   Validation labels ``(n_val,)`` — float binary (0 or 1).
        config:  MLPConfig; if None, uses ``model.config``.

    Returns:
        TrainingHistory with per-epoch metrics and best_epoch index.
    """
    cfg = config or model.config
    device = next(model.parameters()).device

    # Move data to model device
    X_train = X_train.to(device)
    y_train = y_train.to(device).float()
    X_val   = X_val.to(device)
    y_val   = y_val.to(device).float()

    train_ds = TensorDataset(X_train, y_train)
    loader   = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history  = TrainingHistory()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    no_improve = 0

    for epoch in range(cfg.epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(X_train)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze(1)
            val_loss   = criterion(val_logits, y_val).item()
            val_probs  = torch.sigmoid(val_logits)
            val_preds  = (val_probs >= cfg.threshold).float()
            val_acc    = (val_preds == y_val).float().mean().item()

        scheduler.step(val_loss)
        history.train_loss.append(round(train_loss, 6))
        history.val_loss.append(round(val_loss, 6))
        history.val_acc.append(round(val_acc, 4))

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        if (epoch + 1) % 10 == 0:
            log.debug(
                "Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                epoch + 1, cfg.epochs, train_loss, val_loss, val_acc,
            )

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    log.info(
        "Training complete — best_epoch=%d val_loss=%.4f val_acc=%.3f",
        history.best_epoch + 1,
        history.val_loss[history.best_epoch],
        history.val_acc[history.best_epoch],
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model:     MLP,
    X:         torch.Tensor,
    y:         torch.Tensor,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate the model on a held-out split.

    Args:
        model:     Trained MLP.
        X:         Feature tensor ``(n, input_size)``.
        y:         Label tensor ``(n,)`` — float binary (0 or 1).
        threshold: Decision threshold (defaults to ``model.config.threshold``).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc (all floats).
    """
    thr = threshold if threshold is not None else model.config.threshold
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device).float()

    probs = model.predict_proba(X)
    preds = (probs >= thr).float()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # AUC approximation via sorted threshold sweep
    auc = _roc_auc(probs.cpu().numpy(), y.cpu().numpy())

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc":       round(auc, 4),
    }


def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC using the trapezoidal rule.

    Args:
        probs:  Predicted probabilities, shape ``(n,)``.
        labels: True binary labels, shape ``(n,)``.

    Returns:
        AUC score in [0, 1].
    """
    thresholds = np.linspace(0, 1, 101)[::-1]
    tprs, fprs = [], []
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    for t in thresholds:
        preds = (probs >= t).astype(float)
        tpr   = ((preds == 1) & (labels == 1)).sum() / n_pos
        fpr   = ((preds == 1) & (labels == 0)).sum() / n_neg
        tprs.append(tpr)
        fprs.append(fpr)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(tprs, fprs))  # fprs increases as threshold decreases


# ── Single-sample prediction ──────────────────────────────────────────────────

def predict(
    model:    MLP,
    x:        torch.Tensor | np.ndarray,
) -> float:
    """Return the up-direction probability for a single sample.

    Args:
        model: Trained MLP.
        x:     Feature vector ``(input_size,)`` — numpy array or tensor.

    Returns:
        Probability in [0, 1] that the next move is up.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    device = next(model.parameters()).device
    return model.predict_proba(x.to(device)).item()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model:   MLP,
    path:    Path | str,
    history: TrainingHistory | None = None,
    extra:   dict[str, Any] | None  = None,
) -> None:
    """Save the model and its config to a .pt checkpoint.

    Args:
        model:   Trained MLP to save.
        path:    Destination file path.
        history: Optional TrainingHistory to embed in the checkpoint.
        extra:   Optional extra metadata dict to store alongside the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config":     model.config.to_dict(),
    }
    if history is not None:
        payload["history"] = history.to_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    log.info("Checkpoint saved: %s", path)


def load_checkpoint(path: Path | str) -> tuple[MLP, dict[str, Any]]:
    """Load an MLP and its metadata from a .pt checkpoint.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Tuple of (model, metadata_dict).  The metadata dict contains at
        minimum ``'config'``; optionally ``'history'`` and ``'extra'``.
    """
    path    = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg     = MLPConfig.from_dict(payload["config"])
    model   = MLP(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {k: v for k, v in payload.items() if k not in ("state_dict",)}
    log.info("Checkpoint loaded: %s", path)
    return model, meta


# ── Normalisation helpers ─────────────────────────────────────────────────────

@dataclass
class FeatureScaler:
    """Z-score normaliser fitted on training data.

    Attributes:
        mean_: Per-feature mean vector.
        std_:  Per-feature std vector (NaN-safe; zeros replaced by 1).
    """
    mean_: np.ndarray | None = None
    std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit on a training matrix ``(n_samples, n_features)``.

        Args:
            X: Training feature matrix.

        Returns:
            Self (for chaining).
        """
        self.mean_ = np.nanmean(X, axis=0)
        self.std_  = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted normalisation.

        Args:
            X: Feature matrix ``(n_samples, n_features)`` or ``(n_features,)``.

        Returns:
            Z-score normalised array of the same shape.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler has not been fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and immediately transform.

        Args:
            X: Training feature matrix.

        Returns:
            Normalised matrix.
        """
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "mean_": self.mean_.tolist() if self.mean_ is not None else None,
            "std_":  self.std_.tolist()  if self.std_  is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureScaler":
        """Reconstruct from a serialised dict."""
        scaler = cls()
        if d.get("mean_") is not None:
            scaler.mean_ = np.array(d["mean_"])
        if d.get("std_") is not None:
            scaler.std_  = np.array(d["std_"])
        return scaler
