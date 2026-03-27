# MLP Baseline Model

**Module:** `ml/patterns/mlp.py`

---

## Purpose

A shallow multi-layer perceptron trained on engineered features to predict next-day price direction (up vs. down/flat). Designed to be simple, fast to train, and easy to interpret — the baseline that every more complex model must beat.

---

## Quick start

```python
import torch
from ml.patterns.mlp import MLP, MLPConfig, train, evaluate, predict, save_checkpoint

# 1. Configure
cfg = MLPConfig(
    input_size   = 60,
    hidden_sizes = [128, 64],
    dropout      = 0.2,
    lr           = 0.001,
    epochs       = 100,
    patience     = 10,
)

# 2. Build model
model = MLP(cfg)

# 3. Prepare tensors  (features already normalised)
X_train = torch.randn(800, 60)
y_train = torch.randint(0, 2, (800,)).float()
X_val   = torch.randn(200, 60)
y_val   = torch.randint(0, 2, (200,)).float()

# 4. Train
history = train(model, X_train, y_train, X_val, y_val)

# 5. Evaluate
metrics = evaluate(model, X_val, y_val)
# {'accuracy': 0.62, 'precision': 0.61, 'recall': 0.64, 'f1': 0.62, 'auc': 0.65}

# 6. Save
save_checkpoint(model, "ml/checkpoints/mlp_v1.pt", history=history)
```

---

## Feature normalisation

Always normalise inputs with `FeatureScaler` before training or inference:

```python
from ml.patterns.mlp import FeatureScaler
import numpy as np

scaler = FeatureScaler()
X_train_norm = scaler.fit_transform(X_train_raw)   # fit on train only
X_val_norm   = scaler.transform(X_val_raw)
X_test_norm  = scaler.transform(X_test_raw)

# Serialise with the checkpoint
import json
Path("scaler.json").write_text(json.dumps(scaler.to_dict()))

# Reload
scaler2 = FeatureScaler.from_dict(json.loads(Path("scaler.json").read_text()))
```

---

## MLPConfig reference

| Field | Default | Description |
|-------|---------|-------------|
| `input_size` | 60 | Number of input features |
| `hidden_sizes` | `[128, 64]` | Hidden layer widths |
| `dropout` | 0.2 | Dropout rate (0 = off) |
| `activation` | `"relu"` | `relu`, `gelu`, or `tanh` |
| `batch_norm` | True | BatchNorm after each hidden layer |
| `lr` | 0.001 | AdamW learning rate |
| `weight_decay` | 1e-4 | L2 regularisation |
| `epochs` | 100 | Max training epochs |
| `batch_size` | 64 | Mini-batch size |
| `patience` | 10 | Early stopping patience |
| `threshold` | 0.5 | Probability threshold for positive class |

---

## Training details

- **Optimiser:** AdamW with gradient clipping (`max_norm=1.0`)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss:** BCEWithLogitsLoss
- **Early stopping:** Restores best weights when validation loss stops improving
- **Initialisation:** Kaiming normal for all linear layers

---

## Checkpoints

```python
from ml.patterns.mlp import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, "ml/checkpoints/mlp_v1.pt", history=history, extra={"symbols": ["AAPL"]})

# Load
model, meta = load_checkpoint("ml/checkpoints/mlp_v1.pt")
print(meta["history"])   # TrainingHistory dict
print(meta["extra"])     # {"symbols": ["AAPL"]}
```

---

## Evaluation metrics

`evaluate()` returns:

| Metric | Description |
|--------|-------------|
| `accuracy` | Fraction of correct predictions |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1` | Harmonic mean of precision and recall |
| `auc` | Area under the ROC curve |

---

## Integration with Experiment Registry

```python
from ml.registry import ExperimentRegistry

reg = ExperimentRegistry()
exp = reg.create(
    name        = "mlp_baseline_v1",
    model_type  = "mlp",
    hyperparams = cfg.to_dict(),
    dataset_info= {"symbols": ["AAPL", "MSFT"], "n_train": 800},
)

history = train(model, X_train, y_train, X_val, y_val, cfg)
metrics = evaluate(model, X_val, y_val)

save_checkpoint(model, f"ml/checkpoints/{exp.experiment_id}.pt", history=history)
reg.finish(exp.experiment_id, metrics=metrics, checkpoint_path=str(checkpoint_path))
```
