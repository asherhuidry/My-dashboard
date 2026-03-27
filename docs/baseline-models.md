# Baseline Models

`ml/patterns/baseline.py` provides a logistic regression baseline with the same
interface as the MLP so both can participate in the same comparison flow.

## Why logistic regression

- **Transparent** — coefficients are interpretable; overfitting is easy to detect
- **Fast** — trains in milliseconds even on 10k rows
- **Strong sanity check** — if the MLP cannot beat it, the features are not
  adding non-linear value
- **No special infrastructure** — uses scikit-learn which is already a dependency

## Quick start

```python
from ml.patterns.baseline import run_baseline_pipeline

result = run_baseline_pipeline(
    symbol         = "AAPL",
    df             = ohlcv_df,
    registry       = registry,
    checkpoint_dir = Path("ml/checkpoints"),
)
result.print_summary()
```

## Sharing a dataset with the MLP

```python
from ml.data.dataset_builder import assemble_dataset
from ml.patterns.baseline import run_baseline_pipeline
from ml.patterns.train_mlp import run_pipeline as run_mlp

data, meta = assemble_dataset(ohlcv_df, symbol="AAPL")

baseline_result = run_baseline_pipeline(
    symbol       = "AAPL",
    dataset      = data,
    dataset_meta = meta,
    registry     = registry,
)

mlp_result = run_mlp(
    symbol   = "AAPL",
    df       = ohlcv_df,
    registry = registry,
)
```

Passing `dataset=data` ensures both models use *exactly* the same feature
arrays and splits.  Both experiment records will share the same
`dataset_version` in the registry.

## BaselineConfig

```python
from ml.patterns.baseline import BaselineConfig

cfg = BaselineConfig(
    C            = 1.0,       # regularisation inverse strength
    max_iter     = 1000,      # solver iterations
    class_weight = "balanced",# compensate for class imbalance
    solver       = "lbfgs",   # or "saga" for very large datasets
)
```

## LogisticBaseline API

| Method | Description |
|--------|-------------|
| `fit(X_train, y_train)` | Fit and return `{"train_accuracy": ...}` |
| `predict_proba(X)` | Return float32 probability array in `[0, 1]` |
| `evaluate(X, y, threshold=0.5)` | Return `accuracy, precision, recall, f1, auc` |
| `save(path)` | Pickle model + JSON sidecar |
| `LogisticBaseline.load(path)` | Reload from pickle |

## Pipeline output

`run_baseline_pipeline` returns a `PipelineResult` identical in structure to
the MLP pipeline:

```python
result.experiment_id        # registry ID
result.metrics              # accuracy, f1, auc, precision, recall
result.backtest_summary     # cumulative_return, hit_rate, sharpe, ...
result.checkpoint_path      # path to .pkl
result.scaler_path          # path to scaler JSON
result.promotion_recommended # bool
result.promotion_reasons    # list of threshold messages
```

## Promotion thresholds

Same thresholds as the MLP (defined in `train_mlp.py`):

| Criterion | Threshold |
|-----------|-----------|
| Accuracy | ≥ 0.55 |
| Backtest hit rate | ≥ 0.52 |
| Cumulative return > benchmark | > 0.0 |

All three must pass for `promotion_recommended = True`.  Actual promotion
requires a human call to `registry.promote(experiment_id)`.

## Checkpoint format

The checkpoint is a pickle file containing:

```python
{
    "config": {"C": 1.0, "max_iter": 1000, ...},
    "model":  <sklearn LogisticRegression>
}
```

A `.json` sidecar (same stem, `.json` extension) stores the config as plain
text for inspection without loading the pickle.
