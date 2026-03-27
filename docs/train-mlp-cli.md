# MLP Training CLI

**Module:** `ml/patterns/train_mlp.py`

---

## Purpose

A single end-to-end command that downloads price history, engineers features, trains the MLP baseline model, evaluates it on a held-out test set, runs a backtest, and records everything in the experiment registry. Designed to be runnable with a single command with sensible defaults.

---

## Quick start

```bash
# Basic run — trains on 3 years of AAPL data
python -m ml.patterns.train_mlp --symbol AAPL

# Custom architecture and longer period
python -m ml.patterns.train_mlp \
  --symbol MSFT \
  --period 4y \
  --hidden 256 128 64 \
  --dropout 0.3 \
  --lr 0.0005 \
  --epochs 200 \
  --patience 15

# JSON output (for scripting)
python -m ml.patterns.train_mlp --symbol NVDA --json
```

---

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | `AAPL` | Ticker symbol to train on |
| `--period` | `3y` | yfinance period string (e.g. `1y`, `2y`, `5y`) |
| `--horizon` | `1` | Prediction horizon in bars (1 = next-day direction) |
| `--hidden` | `128 64` | Hidden layer widths (space-separated) |
| `--dropout` | `0.2` | Dropout rate |
| `--lr` | `0.001` | AdamW learning rate |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `10` | Early-stopping patience |
| `--batch` | `64` | Mini-batch size |
| `--threshold` | `0.55` | Backtest signal threshold (go long ≥ this) |
| `--tags` | (none) | Space-separated experiment tags |
| `--notes` | (none) | Free-text notes attached to the experiment |
| `--json` | off | Print result as JSON instead of human-readable table |

---

## What it does

1. **Download** — fetches OHLCV history via yfinance for `--period`
2. **Feature engineering** — calls `ml.patterns.features.build_features()` (~80 features)
3. **Target** — binary next-day direction: 1 if `close[t+1] > close[t]`, else 0
4. **Chronological split** — 70% train / 15% val / 15% test (no shuffling)
5. **Normalise** — Z-score via `FeatureScaler` fitted on train set only
6. **Train** — MLP with AdamW + early stopping; best weights restored
7. **Evaluate** — accuracy, precision, recall, F1, AUC on test set
8. **Save** — checkpoint `.pt` and scaler `.json` to `ml/checkpoints/`
9. **Backtest** — vectorised backtest on test-set prices using model scores as signals
10. **Register** — creates/updates an `ExperimentRecord` with all outputs
11. **Recommend** — prints whether the model meets promotion thresholds

---

## Promotion thresholds

A model is flagged as **PROMOTION RECOMMENDED** if it meets all three:

| Threshold | Value | Description |
|-----------|-------|-------------|
| Minimum accuracy | 55% | Test-set direction accuracy |
| Minimum hit rate | 52% | Backtest fraction of profitable trades |
| Beat benchmark | > 0% | Cumulative return exceeds buy-and-hold |

The CLI prints a recommendation but does **not** auto-promote. To promote:

```python
from ml.registry import ExperimentRegistry, ExperimentStatus

reg = ExperimentRegistry()
reg.promote("your-experiment-id", notes="Manual review passed")
```

---

## Programmatic usage

```python
from ml.patterns.train_mlp import run_pipeline

result = run_pipeline(
    symbol             = "AAPL",
    period             = "3y",
    hidden_sizes       = [128, 64],
    backtest_threshold = 0.55,
)

print(result.experiment_id)        # experiment UUID
print(result.metrics)              # {'accuracy': 0.58, 'f1': 0.57, ...}
print(result.backtest_summary)     # backtest dict
print(result.promotion_recommended) # True / False
```

You can also pass a pre-built DataFrame to skip the yfinance download:

```python
result = run_pipeline(symbol="AAPL", df=my_df)
```

---

## Output files

| File | Description |
|------|-------------|
| `ml/checkpoints/<exp_id>.pt` | PyTorch model checkpoint + training history |
| `ml/checkpoints/<exp_id>_scaler.json` | Fitted FeatureScaler (needed for inference) |
| `data/registry/experiments.json` | Updated experiment registry |

---

## Assumptions and limitations

- **Feature set**: uses `build_features()` from `ml/patterns/features.py` (~80 features). No macro data is joined in this version (the function accepts it as an optional parameter but this CLI does not fetch it yet).
- **Target**: simple next-day binary direction. Does not account for transaction costs in the target label.
- **No walk-forward CV**: training and evaluation are single-window chronological splits, not rolling walk-forward. This is intentional for the baseline — walk-forward is Phase 5+.
- **Backtest**: runs only on the test set (last 15% of history). This is short but avoids leaking the training period into the backtest.
- **No auto-promotion**: the pipeline logs a recommendation but does not call `reg.promote()`. A human must review and promote manually.
