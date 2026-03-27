# Walk-Forward Validation

`ml/validation/` provides a reusable, leakage-free walk-forward cross-validation
workflow for the FinBrain model research loop.

Walk-forward validation addresses the core limitation of a single train/val/test
split: one test window is too small a sample to distinguish a genuinely
generalising model from a lucky draw.  Multiple chronological folds provide
evidence of consistency.

---

## Architecture

```
OHLCV data
    │
    ▼
assemble_walk_forward_dataset()   ←  feature engineering once
    │
    ├── make_folds()              ←  fold index specs (no data copied)
    │
    └── for each fold:
            slice_fold()          ←  train / val / test arrays (unscaled)
            fit scaler on train   ←  per-fold, no look-ahead in normalisation
            │
            ├── _eval_baseline_fold()
            ├── _eval_mlp_fold()
            └── _eval_lstm_fold()  (best-effort; may skip if too few sequences)
                │
                ├── FoldResult
                └── ...
                        │
                        ▼
                aggregate_folds()          →  FoldAggregate (mean/std/min/max)
                wf_promotion_recommend()   →  WalkForwardPromotion
```

---

## Fold structure

Each fold has three non-overlapping windows:

```
Time →

|── train ──────────────────────| val |  gap  | test |
^                                             ^      ^
train_start                         test_start      test_end
```

- **train**: rows used to fit the model and scaler.
- **val**: the tail of the training window, used for early stopping (MLP/LSTM).
  The scaler is fit on pure training rows only; val rows are then transformed
  without fitting.
- **gap**: optional rows skipped between train and test (default 0).
- **test**: held-out rows never seen during training or tuning.

For `window='expanding'` (default), the train window grows each fold:

```
Fold 0:  [===train0===|val0] [test0]
Fold 1:  [=====train1=======|val1] [test1]
Fold 2:  [========train2=========|val2] [test2]
```

For `window='rolling'`, train size is fixed and the window slides:

```
Fold 0:  [===train0===|val0] [test0]
Fold 1:      [===train1===|val1] [test1]
Fold 2:          [===train2===|val2] [test2]
```

---

## Configuring folds

```python
from ml.validation.walk_forward import WalkForwardConfig

cfg = WalkForwardConfig(
    n_folds        = 5,      # number of train/test folds
    min_train_size = 120,    # minimum rows in first train window
    test_size      = None,   # auto-computed; override with an int
    val_frac       = 0.15,   # tail of train window used for validation
    gap            = 0,      # rows skipped between train end and test start
    window         = "expanding",  # or "rolling"
)
```

**Minimum data requirement (rough guide):**

```
n_valid_rows ≥ min_train_size + n_folds × test_size
```

With `min_train_size=120`, `n_folds=5`, and `n_valid_rows=250`, the auto-computed
`test_size = (250 − 120) / 5 = 26` rows per fold.

---

## Running walk-forward evaluation

### Single model

```python
from ml.validation.walk_forward import WalkForwardConfig
from ml.validation.wf_runner import run_walk_forward_model

cfg = WalkForwardConfig(n_folds=5, min_train_size=120)

wf = run_walk_forward_model(
    model_key  = "baseline",   # "baseline" | "mlp" | "lstm"
    symbol     = "AAPL",
    df         = ohlcv_df,
    config     = cfg,
    epochs     = 20,
    patience   = 5,
)

print(f"Completed {wf.n_folds_run}/{wf.n_folds_total} folds")
for fr in wf.fold_results:
    print(fr.fold_idx, fr.metrics["accuracy"], fr.backtest_summary["hit_rate"])
```

### Multiple models (shared dataset)

```python
from ml.validation.wf_runner import run_walk_forward

results = run_walk_forward(
    symbol   = "AAPL",
    df       = ohlcv_df,
    models   = ("baseline", "mlp"),
    config   = cfg,
    epochs   = 20,
    patience = 5,
)
# results: dict[model_type, WalkForwardResult]
```

The full dataset is assembled **once** and shared across all models.

### Via the comparison runner

```python
from ml.comparison.runner import run_comparison
from ml.validation.walk_forward import WalkForwardConfig

wf_result = run_comparison(
    symbol       = "AAPL",
    df           = ohlcv_df,
    models       = ("baseline", "mlp"),
    walk_forward = True,
    wf_config    = WalkForwardConfig(n_folds=5),
    epochs       = 20,
    patience     = 5,
)
wf_result.print_summary()
print("Winner:", wf_result.winner)
```

---

## Aggregated fold statistics

```python
from ml.validation.wf_aggregation import aggregate_folds

agg = aggregate_folds(wf.fold_results, model_type="baseline")

print(agg.mean_accuracy)          # mean accuracy across folds
print(agg.std_accuracy)           # standard deviation — stability signal
print(agg.min_accuracy)           # worst fold
print(agg.max_accuracy)           # best fold
print(agg.n_folds_beat_benchmark) # how many folds beat buy-and-hold
print(agg.n_folds_promo_recommended) # how many folds passed all single-fold gates
```

---

## Variance-aware promotion

```python
from ml.validation.wf_aggregation import wf_promotion_recommend, WalkForwardPromotionConfig

promo = wf_promotion_recommend(agg)
print(promo.summary)
for criterion in promo.criteria:
    print(criterion["criterion"], criterion["passed"], criterion["message"])
```

### Promotion criteria

| Criterion | Gate? | Default threshold |
|-----------|-------|-------------------|
| mean_accuracy | ✓ gate | ≥ 0.55 |
| mean_hit_rate | ✓ gate | ≥ 0.52 |
| folds_beat_benchmark | ✓ gate | ≥ ⌈n_folds / 2⌉ |
| std_accuracy | advisory (gate if `std_is_gate=True`) | ≤ 0.08 |
| mean_auc | advisory | ≥ 0.52 |
| mean_sharpe | advisory | > 0.0 |

All gate criteria must pass for `overall_recommended = True`.
Advisory criteria are reported but do not block promotion.

Custom thresholds:

```python
from ml.validation.wf_aggregation import WalkForwardPromotionConfig, wf_promotion_recommend

cfg = WalkForwardPromotionConfig(
    min_mean_accuracy  = 0.56,
    min_mean_hit_rate  = 0.53,
    min_folds_beat_bm  = 4,      # must beat benchmark on 4 of 5 folds
    max_std_accuracy   = 0.06,
    std_is_gate        = True,   # variance becomes a hard gate
)
promo = wf_promotion_recommend(agg, config=cfg)
```

`overall_recommended` is a flag — **it does NOT call `registry.promote()`**.
Promotion requires a deliberate human decision.

---

## Walk-forward comparison result

`WalkForwardComparisonResult` exposes:

```python
wf_result.winner          # model_type with highest mean composite score
wf_result.ranked()        # list of (model_type, FoldAggregate) best-first
wf_result.aggregates      # dict[model_type, FoldAggregate]
wf_result.promotions      # dict[model_type, WalkForwardPromotion]
wf_result.to_dict()       # JSON-serialisable
```

### Composite score ranking

The per-fold composite score uses the same weighted formula as the single-split
comparison:

| Component | Weight |
|-----------|--------|
| Accuracy  | 30%    |
| AUC       | 25%    |
| Hit rate  | 25%    |
| Sharpe (normalised) | 20% |

`WalkForwardComparisonResult.winner` is the model with the highest
**mean composite score** across folds — not just the best single fold.

---

## LSTM support and limitations

LSTM walk-forward is supported on a **best-effort** basis.

Each LSTM fold uses the combined feature window `[train_start, test_end)` to
construct rolling sequences, then splits chronologically into train / val / test.
If a fold contains fewer than 40 total sequences, it is **skipped** with a
warning and excluded from the `WalkForwardResult`.

When using LSTM in walk-forward mode:

- Use `n_folds ≤ 3` for datasets shorter than 500 bars.
- The test set close prices for LSTM are offset by `seq_len` rows from the
  flat-model test set.  Backtest results are comparable in direction but not
  exactly aligned by date.
- LSTM fold training is slower than baseline/MLP — budget ~10× more time per
  fold with the default `LSTMConfig`.

For short datasets (< 200 bars), use baseline and MLP only:

```python
results = run_walk_forward(
    symbol = "AAPL",
    df     = ohlcv_df,
    models = ("baseline", "mlp"),  # omit "lstm"
    config = WalkForwardConfig(n_folds=3, min_train_size=100),
)
```
