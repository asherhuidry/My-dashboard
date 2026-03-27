# Model Comparison and Promotion Workflow

`ml/comparison/` provides a repeatable workflow for evaluating multiple models
on the same dataset version and surfacing clear promotion recommendations.

## Overview

```
OHLCV data
    │
    ▼
assemble_dataset()  ─────────────────────────────────────────────────────────────┐
    │                                                                             │
    ├── run_baseline_pipeline()  →  PipelineResult (logistic)    dataset_version │ flat
    ├── run_pipeline()           →  PipelineResult (MLP)         dataset_version │ version
    └── run_lstm_pipeline()      →  PipelineResult (LSTM)        seq dataset_version
                                                                  comparison_group_version ← same flat version
    All results flow into ────────────────────────────────────────────────────────┘
    ComparisonResult (ranked by composite score)
```

Baseline and MLP store the flat `dataset_version` directly.  The LSTM stores its
own sequence-dataset version under `dataset_version` and records the flat version
under `comparison_group_version` so all three models are groupable by the flat key.

## Running a comparison

```python
from ml.comparison.runner import run_comparison

result = run_comparison(
    symbol         = "AAPL",
    df             = ohlcv_df,          # or omit to fetch from yfinance
    models         = ("baseline", "mlp"),
    registry       = registry,
    checkpoint_dir = Path("ml/checkpoints"),
    epochs         = 50,
    patience       = 10,
)
result.print_summary()
```

### LSTM participation

Pass `"lstm"` in `models` to include the LSTM.  It requires sufficient data
for the rolling window (seq_len=20 default, needs > ~100 sequences per split).
If the LSTM fails for any reason it is skipped with a warning and the
comparison completes with the remaining models.

```python
result = run_comparison(
    symbol = "AAPL",
    df     = ohlcv_df,
    models = ("baseline", "mlp", "lstm"),
    ...
)
```

## ComparisonResult

```python
result.symbol           # "AAPL"
result.dataset_version  # "3f9b1a2c7e4d"  — shared across all models
result.winner           # "mlp"  — highest composite score
result.generated_at     # ISO-8601

for mr in result.ranked():
    print(mr.rank, mr.model_type, mr.composite_score)
    print(mr.promotion_recommended, mr.promotion_reasons)
    print(mr.experiment_id)   # look up full record in registry

result.to_dict()        # JSON-serialisable
```

## Composite score

The composite score combines four metrics into a single ranking scalar:

| Component | Weight | Notes |
|-----------|--------|-------|
| Accuracy | 30% | Out-of-sample fraction correct |
| AUC | 25% | ROC area under curve |
| Hit rate | 25% | Fraction profitable backtest trades |
| Sharpe | 20% | Clamped to [-3, 3], rescaled to [0, 1] |

The composite score does **not** gate promotion — it is used only for ranking.
Promotion decisions use the hard-gate criteria below.

## Promotion criteria

All gate criteria must pass for `promotion_recommended = True`:

| Criterion | Gate? | Threshold |
|-----------|-------|-----------|
| Accuracy | ✓ gate | ≥ 0.55 |
| Hit rate | ✓ gate | ≥ 0.52 |
| Beat benchmark | ✓ gate | cumulative return > benchmark return |
| AUC | advisory | ≥ 0.52 |
| Sharpe | advisory | > 0.0 |

`promotion_recommended` is a flag — it does **not** call `registry.promote()`.
Promotion requires a deliberate human decision.

## Ranking utilities

```python
from ml.comparison.ranking import (
    rank_experiments,
    explain_promotion,
    compare_dataset_version,
    top_n,
)

# Rank all completed experiments sharing a dataset version
ranked = rank_experiments(registry, dataset_version="3f9b1a2c7e4d")
for rank, record in ranked:
    print(rank, record.model_type, record.metrics.get("accuracy"))

# Detailed promotion explanation
detail = explain_promotion(record)
print(detail["summary"])
for criterion in detail["criteria"]:
    print(criterion["criterion"], criterion["passed"], criterion["message"])

# All experiments on the same dataset
records = compare_dataset_version(registry, "3f9b1a2c7e4d")

# Top 5 by a single metric
best = top_n(registry, n=5, metric="sharpe")
```

## LSTM status

The LSTM path (`ml/patterns/train_lstm.py`) is complete and registry-integrated:

- Uses the existing `FinBrainLSTM` model and `PriceSequenceDataset`
- 70 / 15 / 15 split (train / val for early stopping / held-out test)
- Same `PipelineResult` output as MLP and baseline
- Backtest run on the held-out test set close prices
- Cross-references flat dataset version in `dataset_info.comparison_group_version`

**Limitation:** the LSTM test set uses the same close price window as MLP and
baseline only if `seq_len` is small relative to the data length.  Longer
sequences reduce the test window slightly.  For short datasets (< 200 bars),
prefer baseline and MLP comparison only.

## Example: three-model comparison with promotion review

```python
from ml.comparison.runner import run_comparison
from ml.comparison.ranking import rank_experiments, explain_promotion

# 1. Run all models
result = run_comparison("AAPL", df=ohlcv_df, models=("baseline", "mlp", "lstm"),
                        registry=reg, checkpoint_dir=Path("ml/checkpoints"))
result.print_summary()

# 2. Rank all experiments in the registry for this dataset
ranked = rank_experiments(reg, dataset_version=result.dataset_version)

# 3. Explain the winner's promotion status
winner_record = reg.get(result.ranked()[0].experiment_id)
detail = explain_promotion(winner_record)
print(detail["summary"])

# 4. Promote if appropriate (human decision)
if detail["overall_pass"]:
    reg.promote(winner_record.experiment_id, notes="Approved after comparison run")
```
