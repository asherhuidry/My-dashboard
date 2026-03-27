# Experiment Registry

**Module:** `ml/registry/`
**Persistence:** `data/registry/experiments.json` (local-first)

---

## Purpose

Every model training run is tracked in the experiment registry with its hyperparameters, dataset info, evaluation metrics, and backtest results. The registry enforces a promotion gate: a model cannot be marked PROMOTED unless it has an attached `BacktestResult`.

---

## Quick start

```python
from ml.registry import ExperimentRegistry, BacktestResult

reg = ExperimentRegistry()

# 1. Start a run
exp = reg.create(
    name        = "mlp_baseline_v2",
    model_type  = "mlp",
    hyperparams = {"hidden_sizes": [128, 64], "lr": 0.001, "epochs": 50},
    dataset_info= {"symbols": ["AAPL", "MSFT"], "start": "2020-01-01", "end": "2024-12-31"},
    notes       = "Wider hidden layer test",
    tags        = ["baseline", "ohlcv"],
)

# 2. Train model ...

# 3. Record results
reg.finish(
    exp.experiment_id,
    metrics         = {"val_accuracy": 0.63, "val_loss": 0.54},
    checkpoint_path = "ml/checkpoints/mlp_v2.pt",
)

# 4. Run backtest and attach
result = BacktestResult(
    cumulative_return = 0.18,
    annualised_return = 0.11,
    hit_rate          = 0.58,
    max_drawdown      = -0.09,
    sharpe            = 1.2,
    trade_count       = 142,
    benchmark_return  = 0.14,
    period_start      = "2023-01-01",
    period_end        = "2024-12-31",
)
reg.attach_backtest(exp.experiment_id, result)

# 5. Promote (only works if backtest is attached)
reg.promote(exp.experiment_id, notes="Beats benchmark by 4pp")
```

---

## Lifecycle states

```
RUNNING → COMPLETED → PROMOTED
        → FAILED
  any state → ARCHIVED
```

| Status | Meaning |
|--------|---------|
| `running` | Training is in progress |
| `completed` | Training finished; metrics recorded |
| `failed` | Training raised an unhandled exception |
| `promoted` | Passed backtest gate; production candidate |
| `archived` | No longer active; retained for reference |

---

## ExperimentRecord fields

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id` | `str` | UUID |
| `name` | `str` | Human-readable run name |
| `model_type` | `str` | Model family tag (`mlp`, `lstm`, `gnn`, …) |
| `status` | `ExperimentStatus` | Current lifecycle state |
| `hyperparams` | `dict` | Hyperparameter values used |
| `dataset_info` | `dict` | Training dataset description |
| `metrics` | `dict[str, float]` | Evaluation metrics (set on `finish()`) |
| `backtest` | `BacktestResult \| None` | Attached backtest result |
| `notes` | `str` | Free-text notes (appended, never overwritten) |
| `checkpoint_path` | `str` | Path to saved model file |
| `started_at` | `str` | ISO-8601 timestamp |
| `finished_at` | `str \| None` | ISO-8601 timestamp when done |
| `tags` | `list[str]` | Filterable string tags |

Computed properties:
- `is_running` — True if status is RUNNING
- `duration_seconds` — wall-clock duration, or None if not finished

---

## BacktestResult fields

| Field | Type | Description |
|-------|------|-------------|
| `cumulative_return` | `float` | Total return (e.g. `0.18` = 18%) |
| `annualised_return` | `float` | Annualised equivalent |
| `hit_rate` | `float` | Fraction of correct direction calls (0–1) |
| `max_drawdown` | `float` | Max peak-to-trough loss (negative, e.g. `-0.09`) |
| `sharpe` | `float` | Sharpe-like ratio |
| `trade_count` | `int` | Number of trades executed |
| `benchmark_return` | `float` | Buy-and-hold return over same period |
| `period_start` | `str` | ISO-8601 backtest start date |
| `period_end` | `str` | ISO-8601 backtest end date |
| `extra` | `dict` | Additional metrics (per-symbol breakdown, etc.) |

---

## Querying

```python
# All experiments, most recent first
reg.all()

# Filter
reg.filter(model_type="mlp", status=ExperimentStatus.PROMOTED)
reg.filter(tag="baseline", has_backtest=True)

# Best model by metric
best = reg.best("val_accuracy", model_type="mlp")

# Summary
reg.summary()
# {
#   "total": 12,
#   "by_status": {"completed": 8, "promoted": 2, "failed": 2},
#   "by_model":  {"mlp": 7, "lstm": 5},
#   "promoted": 2,
#   "with_backtest": 4
# }
```

---

## Promotion gate

`promote()` raises `ValueError` if:
1. The experiment is not in `COMPLETED` status, or
2. No `BacktestResult` has been attached via `attach_backtest()`.

This ensures no model reaches PROMOTED status without measured backtest evidence.

---

## Environment variable

Override the default storage path:

```bash
export FINBRAIN_EXPERIMENT_REGISTRY_PATH=/path/to/experiments.json
```
