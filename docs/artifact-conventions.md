# Artifact and Storage Conventions

This document describes where generated files land, what is tracked in git, and
what is treated as runtime output.

---

## Checkpoint files (`ml/checkpoints/`)

Model checkpoints and scaler files from training runs land in `ml/checkpoints/`.

| File pattern | Contents |
|---|---|
| `{experiment_id}.pt` | PyTorch model state dict + config + training history |
| `{experiment_id}_scaler.json` | `FeatureScaler` mean/std arrays for inference |

**Git status:** ignored (`ml/checkpoints/` is in `.gitignore`).

Checkpoints are referenced by path in the experiment registry but are not
committed.  To reproduce a run, retrain from the registry record's
`hyperparams` and `dataset_info`.

---

## Experiment registry (`data/registry/experiments.json`)

Flat JSON log of every experiment recorded through `ExperimentRegistry`.  Each
record holds: id, model type, hyperparams, dataset info, metrics, backtest
results, promotion status, and the checkpoint path.

**Git status:** tracked.

The registry accumulates over time and is the primary audit trail for model
comparisons.  It is intentionally committed so the research history survives
across machines and clones.

---

## Run output summaries (`ml/outputs/`)

JSON summaries written by `result.save_summary()` after comparison or
walk-forward runs.

| Directory | Contents |
|---|---|
| `ml/outputs/comparison/` | Single-split comparison results (`comparison_{symbol}_{ts}.json`) |
| `ml/outputs/walk_forward/` | Walk-forward comparison results (`wf_{symbol}_{ts}.json`) |

**Git status:** JSON/JSONL files in these directories are ignored; the
`.gitkeep` stubs that create the directories are tracked.

Summary files are runtime inspection artifacts.  Load them with
`json.loads(path.read_text())` or pass the dict to
`WalkForwardComparisonResult` / `ComparisonResult` for re-display.

---

## Source registry (`data/registry/source_registry.py`, `seed_sources.py`)

Python modules defining known data sources.  Tracked in git.

---

## What to commit vs. ignore

| Artifact | Commit? | Notes |
|---|---|---|
| `ml/checkpoints/*.pt` | No | Gitignored; reproduce by retraining |
| `ml/checkpoints/*_scaler.json` | No | Gitignored; saved alongside .pt files |
| `data/registry/experiments.json` | Yes | Audit trail of all training runs |
| `ml/outputs/**/*.json` | No | Runtime inspection output; gitignored |
| `ml/outputs/**/.gitkeep` | Yes | Directory stubs only |
| `docs/*.md` | Yes | |
| `tests/*.py` | Yes | |

---

## Walk-forward vs. single-split artifacts

**Single-split** (`run_comparison(walk_forward=False)`):

- One `ExperimentRecord` per model written to the registry.
- One checkpoint + scaler per model in `ml/checkpoints/`.
- Optionally one `comparison_{symbol}_{ts}.json` via `result.save_summary()`.

**Walk-forward** (`run_comparison(walk_forward=True)`):

- Per-fold results are in-memory only by default (`register_folds=False`).
- No per-fold checkpoints are written (fold training is ephemeral).
- The full comparison summary can be saved via `result.save_summary()`,
  which writes a single `wf_{symbol}_{ts}.json` containing all fold metrics,
  aggregates, and promotion recommendations.
- To persist fold-level registry entries (for deeper audit), pass
  `register_folds=True` to `run_walk_forward()`.

---

## Reproducing a walk-forward run

```python
import json
from pathlib import Path
from ml.comparison.runner import run_comparison
from ml.validation.walk_forward import WalkForwardConfig

# Load a previously saved summary for inspection
summary = json.loads(
    Path("ml/outputs/walk_forward/wf_AAPL_2026-03-27_023409.json").read_text()
)
# Re-run with the same configuration
cfg = WalkForwardConfig(**summary["wf_config"])
result = run_comparison(
    symbol       = summary["symbol"],
    models       = list(summary["aggregates"].keys()),
    walk_forward = True,
    wf_config    = cfg,
)
result.print_summary()
```
