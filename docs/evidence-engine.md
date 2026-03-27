# Evidence Engine

`ml/evidence/` provides a structured, local-first layer for representing and
evaluating reasoning-backed claims about data sources, features, and
relationships.

---

## What it is

A deterministic, serializable system for recording *structured claims* about
the research domain and attaching *evidence items* to those claims.

The evaluation logic is simple rule-based scoring — it tells you how well a
claim is supported by its attached evidence.  It does not generate hypotheses,
does not talk to an LLM, and does not auto-promote anything.

---

## What it is NOT

- **Not a reasoning model.** No inference, no language model, no narrative
  confidence treated as truth.
- **Not auto-promotion.** A claim with a "strong" evaluation is a signal to a
  human, not a trigger for action.
- **Not ground truth.** Claims are assertions.  They may be wrong.
- **Not a graph database.** Relationship claims are stored as structured data,
  not as graph edges (yet).

---

## Core concepts

### EvidenceItem

A single auditable piece of evidence — an experiment result, a dataset
observation, a backtest summary, a manually recorded note.

```python
from ml.evidence import EvidenceItem, EvidenceSourceType

ev = EvidenceItem.new(
    source_type     = EvidenceSourceType.EXPERIMENT,
    source_ref      = "exp_abc123",
    summary         = "MLP achieved 0.58 accuracy on AAPL fold 2 of 3",
    structured_data = {"accuracy": 0.58, "fold": 2, "symbol": "AAPL"},
    citation        = "",   # optional URL or reference string
)
```

`source_type` values: `dataset`, `backtest`, `experiment`, `document`,
`source_registry`, `manual`.

To signal that an evidence item *contradicts* the claim it is attached to,
set `structured_data["conflicts"] = True`.  The evaluator counts these as
conflicting evidence.

### Claim

A structured assertion with a `(subject, predicate, object)` triple, a
human-set confidence score, uncertainty notes, counterpoints, and attached
evidence IDs.

```python
from ml.evidence import Claim, ClaimType

claim = Claim.new(
    claim_type        = ClaimType.FEATURE_USEFULNESS,
    subject           = "feature:rsi_14",
    predicate         = "improves_accuracy_on",
    obj               = "model:mlp_dataset:abc123",
    confidence        = 0.6,
    uncertainty_notes = "Only tested on AAPL 2y window",
    counterpoints     = ["Feature may be noisy on low-vol regimes"],
)
```

**Confidence is human-set.** Do not feed an LLM output directly into this
field.  It should reflect your considered judgement after reviewing evidence.

### ClaimStatus lifecycle

```
PROPOSED → SUPPORTED   (human reviews evidence and is satisfied)
PROPOSED → WEAK        (evaluator flags insufficient evidence)
PROPOSED → REJECTED    (evidence contradicts the claim)
* → ARCHIVED           (claim is no longer relevant)
```

Status is **never changed automatically**.

---

## Using the store

```python
from ml.evidence import EvidenceItem, EvidenceSourceType, Claim, ClaimType
from ml.evidence import ClaimStore

store = ClaimStore()   # persists to data/evidence/claims.json

# 1. Record evidence
ev = EvidenceItem.new(
    source_type     = EvidenceSourceType.BACKTEST,
    source_ref      = "wf_AAPL_2026-03-27",
    summary         = "Baseline model beat buy-and-hold on 2/3 folds",
    structured_data = {"beat_bm_folds": 2, "total_folds": 3},
)
store.add_evidence(ev)

# 2. Create a claim
claim = Claim.new(
    claim_type = ClaimType.PERFORMANCE,
    subject    = "model:baseline_on_AAPL",
    predicate  = "meets_performance_bar",
    obj        = "metric:beat_bm_majority",
    confidence = 0.55,
)
store.add_claim(claim)

# 3. Link evidence to claim
store.link_evidence(claim.claim_id, ev.evidence_id)

# 4. Evaluate
from ml.evidence import evaluate_claim
items = store.get_evidence_for_claim(claim.claim_id)
ev_result = evaluate_claim(claim, items)
print(ev_result.support_level)    # "weak" / "moderate" / "strong" / "none"
print(ev_result.recommendation)   # "propose" / "support" / "flag_weak" / "reject"
print(ev_result.notes)            # list of human-readable notes

# 5. Update status manually after review
store.update_status(claim.claim_id, ClaimStatus.SUPPORTED,
                    notes="Reviewed 2026-03-27; evidence deemed sufficient")
```

---

## Claim templates

Use the template helpers to create common claim types with consistent naming:

```python
from ml.evidence.templates import (
    source_usefulness_claim,
    feature_usefulness_claim,
    relationship_claim,
    performance_claim,
)

# "FRED GDP series is useful for macro regime detection"
c1 = source_usefulness_claim(
    source_id  = "FRED_GDP",
    useful_for = "macro_regime",
    notes      = "GDP growth direction aligns with broad equity trend",
    confidence = 0.55,
)

# "RSI-14 improves MLP accuracy on dataset abc123"
c2 = feature_usefulness_claim(
    feature_name    = "rsi_14",
    model_type      = "mlp",
    dataset_version = "abc123",
    improvement_note = "accuracy +0.03 over 3-fold walk-forward on AAPL",
    confidence      = 0.6,
)

# "AAPL and QQQ are correlated (60d rolling r > 0.85)"
c3 = relationship_claim(
    asset_a    = "AAPL",
    asset_b    = "QQQ",
    criterion  = "rolling_60d_correlation > 0.85",
    confidence = 0.7,
)
```

---

## Evaluation logic

`evaluate_claim(claim, evidence_items)` returns a `ClaimEvaluation`:

| Support level | Condition |
|---|---|
| `none` | 0 evidence items, or majority conflict |
| `weak` | 1 item, or 2+ items from only 1 source type |
| `moderate` | 2+ items from 2+ distinct source types |
| `strong` | 4+ items from 3+ distinct source types, 0 conflicts |

The `support_score` (0.0–1.0) is indicative only — do not threshold it for
hard decisions.

The `recommendation` is one of:

| Recommendation | Meaning |
|---|---|
| `propose` | Keep as PROPOSED; not enough evidence yet |
| `support` | Evidence sufficient — consider SUPPORTED after human review |
| `flag_weak` | Some evidence but insufficient; add more before deciding |
| `reject` | Majority evidence conflicts; consider REJECTED |

**The evaluation never changes the claim's status.**  It only informs you.

---

## Query helpers

```python
# All active (non-rejected, non-archived) claims
active = store.list_claims(active_only=True)

# All proposed feature-usefulness claims
proposed = store.list_claims(
    status     = ClaimStatus.PROPOSED,
    claim_type = ClaimType.FEATURE_USEFULNESS,
)

# All claims about a specific feature
rsi_claims = store.claims_by_subject("feature:rsi_14")

# All claims that target a specific model
model_claims = store.claims_by_object("model:mlp_on_AAPL")

# Summary statistics
print(store.summary_stats())
# {'n_claims': 12, 'n_evidence': 31, 'by_status': {...}, 'by_type': {...}}
```

---

## Pipeline bridge

`ml/evidence/bridge.py` converts empirical pipeline outputs into evidence items
and claims automatically.  Use it immediately after a comparison or walk-forward
run so results flow into the evidence layer without manual bookkeeping.

### After a walk-forward comparison

```python
from ml.evidence import ClaimStore
from ml.evidence.bridge import bridge_wf_comparison
from ml.comparison.runner import run_comparison

store  = ClaimStore()
result = run_comparison("AAPL", df, walk_forward=True)

# Generates per-fold EvidenceItems, aggregate EvidenceItem, and
# performance Claims — all PROPOSED, nothing auto-promoted.
saved = bridge_wf_comparison(result, store, wf_results=result.wf_results)
print(saved.evidence_ids)   # list of new evidence IDs
print(saved.claim_ids)      # list of new claim IDs
```

### After a single-split comparison

```python
from ml.evidence.bridge import bridge_comparison

result = run_comparison("SPY", df, walk_forward=False)
saved  = bridge_comparison(result, store, pipeline_results=result.pipeline_results)
```

### Manual bridging (fine-grained control)

```python
from ml.evidence.bridge import (
    evidence_from_wf_result,
    claims_from_wf_comparison,
    save_evidence_bundle,
)

# Collect evidence from each model's WalkForwardResult
items = []
for model_key, wf in wf_results.items():
    items.extend(evidence_from_wf_result(wf))

# Generate claim bundles from the comparison result
bundles = claims_from_wf_comparison(wf_comparison_result)

# Persist everything and link evidence to claims
saved = save_evidence_bundle(store, items, bundles)
```

### Claim confidence derivation

Bridge-generated confidence values come from aggregate fold statistics, **never
from LLM output or narrative text**:

| Condition | Confidence |
|---|---|
| `overall_recommended=True` | 0.70 |
| `mean_accuracy >= 0.52` and majority folds beat benchmark | 0.58 |
| Otherwise | 0.42 |

Winner claims receive a fixed 0.55 regardless of whether promotion was granted.

---

## Persistence

All data is stored in `data/evidence/claims.json` (override with
`FINBRAIN_CLAIMS_PATH` environment variable).  The format is a plain JSON
file with `version`, `updated_at`, `evidence`, and `claims` arrays.

The file is **not** ignored by git.  Commit it to track the research history
alongside the experiment registry.

---

## Naming conventions

| Namespace | Format | Example |
|---|---|---|
| Data source | `source:<id>` | `source:FRED_GDP` |
| Feature | `feature:<name>` | `feature:rsi_14` |
| Asset | `asset:<ticker>` | `asset:AAPL` |
| Model | `model:<type>_on_<symbol>` | `model:mlp_on_AAPL` |
| Domain | `domain:<description>` | `domain:macro_regime` |
| Dataset | `dataset:<version>` | `dataset:abc123` |

---

## How this supports future autonomy safely

The evidence layer is designed to be the **anchor** for any future autonomous
decision-making:

1. **Sources** discovered by the API hunter can be recorded as evidence items
   before any source is promoted.
2. **Features** identified by the self-improve loop can have
   `feature_usefulness_claim`s created before being activated.
3. **Graph edges** proposed by the reasoning layer can be represented as
   `relationship_claim`s with evidence, rather than being inserted directly.
4. **Every autonomous action** that creates or modifies a claim must record the
   evidence it is based on — no claim without a source reference.

Nothing in this layer calls external services, mutates model weights, or
triggers downstream actions.  It is a structured memo system, not an agent.
