# Source Scout

`data/scout/` provides a bounded, local-first system for discovering,
evaluating, and proposing new financial data sources.

---

## What it is

A pipeline that converts raw candidate source descriptions into:

1. **Normalized SourceCandidates** — consistent, validated intake records
2. **Usefulness scores** — deterministic 0–1 quality signals
3. **Connector proposals** — lightweight specs describing how to build a connector
4. **Registry entries** — DISCOVERED-status records in the source registry
5. **Evidence/claim stubs** — optional PROPOSED claims for the evidence layer

---

## What it is NOT

- **Not a web crawler.**  No automatic URL discovery or scraping.
- **Not an auto-approver.**  Every candidate enters the registry as `DISCOVERED`.
  A human must advance the status to `SAMPLED → VALIDATED → APPROVED`.
- **Not a connector builder.**  Proposals describe what to build; they do not
  write connector code.
- **Not an LLM.**  All scoring and proposal logic is rule-based and deterministic.
- **Not a production ingestion path.**  Scout output feeds the registry and
  evidence layer; ingestion pipelines pull from `APPROVED` sources only.

---

## How candidate sources flow

```
Raw dict / CANDIDATE_CATALOG
        │
        ▼
normalize_source_candidate()
        │
        ▼ SourceCandidate
        ├──► score_source_candidate()  ──► float score
        │
        ├──► propose_connector_spec()  ──► ConnectorProposal (spec only)
        │
        ├──► register_candidate_source() ──► SourceRegistry (DISCOVERED)
        │
        └──► evidence_from_candidate()  ──► EvidenceItem
             source_claim_from_candidate() ──► Claim (PROPOSED)
```

---

## Eligibility and status rules

| Stage | Status | Who advances it |
|---|---|---|
| Intake | DISCOVERED | Source scout |
| First fetch | SAMPLED | Connector dev / ingest test |
| Quality check | VALIDATED | Validator / noise_filter |
| In active use | APPROVED | Human sign-off |
| Failing | QUARANTINED | Monitoring / human |
| Not useful | REJECTED | Human decision |

The scout only ever writes `DISCOVERED`.  It cannot skip or reverse-advance any status.

---

## Scoring factors

| Factor | Weight | Signal |
|---|---|---|
| `official_source` | 0.20 | Government, central bank, exchange, regulator |
| `free_tier` | 0.15 | No-cost access available |
| `no_auth` | 0.08 | No API key required |
| `cadence_clarity` | 0.10 | Update cadence is known (not "unknown") |
| `schema_clarity` | 0.15 | Documentation quality 1–5 (normalized) |
| `financial_relevance` | 0.20 | Category + high-value data types |
| `broad_utility` | 0.12 | Multiple asset classes or data types |

Scores are first-pass triage signals — they do not replace empirical validation.

---

## Connector proposal priority

| Score | Priority |
|---|---|
| ≥ 0.65 | high |
| 0.40 – 0.64 | medium |
| < 0.40 | low |

---

## Usage

### Normalize and score a single candidate

```python
from data.scout import normalize_source_candidate, score_source_candidate

candidate = normalize_source_candidate({
    "name":            "ECB Statistical Data Warehouse",
    "url":             "https://sdw-wsrest.ecb.europa.eu/service",
    "category":        "macro",
    "official_source": True,
    "free_tier":       True,
    "auth_required":   False,
    "update_cadence":  "daily",
    "asset_types":     ["macro"],
    "data_types":      ["yields", "inflation", "exchange_rates"],
    "schema_clarity":  4,
})

score = score_source_candidate(candidate)
print(candidate.source_id)  # "ecb_statistical_data_warehouse"
print(score)                # e.g. 0.8350
```

### Generate a connector proposal

```python
from data.scout import propose_connector_spec

spec = propose_connector_spec(candidate, score)
print(spec.priority)                     # "high"
print(spec.expected_payload_shape)       # ["date", "maturity", "yield_pct", ...]
print(spec.refresh_frequency_suggestion) # "Daily cron after market close..."
print(spec.implementation_notes)         # "Score: 0.84 (high priority)..."
```

### Register in the source registry

```python
from data.registry.source_registry import SourceRegistry
from data.scout import register_candidate_source

registry = SourceRegistry()
result   = register_candidate_source(candidate, registry, score=score)
print(result.action)        # "created"
print(result.record.status) # SourceStatus.DISCOVERED
```

### Run the full built-in catalog

```python
from data.scout import (
    CANDIDATE_CATALOG,
    normalize_source_candidate,
    score_source_candidate,
    propose_connector_spec,
    register_catalog,
)
from data.registry.source_registry import SourceRegistry

registry   = SourceRegistry()
candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
scores     = {c.source_id: score_source_candidate(c) for c in candidates}
results    = register_catalog(candidates, registry, scores=scores)

for r in results:
    print(r.source_id, r.action, scores[r.source_id])
```

### Create evidence and claim stubs (optional)

```python
from data.scout import evidence_from_candidate, source_claim_from_candidate
from ml.evidence.store import ClaimStore

store = ClaimStore()
ev    = evidence_from_candidate(candidate, score)
claim = source_claim_from_candidate(candidate, score)

store.add_evidence(ev)
store.add_claim(claim)
store.link_evidence(claim.claim_id, ev.evidence_id)
# Claim is PROPOSED — review and promote manually after validation
```

---

## Built-in candidate catalog

`CANDIDATE_CATALOG` contains 8 high-signal, no-cost public financial data
sources not yet in the seed registry:

| Source | Category | Official | Auth |
|---|---|---|---|
| ECB Statistical Data Warehouse | macro | Yes | No |
| US Treasury Yield Curve | macro | Yes | No |
| OECD Data API | macro | Yes | No |
| SEC EDGAR XBRL | fundamental | Yes | No |
| Yahoo Finance RSS | news | No | No |
| CBOE VIX Historical Data | alternative | Yes | No |
| US BLS Employment Statistics | macro | Yes | No |
| Open Exchange Rates | forex | No | Yes (free tier) |

---

## How connector proposals are used

A `ConnectorProposal` is a specification document, not executable code.  After
a source is promoted to `SAMPLED`, the proposal's fields guide implementation:

- `acquisition_method` → which pattern to follow (`yfinance_connector.py`,
  `fred_connector.py`, etc.)
- `expected_payload_shape` → the fields the connector should produce
- `refresh_frequency_suggestion` → the GitHub Actions cron schedule
- `validation_strategy_hint` → the checks to add to `validator.py`
- `auth_notes` → which environment variable to document

A future automation pass could consume these proposals to scaffold connector
files; for now they are human-readable specs.
