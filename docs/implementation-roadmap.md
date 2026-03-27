# FinBrain — Implementation Roadmap

**Created:** 2026-03-26
**Based on:** repo-reality-audit.md

This roadmap prioritises correctness and immediate usefulness over ambitious autonomy.
Each phase produces a working, testable artefact before the next begins.

---

## Phase 1 — Foundation (now → stable baseline)

**Goal:** Make the existing core trustworthy and reproducible.

| Task | Module | Priority |
|------|--------|----------|
| Source registry with status lifecycle | `data/registry/source_registry.py` | **P0** |
| Data validation + quarantine layer | `data/validation/validator.py` | **P0** |
| Experiment registry (local-first) | `ml/registry/experiment_registry.py` | **P0** |
| MLP baseline model | `ml/patterns/mlp.py` | **P1** |
| Clean backtest engine module | `ml/backtest/engine.py` | **P1** |
| Run LSTM training to produce first checkpoint | `ml/patterns/train.py` | **P1** |
| Wire validation into all ingest connectors | `data/ingest/*.py` | **P2** |
| Raise test coverage to ≥ 30% | `tests/` | **P2** |

**Definition of done:** `pytest` passes; a model checkpoint exists; every ingest run produces a validation report; experiments are logged locally.

---

## Phase 2 — Evidence-driven autonomy

**Goal:** The system discovers, validates, and tracks new evidence without manual intervention, but with evaluation gates before any data or model is promoted.

| Task | Module | Notes |
|------|--------|-------|
| Source discovery agent (search known registries) | `data/agents/source_discovery.py` | Searches SEC EDGAR, FRED, World Bank; no uncontrolled crawling |
| Automated source sampling + validation | `data/registry/sampler.py` | Fetches a small sample from each `discovered` source and runs the validator |
| Source promotion gate | `data/registry/source_registry.py` | `discovered` → `sampled` → `validated` (human or threshold-based) → `approved` |
| Feature importance tracking | `ml/registry/experiment_registry.py` | Record SHAP / permutation importance per experiment |
| Walk-forward experiment runner | `ml/backtest/walk_forward.py` | Runs model on rolling out-of-sample windows; logs each fold as an experiment |
| Automated signal evaluation | `ml/backtest/signal_evaluator.py` | Hit rate, Kelly criterion, Sharpe by regime |
| Wire experiment registry to training scripts | `ml/patterns/train.py`, `ml/patterns/train_mlp.py` | Every run writes a local experiment JSON |

**Definition of done:** A daily GitHub Actions run ingests data, validates it, runs the models, logs experiments, and flags anything anomalous — all without human intervention. No model is promoted without meeting a Sharpe threshold.

---

## Phase 3 — Model and UI expansion

**Goal:** Richer models, richer UI, richer reasoning.

| Task | Module | Notes |
|------|--------|-------|
| Train FinBrainNet GNN on real data | `ml/neural/train_unified.py` | Requires Phase 1 checkpoint + validated features |
| Pattern discovery VAE training | `ml/neural/pattern_discovery.py` | After GNN is trained |
| Semantic embedding pipeline | Qdrant + news/filings | Index news, filings in Qdrant for Research page |
| Intelligence page — live correlations | `dashboard/src/pages/Intelligence.jsx` | Wire to `/api/intelligence/*` endpoints |
| Regime overlay in Analyzer | `dashboard/src/pages/Analyzer.jsx` | Show macro regime badge |
| Architect agent (read-only) | `architect/master_architect.py` | Reads system state; files roadmap tasks; does NOT auto-modify models |
| Model tournament | `ml/improve/model_tournament.py` | Weekly comparison of registered models; human promotes winner |

**Definition of done:** Three model types trained and compared (MLP, LSTM, GNN); semantic search works on news; architect agent produces weekly reports.

---

## Recommended sequencing

```
Week 1:  Phase 1 foundation  (source registry + validation + experiment registry + MLP + backtest engine)
Week 2:  Run training pipeline end-to-end; produce first real checkpoints; close test coverage gap
Week 3:  Phase 2 automation  (source discovery + walk-forward runner + signal evaluator)
Week 4:  Wire Phase 2 into GitHub Actions; validate daily pipeline runs cleanly
Month 2: Phase 3 (GNN training + UI + architect agent)
```

---

## Deferred work

The following items are explicitly deferred until the foundation phases are stable:

- **Uncontrolled web crawling** — source discovery will target known registries only
- **Master architect self-modification** — read-only reports only until Phase 2 is proven
- **Model auto-promotion** — always requires a validation gate; never fully autonomous in Phase 1–2
- **arXiv paper review agent** — deferred to Phase 3+
- **db_evolution_agent** — deferred until Supabase schema is stable
- **Capability expansion agent** — deferred to Phase 3+
- **FinBrainNet GNN training at scale** — deferred to Phase 3 (depends on validated data pipeline)
- **UI visualization of latent factors** — deferred to Phase 3

---

## Non-negotiable constraints

These must be true at every phase:

1. Every data source must pass validation before writing to any store.
2. Every model must have a backtest result before being registered as `promoted`.
3. Every experiment must be reproducible from its logged metadata.
4. All automation runs through GitHub Actions with explicit approval gates for destructive operations.
5. No API key or credential is ever hardcoded or logged.
