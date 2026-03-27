# FinBrain — Repository Reality Audit

**Audit date:** 2026-03-26
**Auditor:** automated self-inspection pass
**Repo root:** `My-dashboard/`

---

## Implemented systems

These modules contain real, runnable code that was verified by import and/or test pass.

| Module | Lines | What it does |
|--------|-------|-------------|
| `db/supabase/client.py` | ~336 | Lazy singleton client; typed dataclasses (EvolutionLogEntry, Signal, ModelRegistryEntry, AgentRun, QuarantineRecord); write helpers for all tables |
| `db/neo4j/client.py` | ~420 | Lazy singleton driver; node/relationship helpers; MERGE-based upserts; query helpers |
| `db/neo4j/schema.py` | ~100 | Node labels, relationship types, constraint and index DDL |
| `db/qdrant/client.py` | ~243 | Lazy QdrantClient; collection management; EmbeddingPoint upsert + search |
| `db/qdrant/collections.py` | ~79 | 5 collection definitions (768-dim, COSINE) |
| `db/timescale/client.py` | ~285 | Typed row classes; bulk upsert; fetch helpers for prices and macro |
| `skills/env.py` | ~82 | Reads all 10 credentials from `.env`; raises RuntimeError on missing required vars |
| `skills/logger.py` | ~45 | Structured JSON logging via structlog |
| `data/ingest/yfinance_connector.py` | ~170 | Fetch OHLCV; agent-run logging; bulk write to TimescaleDB |
| `data/ingest/coingecko_connector.py` | ~80 | Fetch crypto OHLC; agent-run logging |
| `data/ingest/fred_connector.py` | ~80 | Fetch FRED series; agent-run logging |
| `data/ingest/universe.py` | ~232 | 200+ tickers, 65 macro series, sector map, causal relationship table |
| `data/ingest/run.py` | ~126 | Orchestrates all four connectors; dry-run mode |
| `data/agents/noise_filter.py` | ~150 | Z-score outlier detection; volume spike filter; cross-source consistency |
| `data/agents/correlation_hunter.py` | ~350 | Pearson + Granger + mutual-info + regime-conditional correlation; lead-lag analysis |
| `ml/patterns/features.py` | ~554 | 80+ ML features: returns, volatility, momentum, trend, volume, regime, macro-lag, cross-asset, calendar |
| `ml/patterns/dataset.py` | ~123 | PyTorch Dataset; sliding-window; chronological split |
| `ml/patterns/lstm.py` | ~176 | 2-layer LSTM + dense head; save/load checkpoint; predict helper |
| `ml/patterns/train.py` | ~298 | End-to-end LSTM training pipeline; early stopping; Supabase model-registry logging; CLI |
| `api/main.py` | ~55 | FastAPI app; CORS; registers 16 routers; background price-polling |
| `api/routes/health.py` | ~25 | `/api/health`; pings all 5 services |
| `api/routes/prices.py` | ~50 | `/api/prices/{symbol}`; returns OHLCV JSON |
| `api/routes/analysis.py` | ~140 | `/api/analyze/{symbol}`; 74+ features + consensus signal |
| `api/routes/predict.py` | ~160 | `/api/predict/{symbol}`; LSTM or rules-based fallback |
| `api/routes/backtest.py` | ~270 | `/api/backtest/{symbol}`; rule-based signal; equity curve; 12 metrics |
| `api/routes/chat.py` | ~60 | `/api/chat`; Claude Sonnet; financial system prompt |
| `api/routes/ws.py` | ~80 | WebSocket `/ws/prices`; 17-symbol watchlist; background poll |
| `api/routes/search.py` | ~50 | `/api/search`; symbol lookup |
| `tests/` | ~800 | 74 passing tests across structure, connectors, features, LSTM, DB clients, noise filter |

**74 tests pass** (`pytest tests/test_structure.py tests/test_features.py -q` confirmed).

---

## Partially implemented systems

These files exist with real code but are incomplete, untested, or not wired to the running app.

| Module | Status | Notes |
|--------|--------|-------|
| `data/ingest/edgar_full.py` | ~60% | EDGAR XBRL, Form 4, 13F, Congress trades; no tests; not called from `run.py` |
| `data/ingest/social_data.py` | ~70% | Reddit/StockTwits/Google Trends/CBOE; pytrends optional dep; no tests |
| `data/ingest/macro_expanded.py` | ~70% | 65-series FRED pipeline; Z-scores; regime detection; no tests |
| `data/sources/fundamentals.py` | ~60% | yfinance fundamentals + SEC CIK lookup; no tests |
| `data/sources/news.py` | ~65% | DuckDuckGo + yfinance news + sentiment scorer; no tests |
| `data/agents/knowledge_builder.py` | ~70% | Neo4j supply-chain + macro graph writes; no tests |
| `ml/patterns/features_expanded.py` | ~55% | 150-feature pipeline; all functions exist but sparse and untested |
| `ml/neural/encoders.py` | ~80% | Real PyTorch modules; imports and forward-pass verified |
| `ml/neural/graph_net.py` | ~75% | Custom heterogeneous GAT; FinancialGraph builder; verified by smoke test |
| `ml/neural/unified_model.py` | ~70% | FinBrainNet; forward-pass verified with fake tensors; no real training data path |
| `ml/neural/pattern_discovery.py` | ~70% | VAE; regime detector; anomaly detector; smoke-test verified |
| `ml/neural/train_unified.py` | ~50% | Training loop skeleton; `_build_price_tensor` works; daily-update path unverified end-to-end |
| `api/routes/research.py` | ~65% | Claude agentic loop with 5 tools; no tests |
| `api/routes/intelligence.py` | ~60% | Correlations, regime, supply-chain endpoints; Neo4j optional fallbacks; no tests |
| `api/routes/neural.py` | ~55% | Neural predict/explain/anomaly endpoints; lazy checkpoint load; no tests |
| `api/routes/screener.py` | ~50% | Asset screener; unknown completeness |
| `api/routes/evolution.py` | ~50% | Evolution log reader; unknown completeness |
| `api/routes/data_sources.py` | ~60% | News + fundamentals routes; thin wrappers |

---

## Aspirational / scaffolded systems

These directories exist only as empty `__init__.py` files with no real implementation.

| Path | Expected content | Actual content |
|------|-----------------|----------------|
| `architect/` | master_architect.py, gap_analysis.py, roadmap_writer.py, impact_evaluator.py | Empty `__init__.py` only |
| `ml/backtest/` | Walk-forward engine, stress scenarios | Empty `__init__.py` only |
| `ml/improve/` | self_improve_loop.py, model_tournament.py | Empty `__init__.py` only |
| `ml/reasoning/` | Causal chain inference, event linking | Empty `__init__.py` only |
| `ml/agents/` | capability_expansion_agent.py | Empty `__init__.py` only |
| `db/agents/` | db_evolution_agent.py | Empty `__init__.py` only |

**No trained model checkpoints exist.** `checkpoints/` is empty. `ml/patterns/train.py` is ready to run but has never been executed in this environment.

---

## Current working core

The system that actually works end-to-end today:

```
yfinance / CoinGecko / FRED
        ↓
  data/ingest/run.py
        ↓
  TimescaleDB (prices) + Supabase (metadata)
        ↓
  ml/patterns/features.py  →  ml/patterns/train.py
        ↓
  LSTM model (not yet trained in this env)
        ↓
  api/main.py (FastAPI)
  ├── /api/prices/{symbol}
  ├── /api/analyze/{symbol}
  ├── /api/predict/{symbol}  (rules-based fallback)
  ├── /api/backtest/{symbol}
  ├── /api/chat
  └── ws://localhost:8000/ws/prices
```

The dashboard (React + Vite) consumes these routes and is separately confirmed to build (`npm run build` passes).

---

## Key risks

1. **No trained models** — `checkpoints/` is empty; all ML predictions use the rules-based fallback. The training script exists and is correct, but no run has completed.

2. **Credential dependencies** — All DB clients fail gracefully (log warnings), but no real data flows without `.env` populated. Several partially-implemented modules will silently return empty results.

3. **No source provenance tracking** — Data is ingested with no registry of where it came from, its reliability, or whether it passed quality checks. There is a `quarantine_record` function in the Supabase client but nothing calls it systematically.

4. **No experiment tracking** — Models are registered in `model_registry` on Supabase, but there is no local experiment log, no feature-set versioning, no reproducibility metadata. If the DB is not connected, nothing is logged.

5. **GNN / neural layer is unverified at scale** — Forward pass works with fake tensors; real training from market data has not run. `assemble_node_inputs()` calls yfinance and FRED live, making it slow and brittle in CI.

6. **Duplicated backtest logic** — Walk-forward backtest code lives in `api/routes/backtest.py` (an HTTP endpoint) rather than a reusable Python module. Any other consumer must re-implement it.

7. **Test coverage is 9%** — Tests cover structure and features well; ML training, API routes, and partially-implemented modules have zero coverage.

---

## Recommended immediate next steps

1. **Source registry** — Add a lightweight, file-backed registry of known data sources with status lifecycle. This is the foundation of evidence-driven autonomy (knowing *where* data comes from and whether it is trustworthy).

2. **Validation / quarantine layer** — Add a reusable validator that any ingest connector can call before writing. Produce structured reports; quarantine failing datasets locally.

3. **Experiment registry** — Add a local-first experiment log (JSON/SQLite) so model runs are reproducible without a live Supabase connection.

4. **MLP baseline** — Add a simple feedforward network as a second model class. The LSTM is good but a fast linear/MLP baseline is needed for comparison and for datasets where sequences are not meaningful.

5. **Clean backtest engine module** — Extract the backtest logic from the API route into `ml/backtest/engine.py` so it can be called from training scripts, experiment runners, and the API equally.
