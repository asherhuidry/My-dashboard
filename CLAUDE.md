# CLAUDE.md

## Identity

This project is a **governed market graph + research engine**.

It is **not**:
- a generic finance dashboard
- a vague self-evolving AI app
- a collection of flashy but weak features

Its purpose is to:
- find and validate useful sources
- ingest market/macro knowledge
- discover meaningful relationships
- persist them
- materialize a living market graph
- track how graph structure changes over time
- prepare for future graph-ML / neural representations

---

## North Star

Build a system that becomes more useful over time because it accumulates:

- better sources
- better nodes
- better edges
- better structural memory
- better provenance
- better graph intelligence

---

## Priorities

Optimize for, in order:

1. source governance / source expansion
2. automated ingestion of high-value structured data
3. meaningful edge discovery
4. persistence of graph-relevant state
5. graph materialization + visibility
6. temporal graph memory
7. provenance / trust
8. future graph-ML readiness
9. only then broader UI polish

Every good task should improve one or more of:

- node coverage
- edge discovery
- edge persistence
- automation
- graph visibility
- structural memory
- provenance
- evidence-backed state
- ML readiness

---

## What To Avoid

Avoid:
- fake autonomy
- broad speculative architecture
- decorative UI
- unsupported intelligence claims
- big redesigns without leverage
- many weak features instead of one strong one
- pages/features not backed by real system state

If something is not real, useful, and fed by substantive state:
**hide it, simplify it, or defer it.**

---

## Default Work Style

For each task:

1. inspect current code/state first
2. choose the **single highest-leverage bounded task**
3. implement only that task
4. add/update targeted tests
5. verify concretely
6. commit and push
7. report:
- what you chose
- why it was the best next move
- what changed
- what was verified
- the next best step

One strong upgrade > several partial ones.

---

## Tech Stack

- Python 3.11+ with full type hints everywhere
- FastAPI for the backend API
- React + D3.js + react-force-graph-2d for the visualization dashboard
- Supabase (relational + auth + evolution_log)
- TimescaleDB (time series via Supabase extension)
- Qdrant Cloud free tier (vector embeddings)
- Neo4j Aura free tier (relationship graph)
- GitHub Actions for all scheduled automation
- PyTorch + scikit-learn for ML
- yfinance, fredapi, alpha_vantage for data

## How to Run

- Start API: `uvicorn api.main:app --reload`
- Start dashboard: `cd dashboard && npm run dev`
- Run tests: `python -m pytest tests/ -v`
- Run ingestion: `python data/ingest/run.py`
- Run graph materializer: `python data/agents/graph_materializer.py`
- Run scout pipeline: `python data/scout/auto_runner.py`

## Environment Variables (never hardcode)

ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY, QDRANT_URL, QDRANT_API_KEY,
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ALPHA_VANTAGE_KEY, FRED_API_KEY

---

## Product Guidance

The UI should feel like a **market graph workbench**, centered on:

- overview / structural state
- sources
- discoveries
- graph
- structural intelligence

Not a cluttered finance dashboard.

Prefer fewer honest surfaces over many premature ones.

---

## Discovery Guidance

Better discoveries means more than more correlations.

Prefer:
- sensitivity / exposure edges
- regime-shift / edge-change behavior
- macro-to-asset structure
- bridge factors
- hubs / centrality
- anomaly detection
- provenance-backed discoveries
- confidence/evidence scoring

Quality of graph meaning > quantity of edges.

---

## Provenance / Trust

The system should increasingly answer:

- what changed?
- why did it change?
- what source supports this?
- what run created this?
- how confident are we?

Preserve:
- run_id
- timestamps
- source provenance
- structured outputs suitable for comparison over time

---

## Automation Guidance

Prefer bounded, trustworthy automation:

- trusted-domain source scouting
- scheduled ingestion
- scheduled discovery
- scheduled graph materialization
- scheduled structural snapshots
- anomaly checks over persisted graph memory

Avoid uncontrolled broad scraping.

---

## Coding Rules

- Always use Python type hints on every function
- Always write a docstring on every function
- Always add logging so the evolution log can track what ran
- Always write tests for new functions before considering them done
- Never hardcode any API keys, URLs, or credentials
- Every agent must write its results to the evolution log in Supabase
- Keep every agent modular — one file per agent, no monoliths
- All scheduled jobs run through GitHub Actions cron
- Use deferred imports in modules that depend on external services (Neo4j, Supabase) so tests can mock at the source path

## Agent Rules

- Agents never delete data — they quarantine or archive
- Agents always log what they did, what they found, and what they changed
- Agents always measure before and after any change
- Agents file tasks to the roadmap table, they do not self-execute destructive operations without a backtest gate

---

## If Unsure

Choose the highest-leverage bounded task that makes the market graph more:

- real
- meaningful
- persistent
- queryable
- explainable
- trustworthy
- automatable
- learnable later

When in doubt:
**prefer better graph structure and graph memory over broader product surface.**
