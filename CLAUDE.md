# FinBrain — Autonomous Financial Intelligence System

## What this project is
A self-evolving financial AI that autonomously ingests data, trains ML models,
discovers patterns across all asset classes, and improves itself continuously
without human intervention. Every layer evolves on its own schedule.

## Tech stack
- Python 3.11+ with full type hints everywhere
- FastAPI for the backend API
- React + D3.js for the visualization dashboard
- Supabase (relational + auth)
- TimescaleDB (time series via Supabase extension)
- Qdrant Cloud free tier (vector embeddings)
- Neo4j Aura free tier (relationship graph)
- GitHub Actions for all scheduled automation
- PyTorch + scikit-learn for ML
- yfinance, ccxt, fredapi, alpha_vantage for data

## Project structure
/data
  ingest/         — API connectors, scrapers, schedulers
  agents/         — api_hunter.py, context_agent.py, noise_filter.py
/db
  timescale/      — schemas, connectors, migrations
  qdrant/         — collections, embedding models, reindex logic
  neo4j/          — graph schemas, relationship discovery
  supabase/       — relational schemas, signal tables, audit log
  agents/         — db_evolution_agent.py
/ml
  patterns/       — LSTM, transformer, ensemble models
  backtest/       — walk-forward engine, stress scenarios
  reasoning/      — causal chain inference, event linking
  improve/        — self_improve_loop.py, model_tournament.py
  agents/         — capability_expansion_agent.py
/architect
  — master_architect.py
  — gap_analysis.py
  — roadmap_writer.py
  — impact_evaluator.py
  — system_health.py
/dashboard
  — React + D3.js knowledge network visualization
  — signals dashboard
  — evolution log viewer
  — reasoning explainer
/api
  — FastAPI routes serving the dashboard
/skills
  — reusable utilities used by all agents
/tests
  — tests for every module

## How to run
- Start API: uvicorn api.main:app --reload
- Start dashboard: cd dashboard && npm run dev
- Run ingestion manually: python data/ingest/run.py
- Run master architect manually: python architect/master_architect.py

## Environment variables (never hardcode these)
ANTHROPIC_API_KEY
SUPABASE_URL
SUPABASE_KEY
QDRANT_URL
QDRANT_API_KEY
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
ALPHA_VANTAGE_KEY
FRED_API_KEY

## Coding rules (always follow these)
- Always use Python type hints on every function
- Always write a docstring on every function explaining what it does
- Always add logging so the evolution log can track what ran
- Always write tests for new functions before considering them done
- Never hardcode any API keys, URLs, or credentials
- Every agent must write its results to the evolution log in Supabase
- Every model must be backtested before it gets used in production
- Every improvement must have a rollback plan
- Keep every agent modular — one file per agent, no monoliths
- All scheduled jobs run through GitHub Actions cron

## Agent rules
- Agents never delete data — they quarantine or archive
- Agents always log what they did, what they found, and what they changed
- Agents always measure before and after any change
- Agents file tasks to the roadmap table in Supabase, they do not self-execute
  destructive operations without a backtest gate
- Master architect has read access to everything but only writes to the roadmap
  and evolution log — it does not directly modify models or schemas

## Evolution schedule
- Hourly: data ingestion, live accuracy monitoring, signal generation
- Daily: master architect reads system state, gap analysis, impact evaluation
- Weekly: API hunter, DB evolution audit, self-improve loop
- Bi-weekly: model tournament, capability expansion agent, research review

## What the master architect prioritizes
1. Accuracy — if any asset class signal accuracy drops below 55%, it fires
2. Coverage — if a new asset class has less than 30 days of data, it flags it
3. Database health — if any query exceeds 500ms, it files a DB task
4. Model freshness — if any model has not been retrained in 14 days, it flags it
5. New capabilities — reviews arXiv ML papers weekly for applicable techniques

## Phase 1 build order
Build in exactly this order:
1. Create the complete folder structure for all layers
2. Set up Supabase — create all tables: evolution_log, roadmap, signals,
   model_registry, api_sources, system_health, agent_runs, quarantine
3. Set up TimescaleDB — hypertables for prices, volume, macro_events
4. Set up Qdrant — 5 collections with 768-dim vectors
5. Set up Neo4j — node types and initial relationship schema
6. Build core data connectors — yfinance, CoinGecko, FRED, Alpha Vantage
7. Build the noise filter
8. Run first data ingestion — 2 years history for top 50 US stocks,
   top 20 crypto, major forex pairs, key commodities, 10 FRED indicators
9. Build feature engineering pipeline (80+ features)
10. Train first LSTM model, log to model_registry
11. Run first backtest, log results
12. Build FastAPI backend with all routes
13. Build React dashboard with D3.js knowledge network
14. Set up all 4 GitHub Actions workflows
15. Build master architect agent last — it needs data to read

Show me the full plan for Phase 1 before touching any files.
After I approve, execute each step one at a time, run tests
after each step, and commit to GitHub after each step passes.
Do not move to the next step until the current one works and is committed.
