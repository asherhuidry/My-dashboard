-- FinBrain — Supabase Relational Schema
-- Run this against your Supabase project via the SQL editor or psql.
-- All tables use UUIDs as primary keys and timestamptz for all timestamps.

-- ─────────────────────────────────────────────
-- Enable extensions
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────
-- evolution_log
-- Every agent writes one row per run describing what it did.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evolution_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id        TEXT        NOT NULL,
    action          TEXT        NOT NULL,
    before_state    JSONB,
    after_state     JSONB,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evolution_log_agent_id   ON evolution_log (agent_id);
CREATE INDEX IF NOT EXISTS idx_evolution_log_created_at ON evolution_log (created_at DESC);

ALTER TABLE evolution_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON evolution_log
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- roadmap
-- Agents file tasks here; they never self-execute destructive ops.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS roadmap (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filed_by        TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    description     TEXT,
    priority        SMALLINT    NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    status          TEXT        NOT NULL DEFAULT 'open'
                                CHECK (status IN ('open', 'in_progress', 'blocked', 'done', 'cancelled')),
    backtest_gate   BOOLEAN     NOT NULL DEFAULT FALSE,
    backtest_result JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roadmap_status   ON roadmap (status);
CREATE INDEX IF NOT EXISTS idx_roadmap_priority ON roadmap (priority);

ALTER TABLE roadmap ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON roadmap
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- signals
-- Generated trading signals per asset.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset           TEXT        NOT NULL,
    asset_class     TEXT        NOT NULL
                                CHECK (asset_class IN ('equity', 'crypto', 'forex', 'commodity', 'macro')),
    direction       TEXT        NOT NULL CHECK (direction IN ('long', 'short', 'neutral')),
    confidence      NUMERIC(5,4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    model_id        UUID,
    features        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_asset       ON signals (asset);
CREATE INDEX IF NOT EXISTS idx_signals_asset_class ON signals (asset_class);
CREATE INDEX IF NOT EXISTS idx_signals_created_at  ON signals (created_at DESC);

ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON signals
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- model_registry
-- All trained models with versioning and accuracy tracking.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT        NOT NULL,
    version         TEXT        NOT NULL,
    model_type      TEXT        NOT NULL,
    asset_class     TEXT,
    accuracy        NUMERIC(5,4),
    val_loss        NUMERIC(10,6),
    train_samples   INTEGER,
    hyperparams     JSONB,
    artifact_path   TEXT,
    status          TEXT        NOT NULL DEFAULT 'staging'
                                CHECK (status IN ('staging', 'production', 'retired', 'failed')),
    last_trained_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry (status);
CREATE INDEX IF NOT EXISTS idx_model_registry_name   ON model_registry (name);

ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON model_registry
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- api_sources
-- Tracks all external data sources and their health.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT        NOT NULL UNIQUE,
    endpoint        TEXT,
    asset_classes   TEXT[]      NOT NULL DEFAULT '{}',
    last_fetched_at TIMESTAMPTZ,
    health_status   TEXT        NOT NULL DEFAULT 'unknown'
                                CHECK (health_status IN ('healthy', 'degraded', 'down', 'unknown')),
    error_count     INTEGER     NOT NULL DEFAULT 0,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE api_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON api_sources
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- system_health
-- Time-series of system-level metrics with threshold flags.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_health (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric          TEXT        NOT NULL,
    value           NUMERIC     NOT NULL,
    threshold       NUMERIC,
    flagged         BOOLEAN     NOT NULL DEFAULT FALSE,
    source          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_health_metric     ON system_health (metric);
CREATE INDEX IF NOT EXISTS idx_system_health_flagged    ON system_health (flagged) WHERE flagged = TRUE;
CREATE INDEX IF NOT EXISTS idx_system_health_created_at ON system_health (created_at DESC);

ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON system_health
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- agent_runs
-- One row per agent execution with timing and outcome.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name      TEXT        NOT NULL,
    trigger         TEXT        NOT NULL DEFAULT 'manual'
                                CHECK (trigger IN ('manual', 'scheduled', 'event')),
    status          TEXT        NOT NULL DEFAULT 'running'
                                CHECK (status IN ('running', 'completed', 'failed')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    duration_ms     INTEGER,
    result_summary  TEXT,
    error_message   TEXT,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_name ON agent_runs (agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status     ON agent_runs (status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at ON agent_runs (started_at DESC);

ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON agent_runs
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- quarantine
-- Agents never delete data — they quarantine it here.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quarantine (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_table    TEXT        NOT NULL,
    original_id       TEXT,
    data              JSONB       NOT NULL,
    reason            TEXT        NOT NULL,
    quarantined_by    TEXT        NOT NULL,
    reviewed          BOOLEAN     NOT NULL DEFAULT FALSE,
    reviewed_by       TEXT,
    reviewed_at       TIMESTAMPTZ,
    quarantined_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quarantine_original_table ON quarantine (original_table);
CREATE INDEX IF NOT EXISTS idx_quarantine_reviewed       ON quarantine (reviewed) WHERE reviewed = FALSE;

ALTER TABLE quarantine ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON quarantine
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- discoveries
-- Persisted correlation findings from the weekly correlation hunter.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS discoveries (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_a          TEXT        NOT NULL,
    series_b          TEXT        NOT NULL,
    lag_days          INTEGER     NOT NULL,
    pearson_r         NUMERIC(8,6) NOT NULL,
    granger_p         NUMERIC(8,6),
    mutual_info       NUMERIC(8,6),
    regime            TEXT        NOT NULL DEFAULT 'all',
    strength          TEXT        NOT NULL DEFAULT 'moderate'
                                  CHECK (strength IN ('strong', 'moderate', 'weak')),
    relationship_type TEXT        NOT NULL DEFAULT 'discovered',
    run_id            UUID        NOT NULL,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discoveries_series_a   ON discoveries (series_a);
CREATE INDEX IF NOT EXISTS idx_discoveries_series_b   ON discoveries (series_b);
CREATE INDEX IF NOT EXISTS idx_discoveries_strength   ON discoveries (strength);
CREATE INDEX IF NOT EXISTS idx_discoveries_run_id     ON discoveries (run_id);
CREATE INDEX IF NOT EXISTS idx_discoveries_pearson_r  ON discoveries (pearson_r DESC);
CREATE INDEX IF NOT EXISTS idx_discoveries_created_at ON discoveries (created_at DESC);

ALTER TABLE discoveries ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service role full access" ON discoveries
    USING (auth.role() = 'service_role');

-- ─────────────────────────────────────────────
-- updated_at trigger (applied to tables that have it)
-- ─────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_roadmap_updated_at
    BEFORE UPDATE ON roadmap
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_api_sources_updated_at
    BEFORE UPDATE ON api_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
