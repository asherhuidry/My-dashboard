-- FinBrain — TimescaleDB Schema
-- Run against a PostgreSQL instance with the TimescaleDB extension enabled
-- (available via Supabase under Database → Extensions → timescaledb).
-- All time columns use TIMESTAMPTZ and are the partitioning dimension.

-- ─────────────────────────────────────────────
-- Enable TimescaleDB
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─────────────────────────────────────────────
-- prices
-- Daily / intraday OHLCV for all asset classes.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    time            TIMESTAMPTZ     NOT NULL,
    asset           TEXT            NOT NULL,
    asset_class     TEXT            NOT NULL
                                    CHECK (asset_class IN ('equity', 'crypto', 'forex', 'commodity')),
    source          TEXT            NOT NULL DEFAULT 'yfinance',
    open            NUMERIC(18, 6)  NOT NULL,
    high            NUMERIC(18, 6)  NOT NULL,
    low             NUMERIC(18, 6)  NOT NULL,
    close           NUMERIC(18, 6)  NOT NULL,
    volume          NUMERIC(24, 2)  NOT NULL DEFAULT 0,
    adj_close       NUMERIC(18, 6),
    PRIMARY KEY (time, asset, source)
);

-- Convert to hypertable partitioned by time, 7-day chunks
SELECT create_hypertable(
    'prices', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_prices_asset_time
    ON prices (asset, time DESC);

CREATE INDEX IF NOT EXISTS idx_prices_asset_class_time
    ON prices (asset_class, time DESC);

-- Continuous aggregate: daily OHLCV (materialized every hour)
CREATE MATERIALIZED VIEW IF NOT EXISTS prices_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)  AS bucket,
    asset,
    asset_class,
    FIRST(open, time)           AS open,
    MAX(high)                   AS high,
    MIN(low)                    AS low,
    LAST(close, time)           AS close,
    SUM(volume)                 AS volume
FROM prices
GROUP BY bucket, asset, asset_class
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'prices_daily',
    start_offset  => INTERVAL '3 days',
    end_offset    => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ─────────────────────────────────────────────
-- volume
-- Exchange-level buy/sell volume breakdown.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS volume (
    time            TIMESTAMPTZ     NOT NULL,
    asset           TEXT            NOT NULL,
    exchange        TEXT            NOT NULL DEFAULT 'aggregate',
    buy_vol         NUMERIC(24, 2)  NOT NULL DEFAULT 0,
    sell_vol        NUMERIC(24, 2)  NOT NULL DEFAULT 0,
    total_vol       NUMERIC(24, 2)  GENERATED ALWAYS AS (buy_vol + sell_vol) STORED,
    PRIMARY KEY (time, asset, exchange)
);

SELECT create_hypertable(
    'volume', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_volume_asset_time
    ON volume (asset, time DESC);

-- ─────────────────────────────────────────────
-- macro_events
-- FRED / central bank macro indicator releases.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS macro_events (
    time            TIMESTAMPTZ     NOT NULL,
    indicator       TEXT            NOT NULL,
    value           NUMERIC(18, 6)  NOT NULL,
    prior_value     NUMERIC(18, 6),
    revision        BOOLEAN         NOT NULL DEFAULT FALSE,
    source          TEXT            NOT NULL DEFAULT 'fred',
    unit            TEXT,
    frequency       TEXT,
    PRIMARY KEY (time, indicator, source)
);

SELECT create_hypertable(
    'macro_events', 'time',
    chunk_time_interval => INTERVAL '90 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_macro_events_indicator_time
    ON macro_events (indicator, time DESC);

-- ─────────────────────────────────────────────
-- Compression policies (compress chunks older than 30 days)
-- ─────────────────────────────────────────────
ALTER TABLE prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset, asset_class'
);
SELECT add_compression_policy('prices',   INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('volume',   INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('macro_events', INTERVAL '90 days', if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- Retention policies (keep 5 years of raw data)
-- ─────────────────────────────────────────────
SELECT add_retention_policy('prices',       INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('volume',       INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('macro_events', INTERVAL '10 years', if_not_exists => TRUE);
