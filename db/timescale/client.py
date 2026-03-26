"""TimescaleDB client for FinBrain.

Wraps psycopg2 with typed dataclasses and bulk-insert helpers for
the prices, volume, and macro_events hypertables.

Connection is pooled via a module-level connection that reconnects
on failure. All writes use COPY-style executemany for throughput.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection, cursor as PgCursor

from skills.env import get_supabase_url
from skills.logger import get_logger

logger = get_logger(__name__)

_conn: PgConnection | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────────────────────

def _build_dsn() -> str:
    """Build a psycopg2 DSN from the Supabase connection string.

    Supabase exposes TimescaleDB via a direct Postgres connection on port 5432.
    The SUPABASE_URL env var is expected to be the full postgres:// DSN
    (available under Project Settings → Database → Connection string).

    Returns:
        A psycopg2-compatible DSN string.
    """
    url = get_supabase_url()
    # Accept both https:// (REST) and postgres:// (direct) forms
    if url.startswith("https://"):
        # Derive the postgres DSN from the Supabase REST URL:
        # https://xyz.supabase.co  →  postgresql://postgres:pass@db.xyz.supabase.co:5432/postgres
        # In practice users should set TIMESCALE_DSN explicitly.
        raise RuntimeError(
            "SUPABASE_URL looks like a REST URL. "
            "Set TIMESCALE_DSN=postgresql://... for the direct Postgres connection."
        )
    return url


def get_connection() -> PgConnection:
    """Return a cached psycopg2 connection, reconnecting if closed.

    Returns:
        An open psycopg2 connection to TimescaleDB.
    """
    global _conn
    if _conn is None or _conn.closed:
        import os
        dsn = os.getenv("TIMESCALE_DSN") or _build_dsn()
        _conn = psycopg2.connect(dsn)
        _conn.autocommit = False
        logger.info("timescale_connected")
    return _conn


@contextlib.contextmanager
def transaction() -> Generator[PgCursor, None, None]:
    """Context manager that yields a cursor inside a transaction.

    Commits on clean exit, rolls back on any exception.

    Yields:
        An open psycopg2 cursor.
    """
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# Typed dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceRow:
    """One OHLCV row destined for the prices hypertable."""
    time: datetime
    asset: str
    asset_class: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    adj_close: float | None = None
    source: str = "yfinance"

    def __post_init__(self) -> None:
        """Ensure time is timezone-aware UTC.

        Raises:
            ValueError: If close price is negative (basic sanity check).
        """
        if self.time.tzinfo is None:
            self.time = self.time.replace(tzinfo=timezone.utc)
        if self.close < 0:
            raise ValueError(f"Negative close price for {self.asset}: {self.close}")


@dataclass
class VolumeRow:
    """One buy/sell volume row for the volume hypertable."""
    time: datetime
    asset: str
    buy_vol: float
    sell_vol: float
    exchange: str = "aggregate"

    def __post_init__(self) -> None:
        """Ensure time is timezone-aware UTC."""
        if self.time.tzinfo is None:
            self.time = self.time.replace(tzinfo=timezone.utc)


@dataclass
class MacroEventRow:
    """One macro indicator release for the macro_events hypertable."""
    time: datetime
    indicator: str
    value: float
    prior_value: float | None = None
    revision: bool = False
    source: str = "fred"
    unit: str | None = None
    frequency: str | None = None

    def __post_init__(self) -> None:
        """Ensure time is timezone-aware UTC."""
        if self.time.tzinfo is None:
            self.time = self.time.replace(tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

_PRICES_INSERT = """
    INSERT INTO prices (time, asset, asset_class, source, open, high, low, close, volume, adj_close)
    VALUES (%(time)s, %(asset)s, %(asset_class)s, %(source)s,
            %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(adj_close)s)
    ON CONFLICT (time, asset, source) DO UPDATE SET
        open      = EXCLUDED.open,
        high      = EXCLUDED.high,
        low       = EXCLUDED.low,
        close     = EXCLUDED.close,
        volume    = EXCLUDED.volume,
        adj_close = EXCLUDED.adj_close;
"""

_VOLUME_INSERT = """
    INSERT INTO volume (time, asset, exchange, buy_vol, sell_vol)
    VALUES (%(time)s, %(asset)s, %(exchange)s, %(buy_vol)s, %(sell_vol)s)
    ON CONFLICT (time, asset, exchange) DO UPDATE SET
        buy_vol  = EXCLUDED.buy_vol,
        sell_vol = EXCLUDED.sell_vol;
"""

_MACRO_INSERT = """
    INSERT INTO macro_events (time, indicator, value, prior_value, revision, source, unit, frequency)
    VALUES (%(time)s, %(indicator)s, %(value)s, %(prior_value)s,
            %(revision)s, %(source)s, %(unit)s, %(frequency)s)
    ON CONFLICT (time, indicator, source) DO UPDATE SET
        value       = EXCLUDED.value,
        prior_value = EXCLUDED.prior_value,
        revision    = EXCLUDED.revision;
"""


def bulk_insert_prices(rows: list[PriceRow]) -> int:
    """Bulk-insert OHLCV rows into the prices hypertable.

    Uses ON CONFLICT DO UPDATE so re-running ingestion is idempotent.

    Args:
        rows: List of PriceRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0
    params = [
        {
            "time": r.time, "asset": r.asset, "asset_class": r.asset_class,
            "source": r.source, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
            "adj_close": r.adj_close,
        }
        for r in rows
    ]
    with transaction() as cur:
        psycopg2.extras.execute_batch(cur, _PRICES_INSERT, params, page_size=500)
    logger.info("prices_inserted", count=len(rows), asset=rows[0].asset if rows else "?")
    return len(rows)


def bulk_insert_volume(rows: list[VolumeRow]) -> int:
    """Bulk-insert volume rows into the volume hypertable.

    Args:
        rows: List of VolumeRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0
    params = [
        {
            "time": r.time, "asset": r.asset, "exchange": r.exchange,
            "buy_vol": r.buy_vol, "sell_vol": r.sell_vol,
        }
        for r in rows
    ]
    with transaction() as cur:
        psycopg2.extras.execute_batch(cur, _VOLUME_INSERT, params, page_size=500)
    logger.info("volume_inserted", count=len(rows))
    return len(rows)


def bulk_insert_macro(rows: list[MacroEventRow]) -> int:
    """Bulk-insert macro indicator rows into the macro_events hypertable.

    Args:
        rows: List of MacroEventRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0
    params = [
        {
            "time": r.time, "indicator": r.indicator, "value": r.value,
            "prior_value": r.prior_value, "revision": r.revision,
            "source": r.source, "unit": r.unit, "frequency": r.frequency,
        }
        for r in rows
    ]
    with transaction() as cur:
        psycopg2.extras.execute_batch(cur, _MACRO_INSERT, params, page_size=500)
    logger.info("macro_events_inserted", count=len(rows))
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(asset: str, start: datetime, end: datetime,
                 source: str = "yfinance") -> list[dict[str, Any]]:
    """Fetch OHLCV rows for one asset within a time range.

    Args:
        asset: Ticker symbol (e.g. 'AAPL').
        start: Range start (inclusive), timezone-aware.
        end: Range end (inclusive), timezone-aware.
        source: Data source label to filter on.

    Returns:
        List of row dicts ordered by time ascending.
    """
    sql = """
        SELECT time, asset, open, high, low, close, volume, adj_close
        FROM prices
        WHERE asset = %s AND source = %s AND time BETWEEN %s AND %s
        ORDER BY time ASC;
    """
    with transaction() as cur:
        cur.execute(sql, (asset, source, start, end))
        return list(cur.fetchall())


def fetch_macro(indicator: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch macro indicator readings within a time range.

    Args:
        indicator: FRED series ID or indicator name (e.g. 'GDP').
        start: Range start (inclusive), timezone-aware.
        end: Range end (inclusive), timezone-aware.

    Returns:
        List of row dicts ordered by time ascending.
    """
    sql = """
        SELECT time, indicator, value, prior_value, revision, source
        FROM macro_events
        WHERE indicator = %s AND time BETWEEN %s AND %s
        ORDER BY time ASC;
    """
    with transaction() as cur:
        cur.execute(sql, (indicator, start, end))
        return list(cur.fetchall())
