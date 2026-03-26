"""TimescaleDB client for FinBrain.

Writes to the prices, volume, and macro_events hypertables via the
Supabase REST API (supabase-py). This avoids needing a direct TCP
connection to the database, which is blocked on Supabase's free tier.

The hypertables themselves are created by db/timescale/schema.sql and
live in your Supabase project alongside the relational tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from skills.env import get_supabase_key, get_supabase_url
from skills.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Supabase REST client (reuse singleton from db.supabase.client)
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """Return the shared Supabase REST client.

    Returns:
        An authenticated supabase.Client instance.
    """
    from supabase import create_client
    return create_client(get_supabase_url(), get_supabase_key())


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
            ValueError: If close price is negative.
        """
        if self.time.tzinfo is None:
            self.time = self.time.replace(tzinfo=timezone.utc)
        if self.close < 0:
            raise ValueError(f"Negative close price for {self.asset}: {self.close}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict safe for Supabase REST upsert.

        Returns:
            Dict with all fields, time as ISO string.
        """
        return {
            "time":       self.time.isoformat(),
            "asset":      self.asset,
            "asset_class": self.asset_class,
            "source":     self.source,
            "open":       self.open,
            "high":       self.high,
            "low":        self.low,
            "close":      self.close,
            "volume":     self.volume,
            "adj_close":  self.adj_close,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict safe for Supabase REST upsert."""
        return {
            "time":     self.time.isoformat(),
            "asset":    self.asset,
            "exchange": self.exchange,
            "buy_vol":  self.buy_vol,
            "sell_vol": self.sell_vol,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict safe for Supabase REST upsert."""
        return {
            "time":        self.time.isoformat(),
            "indicator":   self.indicator,
            "value":       self.value,
            "prior_value": self.prior_value,
            "revision":    self.revision,
            "source":      self.source,
            "unit":        self.unit,
            "frequency":   self.frequency,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers — batch upsert via Supabase REST API
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 500   # Supabase REST handles up to 500 rows per request comfortably


def bulk_insert_prices(rows: list[PriceRow]) -> int:
    """Upsert OHLCV rows into the prices table via the Supabase REST API.

    Splits large batches into chunks of 500 to stay within request limits.
    Uses upsert with on_conflict so re-running ingestion is idempotent.

    Args:
        rows: List of PriceRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0

    client = _get_client()
    total = 0
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        payload = [r.to_dict() for r in batch]
        client.table("prices").upsert(
            payload, on_conflict="time,asset,source"
        ).execute()
        total += len(batch)

    logger.info("prices_inserted", count=total,
                asset=rows[0].asset if rows else "?")
    return total


def bulk_insert_volume(rows: list[VolumeRow]) -> int:
    """Upsert volume rows into the volume table via the Supabase REST API.

    Args:
        rows: List of VolumeRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0

    client = _get_client()
    total = 0
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        payload = [r.to_dict() for r in batch]
        client.table("volume").upsert(
            payload, on_conflict="time,asset,exchange"
        ).execute()
        total += len(batch)

    logger.info("volume_inserted", count=total)
    return total


def bulk_insert_macro(rows: list[MacroEventRow]) -> int:
    """Upsert macro indicator rows into macro_events via the Supabase REST API.

    Args:
        rows: List of MacroEventRow objects to insert.

    Returns:
        Number of rows inserted/updated.
    """
    if not rows:
        return 0

    client = _get_client()
    total = 0
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        payload = [r.to_dict() for r in batch]
        client.table("macro_events").upsert(
            payload, on_conflict="time,indicator,source"
        ).execute()
        total += len(batch)

    logger.info("macro_events_inserted", count=total)
    return total


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
    client = _get_client()
    result = (
        client.table("prices")
        .select("time,asset,open,high,low,close,volume,adj_close")
        .eq("asset", asset)
        .eq("source", source)
        .gte("time", start.isoformat())
        .lte("time", end.isoformat())
        .order("time")
        .execute()
    )
    return result.data


def fetch_macro(indicator: str, start: datetime,
                end: datetime) -> list[dict[str, Any]]:
    """Fetch macro indicator readings within a time range.

    Args:
        indicator: FRED series ID (e.g. 'GDP').
        start: Range start (inclusive), timezone-aware.
        end: Range end (inclusive), timezone-aware.

    Returns:
        List of row dicts ordered by time ascending.
    """
    client = _get_client()
    result = (
        client.table("macro_events")
        .select("time,indicator,value,prior_value,revision,source")
        .eq("indicator", indicator)
        .gte("time", start.isoformat())
        .lte("time", end.isoformat())
        .order("time")
        .execute()
    )
    return result.data
