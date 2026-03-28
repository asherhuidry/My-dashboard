"""Finnhub earnings calendar connector for FinBrain.

Fetches upcoming and recent earnings events via the Finnhub REST API
and writes EarningsEvent nodes + REPORTS edges to Neo4j, plus summary
rows to the TimescaleDB macro_events hypertable.

API docs: https://finnhub.io/docs/api/earnings-calendar
Rate limit: 60 calls/min on free tier.
Authentication: API key via query parameter.

Usage::

    from data.ingest.finnhub_connector import fetch_earnings, run

    # Fetch earnings for a date range
    results = fetch_earnings("2026-03-01", "2026-03-31")

    # Full run: fetch + write to Neo4j + log
    summary = run()
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Any

from skills.env import get_finnhub_api_key
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "finnhub_connector"
SOURCE_ID = "finnhub_earnings_calendar"

_BASE_URL = "https://finnhub.io/api/v1"
_TIMEOUT = 15  # seconds
_RATE_DELAY = 1.1  # seconds between requests (60/min limit → 1 req/sec)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class EarningsEvent:
    """One earnings release event from Finnhub.

    Attributes:
        symbol:       Ticker symbol (e.g. "AAPL").
        date:         Earnings release date.
        hour:         "bmo" (before market open), "amc" (after market close), or "".
        eps_actual:   Actual EPS reported (None if not yet reported).
        eps_estimate: Consensus EPS estimate.
        revenue_actual:   Actual revenue (None if not yet reported).
        revenue_estimate: Consensus revenue estimate.
        quarter:      Fiscal quarter (1-4).
        year:         Fiscal year.
    """
    symbol:           str
    date:             date
    hour:             str = ""
    eps_actual:       float | None = None
    eps_estimate:     float | None = None
    revenue_actual:   float | None = None
    revenue_estimate: float | None = None
    quarter:          int | None = None
    year:             int | None = None

    @property
    def eps_surprise(self) -> float | None:
        """EPS surprise (actual - estimate). None if either is missing."""
        if self.eps_actual is not None and self.eps_estimate is not None:
            return round(self.eps_actual - self.eps_estimate, 4)
        return None

    @property
    def eps_surprise_pct(self) -> float | None:
        """EPS surprise as percentage. None if estimate is zero or missing."""
        if self.eps_surprise is not None and self.eps_estimate and self.eps_estimate != 0:
            return round(self.eps_surprise / abs(self.eps_estimate) * 100, 2)
        return None

    @property
    def has_actuals(self) -> bool:
        """Whether actual results have been reported."""
        return self.eps_actual is not None


@dataclass
class FinnhubFetchResult:
    """Result of one Finnhub earnings calendar fetch.

    Attributes:
        from_date:     Start of date range queried.
        to_date:       End of date range queried.
        events_fetched: Number of earnings events returned.
        events_written: Number of events written to Neo4j.
        symbols:        Unique tickers with events.
        error:          Error message if fetch failed.
    """
    from_date:      str
    to_date:        str
    events_fetched: int = 0
    events_written: int = 0
    symbols:        list[str] = field(default_factory=list)
    error:          str | None = None


# ── API fetch ────────────────────────────────────────────────────────────────

def fetch_earnings(
    from_date: str,
    to_date: str,
    symbol: str | None = None,
) -> list[EarningsEvent]:
    """Fetch earnings calendar events from Finnhub.

    Args:
        from_date: Start date (YYYY-MM-DD).
        to_date:   End date (YYYY-MM-DD).
        symbol:    Optional ticker to filter for.

    Returns:
        List of EarningsEvent objects.

    Raises:
        ConnectionError: If the API request fails.
    """
    api_key = get_finnhub_api_key()
    url = f"{_BASE_URL}/calendar/earnings?from={from_date}&to={to_date}&token={api_key}"
    if symbol:
        url += f"&symbol={symbol}"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "FinBrain-FinnhubConnector/1.0")

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ConnectionError(
            f"Finnhub API error {exc.code}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Finnhub API network error: {exc.reason}"
        ) from exc

    raw_events = data.get("earningsCalendar", [])
    events: list[EarningsEvent] = []

    for raw in raw_events:
        try:
            event_date = date.fromisoformat(raw["date"])
            events.append(EarningsEvent(
                symbol=raw.get("symbol", ""),
                date=event_date,
                hour=raw.get("hour", ""),
                eps_actual=_safe_float(raw.get("epsActual")),
                eps_estimate=_safe_float(raw.get("epsEstimate")),
                revenue_actual=_safe_float(raw.get("revenueActual")),
                revenue_estimate=_safe_float(raw.get("revenueEstimate")),
                quarter=raw.get("quarter"),
                year=raw.get("year"),
            ))
        except (KeyError, ValueError) as exc:
            logger.debug("finnhub_parse_skip", raw=raw, error=str(exc))

    logger.info(
        "finnhub_earnings_fetched",
        from_date=from_date,
        to_date=to_date,
        events=len(events),
    )
    return events


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for null/invalid."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Neo4j graph write ────────────────────────────────────────────────────────

def write_earnings_to_graph(
    events: list[EarningsEvent],
    universe_symbols: set[str] | None = None,
) -> int:
    """Write earnings events to Neo4j as Event nodes with REPORTS edges.

    Creates:
      (Asset)-[:REPORTS]->(Event:EarningsEvent)

    Only writes events for symbols in the universe if universe_symbols is provided.

    Args:
        events:           List of EarningsEvent objects.
        universe_symbols: Optional set of tickers to filter on.

    Returns:
        Number of events written.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
    except Exception as exc:
        logger.warning("Neo4j unavailable, skipping earnings graph write: %s", exc)
        return 0

    written = 0
    ts = datetime.now(tz=timezone.utc).isoformat()

    with driver.session() as session:
        for event in events:
            if universe_symbols and event.symbol not in universe_symbols:
                continue

            event_id = f"earnings_{event.symbol}_{event.date.isoformat()}"
            props: dict[str, Any] = {
                "event_id": event_id,
                "event_type": "earnings",
                "event_date": event.date.isoformat(),
                "symbol": event.symbol,
                "hour": event.hour,
                "quarter": event.quarter,
                "year": event.year,
                "updated_at": ts,
            }
            if event.eps_actual is not None:
                props["eps_actual"] = event.eps_actual
            if event.eps_estimate is not None:
                props["eps_estimate"] = event.eps_estimate
            if event.eps_surprise is not None:
                props["eps_surprise"] = event.eps_surprise
            if event.eps_surprise_pct is not None:
                props["eps_surprise_pct"] = event.eps_surprise_pct
            if event.revenue_actual is not None:
                props["revenue_actual"] = event.revenue_actual
            if event.revenue_estimate is not None:
                props["revenue_estimate"] = event.revenue_estimate

            try:
                session.run(
                    "MERGE (e:Event {event_id: $event_id}) "
                    "SET e += $props, e:EarningsEvent "
                    "WITH e "
                    "MERGE (a:Asset {ticker: $symbol}) "
                    "MERGE (a)-[r:REPORTS]->(e) "
                    "SET r.updated_at = $ts",
                    event_id=event_id, props=props, symbol=event.symbol, ts=ts,
                )
                written += 1
            except Exception as exc:
                logger.debug(
                    "finnhub_graph_write_failed",
                    symbol=event.symbol, error=str(exc),
                )

    return written


# ── Batch runner ─────────────────────────────────────────────────────────────

def run(
    lookback_days: int = 7,
    lookahead_days: int = 14,
    write_to_graph: bool = True,
    filter_universe: bool = True,
) -> FinnhubFetchResult:
    """Fetch earnings calendar and write to Neo4j.

    Fetches a window of [today - lookback_days, today + lookahead_days].

    Args:
        lookback_days:   Days of historical earnings to include.
        lookahead_days:  Days of upcoming earnings to include.
        write_to_graph:  If True, write events to Neo4j.
        filter_universe: If True, only write events for universe equities.

    Returns:
        FinnhubFetchResult summary.
    """
    today = date.today()
    from_date = (today - timedelta(days=lookback_days)).isoformat()
    to_date = (today + timedelta(days=lookahead_days)).isoformat()

    try:
        events = fetch_earnings(from_date, to_date)
    except ConnectionError as exc:
        logger.error("finnhub_run_error", error=str(exc))
        return FinnhubFetchResult(
            from_date=from_date, to_date=to_date, error=str(exc),
        )

    universe_symbols: set[str] | None = None
    if filter_universe:
        try:
            from data.ingest.universe import EQUITIES
            universe_symbols = set(EQUITIES)
        except ImportError:
            universe_symbols = None

    unique_symbols = sorted({e.symbol for e in events})
    events_written = 0

    if write_to_graph and events:
        events_written = write_earnings_to_graph(events, universe_symbols)

    result = FinnhubFetchResult(
        from_date=from_date,
        to_date=to_date,
        events_fetched=len(events),
        events_written=events_written,
        symbols=unique_symbols,
    )

    # Log to evolution log
    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id=AGENT_ID,
            action="fetch_earnings_calendar",
            after_state={
                "from_date": from_date,
                "to_date": to_date,
                "events_fetched": len(events),
                "events_written": events_written,
                "unique_symbols": len(unique_symbols),
                "with_actuals": sum(1 for e in events if e.has_actuals),
            },
        ))
    except Exception:
        pass

    logger.info(
        "finnhub_run_complete",
        events_fetched=len(events),
        events_written=events_written,
        symbols=len(unique_symbols),
    )
    return result
