"""CoinGecko data connector for FinBrain.

Fetches daily OHLCV history for crypto assets via the CoinGecko
public API (no key required for free tier) and writes to TimescaleDB.

CoinGecko rate-limit: 10–30 req/min on the free tier.
We sleep 2 seconds between requests to stay well within limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from db.supabase.client import (
    AgentRun,
    EvolutionLogEntry,
    end_agent_run,
    log_evolution,
    start_agent_run,
)
from db.timescale.client import PriceRow, bulk_insert_prices
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "coingecko_connector"
BASE_URL = "https://api.coingecko.com/api/v3"
REQUEST_DELAY_SECONDS = 2.0


@dataclass
class CoinGeckoFetchResult:
    """Result of one CoinGecko fetch operation.

    Attributes:
        coin_id: The CoinGecko coin ID (e.g. 'bitcoin').
        ticker: Normalised ticker symbol stored in TimescaleDB (e.g. 'BTC').
        rows_written: Number of daily OHLCV rows written.
        start_date: ISO string of the earliest date fetched.
        end_date: ISO string of the latest date fetched.
        error: Error message if the fetch failed, else None.
    """
    coin_id: str
    ticker: str
    rows_written: int
    start_date: str | None
    end_date: str | None
    error: str | None = None


def _fetch_ohlc(coin_id: str, vs_currency: str = "usd", days: int = 730) -> list[list]:
    """Call the CoinGecko OHLC endpoint and return raw data.

    Args:
        coin_id: CoinGecko coin ID (e.g. 'bitcoin').
        vs_currency: Quote currency, default 'usd'.
        days: Number of days of history to fetch (max 730 for free tier daily).

    Returns:
        List of [timestamp_ms, open, high, low, close] lists.

    Raises:
        httpx.HTTPStatusError: If the API returns a non-2xx status.
    """
    url = f"{BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    response = httpx.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_crypto(
    coin_id: str,
    ticker: str,
    days: int = 730,
) -> CoinGeckoFetchResult:
    """Fetch daily OHLCV for one crypto coin and write to TimescaleDB.

    CoinGecko OHLC endpoint returns [timestamp_ms, open, high, low, close].
    Volume is not included in the OHLC endpoint; volume column defaults to 0.

    Args:
        coin_id: CoinGecko coin ID (e.g. 'bitcoin').
        ticker: Short ticker to store (e.g. 'BTC').
        days: History depth in days (max 730 for free tier daily bars).

    Returns:
        A CoinGeckoFetchResult describing what was written.
    """
    logger.info("coingecko_fetch_start", coin_id=coin_id, days=days)
    try:
        raw = _fetch_ohlc(coin_id, days=days)

        if not raw:
            return CoinGeckoFetchResult(
                coin_id=coin_id, ticker=ticker,
                rows_written=0, start_date=None, end_date=None,
                error="empty response from CoinGecko"
            )

        rows: list[PriceRow] = []
        for candle in raw:
            ts_ms, open_, high, low, close = candle
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            rows.append(PriceRow(
                time=dt,
                asset=ticker,
                asset_class="crypto",
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=0.0,
                source="coingecko",
            ))

        written = bulk_insert_prices(rows)
        dates = [r.time.isoformat() for r in rows]

        logger.info("coingecko_fetch_done", ticker=ticker, rows=written)
        return CoinGeckoFetchResult(
            coin_id=coin_id, ticker=ticker, rows_written=written,
            start_date=min(dates), end_date=max(dates),
        )

    except Exception as exc:
        logger.error("coingecko_fetch_error", coin_id=coin_id, error=str(exc))
        return CoinGeckoFetchResult(
            coin_id=coin_id, ticker=ticker,
            rows_written=0, start_date=None, end_date=None,
            error=str(exc),
        )


def run(coins: list[tuple[str, str]], days: int = 730) -> list[CoinGeckoFetchResult]:
    """Fetch history for a list of (coin_id, ticker) pairs.

    Logs the run to agent_runs and evolution_log in Supabase.

    Args:
        coins: List of (coin_id, ticker) tuples, e.g. [('bitcoin', 'BTC')].
        days: History depth in days for all coins.

    Returns:
        List of CoinGeckoFetchResult — one per coin.
    """
    agent_run = AgentRun(agent_name=AGENT_ID, trigger="manual")
    start_agent_run(agent_run)

    results: list[CoinGeckoFetchResult] = []
    for coin_id, ticker in coins:
        result = fetch_crypto(coin_id, ticker, days=days)
        results.append(result)
        time.sleep(REQUEST_DELAY_SECONDS)

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    total_rows = sum(r.rows_written for r in successes)

    summary = (
        f"Fetched {len(successes)}/{len(results)} coins. "
        f"Total rows: {total_rows}. "
        f"Failures: {[r.coin_id for r in failures]}"
    )
    end_agent_run(agent_run, success=len(failures) == 0, summary=summary)

    log_evolution(EvolutionLogEntry(
        agent_id=AGENT_ID,
        action="fetch_crypto_ohlcv",
        after_state={"coins_fetched": len(successes), "rows_written": total_rows},
        metadata={"failures": [r.coin_id for r in failures]},
    ))

    return results
