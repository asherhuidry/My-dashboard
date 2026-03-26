"""Alpha Vantage data connector for FinBrain.

Fetches supplemental daily adjusted OHLCV for equities via the
Alpha Vantage TIME_SERIES_DAILY_ADJUSTED endpoint and writes to
TimescaleDB. Used as a cross-validation source alongside yfinance.

Alpha Vantage free tier: 25 requests/day, 5 requests/minute.
We respect this with conservative rate limiting.
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
from skills.env import get_alpha_vantage_key
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "alpha_vantage_connector"
BASE_URL = "https://www.alphavantage.co/query"
# Free tier: 5 req/min — we wait 13 seconds to be safe
REQUEST_DELAY_SECONDS = 13.0


@dataclass
class AlphaVantageFetchResult:
    """Result of one Alpha Vantage fetch operation.

    Attributes:
        ticker: The ticker symbol fetched.
        rows_written: Number of OHLCV rows written to TimescaleDB.
        start_date: ISO string of the earliest date in the response.
        end_date: ISO string of the latest date in the response.
        error: Error message if the fetch failed, else None.
    """
    ticker: str
    rows_written: int
    start_date: str | None
    end_date: str | None
    error: str | None = None


def _fetch_daily_adjusted(ticker: str, outputsize: str = "full") -> dict:
    """Call the Alpha Vantage TIME_SERIES_DAILY_ADJUSTED endpoint.

    Args:
        ticker: Equity ticker symbol.
        outputsize: 'compact' (100 rows) or 'full' (20+ years).

    Returns:
        The raw JSON response dict.

    Raises:
        httpx.HTTPStatusError: For non-2xx HTTP responses.
        ValueError: If the response contains an error message.
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": get_alpha_vantage_key(),
    }
    response = httpx.get(BASE_URL, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit hit for {ticker}: {data['Note']}")
    if "Information" in data:
        raise ValueError(f"Alpha Vantage info message for {ticker}: {data['Information']}")

    return data


def fetch_equity(ticker: str, outputsize: str = "full") -> AlphaVantageFetchResult:
    """Fetch daily adjusted OHLCV for one equity and write to TimescaleDB.

    Args:
        ticker: The equity ticker symbol (e.g. 'AAPL').
        outputsize: 'compact' (100 rows) or 'full' (full history).

    Returns:
        An AlphaVantageFetchResult describing what was written.
    """
    logger.info("alpha_vantage_fetch_start", ticker=ticker)
    try:
        data = _fetch_daily_adjusted(ticker, outputsize=outputsize)
        time_series: dict = data.get("Time Series (Daily)", {})

        if not time_series:
            return AlphaVantageFetchResult(
                ticker=ticker, rows_written=0,
                start_date=None, end_date=None,
                error="no time series data in response"
            )

        rows: list[PriceRow] = []
        for date_str, bar in time_series.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            close = float(bar["4. close"])
            adj_close = float(bar["5. adjusted close"])
            if close <= 0:
                continue  # skip bad rows — noise filter will catch in DB
            rows.append(PriceRow(
                time=dt,
                asset=ticker,
                asset_class="equity",
                open=float(bar["1. open"]),
                high=float(bar["2. high"]),
                low=float(bar["3. low"]),
                close=close,
                volume=float(bar["6. volume"]),
                adj_close=adj_close,
                source="alpha_vantage",
            ))

        if not rows:
            return AlphaVantageFetchResult(
                ticker=ticker, rows_written=0,
                start_date=None, end_date=None,
                error="all rows filtered as invalid"
            )

        written = bulk_insert_prices(rows)
        dates = [r.time.isoformat() for r in rows]

        logger.info("alpha_vantage_fetch_done", ticker=ticker, rows=written)
        return AlphaVantageFetchResult(
            ticker=ticker, rows_written=written,
            start_date=min(dates), end_date=max(dates),
        )

    except Exception as exc:
        logger.error("alpha_vantage_fetch_error", ticker=ticker, error=str(exc))
        return AlphaVantageFetchResult(
            ticker=ticker, rows_written=0,
            start_date=None, end_date=None,
            error=str(exc),
        )


def run(tickers: list[str], outputsize: str = "full") -> list[AlphaVantageFetchResult]:
    """Fetch daily adjusted history for a list of equity tickers.

    Logs the run to agent_runs and evolution_log in Supabase.

    Note: Free tier allows 25 requests/day. Keep tickers list short
    or use the 'compact' outputsize for faster runs.

    Args:
        tickers: List of equity ticker symbols to fetch.
        outputsize: 'compact' or 'full'.

    Returns:
        List of AlphaVantageFetchResult — one per ticker.
    """
    agent_run = AgentRun(agent_name=AGENT_ID, trigger="manual")
    start_agent_run(agent_run)

    results: list[AlphaVantageFetchResult] = []
    for ticker in tickers:
        result = fetch_equity(ticker, outputsize=outputsize)
        results.append(result)
        time.sleep(REQUEST_DELAY_SECONDS)

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    total_rows = sum(r.rows_written for r in successes)

    summary = (
        f"Fetched {len(successes)}/{len(results)} tickers. "
        f"Total rows: {total_rows}. "
        f"Failures: {[r.ticker for r in failures]}"
    )
    end_agent_run(agent_run, success=len(failures) == 0, summary=summary)

    log_evolution(EvolutionLogEntry(
        agent_id=AGENT_ID,
        action="fetch_equity_ohlcv",
        after_state={"tickers_fetched": len(successes), "rows_written": total_rows},
        metadata={"failures": [r.ticker for r in failures]},
    ))

    return results
