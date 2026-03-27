"""yfinance data connector for FinBrain.

Fetches daily OHLCV history for equities, ETFs, forex pairs, and
commodities from Yahoo Finance and writes them to TimescaleDB.

Every run logs itself to the agent_runs and evolution_log tables.

Validation is applied before any write: if the fetched DataFrame fails
ERROR-level checks, the payload is routed to the QuarantineStore and
no rows are written to TimescaleDB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from data.validation.quarantine import QuarantineStore
from data.validation.validator import validate_ohlcv
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

AGENT_ID = "yfinance_connector"

# Module-level QuarantineStore — created once, reused across calls.
_quarantine = QuarantineStore()


@dataclass
class YFinanceFetchResult:
    """Result of one yfinance fetch operation.

    Attributes:
        ticker: The ticker symbol fetched.
        asset_class: The asset class (equity, forex, commodity).
        rows_written: Number of OHLCV rows written to TimescaleDB.
        start_date: Earliest date in the fetched range.
        end_date: Latest date in the fetched range.
        error: Error message if the fetch failed, else None.
        quarantine_id: QuarantineStore entry ID if data was quarantined,
                       else None.
    """
    ticker: str
    asset_class: str
    rows_written: int
    start_date: str | None
    end_date: str | None
    error: str | None = None
    quarantine_id: str | None = None


def fetch_ohlcv(
    ticker: str,
    asset_class: str,
    period: str = "2y",
    interval: str = "1d",
) -> YFinanceFetchResult:
    """Fetch OHLCV history for one ticker and write to TimescaleDB.

    Uses yfinance Ticker.history() which returns adjusted prices by default.
    Rows with any NaN in open/high/low/close are dropped before writing.

    Args:
        ticker: The Yahoo Finance ticker symbol (e.g. 'AAPL', 'BTC-USD').
        asset_class: One of 'equity', 'forex', 'commodity'.
        period: yfinance period string (e.g. '2y', '1y', '6mo').
        interval: Bar interval (e.g. '1d', '1h').

    Returns:
        A YFinanceFetchResult describing what was written.
    """
    logger.info("yfinance_fetch_start", ticker=ticker, period=period)
    try:
        df: pd.DataFrame = yf.Ticker(ticker).history(period=period, interval=interval)

        if df.empty:
            logger.warning("yfinance_empty_response", ticker=ticker)
            return YFinanceFetchResult(
                ticker=ticker, asset_class=asset_class,
                rows_written=0, start_date=None, end_date=None,
                error="empty response from yfinance"
            )

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(df.index, utc=True)

        # ── Validation gate ───────────────────────────────────────────────
        # Build a lowercase-column view for the validator (expects lowercase).
        df_val = df.rename(columns=str.lower).copy()
        df_val["date"] = df_val.index
        report = validate_ohlcv(df_val, source_id="yfinance", symbol=ticker)
        if not report.passed:
            entry = _quarantine.save(report, df_val)
            logger.warning(
                "yfinance_validation_failed",
                ticker=ticker, errors=len(report.errors),
                quarantine_id=entry.entry_id,
            )
            return YFinanceFetchResult(
                ticker=ticker, asset_class=asset_class,
                rows_written=0, start_date=None, end_date=None,
                error=f"validation failed ({len(report.errors)} errors)",
                quarantine_id=entry.entry_id,
            )
        # ── End validation gate ───────────────────────────────────────────

        rows: list[PriceRow] = []
        for ts, row in df.iterrows():
            rows.append(PriceRow(
                time=ts.to_pydatetime(),
                asset=ticker,
                asset_class=asset_class,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0) or 0),
                adj_close=float(row["Close"]),  # history() already returns adjusted
                source="yfinance",
            ))

        written = bulk_insert_prices(rows)
        start_date = df.index.min().isoformat()
        end_date = df.index.max().isoformat()

        logger.info("yfinance_fetch_done", ticker=ticker, rows=written)
        return YFinanceFetchResult(
            ticker=ticker, asset_class=asset_class,
            rows_written=written, start_date=start_date, end_date=end_date,
        )

    except Exception as exc:
        logger.error("yfinance_fetch_error", ticker=ticker, error=str(exc))
        return YFinanceFetchResult(
            ticker=ticker, asset_class=asset_class,
            rows_written=0, start_date=None, end_date=None,
            error=str(exc),
        )


def run(
    tickers: dict[str, str],
    period: str = "2y",
) -> list[YFinanceFetchResult]:
    """Fetch OHLCV history for a dict of {ticker: asset_class} pairs.

    Logs the entire run to agent_runs and evolution_log in Supabase.

    Args:
        tickers: Mapping of ticker symbol → asset_class string.
        period: yfinance period string applied to all tickers.

    Returns:
        List of YFinanceFetchResult — one per ticker.
    """
    agent_run = AgentRun(agent_name=AGENT_ID, trigger="manual")
    start_agent_run(agent_run)

    results: list[YFinanceFetchResult] = []
    for ticker, asset_class in tickers.items():
        result = fetch_ohlcv(ticker, asset_class, period=period)
        results.append(result)
        time.sleep(0.3)  # be polite to Yahoo Finance

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    total_rows = sum(r.rows_written for r in successes)

    summary = (
        f"Fetched {len(successes)}/{len(results)} tickers. "
        f"Total rows written: {total_rows}. "
        f"Failures: {[r.ticker for r in failures]}"
    )

    end_agent_run(agent_run, success=len(failures) == 0, summary=summary)

    log_evolution(EvolutionLogEntry(
        agent_id=AGENT_ID,
        action="fetch_ohlcv",
        before_state=None,
        after_state={"tickers_fetched": len(successes), "rows_written": total_rows},
        metadata={"failures": [r.ticker for r in failures]},
    ))

    logger.info("yfinance_run_complete", total=len(results),
                successes=len(successes), rows=total_rows)
    return results
