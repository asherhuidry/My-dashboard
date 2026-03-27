"""FRED (Federal Reserve Economic Data) connector for FinBrain.

Fetches macro indicator time series via the fredapi library and
writes them to the TimescaleDB macro_events hypertable.

Validation is applied before any write: if the fetched series fails
ERROR-level checks, the payload is routed to the QuarantineStore and
no rows are written to TimescaleDB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from fredapi import Fred

from data.validation.quarantine import QuarantineStore
from data.validation.validator import validate_timeseries
from db.supabase.client import (
    AgentRun,
    EvolutionLogEntry,
    end_agent_run,
    log_evolution,
    start_agent_run,
)
from db.timescale.client import MacroEventRow, bulk_insert_macro
from skills.env import get_fred_api_key
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "fred_connector"

# Module-level QuarantineStore — created once, reused across calls.
_quarantine = QuarantineStore()

# The 10 macro indicators specified in CLAUDE.md
DEFAULT_SERIES: list[tuple[str, str, str]] = [
    ("GDP",       "Gross Domestic Product",              "quarterly"),
    ("CPIAUCSL",  "Consumer Price Index (All Urban)",    "monthly"),
    ("FEDFUNDS",  "Federal Funds Effective Rate",        "monthly"),
    ("UNRATE",    "Unemployment Rate",                   "monthly"),
    ("M2SL",      "M2 Money Supply",                    "monthly"),
    ("GS10",      "10-Year Treasury Constant Maturity", "daily"),
    ("GS2",       "2-Year Treasury Constant Maturity",  "daily"),
    ("VIXCLS",    "CBOE Volatility Index (VIX)",         "daily"),
    ("RSAFS",     "Retail Sales",                        "monthly"),
    ("MANEMP",    "ISM Manufacturing Employment",        "monthly"),
]


@dataclass
class FredFetchResult:
    """Result of one FRED series fetch operation.

    Attributes:
        series_id: The FRED series ID (e.g. 'GDP').
        name: Human-readable indicator name.
        rows_written: Number of observations written to TimescaleDB.
        start_date: ISO string of the earliest date fetched.
        end_date: ISO string of the latest date fetched.
        error: Error message if the fetch failed, else None.
    """
    series_id: str
    name: str
    rows_written: int
    start_date: str | None
    end_date: str | None
    error: str | None = None
    quarantine_id: str | None = None


def _get_fred_client() -> Fred:
    """Return an authenticated Fred client.

    Returns:
        A fredapi.Fred instance authenticated with FRED_API_KEY.
    """
    return Fred(api_key=get_fred_api_key())


def fetch_series(
    series_id: str,
    name: str,
    frequency: str,
    observation_start: str = "2000-01-01",
) -> FredFetchResult:
    """Fetch one FRED series and write it to the macro_events hypertable.

    Args:
        series_id: The FRED series ID to fetch.
        name: Human-readable name stored as the indicator label.
        frequency: Frequency string for metadata (daily/monthly/quarterly).
        observation_start: ISO date string; fetch data from this date onward.

    Returns:
        A FredFetchResult describing what was written.
    """
    logger.info("fred_fetch_start", series_id=series_id)
    try:
        fred = _get_fred_client()
        series: pd.Series = fred.get_series(
            series_id, observation_start=observation_start
        )

        if series.empty:
            return FredFetchResult(
                series_id=series_id, name=name,
                rows_written=0, start_date=None, end_date=None,
                error="empty series from FRED"
            )

        series = series.dropna()

        # ── Validation gate ───────────────────────────────────────────────
        # Build a two-column DataFrame (date, value) for the validator.
        df_val = pd.DataFrame({
            "date":  pd.to_datetime(series.index),
            "value": series.values,
        })
        # FRED monthly/quarterly data can be 30-100 days stale by design;
        # use a generous max_stale_days to avoid WARNING noise on these series.
        report = validate_timeseries(
            df_val,
            source_id    = "fred_api",
            dataset_key  = series_id,
            required_cols = ["date", "value"],
            time_col     = "date",
            min_rows     = 5,
            max_stale_days = 100,   # quarterly GDP is never "fresh"
        )
        if not report.passed:
            entry = _quarantine.save(report, df_val)
            logger.warning(
                "fred_validation_failed",
                series_id=series_id, errors=len(report.errors),
                quarantine_id=entry.entry_id,
            )
            return FredFetchResult(
                series_id=series_id, name=name,
                rows_written=0, start_date=None, end_date=None,
                error=f"validation failed ({len(report.errors)} errors)",
                quarantine_id=entry.entry_id,
            )
        # ── End validation gate ───────────────────────────────────────────

        rows: list[MacroEventRow] = []
        for date, value in series.items():
            dt = pd.Timestamp(date).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            rows.append(MacroEventRow(
                time=dt,
                indicator=series_id,
                value=float(value),
                source="fred",
                frequency=frequency,
            ))

        written = bulk_insert_macro(rows)
        dates = [r.time.isoformat() for r in rows]

        logger.info("fred_fetch_done", series_id=series_id, rows=written)
        return FredFetchResult(
            series_id=series_id, name=name, rows_written=written,
            start_date=min(dates), end_date=max(dates),
        )

    except Exception as exc:
        logger.error("fred_fetch_error", series_id=series_id, error=str(exc))
        return FredFetchResult(
            series_id=series_id, name=name,
            rows_written=0, start_date=None, end_date=None,
            error=str(exc),
        )


def run(
    series: list[tuple[str, str, str]] | None = None,
    observation_start: str = "2000-01-01",
) -> list[FredFetchResult]:
    """Fetch all FRED macro indicator series and write to TimescaleDB.

    Logs the run to agent_runs and evolution_log in Supabase.

    Args:
        series: List of (series_id, name, frequency) tuples.
                Defaults to the 10 series defined in DEFAULT_SERIES.
        observation_start: ISO date string for the start of history.

    Returns:
        List of FredFetchResult — one per series.
    """
    if series is None:
        series = DEFAULT_SERIES

    agent_run = AgentRun(agent_name=AGENT_ID, trigger="manual")
    start_agent_run(agent_run)

    results: list[FredFetchResult] = []
    for series_id, name, frequency in series:
        result = fetch_series(series_id, name, frequency, observation_start)
        results.append(result)
        time.sleep(0.5)

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    total_rows = sum(r.rows_written for r in successes)

    summary = (
        f"Fetched {len(successes)}/{len(results)} series. "
        f"Total rows: {total_rows}. "
        f"Failures: {[r.series_id for r in failures]}"
    )
    end_agent_run(agent_run, success=len(failures) == 0, summary=summary)

    log_evolution(EvolutionLogEntry(
        agent_id=AGENT_ID,
        action="fetch_macro_series",
        after_state={"series_fetched": len(successes), "rows_written": total_rows},
        metadata={"failures": [r.series_id for r in failures]},
    ))

    return results
