"""ECB Statistical Data Warehouse connector for FinBrain.

Fetches Euro area macro time series via the ECB SDMX REST API and
writes them to the TimescaleDB macro_events hypertable.

The ECB API requires no authentication, returns clean CSV, and covers:
- Exchange rates (EUR/USD, EUR/GBP, EUR/JPY, …) — daily
- HICP inflation (annual rate of change) — monthly
- Key ECB interest rates (deposit facility, main refinancing) — event-based
- Money supply (M1, M3) — monthly
- Government bond yields (10-year benchmarks) — monthly

Validation is applied before any write: if the fetched series fails
ERROR-level checks, the payload is routed to the QuarantineStore.

Data source:
    https://data-api.ecb.europa.eu/
    No API key required. Rate limit: ~100 req/min (undocumented).
    Source registry ID: ecb_statistical_data_warehouse

Usage::

    from data.ingest.ecb_connector import fetch_series, run

    # Fetch one series
    result = fetch_series("EUR/USD Exchange Rate", ECB_SERIES[0])

    # Fetch all default series
    results = run()
"""
from __future__ import annotations

import csv
import io
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from data.validation.quarantine import QuarantineStore
from data.validation.validator import validate_timeseries
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "ecb_connector"
SOURCE_ID = "ecb_statistical_data_warehouse"

_quarantine = QuarantineStore()

_BASE_URL    = "https://data-api.ecb.europa.eu/service/data"
_USER_AGENT  = "FinBrain-ECBConnector/1.0 (macro data collection)"
_TIMEOUT     = 20  # seconds
_RATE_DELAY  = 0.5  # seconds between requests


# ── Series definitions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ECBSeriesDef:
    """Definition of one ECB SDMX series to fetch.

    Attributes:
        name:       Human-readable name for logging and the indicator field.
        flow_ref:   SDMX dataflow (e.g. "EXR", "FM", "ICP", "BSI").
        key:        SDMX series key (e.g. "D.USD.EUR.SP00.A").
        frequency:  Data frequency string (daily, monthly, quarterly, event).
        unit:       Unit of measurement (e.g. "rate", "percent", "millions_eur").
        indicator:  Short stable indicator ID stored in macro_events.
    """
    name:       str
    flow_ref:   str
    key:        str
    frequency:  str
    unit:       str
    indicator:  str


#: Default series covering the most research-useful ECB data.
#: Each tuple maps to a single SDMX series key.
ECB_SERIES: list[ECBSeriesDef] = [
    # ── Exchange rates (daily) ───────────────────────────────────────────
    ECBSeriesDef(
        name="EUR/USD Exchange Rate",
        flow_ref="EXR", key="D.USD.EUR.SP00.A",
        frequency="daily", unit="rate", indicator="ECB_EURUSD",
    ),
    ECBSeriesDef(
        name="EUR/GBP Exchange Rate",
        flow_ref="EXR", key="D.GBP.EUR.SP00.A",
        frequency="daily", unit="rate", indicator="ECB_EURGBP",
    ),
    ECBSeriesDef(
        name="EUR/JPY Exchange Rate",
        flow_ref="EXR", key="D.JPY.EUR.SP00.A",
        frequency="daily", unit="rate", indicator="ECB_EURJPY",
    ),
    ECBSeriesDef(
        name="EUR/CHF Exchange Rate",
        flow_ref="EXR", key="D.CHF.EUR.SP00.A",
        frequency="daily", unit="rate", indicator="ECB_EURCHF",
    ),
    # ── Key ECB interest rates ───────────────────────────────────────────
    ECBSeriesDef(
        name="ECB Deposit Facility Rate",
        flow_ref="FM", key="B.U2.EUR.4F.KR.DFR.LEV",
        frequency="event", unit="percent", indicator="ECB_DFR",
    ),
    ECBSeriesDef(
        name="ECB Main Refinancing Rate",
        flow_ref="FM", key="B.U2.EUR.4F.KR.MRR_FR.LEV",
        frequency="event", unit="percent", indicator="ECB_MRR",
    ),
    # ── Inflation (monthly) ──────────────────────────────────────────────
    ECBSeriesDef(
        name="Euro Area HICP Inflation (YoY %)",
        flow_ref="ICP", key="M.U2.N.000000.4.ANR",
        frequency="monthly", unit="percent", indicator="ECB_HICP",
    ),
    # ── Government bond yields (monthly, 10-year benchmark) ──────────────
    ECBSeriesDef(
        name="Euro Area 10Y Government Bond Yield",
        flow_ref="YC", key="B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
        frequency="daily", unit="percent", indicator="ECB_10Y_YIELD",
    ),
]


# ── Fetch result ─────────────────────────────────────────────────────────────

@dataclass
class ECBFetchResult:
    """Result of one ECB series fetch operation.

    Attributes:
        indicator:    The indicator ID (e.g. 'ECB_EURUSD').
        name:         Human-readable series name.
        rows_fetched: Number of observations parsed from the API response.
        rows_written: Number of rows written to TimescaleDB.
        start_date:   ISO string of the earliest observation.
        end_date:     ISO string of the latest observation.
        error:        Error message if the fetch failed, else None.
        quarantine_id: ID of the quarantine entry if validation failed.
    """
    indicator:     str
    name:          str
    rows_fetched:  int
    rows_written:  int
    start_date:    str | None
    end_date:      str | None
    error:         str | None       = None
    quarantine_id: str | None       = None


# ── Core fetch logic ─────────────────────────────────────────────────────────

def build_url(
    series_def:     ECBSeriesDef,
    start_period:   str = "2000-01-01",
) -> str:
    """Build the ECB SDMX CSV data URL for a series definition.

    Args:
        series_def:   The ECBSeriesDef to fetch.
        start_period: ISO date string for the start of the data window.

    Returns:
        The full URL string.
    """
    return (
        f"{_BASE_URL}/{series_def.flow_ref}/{series_def.key}"
        f"?format=csvdata&startPeriod={start_period}&detail=dataonly"
    )


def fetch_csv(url: str, timeout: int = _TIMEOUT) -> str:
    """Fetch raw CSV text from an ECB SDMX URL.

    Args:
        url:     The full ECB API URL.
        timeout: HTTP timeout in seconds.

    Returns:
        The response body as a string.

    Raises:
        ConnectionError: On HTTP or network errors.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ConnectionError(f"ECB API HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"ECB API network error: {exc.reason}") from exc


def parse_ecb_csv(raw_csv: str, series_def: ECBSeriesDef) -> pd.DataFrame:
    """Parse ECB SDMX CSV into a normalized DataFrame.

    The ECB CSV has columns like KEY, FREQ, ..., TIME_PERIOD, OBS_VALUE.
    We extract only TIME_PERIOD and OBS_VALUE and rename them.

    Args:
        raw_csv:    Raw CSV text from the ECB API.
        series_def: The series definition (for indicator/unit metadata).

    Returns:
        DataFrame with columns: date, value, indicator, unit, source.
        Empty DataFrame if the CSV contains no data rows.
    """
    reader = csv.DictReader(io.StringIO(raw_csv))
    rows: list[dict[str, Any]] = []
    for row in reader:
        time_period = row.get("TIME_PERIOD", "").strip()
        obs_value = row.get("OBS_VALUE", "").strip()
        if not time_period or not obs_value:
            continue
        try:
            value = float(obs_value)
        except (ValueError, TypeError):
            continue
        rows.append({
            "date":      time_period,
            "value":     value,
            "indicator": series_def.indicator,
            "unit":      series_def.unit,
            "source":    "ecb",
        })

    if not rows:
        return pd.DataFrame(columns=["date", "value", "indicator", "unit", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_series(
    series_def:     ECBSeriesDef,
    start_period:   str = "2000-01-01",
    write_to_db:    bool = True,
) -> ECBFetchResult:
    """Fetch one ECB series, validate it, and optionally write to TimescaleDB.

    Args:
        series_def:   Which series to fetch.
        start_period: ISO date string; fetch data from this date onward.
        write_to_db:  If True (default), write valid data to TimescaleDB.
                      Set False for dry-run / testing.

    Returns:
        An ECBFetchResult describing what was fetched and written.
    """
    logger.info("ecb_fetch_start", indicator=series_def.indicator)
    try:
        url = build_url(series_def, start_period)
        raw = fetch_csv(url)
        df = parse_ecb_csv(raw, series_def)

        if df.empty:
            return ECBFetchResult(
                indicator=series_def.indicator, name=series_def.name,
                rows_fetched=0, rows_written=0,
                start_date=None, end_date=None,
                error="Empty response from ECB API",
            )

        rows_fetched = len(df)

        # ── Validation gate ──────────────────────────────────────────────
        # ECB event-based rates (DFR, MRR) may have <10 rows per year —
        # use min_rows=2.  Monthly/daily series use 10.
        min_rows = 2 if series_def.frequency == "event" else 10
        # ECB monthly data can be 30-60 days stale by design.
        max_stale = 100 if series_def.frequency in ("monthly", "event") else 10

        report = validate_timeseries(
            df,
            source_id      = SOURCE_ID,
            dataset_key    = series_def.indicator,
            required_cols  = ["date", "value"],
            time_col       = "date",
            min_rows       = min_rows,
            max_stale_days = max_stale,
        )
        if not report.passed:
            entry = _quarantine.save(report, df)
            logger.warning(
                "ecb_validation_failed",
                indicator=series_def.indicator,
                errors=len(report.errors),
                quarantine_id=entry.entry_id,
            )
            return ECBFetchResult(
                indicator=series_def.indicator, name=series_def.name,
                rows_fetched=rows_fetched, rows_written=0,
                start_date=None, end_date=None,
                error=f"Validation failed ({len(report.errors)} errors)",
                quarantine_id=entry.entry_id,
            )
        # ── End validation gate ──────────────────────────────────────────

        rows_written = 0
        start_date = df["date"].min().isoformat()
        end_date = df["date"].max().isoformat()

        if write_to_db:
            rows_written = _write_to_timescale(df, series_def)

        logger.info(
            "ecb_fetch_done",
            indicator=series_def.indicator,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
        )
        return ECBFetchResult(
            indicator=series_def.indicator, name=series_def.name,
            rows_fetched=rows_fetched, rows_written=rows_written,
            start_date=start_date, end_date=end_date,
        )

    except Exception as exc:
        logger.error("ecb_fetch_error", indicator=series_def.indicator, error=str(exc))
        return ECBFetchResult(
            indicator=series_def.indicator, name=series_def.name,
            rows_fetched=0, rows_written=0,
            start_date=None, end_date=None, error=str(exc),
        )


# ── Batch runner ─────────────────────────────────────────────────────────────

def run(
    series:         list[ECBSeriesDef] | None = None,
    start_period:   str  = "2000-01-01",
    write_to_db:    bool = True,
) -> list[ECBFetchResult]:
    """Fetch all default ECB series, validate, and write to TimescaleDB.

    Args:
        series:       List of ECBSeriesDef to fetch. Defaults to ECB_SERIES.
        start_period: ISO date string for the start of history.
        write_to_db:  If True (default), write to TimescaleDB.

    Returns:
        List of ECBFetchResult — one per series.
    """
    if series is None:
        series = ECB_SERIES

    results: list[ECBFetchResult] = []
    for sdef in series:
        result = fetch_series(sdef, start_period, write_to_db=write_to_db)
        results.append(result)
        time.sleep(_RATE_DELAY)

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    total_fetched = sum(r.rows_fetched for r in successes)
    total_written = sum(r.rows_written for r in successes)

    logger.info(
        "ecb_run_complete",
        series_ok=len(successes),
        series_total=len(results),
        rows_fetched=total_fetched,
        rows_written=total_written,
        failures=[r.indicator for r in failures],
    )
    return results


# ── Internal: write to TimescaleDB ──────────────────────────────────────────

def _write_to_timescale(df: pd.DataFrame, series_def: ECBSeriesDef) -> int:
    """Convert a validated DataFrame to MacroEventRows and bulk insert.

    Args:
        df:         Validated DataFrame with date, value, indicator, unit, source.
        series_def: The series definition.

    Returns:
        Number of rows written.
    """
    from db.timescale.client import MacroEventRow, bulk_insert_macro

    rows: list[MacroEventRow] = []
    for _, row in df.iterrows():
        dt = row["date"].to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        rows.append(MacroEventRow(
            time=dt,
            indicator=series_def.indicator,
            value=float(row["value"]),
            source="ecb",
            unit=series_def.unit,
            frequency=series_def.frequency,
        ))

    return bulk_insert_macro(rows)
