"""Reusable validation layer for time-series and event datasets.

Every ingest connector should call ``validate_timeseries`` (or the more specific
``validate_ohlcv``) before writing data to any store.  If the report's
``passed`` attribute is False the connector should route the payload to
``QuarantineStore`` rather than the live tables.

Usage::

    from data.validation.validator import validate_ohlcv

    df = fetch_prices("AAPL")
    report = validate_ohlcv(df, source_id="yfinance", symbol="AAPL")
    if not report.passed:
        quarantine.save(report, df)
        return
    # safe to write
    db.bulk_insert_prices(df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

class CheckSeverity(str, Enum):
    """How badly a failed check affects the dataset."""
    ERROR   = "error"    # Data cannot be used; must quarantine
    WARNING = "warning"  # Data is usable but degraded; flag for review
    INFO    = "info"     # Informational; never blocks ingestion


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes:
        name:     Short identifier for the check (e.g. 'null_rate').
        passed:   True if the check passed.
        severity: How bad a failure is.
        message:  Human-readable description of what was checked.
        detail:   Optional structured detail (e.g. null count, bad rows).
    """
    name:     str
    passed:   bool
    severity: CheckSeverity
    message:  str
    detail:   dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationCheck:
    """Definition of a single reusable check (parameterised)."""
    name:        str
    description: str
    severity:    CheckSeverity = CheckSeverity.ERROR


@dataclass
class ValidationReport:
    """Aggregated result of all checks run on one dataset.

    Attributes:
        source_id:   Registry ID of the data source.
        dataset_key: Identifier for the dataset (e.g. symbol + date range).
        checked_at:  ISO-8601 timestamp.
        row_count:   Number of rows in the input.
        checks:      List of individual CheckResult objects.
        passed:      True only if no ERROR-severity check failed.
        schema_hash: Hash of column set (for schema drift detection).
    """
    source_id:   str
    dataset_key: str
    checked_at:  str
    row_count:   int
    checks:      list[CheckResult] = field(default_factory=list)
    passed:      bool = True
    schema_hash: str = ""

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == CheckSeverity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == CheckSeverity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id":   self.source_id,
            "dataset_key": self.dataset_key,
            "checked_at":  self.checked_at,
            "row_count":   self.row_count,
            "passed":      self.passed,
            "schema_hash": self.schema_hash,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": [
                {
                    "name":    c.name,
                    "passed":  c.passed,
                    "severity":c.severity.value,
                    "message": c.message,
                    "detail":  c.detail,
                }
                for c in self.checks
            ],
        }

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.source_id}/{self.dataset_key} "
            f"rows={self.row_count} errors={len(self.errors)} "
            f"warnings={len(self.warnings)}"
        )


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_required_columns(
    df:       pd.DataFrame,
    required: list[str],
) -> CheckResult:
    missing = [c for c in required if c not in df.columns]
    return CheckResult(
        name    = "required_columns",
        passed  = len(missing) == 0,
        severity= CheckSeverity.ERROR,
        message = f"Required columns present: {required}",
        detail  = {"missing": missing},
    )


def _check_null_rate(
    df:           pd.DataFrame,
    max_null_pct: float = 0.05,
    key_columns:  list[str] | None = None,
) -> CheckResult:
    cols = key_columns if key_columns else list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return CheckResult("null_rate", True, CheckSeverity.INFO, "No columns to check.")
    null_rates = {c: float(df[c].isnull().mean()) for c in cols}
    worst      = max(null_rates.values())
    bad_cols   = {k: v for k, v in null_rates.items() if v > max_null_pct}
    return CheckResult(
        name    = "null_rate",
        passed  = len(bad_cols) == 0,
        severity= CheckSeverity.ERROR,
        message = f"Null rate ≤ {max_null_pct*100:.0f}% per column",
        detail  = {"max_null_pct": worst, "bad_columns": bad_cols},
    )


def _check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    n_dup = int(df.duplicated().sum())
    return CheckResult(
        name    = "duplicate_rows",
        passed  = n_dup == 0,
        severity= CheckSeverity.WARNING,
        message = "No duplicate rows",
        detail  = {"duplicate_count": n_dup},
    )


def _check_duplicate_timestamps(
    df:         pd.DataFrame,
    time_col:   str = "date",
) -> CheckResult:
    if time_col not in df.columns:
        return CheckResult("duplicate_timestamps", True, CheckSeverity.INFO,
                           f"Column '{time_col}' not found; skipped.")
    n_dup = int(df[time_col].duplicated().sum())
    return CheckResult(
        name    = "duplicate_timestamps",
        passed  = n_dup == 0,
        severity= CheckSeverity.ERROR,
        message = f"No duplicate values in '{time_col}'",
        detail  = {"duplicate_count": n_dup},
    )


def _check_stale_data(
    df:              pd.DataFrame,
    time_col:        str = "date",
    max_stale_days:  int = 5,
) -> CheckResult:
    if time_col not in df.columns:
        return CheckResult("stale_data", True, CheckSeverity.INFO,
                           f"Column '{time_col}' not found; skipped.")
    try:
        ts = pd.to_datetime(df[time_col])
        latest = ts.max()
        now    = pd.Timestamp.now(tz=latest.tzinfo)
        staleness_days = (now - latest).days
        passed = staleness_days <= max_stale_days
        return CheckResult(
            name    = "stale_data",
            passed  = passed,
            severity= CheckSeverity.WARNING,
            message = f"Latest row within {max_stale_days} calendar days",
            detail  = {
                "latest_date":    str(latest.date()),
                "staleness_days": staleness_days,
            },
        )
    except Exception as exc:
        return CheckResult("stale_data", True, CheckSeverity.INFO,
                           f"Could not parse timestamps: {exc}")


def _check_row_count(
    df:       pd.DataFrame,
    min_rows: int = 10,
) -> CheckResult:
    return CheckResult(
        name    = "row_count",
        passed  = len(df) >= min_rows,
        severity= CheckSeverity.ERROR,
        message = f"At least {min_rows} rows required",
        detail  = {"row_count": len(df), "min_rows": min_rows},
    )


def _check_numeric_range(
    df:        pd.DataFrame,
    col:       str,
    min_val:   float | None = None,
    max_val:   float | None = None,
    max_pct_violations: float = 0.001,
) -> CheckResult:
    if col not in df.columns:
        return CheckResult(f"numeric_range_{col}", True, CheckSeverity.INFO,
                           f"Column '{col}' not found; skipped.")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) == 0:
        return CheckResult(f"numeric_range_{col}", False, CheckSeverity.ERROR,
                           f"Column '{col}' has no numeric values.")
    violations = pd.Series(False, index=series.index)
    if min_val is not None:
        violations |= series < min_val
    if max_val is not None:
        violations |= series > max_val
    pct = float(violations.mean())
    return CheckResult(
        name    = f"numeric_range_{col}",
        passed  = pct <= max_pct_violations,
        severity= CheckSeverity.ERROR,
        message = f"'{col}' values within [{min_val}, {max_val}]",
        detail  = {
            "violation_pct": round(pct * 100, 3),
            "min_seen":      float(series.min()),
            "max_seen":      float(series.max()),
        },
    )


def _check_price_sanity(df: pd.DataFrame) -> CheckResult:
    """OHLCV-specific: high >= low, close within (low, high), volume >= 0."""
    issues = []
    if all(c in df.columns for c in ["high", "low"]):
        bad = (df["high"] < df["low"]).sum()
        if bad > 0:
            issues.append(f"{bad} rows where high < low")
    if all(c in df.columns for c in ["high", "low", "close"]):
        bad = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
        if bad > 0:
            issues.append(f"{bad} rows where close outside [low, high]")
    if "volume" in df.columns:
        bad = (df["volume"] < 0).sum()
        if bad > 0:
            issues.append(f"{bad} rows with negative volume")
    return CheckResult(
        name    = "price_sanity",
        passed  = len(issues) == 0,
        severity= CheckSeverity.ERROR,
        message = "OHLCV price relationships are valid",
        detail  = {"issues": issues},
    )


def _check_schema_drift(
    df:            pd.DataFrame,
    expected_cols: list[str] | None = None,
) -> CheckResult:
    """Detect unexpected addition or removal of columns."""
    current_cols = sorted(df.columns.tolist())
    schema_hash  = str(hash(tuple(current_cols)))
    if expected_cols is None:
        return CheckResult(
            "schema_drift", True, CheckSeverity.INFO,
            "No expected schema provided; hash recorded.",
            detail={"schema_hash": schema_hash, "columns": current_cols},
        )
    expected_sorted = sorted(expected_cols)
    added   = [c for c in current_cols if c not in expected_sorted]
    removed = [c for c in expected_sorted if c not in current_cols]
    passed  = len(added) == 0 and len(removed) == 0
    return CheckResult(
        name    = "schema_drift",
        passed  = passed,
        severity= CheckSeverity.WARNING,
        message = "Column schema matches expected",
        detail  = {
            "schema_hash": schema_hash,
            "added_cols":   added,
            "removed_cols": removed,
        },
    )


def _check_monotonic_timestamps(
    df:       pd.DataFrame,
    time_col: str = "date",
) -> CheckResult:
    """Check that the time column is monotonically increasing."""
    if time_col not in df.columns:
        return CheckResult("monotonic_timestamps", True, CheckSeverity.INFO,
                           f"Column '{time_col}' not found; skipped.")
    ts = pd.to_datetime(df[time_col])
    is_mono = ts.is_monotonic_increasing
    return CheckResult(
        name    = "monotonic_timestamps",
        passed  = bool(is_mono),
        severity= CheckSeverity.WARNING,
        message = f"'{time_col}' is monotonically increasing",
        detail  = {"is_monotonic": bool(is_mono)},
    )


# ── Schema hash helper ────────────────────────────────────────────────────────

def _schema_hash(df: pd.DataFrame) -> str:
    return str(hash(tuple(sorted(df.columns.tolist()))))


# ── High-level validators ─────────────────────────────────────────────────────

def validate_timeseries(
    df:              pd.DataFrame,
    source_id:       str,
    dataset_key:     str,
    required_cols:   list[str] | None = None,
    key_cols_for_nulls: list[str] | None = None,
    time_col:        str       = "date",
    min_rows:        int       = 10,
    max_null_pct:    float     = 0.05,
    max_stale_days:  int       = 7,
    expected_cols:   list[str] | None = None,
) -> ValidationReport:
    """Run a standard suite of checks on any time-series DataFrame.

    Args:
        df:                   Input DataFrame.
        source_id:            Registry ID of the data source.
        dataset_key:          Identifier for the dataset (e.g. 'AAPL_2023').
        required_cols:        Columns that must be present.
        key_cols_for_nulls:   Columns to check null rates on (default: all).
        time_col:             Name of the timestamp column.
        min_rows:             Minimum acceptable row count.
        max_null_pct:         Maximum acceptable null rate per key column.
        max_stale_days:       Maximum acceptable staleness in calendar days.
        expected_cols:        Expected column set for schema-drift detection.

    Returns:
        ValidationReport with all check results and a top-level ``passed`` flag.
    """
    now    = datetime.now(tz=timezone.utc).isoformat()
    checks: list[CheckResult] = []

    if required_cols:
        checks.append(_check_required_columns(df, required_cols))

    checks.append(_check_row_count(df, min_rows=min_rows))
    checks.append(_check_null_rate(df, max_null_pct=max_null_pct, key_columns=key_cols_for_nulls))
    checks.append(_check_duplicate_rows(df))
    checks.append(_check_duplicate_timestamps(df, time_col=time_col))
    checks.append(_check_stale_data(df, time_col=time_col, max_stale_days=max_stale_days))
    checks.append(_check_monotonic_timestamps(df, time_col=time_col))
    checks.append(_check_schema_drift(df, expected_cols=expected_cols))

    passed = all(
        c.passed or c.severity != CheckSeverity.ERROR
        for c in checks
    )

    report = ValidationReport(
        source_id   = source_id,
        dataset_key = dataset_key,
        checked_at  = now,
        row_count   = len(df),
        checks      = checks,
        passed      = passed,
        schema_hash = _schema_hash(df),
    )
    if not passed:
        log.warning("Validation FAILED: %s", report.summary_line())
    else:
        log.debug("Validation PASSED: %s", report.summary_line())
    return report


def validate_ohlcv(
    df:             pd.DataFrame,
    source_id:      str,
    symbol:         str,
    max_null_pct:   float = 0.01,
    max_stale_days: int   = 5,
) -> ValidationReport:
    """Validate an OHLCV DataFrame (specialised for price data).

    Runs all generic time-series checks plus OHLCV-specific sanity:
    high ≥ low, close within (low, high), volume ≥ 0.

    Args:
        df:             DataFrame with columns open, high, low, close, volume.
        source_id:      Registry ID.
        symbol:         Ticker symbol (used as dataset_key).
        max_null_pct:   Max null rate per column (stricter default for price data).
        max_stale_days: Max staleness in calendar days.

    Returns:
        ValidationReport.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    report = validate_timeseries(
        df             = df,
        source_id      = source_id,
        dataset_key    = symbol,
        required_cols  = ohlcv_cols,
        key_cols_for_nulls = ohlcv_cols,
        time_col       = "date",
        min_rows       = 20,
        max_null_pct   = max_null_pct,
        max_stale_days = max_stale_days,
        expected_cols  = ohlcv_cols,
    )

    # Add OHLCV-specific checks
    report.checks.append(_check_price_sanity(df))
    for col in ["open", "high", "low", "close"]:
        report.checks.append(_check_numeric_range(df, col, min_val=0.0))
    report.checks.append(_check_numeric_range(df, "volume", min_val=0.0))

    # Re-evaluate passed after new checks
    report.passed = all(
        c.passed or c.severity != CheckSeverity.ERROR
        for c in report.checks
    )
    return report
