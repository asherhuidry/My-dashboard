"""Tests for data.validation.validator and data.validation.quarantine."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.validation.validator import (
    CheckSeverity,
    CheckResult,
    ValidationReport,
    validate_timeseries,
    validate_ohlcv,
    _check_required_columns,
    _check_null_rate,
    _check_duplicate_rows,
    _check_duplicate_timestamps,
    _check_stale_data,
    _check_row_count,
    _check_numeric_range,
    _check_price_sanity,
    _check_schema_drift,
    _check_monotonic_timestamps,
)
from data.validation.quarantine import QuarantineStore, QuarantineEntry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 30,
    days_back: int = 0,
    include_bad_price: bool = False,
    null_col: str | None = None,
) -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame."""
    end   = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    dates = pd.date_range(end=end, periods=n, freq="B")
    base  = 100.0
    data  = {
        "date":   dates,
        "open":   [base + i * 0.1 for i in range(n)],
        "high":   [base + i * 0.1 + 1.0 for i in range(n)],
        "low":    [base + i * 0.1 - 0.5 for i in range(n)],
        "close":  [base + i * 0.1 + 0.3 for i in range(n)],
        "volume": [1_000_000 + i * 1000 for i in range(n)],
    }
    df = pd.DataFrame(data)
    if include_bad_price:
        df.loc[0, "high"] = df.loc[0, "low"] - 1  # high < low
    if null_col:
        df.loc[:5, null_col] = np.nan
    return df


def _make_timeseries(n: int = 20, days_back: int = 0) -> pd.DataFrame:
    end   = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    dates = pd.date_range(end=end, periods=n, freq="B")
    return pd.DataFrame({"date": dates, "value": range(n)})


# ── Individual checks ─────────────────────────────────────────────────────────

class TestCheckRequiredColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        r  = _check_required_columns(df, ["a", "b"])
        assert r.passed

    def test_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        r  = _check_required_columns(df, ["a", "b"])
        assert not r.passed
        assert "b" in r.detail["missing"]

    def test_severity_is_error(self):
        df = pd.DataFrame({"a": [1]})
        r  = _check_required_columns(df, ["z"])
        assert r.severity == CheckSeverity.ERROR


class TestCheckNullRate:
    def test_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _check_null_rate(df).passed

    def test_high_null_rate_fails(self):
        df = pd.DataFrame({"a": [None, None, None, None, 1.0]})
        r  = _check_null_rate(df, max_null_pct=0.5)
        assert not r.passed

    def test_below_threshold_passes(self):
        df = pd.DataFrame({"a": [None] + [1.0] * 99})
        r  = _check_null_rate(df, max_null_pct=0.05)
        assert r.passed  # 1% nulls < 5% threshold

    def test_key_columns_subset(self):
        # Only check column 'a'; 'b' has lots of nulls but is not key
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [None, None, None]})
        r  = _check_null_rate(df, max_null_pct=0.05, key_columns=["a"])
        assert r.passed


class TestCheckDuplicateRows:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _check_duplicate_rows(df).passed

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        r  = _check_duplicate_rows(df)
        assert not r.passed
        assert r.detail["duplicate_count"] == 1

    def test_severity_is_warning(self):
        df = pd.DataFrame({"a": [1, 1]})
        assert _check_duplicate_rows(df).severity == CheckSeverity.WARNING


class TestCheckDuplicateTimestamps:
    def test_no_duplicates(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
        assert _check_duplicate_timestamps(df).passed

    def test_with_duplicates(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-02"]})
        r  = _check_duplicate_timestamps(df)
        assert not r.passed
        assert r.detail["duplicate_count"] == 1

    def test_missing_col_returns_info(self):
        df = pd.DataFrame({"value": [1, 2]})
        r  = _check_duplicate_timestamps(df, time_col="date")
        assert r.severity == CheckSeverity.INFO
        assert r.passed


class TestCheckStaleData:
    def test_fresh_data_passes(self):
        df = _make_timeseries(days_back=0)
        assert _check_stale_data(df, max_stale_days=7).passed

    def test_stale_data_fails(self):
        df = _make_timeseries(days_back=30)
        r  = _check_stale_data(df, max_stale_days=7)
        assert not r.passed

    def test_missing_col_returns_info(self):
        df = pd.DataFrame({"value": [1, 2]})
        r  = _check_stale_data(df, time_col="date")
        assert r.severity == CheckSeverity.INFO


class TestCheckRowCount:
    def test_enough_rows(self):
        df = pd.DataFrame({"a": range(20)})
        assert _check_row_count(df, min_rows=10).passed

    def test_too_few_rows(self):
        df = pd.DataFrame({"a": range(5)})
        r  = _check_row_count(df, min_rows=10)
        assert not r.passed
        assert r.severity == CheckSeverity.ERROR


class TestCheckNumericRange:
    def test_all_within_range(self):
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        assert _check_numeric_range(df, "price", min_val=0.0).passed

    def test_violations_below_threshold(self):
        # 0 out of 1000 is fine even with strict threshold
        df = pd.DataFrame({"price": [1.0] * 1000})
        assert _check_numeric_range(df, "price", min_val=0.0, max_pct_violations=0.001).passed

    def test_too_many_violations(self):
        df = pd.DataFrame({"price": [-1.0, -2.0] + [1.0] * 8})
        r  = _check_numeric_range(df, "price", min_val=0.0, max_pct_violations=0.001)
        assert not r.passed

    def test_missing_col_returns_info(self):
        df = pd.DataFrame({"other": [1, 2]})
        r  = _check_numeric_range(df, "price", min_val=0.0)
        assert r.severity == CheckSeverity.INFO
        assert r.passed


class TestCheckPriceSanity:
    def test_valid_ohlcv(self):
        df = _make_ohlcv()
        assert _check_price_sanity(df).passed

    def test_high_less_than_low(self):
        df = _make_ohlcv()
        df.loc[0, "high"] = df.loc[0, "low"] - 1
        r  = _check_price_sanity(df)
        assert not r.passed
        assert any("high < low" in issue for issue in r.detail["issues"])

    def test_close_outside_high_low(self):
        df = _make_ohlcv()
        df.loc[0, "close"] = df.loc[0, "high"] + 100
        r  = _check_price_sanity(df)
        assert not r.passed

    def test_negative_volume(self):
        df = _make_ohlcv()
        df.loc[0, "volume"] = -1
        r  = _check_price_sanity(df)
        assert not r.passed


class TestCheckSchemaDrift:
    def test_no_expected_schema(self):
        df = pd.DataFrame({"a": [1]})
        r  = _check_schema_drift(df, expected_cols=None)
        assert r.passed
        assert r.severity == CheckSeverity.INFO

    def test_matching_schema(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        r  = _check_schema_drift(df, expected_cols=["a", "b"])
        assert r.passed

    def test_added_column(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        r  = _check_schema_drift(df, expected_cols=["a", "b"])
        assert not r.passed
        assert "c" in r.detail["added_cols"]

    def test_removed_column(self):
        df = pd.DataFrame({"a": [1]})
        r  = _check_schema_drift(df, expected_cols=["a", "b"])
        assert not r.passed
        assert "b" in r.detail["removed_cols"]


class TestCheckMonotonicTimestamps:
    def test_monotonic(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
        assert _check_monotonic_timestamps(df).passed

    def test_non_monotonic(self):
        df = pd.DataFrame({"date": ["2024-01-03", "2024-01-01", "2024-01-02"]})
        r  = _check_monotonic_timestamps(df)
        assert not r.passed

    def test_missing_col_returns_info(self):
        df = pd.DataFrame({"value": [1]})
        r  = _check_monotonic_timestamps(df, time_col="date")
        assert r.severity == CheckSeverity.INFO


# ── ValidationReport ──────────────────────────────────────────────────────────

class TestValidationReport:
    def _make_report(self) -> ValidationReport:
        return ValidationReport(
            source_id   = "test",
            dataset_key = "KEY",
            checked_at  = datetime.now(tz=timezone.utc).isoformat(),
            row_count   = 10,
        )

    def test_passed_initially(self):
        r = self._make_report()
        assert r.passed

    def test_errors_property(self):
        r = self._make_report()
        r.checks.append(CheckResult("c1", False, CheckSeverity.ERROR, "bad"))
        r.checks.append(CheckResult("c2", False, CheckSeverity.WARNING, "warn"))
        assert len(r.errors) == 1
        assert r.errors[0].name == "c1"

    def test_warnings_property(self):
        r = self._make_report()
        r.checks.append(CheckResult("c1", False, CheckSeverity.WARNING, "warn"))
        assert len(r.warnings) == 1

    def test_to_dict_serializable(self):
        r = self._make_report()
        r.checks.append(CheckResult("c1", True, CheckSeverity.INFO, "ok"))
        d = r.to_dict()
        assert json.dumps(d)  # must be JSON-serializable
        assert d["source_id"] == "test"
        assert len(d["checks"]) == 1

    def test_summary_line_pass(self):
        r = self._make_report()
        assert r.summary_line().startswith("[PASS]")

    def test_summary_line_fail(self):
        r = self._make_report()
        r.passed = False
        assert r.summary_line().startswith("[FAIL]")


# ── High-level validators ─────────────────────────────────────────────────────

class TestValidateTimeseries:
    def test_valid_data_passes(self):
        df = _make_timeseries()
        r  = validate_timeseries(df, "src", "key", required_cols=["date", "value"])
        assert r.passed

    def test_missing_required_col_fails(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20)})
        r  = validate_timeseries(df, "src", "key", required_cols=["date", "value"])
        assert not r.passed
        assert any(c.name == "required_columns" and not c.passed for c in r.checks)

    def test_too_few_rows_fails(self):
        df = _make_timeseries(n=3)
        r  = validate_timeseries(df, "src", "key", min_rows=10)
        assert not r.passed

    def test_stale_is_warning_not_error(self):
        df = _make_timeseries(days_back=30)
        r  = validate_timeseries(df, "src", "key", max_stale_days=7, min_rows=1)
        stale = next(c for c in r.checks if c.name == "stale_data")
        assert stale.severity == CheckSeverity.WARNING
        # Stale alone should not fail report (WARNING only)
        error_checks = [c for c in r.checks if not c.passed and c.severity == CheckSeverity.ERROR]
        assert len(error_checks) == 0

    def test_report_fields_populated(self):
        df = _make_timeseries()
        r  = validate_timeseries(df, "fred_api", "DGS10")
        assert r.source_id == "fred_api"
        assert r.dataset_key == "DGS10"
        assert r.row_count == len(df)
        assert r.schema_hash != ""
        assert r.checked_at != ""


class TestValidateOhlcv:
    def test_valid_ohlcv_passes(self):
        df = _make_ohlcv()
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        assert r.passed

    def test_missing_ohlcv_col_fails(self):
        df = _make_ohlcv().drop(columns=["volume"])
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        assert not r.passed

    def test_price_sanity_check_present(self):
        df = _make_ohlcv()
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        assert any(c.name == "price_sanity" for c in r.checks)

    def test_bad_price_fails(self):
        df = _make_ohlcv(include_bad_price=True)
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        assert not r.passed

    def test_numeric_range_checks_present(self):
        df = _make_ohlcv()
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        names = {c.name for c in r.checks}
        assert "numeric_range_open" in names
        assert "numeric_range_close" in names

    def test_high_null_rate_fails(self):
        df = _make_ohlcv(null_col="close")
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        assert not r.passed

    def test_stale_data_is_warning(self):
        df = _make_ohlcv(days_back=20)
        r  = validate_ohlcv(df, "yfinance", "AAPL")
        stale = next(c for c in r.checks if c.name == "stale_data")
        assert stale.severity == CheckSeverity.WARNING


# ── QuarantineEntry ───────────────────────────────────────────────────────────

class TestQuarantineEntry:
    def _make_entry(self) -> QuarantineEntry:
        return QuarantineEntry(
            entry_id       = "abc12345",
            source_id      = "yfinance",
            dataset_key    = "AAPL",
            reason         = "high null rate",
            error_count    = 1,
            warning_count  = 0,
            row_count      = 30,
            quarantined_at = datetime.now(tz=timezone.utc).isoformat(),
        )

    def test_is_resolved_false_by_default(self):
        e = self._make_entry()
        assert not e.is_resolved

    def test_is_resolved_true_when_set(self):
        e = self._make_entry()
        e.resolved_at = datetime.now(tz=timezone.utc).isoformat()
        assert e.is_resolved

    def test_to_dict_round_trip(self):
        e  = self._make_entry()
        d  = e.to_dict()
        e2 = QuarantineEntry.from_dict(d)
        assert e2.entry_id == e.entry_id
        assert e2.source_id == e.source_id
        assert e2.resolved_at is None


# ── QuarantineStore ───────────────────────────────────────────────────────────

@pytest.fixture()
def qs(tmp_path: Path) -> QuarantineStore:
    """Fresh QuarantineStore backed by a temp directory."""
    return QuarantineStore(directory=tmp_path / "quarantine")


def _valid_report(source_id: str = "yfinance", dataset_key: str = "AAPL") -> ValidationReport:
    """Build a failed ValidationReport for quarantine testing."""
    return ValidationReport(
        source_id   = source_id,
        dataset_key = dataset_key,
        checked_at  = datetime.now(tz=timezone.utc).isoformat(),
        row_count   = 5,
        checks      = [
            CheckResult("null_rate", False, CheckSeverity.ERROR, "Too many nulls",
                        detail={"max_null_pct": 0.8, "bad_columns": {"close": 0.8}})
        ],
        passed      = False,
    )


class TestQuarantineStoreSave:
    def test_save_creates_entry(self, qs):
        report = _valid_report()
        entry  = qs.save(report, _make_ohlcv())
        assert entry.entry_id != ""
        assert entry.source_id == "yfinance"
        assert not entry.is_resolved

    def test_save_writes_report_json(self, qs, tmp_path):
        report = _valid_report()
        entry  = qs.save(report)
        assert Path(entry.report_path).exists()
        d = json.loads(Path(entry.report_path).read_text())
        assert d["source_id"] == "yfinance"

    def test_save_writes_data_csv(self, qs):
        df     = _make_ohlcv()
        report = _valid_report()
        entry  = qs.save(report, df)
        assert entry.data_path != ""
        assert Path(entry.data_path).exists()

    def test_save_without_df_has_no_data_path(self, qs):
        entry = qs.save(_valid_report())
        assert entry.data_path == ""

    def test_save_updates_index(self, qs):
        qs.save(_valid_report())
        qs.save(_valid_report(dataset_key="MSFT"))
        assert len(qs) == 2

    def test_save_index_persists_to_disk(self, tmp_path):
        store = QuarantineStore(directory=tmp_path / "q")
        store.save(_valid_report())
        store2 = QuarantineStore(directory=tmp_path / "q")
        assert len(store2) == 1

    def test_custom_reason(self, qs):
        entry = qs.save(_valid_report(), reason="manual override")
        assert entry.reason == "manual override"

    def test_auto_reason_from_errors(self, qs):
        entry = qs.save(_valid_report())
        assert "null" in entry.reason.lower() or entry.reason != ""


class TestQuarantineStoreResolve:
    def test_resolve_marks_entry(self, qs):
        entry = qs.save(_valid_report())
        qs.resolve(entry.entry_id)
        assert qs.list(resolved=True)[0].is_resolved

    def test_resolve_updates_index(self, qs):
        entry = qs.save(_valid_report())
        qs.resolve(entry.entry_id)
        assert qs.list(resolved=False) == []

    def test_resolve_unknown_raises(self, qs):
        with pytest.raises(KeyError):
            qs.resolve("nonexistent")

    def test_resolve_with_notes_appends_to_report(self, qs):
        entry = qs.save(_valid_report())
        qs.resolve(entry.entry_id, notes="Fixed upstream")
        d = json.loads(Path(entry.report_path).read_text())
        assert d.get("resolution_notes") == "Fixed upstream"


class TestQuarantineStoreLoad:
    def test_load_report(self, qs):
        entry = qs.save(_valid_report())
        d     = qs.load_report(entry.entry_id)
        assert d["source_id"] == "yfinance"

    def test_load_report_missing_raises(self, qs):
        qs.save(_valid_report())  # create a real entry
        # Corrupt the path
        entry = qs._index[0]
        entry.report_path = "/nonexistent/path/report.json"
        with pytest.raises(FileNotFoundError):
            qs.load_report(entry.entry_id)

    def test_load_data_returns_dataframe(self, qs):
        df    = _make_ohlcv()
        entry = qs.save(_valid_report(), df)
        df2   = qs.load_data(entry.entry_id)
        assert df2 is not None
        assert len(df2) == len(df)

    def test_load_data_none_when_no_snapshot(self, qs):
        entry = qs.save(_valid_report())
        assert qs.load_data(entry.entry_id) is None


class TestQuarantineStoreList:
    def _populate(self, qs: QuarantineStore) -> None:
        qs.save(_valid_report("yfinance", "AAPL"))
        qs.save(_valid_report("yfinance", "MSFT"))
        qs.save(_valid_report("fred_api", "DGS10"))
        entry = qs.save(_valid_report("fred_api", "DGS2"))
        qs.resolve(entry.entry_id)

    def test_list_all(self, qs):
        self._populate(qs)
        assert len(qs.list()) == 4

    def test_list_by_source(self, qs):
        self._populate(qs)
        assert len(qs.list(source_id="yfinance")) == 2

    def test_list_unresolved(self, qs):
        self._populate(qs)
        assert len(qs.list(resolved=False)) == 3

    def test_list_resolved(self, qs):
        self._populate(qs)
        assert len(qs.list(resolved=True)) == 1

    def test_list_by_dataset_key(self, qs):
        self._populate(qs)
        assert len(qs.list(dataset_key="AAPL")) == 1

    def test_list_sorted_most_recent_first(self, qs):
        self._populate(qs)
        entries = qs.list()
        times   = [e.quarantined_at for e in entries]
        assert times == sorted(times, reverse=True)


class TestQuarantineStoreSummary:
    def test_summary_counts(self, qs):
        qs.save(_valid_report("yfinance", "AAPL"))
        qs.save(_valid_report("yfinance", "MSFT"))
        entry = qs.save(_valid_report("fred_api", "DGS10"))
        qs.resolve(entry.entry_id)

        s = qs.summary()
        assert s["total"]      == 3
        assert s["unresolved"] == 2
        assert s["resolved"]   == 1
        assert s["by_source"]["yfinance"] == 2
        assert s["by_source"]["fred_api"] == 1

    def test_summary_empty(self, qs):
        s = qs.summary()
        assert s["total"] == 0
        assert s["unresolved"] == 0


class TestQuarantineStoreLen:
    def test_len(self, qs):
        assert len(qs) == 0
        qs.save(_valid_report())
        assert len(qs) == 1
        qs.save(_valid_report(dataset_key="X"))
        assert len(qs) == 2
