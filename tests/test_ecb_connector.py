"""Tests for data.ingest.ecb_connector — ECB SDMX data connector.

All HTTP I/O and DB writes are mocked so the suite is deterministic.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.ingest.ecb_connector import (
    ECB_SERIES,
    ECBFetchResult,
    ECBSeriesDef,
    build_url,
    fetch_csv,
    fetch_series,
    parse_ecb_csv,
    run,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_EURUSD_DEF = ECBSeriesDef(
    name="EUR/USD Exchange Rate",
    flow_ref="EXR", key="D.USD.EUR.SP00.A",
    frequency="daily", unit="rate", indicator="ECB_EURUSD",
)

_DFR_DEF = ECBSeriesDef(
    name="ECB Deposit Facility Rate",
    flow_ref="FM", key="B.U2.EUR.4F.KR.DFR.LEV",
    frequency="event", unit="percent", indicator="ECB_DFR",
)

_HICP_DEF = ECBSeriesDef(
    name="Euro Area HICP Inflation",
    flow_ref="ICP", key="M.U2.N.000000.4.ANR",
    frequency="monthly", unit="percent", indicator="ECB_HICP",
)

SAMPLE_EXCHANGE_CSV = """\
KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-01,1.0834
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-04,1.0849
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-05,1.0864
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-06,1.0894
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-07,1.0944
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-08,1.0937
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-11,1.0928
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-12,1.0929
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-13,1.0944
EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-03-14,1.0889
"""

SAMPLE_RATE_CSV = """\
KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE
FM.B.U2.EUR.4F.KR.DFR.LEV,B,U2,EUR,4F,KR,DFR,LEV,2024-06-12,3.75
FM.B.U2.EUR.4F.KR.DFR.LEV,B,U2,EUR,4F,KR,DFR,LEV,2024-09-18,3.5
FM.B.U2.EUR.4F.KR.DFR.LEV,B,U2,EUR,4F,KR,DFR,LEV,2024-10-23,3.25
FM.B.U2.EUR.4F.KR.DFR.LEV,B,U2,EUR,4F,KR,DFR,LEV,2024-12-18,3
"""

SAMPLE_INFLATION_CSV = """\
KEY,FREQ,REF_AREA,ADJUSTMENT,ICP_ITEM,STS_INSTITUTION,ICP_SUFFIX,TIME_PERIOD,OBS_VALUE
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-01,2.8
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-02,2.6
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-03,2.4
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-04,2.4
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-05,2.6
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-06,2.5
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-07,2.6
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-08,2.2
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-09,1.7
ICP.M.U2.N.000000.4.ANR,M,U2,N,000000,4,ANR,2024-10,2.0
"""

EMPTY_CSV = "KEY,FREQ,TIME_PERIOD,OBS_VALUE\n"

BAD_VALUES_CSV = """\
KEY,FREQ,TIME_PERIOD,OBS_VALUE
X,D,2024-01-01,not_a_number
X,D,2024-01-02,
X,D,,1.5
"""


# ── build_url tests ──────────────────────────────────────────────────────────

class TestBuildUrl:
    def test_exchange_rate_url(self):
        url = build_url(_EURUSD_DEF, "2024-01-01")
        assert "EXR/D.USD.EUR.SP00.A" in url
        assert "startPeriod=2024-01-01" in url
        assert "format=csvdata" in url
        assert "detail=dataonly" in url

    def test_interest_rate_url(self):
        url = build_url(_DFR_DEF)
        assert "FM/B.U2.EUR.4F.KR.DFR.LEV" in url

    def test_default_start_period(self):
        url = build_url(_EURUSD_DEF)
        assert "startPeriod=2000-01-01" in url


# ── parse_ecb_csv tests ──────────────────────────────────────────────────────

class TestParseEcbCsv:
    def test_exchange_rate_parsing(self):
        df = parse_ecb_csv(SAMPLE_EXCHANGE_CSV, _EURUSD_DEF)
        assert len(df) == 10
        assert list(df.columns) == ["date", "value", "indicator", "unit", "source"]
        assert df["indicator"].iloc[0] == "ECB_EURUSD"
        assert df["unit"].iloc[0] == "rate"
        assert df["source"].iloc[0] == "ecb"
        assert df["value"].dtype == float
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_interest_rate_parsing(self):
        df = parse_ecb_csv(SAMPLE_RATE_CSV, _DFR_DEF)
        assert len(df) == 4
        assert df["indicator"].iloc[0] == "ECB_DFR"
        assert df["value"].iloc[0] == 3.75
        assert df["value"].iloc[-1] == 3.0

    def test_inflation_parsing(self):
        df = parse_ecb_csv(SAMPLE_INFLATION_CSV, _HICP_DEF)
        assert len(df) == 10
        assert df["indicator"].iloc[0] == "ECB_HICP"

    def test_empty_csv(self):
        df = parse_ecb_csv(EMPTY_CSV, _EURUSD_DEF)
        assert df.empty
        assert "date" in df.columns

    def test_bad_values_skipped(self):
        df = parse_ecb_csv(BAD_VALUES_CSV, _EURUSD_DEF)
        # "not_a_number" → skipped, empty value → skipped, empty date → skipped
        assert len(df) == 0

    def test_sorted_by_date(self):
        # Reverse the CSV order
        lines = SAMPLE_EXCHANGE_CSV.strip().splitlines()
        reversed_csv = lines[0] + "\n" + "\n".join(reversed(lines[1:]))
        df = parse_ecb_csv(reversed_csv, _EURUSD_DEF)
        assert df["date"].is_monotonic_increasing

    def test_monthly_date_parsing(self):
        """Monthly dates like '2024-01' should parse correctly."""
        df = parse_ecb_csv(SAMPLE_INFLATION_CSV, _HICP_DEF)
        assert df["date"].iloc[0].year == 2024
        assert df["date"].iloc[0].month == 1


# ── fetch_csv tests ──────────────────────────────────────────────────────────

class TestFetchCsv:
    @patch("data.ingest.ecb_connector.urllib.request.urlopen")
    def test_returns_body(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"header\nrow1\n"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        body = fetch_csv("https://example.com/data")
        assert body == "header\nrow1\n"

    @patch("data.ingest.ecb_connector.urllib.request.urlopen")
    def test_http_error_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://x", 500, "Server Error", {}, None,
        )
        with pytest.raises(ConnectionError, match="HTTP 500"):
            fetch_csv("https://x")

    @patch("data.ingest.ecb_connector.urllib.request.urlopen")
    def test_url_error_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("DNS failed")
        with pytest.raises(ConnectionError, match="network error"):
            fetch_csv("https://x")


# ── fetch_series tests ───────────────────────────────────────────────────────

class TestFetchSeries:
    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_valid_series_no_db(self, mock_csv):
        mock_csv.return_value = SAMPLE_EXCHANGE_CSV
        result = fetch_series(_EURUSD_DEF, write_to_db=False)
        assert result.error is None
        assert result.rows_fetched == 10
        assert result.rows_written == 0  # write_to_db=False
        assert result.indicator == "ECB_EURUSD"
        assert result.start_date is not None
        assert result.end_date is not None

    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_empty_response(self, mock_csv):
        mock_csv.return_value = EMPTY_CSV
        result = fetch_series(_EURUSD_DEF, write_to_db=False)
        assert result.error == "Empty response from ECB API"
        assert result.rows_fetched == 0

    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_connection_error(self, mock_csv):
        mock_csv.side_effect = ConnectionError("timeout")
        result = fetch_series(_EURUSD_DEF, write_to_db=False)
        assert result.error is not None
        assert "timeout" in result.error

    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_interest_rate_event_series(self, mock_csv):
        """Event-frequency series with few rows should still pass validation."""
        mock_csv.return_value = SAMPLE_RATE_CSV
        result = fetch_series(_DFR_DEF, write_to_db=False)
        assert result.error is None
        assert result.rows_fetched == 4
        assert result.indicator == "ECB_DFR"

    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_inflation_monthly(self, mock_csv):
        mock_csv.return_value = SAMPLE_INFLATION_CSV
        result = fetch_series(_HICP_DEF, write_to_db=False)
        assert result.error is None
        assert result.rows_fetched == 10

    @patch("data.ingest.ecb_connector._write_to_timescale")
    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_write_to_db_called(self, mock_csv, mock_write):
        mock_csv.return_value = SAMPLE_EXCHANGE_CSV
        mock_write.return_value = 10
        result = fetch_series(_EURUSD_DEF, write_to_db=True)
        assert result.rows_written == 10
        mock_write.assert_called_once()

    @patch("data.ingest.ecb_connector.fetch_csv")
    def test_validation_failure_quarantines(self, mock_csv):
        """A series with only 1 row should fail validation (min_rows=10 for daily)."""
        one_row_csv = (
            "KEY,FREQ,TIME_PERIOD,OBS_VALUE\n"
            "EXR.D.USD.EUR.SP00.A,D,2024-03-01,1.0834\n"
        )
        mock_csv.return_value = one_row_csv
        result = fetch_series(_EURUSD_DEF, write_to_db=False)
        assert result.error is not None
        assert "Validation failed" in result.error
        assert result.rows_written == 0


# ── run() tests ──────────────────────────────────────────────────────────────

class TestRun:
    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_runs_all_default_series(self, mock_fetch, mock_sleep):
        mock_fetch.return_value = ECBFetchResult(
            indicator="x", name="x", rows_fetched=10,
            rows_written=10, start_date="2024-01-01", end_date="2024-03-01",
        )
        results = run(write_to_db=False)
        assert len(results) == len(ECB_SERIES)
        assert mock_fetch.call_count == len(ECB_SERIES)

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_custom_series_list(self, mock_fetch, mock_sleep):
        mock_fetch.return_value = ECBFetchResult(
            indicator="x", name="x", rows_fetched=5,
            rows_written=0, start_date="2024-01-01", end_date="2024-03-01",
        )
        results = run(series=[_EURUSD_DEF, _DFR_DEF], write_to_db=False)
        assert len(results) == 2

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_partial_failure(self, mock_fetch, mock_sleep):
        ok = ECBFetchResult(
            indicator="ok", name="ok", rows_fetched=10,
            rows_written=10, start_date="2024-01-01", end_date="2024-03-01",
        )
        fail = ECBFetchResult(
            indicator="fail", name="fail", rows_fetched=0,
            rows_written=0, start_date=None, end_date=None, error="boom",
        )
        mock_fetch.side_effect = [ok, fail]
        results = run(series=[_EURUSD_DEF, _DFR_DEF], write_to_db=False)
        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error == "boom"

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_rate_delay_between_fetches(self, mock_fetch, mock_sleep):
        mock_fetch.return_value = ECBFetchResult(
            indicator="x", name="x", rows_fetched=5,
            rows_written=0, start_date=None, end_date=None,
        )
        run(series=[_EURUSD_DEF, _DFR_DEF], write_to_db=False)
        assert mock_sleep.call_count == 2


# ── ECBSeriesDef tests ───────────────────────────────────────────────────────

class TestECBSeriesDef:
    def test_frozen(self):
        with pytest.raises(AttributeError):
            _EURUSD_DEF.name = "changed"

    def test_default_series_count(self):
        assert len(ECB_SERIES) == 8

    def test_all_have_required_fields(self):
        for s in ECB_SERIES:
            assert s.name
            assert s.flow_ref
            assert s.key
            assert s.frequency
            assert s.unit
            assert s.indicator
            assert s.indicator.startswith("ECB_")

    def test_unique_indicators(self):
        indicators = [s.indicator for s in ECB_SERIES]
        assert len(indicators) == len(set(indicators))
