"""Integration tests: validation gate wired into ingest connectors.

All external HTTP calls and DB writes are mocked so tests run fully offline.
The tests focus on the *new* validation/quarantine branch paths rather than
repeating the CRUD tests in test_connectors.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.validation.quarantine import QuarantineStore


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _bdate_index(n: int = 30, days_back: int = 0) -> pd.DatetimeIndex:
    end = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    return pd.bdate_range(end=end, periods=n, tz="UTC")


def _valid_ohlcv_df(n: int = 30) -> pd.DataFrame:
    idx = _bdate_index(n)
    return pd.DataFrame({
        "Open":   100.0 + np.arange(n) * 0.1,
        "High":   101.0 + np.arange(n) * 0.1,
        "Low":    99.0  + np.arange(n) * 0.1,
        "Close":  100.5 + np.arange(n) * 0.1,
        "Volume": 1_000_000.0 + np.arange(n) * 100,
    }, index=idx)


def _bad_ohlcv_df(n: int = 30) -> pd.DataFrame:
    """Return a DataFrame where High < Low on every row (price_sanity ERROR)."""
    df = _valid_ohlcv_df(n)
    df["High"] = df["Low"] - 5.0  # force high < low
    return df


def _tiny_ohlcv_df() -> pd.DataFrame:
    """Return a DataFrame too small to pass the min_rows check (need 20)."""
    return _valid_ohlcv_df(n=3)


# ══════════════════════════════════════════════════════════════════════════════
# yfinance_connector
# ══════════════════════════════════════════════════════════════════════════════

class TestYFinanceValidation:
    """Validation gate in yfinance_connector.fetch_ohlcv."""

    def _make_ticker_mock(self, df: pd.DataFrame) -> MagicMock:
        m = MagicMock()
        m.history.return_value = df
        return m

    @patch("data.ingest.yfinance_connector.bulk_insert_prices", return_value=30)
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_valid_data_writes_to_db(self, mock_ticker, mock_insert, tmp_path):
        """Clean data should pass validation and call bulk_insert_prices."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_ticker.return_value = self._make_ticker_mock(_valid_ohlcv_df())
        result = fetch_ohlcv("AAPL", "equity")

        assert result.error is None
        assert result.quarantine_id is None
        mock_insert.assert_called_once()

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_bad_data_quarantined_not_written(self, mock_ticker, mock_insert, tmp_path):
        """Data with price_sanity errors must be quarantined, not written to DB."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_ticker.return_value = self._make_ticker_mock(_bad_ohlcv_df())
        result = fetch_ohlcv("AAPL", "equity")

        assert result.error is not None
        assert "validation failed" in result.error
        assert result.quarantine_id is not None
        assert result.rows_written == 0
        mock_insert.assert_not_called()

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_bad_data_creates_quarantine_entry(self, mock_ticker, mock_insert, tmp_path):
        """Quarantine entry should be persisted to disk."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_ticker.return_value = self._make_ticker_mock(_bad_ohlcv_df())
        result = fetch_ohlcv("AAPL", "equity")

        assert len(qs) == 1
        entry = qs.list()[0]
        assert entry.source_id == "yfinance"
        assert entry.dataset_key == "AAPL"

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_too_few_rows_quarantined(self, mock_ticker, mock_insert, tmp_path):
        """DataFrames below min_rows threshold must be quarantined."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_ticker.return_value = self._make_ticker_mock(_tiny_ohlcv_df())
        result = fetch_ohlcv("AAPL", "equity")

        assert result.quarantine_id is not None
        mock_insert.assert_not_called()

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_empty_response_returns_error_no_quarantine(self, mock_ticker, mock_insert, tmp_path):
        """Empty yfinance response is handled before validation — no quarantine entry."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_ticker.return_value = self._make_ticker_mock(pd.DataFrame())
        result = fetch_ohlcv("AAPL", "equity")

        assert result.error == "empty response from yfinance"
        assert result.quarantine_id is None
        assert len(qs) == 0


# ══════════════════════════════════════════════════════════════════════════════
# fred_connector
# ══════════════════════════════════════════════════════════════════════════════

class TestFredValidation:
    """Validation gate in fred_connector.fetch_series."""

    def _valid_series(self, n: int = 60) -> pd.Series:
        end = datetime.now(tz=timezone.utc) - timedelta(days=5)
        idx = pd.bdate_range(end=end, periods=n)
        return pd.Series(np.random.uniform(2.0, 5.0, size=len(idx)), index=idx)

    def _tiny_series(self) -> pd.Series:
        idx = pd.bdate_range(end=datetime.now(tz=timezone.utc), periods=2)
        return pd.Series([3.0, 3.1], index=idx)

    def _null_heavy_series(self, n: int = 60) -> pd.Series:
        # dropna() is called before validation, so we can't inject NaNs via series.
        # Instead test the too-few-rows path after dropna via a tiny series.
        return self._tiny_series()

    @patch("data.ingest.fred_connector.bulk_insert_macro", return_value=60)
    @patch("data.ingest.fred_connector.Fred")
    @patch("data.ingest.fred_connector.get_fred_api_key", return_value="test_key")
    def test_valid_series_writes_to_db(self, mock_key, mock_fred_cls, mock_insert, tmp_path):
        """Clean series passes validation and calls bulk_insert_macro."""
        from data.ingest.fred_connector import fetch_series
        import data.ingest.fred_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_fred_cls.return_value.get_series.return_value = self._valid_series()
        result = fetch_series("GS10", "10-Year Treasury", "daily")

        assert result.error is None
        assert result.quarantine_id is None
        mock_insert.assert_called_once()

    @patch("data.ingest.fred_connector.bulk_insert_macro")
    @patch("data.ingest.fred_connector.Fred")
    @patch("data.ingest.fred_connector.get_fred_api_key", return_value="test_key")
    def test_too_few_rows_quarantined(self, mock_key, mock_fred_cls, mock_insert, tmp_path):
        """Series with fewer than min_rows (5) observations must be quarantined."""
        from data.ingest.fred_connector import fetch_series
        import data.ingest.fred_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        # 2 rows < min_rows=5
        mock_fred_cls.return_value.get_series.return_value = self._tiny_series()
        result = fetch_series("GDP", "Gross Domestic Product", "quarterly")

        assert result.error is not None
        assert result.quarantine_id is not None
        mock_insert.assert_not_called()
        assert len(qs) == 1
        assert qs.list()[0].source_id == "fred_api"

    @patch("data.ingest.fred_connector.bulk_insert_macro")
    @patch("data.ingest.fred_connector.Fred")
    @patch("data.ingest.fred_connector.get_fred_api_key", return_value="test_key")
    def test_empty_series_returns_error_no_quarantine(self, mock_key, mock_fred_cls,
                                                       mock_insert, tmp_path):
        """Empty FRED response is handled before validation — no quarantine entry."""
        from data.ingest.fred_connector import fetch_series
        import data.ingest.fred_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_fred_cls.return_value.get_series.return_value = pd.Series([], dtype=float)
        result = fetch_series("GDP", "GDP", "quarterly")

        assert result.error == "empty series from FRED"
        assert result.quarantine_id is None
        assert len(qs) == 0


# ══════════════════════════════════════════════════════════════════════════════
# coingecko_connector
# ══════════════════════════════════════════════════════════════════════════════

class TestCoinGeckoValidation:
    """Validation gate in coingecko_connector.fetch_crypto."""

    def _valid_raw(self, n: int = 30) -> list[list]:
        """Build synthetic CoinGecko OHLC payload [ts_ms, o, h, l, c]."""
        base_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=n)).timestamp() * 1000)
        day_ms  = 86_400_000
        rows    = []
        for i in range(n):
            o = 40000.0 + i * 10
            rows.append([base_ts + i * day_ms, o, o + 500, o - 300, o + 200])
        return rows

    def _bad_raw(self, n: int = 30) -> list[list]:
        """High < Low on every candle — triggers price_sanity ERROR."""
        base_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=n)).timestamp() * 1000)
        day_ms  = 86_400_000
        rows    = []
        for i in range(n):
            o = 40000.0 + i * 10
            rows.append([base_ts + i * day_ms, o, o - 500, o + 300, o + 100])
        return rows

    @patch("data.ingest.coingecko_connector.bulk_insert_prices", return_value=30)
    @patch("data.ingest.coingecko_connector._fetch_ohlc")
    def test_valid_data_writes_to_db(self, mock_fetch, mock_insert, tmp_path):
        """Clean crypto OHLCV passes validation and calls bulk_insert_prices."""
        from data.ingest.coingecko_connector import fetch_crypto
        import data.ingest.coingecko_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_fetch.return_value = self._valid_raw()
        result = fetch_crypto("bitcoin", "BTC")

        assert result.error is None
        assert result.quarantine_id is None
        mock_insert.assert_called_once()

    @patch("data.ingest.coingecko_connector.bulk_insert_prices")
    @patch("data.ingest.coingecko_connector._fetch_ohlc")
    def test_bad_data_quarantined_not_written(self, mock_fetch, mock_insert, tmp_path):
        """Data with price_sanity errors is quarantined, not written to DB."""
        from data.ingest.coingecko_connector import fetch_crypto
        import data.ingest.coingecko_connector as mod
        mod._quarantine = QuarantineStore(directory=tmp_path / "q")

        mock_fetch.return_value = self._bad_raw()
        result = fetch_crypto("bitcoin", "BTC")

        assert result.error is not None
        assert result.quarantine_id is not None
        assert result.rows_written == 0
        mock_insert.assert_not_called()

    @patch("data.ingest.coingecko_connector.bulk_insert_prices")
    @patch("data.ingest.coingecko_connector._fetch_ohlc")
    def test_empty_response_returns_error_no_quarantine(self, mock_fetch, mock_insert, tmp_path):
        """Empty CoinGecko response is handled before validation."""
        from data.ingest.coingecko_connector import fetch_crypto
        import data.ingest.coingecko_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_fetch.return_value = []
        result = fetch_crypto("bitcoin", "BTC")

        assert result.error == "empty response from CoinGecko"
        assert result.quarantine_id is None
        assert len(qs) == 0

    @patch("data.ingest.coingecko_connector.bulk_insert_prices")
    @patch("data.ingest.coingecko_connector._fetch_ohlc")
    def test_quarantine_entry_persisted(self, mock_fetch, mock_insert, tmp_path):
        """Quarantine entry from a failed fetch is saved to disk."""
        from data.ingest.coingecko_connector import fetch_crypto
        import data.ingest.coingecko_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_fetch.return_value = self._bad_raw()
        result = fetch_crypto("bitcoin", "BTC")

        assert len(qs) == 1
        entry = qs.list()[0]
        assert entry.source_id == "coingecko"
        assert entry.dataset_key == "BTC"
        assert not entry.is_resolved


# ══════════════════════════════════════════════════════════════════════════════
# Quarantine interaction across connectors
# ══════════════════════════════════════════════════════════════════════════════

class TestQuarantineInteraction:
    """Confirm the quarantine entries are usable after ingestion."""

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_quarantined_data_can_be_loaded(self, mock_ticker, mock_insert, tmp_path):
        """Data snapshot written during quarantine is readable via QuarantineStore."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        bad_df = _bad_ohlcv_df()
        mock_ticker.return_value = MagicMock()
        mock_ticker.return_value.history.return_value = bad_df

        result = fetch_ohlcv("MSFT", "equity")
        entry  = qs.list()[0]

        df2 = qs.load_data(entry.entry_id)
        assert df2 is not None
        assert len(df2) == len(bad_df)

    @patch("data.ingest.yfinance_connector.bulk_insert_prices")
    @patch("data.ingest.yfinance_connector.yf.Ticker")
    def test_quarantined_report_readable(self, mock_ticker, mock_insert, tmp_path):
        """Validation report saved by quarantine is readable and contains check data."""
        from data.ingest.yfinance_connector import fetch_ohlcv
        import data.ingest.yfinance_connector as mod
        qs = QuarantineStore(directory=tmp_path / "q")
        mod._quarantine = qs

        mock_ticker.return_value = MagicMock()
        mock_ticker.return_value.history.return_value = _bad_ohlcv_df()

        fetch_ohlcv("TSLA", "equity")
        entry  = qs.list()[0]
        report = qs.load_report(entry.entry_id)

        assert report["source_id"] == "yfinance"
        assert report["passed"] is False
        assert len(report["checks"]) > 0
