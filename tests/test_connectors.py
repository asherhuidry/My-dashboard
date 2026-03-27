"""Tests for all four data ingest connectors and the universe module.

All external HTTP calls and DB writes are mocked so tests run offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _mock_supabase_writes() -> tuple[Any, ...]:
    """Return patches for all Supabase write helpers used by connectors."""
    return (
        patch("db.supabase.client.get_client"),
        patch("data.ingest.yfinance_connector.start_agent_run"),
        patch("data.ingest.yfinance_connector.end_agent_run"),
        patch("data.ingest.yfinance_connector.log_evolution"),
    )


def _utc(year: int, month: int, day: int) -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

class TestUniverse:
    """Tests for universe.py definitions."""

    def test_equities_count(self) -> None:
        """EQUITIES contains the expanded universe."""
        from data.ingest.universe import EQUITIES
        assert len(EQUITIES) >= 90  # 96 as of last expansion

    def test_crypto_count(self) -> None:
        """CRYPTO contains top 20 (coin_id, ticker) pairs."""
        from data.ingest.universe import CRYPTO
        assert len(CRYPTO) == 20

    def test_forex_count(self) -> None:
        """FOREX contains major + emerging pairs."""
        from data.ingest.universe import FOREX
        assert len(FOREX) >= 10

    def test_commodities_count(self) -> None:
        """COMMODITIES contains futures + spot tickers."""
        from data.ingest.universe import COMMODITIES
        assert len(COMMODITIES) >= 10

    def test_macro_series_count(self) -> None:
        """MACRO_SERIES contains 58+ FRED series."""
        from data.ingest.universe import MACRO_SERIES
        assert len(MACRO_SERIES) >= 50

    def test_get_yfinance_universe_keys(self) -> None:
        """get_yfinance_universe returns all asset classes."""
        from data.ingest.universe import (
            COMMODITIES, EQUITIES, FOREX, ETFS, CRYPTO_YF,
            get_yfinance_universe,
        )
        universe = get_yfinance_universe(extended=True)
        expected = len(EQUITIES) + len(CRYPTO_YF) + len(ETFS) + len(FOREX) + len(COMMODITIES)
        assert len(universe) == expected

    def test_yfinance_universe_asset_classes(self) -> None:
        """Every value in get_yfinance_universe is a valid asset class."""
        from data.ingest.universe import get_yfinance_universe
        valid = {"equity", "crypto", "etf", "forex", "commodity"}
        for ticker, asset_class in get_yfinance_universe().items():
            assert asset_class in valid, f"{ticker} has invalid asset_class: {asset_class}"

    def test_crypto_tickers_unique(self) -> None:
        """All crypto tickers in CRYPTO are unique."""
        from data.ingest.universe import CRYPTO
        tickers = [t for _, t in CRYPTO]
        assert len(tickers) == len(set(tickers))


# ---------------------------------------------------------------------------
# yfinance connector
# ---------------------------------------------------------------------------

class TestYFinanceConnector:
    """Tests for yfinance_connector.py."""

    def _make_mock_df(self, n: int = 30) -> pd.DataFrame:
        """Return a realistic OHLCV DataFrame as yfinance would return."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        base = 150.0
        return pd.DataFrame({
            "Open":   [base + i * 0.5 for i in range(n)],
            "High":   [base + i * 0.5 + 3.0 for i in range(n)],
            "Low":    [base + i * 0.5 - 1.0 for i in range(n)],
            "Close":  [base + i * 0.5 + 1.0 for i in range(n)],
            "Volume": [1e6 + i * 1e4 for i in range(n)],
        }, index=dates)

    def test_fetch_ohlcv_success(self) -> None:
        """fetch_ohlcv returns a result with correct row count on success."""
        from data.ingest.yfinance_connector import fetch_ohlcv

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_mock_df(30)

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("data.ingest.yfinance_connector.bulk_insert_prices", return_value=30):
            result = fetch_ohlcv("AAPL", "equity", period="2y")

        assert result.rows_written == 30
        assert result.error is None
        assert result.ticker == "AAPL"

    def test_fetch_ohlcv_empty_response(self) -> None:
        """fetch_ohlcv returns error result when yfinance returns empty DataFrame."""
        from data.ingest.yfinance_connector import fetch_ohlcv

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_ohlcv("FAKE", "equity")

        assert result.rows_written == 0
        assert result.error is not None
        assert "empty" in result.error

    def test_fetch_ohlcv_exception_returns_error(self) -> None:
        """fetch_ohlcv catches exceptions and returns an error result."""
        from data.ingest.yfinance_connector import fetch_ohlcv

        with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
            result = fetch_ohlcv("AAPL", "equity")

        assert result.error == "network error"
        assert result.rows_written == 0

    def test_fetch_ohlcv_drops_nan_rows(self) -> None:
        """fetch_ohlcv drops rows with NaN close before inserting."""
        from data.ingest.yfinance_connector import fetch_ohlcv

        df = self._make_mock_df(30)
        df.loc[df.index[2], "Close"] = float("nan")

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df

        captured: list = []

        def capture_rows(rows):
            captured.extend(rows)
            return len(rows)

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("data.ingest.yfinance_connector.bulk_insert_prices", side_effect=capture_rows):
            result = fetch_ohlcv("AAPL", "equity")

        # NaN row should be dropped — 29 of 30 rows written
        assert result.rows_written == 29

    def test_run_logs_to_supabase(self) -> None:
        """run() calls start_agent_run, end_agent_run, and log_evolution."""
        from data.ingest.yfinance_connector import run

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._make_mock_df()

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("data.ingest.yfinance_connector.bulk_insert_prices", return_value=30), \
             patch("data.ingest.yfinance_connector.start_agent_run") as mock_start, \
             patch("data.ingest.yfinance_connector.end_agent_run") as mock_end, \
             patch("data.ingest.yfinance_connector.log_evolution") as mock_log, \
             patch("time.sleep"):
            run({"AAPL": "equity"})

        mock_start.assert_called_once()
        mock_end.assert_called_once()
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# CoinGecko connector
# ---------------------------------------------------------------------------

class TestCoinGeckoConnector:
    """Tests for coingecko_connector.py."""

    def _mock_ohlc_response(self, n: int = 30) -> list[list]:
        """Return mock CoinGecko OHLC data with enough rows to pass validation."""
        base_ms = int(_utc(2024, 1, 1).timestamp() * 1000)
        return [
            [base_ms + i * 86400000, 40000 + i * 10, 41000 + i * 10, 39000 + i * 10, 40500 + i * 10]
            for i in range(n)
        ]

    def test_fetch_crypto_success(self) -> None:
        """fetch_crypto returns correct row count on success."""
        from data.ingest.coingecko_connector import fetch_crypto

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_ohlc_response()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.coingecko_connector.bulk_insert_prices", return_value=30):
            result = fetch_crypto("bitcoin", "BTC", days=730)

        assert result.rows_written == 30
        assert result.error is None
        assert result.ticker == "BTC"

    def test_fetch_crypto_empty_response(self) -> None:
        """fetch_crypto returns error result for empty API response."""
        from data.ingest.coingecko_connector import fetch_crypto

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            result = fetch_crypto("bitcoin", "BTC")

        assert result.error is not None
        assert result.rows_written == 0

    def test_fetch_crypto_http_error(self) -> None:
        """fetch_crypto catches HTTP errors and returns error result."""
        from data.ingest.coingecko_connector import fetch_crypto
        import httpx

        with patch("httpx.get", side_effect=httpx.RequestError("timeout")):
            result = fetch_crypto("bitcoin", "BTC")

        assert result.error is not None

    def test_rows_have_crypto_asset_class(self) -> None:
        """fetch_crypto writes rows with asset_class='crypto'."""
        from data.ingest.coingecko_connector import fetch_crypto

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_ohlc_response()
        mock_response.raise_for_status = MagicMock()

        captured: list = []

        def capture(rows):
            captured.extend(rows)
            return len(rows)

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.coingecko_connector.bulk_insert_prices", side_effect=capture):
            fetch_crypto("bitcoin", "BTC")

        assert all(r.asset_class == "crypto" for r in captured)
        assert all(r.source == "coingecko" for r in captured)

    def test_run_logs_to_supabase(self) -> None:
        """run() calls start_agent_run, end_agent_run, and log_evolution."""
        from data.ingest.coingecko_connector import run

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_ohlc_response()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.coingecko_connector.bulk_insert_prices", return_value=30), \
             patch("data.ingest.coingecko_connector.start_agent_run") as mock_start, \
             patch("data.ingest.coingecko_connector.end_agent_run") as mock_end, \
             patch("data.ingest.coingecko_connector.log_evolution") as mock_log, \
             patch("time.sleep"):
            run([("bitcoin", "BTC")])

        mock_start.assert_called_once()
        mock_end.assert_called_once()
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# FRED connector
# ---------------------------------------------------------------------------

class TestFredConnector:
    """Tests for fred_connector.py."""

    def _mock_series(self, n: int = 30) -> pd.Series:
        """Return a realistic FRED series as fredapi would return it."""
        dates = pd.date_range("2020-01-01", periods=n, freq="QS")
        return pd.Series([21500.0 + i * 100 for i in range(n)], index=dates)

    def test_fetch_series_success(self) -> None:
        """fetch_series returns correct row count on success."""
        from data.ingest.fred_connector import fetch_series

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = self._mock_series()

        with patch("data.ingest.fred_connector._get_fred_client", return_value=mock_fred), \
             patch("data.ingest.fred_connector.bulk_insert_macro", return_value=30):
            result = fetch_series("GDP", "Gross Domestic Product", "quarterly")

        assert result.rows_written == 30
        assert result.error is None
        assert result.series_id == "GDP"

    def test_fetch_series_empty(self) -> None:
        """fetch_series returns error result for empty series."""
        from data.ingest.fred_connector import fetch_series

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series([], dtype=float)

        with patch("data.ingest.fred_connector._get_fred_client", return_value=mock_fred):
            result = fetch_series("GDP", "GDP", "quarterly")

        assert result.error is not None
        assert result.rows_written == 0

    def test_fetch_series_drops_nan(self) -> None:
        """fetch_series drops NaN observations before writing."""
        from data.ingest.fred_connector import fetch_series
        import numpy as np

        series = self._mock_series()
        series.iloc[2] = float("nan")

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = series

        captured: list = []

        def capture(rows):
            captured.extend(rows)
            return len(rows)

        with patch("data.ingest.fred_connector._get_fred_client", return_value=mock_fred), \
             patch("data.ingest.fred_connector.bulk_insert_macro", side_effect=capture):
            result = fetch_series("GDP", "GDP", "quarterly")

        assert result.rows_written == 29  # one NaN dropped

    def test_rows_have_fred_source(self) -> None:
        """fetch_series writes rows with source='fred'."""
        from data.ingest.fred_connector import fetch_series

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = self._mock_series()
        captured: list = []

        with patch("data.ingest.fred_connector._get_fred_client", return_value=mock_fred), \
             patch("data.ingest.fred_connector.bulk_insert_macro",
                   side_effect=lambda rows: captured.extend(rows) or len(rows)):
            fetch_series("GDP", "GDP", "quarterly")

        assert all(r.source == "fred" for r in captured)

    def test_run_logs_to_supabase(self) -> None:
        """run() calls start_agent_run, end_agent_run, and log_evolution."""
        from data.ingest.fred_connector import run

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = self._mock_series()

        with patch("data.ingest.fred_connector._get_fred_client", return_value=mock_fred), \
             patch("data.ingest.fred_connector.bulk_insert_macro", return_value=30), \
             patch("data.ingest.fred_connector.start_agent_run") as mock_start, \
             patch("data.ingest.fred_connector.end_agent_run") as mock_end, \
             patch("data.ingest.fred_connector.log_evolution") as mock_log, \
             patch("time.sleep"):
            run([("GDP", "GDP", "quarterly")])

        mock_start.assert_called_once()
        mock_end.assert_called_once()
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Alpha Vantage connector
# ---------------------------------------------------------------------------

class TestAlphaVantageConnector:
    """Tests for alpha_vantage_connector.py."""

    def _mock_av_response(self, n: int = 30) -> dict:
        """Return a realistic Alpha Vantage TIME_SERIES_DAILY_ADJUSTED response."""
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        series = {}
        for i, dt in enumerate(dates):
            key = dt.strftime("%Y-%m-%d")
            base = 180.0 + i * 0.5
            series[key] = {
                "1. open": str(base),
                "2. high": str(base + 3.0),
                "3. low": str(base - 1.0),
                "4. close": str(base + 1.0),
                "5. adjusted close": str(base + 1.0),
                "6. volume": str(int(5e7 + i * 1e5)),
            }
        return {
            "Meta Data": {"2. Symbol": "AAPL"},
            "Time Series (Daily)": series,
        }

    def test_fetch_equity_success(self) -> None:
        """fetch_equity returns correct row count on success."""
        from data.ingest.alpha_vantage_connector import fetch_equity

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_av_response()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.bulk_insert_prices", return_value=30), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"):
            result = fetch_equity("AAPL")

        assert result.rows_written == 30
        assert result.error is None

    def test_fetch_equity_error_message_in_response(self) -> None:
        """fetch_equity returns error result when response contains Error Message."""
        from data.ingest.alpha_vantage_connector import fetch_equity

        mock_response = MagicMock()
        mock_response.json.return_value = {"Error Message": "Invalid API call."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"):
            result = fetch_equity("FAKE")

        assert result.error is not None
        assert result.rows_written == 0

    def test_fetch_equity_rate_limit_note(self) -> None:
        """fetch_equity returns error result when response contains Note (rate limit)."""
        from data.ingest.alpha_vantage_connector import fetch_equity

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is..."
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"):
            result = fetch_equity("AAPL")

        assert result.error is not None

    def test_fetch_equity_skips_zero_close(self) -> None:
        """fetch_equity skips rows where close price is zero."""
        from data.ingest.alpha_vantage_connector import fetch_equity

        data = self._mock_av_response(31)
        # Overwrite one date entry with zero-close data
        first_key = next(iter(data["Time Series (Daily)"]))
        data["Time Series (Daily)"][first_key] = {
            "1. open": "0", "2. high": "0", "3. low": "0",
            "4. close": "0", "5. adjusted close": "0", "6. volume": "0",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = data
        mock_response.raise_for_status = MagicMock()

        captured: list = []

        def capture(rows):
            captured.extend(rows)
            return len(rows)

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.bulk_insert_prices",
                   side_effect=capture), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"):
            result = fetch_equity("AAPL")

        # Zero-close row must be skipped
        assert all(r.close > 0 for r in captured)

    def test_rows_have_alpha_vantage_source(self) -> None:
        """fetch_equity writes rows with source='alpha_vantage'."""
        from data.ingest.alpha_vantage_connector import fetch_equity

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_av_response()
        mock_response.raise_for_status = MagicMock()

        captured: list = []

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.bulk_insert_prices",
                   side_effect=lambda rows: captured.extend(rows) or len(rows)), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"):
            fetch_equity("AAPL")

        assert all(r.source == "alpha_vantage" for r in captured)

    def test_run_logs_to_supabase(self) -> None:
        """run() calls start_agent_run, end_agent_run, and log_evolution."""
        from data.ingest.alpha_vantage_connector import run

        mock_response = MagicMock()
        mock_response.json.return_value = self._mock_av_response()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response), \
             patch("data.ingest.alpha_vantage_connector.bulk_insert_prices", return_value=30), \
             patch("data.ingest.alpha_vantage_connector.get_alpha_vantage_key",
                   return_value="test_key"), \
             patch("data.ingest.alpha_vantage_connector.start_agent_run") as mock_start, \
             patch("data.ingest.alpha_vantage_connector.end_agent_run") as mock_end, \
             patch("data.ingest.alpha_vantage_connector.log_evolution") as mock_log, \
             patch("time.sleep"):
            run(["AAPL"])

        mock_start.assert_called_once()
        mock_end.assert_called_once()
        mock_log.assert_called_once()
