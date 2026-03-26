"""Tests for the TimescaleDB client module.

All tests mock psycopg2 so they run without a live database.
Integration tests require TIMESCALE_DSN to be set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year: int, month: int, day: int) -> datetime:
    """Return a timezone-aware UTC datetime for the given date."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _mock_conn() -> MagicMock:
    """Return a mock psycopg2 connection with a cursor that does nothing."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = []
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.closed = False
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


# ---------------------------------------------------------------------------
# PriceRow dataclass
# ---------------------------------------------------------------------------

class TestPriceRow:
    """Tests for the PriceRow dataclass validation."""

    def test_naive_datetime_gets_utc(self) -> None:
        """PriceRow converts naive datetime to UTC-aware."""
        from db.timescale.client import PriceRow
        row = PriceRow(
            time=datetime(2024, 1, 15),
            asset="AAPL", asset_class="equity",
            open=180.0, high=185.0, low=179.0, close=182.0
        )
        assert row.time.tzinfo == timezone.utc

    def test_negative_close_raises(self) -> None:
        """PriceRow raises ValueError for negative close price."""
        from db.timescale.client import PriceRow
        with pytest.raises(ValueError, match="Negative close"):
            PriceRow(
                time=_utc(2024, 1, 15),
                asset="AAPL", asset_class="equity",
                open=180.0, high=185.0, low=179.0, close=-1.0
            )

    def test_defaults(self) -> None:
        """PriceRow source defaults to 'yfinance' and volume defaults to 0."""
        from db.timescale.client import PriceRow
        row = PriceRow(
            time=_utc(2024, 1, 15), asset="BTC", asset_class="crypto",
            open=40000.0, high=41000.0, low=39000.0, close=40500.0
        )
        assert row.source == "yfinance"
        assert row.volume == 0.0


# ---------------------------------------------------------------------------
# VolumeRow dataclass
# ---------------------------------------------------------------------------

class TestVolumeRow:
    """Tests for the VolumeRow dataclass."""

    def test_naive_datetime_gets_utc(self) -> None:
        """VolumeRow converts naive datetime to UTC-aware."""
        from db.timescale.client import VolumeRow
        row = VolumeRow(time=datetime(2024, 1, 15), asset="ETH",
                        buy_vol=1000.0, sell_vol=800.0)
        assert row.time.tzinfo == timezone.utc

    def test_exchange_defaults_to_aggregate(self) -> None:
        """VolumeRow exchange defaults to 'aggregate'."""
        from db.timescale.client import VolumeRow
        row = VolumeRow(time=_utc(2024, 1, 15), asset="ETH",
                        buy_vol=1000.0, sell_vol=800.0)
        assert row.exchange == "aggregate"


# ---------------------------------------------------------------------------
# MacroEventRow dataclass
# ---------------------------------------------------------------------------

class TestMacroEventRow:
    """Tests for the MacroEventRow dataclass."""

    def test_naive_datetime_gets_utc(self) -> None:
        """MacroEventRow converts naive datetime to UTC-aware."""
        from db.timescale.client import MacroEventRow
        row = MacroEventRow(time=datetime(2024, 3, 28),
                            indicator="GDP", value=2.4)
        assert row.time.tzinfo == timezone.utc

    def test_source_defaults_to_fred(self) -> None:
        """MacroEventRow source defaults to 'fred'."""
        from db.timescale.client import MacroEventRow
        row = MacroEventRow(time=_utc(2024, 3, 28), indicator="CPI", value=3.2)
        assert row.source == "fred"


# ---------------------------------------------------------------------------
# bulk_insert_prices
# ---------------------------------------------------------------------------

class TestBulkInsertPrices:
    """Tests for bulk_insert_prices()."""

    def test_returns_row_count(self) -> None:
        """bulk_insert_prices returns the number of rows submitted."""
        from db.timescale.client import PriceRow, bulk_insert_prices

        rows = [
            PriceRow(time=_utc(2024, 1, i), asset="AAPL", asset_class="equity",
                     open=180.0, high=185.0, low=179.0, close=182.0)
            for i in range(1, 4)
        ]
        mock_conn = _mock_conn()
        with patch("db.timescale.client.get_connection", return_value=mock_conn), \
             patch("psycopg2.extras.execute_batch"):
            count = bulk_insert_prices(rows)

        assert count == 3

    def test_empty_list_returns_zero(self) -> None:
        """bulk_insert_prices returns 0 for an empty list without hitting DB."""
        from db.timescale.client import bulk_insert_prices
        with patch("db.timescale.client.get_connection") as mock_get:
            count = bulk_insert_prices([])
        mock_get.assert_not_called()
        assert count == 0

    def test_calls_execute_batch(self) -> None:
        """bulk_insert_prices calls psycopg2.extras.execute_batch."""
        from db.timescale.client import PriceRow, bulk_insert_prices

        rows = [PriceRow(time=_utc(2024, 1, 1), asset="TSLA", asset_class="equity",
                         open=200.0, high=210.0, low=198.0, close=205.0)]
        mock_conn = _mock_conn()
        with patch("db.timescale.client.get_connection", return_value=mock_conn), \
             patch("psycopg2.extras.execute_batch") as mock_eb:
            bulk_insert_prices(rows)

        mock_eb.assert_called_once()


# ---------------------------------------------------------------------------
# bulk_insert_volume
# ---------------------------------------------------------------------------

class TestBulkInsertVolume:
    """Tests for bulk_insert_volume()."""

    def test_returns_row_count(self) -> None:
        """bulk_insert_volume returns the number of rows submitted."""
        from db.timescale.client import VolumeRow, bulk_insert_volume

        rows = [VolumeRow(time=_utc(2024, 1, 1), asset="BTC",
                          buy_vol=500.0, sell_vol=400.0)]
        mock_conn = _mock_conn()
        with patch("db.timescale.client.get_connection", return_value=mock_conn), \
             patch("psycopg2.extras.execute_batch"):
            count = bulk_insert_volume(rows)

        assert count == 1

    def test_empty_list_returns_zero(self) -> None:
        """bulk_insert_volume returns 0 for an empty list."""
        from db.timescale.client import bulk_insert_volume
        assert bulk_insert_volume([]) == 0


# ---------------------------------------------------------------------------
# bulk_insert_macro
# ---------------------------------------------------------------------------

class TestBulkInsertMacro:
    """Tests for bulk_insert_macro()."""

    def test_returns_row_count(self) -> None:
        """bulk_insert_macro returns the number of rows submitted."""
        from db.timescale.client import MacroEventRow, bulk_insert_macro

        rows = [
            MacroEventRow(time=_utc(2024, 1, 1), indicator="GDP", value=2.4),
            MacroEventRow(time=_utc(2024, 4, 1), indicator="GDP", value=2.8),
        ]
        mock_conn = _mock_conn()
        with patch("db.timescale.client.get_connection", return_value=mock_conn), \
             patch("psycopg2.extras.execute_batch"):
            count = bulk_insert_macro(rows)

        assert count == 2

    def test_empty_list_returns_zero(self) -> None:
        """bulk_insert_macro returns 0 for an empty list."""
        from db.timescale.client import bulk_insert_macro
        assert bulk_insert_macro([]) == 0


# ---------------------------------------------------------------------------
# fetch_prices / fetch_macro
# ---------------------------------------------------------------------------

class TestFetchPrices:
    """Tests for fetch_prices()."""

    def test_returns_rows_from_cursor(self) -> None:
        """fetch_prices returns whatever the cursor returns."""
        from db.timescale.client import fetch_prices

        fake_rows = [{"time": _utc(2024, 1, 1), "asset": "AAPL", "close": 182.0}]
        mock_conn = _mock_conn()
        mock_conn.cursor.return_value.fetchall.return_value = fake_rows

        with patch("db.timescale.client.get_connection", return_value=mock_conn):
            rows = fetch_prices("AAPL", _utc(2024, 1, 1), _utc(2024, 1, 31))

        assert rows == fake_rows

    def test_executes_sql_with_correct_params(self) -> None:
        """fetch_prices passes asset and date range to cursor.execute."""
        from db.timescale.client import fetch_prices

        mock_conn = _mock_conn()
        mock_cur = mock_conn.cursor.return_value
        mock_cur.fetchall.return_value = []

        start = _utc(2024, 1, 1)
        end = _utc(2024, 1, 31)

        with patch("db.timescale.client.get_connection", return_value=mock_conn):
            fetch_prices("MSFT", start, end)

        execute_call = mock_cur.execute.call_args
        params = execute_call[0][1]
        assert params[0] == "MSFT"
        assert params[2] == start
        assert params[3] == end


class TestFetchMacro:
    """Tests for fetch_macro()."""

    def test_returns_rows_from_cursor(self) -> None:
        """fetch_macro returns whatever the cursor returns."""
        from db.timescale.client import fetch_macro

        fake_rows = [{"indicator": "CPI", "value": 3.2}]
        mock_conn = _mock_conn()
        mock_conn.cursor.return_value.fetchall.return_value = fake_rows

        with patch("db.timescale.client.get_connection", return_value=mock_conn):
            rows = fetch_macro("CPI", _utc(2023, 1, 1), _utc(2024, 1, 1))

        assert rows == fake_rows


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnection:
    """Tests for get_connection()."""

    def test_raises_for_https_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_connection raises RuntimeError if SUPABASE_URL is an https:// URL."""
        import db.timescale.client as mod
        monkeypatch.delenv("TIMESCALE_DSN", raising=False)
        monkeypatch.setenv("SUPABASE_URL", "https://xyz.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "fake-key")
        mod._conn = None

        with pytest.raises(RuntimeError, match="TIMESCALE_DSN"):
            mod.get_connection()

    def test_reuses_open_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_connection returns the cached connection if it is still open."""
        import db.timescale.client as mod

        mock_conn = _mock_conn()
        mod._conn = mock_conn

        with patch("psycopg2.connect") as mock_connect:
            conn = mod.get_connection()

        mock_connect.assert_not_called()
        assert conn is mock_conn
