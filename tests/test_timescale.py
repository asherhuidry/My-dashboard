"""Tests for the TimescaleDB client module.

All tests mock the Supabase REST client so they run without live credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year: int, month: int, day: int) -> datetime:
    """Return a timezone-aware UTC datetime for the given date."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _mock_supabase_client() -> MagicMock:
    """Return a mock Supabase client whose table().upsert().execute() does nothing."""
    mock_execute = MagicMock()
    mock_execute.data = []

    mock_query = MagicMock()
    mock_query.execute.return_value = mock_execute
    mock_query.eq.return_value = mock_query
    mock_query.gte.return_value = mock_query
    mock_query.lte.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.select.return_value = mock_query
    mock_query.upsert.return_value = mock_query

    mock_table = MagicMock()
    mock_table.upsert.return_value = mock_query
    mock_table.select.return_value = mock_query

    mock_client = MagicMock()
    mock_client.table.return_value = mock_table
    return mock_client


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

    def test_to_dict_serialises_time_as_iso(self) -> None:
        """PriceRow.to_dict() returns time as an ISO string."""
        from db.timescale.client import PriceRow
        row = PriceRow(
            time=_utc(2024, 1, 15), asset="AAPL", asset_class="equity",
            open=180.0, high=185.0, low=179.0, close=182.0
        )
        d = row.to_dict()
        assert isinstance(d["time"], str)
        assert "2024-01-15" in d["time"]


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

    def test_to_dict_contains_all_fields(self) -> None:
        """VolumeRow.to_dict() contains all expected keys."""
        from db.timescale.client import VolumeRow
        row = VolumeRow(time=_utc(2024, 1, 15), asset="ETH",
                        buy_vol=1000.0, sell_vol=800.0)
        d = row.to_dict()
        assert set(d.keys()) == {"time", "asset", "exchange", "buy_vol", "sell_vol"}


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

    def test_to_dict_contains_all_fields(self) -> None:
        """MacroEventRow.to_dict() contains all expected keys."""
        from db.timescale.client import MacroEventRow
        row = MacroEventRow(time=_utc(2024, 3, 28), indicator="GDP", value=2.4)
        d = row.to_dict()
        assert "time" in d and "indicator" in d and "value" in d


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
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            count = bulk_insert_prices(rows)

        assert count == 3

    def test_empty_list_returns_zero(self) -> None:
        """bulk_insert_prices returns 0 for an empty list without hitting DB."""
        from db.timescale.client import bulk_insert_prices
        with patch("db.timescale.client._get_client") as mock_get:
            count = bulk_insert_prices([])
        mock_get.assert_not_called()
        assert count == 0

    def test_calls_upsert_on_prices_table(self) -> None:
        """bulk_insert_prices calls upsert on the 'prices' table."""
        from db.timescale.client import PriceRow, bulk_insert_prices

        rows = [PriceRow(time=_utc(2024, 1, 1), asset="TSLA", asset_class="equity",
                         open=200.0, high=210.0, low=198.0, close=205.0)]
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            bulk_insert_prices(rows)

        mock_client.table.assert_called_with("prices")

    def test_batches_large_inserts(self) -> None:
        """bulk_insert_prices splits 600 rows into 2 batches of 500 and 100."""
        from db.timescale.client import PriceRow, bulk_insert_prices

        rows = [
            PriceRow(time=_utc(2024, 1, 1), asset="AAPL", asset_class="equity",
                     open=180.0, high=185.0, low=179.0, close=182.0)
            for _ in range(600)
        ]
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            count = bulk_insert_prices(rows)

        assert count == 600
        # 2 upsert calls: batch of 500 + batch of 100
        assert mock_client.table().upsert.call_count == 2


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
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
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
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            count = bulk_insert_macro(rows)

        assert count == 2

    def test_empty_list_returns_zero(self) -> None:
        """bulk_insert_macro returns 0 for an empty list."""
        from db.timescale.client import bulk_insert_macro
        assert bulk_insert_macro([]) == 0

    def test_calls_upsert_on_macro_events_table(self) -> None:
        """bulk_insert_macro calls upsert on the 'macro_events' table."""
        from db.timescale.client import MacroEventRow, bulk_insert_macro

        rows = [MacroEventRow(time=_utc(2024, 1, 1), indicator="CPI", value=3.2)]
        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            bulk_insert_macro(rows)

        mock_client.table.assert_called_with("macro_events")


# ---------------------------------------------------------------------------
# fetch_prices / fetch_macro
# ---------------------------------------------------------------------------

class TestFetchPrices:
    """Tests for fetch_prices()."""

    def test_returns_rows_from_client(self) -> None:
        """fetch_prices returns whatever the Supabase client returns."""
        from db.timescale.client import fetch_prices

        fake_rows = [{"time": "2024-01-01T00:00:00+00:00", "asset": "AAPL", "close": 182.0}]
        mock_client = _mock_supabase_client()
        mock_client.table().select().eq().eq().gte().lte().order().execute.return_value.data = fake_rows

        with patch("db.timescale.client._get_client", return_value=mock_client):
            rows = fetch_prices("AAPL", _utc(2024, 1, 1), _utc(2024, 1, 31))

        assert isinstance(rows, list)

    def test_filters_by_asset(self) -> None:
        """fetch_prices passes asset to .eq() filter."""
        from db.timescale.client import fetch_prices

        mock_client = _mock_supabase_client()
        with patch("db.timescale.client._get_client", return_value=mock_client):
            fetch_prices("MSFT", _utc(2024, 1, 1), _utc(2024, 1, 31))

        eq_calls = [str(c) for c in mock_client.table().select().eq.call_args_list]
        assert any("MSFT" in str(c) for c in mock_client.table().select().eq.call_args_list)


class TestFetchMacro:
    """Tests for fetch_macro()."""

    def test_returns_rows_from_client(self) -> None:
        """fetch_macro returns whatever the Supabase client returns."""
        from db.timescale.client import fetch_macro

        mock_client = _mock_supabase_client()
        mock_client.table().select().eq().gte().lte().order().execute.return_value.data = []

        with patch("db.timescale.client._get_client", return_value=mock_client):
            rows = fetch_macro("CPI", _utc(2023, 1, 1), _utc(2024, 1, 1))

        assert isinstance(rows, list)
