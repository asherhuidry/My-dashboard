"""Tests for the Finnhub earnings calendar connector.

All tests mock external calls (Finnhub API, Neo4j, Supabase).
"""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_API_RESPONSE = {
    "earningsCalendar": [
        {
            "date": "2026-04-01",
            "epsActual": 1.52,
            "epsEstimate": 1.45,
            "hour": "amc",
            "quarter": 1,
            "revenueActual": 94500000000,
            "revenueEstimate": 93200000000,
            "symbol": "AAPL",
            "year": 2026,
        },
        {
            "date": "2026-04-02",
            "epsActual": None,
            "epsEstimate": 2.10,
            "hour": "bmo",
            "quarter": 1,
            "revenueActual": None,
            "revenueEstimate": 61000000000,
            "symbol": "MSFT",
            "year": 2026,
        },
        {
            "date": "2026-04-01",
            "epsActual": 0.88,
            "epsEstimate": 0.90,
            "hour": "amc",
            "quarter": 1,
            "revenueActual": 5200000000,
            "revenueEstimate": 5400000000,
            "symbol": "UNKNOWN_TICKER",
            "year": 2026,
        },
    ],
}


def _mock_urlopen(response_data: dict) -> MagicMock:
    """Create a mock urllib response context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _mock_neo4j_driver() -> MagicMock:
    """Return a mock Neo4j Driver."""
    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver


# ---------------------------------------------------------------------------
# EarningsEvent dataclass tests
# ---------------------------------------------------------------------------

class TestEarningsEvent:
    """Test the EarningsEvent dataclass and its computed properties."""

    def test_eps_surprise_positive(self) -> None:
        """Positive surprise when actual > estimate."""
        from data.ingest.finnhub_connector import EarningsEvent
        event = EarningsEvent(
            symbol="AAPL", date=date(2026, 4, 1),
            eps_actual=1.52, eps_estimate=1.45,
        )
        assert event.eps_surprise == 0.07
        assert event.eps_surprise_pct is not None
        assert event.eps_surprise_pct > 0

    def test_eps_surprise_negative(self) -> None:
        """Negative surprise when actual < estimate."""
        from data.ingest.finnhub_connector import EarningsEvent
        event = EarningsEvent(
            symbol="TEST", date=date(2026, 4, 1),
            eps_actual=0.88, eps_estimate=0.90,
        )
        assert event.eps_surprise is not None
        assert event.eps_surprise < 0

    def test_eps_surprise_none_when_no_actual(self) -> None:
        """Surprise is None when actual is not yet reported."""
        from data.ingest.finnhub_connector import EarningsEvent
        event = EarningsEvent(
            symbol="MSFT", date=date(2026, 4, 2),
            eps_actual=None, eps_estimate=2.10,
        )
        assert event.eps_surprise is None
        assert event.eps_surprise_pct is None

    def test_has_actuals(self) -> None:
        """has_actuals reflects whether eps_actual is set."""
        from data.ingest.finnhub_connector import EarningsEvent
        reported = EarningsEvent(
            symbol="AAPL", date=date(2026, 4, 1), eps_actual=1.52,
        )
        pending = EarningsEvent(
            symbol="MSFT", date=date(2026, 4, 2), eps_actual=None,
        )
        assert reported.has_actuals is True
        assert pending.has_actuals is False


# ---------------------------------------------------------------------------
# API fetch tests
# ---------------------------------------------------------------------------

class TestFetchEarnings:
    """Test the fetch_earnings() function with mocked HTTP."""

    @patch("data.ingest.finnhub_connector.get_finnhub_api_key", return_value="test_key")
    @patch("data.ingest.finnhub_connector.urllib.request.urlopen")
    def test_parses_api_response(
        self, mock_urlopen: MagicMock, mock_key: MagicMock,
    ) -> None:
        """Correctly parses Finnhub JSON into EarningsEvent objects."""
        mock_urlopen.return_value = _mock_urlopen(SAMPLE_API_RESPONSE)

        from data.ingest.finnhub_connector import fetch_earnings
        events = fetch_earnings("2026-04-01", "2026-04-07")

        assert len(events) == 3
        aapl = [e for e in events if e.symbol == "AAPL"][0]
        assert aapl.eps_actual == 1.52
        assert aapl.eps_estimate == 1.45
        assert aapl.quarter == 1
        assert aapl.year == 2026
        assert aapl.hour == "amc"

    @patch("data.ingest.finnhub_connector.get_finnhub_api_key", return_value="test_key")
    @patch("data.ingest.finnhub_connector.urllib.request.urlopen")
    def test_handles_empty_response(
        self, mock_urlopen: MagicMock, mock_key: MagicMock,
    ) -> None:
        """Returns empty list when API has no events."""
        mock_urlopen.return_value = _mock_urlopen({"earningsCalendar": []})

        from data.ingest.finnhub_connector import fetch_earnings
        events = fetch_earnings("2026-04-01", "2026-04-07")
        assert events == []

    @patch("data.ingest.finnhub_connector.get_finnhub_api_key", return_value="test_key")
    @patch("data.ingest.finnhub_connector.urllib.request.urlopen")
    def test_skips_malformed_entries(
        self, mock_urlopen: MagicMock, mock_key: MagicMock,
    ) -> None:
        """Skips entries that are missing required fields."""
        bad_data = {"earningsCalendar": [
            {"symbol": "AAPL"},  # no date
            {"date": "2026-04-01", "symbol": "MSFT", "epsActual": 1.0},  # valid
        ]}
        mock_urlopen.return_value = _mock_urlopen(bad_data)

        from data.ingest.finnhub_connector import fetch_earnings
        events = fetch_earnings("2026-04-01", "2026-04-07")
        assert len(events) == 1
        assert events[0].symbol == "MSFT"


# ---------------------------------------------------------------------------
# Graph write tests
# ---------------------------------------------------------------------------

class TestWriteEarningsToGraph:
    """Test the write_earnings_to_graph() function with mocked Neo4j."""

    @patch("db.neo4j.client.get_driver")
    def test_writes_events_to_neo4j(self, mock_get_driver: MagicMock) -> None:
        """Creates Event nodes and REPORTS edges in Neo4j."""
        driver = _mock_neo4j_driver()
        mock_get_driver.return_value = driver

        from data.ingest.finnhub_connector import EarningsEvent, write_earnings_to_graph
        events = [
            EarningsEvent(symbol="AAPL", date=date(2026, 4, 1), eps_actual=1.52),
            EarningsEvent(symbol="MSFT", date=date(2026, 4, 2), eps_estimate=2.10),
        ]
        written = write_earnings_to_graph(events)

        assert written == 2
        session = driver.session.return_value
        assert session.run.call_count == 2

    @patch("db.neo4j.client.get_driver")
    def test_filters_by_universe(self, mock_get_driver: MagicMock) -> None:
        """Only writes events for symbols in the universe set."""
        driver = _mock_neo4j_driver()
        mock_get_driver.return_value = driver

        from data.ingest.finnhub_connector import EarningsEvent, write_earnings_to_graph
        events = [
            EarningsEvent(symbol="AAPL", date=date(2026, 4, 1)),
            EarningsEvent(symbol="OBSCURE_TICKER", date=date(2026, 4, 1)),
        ]
        written = write_earnings_to_graph(events, universe_symbols={"AAPL", "MSFT"})

        assert written == 1

    @patch("db.neo4j.client.get_driver")
    def test_cypher_contains_reports_edge(self, mock_get_driver: MagicMock) -> None:
        """REPORTS edge is created linking Asset to EarningsEvent."""
        driver = _mock_neo4j_driver()
        mock_get_driver.return_value = driver

        from data.ingest.finnhub_connector import EarningsEvent, write_earnings_to_graph
        events = [EarningsEvent(symbol="AAPL", date=date(2026, 4, 1))]
        write_earnings_to_graph(events)

        session = driver.session.return_value
        cypher = str(session.run.call_args_list[0])
        assert "REPORTS" in cypher
        assert "EarningsEvent" in cypher

    @patch("db.neo4j.client.get_driver", side_effect=Exception("no neo4j"))
    def test_handles_neo4j_unavailable(self, mock_get_driver: MagicMock) -> None:
        """Returns 0 when Neo4j is unavailable."""
        from data.ingest.finnhub_connector import EarningsEvent, write_earnings_to_graph
        events = [EarningsEvent(symbol="AAPL", date=date(2026, 4, 1))]
        written = write_earnings_to_graph(events)
        assert written == 0


# ---------------------------------------------------------------------------
# Run function tests
# ---------------------------------------------------------------------------

class TestRun:
    """Test the run() function end-to-end with mocks."""

    @patch("data.ingest.finnhub_connector.get_finnhub_api_key", return_value="test_key")
    @patch("data.ingest.finnhub_connector.urllib.request.urlopen")
    def test_run_returns_result(
        self, mock_urlopen: MagicMock, mock_key: MagicMock,
    ) -> None:
        """run() returns a FinnhubFetchResult with correct counts."""
        mock_urlopen.return_value = _mock_urlopen(SAMPLE_API_RESPONSE)

        from data.ingest.finnhub_connector import run
        result = run(write_to_graph=False)

        assert result.error is None
        assert result.events_fetched == 3
        assert len(result.symbols) == 3
