"""Tests for the macro calendar event generation."""
from __future__ import annotations

import pytest

from data.agents.macro_calendar import (
    generate_events,
    EVENT_DEFINITIONS,
    FOMC_2026_DATES,
    CPI_2026_DATES,
    NFP_2026_DATES,
)


class TestGenerateEvents:
    """Tests for event generation."""

    def test_returns_list(self) -> None:
        events = generate_events(2026)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_event_has_required_fields(self) -> None:
        events = generate_events(2026)
        required = {"event_id", "event_type", "label", "event_date", "impact", "description", "related_series"}
        for ev in events:
            assert required.issubset(ev.keys()), f"Missing fields in {ev['event_id']}"

    def test_event_ids_are_unique(self) -> None:
        events = generate_events(2026)
        ids = [e["event_id"] for e in events]
        assert len(ids) == len(set(ids))

    def test_fomc_count_matches(self) -> None:
        """Should generate one FOMC event per scheduled date."""
        events = generate_events(2026)
        fomc = [e for e in events if e["event_type"] == "FOMC"]
        assert len(fomc) == len(FOMC_2026_DATES)

    def test_monthly_events_have_12(self) -> None:
        """Monthly event types should have ~12 events."""
        events = generate_events(2026)
        cpi = [e for e in events if e["event_type"] == "CPI"]
        assert len(cpi) == 12

    def test_quarterly_events_have_4(self) -> None:
        """Quarterly event types should have 4 events."""
        events = generate_events(2026)
        gdp = [e for e in events if e["event_type"] == "GDP"]
        assert len(gdp) == 4

    def test_related_series_not_empty(self) -> None:
        events = generate_events(2026)
        for ev in events:
            assert len(ev["related_series"]) > 0

    def test_impact_is_valid(self) -> None:
        events = generate_events(2026)
        for ev in events:
            assert ev["impact"] in ("high", "moderate", "low")

    def test_all_event_types_present(self) -> None:
        events = generate_events(2026)
        types = {e["event_type"] for e in events}
        expected = {d["event_type"] for d in EVENT_DEFINITIONS}
        assert types == expected

    def test_event_id_format(self) -> None:
        """Event IDs should be TYPE_YYYY-MM-DD."""
        events = generate_events(2026)
        for ev in events:
            assert ev["event_id"].startswith(ev["event_type"] + "_")
            # Date is always the last 10 chars
            date_part = ev["event_id"][-10:]
            assert date_part.startswith("2026-")

    def test_fomc_dates_match_fed_schedule(self) -> None:
        """FOMC dates should match the Federal Reserve published schedule."""
        events = generate_events(2026)
        fomc = sorted(e["event_date"] for e in events if e["event_type"] == "FOMC")
        # Verified against federalreserve.gov/monetarypolicy/fomccalendars.htm
        assert fomc[0] == "2026-01-28"
        assert fomc[2] == "2026-04-29"  # was incorrectly May 6
        assert fomc[6] == "2026-10-28"  # was incorrectly Nov 4
        assert fomc[7] == "2026-12-09"  # was incorrectly Dec 16

    def test_cpi_uses_exact_2026_dates(self) -> None:
        """CPI events for 2026 should use verified BLS dates, not approximations."""
        events = generate_events(2026)
        cpi_dates = sorted(e["event_date"] for e in events if e["event_type"] == "CPI")
        assert cpi_dates[0] == "2026-01-13"
        assert cpi_dates[1] == "2026-02-11"  # approximate was Feb 13
        assert cpi_dates[5] == "2026-06-10"  # approximate was Jun 13

    def test_nfp_uses_exact_2026_dates(self) -> None:
        """NFP events for 2026 should use verified BLS dates."""
        events = generate_events(2026)
        nfp_dates = sorted(e["event_date"] for e in events if e["event_type"] == "NFP")
        assert nfp_dates[0] == "2026-01-09"
        assert nfp_dates[3] == "2026-04-03"  # approximate was Apr 5

    def test_pce_count_for_2026(self) -> None:
        """PCE uses hardcoded 2026 dates — may have >12 releases."""
        events = generate_events(2026)
        pce = [e for e in events if e["event_type"] == "PCE"]
        assert len(pce) >= 12
