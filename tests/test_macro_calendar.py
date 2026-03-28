"""Tests for the macro calendar event generation."""
from __future__ import annotations

import pytest

from data.agents.macro_calendar import (
    generate_events,
    EVENT_DEFINITIONS,
    FOMC_2026_DATES,
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
