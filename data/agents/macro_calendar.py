"""Macro calendar — scheduled economic event nodes for the knowledge graph.

Creates Event nodes for recurring, high-impact macro releases (FOMC, CPI,
NFP, GDP) and links them to the MacroIndicator nodes they relate to via
TRIGGERED_BY edges.  This gives the graph a temporal dimension: agents and
the UI can query "what events are coming up that affect indicators X, Y?"

Events are deterministic (dates derived from known schedules) so this
module needs no external API — it generates events from rules.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Any

from skills.logger import get_logger

log = get_logger(__name__)

AGENT_ID = "macro_calendar"


# ─────────────────────────────────────────────────────────────────────────────
# Event definitions — recurring macro releases
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (event_type, label, related_fred_series, typical_impact)
EVENT_DEFINITIONS: list[dict[str, Any]] = [
    {
        "event_type": "FOMC",
        "label": "FOMC Rate Decision",
        "related_series": ["FEDFUNDS", "DFF"],
        "impact": "high",
        "frequency": "6_weeks",
        "description": "Federal Open Market Committee interest rate decision",
    },
    {
        "event_type": "CPI",
        "label": "CPI Release",
        "related_series": ["CPIAUCSL", "CPILFESL"],
        "impact": "high",
        "frequency": "monthly",
        "description": "Consumer Price Index monthly release",
    },
    {
        "event_type": "NFP",
        "label": "Non-Farm Payrolls",
        "related_series": ["UNRATE", "PAYEMS"],
        "impact": "high",
        "frequency": "monthly",
        "description": "Bureau of Labor Statistics employment report",
    },
    {
        "event_type": "GDP",
        "label": "GDP Release",
        "related_series": ["GDP", "GDPC1"],
        "impact": "high",
        "frequency": "quarterly",
        "description": "Bureau of Economic Analysis GDP estimate",
    },
    {
        "event_type": "PCE",
        "label": "PCE Price Index",
        "related_series": ["PCEPI", "PCEPILFE"],
        "impact": "high",
        "frequency": "monthly",
        "description": "Personal Consumption Expenditures price data (Fed preferred inflation)",
    },
    {
        "event_type": "PPI",
        "label": "PPI Release",
        "related_series": ["PPIFIS"],
        "impact": "moderate",
        "frequency": "monthly",
        "description": "Producer Price Index monthly release",
    },
    {
        "event_type": "ISM_MFG",
        "label": "ISM Manufacturing PMI",
        "related_series": ["INDPRO", "TCU"],
        "impact": "moderate",
        "frequency": "monthly",
        "description": "Institute for Supply Management manufacturing survey",
    },
    {
        "event_type": "RETAIL_SALES",
        "label": "Retail Sales",
        "related_series": ["RSAFS"],
        "impact": "moderate",
        "frequency": "monthly",
        "description": "Census Bureau monthly retail sales report",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2026 FOMC meeting dates (known schedule)
# ─────────────────────────────────────────────────────────────────────────────

FOMC_2026_DATES: list[date] = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
]


def _generate_monthly_dates(
    year: int,
    typical_day: int = 12,
    months: list[int] | None = None,
) -> list[date]:
    """Generate approximate monthly release dates for a given year.

    Args:
        year: Calendar year.
        typical_day: Day of month the release typically falls on.
        months: Specific months (1-12). None = all 12 months.

    Returns:
        List of dates.
    """
    target_months = months or list(range(1, 13))
    dates = []
    for m in target_months:
        day = min(typical_day, 28)  # safe for February
        dates.append(date(year, m, day))
    return dates


def _generate_quarterly_dates(year: int, typical_day: int = 28) -> list[date]:
    """Generate approximate quarterly release dates.

    Args:
        year: Calendar year.
        typical_day: Day of month the release typically falls on.

    Returns:
        List of dates for Q1-Q4 releases.
    """
    return [date(year, m, min(typical_day, 28)) for m in [1, 4, 7, 10]]


# ─────────────────────────────────────────────────────────────────────────────
# Event generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_events(year: int = 2026) -> list[dict[str, Any]]:
    """Generate all macro calendar events for a given year.

    Creates deterministic event nodes from known schedules. Each event
    has a unique event_id, date, type, impact level, and list of related
    FRED series that it could affect.

    Args:
        year: Calendar year to generate events for.

    Returns:
        List of event dicts ready for batch_merge_events.
    """
    events: list[dict[str, Any]] = []

    for defn in EVENT_DEFINITIONS:
        event_type = defn["event_type"]

        if event_type == "FOMC":
            dates = [d for d in FOMC_2026_DATES if d.year == year]
        elif defn["frequency"] == "quarterly":
            dates = _generate_quarterly_dates(year)
        else:
            # Monthly releases: approximate day varies by type
            day_map = {
                "CPI": 13, "NFP": 5, "PCE": 28, "PPI": 14,
                "ISM_MFG": 1, "RETAIL_SALES": 16,
            }
            dates = _generate_monthly_dates(year, typical_day=day_map.get(event_type, 15))

        for event_date in dates:
            event_id = f"{event_type}_{event_date.isoformat()}"
            events.append({
                "event_id": event_id,
                "event_type": event_type,
                "label": defn["label"],
                "event_date": event_date.isoformat(),
                "impact": defn["impact"],
                "description": defn["description"],
                "related_series": defn["related_series"],
            })

    log.info("Generated %d macro calendar events for %d", len(events), year)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Graph materialization
# ─────────────────────────────────────────────────────────────────────────────

def materialize_events(year: int = 2026) -> dict[str, Any]:
    """Generate events and merge them into Neo4j with TRIGGERED_BY edges.

    Steps:
    1. Generate events for the year
    2. Batch-merge Event nodes
    3. Create TRIGGERED_BY edges to related MacroIndicator nodes

    Args:
        year: Calendar year.

    Returns:
        Summary dict with counts.
    """
    from db.neo4j.client import batch_merge_events, batch_merge_event_edges

    events = generate_events(year)
    n_events = batch_merge_events(events)

    # Build TRIGGERED_BY edges: Event -> MacroIndicator
    trigger_edges: list[dict[str, Any]] = []
    for ev in events:
        for series_id in ev.get("related_series", []):
            trigger_edges.append({
                "event_id": ev["event_id"],
                "series_id": series_id,
                "rel_type": "TRIGGERED_BY",
            })

    n_edges = batch_merge_event_edges(trigger_edges)

    summary = {
        "events_merged": n_events,
        "trigger_edges_merged": n_edges,
        "year": year,
    }

    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id=AGENT_ID,
            action="materialize_events",
            after_state=summary,
        ))
    except Exception as exc:
        log.warning("Could not log to evolution trail: %s", exc)

    log.info("Event materialization complete: %s", summary)
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    result = materialize_events(yr)
    print(f"\nEvents materialized for {yr}:")
    print(f"  Event nodes: {result['events_merged']}")
    print(f"  TRIGGERED_BY edges: {result['trigger_edges_merged']}")
