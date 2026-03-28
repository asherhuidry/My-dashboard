"""Neo4j graph schema definitions for FinBrain.

Defines node labels, relationship types, and the Cypher statements
needed to create uniqueness constraints and indexes on Neo4j Aura.

Run apply_schema() once at startup (or via a migration script) to
ensure the graph is structured correctly before agents write to it.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Node labels
# ─────────────────────────────────────────────────────────────────────────────

class NodeLabel:
    """All node label constants used in the FinBrain graph."""
    ASSET = "Asset"
    SECTOR = "Sector"
    MACRO_INDICATOR = "MacroIndicator"
    EVENT = "Event"
    SIGNAL = "Signal"
    MODEL = "Model"


# ─────────────────────────────────────────────────────────────────────────────
# Relationship types
# ─────────────────────────────────────────────────────────────────────────────

class RelType:
    """All relationship type constants used in the FinBrain graph."""
    CORRELATED_WITH = "CORRELATED_WITH"   # Asset ↔ Asset
    CAUSES = "CAUSES"                     # MacroIndicator → Asset or Event → Asset
    SENSITIVE_TO = "SENSITIVE_TO"         # Asset → MacroIndicator (factor exposure)
    BELONGS_TO = "BELONGS_TO"             # Asset → Sector
    GENERATES = "GENERATES"               # Model → Signal
    TRAINED_ON = "TRAINED_ON"             # Model → Asset
    TRIGGERED_BY = "TRIGGERED_BY"         # Signal → Event
    IMPACTS = "IMPACTS"                   # Event → Asset


# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL — uniqueness constraints and indexes
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (constraint_name, cypher_statement)
CONSTRAINTS: list[tuple[str, str]] = [
    (
        "asset_ticker_unique",
        "CREATE CONSTRAINT asset_ticker_unique IF NOT EXISTS "
        "FOR (a:Asset) REQUIRE a.ticker IS UNIQUE",
    ),
    (
        "sector_name_unique",
        "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS "
        "FOR (s:Sector) REQUIRE s.name IS UNIQUE",
    ),
    (
        "macro_indicator_series_unique",
        "CREATE CONSTRAINT macro_indicator_series_unique IF NOT EXISTS "
        "FOR (m:MacroIndicator) REQUIRE m.series_id IS UNIQUE",
    ),
    (
        "event_id_unique",
        "CREATE CONSTRAINT event_id_unique IF NOT EXISTS "
        "FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    ),
    (
        "signal_id_unique",
        "CREATE CONSTRAINT signal_id_unique IF NOT EXISTS "
        "FOR (sig:Signal) REQUIRE sig.signal_id IS UNIQUE",
    ),
    (
        "model_id_unique",
        "CREATE CONSTRAINT model_id_unique IF NOT EXISTS "
        "FOR (m:Model) REQUIRE m.model_id IS UNIQUE",
    ),
]

INDEXES: list[tuple[str, str]] = [
    (
        "asset_class_index",
        "CREATE INDEX asset_class_index IF NOT EXISTS "
        "FOR (a:Asset) ON (a.asset_class)",
    ),
    (
        "signal_created_at_index",
        "CREATE INDEX signal_created_at_index IF NOT EXISTS "
        "FOR (sig:Signal) ON (sig.created_at)",
    ),
    (
        "event_date_index",
        "CREATE INDEX event_date_index IF NOT EXISTS "
        "FOR (e:Event) ON (e.event_date)",
    ),
    (
        "model_status_index",
        "CREATE INDEX model_status_index IF NOT EXISTS "
        "FOR (m:Model) ON (m.status)",
    ),
]
