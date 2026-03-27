"""Neo4j Aura client for FinBrain.

Provides a lazy singleton GraphDatabase driver and typed helpers for:
- Applying the schema (constraints + indexes)
- Merging nodes (upsert by unique key)
- Creating relationships between nodes
- Querying neighbours for a given asset (used by the D3 dashboard)
- Running arbitrary Cypher read queries
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Generator

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

from db.neo4j.schema import CONSTRAINTS, INDEXES, NodeLabel, RelType
from skills.env import get_neo4j_password, get_neo4j_uri, get_neo4j_user
from skills.logger import get_logger

logger = get_logger(__name__)

_driver: Driver | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Driver management
# ─────────────────────────────────────────────────────────────────────────────

def get_driver() -> Driver:
    """Return a cached Neo4j Driver, initialising it on first call.

    Returns:
        An authenticated Neo4j Driver connected to Aura.
    """
    global _driver
    if _driver is None:
        uri = get_neo4j_uri()
        user = get_neo4j_user()
        password = get_neo4j_password()
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("neo4j_driver_initialised", uri=uri[:40] + "...")
    return _driver


def close_driver() -> None:
    """Close the cached Neo4j driver and reset the singleton.

    Call this on application shutdown to release connection resources.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("neo4j_driver_closed")


@contextlib.contextmanager
def session() -> Generator[Session, None, None]:
    """Context manager that yields an open Neo4j Session.

    Yields:
        An open neo4j.Session.
    """
    driver = get_driver()
    with driver.session() as s:
        yield s


# ─────────────────────────────────────────────────────────────────────────────
# Schema application
# ─────────────────────────────────────────────────────────────────────────────

def apply_schema() -> dict[str, int]:
    """Apply all constraints and indexes defined in schema.py.

    Idempotent — uses IF NOT EXISTS so safe to run on every startup.

    Returns:
        Dict with keys 'constraints' and 'indexes' showing counts applied.
    """
    with session() as s:
        for name, cypher in CONSTRAINTS:
            s.run(cypher)
            logger.info("neo4j_constraint_applied", name=name)
        for name, cypher in INDEXES:
            s.run(cypher)
            logger.info("neo4j_index_applied", name=name)

    return {"constraints": len(CONSTRAINTS), "indexes": len(INDEXES)}


# ─────────────────────────────────────────────────────────────────────────────
# Typed node dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetNode:
    """Properties for an Asset node."""
    ticker: str
    name: str
    asset_class: str   # equity | crypto | forex | commodity
    sector: str | None = None
    exchange: str | None = None


@dataclass
class SectorNode:
    """Properties for a Sector node."""
    name: str
    description: str = ""


@dataclass
class MacroIndicatorNode:
    """Properties for a MacroIndicator node."""
    series_id: str    # e.g. 'GDP', 'CPIAUCSL'
    name: str
    source: str = "fred"
    frequency: str = "monthly"
    unit: str = ""


@dataclass
class EventNode:
    """Properties for an Event node (earnings, FOMC, macro release, etc.)."""
    event_id: str
    event_type: str   # earnings | fomc | macro_release | geopolitical
    title: str
    event_date: str   # ISO date string
    description: str = ""


@dataclass
class SignalNode:
    """Properties for a Signal node."""
    signal_id: str
    asset: str
    direction: str    # long | short | neutral
    confidence: float
    created_at: str   # ISO datetime string


@dataclass
class ModelNode:
    """Properties for a Model node."""
    model_id: str
    name: str
    model_type: str
    version: str
    status: str = "staging"
    accuracy: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Node merge helpers (MERGE = upsert by unique key)
# ─────────────────────────────────────────────────────────────────────────────

def merge_asset(node: AssetNode) -> dict[str, Any]:
    """Merge an Asset node into the graph (upsert by ticker).

    Args:
        node: The AssetNode to merge.

    Returns:
        The node properties as stored in Neo4j.
    """
    cypher = """
        MERGE (a:Asset {ticker: $ticker})
        SET a.name        = $name,
            a.asset_class = $asset_class,
            a.sector      = $sector,
            a.exchange    = $exchange
        RETURN properties(a) AS props
    """
    with session() as s:
        result = s.run(cypher, ticker=node.ticker, name=node.name,
                       asset_class=node.asset_class, sector=node.sector,
                       exchange=node.exchange)
        record = result.single()
        logger.info("neo4j_asset_merged", ticker=node.ticker)
        return dict(record["props"]) if record else {}


def merge_sector(node: SectorNode) -> dict[str, Any]:
    """Merge a Sector node into the graph (upsert by name).

    Args:
        node: The SectorNode to merge.

    Returns:
        The node properties as stored in Neo4j.
    """
    cypher = """
        MERGE (s:Sector {name: $name})
        SET s.description = $description
        RETURN properties(s) AS props
    """
    with session() as s:
        result = s.run(cypher, name=node.name, description=node.description)
        record = result.single()
        logger.info("neo4j_sector_merged", name=node.name)
        return dict(record["props"]) if record else {}


def merge_macro_indicator(node: MacroIndicatorNode) -> dict[str, Any]:
    """Merge a MacroIndicator node into the graph (upsert by series_id).

    Args:
        node: The MacroIndicatorNode to merge.

    Returns:
        The node properties as stored in Neo4j.
    """
    cypher = """
        MERGE (m:MacroIndicator {series_id: $series_id})
        SET m.name      = $name,
            m.source    = $source,
            m.frequency = $frequency,
            m.unit      = $unit
        RETURN properties(m) AS props
    """
    with session() as s:
        result = s.run(cypher, series_id=node.series_id, name=node.name,
                       source=node.source, frequency=node.frequency, unit=node.unit)
        record = result.single()
        logger.info("neo4j_macro_indicator_merged", series_id=node.series_id)
        return dict(record["props"]) if record else {}


def merge_signal(node: SignalNode) -> dict[str, Any]:
    """Merge a Signal node into the graph (upsert by signal_id).

    Args:
        node: The SignalNode to merge.

    Returns:
        The node properties as stored in Neo4j.
    """
    cypher = """
        MERGE (sig:Signal {signal_id: $signal_id})
        SET sig.asset      = $asset,
            sig.direction  = $direction,
            sig.confidence = $confidence,
            sig.created_at = $created_at
        RETURN properties(sig) AS props
    """
    with session() as s:
        result = s.run(cypher, signal_id=node.signal_id, asset=node.asset,
                       direction=node.direction, confidence=node.confidence,
                       created_at=node.created_at)
        record = result.single()
        logger.info("neo4j_signal_merged", signal_id=node.signal_id)
        return dict(record["props"]) if record else {}


def merge_model(node: ModelNode) -> dict[str, Any]:
    """Merge a Model node into the graph (upsert by model_id).

    Args:
        node: The ModelNode to merge.

    Returns:
        The node properties as stored in Neo4j.
    """
    cypher = """
        MERGE (m:Model {model_id: $model_id})
        SET m.name       = $name,
            m.model_type = $model_type,
            m.version    = $version,
            m.status     = $status,
            m.accuracy   = $accuracy
        RETURN properties(m) AS props
    """
    with session() as s:
        result = s.run(cypher, model_id=node.model_id, name=node.name,
                       model_type=node.model_type, version=node.version,
                       status=node.status, accuracy=node.accuracy)
        record = result.single()
        logger.info("neo4j_model_merged", model_id=node.model_id)
        return dict(record["props"]) if record else {}


# ─────────────────────────────────────────────────────────────────────────────
# Relationship helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_belongs_to(ticker: str, sector_name: str) -> None:
    """Create or update a BELONGS_TO relationship from an Asset to a Sector.

    Args:
        ticker: The asset ticker (must already exist as an Asset node).
        sector_name: The sector name (must already exist as a Sector node).
    """
    cypher = """
        MATCH (a:Asset {ticker: $ticker})
        MATCH (s:Sector {name: $sector_name})
        MERGE (a)-[:BELONGS_TO]->(s)
    """
    with session() as s:
        s.run(cypher, ticker=ticker, sector_name=sector_name)
        logger.info("neo4j_belongs_to_created", ticker=ticker, sector=sector_name)


def create_correlated_with(ticker_a: str, ticker_b: str,
                           correlation: float, window_days: int = 90) -> None:
    """Create or update a CORRELATED_WITH relationship between two assets.

    The relationship is undirected by convention — we always write from
    the lexicographically smaller ticker to avoid duplicates.

    Args:
        ticker_a: First asset ticker.
        ticker_b: Second asset ticker.
        correlation: Pearson correlation coefficient (-1 to 1).
        window_days: Rolling window length used to compute the correlation.
    """
    if ticker_a > ticker_b:
        ticker_a, ticker_b = ticker_b, ticker_a  # canonical ordering

    cypher = """
        MATCH (a:Asset {ticker: $ticker_a})
        MATCH (b:Asset {ticker: $ticker_b})
        MERGE (a)-[r:CORRELATED_WITH]-(b)
        SET r.correlation  = $correlation,
            r.window_days  = $window_days,
            r.updated_at   = datetime()
    """
    with session() as s:
        s.run(cypher, ticker_a=ticker_a, ticker_b=ticker_b,
              correlation=correlation, window_days=window_days)
        logger.info("neo4j_correlation_created", a=ticker_a, b=ticker_b,
                    correlation=correlation)


def create_generates(model_id: str, signal_id: str) -> None:
    """Create a GENERATES relationship from a Model to a Signal.

    Args:
        model_id: The model's unique ID.
        signal_id: The signal's unique ID.
    """
    cypher = """
        MATCH (m:Model   {model_id:  $model_id})
        MATCH (s:Signal  {signal_id: $signal_id})
        MERGE (m)-[:GENERATES]->(s)
    """
    with session() as s:
        s.run(cypher, model_id=model_id, signal_id=signal_id)
        logger.info("neo4j_generates_created", model=model_id, signal=signal_id)


def create_trained_on(model_id: str, ticker: str) -> None:
    """Create a TRAINED_ON relationship from a Model to an Asset.

    Args:
        model_id: The model's unique ID.
        ticker: The asset ticker the model was trained on.
    """
    cypher = """
        MATCH (m:Model  {model_id: $model_id})
        MATCH (a:Asset  {ticker:   $ticker})
        MERGE (m)-[:TRAINED_ON]->(a)
    """
    with session() as s:
        s.run(cypher, model_id=model_id, ticker=ticker)
        logger.info("neo4j_trained_on_created", model=model_id, ticker=ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_asset_neighbours(ticker: str, max_depth: int = 2) -> list[dict[str, Any]]:
    """Return all nodes and relationships reachable from an Asset within max_depth hops.

    Used by the D3 knowledge network endpoint to build the force graph.

    Args:
        ticker: The asset ticker to start from.
        max_depth: Maximum relationship hops to traverse.

    Returns:
        List of dicts with keys 'source', 'target', 'relationship', 'properties'.
    """
    cypher = f"""
        MATCH path = (a:Asset {{ticker: $ticker}})-[r*1..{max_depth}]-(neighbour)
        UNWIND relationships(path) AS rel
        RETURN
            startNode(rel).ticker   AS source,
            endNode(rel).ticker     AS target,
            labels(endNode(rel))[0] AS target_label,
            type(rel)               AS relationship,
            properties(rel)         AS properties
        LIMIT 200
    """
    with session() as s:
        result = s.run(cypher, ticker=ticker)
        rows = [dict(r) for r in result]
        logger.info("neo4j_neighbours_fetched", ticker=ticker, count=len(rows))
        return rows


def run_read_query(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run an arbitrary read-only Cypher query and return all results.

    Args:
        cypher: The Cypher query string.
        params: Optional parameter dict for the query.

    Returns:
        List of result records as dicts.
    """
    with session() as s:
        result = s.run(cypher, **(params or {}))
        return [dict(r) for r in result]


# ─────────────────────────────────────────────────────────────────────────────
# Batch merge helpers (used by graph materializer for efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def batch_merge_assets(nodes: list[dict[str, Any]]) -> int:
    """Batch-merge Asset nodes using UNWIND for efficiency.

    Each dict must have keys: ticker, name, asset_class.
    Optional keys: sector, exchange.

    Args:
        nodes: List of asset property dicts.

    Returns:
        Number of nodes merged.
    """
    if not nodes:
        return 0
    cypher = """
        UNWIND $nodes AS n
        MERGE (a:Asset {ticker: n.ticker})
        SET a.name        = n.name,
            a.asset_class = n.asset_class,
            a.sector      = n.sector,
            a.exchange     = n.exchange
    """
    with session() as s:
        s.run(cypher, nodes=nodes)
    logger.info("neo4j_batch_assets_merged", count=len(nodes))
    return len(nodes)


def batch_merge_macro_indicators(nodes: list[dict[str, Any]]) -> int:
    """Batch-merge MacroIndicator nodes using UNWIND.

    Each dict must have keys: series_id, name, source, frequency.

    Args:
        nodes: List of macro indicator property dicts.

    Returns:
        Number of nodes merged.
    """
    if not nodes:
        return 0
    cypher = """
        UNWIND $nodes AS n
        MERGE (m:MacroIndicator {series_id: n.series_id})
        SET m.name      = n.name,
            m.source    = n.source,
            m.frequency = n.frequency
    """
    with session() as s:
        s.run(cypher, nodes=nodes)
    logger.info("neo4j_batch_macro_merged", count=len(nodes))
    return len(nodes)


def batch_merge_edges(edges: list[dict[str, Any]]) -> int:
    """Batch-merge relationship edges using UNWIND.

    Each dict must have keys:
        source_id, source_label, target_id, target_label, rel_type,
        pearson_r, lag_days, strength.
    Optional: granger_p, mutual_info, regime, relationship_type.

    Supports cross-label edges (Asset↔Asset, Asset↔MacroIndicator, etc).

    Args:
        edges: List of edge property dicts.

    Returns:
        Number of edges merged.
    """
    if not edges:
        return 0

    # Group by label pair + rel_type to build correct MATCH clauses.
    # Most common cases: Asset↔Asset, MacroIndicator↔Asset
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for e in edges:
        key = (e["source_label"], e["target_label"], e["rel_type"])
        groups.setdefault(key, []).append(e)

    total = 0
    with session() as s:
        for (src_label, tgt_label, rel_type), batch in groups.items():
            src_key = "ticker" if src_label == "Asset" else "series_id"
            tgt_key = "ticker" if tgt_label == "Asset" else "series_id"

            # SENSITIVE_TO edges need (factor_group, regime) in the MERGE key
            # so that bear/stress betas get their own edges rather than
            # overwriting the all-regime beta.
            if rel_type == "SENSITIVE_TO":
                merge_clause = (
                    f"MERGE (a)-[r:{rel_type} "
                    "{factor_group: e.factor_group, regime: e.regime}]->(b)"
                )
            else:
                merge_clause = f"MERGE (a)-[r:{rel_type}]->(b)"

            cypher = f"""
                UNWIND $edges AS e
                MATCH (a:{src_label} {{{src_key}: e.source_id}})
                MATCH (b:{tgt_label} {{{tgt_key}: e.target_id}})
                {merge_clause}
                SET r.pearson_r         = e.pearson_r,
                    r.lag_days          = e.lag_days,
                    r.granger_p         = e.granger_p,
                    r.mutual_info       = e.mutual_info,
                    r.strength          = e.strength,
                    r.regime            = e.regime,
                    r.relationship_type = e.relationship_type,
                    r.factor_group      = e.factor_group,
                    r.beta              = e.beta,
                    r.updated_at        = datetime()
            """
            s.run(cypher, edges=batch)
            total += len(batch)
            logger.info("neo4j_batch_edges_merged",
                        src_label=src_label, tgt_label=tgt_label,
                        rel_type=rel_type, count=len(batch))

    return total


def get_graph_stats() -> dict[str, Any]:
    """Return counts of nodes by label and edges by type.

    Returns:
        Dict with 'nodes' (label→count) and 'edges' (type→count) and 'total_nodes', 'total_edges'.
    """
    node_cypher = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n)
            WHERE label IN labels(n)
            RETURN count(n) AS cnt
        }
        RETURN label, cnt
    """
    edge_cypher = """
        CALL db.relationshipTypes() YIELD relationshipType AS rtype
        CALL {
            WITH rtype
            MATCH ()-[r]->()
            WHERE type(r) = rtype
            RETURN count(r) AS cnt
        }
        RETURN rtype, cnt
    """
    nodes: dict[str, int] = {}
    edges: dict[str, int] = {}
    try:
        with session() as s:
            for record in s.run(node_cypher):
                nodes[record["label"]] = record["cnt"]
            for record in s.run(edge_cypher):
                edges[record["rtype"]] = record["cnt"]
    except Exception as exc:
        logger.warning("neo4j_stats_failed", error=str(exc))

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": sum(nodes.values()),
        "total_edges": sum(edges.values()),
    }
