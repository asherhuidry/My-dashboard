"""Graph Materializer — turns persisted discoveries into a live Neo4j market graph.

Reads all discovery rows from Supabase, resolves each series to its node type
(Asset or MacroIndicator), batch-merges nodes, then batch-merges edges with
full statistical properties (pearson_r, granger_p, lag_days, strength, etc.).

Idempotent: safe to run repeatedly.  Every run brings the graph up to date
with whatever discoveries exist in Supabase.

This is the bridge between the flat discovery table and the queryable graph
that future graph-ML, neural representations, and the D3 network visualization
consume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from skills.logger import get_logger

log = get_logger(__name__)

AGENT_ID = "graph_materializer"

# Granger p-value threshold: below this we create a directed CAUSES edge
# in addition to the undirected CORRELATED_WITH edge.
GRANGER_CAUSAL_THRESHOLD = 0.05

# relationship_type values that denote factor sensitivity discoveries.
# These get SENSITIVE_TO edges instead of CORRELATED_WITH.
SENSITIVITY_SUFFIX = "_sensitive"


# ─────────────────────────────────────────────────────────────────────────────
# Series classification
# ─────────────────────────────────────────────────────────────────────────────

def _build_series_lookup() -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Build lookup tables for classifying series names.

    Returns:
        Tuple of (asset_lookup, macro_lookup) where:
        - asset_lookup: {ticker: asset_class} for all yfinance assets
        - macro_lookup: {series_id: (label, frequency)} for all FRED series
    """
    from data.ingest.universe import get_yfinance_universe, MACRO_SERIES

    asset_lookup = get_yfinance_universe(extended=True)
    macro_lookup = {sid: (label, freq) for sid, label, freq in MACRO_SERIES}

    return asset_lookup, macro_lookup


def classify_series(series_id: str,
                    asset_lookup: dict[str, str],
                    macro_lookup: dict[str, tuple[str, str]]) -> str:
    """Classify a series ID as 'Asset' or 'MacroIndicator'.

    Args:
        series_id: The series identifier from a discovery row.
        asset_lookup: {ticker: asset_class} mapping.
        macro_lookup: {fred_id: (label, frequency)} mapping.

    Returns:
        'Asset' or 'MacroIndicator'.
    """
    if series_id in asset_lookup:
        return "Asset"
    if series_id in macro_lookup:
        return "MacroIndicator"
    # Heuristic: FRED series IDs are typically uppercase alphanumeric, often
    # longer than 4 chars, and don't contain hyphens or '=' (unlike tickers).
    # Default to Asset for unknown series (more common case).
    log.debug("Unknown series '%s' — defaulting to Asset", series_id)
    return "Asset"


# ─────────────────────────────────────────────────────────────────────────────
# Node builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_asset_node(ticker: str, asset_lookup: dict[str, str]) -> dict[str, Any]:
    """Build a node property dict for an Asset.

    Args:
        ticker: The asset ticker.
        asset_lookup: {ticker: asset_class} mapping.

    Returns:
        Dict with keys: ticker, name, asset_class, sector, exchange.
    """
    return {
        "ticker": ticker,
        "name": ticker,  # We use ticker as display name for now
        "asset_class": asset_lookup.get(ticker, "unknown"),
        "sector": None,
        "exchange": None,
    }


def _build_macro_node(series_id: str,
                      macro_lookup: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Build a node property dict for a MacroIndicator.

    Args:
        series_id: The FRED series identifier.
        macro_lookup: {series_id: (label, frequency)} mapping.

    Returns:
        Dict with keys: series_id, name, source, frequency.
    """
    label, freq = macro_lookup.get(series_id, (series_id, "unknown"))
    return {
        "series_id": series_id,
        "name": label,
        "source": "fred",
        "frequency": freq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edge builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_edge(row: dict[str, Any],
                asset_lookup: dict[str, str],
                macro_lookup: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """Build edge dicts from a single discovery row.

    For correlation discoveries, may return 1-2 edges:
    - CORRELATED_WITH (undirected, canonical order)
    - CAUSES (directed, if Granger p < threshold)

    For sensitivity discoveries (relationship_type ends with '_sensitive'),
    returns exactly 1 directed SENSITIVE_TO edge:
    - Asset -[SENSITIVE_TO]-> MacroIndicator

    Args:
        row: A discovery row from Supabase.
        asset_lookup: {ticker: asset_class} mapping.
        macro_lookup: {series_id: (label, frequency)} mapping.

    Returns:
        List of edge dicts ready for batch_merge_edges.
    """
    rel_type_raw = row.get("relationship_type", "discovered")

    # ── Sensitivity edges ──────────────────────────────────────────────
    if rel_type_raw.endswith(SENSITIVITY_SUFFIX):
        return _build_sensitivity_edge(row, asset_lookup, macro_lookup)

    # ── Correlation edges (existing logic) ─────────────────────────────
    from data.agents.edge_confidence import score_edge

    series_a = row["series_a"]
    series_b = row["series_b"]
    label_a = classify_series(series_a, asset_lookup, macro_lookup)
    label_b = classify_series(series_b, asset_lookup, macro_lookup)

    # Canonical ordering for undirected CORRELATED_WITH
    if series_a > series_b:
        series_a, series_b = series_b, series_a
        label_a, label_b = label_b, label_a

    # Discovery provenance — carried through to Neo4j edges
    provenance = {
        "discovery_id": row.get("id"),
        "run_id": row.get("run_id"),
        "discovered_at": row.get("computed_at"),
    }

    base = {
        "pearson_r": row.get("pearson_r", 0),
        "lag_days": row.get("lag_days", 0),
        "granger_p": row.get("granger_p"),
        "mutual_info": row.get("mutual_info"),
        "strength": row.get("strength", "weak"),
        "regime": row.get("regime", "all"),
        "relationship_type": row.get("relationship_type", "discovered"),
        "factor_group": None,
        "beta": None,
        **provenance,
    }

    edges = []

    # 1. Always create CORRELATED_WITH
    corr_props = {**base, "rel_type": "CORRELATED_WITH"}
    corr_props["confidence"] = score_edge(corr_props)
    edges.append({
        "source_id": series_a,
        "source_label": label_a,
        "target_id": series_b,
        "target_label": label_b,
        **corr_props,
    })

    # 2. If Granger-causal, also create directed CAUSES edge
    granger_p = row.get("granger_p")
    if granger_p is not None and granger_p < GRANGER_CAUSAL_THRESHOLD:
        orig_a = row["series_a"]
        orig_b = row["series_b"]
        orig_label_a = classify_series(orig_a, asset_lookup, macro_lookup)
        orig_label_b = classify_series(orig_b, asset_lookup, macro_lookup)
        causal_props = {**base, "rel_type": "CAUSES"}
        causal_props["confidence"] = score_edge(causal_props)
        edges.append({
            "source_id": orig_a,
            "source_label": orig_label_a,
            "target_id": orig_b,
            "target_label": orig_label_b,
            **causal_props,
        })

    return edges


def _build_sensitivity_edge(row: dict[str, Any],
                            asset_lookup: dict[str, str],
                            macro_lookup: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """Build a directed SENSITIVE_TO edge from a sensitivity discovery.

    Direction is always Asset -> MacroIndicator (asset is exposed to factor).
    The beta coefficient (stored in mutual_info) becomes an explicit edge property.

    Args:
        row: A sensitivity discovery row from Supabase.
        asset_lookup: {ticker: asset_class} mapping.
        macro_lookup: {series_id: (label, frequency)} mapping.

    Returns:
        Single-element list with the SENSITIVE_TO edge dict.
    """
    from data.agents.edge_confidence import score_edge

    series_a = row["series_a"]  # asset
    series_b = row["series_b"]  # macro factor
    label_a = classify_series(series_a, asset_lookup, macro_lookup)
    label_b = classify_series(series_b, asset_lookup, macro_lookup)

    rel_type_raw = row.get("relationship_type", "")
    factor_group = rel_type_raw.replace(SENSITIVITY_SUFFIX, "")

    # Beta is stored in mutual_info field for sensitivity discoveries
    beta = row.get("mutual_info")

    props = {
        "rel_type": "SENSITIVE_TO",
        "pearson_r": row.get("pearson_r", 0),
        "lag_days": row.get("lag_days", 0),
        "granger_p": row.get("granger_p"),
        "mutual_info": row.get("mutual_info"),
        "strength": row.get("strength", "weak"),
        "regime": row.get("regime", "all"),
        "relationship_type": rel_type_raw,
        "factor_group": factor_group,
        "beta": beta,
        "discovery_id": row.get("id"),
        "run_id": row.get("run_id"),
        "discovered_at": row.get("computed_at"),
    }
    props["confidence"] = score_edge(props)

    return [{
        "source_id": series_a,
        "source_label": label_a,
        "target_id": series_b,
        "target_label": label_b,
        **props,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# Main materializer
# ─────────────────────────────────────────────────────────────────────────────

def materialize(
    min_strength: str | None = None,
    min_abs_r: float = 0.0,
) -> dict[str, Any]:
    """Read discoveries from Supabase and materialize them as a Neo4j graph.

    Steps:
    1. Apply Neo4j schema (constraints + indexes)
    2. Fetch all discoveries from Supabase
    3. Classify each series as Asset or MacroIndicator
    4. Batch-merge all nodes
    5. Batch-merge all edges (CORRELATED_WITH + CAUSES)
    6. Return summary stats

    Args:
        min_strength: If set, only materialize 'strong', 'moderate', or 'weak' and above.
        min_abs_r: Minimum |pearson_r| to include (default 0 = all).

    Returns:
        Summary dict with node/edge counts.
    """
    from db.neo4j.client import (
        apply_schema, batch_merge_assets, batch_merge_macro_indicators,
        batch_merge_edges, get_graph_stats,
    )
    from db.supabase.client import get_client

    # ── Step 1: Schema ────────────────────────────────────────────────────
    log.info("Applying Neo4j schema...")
    schema_result = apply_schema()
    log.info("Schema applied: %s", schema_result)

    # ── Step 2: Fetch discoveries ─────────────────────────────────────────
    log.info("Fetching discoveries from Supabase...")
    client = get_client()

    # Fetch all discoveries (paginate if needed — Supabase default limit is 1000)
    query = client.table("discoveries").select("*")
    if min_strength:
        strength_order = ["strong", "moderate", "weak"]
        idx = strength_order.index(min_strength) if min_strength in strength_order else 2
        allowed = strength_order[:idx + 1]
        query = query.in_("strength", allowed)

    result = query.order("pearson_r", desc=True).limit(2000).execute()
    rows = result.data
    log.info("Fetched %d discoveries from Supabase", len(rows))

    if not rows:
        log.warning("No discoveries to materialize")
        return {"discoveries_fetched": 0, "nodes_merged": 0, "edges_merged": 0}

    # Apply min_abs_r filter client-side
    if min_abs_r > 0:
        rows = [r for r in rows if abs(r.get("pearson_r", 0)) >= min_abs_r]
        log.info("After |r| >= %.2f filter: %d discoveries", min_abs_r, len(rows))

    # ── Step 3: Classify series ───────────────────────────────────────────
    asset_lookup, macro_lookup = _build_series_lookup()

    # Collect unique series IDs
    all_series: set[str] = set()
    for r in rows:
        all_series.add(r["series_a"])
        all_series.add(r["series_b"])

    asset_ids: set[str] = set()
    macro_ids: set[str] = set()
    for sid in all_series:
        if classify_series(sid, asset_lookup, macro_lookup) == "Asset":
            asset_ids.add(sid)
        else:
            macro_ids.add(sid)

    log.info("Unique series: %d assets, %d macro indicators", len(asset_ids), len(macro_ids))

    # ── Step 4: Merge nodes ───────────────────────────────────────────────
    asset_nodes = [_build_asset_node(t, asset_lookup) for t in sorted(asset_ids)]
    macro_nodes = [_build_macro_node(s, macro_lookup) for s in sorted(macro_ids)]

    n_assets = batch_merge_assets(asset_nodes)
    n_macros = batch_merge_macro_indicators(macro_nodes)
    log.info("Nodes merged: %d assets, %d macro indicators", n_assets, n_macros)

    # ── Step 5: Build and merge edges ─────────────────────────────────────
    all_edges: list[dict[str, Any]] = []
    for r in rows:
        all_edges.extend(_build_edge(r, asset_lookup, macro_lookup))

    n_edges = batch_merge_edges(all_edges)

    correlated = sum(1 for e in all_edges if e["rel_type"] == "CORRELATED_WITH")
    causal = sum(1 for e in all_edges if e["rel_type"] == "CAUSES")
    sensitive = sum(1 for e in all_edges if e["rel_type"] == "SENSITIVE_TO")
    log.info("Edges merged: %d CORRELATED_WITH, %d CAUSES, %d SENSITIVE_TO (%d total)",
             correlated, causal, sensitive, n_edges)

    # ── Step 6: Final stats ───────────────────────────────────────────────
    stats = get_graph_stats()
    log.info("Graph state: %s", stats)

    summary = {
        "discoveries_fetched": len(rows),
        "unique_series": len(all_series),
        "asset_nodes_merged": n_assets,
        "macro_nodes_merged": n_macros,
        "edges_merged": n_edges,
        "correlated_with_edges": correlated,
        "causes_edges": causal,
        "sensitive_to_edges": sensitive,
        "graph_stats": stats,
    }

    # Log to evolution trail
    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id=AGENT_ID,
            action="materialize_graph",
            after_state=summary,
        ))
    except Exception as exc:
        log.warning("Could not log to evolution trail: %s", exc)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(min_strength: str | None = "moderate") -> dict[str, Any]:
    """Main entry point for the graph materializer.

    Called by GitHub Actions on the weekly schedule or manually.
    Default: only materializes moderate+ strength discoveries to keep
    the graph focused on meaningful relationships.

    Args:
        min_strength: Minimum strength to include. None = all.

    Returns:
        Summary dict.
    """
    log.info("=== Graph Materializer starting ===")
    result = materialize(min_strength=min_strength)
    log.info("=== Graph Materializer complete: %d nodes, %d edges ===",
             result["asset_nodes_merged"] + result["macro_nodes_merged"],
             result["edges_merged"])
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    strength = sys.argv[1] if len(sys.argv) > 1 else "moderate"
    if strength == "all":
        strength = None
    result = run(min_strength=strength)
    print(f"\nGraph materialized:")
    print(f"  Discoveries read:  {result['discoveries_fetched']}")
    print(f"  Asset nodes:       {result['asset_nodes_merged']}")
    print(f"  Macro nodes:       {result['macro_nodes_merged']}")
    print(f"  Edges (total):     {result['edges_merged']}")
    print(f"    CORRELATED_WITH: {result['correlated_with_edges']}")
    print(f"    CAUSES:          {result['causes_edges']}")
    if result.get("graph_stats"):
        gs = result["graph_stats"]
        print(f"  Graph totals:      {gs['total_nodes']} nodes, {gs['total_edges']} edges")
