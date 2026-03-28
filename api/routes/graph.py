"""Graph data routes — correlation network, knowledge graph, and node detail."""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query

log = logging.getLogger(__name__)
router = APIRouter()

# Asset universe for correlation network
_ASSETS = {
    "equity":    ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","BAC","GS",
                  "JNJ","PFE","XOM","CVX","WMT","HD","V","MA","UNH","NFLX"],
    "crypto":    ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
                  "ADA-USD","DOGE-USD","AVAX-USD","LINK-USD"],
    "forex":     ["EURUSD=X","GBPUSD=X","JPYUSD=X"],
    "commodity": ["GLD","SLV","USO","^GSPC"],
}

_ASSET_CLASS_COLOR = {
    "equity":    "#3b82f6",
    "crypto":    "#f59e0b",
    "forex":     "#10b981",
    "commodity": "#8b5cf6",
}

_REL_COLORS: dict[str, str] = {
    "SENSITIVE_TO":     "#f59e0b",
    "CORRELATED_WITH":  "#10b981",
    "BELONGS_TO":       "#8b5cf6",
    "HAS_FEATURES":     "#3b82f6",
}

_NODE_COLORS: dict[str, str] = {
    "Asset":          "#3b82f6",
    "Sector":         "#8b5cf6",
    "MacroIndicator": "#10b981",
    "Signal":         "#f59e0b",
    "Model":          "#ef4444",
    "Pattern":        "#06b6d4",
}


def _symbol_to_class(sym: str) -> str:
    for cls, syms in _ASSETS.items():
        if sym in syms:
            return cls
    return "equity"


# ── Correlation network ──────────────────────────────────────────────────────

@router.get("/graph/correlations")
def get_correlation_graph(
    days: int = Query(default=63, ge=21, le=252),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
) -> dict:
    """Return nodes and edges for the asset correlation force graph."""
    all_symbols = [s for syms in _ASSETS.values() for s in syms]

    closes: dict[str, pd.Series] = {}
    for sym in all_symbols:
        try:
            df = yf.Ticker(sym).history(period="6mo")
            if not df.empty:
                closes[sym] = df["Close"].tail(days)
        except Exception:
            pass

    if len(closes) < 2:
        return {"nodes": [], "edges": [], "error": "Insufficient data"}

    price_df = pd.DataFrame(closes).dropna(axis=1, how="all").ffill().bfill()
    ret_df   = price_df.pct_change().dropna()
    corr_mat = ret_df.corr()
    symbols  = list(corr_mat.columns)

    nodes = []
    for sym in symbols:
        cls    = _symbol_to_class(sym)
        close  = price_df[sym].iloc[-1]
        ret_1d = float(ret_df[sym].iloc[-1]) if sym in ret_df.columns else 0
        nodes.append({
            "id":    sym,
            "label": sym.replace("-USD","").replace("=X",""),
            "class": cls,
            "color": _ASSET_CLASS_COLOR.get(cls, "#6b7280"),
            "price": round(float(close), 4),
            "ret_1d": round(ret_1d * 100, 2),
            "val":   max(abs(ret_1d) * 800 + 4, 4),
        })

    edges = []
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if j <= i:
                continue
            c = float(corr_mat.loc[s1, s2])
            if np.isnan(c) or abs(c) < threshold:
                continue
            edges.append({
                "source": s1,
                "target": s2,
                "value":  round(c, 3),
                "color":  "#10b981" if c > 0 else "#ef4444",
                "width":  round(abs(c) * 4, 2),
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "meta":  {"days": days, "threshold": threshold, "asset_count": len(nodes)},
    }


# ── Knowledge graph (Neo4j) ─────────────────────────────────────────────────

def _build_knowledge_query(
    rel_type: str | None,
    asset_class: str | None,
    regime: str | None,
) -> tuple[str, dict]:
    """Build a filtered Cypher query for the knowledge graph."""
    where_parts: list[str] = []
    params: dict = {}

    if rel_type:
        where_parts.append("type(r) = $rel_type")
        params["rel_type"] = rel_type
    if asset_class:
        where_parts.append(
            "(n.asset_class = $asset_class OR m.asset_class = $asset_class)"
        )
        params["asset_class"] = asset_class
    if regime:
        where_parts.append("r.regime = $regime")
        params["regime"] = regime

    where = " WHERE " + " AND ".join(where_parts) if where_parts else ""

    query = f"""
        MATCH (n)-[r]->(m)
        {where}
        RETURN
          id(n) AS src_id, labels(n)[0] AS src_label,
          n.name AS src_name, n.asset_class AS src_class,
          type(r) AS rel, properties(r) AS rel_props,
          id(m) AS tgt_id, labels(m)[0] AS tgt_label,
          m.name AS tgt_name, m.asset_class AS tgt_class
        LIMIT 800
    """
    return query, params


@router.get("/graph/knowledge")
def get_knowledge_graph(
    rel_type: str | None = Query(None, description="Filter by relationship type"),
    asset_class: str | None = Query(None, description="Filter by asset class"),
    regime: str | None = Query(None, description="Filter by regime (all, bear, stress)"),
) -> dict:
    """Return the live market graph from Neo4j with optional filters.

    Supports filtering by relationship type, asset class, and regime.
    Computes node degree for sizing.  Returns an honest empty graph
    if Neo4j is unavailable.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        query, params = _build_knowledge_query(rel_type, asset_class, regime)

        with driver.session() as session:
            result = session.run(query, params)
            rows = result.data()

        node_map: dict[int, dict] = {}
        edges = []
        degree: Counter = Counter()
        rel_type_counts: Counter = Counter()

        for row in rows:
            for nid, nlabel, nname, ncls in [
                (row["src_id"], row["src_label"], row["src_name"], row.get("src_class")),
                (row["tgt_id"], row["tgt_label"], row["tgt_name"], row.get("tgt_class")),
            ]:
                if nid not in node_map:
                    node_map[nid] = {
                        "id":    str(nid),
                        "label": nname or str(nid),
                        "type":  nlabel or "Node",
                        "color": _NODE_COLORS.get(nlabel, "#6b7280"),
                    }
                    if ncls:
                        node_map[nid]["class"] = ncls
                degree[nid] += 1

            rtype = row["rel"]
            rel_type_counts[rtype] += 1
            props = row.get("rel_props") or {}
            beta = props.get("beta")
            corr = props.get("correlation")

            # Edge width from strongest available numeric property
            if beta is not None:
                width = round(min(abs(beta) * 6, 5), 2)
            elif corr is not None:
                width = round(abs(corr) * 4, 2)
            else:
                width = 1

            edge: dict = {
                "source": str(row["src_id"]),
                "target": str(row["tgt_id"]),
                "label":  rtype,
                "color":  _REL_COLORS.get(rtype, "#475569"),
                "width":  width,
            }
            # Copy useful edge properties
            for k in ("beta", "p_value", "regime", "factor_group",
                       "correlation", "window_days", "strength",
                       "pearson_r", "granger_p", "lag_days"):
                v = props.get(k)
                if v is not None:
                    edge[k] = v
            edges.append(edge)

        # Set node size by degree (more connections = bigger)
        for nid, node in node_map.items():
            d = degree.get(nid, 0)
            node["val"] = max(d * 2 + 3, 4)
            node["degree"] = d

        return {
            "nodes": list(node_map.values()),
            "edges": edges,
            "source": "neo4j",
            "meta": {
                "node_count": len(node_map),
                "edge_count": len(edges),
                "rel_types": dict(rel_type_counts),
                "filters": {
                    "rel_type": rel_type,
                    "asset_class": asset_class,
                    "regime": regime,
                },
            },
        }

    except Exception as exc:
        log.debug("Knowledge graph unavailable: %s", exc)
        return {
            "nodes": [],
            "edges": [],
            "source": "unavailable",
            "meta": {"node_count": 0, "edge_count": 0, "rel_types": {}, "filters": {}},
        }


# ── Node detail ─────────────────────────────────────────────────────────────

@router.get("/graph/node/{node_name}")
def get_node_detail(node_name: str) -> dict:
    """Return all relationships for a specific node, grouped by type.

    Accepts the node name (ticker or series_id) and returns every
    connected edge with full properties, plus the node's own metadata.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n {name: $name})-[r]-(m)
                RETURN
                  labels(n)[0] AS node_label, n.name AS node_name,
                  n.asset_class AS node_class, n.sector AS node_sector,
                  type(r) AS rel, properties(r) AS rel_props,
                  labels(m)[0] AS neighbor_label, m.name AS neighbor_name,
                  m.asset_class AS neighbor_class,
                  CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
                ORDER BY type(r), m.name
            """, {"name": node_name})
            rows = result.data()

        if not rows:
            return {"found": False, "node_name": node_name}

        first = rows[0]
        node_info = {
            "name":  first["node_name"],
            "type":  first["node_label"],
            "class": first.get("node_class"),
            "sector": first.get("node_sector"),
        }

        # Group edges by relationship type
        by_type: dict[str, list] = {}
        for row in rows:
            rtype = row["rel"]
            props = row.get("rel_props") or {}
            entry = {
                "neighbor":       row["neighbor_name"],
                "neighbor_type":  row["neighbor_label"],
                "neighbor_class": row.get("neighbor_class"),
                "direction":      row["direction"],
            }
            # Include all non-null properties
            for k, v in props.items():
                if v is not None and k != "updated_at":
                    entry[k] = v
            by_type.setdefault(rtype, []).append(entry)

        return {
            "found": True,
            "node": node_info,
            "relationships": by_type,
            "total_edges": len(rows),
        }

    except Exception as exc:
        log.debug("Node detail unavailable: %s", exc)
        return {"found": False, "node_name": node_name, "error": str(exc)}


# ── Graph stats ──────────────────────────────────────────────────────────────

@router.get("/graph/stats")
def get_graph_stats() -> dict:
    """Return live node/edge counts from the Neo4j market graph."""
    try:
        from db.neo4j.client import get_graph_stats as neo4j_stats
        stats = neo4j_stats()
        return {
            "nodes": stats.get("nodes", {}),
            "edges": stats.get("edges", {}),
            "total_nodes": stats.get("total_nodes", 0),
            "total_edges": stats.get("total_edges", 0),
            "source": "neo4j",
        }
    except Exception:
        return {
            "nodes": {},
            "edges": {},
            "total_nodes": 0,
            "total_edges": 0,
            "source": "unavailable",
        }
