"""Graph data routes — correlation network and knowledge graph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query

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


def _symbol_to_class(sym: str) -> str:
    for cls, syms in _ASSETS.items():
        if sym in syms:
            return cls
    return "equity"


@router.get("/graph/correlations")
def get_correlation_graph(
    days: int = Query(default=63, ge=21, le=252),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
) -> dict:
    """
    Return nodes and edges for the asset correlation force graph.
    Edges only included where |correlation| >= threshold.
    """
    all_symbols = [s for syms in _ASSETS.values() for s in syms]

    # Fetch closing prices
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

    # Build returns matrix
    price_df = pd.DataFrame(closes).dropna(axis=1, how="all").ffill().bfill()
    ret_df   = price_df.pct_change().dropna()
    corr_mat = ret_df.corr()

    symbols = list(corr_mat.columns)

    # Nodes
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
            "val":   max(abs(ret_1d) * 800 + 4, 4),  # node size by volatility
        })

    # Edges (upper triangle only)
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


@router.get("/graph/knowledge")
def get_knowledge_graph() -> dict:
    """Return the live market graph from Neo4j.

    Includes relationship properties (beta, p_value, regime) so edges
    carry meaning.  Returns an honest empty graph if Neo4j is
    unavailable — no static/demo fallback.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN
                  id(n) AS src_id, labels(n)[0] AS src_label,
                  n.name AS src_name, n.asset_class AS src_class,
                  type(r) AS rel, properties(r) AS rel_props,
                  id(m) AS tgt_id, labels(m)[0] AS tgt_label,
                  m.name AS tgt_name, m.asset_class AS tgt_class
                LIMIT 500
            """)
            rows = result.data()

        node_map: dict[int, dict] = {}
        edges = []
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

            rel_type = row["rel"]
            props = row.get("rel_props") or {}
            beta = props.get("beta")
            edge: dict = {
                "source": str(row["src_id"]),
                "target": str(row["tgt_id"]),
                "label":  rel_type,
                "color":  _REL_COLORS.get(rel_type, "#475569"),
                "width":  round(min(abs(beta) * 6, 4), 2) if beta is not None else 1,
            }
            if props:
                for k in ("beta", "p_value", "regime", "factor_group"):
                    if k in props:
                        edge[k] = props[k]
            edges.append(edge)

        return {
            "nodes": list(node_map.values()),
            "edges": edges,
            "source": "neo4j",
            "meta": {"node_count": len(node_map), "edge_count": len(edges)},
        }

    except Exception:
        return {
            "nodes": [],
            "edges": [],
            "source": "unavailable",
            "meta": {"node_count": 0, "edge_count": 0},
        }


@router.get("/graph/stats")
def get_graph_stats() -> dict:
    """Return live node/edge counts from the Neo4j market graph.

    Falls back to zeros if Neo4j is unavailable.
    """
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
