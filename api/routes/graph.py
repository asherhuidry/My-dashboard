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


@router.get("/graph/knowledge")
def get_knowledge_graph() -> dict:
    """
    Return Neo4j-style knowledge graph nodes and relationships.
    Falls back to a static representative graph if Neo4j is unavailable.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN
                  id(n) AS src_id, labels(n)[0] AS src_label, n.name AS src_name,
                  type(r) AS rel,
                  id(m) AS tgt_id, labels(m)[0] AS tgt_label, m.name AS tgt_name
                LIMIT 300
            """)
            rows = result.data()

        node_map: dict[int, dict] = {}
        edges = []
        label_colors = {
            "Asset":          "#3b82f6",
            "Sector":         "#8b5cf6",
            "MacroIndicator": "#10b981",
            "Signal":         "#f59e0b",
            "Model":          "#ef4444",
            "Pattern":        "#06b6d4",
        }
        for row in rows:
            for nid, nlabel, nname in [
                (row["src_id"], row["src_label"], row["src_name"]),
                (row["tgt_id"], row["tgt_label"], row["tgt_name"]),
            ]:
                if nid not in node_map:
                    node_map[nid] = {
                        "id":    str(nid),
                        "label": nname or str(nid),
                        "type":  nlabel or "Node",
                        "color": label_colors.get(nlabel, "#6b7280"),
                    }
            edges.append({
                "source": str(row["src_id"]),
                "target": str(row["tgt_id"]),
                "label":  row["rel"],
            })

        return {"nodes": list(node_map.values()), "edges": edges, "source": "neo4j"}

    except Exception:
        # Static representative graph when Neo4j not populated
        return _static_knowledge_graph()


def _static_knowledge_graph() -> dict:
    """Demo knowledge graph showing system architecture."""
    nodes = [
        # Data sources
        {"id": "yf",    "label": "yfinance",     "type": "DataSource",    "color": "#06b6d4", "group": "source"},
        {"id": "fred",  "label": "FRED API",      "type": "DataSource",    "color": "#06b6d4", "group": "source"},
        {"id": "av",    "label": "Alpha Vantage", "type": "DataSource",    "color": "#06b6d4", "group": "source"},
        # Storage
        {"id": "ts",    "label": "TimescaleDB",   "type": "Database",      "color": "#3b82f6", "group": "db"},
        {"id": "qd",    "label": "Qdrant",        "type": "VectorDB",      "color": "#3b82f6", "group": "db"},
        {"id": "neo",   "label": "Neo4j",         "type": "GraphDB",       "color": "#3b82f6", "group": "db"},
        {"id": "sb",    "label": "Supabase",      "type": "Database",      "color": "#3b82f6", "group": "db"},
        # Assets
        {"id": "aapl",  "label": "AAPL",          "type": "Asset",         "color": "#a78bfa", "group": "asset"},
        {"id": "btc",   "label": "BTC",            "type": "Asset",         "color": "#f59e0b", "group": "asset"},
        {"id": "spy",   "label": "SPY",            "type": "Asset",         "color": "#a78bfa", "group": "asset"},
        {"id": "nvda",  "label": "NVDA",           "type": "Asset",         "color": "#a78bfa", "group": "asset"},
        # Features/ML
        {"id": "feat",  "label": "Features (74)", "type": "Pipeline",      "color": "#10b981", "group": "ml"},
        {"id": "lstm",  "label": "LSTM Model",     "type": "Model",         "color": "#ef4444", "group": "ml"},
        {"id": "sig",   "label": "Signals",        "type": "Signal",        "color": "#f59e0b", "group": "output"},
        # Macro
        {"id": "gdp",   "label": "GDP",            "type": "MacroIndicator","color": "#10b981", "group": "macro"},
        {"id": "cpi",   "label": "CPI",            "type": "MacroIndicator","color": "#10b981", "group": "macro"},
        {"id": "rates", "label": "Fed Rates",      "type": "MacroIndicator","color": "#10b981", "group": "macro"},
        # Sectors
        {"id": "tech",  "label": "Technology",     "type": "Sector",        "color": "#8b5cf6", "group": "sector"},
        {"id": "fin",   "label": "Financials",     "type": "Sector",        "color": "#8b5cf6", "group": "sector"},
        # Agents
        {"id": "noise", "label": "Noise Filter",   "type": "Agent",         "color": "#f97316", "group": "agent"},
        {"id": "arch",  "label": "Architect",      "type": "Agent",         "color": "#f97316", "group": "agent"},
    ]

    edges = [
        {"source": "yf",   "target": "ts",    "label": "WRITES_TO"},
        {"source": "fred", "target": "ts",    "label": "WRITES_TO"},
        {"source": "av",   "target": "ts",    "label": "WRITES_TO"},
        {"source": "ts",   "target": "feat",  "label": "FEEDS"},
        {"source": "ts",   "target": "noise", "label": "SCANNED_BY"},
        {"source": "feat", "target": "lstm",  "label": "TRAINS"},
        {"source": "feat", "target": "qd",    "label": "EMBEDDED_IN"},
        {"source": "lstm", "target": "sig",   "label": "GENERATES"},
        {"source": "sig",  "target": "sb",    "label": "STORED_IN"},
        {"source": "aapl", "target": "tech",  "label": "BELONGS_TO"},
        {"source": "nvda", "target": "tech",  "label": "BELONGS_TO"},
        {"source": "aapl", "target": "feat",  "label": "HAS_FEATURES"},
        {"source": "btc",  "target": "feat",  "label": "HAS_FEATURES"},
        {"source": "spy",  "target": "feat",  "label": "HAS_FEATURES"},
        {"source": "nvda", "target": "feat",  "label": "HAS_FEATURES"},
        {"source": "gdp",  "target": "feat",  "label": "MACRO_INPUT"},
        {"source": "cpi",  "target": "feat",  "label": "MACRO_INPUT"},
        {"source": "rates","target": "feat",  "label": "MACRO_INPUT"},
        {"source": "aapl", "target": "btc",   "label": "CORRELATED_WITH"},
        {"source": "spy",  "target": "aapl",  "label": "CORRELATED_WITH"},
        {"source": "gdp",  "target": "sb",    "label": "STORED_IN"},
        {"source": "cpi",  "target": "sb",    "label": "STORED_IN"},
        {"source": "arch", "target": "sb",    "label": "WRITES_TO"},
        {"source": "arch", "target": "neo",   "label": "READS"},
        {"source": "noise","target": "sb",    "label": "QUARANTINE_TO"},
        {"source": "aapl", "target": "neo",   "label": "NODE_IN"},
        {"source": "btc",  "target": "neo",   "label": "NODE_IN"},
        {"source": "tech", "target": "neo",   "label": "NODE_IN"},
    ]

    return {"nodes": nodes, "edges": edges, "source": "static"}


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
