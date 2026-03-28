"""Intelligence API — correlation discovery, semantic search, macro regime, knowledge graph queries.

Endpoints:
  GET  /api/intelligence/correlations/{symbol}   — top correlated assets/macro series
  GET  /api/intelligence/regime                  — current macro regime classification
  GET  /api/intelligence/supply-chain/{symbol}   — supply chain exposure graph
  GET  /api/intelligence/lead-indicators/{symbol}— leading indicators for a symbol
  POST /api/intelligence/semantic-search         — semantic similarity search over news/filings
  GET  /api/intelligence/social/{symbol}         — social sentiment intelligence
  GET  /api/intelligence/fundamentals-deep/{symbol} — deep fundamental profile with EDGAR
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()


# ── Request/Response models ───────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query:   str
    limit:   int = 10
    filters: dict[str, Any] | None = None


class CorrelationResult(BaseModel):
    series_a:     str
    series_b:     str
    lag_days:     int
    pearson_r:    float
    strength:     str
    relationship_type: str
    granger_p:    float | None = None


# ── Correlations endpoint ─────────────────────────────────────────────────────

@router.get("/intelligence/correlations/{symbol}")
async def get_correlations(
    symbol:     str,
    max_pairs:  int  = Query(20, ge=1, le=100),
    min_abs_r:  float = Query(0.4, ge=0.0, le=1.0),
    include_macro: bool = Query(True),
) -> dict[str, Any]:
    """Find assets/macro series most correlated with the given symbol.

    Returns:
        Top correlations with lag, strength, and relationship type.
    """
    symbol = symbol.upper()

    # Try to get from Neo4j knowledge graph first
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Asset {ticker: $sym})-[r:CORRELATES_WITH]-(b) "
                "WHERE abs(r.pearson_r) >= $min_r "
                "RETURN b.ticker as series_b, r.lag_days as lag, "
                "       r.pearson_r as r_val, r.strength as strength, "
                "       r.type as rel_type, r.granger_p as granger "
                "ORDER BY abs(r.pearson_r) DESC LIMIT $limit",
                sym=symbol, min_r=min_abs_r, limit=max_pairs
            )
            graph_correlations = [dict(row) for row in result]
    except Exception:
        graph_correlations = []

    # If graph is empty, run correlation hunter live
    if not graph_correlations:
        try:
            from data.agents.correlation_hunter import hunt_correlations, find_leading_indicators
            from data.ingest.universe import ETFS, MACRO_SERIES

            macro_daily = [s[0] for s in MACRO_SERIES if s[2] == "daily"][:20]
            candidates  = ETFS[:15] + macro_daily if include_macro else ETFS[:20]

            findings = hunt_correlations(
                symbols      = [symbol],
                macro_series = macro_daily if include_macro else [],
                max_pairs    = max_pairs,
                min_abs_r    = min_abs_r,
            )
            # Filter to findings involving our symbol
            relevant = [
                f for f in findings
                if f.series_a == symbol or f.series_b == symbol
            ][:max_pairs]

            correlations = [
                {
                    "series_b":         f.series_b if f.series_a == symbol else f.series_a,
                    "lag":              f.lag_days,
                    "r_val":            f.pearson_r,
                    "strength":         f.strength,
                    "rel_type":         f.relationship_type,
                    "granger":          f.granger_p,
                    "mutual_info":      f.mutual_info,
                }
                for f in relevant
            ]
        except Exception as exc:
            log.warning("Correlation hunter failed for %s: %s", symbol, exc)
            correlations = []
    else:
        correlations = graph_correlations

    return {
        "symbol":       symbol,
        "correlations": correlations,
        "source":       "neo4j" if graph_correlations else "live_computation",
        "count":        len(correlations),
    }


# ── Macro regime endpoint ──────────────────────────────────────────────────────

@router.get("/intelligence/regime")
async def get_macro_regime() -> dict[str, Any]:
    """Get current macro regime classification and key indicator levels.

    Returns:
        Regime quadrant (goldilocks/reflation/stagflation/deflation_risk),
        sub-scores (growth, inflation, credit, risk), and key macro levels.
    """
    try:
        from data.ingest.macro_expanded import get_latest_macro_snapshot, detect_macro_regime
        from data.ingest.universe import MACRO_SERIES

        key_series = [
            "T10Y2Y", "T10Y3M", "BAMLH0A0HYM2", "VIXCLS",
            "T10YIE", "DFF", "DTWEXBGS", "WALCL",
            "CPIAUCSL", "STLFSI4", "NFCI",
        ]
        latest = get_latest_macro_snapshot(series_ids=key_series)
        regime = detect_macro_regime(latest)

        # Add human-readable interpretation
        quadrant_descriptions = {
            "goldilocks":    "Growth expanding, inflation contained — historically bullish for equities",
            "reflation":     "Growth expanding, inflation elevated — commodities outperform, tighter Fed ahead",
            "stagflation":   "Growth contracting, inflation elevated — historically worst for equities; commodities, TIPS, cash",
            "deflation_risk":"Growth contracting, inflation falling — bonds outperform, defensive sectors",
        }
        regime["description"] = quadrant_descriptions.get(regime.get("quadrant", ""), "")
        regime["latest_values"] = latest

        return regime
    except Exception as exc:
        log.warning("Regime detection failed: %s", exc)
        # Return minimal regime without DB
        return {
            "quadrant":    "unknown",
            "scores":      {},
            "description": "Unable to compute regime — macro data unavailable",
            "error":       str(exc),
        }


# ── Supply chain endpoint ─────────────────────────────────────────────────────

@router.get("/intelligence/supply-chain/{symbol}")
async def get_supply_chain(symbol: str) -> dict[str, Any]:
    """Get supply chain relationships for a company from the knowledge graph.

    Returns:
        Suppliers, customers, and their relationship details.
    """
    symbol = symbol.upper()

    # Try Neo4j first
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            # Direct suppliers (companies that supply TO this symbol)
            suppliers_result = session.run(
                "MATCH (sup:Company)-[r:SUPPLIES_TO]->(cus:Company {symbol: $sym}) "
                "RETURN sup.symbol as supplier, r.product_category as product, "
                "       r.estimated_revenue_pct as rev_pct",
                sym=symbol
            )
            suppliers = [dict(row) for row in suppliers_result]

            # Direct customers (companies this symbol supplies TO)
            customers_result = session.run(
                "MATCH (sup:Company {symbol: $sym})-[r:SUPPLIES_TO]->(cus:Company) "
                "RETURN cus.symbol as customer, r.product_category as product, "
                "       r.estimated_revenue_pct as rev_pct",
                sym=symbol
            )
            customers = [dict(row) for row in customers_result]

            # 2-hop exposure (companies exposed through a shared supplier)
            indirect_result = session.run(
                "MATCH (sym:Company {symbol: $sym})<-[:SUPPLIES_TO]-(shared:Company)"
                "-[:SUPPLIES_TO]->(other:Company) "
                "WHERE other.symbol <> $sym "
                "RETURN other.symbol as symbol, shared.symbol as via, "
                "       count(*) as exposure_count "
                "ORDER BY exposure_count DESC LIMIT 10",
                sym=symbol
            )
            indirect = [dict(row) for row in indirect_result]

        return {
            "symbol":    symbol,
            "suppliers": suppliers,
            "customers": customers,
            "indirect_exposure": indirect,
            "source":    "neo4j",
        }
    except Exception as exc:
        pass

    # Fallback: static supply chain map
    from data.agents.knowledge_builder import SUPPLY_CHAIN_MAP
    suppliers = [
        {"supplier": sup, "product": prod, "rev_pct": rev}
        for sup, cus, prod, rev in SUPPLY_CHAIN_MAP
        if cus == symbol
    ]
    customers = [
        {"customer": cus, "product": prod, "rev_pct": rev}
        for sup, cus, prod, rev in SUPPLY_CHAIN_MAP
        if sup == symbol
    ]

    return {
        "symbol":    symbol,
        "suppliers": suppliers,
        "customers": customers,
        "indirect_exposure": [],
        "source":    "static",
    }


# ── Leading indicators endpoint ────────────────────────────────────────────────

@router.get("/intelligence/lead-indicators/{symbol}")
async def get_leading_indicators(
    symbol: str,
    max_results: int = Query(10, ge=1, le=30),
) -> dict[str, Any]:
    """Find macro/cross-asset series that lead the given symbol.

    Identifies series where changes TODAY predict price changes in the future,
    using Granger causality and rolling lag analysis.

    Returns:
        Ranked list of leading indicators with lag and strength.
    """
    symbol = symbol.upper()

    try:
        from data.agents.correlation_hunter import find_leading_indicators
        from data.ingest.universe import MACRO_SERIES, ETFS

        candidates = [s[0] for s in MACRO_SERIES if s[2] == "daily"][:30]
        candidates += ETFS[:15]

        leaders = find_leading_indicators(
            target     = symbol,
            candidates = candidates,
            max_lag    = 20,
            top_n      = max_results,
        )

        return {
            "symbol":  symbol,
            "leaders": leaders,
            "count":   len(leaders),
        }
    except Exception as exc:
        log.warning("Lead indicator search failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Semantic search endpoint ──────────────────────────────────────────────────

@router.post("/intelligence/semantic-search")
async def semantic_search(req: SemanticSearchRequest) -> dict[str, Any]:
    """Search news, SEC filings, and research notes by semantic similarity.

    Uses Qdrant vector store for fast approximate nearest-neighbor search.

    Args:
        req.query:   Natural language query.
        req.limit:   Max results to return.
        req.filters: Optional payload filters (e.g. {'symbol': 'AAPL'}).

    Returns:
        List of semantically similar documents with scores.
    """
    try:
        from db.qdrant.client import get_client as get_qdrant
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        import anthropic

        client = anthropic.Anthropic()

        # Embed the query using a simple approach
        # In production, use a dedicated embedding model
        # For now, use Claude to summarize then keyword match
        results = _semantic_search_fallback(req.query, req.limit, req.filters)
        return {"results": results, "source": "keyword_fallback"}

    except Exception as exc:
        log.debug("Semantic search failed: %s", exc)
        results = _semantic_search_fallback(req.query, req.limit, req.filters)
        return {"results": results, "source": "keyword_fallback"}


def _semantic_search_fallback(
    query:   str,
    limit:   int,
    filters: dict | None,
) -> list[dict[str, Any]]:
    """Keyword-based search fallback when Qdrant is unavailable."""
    try:
        from data.sources.news import fetch_all_news, web_search
        symbol = (filters or {}).get("symbol", "")
        if symbol:
            news = fetch_all_news(symbol)
            articles = news.get("articles", [])
            # Simple keyword scoring
            query_words = set(query.lower().split())
            scored = []
            for a in articles:
                text = f"{a.get('title','')} {a.get('summary','')}".lower()
                score = sum(1 for w in query_words if w in text) / max(len(query_words), 1)
                if score > 0:
                    scored.append({**a, "relevance_score": round(score, 3)})
            return sorted(scored, key=lambda x: x["relevance_score"], reverse=True)[:limit]
        else:
            results = web_search(query)
            return results[:limit] if isinstance(results, list) else []
    except Exception:
        return []


# ── Social intelligence endpoint ──────────────────────────────────────────────

@router.get("/intelligence/social/{symbol}")
async def get_social_intelligence(
    symbol:          str,
    include_trends:  bool = Query(True),
) -> dict[str, Any]:
    """Fetch comprehensive social sentiment intelligence for a symbol.

    Aggregates Reddit, StockTwits, Google Trends, and options flow.

    Returns:
        Composite sentiment score, platform breakdown, and top posts.
    """
    symbol = symbol.upper()
    try:
        from data.ingest.social_data import fetch_social_intelligence
        result = fetch_social_intelligence(symbol, include_trends=include_trends)
        return result
    except Exception as exc:
        log.warning("Social intelligence failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Deep fundamentals endpoint ────────────────────────────────────────────────

@router.get("/intelligence/fundamentals-deep/{symbol}")
async def get_deep_fundamentals(symbol: str) -> dict[str, Any]:
    """Get comprehensive fundamental profile combining yfinance + SEC EDGAR.

    Includes:
      - All XBRL financial data from EDGAR (revenue, EPS, FCF, debt, etc.)
      - Recent SEC filings (10-K, 10-Q, 8-K)
      - Insider transactions (Form 4)
      - Congressional stock trades (STOCK Act)
      - Key financial ratios from yfinance
      - Institutional holders

    Returns:
        Deep fundamental profile dict.
    """
    symbol = symbol.upper()
    try:
        from data.ingest.edgar_full import (
            get_company_facts,
            get_recent_filings,
            get_insider_transactions,
            get_congress_trades,
        )
        from data.sources.fundamentals import get_full_fundamental_profile

        # Run all in parallel conceptually (Python GIL means sequential here)
        profile = get_full_fundamental_profile(symbol)

        # Enrich with EDGAR data
        try:
            edgar_facts = get_company_facts(symbol)
            profile["edgar_financials"] = edgar_facts
        except Exception as exc:
            profile["edgar_financials"] = {"error": str(exc)}

        try:
            filings = get_recent_filings(symbol, forms=["10-K", "10-Q", "8-K"])
            profile["recent_filings"] = filings[:10]
        except Exception:
            profile["recent_filings"] = []

        try:
            insiders = get_insider_transactions(symbol)
            profile["insider_transactions_edgar"] = insiders[:20]
        except Exception:
            profile["insider_transactions_edgar"] = []

        try:
            congress = get_congress_trades(symbol)
            profile["congress_trades"] = congress[:20]
        except Exception:
            profile["congress_trades"] = []

        return profile

    except Exception as exc:
        log.warning("Deep fundamentals failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Macro dashboard endpoint ──────────────────────────────────────────────────

@router.get("/intelligence/macro-dashboard")
async def get_macro_dashboard() -> dict[str, Any]:
    """Get a comprehensive macro snapshot for dashboard display.

    Returns latest values for key indicators organized by category.
    """
    try:
        from data.ingest.macro_expanded import get_latest_macro_snapshot, detect_macro_regime

        key_series = {
            "rates": ["DFF", "GS2", "GS10", "GS30", "T10Y2Y", "T10Y3M"],
            "inflation": ["CPIAUCSL", "CPILFESL", "T10YIE", "T5YIE", "DFII10"],
            "credit": ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "NFCI"],
            "employment": ["UNRATE", "PAYEMS", "ICSA", "JTSJOL"],
            "commodities": ["DCOILWTICO", "DCOILBRENTEU", "GOLDAMGBD228NLBM", "DHHNGSP"],
            "risk": ["VIXCLS", "OVXCLS", "GVZCLS", "DTWEXBGS"],
            "fed": ["WALCL", "RRPONTSYD", "M2SL"],
        }

        all_ids = [sid for ids in key_series.values() for sid in ids]
        latest  = get_latest_macro_snapshot(series_ids=all_ids)
        regime  = detect_macro_regime(latest)

        # Organize by category
        organized: dict[str, dict[str, float | None]] = {}
        for category, series_ids in key_series.items():
            organized[category] = {sid: latest.get(sid) for sid in series_ids}

        return {
            "regime":     regime,
            "categories": organized,
            "raw":        latest,
        }

    except Exception as exc:
        log.warning("Macro dashboard failed: %s", exc)
        return {"error": str(exc), "regime": {"quadrant": "unknown"}}


# ── Knowledge graph stats endpoint ────────────────────────────────────────────

@router.get("/intelligence/graph-stats")
async def get_graph_stats() -> dict[str, Any]:
    """Get statistics about the knowledge graph (node/edge counts, coverage)."""
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as session:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as count"
            )
            node_counts = {row["label"]: row["count"] for row in counts}

            edge_counts_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
            )
            edge_counts = {row["rel_type"]: row["count"] for row in edge_counts_result}

        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
        }
    except Exception as exc:
        # Return static counts based on what we've defined
        from data.agents.knowledge_builder import SUPPLY_CHAIN_MAP, MACRO_SECTOR_IMPACTS
        return {
            "nodes": {"Company": 25, "MacroIndicator": 18, "Asset": 45},
            "edges": {
                "SUPPLIES_TO": len(SUPPLY_CHAIN_MAP),
                "IMPACTS":     len(MACRO_SECTOR_IMPACTS),
            },
            "total_nodes": 88,
            "total_edges": len(SUPPLY_CHAIN_MAP) + len(MACRO_SECTOR_IMPACTS),
            "source": "static_estimates",
        }
