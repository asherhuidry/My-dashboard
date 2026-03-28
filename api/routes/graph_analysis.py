"""Graph structural analysis and intelligence API routes.

Exposes read-only structural insights extracted from the Neo4j market
graph: exposure-profile similarity, regime divergence, bridge factors,
degree centrality, and an integrated intelligence report with
provenance, edge-level diffs, and ranked insights.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/graph/analysis")
def graph_analysis(
    top_n: int = Query(20, ge=1, le=100, description="Max results per section"),
) -> dict:
    """Return full structural analysis of the market graph.

    Runs four complementary read-only analyses over the existing graph:
    - exposure_profiles: assets with similar factor-beta DNA
    - regime_divergence: assets whose betas shift most under stress
    - bridge_factors: macro factors spanning multiple asset classes
    - centrality: most-connected structural hubs

    Each section fails gracefully if Neo4j is unreachable.
    """
    from data.agents.graph_analyzer import (
        analyze_exposure_profiles,
        analyze_regime_divergence,
        analyze_bridge_factors,
        analyze_centrality,
    )

    return {
        "exposure_profiles": analyze_exposure_profiles(top_n=top_n),
        "regime_divergence":  analyze_regime_divergence(top_n=top_n),
        "bridge_factors":     analyze_bridge_factors(),
        "centrality":         analyze_centrality(top_n=top_n),
    }


@router.get("/graph/intelligence")
def graph_intelligence(
    run_id: str | None = Query(None, description="Discovery run_id to link report to"),
) -> dict:
    """Return an integrated graph intelligence report.

    Combines structural analysis, source provenance, edge-level diffs
    against the previous snapshot, and ranked insights.  Answers:
    - What changed in the graph since last run?
    - Which edges are new / stronger / weaker?
    - What sources support each node or edge?
    - What are the most important structural findings right now?
    """
    from data.agents.graph_intelligence import build_intelligence_report
    return build_intelligence_report(run_id=run_id)


@router.get("/graph/intelligence/history")
def intelligence_history(
    limit: int = Query(10, ge=1, le=50, description="Max reports to return"),
) -> dict:
    """Return recent intelligence report history from the evolution trail.

    Each entry includes key metrics, reasoning summaries, and insight counts
    so the UI can show trends over time.
    """
    from db.supabase.client import get_intelligence_reports
    reports = get_intelligence_reports(limit=limit)
    return {
        "reports": reports,
        "count": len(reports),
    }
