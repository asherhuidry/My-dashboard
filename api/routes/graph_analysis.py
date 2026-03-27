"""Graph structural analysis API routes.

Exposes read-only structural insights extracted from the Neo4j market
graph: exposure-profile similarity, regime divergence, bridge factors,
and degree centrality.
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
