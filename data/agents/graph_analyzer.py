"""Structural analysis layer for the market graph.

Read-only analysis that extracts non-obvious patterns from the existing
Neo4j graph without adding nodes or edges.  Four complementary analyses:

1. **Exposure profiles** — cosine similarity between asset factor-beta
   vectors.  Assets with similar exposure DNA behave similarly for
   structural reasons, not just because their prices happened to move
   together.

2. **Regime divergence** — assets whose sensitivity betas change most
   between the full sample and bear/stress regimes.  High divergence
   flags hidden crisis exposure that static analysis misses.

3. **Bridge factors** — macro factors (MacroIndicator nodes) that
   connect assets across multiple asset classes.  These are the
   transmission channels through which shocks propagate across
   otherwise unrelated markets.

4. **Centrality** — degree centrality ranking so the most connected
   (and therefore most structurally important) nodes surface first.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from db.neo4j.client import run_read_query

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_graph_structure() -> dict[str, Any]:
    """Run all structural analyses and return a combined report.

    Each sub-analysis is independent and fails gracefully — a Neo4j
    outage or empty graph produces empty results, not exceptions.

    Returns:
        Dict with keys: exposure_profiles, regime_divergence,
        bridge_factors, centrality.
    """
    return {
        "exposure_profiles": analyze_exposure_profiles(),
        "regime_divergence": analyze_regime_divergence(),
        "bridge_factors":    analyze_bridge_factors(),
        "centrality":        analyze_centrality(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exposure-profile similarity
# ─────────────────────────────────────────────────────────────────────────────

_EXPOSURE_QUERY = (
    "MATCH (a:Asset)-[r:SENSITIVE_TO {regime: 'all'}]->(m:MacroIndicator) "
    "RETURN a.ticker AS asset, a.asset_class AS asset_class, "
    "r.factor_group AS factor, r.beta AS beta"
)


def analyze_exposure_profiles(top_n: int = 20) -> dict[str, Any]:
    """Find assets with the most similar factor-exposure profiles.

    Builds a beta vector per asset across all factor groups, then
    computes pairwise cosine similarity.  High similarity means two
    assets share the same macro-structural DNA — useful for portfolio
    construction and substitution analysis.

    Args:
        top_n: Number of most-similar pairs to return.

    Returns:
        Dict with similar_pairs list, asset_count, factor_count.
    """
    try:
        rows = run_read_query(_EXPOSURE_QUERY, {})
    except Exception as exc:
        log.warning("Exposure profile query failed: %s", exc)
        return {"similar_pairs": [], "asset_count": 0, "factor_count": 0}

    if not rows:
        return {"similar_pairs": [], "asset_count": 0, "factor_count": 0}

    # Build per-asset factor vectors
    factors = sorted({r["factor"] for r in rows})
    factor_idx = {f: i for i, f in enumerate(factors)}

    asset_meta: dict[str, str] = {}  # ticker → asset_class
    vectors: dict[str, np.ndarray] = {}

    for r in rows:
        asset = r["asset"]
        if asset not in vectors:
            vectors[asset] = np.zeros(len(factors))
            asset_meta[asset] = r.get("asset_class") or "unknown"
        vectors[asset][factor_idx[r["factor"]]] = r["beta"]

    # Pairwise cosine similarity
    assets = sorted(vectors.keys())
    pairs: list[dict[str, Any]] = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            va, vb = vectors[a], vectors[b]
            norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
            if norm_a == 0 or norm_b == 0:
                continue
            cos_sim = float(np.dot(va, vb) / (norm_a * norm_b))
            shared = int(np.sum((va != 0) & (vb != 0)))
            pairs.append({
                "asset_a":       a,
                "asset_b":       b,
                "class_a":       asset_meta[a],
                "class_b":       asset_meta[b],
                "cosine_similarity": round(cos_sim, 4),
                "shared_factors": shared,
            })

    pairs.sort(key=lambda p: abs(p["cosine_similarity"]), reverse=True)

    log.info("Exposure profiles: %d assets, %d factors, %d pairs computed",
             len(assets), len(factors), len(pairs))
    return {
        "similar_pairs": pairs[:top_n],
        "asset_count":   len(assets),
        "factor_count":  len(factors),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regime divergence
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_QUERY = (
    "MATCH (a:Asset)-[r:SENSITIVE_TO]->(m:MacroIndicator) "
    "RETURN a.ticker AS asset, r.factor_group AS factor, "
    "r.regime AS regime, r.beta AS beta, r.strength AS strength"
)


def analyze_regime_divergence(top_n: int = 20) -> dict[str, Any]:
    """Find assets whose factor betas shift most across regimes.

    For each (asset, factor) pair that has both an all-regime and a
    bear/stress beta, computes the absolute change.  Large shifts
    reveal hidden crisis sensitivity — the most actionable structural
    insight for risk management.

    Args:
        top_n: Number of most-divergent entries to return.

    Returns:
        Dict with divergences list and summary counts.
    """
    try:
        rows = run_read_query(_REGIME_QUERY, {})
    except Exception as exc:
        log.warning("Regime divergence query failed: %s", exc)
        return {"divergences": [], "pairs_analyzed": 0}

    if not rows:
        return {"divergences": [], "pairs_analyzed": 0}

    # Index: (asset, factor) → {regime: beta}
    betas: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        key = (r["asset"], r["factor"])
        betas.setdefault(key, {})[r["regime"]] = r["beta"]

    divergences: list[dict[str, Any]] = []
    for (asset, factor), regime_betas in betas.items():
        base = regime_betas.get("all")
        if base is None:
            continue
        for regime in ("bear", "stress"):
            regime_beta = regime_betas.get(regime)
            if regime_beta is None:
                continue
            abs_change = abs(regime_beta - base)
            # Relative change (guard against zero base)
            rel_change = (abs_change / abs(base)) if abs(base) > 1e-9 else None
            divergences.append({
                "asset":        asset,
                "factor":       factor,
                "regime":       regime,
                "beta_all":     round(base, 6),
                "beta_regime":  round(regime_beta, 6),
                "abs_change":   round(abs_change, 6),
                "rel_change":   round(rel_change, 4) if rel_change is not None else None,
                "direction":    "amplified" if abs(regime_beta) > abs(base) else "dampened",
            })

    divergences.sort(key=lambda d: d["abs_change"], reverse=True)

    log.info("Regime divergence: %d (asset, factor) pairs, %d divergences found",
             len(betas), len(divergences))
    return {
        "divergences":    divergences[:top_n],
        "pairs_analyzed": len(betas),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bridge factors
# ─────────────────────────────────────────────────────────────────────────────

_BRIDGE_QUERY = (
    "MATCH (a:Asset)-[r:SENSITIVE_TO]->(m:MacroIndicator) "
    "WITH m, collect(DISTINCT a.asset_class) AS classes, "
    "     collect(DISTINCT a.ticker) AS assets, "
    "     count(r) AS edge_count "
    "RETURN m.series_id AS factor_id, m.name AS factor_name, "
    "       classes, assets, edge_count "
    "ORDER BY size(classes) DESC, edge_count DESC"
)


def analyze_bridge_factors() -> dict[str, Any]:
    """Find macro factors that connect different asset clusters.

    A bridge factor is a MacroIndicator connected via SENSITIVE_TO to
    assets in 2+ asset classes.  These are the transmission channels
    through which shocks propagate — e.g., if oil sensitivity spans
    equities, commodities, and forex, an oil shock affects all three.

    Returns:
        Dict with bridges list and summary counts.
    """
    try:
        rows = run_read_query(_BRIDGE_QUERY, {})
    except Exception as exc:
        log.warning("Bridge factor query failed: %s", exc)
        return {"bridges": [], "total_factors": 0}

    if not rows:
        return {"bridges": [], "total_factors": 0}

    bridges: list[dict[str, Any]] = []
    for r in rows:
        classes = [c for c in (r["classes"] or []) if c]  # filter None
        bridges.append({
            "factor_id":    r["factor_id"],
            "factor_name":  r["factor_name"],
            "asset_classes": classes,
            "class_count":  len(classes),
            "assets":       r["assets"] or [],
            "asset_count":  len(r["assets"] or []),
            "edge_count":   r["edge_count"],
            "is_bridge":    len(classes) >= 2,
        })

    actual_bridges = [b for b in bridges if b["is_bridge"]]
    log.info("Bridge factors: %d total factors, %d bridge across 2+ classes",
             len(bridges), len(actual_bridges))
    return {
        "bridges":       bridges,
        "total_factors": len(bridges),
        "bridge_count":  len(actual_bridges),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Centrality
# ─────────────────────────────────────────────────────────────────────────────

_CENTRALITY_QUERY = (
    "MATCH (n)-[r]-() "
    "WITH n, labels(n)[0] AS label, count(r) AS degree "
    "RETURN label, "
    "  CASE label "
    "    WHEN 'Asset' THEN n.ticker "
    "    WHEN 'MacroIndicator' THEN n.series_id "
    "    WHEN 'Sector' THEN n.name "
    "    ELSE coalesce(n.ticker, n.series_id, n.name, toString(id(n))) "
    "  END AS node_id, "
    "  degree "
    "ORDER BY degree DESC "
    "LIMIT $limit"
)


def analyze_centrality(top_n: int = 25) -> dict[str, Any]:
    """Rank nodes by degree centrality (total relationship count).

    The most-connected nodes are the structural hubs of the graph.
    High-degree Assets are exposed to many factors; high-degree
    MacroIndicators influence many assets.

    Args:
        top_n: Number of top nodes to return.

    Returns:
        Dict with ranking list and summary.
    """
    try:
        rows = run_read_query(_CENTRALITY_QUERY, {"limit": top_n})
    except Exception as exc:
        log.warning("Centrality query failed: %s", exc)
        return {"ranking": [], "total_ranked": 0}

    if not rows:
        return {"ranking": [], "total_ranked": 0}

    ranking = [
        {
            "node_id": r["node_id"],
            "label":   r["label"],
            "degree":  r["degree"],
        }
        for r in rows
    ]

    log.info("Centrality: top node %s (%s) with degree %d",
             ranking[0]["node_id"] if ranking else "?",
             ranking[0]["label"] if ranking else "?",
             ranking[0]["degree"] if ranking else 0)
    return {
        "ranking":      ranking,
        "total_ranked": len(ranking),
    }
