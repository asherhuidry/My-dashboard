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

Snapshots persist to evolution_log so the system can track how market
structure evolves week over week — not just what it looks like now.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
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
        bridge_factors, centrality, sector_stress, earnings_exposure.
    """
    per_asset = {
        "exposure_profiles": analyze_exposure_profiles(),
        "regime_divergence": analyze_regime_divergence(),
        "bridge_factors":    analyze_bridge_factors(),
        "centrality":        analyze_centrality(),
    }
    # Sector and earnings analyses build on per-asset results
    per_asset["sector_stress"] = analyze_sector_stress(per_asset)
    per_asset["earnings_exposure"] = analyze_earnings_exposure(per_asset)
    return per_asset


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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sector stress aggregation
# ─────────────────────────────────────────────────────────────────────────────

_SECTOR_MEMBERSHIP_QUERY = (
    "MATCH (a)-[:BELONGS_TO]->(s:Sector) "
    "WHERE a:Asset "
    "RETURN a.ticker AS asset, s.name AS sector"
)


def analyze_sector_stress(
    analysis: dict[str, Any],
    top_n: int = 15,
) -> dict[str, Any]:
    """Aggregate per-asset intelligence up to sector level.

    Combines regime divergence, bridge factor exposure, and exposure
    similarity for each sector's member assets to produce a sector
    stress score.  High-stress sectors have members with large regime
    shifts, heavy bridge factor exposure, and convergent exposure profiles.

    Args:
        analysis: Output of the per-asset analyses (regime_divergence,
                  bridge_factors, exposure_profiles).
        top_n:    Number of sectors to return.

    Returns:
        Dict with ranked sectors, each having a stress score and
        contributing components.
    """
    # 1. Get sector memberships from the graph
    try:
        rows = run_read_query(_SECTOR_MEMBERSHIP_QUERY, {})
    except Exception as exc:
        log.warning("Sector membership query failed: %s", exc)
        return {"sectors": [], "sector_count": 0}

    if not rows:
        return {"sectors": [], "sector_count": 0}

    # Build sector → {members}
    sector_members: dict[str, set[str]] = {}
    for r in rows:
        sector_members.setdefault(r["sector"], set()).add(r["asset"])

    # 2. Aggregate regime divergence per sector
    divs = analysis.get("regime_divergence", {}).get("divergences", [])
    asset_div: dict[str, float] = {}
    asset_div_count: dict[str, int] = {}
    for d in divs:
        a = d.get("asset", "")
        asset_div[a] = asset_div.get(a, 0) + d.get("abs_change", 0)
        asset_div_count[a] = asset_div_count.get(a, 0) + 1

    # 3. Aggregate bridge factor exposure per sector
    bridges = analysis.get("bridge_factors", {}).get("bridges", [])
    asset_bridge_exposure: dict[str, int] = {}
    for b in bridges:
        if b.get("is_bridge"):
            for a in b.get("assets", []):
                asset_bridge_exposure[a] = asset_bridge_exposure.get(a, 0) + 1

    # 4. Compute per-sector scores
    sectors: list[dict[str, Any]] = []
    for sector_name, members in sector_members.items():
        member_list = sorted(members)
        n = len(member_list)
        if n == 0:
            continue

        # Regime divergence: mean total divergence across members
        member_divs = [asset_div.get(m, 0) for m in member_list]
        mean_divergence = float(np.mean(member_divs)) if member_divs else 0
        max_divergence = max(member_divs) if member_divs else 0
        most_divergent = member_list[member_divs.index(max_divergence)] if max_divergence > 0 else None

        # Bridge exposure: mean bridge factor count across members
        member_bridges = [asset_bridge_exposure.get(m, 0) for m in member_list]
        mean_bridge_exposure = float(np.mean(member_bridges)) if member_bridges else 0

        # Divergence breadth: how many members have ANY regime divergence
        members_with_div = sum(1 for m in member_list if asset_div.get(m, 0) > 0)
        div_breadth = members_with_div / n if n > 0 else 0

        # Stress score: weighted combination (regime divergence dominates)
        stress_score = round(
            0.50 * min(mean_divergence / 0.3, 1.0)     # divergence intensity
            + 0.25 * min(mean_bridge_exposure / 3.0, 1.0)  # bridge exposure
            + 0.25 * div_breadth,                          # divergence breadth
            4,
        )

        sectors.append({
            "sector": sector_name,
            "stress_score": stress_score,
            "member_count": n,
            "mean_regime_divergence": round(mean_divergence, 6),
            "max_regime_divergence": round(max_divergence, 6),
            "most_divergent_member": most_divergent,
            "mean_bridge_exposure": round(mean_bridge_exposure, 3),
            "divergence_breadth": round(div_breadth, 3),
            "members": member_list,
        })

    sectors.sort(key=lambda s: s["stress_score"], reverse=True)

    log.info(
        "Sector stress: %d sectors analyzed, top=%s (%.3f)",
        len(sectors),
        sectors[0]["sector"] if sectors else "?",
        sectors[0]["stress_score"] if sectors else 0,
    )
    return {
        "sectors": sectors[:top_n],
        "sector_count": len(sectors),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Earnings exposure analysis
# ─────────────────────────────────────────────────────────────────────────────

_EARNINGS_QUERY = (
    "MATCH (a:Asset)-[:REPORTS]->(e:Event) "
    "WHERE e.event_date >= $from_date AND e.event_date <= $to_date "
    "RETURN a.ticker AS asset, "
    "       e.event_id AS event_id, e.event_date AS event_date, "
    "       e.eps_estimate AS eps_estimate, e.eps_actual AS eps_actual, "
    "       e.eps_surprise_pct AS surprise_pct, e.hour AS hour "
    "ORDER BY e.event_date"
)


def analyze_earnings_exposure(
    analysis: dict[str, Any],
    lookahead_days: int = 21,
    lookback_days: int = 7,
) -> dict[str, Any]:
    """Find upcoming earnings events for structurally important assets.

    Crosses earnings calendar data with centrality and regime divergence
    to flag which upcoming earnings events matter most for the graph.

    An earnings event is high-impact if the reporting asset:
    - Has high degree centrality (structural hub)
    - Has large regime divergence (hidden crisis sensitivity)
    - Is connected to many bridge factors

    Args:
        analysis:        Output of the per-asset analyses.
        lookahead_days:  Days ahead to scan for upcoming earnings.
        lookback_days:   Days back to include recent results.

    Returns:
        Dict with upcoming events ranked by structural importance.
    """
    from datetime import date, timedelta

    today = date.today()
    from_date = (today - timedelta(days=lookback_days)).isoformat()
    to_date = (today + timedelta(days=lookahead_days)).isoformat()

    try:
        rows = run_read_query(_EARNINGS_QUERY, {
            "from_date": from_date, "to_date": to_date,
        })
    except Exception as exc:
        log.warning("Earnings exposure query failed: %s", exc)
        return {"events": [], "event_count": 0}

    if not rows:
        return {"events": [], "event_count": 0}

    # Build asset importance scores from existing analysis
    centrality_map: dict[str, int] = {}
    for r in analysis.get("centrality", {}).get("ranking", []):
        if r.get("label") == "Asset":
            centrality_map[r["node_id"]] = r["degree"]

    # Regime divergence total per asset
    asset_div: dict[str, float] = {}
    for d in analysis.get("regime_divergence", {}).get("divergences", []):
        a = d.get("asset", "")
        asset_div[a] = asset_div.get(a, 0) + d.get("abs_change", 0)

    # Bridge exposure per asset
    asset_bridge: dict[str, int] = {}
    for b in analysis.get("bridge_factors", {}).get("bridges", []):
        if b.get("is_bridge"):
            for a in b.get("assets", []):
                asset_bridge[a] = asset_bridge.get(a, 0) + 1

    events: list[dict[str, Any]] = []
    for r in rows:
        asset = r["asset"]
        degree = centrality_map.get(asset, 0)
        divergence = asset_div.get(asset, 0)
        bridges = asset_bridge.get(asset, 0)

        # Structural importance: weighted score
        importance = round(
            0.40 * min(degree / 10.0, 1.0)
            + 0.35 * min(divergence / 0.3, 1.0)
            + 0.25 * min(bridges / 3.0, 1.0),
            4,
        )

        is_upcoming = r["event_date"] >= today.isoformat() if isinstance(r["event_date"], str) else True
        has_actuals = r.get("eps_actual") is not None

        events.append({
            "asset": asset,
            "event_id": r["event_id"],
            "event_date": r["event_date"],
            "eps_estimate": r.get("eps_estimate"),
            "eps_actual": r.get("eps_actual"),
            "surprise_pct": r.get("surprise_pct"),
            "hour": r.get("hour"),
            "is_upcoming": is_upcoming,
            "has_actuals": has_actuals,
            "structural_importance": importance,
            "degree_centrality": degree,
            "regime_divergence": round(divergence, 4),
            "bridge_exposure": bridges,
        })

    events.sort(key=lambda e: e["structural_importance"], reverse=True)

    upcoming = [e for e in events if e["is_upcoming"]]
    recent = [e for e in events if not e["is_upcoming"] and e["has_actuals"]]

    log.info(
        "Earnings exposure: %d events (%d upcoming, %d recent with actuals)",
        len(events), len(upcoming), len(recent),
    )
    return {
        "events": events,
        "event_count": len(events),
        "upcoming_count": len(upcoming),
        "recent_with_actuals": len(recent),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Structural snapshots — persistence and week-over-week diff
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    """Extract scalar summary metrics from a full structural analysis.

    These metrics are designed for time-series tracking: each is a
    single number that can be compared week over week to detect
    structural drift in the market graph.

    Args:
        analysis: Output of analyze_graph_structure().

    Returns:
        Dict of named scalar metrics.
    """
    metrics: dict[str, Any] = {}

    # ── Exposure profiles ────────────────────────────────────────────
    ep = analysis.get("exposure_profiles", {})
    pairs = ep.get("similar_pairs", [])
    sims = [p["cosine_similarity"] for p in pairs]
    metrics["exposure_asset_count"] = ep.get("asset_count", 0)
    metrics["exposure_factor_count"] = ep.get("factor_count", 0)
    metrics["exposure_pair_count"] = len(sims)
    metrics["exposure_sim_mean"] = round(float(np.mean(sims)), 4) if sims else None
    metrics["exposure_sim_max"] = round(max(sims), 4) if sims else None
    metrics["exposure_top_pair"] = (
        f"{pairs[0]['asset_a']}/{pairs[0]['asset_b']}" if pairs else None
    )

    # ── Regime divergence ────────────────────────────────────────────
    rd = analysis.get("regime_divergence", {})
    divs = rd.get("divergences", [])
    abs_changes = [d["abs_change"] for d in divs]
    amplified = [d for d in divs if d.get("direction") == "amplified"]
    metrics["regime_pairs_analyzed"] = rd.get("pairs_analyzed", 0)
    metrics["regime_divergence_count"] = len(divs)
    metrics["regime_divergence_mean"] = (
        round(float(np.mean(abs_changes)), 6) if abs_changes else None
    )
    metrics["regime_divergence_max"] = (
        round(max(abs_changes), 6) if abs_changes else None
    )
    metrics["regime_amplified_pct"] = (
        round(len(amplified) / len(divs), 4) if divs else None
    )
    metrics["regime_top_shift"] = (
        f"{divs[0]['asset']}/{divs[0]['factor']}/{divs[0]['regime']}"
        if divs else None
    )

    # ── Bridge factors ───────────────────────────────────────────────
    bf = analysis.get("bridge_factors", {})
    bridges = bf.get("bridges", [])
    actual = [b for b in bridges if b.get("is_bridge")]
    metrics["bridge_total_factors"] = bf.get("total_factors", 0)
    metrics["bridge_count"] = len(actual)
    metrics["bridge_max_span"] = (
        max(b["class_count"] for b in actual) if actual else 0
    )
    metrics["bridge_factor_ids"] = sorted(
        b["factor_id"] for b in actual
    )

    # ── Centrality ───────────────────────────────────────────────────
    ct = analysis.get("centrality", {})
    ranking = ct.get("ranking", [])
    degrees = [r["degree"] for r in ranking]
    metrics["centrality_top_node"] = (
        ranking[0]["node_id"] if ranking else None
    )
    metrics["centrality_top_degree"] = ranking[0]["degree"] if ranking else 0
    metrics["centrality_mean_degree"] = (
        round(float(np.mean(degrees)), 2) if degrees else None
    )

    # ── Sector stress ─────────────────────────────────────────────
    ss = analysis.get("sector_stress", {})
    sectors = ss.get("sectors", [])
    stress_scores = [s["stress_score"] for s in sectors]
    metrics["sector_count"] = ss.get("sector_count", 0)
    metrics["sector_stress_max"] = max(stress_scores) if stress_scores else None
    metrics["sector_stress_mean"] = (
        round(float(np.mean(stress_scores)), 4) if stress_scores else None
    )
    metrics["sector_most_stressed"] = (
        sectors[0]["sector"] if sectors else None
    )

    # ── Earnings exposure ─────────────────────────────────────────
    ee = analysis.get("earnings_exposure", {})
    metrics["earnings_upcoming"] = ee.get("upcoming_count", 0)
    metrics["earnings_recent_actuals"] = ee.get("recent_with_actuals", 0)
    events = ee.get("events", [])
    high_impact = [e for e in events if e.get("structural_importance", 0) >= 0.5]
    metrics["earnings_high_impact"] = len(high_impact)

    return metrics


def compute_snapshot_diff(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Compute week-over-week changes between two structural snapshots.

    Compares summary metrics and top-N lists to surface what changed
    in the market graph's structure since the last snapshot.

    Args:
        current:  Current snapshot's summary metrics.
        previous: Previous snapshot's summary metrics.

    Returns:
        Dict describing structural changes.
    """
    changes: dict[str, Any] = {}

    # ── Scalar metric deltas ─────────────────────────────────────────
    delta_keys = [
        "exposure_asset_count", "exposure_factor_count",
        "exposure_sim_mean", "exposure_sim_max",
        "regime_pairs_analyzed", "regime_divergence_count",
        "regime_divergence_mean", "regime_divergence_max",
        "regime_amplified_pct",
        "bridge_count", "bridge_max_span",
        "centrality_top_degree", "centrality_mean_degree",
        "sector_stress_max", "sector_stress_mean",
        "earnings_upcoming", "earnings_high_impact",
    ]
    deltas: dict[str, float | None] = {}
    for key in delta_keys:
        cur = current.get(key)
        prev = previous.get(key)
        if cur is not None and prev is not None:
            deltas[key] = round(cur - prev, 6)
        else:
            deltas[key] = None
    changes["metric_deltas"] = deltas

    # ── Bridge factor changes ────────────────────────────────────────
    cur_bridges = set(current.get("bridge_factor_ids", []))
    prev_bridges = set(previous.get("bridge_factor_ids", []))
    changes["bridges_gained"] = sorted(cur_bridges - prev_bridges)
    changes["bridges_lost"] = sorted(prev_bridges - cur_bridges)

    # ── Top-pair shift ───────────────────────────────────────────────
    changes["exposure_top_pair_changed"] = (
        current.get("exposure_top_pair") != previous.get("exposure_top_pair")
    )
    changes["centrality_top_node_changed"] = (
        current.get("centrality_top_node") != previous.get("centrality_top_node")
    )
    changes["regime_top_shift_changed"] = (
        current.get("regime_top_shift") != previous.get("regime_top_shift")
    )
    changes["sector_most_stressed_changed"] = (
        current.get("sector_most_stressed") != previous.get("sector_most_stressed")
    )

    return changes


def snapshot_graph_structure(run_id: str | None = None) -> dict[str, Any]:
    """Run structural analysis, persist snapshot, compute week-over-week diff.

    This is the main entry point for the weekly pipeline.  It:
    1. Runs the full structural analysis against Neo4j
    2. Computes scalar summary metrics for time-series tracking
    3. Fetches the previous snapshot and computes a diff
    4. Persists everything to evolution_log

    Args:
        run_id: Optional discovery run ID to link the snapshot to.

    Returns:
        Dict with analysis, metrics, diff (if previous exists), and
        snapshot timestamp.
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    # 1. Run analysis
    analysis = analyze_graph_structure()

    # 2. Compute metrics
    metrics = compute_summary_metrics(analysis)

    # 3. Fetch previous snapshot and compute diff
    diff: dict[str, Any] | None = None
    previous_timestamp: str | None = None
    try:
        from db.supabase.client import get_structural_snapshots
        prev_snapshots = get_structural_snapshots(limit=1)
        if prev_snapshots:
            prev_state = prev_snapshots[0].get("after_state", {})
            prev_metrics = prev_state.get("metrics", {})
            if prev_metrics:
                diff = compute_snapshot_diff(metrics, prev_metrics)
                previous_timestamp = prev_snapshots[0].get("created_at")
    except Exception as exc:
        log.warning("Could not fetch previous snapshot for diff: %s", exc)

    # 4. Persist to evolution_log
    snapshot_payload: dict[str, Any] = {
        "timestamp": timestamp,
        "run_id": run_id,
        "metrics": metrics,
        "analysis": {
            "exposure_profiles":  analysis["exposure_profiles"],
            "regime_divergence":  analysis["regime_divergence"],
            "bridge_factors":     analysis["bridge_factors"],
            "centrality":         analysis["centrality"],
            "sector_stress":      analysis.get("sector_stress", {}),
            "earnings_exposure":  analysis.get("earnings_exposure", {}),
        },
    }
    if diff is not None:
        snapshot_payload["diff"] = diff
        snapshot_payload["previous_timestamp"] = previous_timestamp

    try:
        from db.supabase.client import EvolutionLogEntry, log_evolution
        log_evolution(EvolutionLogEntry(
            agent_id="graph_analyzer",
            action="structural_snapshot",
            after_state=snapshot_payload,
        ))
        log.info("Structural snapshot persisted (run_id=%s)", run_id)
    except Exception as exc:
        log.warning("Failed to persist structural snapshot: %s", exc)

    return snapshot_payload
