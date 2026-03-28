"""Graph intelligence layer — integrates analysis, provenance, diffs, and insights.

Sits above graph_analyzer and combines five capabilities into a single
coherent intelligence report:

1. **Run diff** — what changed in the graph since the last snapshot
2. **Edge changes** — which exposure pairs, regime divergences, and
   bridges appeared, disappeared, or shifted
3. **Provenance** — which data source feeds each series in the graph
4. **Ranked insights** — the most important structural findings right
   now, prioritised by actionability
5. **Anomaly detection** — z-score based early warnings when structural
   metrics deviate significantly from their rolling history
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Source provenance
# ─────────────────────────────────────────────────────────────────────────────

# Lazy-loaded lookup sets — built once on first call
_SOURCE_CACHE: dict[str, set[str]] | None = None


def _build_source_cache() -> dict[str, set[str]]:
    """Build source → series_ids lookup from universe definitions."""
    from data.ingest.universe import (
        EQUITIES, CRYPTO_YF, ETFS, FOREX, COMMODITIES, MACRO_SERIES,
    )
    return {
        "yfinance":  set(EQUITIES) | set(CRYPTO_YF) | set(ETFS)
                     | set(FOREX) | set(COMMODITIES),
        "fred_api":  {s[0] for s in MACRO_SERIES},
    }


def resolve_series_source(series_id: str) -> dict[str, str | None]:
    """Resolve which data source feeds a given series.

    Uses universe.py definitions to map series_id → source_id.
    Falls back to heuristic matching for series not in the universe.

    Args:
        series_id: Ticker symbol or FRED series ID.

    Returns:
        Dict with source_id and source_name.
    """
    global _SOURCE_CACHE
    if _SOURCE_CACHE is None:
        try:
            _SOURCE_CACHE = _build_source_cache()
        except Exception:
            _SOURCE_CACHE = {"yfinance": set(), "fred_api": set()}

    for source_id, series_set in _SOURCE_CACHE.items():
        if series_id in series_set:
            names = {
                "yfinance": "Yahoo Finance",
                "fred_api": "FRED (Federal Reserve)",
            }
            return {"source_id": source_id, "source_name": names.get(source_id, source_id)}

    # Heuristic fallback: FRED series are typically uppercase with no special chars
    # yfinance tickers can have - or = or .
    if any(c in series_id for c in "-=.^"):
        return {"source_id": "yfinance", "source_name": "Yahoo Finance (inferred)"}
    return {"source_id": "unknown", "source_name": None}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Edge-level changes between snapshots
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_changes(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Compare two structural analysis results at the edge level.

    Detects which exposure pairs, regime divergences, and bridge
    factors appeared, disappeared, or shifted between snapshots.

    Args:
        current:  Analysis output from analyze_graph_structure().
        previous: Previous snapshot's analysis output.

    Returns:
        Dict with edge-level changes per analysis section.
    """
    changes: dict[str, Any] = {}

    # ── Exposure profile changes ─────────────────────────────────────
    cur_pairs = {
        (p["asset_a"], p["asset_b"]): p
        for p in current.get("exposure_profiles", {}).get("similar_pairs", [])
    }
    prev_pairs = {
        (p["asset_a"], p["asset_b"]): p
        for p in previous.get("exposure_profiles", {}).get("similar_pairs", [])
    }
    new_pairs = []
    lost_pairs = []
    shifted_pairs = []

    for key, p in cur_pairs.items():
        if key not in prev_pairs:
            new_pairs.append(p)
        else:
            prev_sim = prev_pairs[key]["cosine_similarity"]
            cur_sim = p["cosine_similarity"]
            delta = round(cur_sim - prev_sim, 4)
            if abs(delta) >= 0.05:
                shifted_pairs.append({
                    **p,
                    "previous_similarity": prev_sim,
                    "delta": delta,
                    "direction": "strengthened" if delta > 0 else "weakened",
                })
    for key, p in prev_pairs.items():
        if key not in cur_pairs:
            lost_pairs.append(p)

    changes["exposure_profiles"] = {
        "new": new_pairs,
        "lost": lost_pairs,
        "shifted": shifted_pairs,
    }

    # ── Regime divergence changes ────────────────────────────────────
    cur_divs = {
        (d["asset"], d["factor"], d["regime"]): d
        for d in current.get("regime_divergence", {}).get("divergences", [])
    }
    prev_divs = {
        (d["asset"], d["factor"], d["regime"]): d
        for d in previous.get("regime_divergence", {}).get("divergences", [])
    }

    new_divs = [d for k, d in cur_divs.items() if k not in prev_divs]
    lost_divs = [d for k, d in prev_divs.items() if k not in cur_divs]
    shifted_divs = []
    for key, d in cur_divs.items():
        if key in prev_divs:
            prev_abs = prev_divs[key]["abs_change"]
            delta = round(d["abs_change"] - prev_abs, 6)
            if abs(delta) >= 0.01:
                shifted_divs.append({
                    **d,
                    "previous_abs_change": prev_abs,
                    "delta": delta,
                    "trend": "amplifying" if delta > 0 else "normalizing",
                })

    changes["regime_divergence"] = {
        "new": new_divs,
        "lost": lost_divs,
        "shifted": shifted_divs,
    }

    # ── Bridge factor changes ────────────────────────────────────────
    cur_bridges = {
        b["factor_id"]: b
        for b in current.get("bridge_factors", {}).get("bridges", [])
        if b.get("is_bridge")
    }
    prev_bridges = {
        b["factor_id"]: b
        for b in previous.get("bridge_factors", {}).get("bridges", [])
        if b.get("is_bridge")
    }
    changes["bridge_factors"] = {
        "gained": [cur_bridges[k] for k in cur_bridges if k not in prev_bridges],
        "lost":   [prev_bridges[k] for k in prev_bridges if k not in cur_bridges],
        "expanded": [
            {**cur_bridges[k], "previous_class_count": prev_bridges[k]["class_count"]}
            for k in cur_bridges
            if k in prev_bridges
            and cur_bridges[k]["class_count"] > prev_bridges[k]["class_count"]
        ],
    }

    # ── Centrality ranking changes ───────────────────────────────────
    cur_rank = {
        r["node_id"]: i for i, r in
        enumerate(current.get("centrality", {}).get("ranking", []))
    }
    prev_rank = {
        r["node_id"]: i for i, r in
        enumerate(previous.get("centrality", {}).get("ranking", []))
    }
    movers = []
    for node_id, cur_pos in cur_rank.items():
        prev_pos = prev_rank.get(node_id)
        if prev_pos is not None and prev_pos != cur_pos:
            movers.append({
                "node_id": node_id,
                "previous_rank": prev_pos + 1,
                "current_rank": cur_pos + 1,
                "direction": "up" if cur_pos < prev_pos else "down",
            })
    new_entrants = [
        {"node_id": n, "current_rank": cur_rank[n] + 1}
        for n in cur_rank if n not in prev_rank
    ]
    changes["centrality"] = {
        "movers": sorted(movers, key=lambda m: abs(m["previous_rank"] - m["current_rank"]), reverse=True),
        "new_entrants": new_entrants,
    }

    return changes


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge confidence summary from Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def query_edge_confidence_stats() -> dict[str, Any] | None:
    """Query aggregate edge confidence and evidence stats from Neo4j.

    Returns a summary of how well-supported the graph edges are,
    including mean confidence, evidence accumulation, and the count
    of low-confidence edges.

    Returns:
        Dict with confidence stats, or None if unavailable.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        with driver.session() as sess:
            result = sess.run("""
                MATCH ()-[r]->()
                WHERE r.confidence IS NOT NULL
                WITH r,
                     duration.between(
                       coalesce(r.last_confirmed_at, r.first_seen_at, datetime()),
                       datetime()
                     ).days AS age_days
                RETURN
                  count(r)                                          AS scored_edges,
                  round(avg(r.confidence), 4)                       AS mean_confidence,
                  round(min(r.confidence), 4)                       AS min_confidence,
                  round(percentileDisc(r.confidence, 0.25), 4)      AS p25,
                  round(percentileDisc(r.confidence, 0.50), 4)      AS median,
                  sum(CASE WHEN r.confidence < 0.4 THEN 1 ELSE 0 END) AS low_confidence,
                  sum(CASE WHEN r.evidence_count > 1 THEN 1 ELSE 0 END) AS reconfirmed,
                  round(avg(coalesce(r.evidence_count, 1)), 2)      AS mean_evidence_count,
                  sum(CASE WHEN age_days > 90 AND age_days <= 180 THEN 1 ELSE 0 END) AS stale_edges,
                  sum(CASE WHEN age_days > 180 THEN 1 ELSE 0 END)   AS expired_edges
            """)
            row = result.single()
            if row and row["scored_edges"] > 0:
                return dict(row)
    except Exception as exc:
        log.debug("Edge confidence query unavailable: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Anomaly detection over structural metric time series
# ─────────────────────────────────────────────────────────────────────────────

# Metrics worth monitoring for structural anomalies — these capture the
# highest-signal dimensions of market structure change.
_MONITORED_METRICS: list[str] = [
    "exposure_sim_mean",
    "exposure_sim_max",
    "regime_divergence_mean",
    "regime_divergence_max",
    "regime_amplified_pct",
    "bridge_count",
    "bridge_max_span",
    "centrality_top_degree",
    "centrality_mean_degree",
]

# Human-readable descriptions for anomaly reports
_METRIC_LABELS: dict[str, str] = {
    "exposure_sim_mean":       "mean exposure similarity",
    "exposure_sim_max":        "max exposure similarity",
    "regime_divergence_mean":  "mean regime divergence",
    "regime_divergence_max":   "max regime divergence",
    "regime_amplified_pct":    "% of regime-amplified pairs",
    "bridge_count":            "cross-class bridge count",
    "bridge_max_span":         "max bridge asset-class span",
    "centrality_top_degree":   "top node degree centrality",
    "centrality_mean_degree":  "mean degree centrality",
}

# z-score threshold for flagging an anomaly
_Z_THRESHOLD: float = 2.0

# Minimum number of historical snapshots required for meaningful z-scores
_MIN_HISTORY: int = 3


def detect_structural_anomalies(
    current_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect anomalies in structural metrics using z-scores.

    Compares each monitored metric in the current snapshot against its
    rolling history.  When |z-score| >= 2.0, the metric is flagged as
    anomalous with direction (spike / drop) and magnitude.

    Args:
        current_metrics: Summary metrics from compute_summary_metrics().
        history: List of previous snapshot metric dicts, most-recent first.
                 Each entry is the ``metrics`` sub-dict from a snapshot's
                 ``after_state``.

    Returns:
        Dict with:
        - ``anomalies``: list of anomaly findings
        - ``metrics_checked``: how many metrics were evaluated
        - ``history_depth``: how many snapshots were used
        - ``sufficient_history``: whether enough data existed
    """
    result: dict[str, Any] = {
        "anomalies": [],
        "metrics_checked": 0,
        "history_depth": len(history),
        "sufficient_history": len(history) >= _MIN_HISTORY,
    }

    if len(history) < _MIN_HISTORY:
        return result

    anomalies: list[dict[str, Any]] = []

    for metric_key in _MONITORED_METRICS:
        current_val = current_metrics.get(metric_key)
        if current_val is None:
            continue

        # Build history array, skipping None values
        hist_vals = [
            h[metric_key] for h in history
            if h.get(metric_key) is not None
        ]
        if len(hist_vals) < _MIN_HISTORY:
            continue

        result["metrics_checked"] += 1

        arr = np.array(hist_vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std < 1e-12:
            # No variance in history — skip (constant metric is not anomalous)
            continue

        z = (float(current_val) - mean) / std

        if abs(z) >= _Z_THRESHOLD:
            anomalies.append({
                "metric": metric_key,
                "label": _METRIC_LABELS.get(metric_key, metric_key),
                "current_value": round(float(current_val), 6),
                "rolling_mean": round(mean, 6),
                "rolling_std": round(std, 6),
                "z_score": round(z, 3),
                "direction": "spike" if z > 0 else "drop",
                "severity": "critical" if abs(z) >= 3.0 else "warning",
            })

    anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)
    result["anomalies"] = anomalies
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Insight ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_insights(
    analysis: dict[str, Any],
    metrics: dict[str, Any],
    edge_changes: dict[str, Any] | None = None,
    anomalies: dict[str, Any] | None = None,
    confidence_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate a prioritised list of the most important structural findings.

    Each insight has a type, priority (1 = most urgent), a human-readable
    description, and the supporting evidence that produced it.

    Args:
        analysis:         Output of analyze_graph_structure().
        metrics:          Summary metrics from compute_summary_metrics().
        edge_changes:     Optional edge-level diff from compute_edge_changes().
        anomalies:        Optional anomaly detection results from
                          detect_structural_anomalies().
        confidence_stats: Optional edge confidence summary from
                          query_edge_confidence_stats().

    Returns:
        List of insight dicts, sorted by priority (ascending = most important first).
    """
    insights: list[dict[str, Any]] = []

    # ── Regime divergence insights ───────────────────────────────────
    divs = analysis.get("regime_divergence", {}).get("divergences", [])
    for d in divs[:3]:  # top 3 most divergent
        insights.append({
            "type": "crisis_exposure",
            "priority": 1 if d["abs_change"] >= 0.25 else 2,
            "title": f"{d['asset']} {d['factor']} sensitivity {d['direction']} in {d['regime']}",
            "detail": (
                f"Beta shifts from {d['beta_all']:.3f} (all) to "
                f"{d['beta_regime']:.3f} ({d['regime']}), "
                f"absolute change {d['abs_change']:.3f}"
            ),
            "evidence": d,
        })

    # ── Bridge factor insights ───────────────────────────────────────
    bridges = analysis.get("bridge_factors", {}).get("bridges", [])
    for b in bridges:
        if b.get("is_bridge") and b["class_count"] >= 3:
            insights.append({
                "type": "systemic_bridge",
                "priority": 1,
                "title": f"{b['factor_id']} bridges {b['class_count']} asset classes",
                "detail": (
                    f"Connected to {b['asset_count']} assets across "
                    f"{', '.join(b['asset_classes'])}"
                ),
                "evidence": b,
            })
        elif b.get("is_bridge"):
            insights.append({
                "type": "bridge_factor",
                "priority": 3,
                "title": f"{b['factor_id']} connects {', '.join(b['asset_classes'])}",
                "detail": f"{b['asset_count']} assets, {b['edge_count']} edges",
                "evidence": b,
            })

    # ── Exposure similarity insights ─────────────────────────────────
    pairs = analysis.get("exposure_profiles", {}).get("similar_pairs", [])
    for p in pairs[:2]:  # top 2 most similar
        if p["cosine_similarity"] >= 0.90:
            insights.append({
                "type": "structural_twin",
                "priority": 2,
                "title": f"{p['asset_a']} and {p['asset_b']} share near-identical exposure DNA",
                "detail": (
                    f"Cosine similarity {p['cosine_similarity']:.2f} across "
                    f"{p['shared_factors']} shared factors"
                ),
                "evidence": p,
            })

    # ── Edge change insights (if diff available) ─────────────────────
    if edge_changes:
        new_divs = edge_changes.get("regime_divergence", {}).get("new", [])
        for d in new_divs[:2]:
            insights.append({
                "type": "new_regime_exposure",
                "priority": 1,
                "title": f"NEW: {d['asset']} now {d['direction']} to {d['factor']} in {d['regime']}",
                "detail": f"Beta = {d['beta_regime']:.3f}, absolute shift = {d['abs_change']:.3f}",
                "evidence": d,
            })

        gained = edge_changes.get("bridge_factors", {}).get("gained", [])
        for b in gained:
            insights.append({
                "type": "new_bridge",
                "priority": 1,
                "title": f"NEW BRIDGE: {b['factor_id']} now spans {', '.join(b['asset_classes'])}",
                "detail": f"Connects {b['asset_count']} assets — new systemic channel",
                "evidence": b,
            })

        lost = edge_changes.get("bridge_factors", {}).get("lost", [])
        for b in lost:
            insights.append({
                "type": "lost_bridge",
                "priority": 2,
                "title": f"LOST: {b['factor_id']} no longer a cross-class bridge",
                "detail": "Systemic channel may be weakening",
                "evidence": b,
            })

    # ── Anomaly insights (if detection results available) ──────────
    if anomalies:
        for a in anomalies.get("anomalies", []):
            is_critical = a["severity"] == "critical"
            insights.append({
                "type": "structural_anomaly",
                "priority": 1 if is_critical else 2,
                "title": (
                    f"ANOMALY: {a['label']} {a['direction']} "
                    f"(z={a['z_score']:+.1f})"
                ),
                "detail": (
                    f"Current {a['current_value']:.4g} vs "
                    f"rolling mean {a['rolling_mean']:.4g} "
                    f"(±{a['rolling_std']:.4g}), "
                    f"severity: {a['severity']}"
                ),
                "evidence": a,
            })

    # ── Edge confidence insights ─────────────────────────────────────
    if confidence_stats:
        total = confidence_stats.get("scored_edges", 0)
        low = confidence_stats.get("low_confidence", 0)
        mean_conf = confidence_stats.get("mean_confidence", 1.0)
        reconfirmed = confidence_stats.get("reconfirmed", 0)

        if total > 0 and mean_conf < 0.45:
            insights.append({
                "type": "weak_evidence_base",
                "priority": 2,
                "title": f"Graph evidence base is thin (mean confidence {mean_conf:.2f})",
                "detail": (
                    f"{low}/{total} edges below 0.4 confidence. "
                    f"Re-running discovery may strengthen the graph."
                ),
                "evidence": confidence_stats,
            })
        elif total > 0 and low > 0:
            low_pct = low / total
            if low_pct > 0.30:
                insights.append({
                    "type": "many_weak_edges",
                    "priority": 3,
                    "title": f"{low_pct:.0%} of edges have low confidence",
                    "detail": (
                        f"{low} of {total} scored edges below 0.4. "
                        f"Mean confidence: {mean_conf:.2f}."
                    ),
                    "evidence": confidence_stats,
                })

        if total > 0 and reconfirmed > 0:
            reconf_pct = reconfirmed / total
            if reconf_pct > 0.5:
                insights.append({
                    "type": "strong_reconfirmation",
                    "priority": 3,
                    "title": f"{reconf_pct:.0%} of edges independently reconfirmed",
                    "detail": (
                        f"{reconfirmed} of {total} edges seen in multiple discovery runs. "
                        f"Mean evidence count: {confidence_stats.get('mean_evidence_count', 1):.1f}."
                    ),
                    "evidence": confidence_stats,
                })

        # Staleness insights
        stale = confidence_stats.get("stale_edges", 0)
        expired = confidence_stats.get("expired_edges", 0)
        if total > 0 and expired > 0:
            expired_pct = expired / total
            if expired_pct > 0.10:
                insights.append({
                    "type": "expired_edges",
                    "priority": 2,
                    "title": f"{expired_pct:.0%} of edges have expired (no reconfirmation in 180+ days)",
                    "detail": (
                        f"{expired} of {total} edges are expired. "
                        f"Consider pruning or re-running discovery on affected pairs."
                    ),
                    "evidence": confidence_stats,
                })
        if total > 0 and stale > 0:
            stale_pct = stale / total
            if stale_pct > 0.25:
                insights.append({
                    "type": "stale_edges",
                    "priority": 3,
                    "title": f"{stale_pct:.0%} of edges are stale (90-180 days without reconfirmation)",
                    "detail": (
                        f"{stale} of {total} edges haven't been reconfirmed recently. "
                        f"Their effective confidence is decaying."
                    ),
                    "evidence": confidence_stats,
                })

    insights.sort(key=lambda i: i["priority"])
    return insights


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph-state reasoning summaries
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_reasoning_summary(
    analysis: dict[str, Any],
    metrics: dict[str, Any],
    confidence_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Distil structural analysis into high-level reasoning conclusions.

    Answers the questions:
    - What is the single most influential macro factor?
    - Which asset class has the highest aggregate exposure?
    - How healthy / trustworthy is the graph overall?
    - What is the dominant structural pattern right now?

    These summaries power the intelligence narrative and can feed
    into LLM-generated briefings.

    Args:
        analysis:         Output of analyze_graph_structure().
        metrics:          Summary metrics from compute_summary_metrics().
        confidence_stats: Optional edge confidence summary.

    Returns:
        Dict with reasoning conclusions.
    """
    reasoning: dict[str, Any] = {}

    # ── Most influential factor ────────────────────────────────────────
    bridges = analysis.get("bridge_factors", {}).get("bridges", [])
    if bridges:
        top_bridge = max(
            [b for b in bridges if b.get("is_bridge")],
            key=lambda b: (b.get("class_count", 0), b.get("asset_count", 0)),
            default=None,
        )
        if top_bridge:
            reasoning["most_influential_factor"] = {
                "factor_id": top_bridge["factor_id"],
                "class_count": top_bridge["class_count"],
                "asset_count": top_bridge["asset_count"],
                "asset_classes": top_bridge.get("asset_classes", []),
                "narrative": (
                    f"{top_bridge['factor_id']} is the most influential factor, "
                    f"connecting {top_bridge['asset_count']} assets across "
                    f"{top_bridge['class_count']} asset classes."
                ),
            }

    # ── Most exposed asset class ───────────────────────────────────────
    divs = analysis.get("regime_divergence", {}).get("divergences", [])
    if divs:
        # Count divergences per asset and find the most exposed
        asset_div_counts: dict[str, float] = {}
        for d in divs:
            asset = d.get("asset", "")
            asset_div_counts[asset] = asset_div_counts.get(asset, 0) + d.get("abs_change", 0)
        if asset_div_counts:
            most_exposed = max(asset_div_counts.items(), key=lambda x: x[1])
            reasoning["most_exposed_asset"] = {
                "asset": most_exposed[0],
                "total_divergence": round(most_exposed[1], 4),
                "divergence_count": sum(1 for d in divs if d["asset"] == most_exposed[0]),
                "narrative": (
                    f"{most_exposed[0]} shows the highest regime sensitivity "
                    f"with total divergence of {most_exposed[1]:.3f} across "
                    f"{sum(1 for d in divs if d['asset'] == most_exposed[0])} factor exposures."
                ),
            }

    # ── Structural health score ────────────────────────────────────────
    health_components: dict[str, float] = {}

    # Confidence quality (0-1)
    if confidence_stats and confidence_stats.get("scored_edges", 0) > 0:
        health_components["confidence"] = min(
            confidence_stats.get("mean_confidence", 0.5) / 0.7, 1.0
        )
        # Reconfirmation ratio
        reconf = confidence_stats.get("reconfirmed", 0) / confidence_stats["scored_edges"]
        health_components["reconfirmation"] = min(reconf / 0.5, 1.0)
        # Freshness (inverse of staleness)
        stale = confidence_stats.get("stale_edges", 0) + confidence_stats.get("expired_edges", 0)
        health_components["freshness"] = max(
            1.0 - stale / confidence_stats["scored_edges"], 0.0
        )

    # Coverage
    centrality = analysis.get("centrality", {}).get("ranking", [])
    if centrality:
        health_components["coverage"] = min(len(centrality) / 50, 1.0)

    # Bridge diversity
    active_bridges = [b for b in bridges if b.get("is_bridge")]
    if active_bridges:
        health_components["bridge_diversity"] = min(len(active_bridges) / 5, 1.0)

    if health_components:
        overall = round(sum(health_components.values()) / len(health_components), 3)
        reasoning["structural_health"] = {
            "score": overall,
            "components": {k: round(v, 3) for k, v in health_components.items()},
            "grade": (
                "excellent" if overall >= 0.8
                else "good" if overall >= 0.6
                else "fair" if overall >= 0.4
                else "poor"
            ),
            "narrative": (
                f"Graph structural health is {overall:.0%} "
                f"({'excellent' if overall >= 0.8 else 'good' if overall >= 0.6 else 'fair' if overall >= 0.4 else 'poor'}). "
                f"Components: {', '.join(f'{k}={v:.0%}' for k, v in health_components.items())}."
            ),
        }

    # ── Dominant structural pattern ────────────────────────────────────
    pairs = analysis.get("exposure_profiles", {}).get("similar_pairs", [])
    high_sim = [p for p in pairs if p.get("cosine_similarity", 0) >= 0.85]
    if high_sim:
        reasoning["dominant_pattern"] = {
            "type": "convergence",
            "pair_count": len(high_sim),
            "top_pair": {
                "assets": [high_sim[0]["asset_a"], high_sim[0]["asset_b"]],
                "similarity": high_sim[0]["cosine_similarity"],
            },
            "narrative": (
                f"{len(high_sim)} asset pairs show convergent exposure profiles "
                f"(cosine ≥ 0.85). Top pair: {high_sim[0]['asset_a']} & "
                f"{high_sim[0]['asset_b']} at {high_sim[0]['cosine_similarity']:.2f}."
            ),
        }
    elif divs:
        amplified = [d for d in divs if d.get("direction") == "amplifies"]
        if len(amplified) > len(divs) * 0.5:
            reasoning["dominant_pattern"] = {
                "type": "regime_amplification",
                "amplified_count": len(amplified),
                "total_divergences": len(divs),
                "narrative": (
                    f"Dominant pattern is regime amplification: "
                    f"{len(amplified)}/{len(divs)} divergences are amplifying."
                ),
            }

    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined intelligence report
# ─────────────────────────────────────────────────────────────────────────────

def build_intelligence_report(run_id: str | None = None) -> dict[str, Any]:
    """Build a complete graph intelligence report.

    Integrates structural analysis, source provenance, edge-level diffs
    against the previous snapshot, and ranked insights into one coherent
    response that can answer:
    - What changed in the graph this run?
    - Which edges are new / stronger / weaker?
    - What sources support this edge or node?
    - What are the most important structural insights right now?

    Args:
        run_id: Optional discovery run_id to link this report to.

    Returns:
        Complete intelligence report dict.
    """
    from data.agents.graph_analyzer import (
        analyze_graph_structure,
        compute_summary_metrics,
        compute_snapshot_diff,
    )

    # 1. Current analysis
    analysis = analyze_graph_structure()
    metrics = compute_summary_metrics(analysis)

    # 2. Previous snapshots → diffs + anomaly detection
    diff: dict[str, Any] | None = None
    edge_changes: dict[str, Any] | None = None
    anomaly_result: dict[str, Any] | None = None
    previous_timestamp: str | None = None

    try:
        from db.supabase.client import get_structural_snapshots
        snapshots = get_structural_snapshots(limit=20)
        if snapshots:
            # Most recent snapshot for edge-level diff
            prev_state = snapshots[0].get("after_state", {})
            prev_metrics = prev_state.get("metrics", {})
            prev_analysis = prev_state.get("analysis", {})
            if prev_metrics:
                diff = compute_snapshot_diff(metrics, prev_metrics)
            if prev_analysis:
                edge_changes = compute_edge_changes(analysis, prev_analysis)
            previous_timestamp = snapshots[0].get("created_at")

            # Rolling history for anomaly detection
            history_metrics = [
                s.get("after_state", {}).get("metrics", {})
                for s in snapshots
                if s.get("after_state", {}).get("metrics")
            ]
            if history_metrics:
                anomaly_result = detect_structural_anomalies(
                    metrics, history_metrics,
                )
    except Exception as exc:
        log.warning("Could not fetch previous snapshot: %s", exc)

    # 3. Edge confidence summary
    confidence_stats = query_edge_confidence_stats()

    # 4. Source provenance for all series in the analysis
    all_series: set[str] = set()
    for p in analysis.get("exposure_profiles", {}).get("similar_pairs", []):
        all_series.add(p["asset_a"])
        all_series.add(p["asset_b"])
    for d in analysis.get("regime_divergence", {}).get("divergences", []):
        all_series.add(d["asset"])
    for b in analysis.get("bridge_factors", {}).get("bridges", []):
        all_series.add(b["factor_id"])
        for a in b.get("assets", []):
            all_series.add(a)
    for r in analysis.get("centrality", {}).get("ranking", []):
        all_series.add(r["node_id"])

    provenance = {s: resolve_series_source(s) for s in sorted(all_series) if s}

    # 5. Ranked insights (including anomalies and confidence when available)
    insights = rank_insights(
        analysis, metrics, edge_changes, anomaly_result, confidence_stats,
    )

    # 6. Graph-state reasoning summary
    reasoning = compute_graph_reasoning_summary(analysis, metrics, confidence_stats)

    report: dict[str, Any] = {
        "run_id": run_id,
        "previous_timestamp": previous_timestamp,
        "insights": insights,
        "reasoning": reasoning,
        "metrics": metrics,
        "provenance": provenance,
        "analysis": analysis,
    }
    if diff:
        report["metric_diff"] = diff
    if edge_changes:
        report["edge_changes"] = edge_changes
    if anomaly_result:
        report["anomalies"] = anomaly_result
    if confidence_stats:
        report["confidence"] = confidence_stats

    anomaly_count = len(anomaly_result["anomalies"]) if anomaly_result else 0
    log.info(
        "Intelligence report: %d insights, %d anomalies, %d series tracked, diff=%s",
        len(insights), anomaly_count, len(provenance),
        "yes" if diff else "first run",
    )

    # Persist report to evolution trail for history
    try:
        from db.supabase.client import persist_intelligence_report
        persist_intelligence_report(report, run_id=run_id)
    except Exception as exc:
        log.warning("Could not persist intelligence report: %s", exc)

    return report
