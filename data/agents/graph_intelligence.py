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
# 3. Anomaly detection over structural metric time series
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
# 4. Insight ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_insights(
    analysis: dict[str, Any],
    metrics: dict[str, Any],
    edge_changes: dict[str, Any] | None = None,
    anomalies: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate a prioritised list of the most important structural findings.

    Each insight has a type, priority (1 = most urgent), a human-readable
    description, and the supporting evidence that produced it.

    Args:
        analysis:     Output of analyze_graph_structure().
        metrics:      Summary metrics from compute_summary_metrics().
        edge_changes: Optional edge-level diff from compute_edge_changes().
        anomalies:    Optional anomaly detection results from
                      detect_structural_anomalies().

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

    insights.sort(key=lambda i: i["priority"])
    return insights


# ─────────────────────────────────────────────────────────────────────────────
# 5. Combined intelligence report
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

    # 3. Source provenance for all series in the analysis
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

    # 4. Ranked insights (including anomalies when available)
    insights = rank_insights(analysis, metrics, edge_changes, anomaly_result)

    report: dict[str, Any] = {
        "run_id": run_id,
        "previous_timestamp": previous_timestamp,
        "insights": insights,
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

    anomaly_count = len(anomaly_result["anomalies"]) if anomaly_result else 0
    log.info(
        "Intelligence report: %d insights, %d anomalies, %d series tracked, diff=%s",
        len(insights), anomaly_count, len(provenance),
        "yes" if diff else "first run",
    )
    return report
