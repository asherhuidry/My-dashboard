"""Edge confidence scoring — unified 0-1 score from statistical evidence.

Every discovery (correlation, sensitivity, or causal) carries raw statistical
measures (pearson_r, granger_p, t_stat, p_value, r_squared, mutual_info).
This module distils those into a single ``confidence`` float that captures
how trustworthy the edge is, accounting for effect size, statistical
significance, model fit, and categorical strength.

The score is deterministic, interpretable, and designed to serve as:
- An edge weight for GNN attention layers (graph embeddings readiness)
- A filter/sort key for intelligence reports (reasoning over graph state)
- A temporal signal when tracked across runs (anomaly / early-warning)

Each edge type has its own scoring profile because the evidence means
different things:
- CORRELATED_WITH: effect size (|r|) + Granger significance + MI
- SENSITIVE_TO:    t-stat significance + beta magnitude + model fit (R²)
- CAUSES:          Granger significance (primary) + effect size + MI
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

_STRENGTH_SCORES: dict[str, float] = {
    "strong": 1.0,
    "moderate": 0.6,
    "weak": 0.3,
}


def _clamp(val: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return round(max(0.0, min(1.0, val)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Per-edge-type scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_correlation_edge(
    *,
    pearson_r: float | None = None,
    granger_p: float | None = None,
    mutual_info: float | None = None,
    strength: str | None = None,
) -> float:
    """Score a CORRELATED_WITH edge.

    Weights:
      40% — effect size (|pearson_r| scaled to 0.8 = perfect)
      30% — Granger significance (1 − p/0.10, clamped)
      15% — non-linear dependency (mutual_info scaled to 0.3 = perfect)
      15% — categorical strength floor

    Args:
        pearson_r:   Pearson correlation at best lag.
        granger_p:   p-value from Granger causality test.
        mutual_info: Mutual information in bits.
        strength:    Categorical strength ('strong'/'moderate'/'weak').

    Returns:
        Confidence score in [0.0, 1.0].
    """
    effect = min(abs(pearson_r or 0) / 0.8, 1.0)

    if granger_p is not None:
        significance = max(0.0, 1.0 - granger_p / 0.10)
    else:
        significance = 0.4  # unknown → moderate default

    mi_score = min((mutual_info or 0) / 0.3, 1.0)
    str_score = _STRENGTH_SCORES.get(strength or "", 0.3)

    return _clamp(effect * 0.40 + significance * 0.30 + mi_score * 0.15 + str_score * 0.15)


def score_sensitivity_edge(
    *,
    beta: float | None = None,
    t_stat: float | None = None,
    p_value: float | None = None,
    r_squared: float | None = None,
    strength: str | None = None,
) -> float:
    """Score a SENSITIVE_TO edge.

    Weights:
      35% — t-stat significance (|t| scaled to 4.0 = perfect)
      25% — effect size (|beta| scaled to 1.0)
      20% — model fit (R²)
      20% — categorical strength floor

    For edges where t_stat is unavailable but p_value is, we derive
    significance from p_value instead.

    Args:
        beta:      OLS regression coefficient.
        t_stat:    t-statistic of the beta estimate.
        p_value:   p-value of the beta estimate.
        r_squared: Model R² (goodness of fit).
        strength:  Categorical strength.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    if t_stat is not None:
        t_score = min(abs(t_stat) / 4.0, 1.0)
    elif p_value is not None:
        t_score = max(0.0, 1.0 - p_value / 0.10)
    else:
        t_score = 0.4

    effect = min(abs(beta or 0) / 1.0, 1.0)
    fit = min(r_squared or 0.1, 1.0)
    str_score = _STRENGTH_SCORES.get(strength or "", 0.3)

    return _clamp(t_score * 0.35 + effect * 0.25 + fit * 0.20 + str_score * 0.20)


def score_causal_edge(
    *,
    pearson_r: float | None = None,
    granger_p: float | None = None,
    mutual_info: float | None = None,
    strength: str | None = None,
) -> float:
    """Score a CAUSES edge.

    Heavier weight on Granger significance (the key causal signal):
      45% — Granger significance
      25% — effect size (|pearson_r|)
      15% — non-linear dependency (mutual_info)
      15% — categorical strength floor

    Args:
        pearson_r:   Pearson correlation at best lag.
        granger_p:   p-value from Granger causality test.
        mutual_info: Mutual information in bits.
        strength:    Categorical strength.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    if granger_p is not None:
        granger_score = max(0.0, 1.0 - granger_p / 0.10)
    else:
        granger_score = 0.3  # lower default for causal claims without evidence

    effect = min(abs(pearson_r or 0) / 0.8, 1.0)
    mi_score = min((mutual_info or 0) / 0.3, 1.0)
    str_score = _STRENGTH_SCORES.get(strength or "", 0.3)

    return _clamp(granger_score * 0.45 + effect * 0.25 + mi_score * 0.15 + str_score * 0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_edge(props: dict[str, Any]) -> float:
    """Auto-detect edge type from properties and compute confidence.

    Dispatches to the appropriate scorer based on ``rel_type``.  Falls back
    to the correlation scorer for unknown edge types.

    Args:
        props: Edge property dict (must include ``rel_type`` for dispatch).

    Returns:
        Confidence score in [0.0, 1.0].
    """
    rel_type = props.get("rel_type", "")

    if rel_type == "SENSITIVE_TO":
        return score_sensitivity_edge(
            beta=props.get("beta"),
            t_stat=props.get("t_stat"),
            p_value=props.get("p_value"),
            r_squared=props.get("r_squared"),
            strength=props.get("strength"),
        )

    if rel_type == "CAUSES":
        return score_causal_edge(
            pearson_r=props.get("pearson_r"),
            granger_p=props.get("granger_p"),
            mutual_info=props.get("mutual_info"),
            strength=props.get("strength"),
        )

    # Default: CORRELATED_WITH or any unknown type
    return score_correlation_edge(
        pearson_r=props.get("pearson_r"),
        granger_p=props.get("granger_p"),
        mutual_info=props.get("mutual_info"),
        strength=props.get("strength"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal confidence decay
# ─────────────────────────────────────────────────────────────────────────────

# Half-life in days: confidence halves every 90 days without reconfirmation.
_HALF_LIFE_DAYS: float = 90.0

# Minimum floor: even fully decayed edges retain 10% of raw confidence.
_DECAY_FLOOR: float = 0.10


def decay_confidence(
    raw_confidence: float,
    days_since_confirmed: float,
    evidence_count: int = 1,
    half_life: float = _HALF_LIFE_DAYS,
) -> float:
    """Apply temporal decay to raw confidence.

    Edges that haven't been reconfirmed recently have their confidence
    reduced via exponential decay.  Edges with higher ``evidence_count``
    decay more slowly (each reconfirmation adds 20% to the half-life).

    Args:
        raw_confidence:       The stored confidence score [0-1].
        days_since_confirmed: Days since ``last_confirmed_at`` (or
                              ``first_seen_at`` if never reconfirmed).
        evidence_count:       How many independent runs confirmed this edge.
        half_life:            Base half-life in days (default 90).

    Returns:
        Effective confidence after decay, floored at 10% of raw.
    """
    if days_since_confirmed <= 0:
        return _clamp(raw_confidence)

    # More reconfirmations → slower decay (each adds 20% to half-life)
    adjusted_hl = half_life * (1.0 + 0.20 * max(evidence_count - 1, 0))

    import math
    decay_factor = math.pow(0.5, days_since_confirmed / adjusted_hl)
    floor = raw_confidence * _DECAY_FLOOR
    decayed = raw_confidence * decay_factor

    return _clamp(max(decayed, floor))


def classify_staleness(
    days_since_confirmed: float,
    evidence_count: int = 1,
) -> str:
    """Classify an edge's staleness level.

    Args:
        days_since_confirmed: Days since last confirmation.
        evidence_count:       Number of independent confirmations.

    Returns:
        One of 'fresh', 'aging', 'stale', 'expired'.
    """
    # Well-confirmed edges get longer freshness windows
    freshness_bonus = min(evidence_count - 1, 5) * 7  # +7 days per extra confirmation

    if days_since_confirmed <= 30 + freshness_bonus:
        return "fresh"
    if days_since_confirmed <= 90 + freshness_bonus:
        return "aging"
    if days_since_confirmed <= 180 + freshness_bonus:
        return "stale"
    return "expired"
