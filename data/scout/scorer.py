"""Deterministic source usefulness/reliability scoring.

Scores a SourceCandidate on a 0–1 scale using a weighted sum of practical
factors.  No LLM, no network calls.  The score is deliberately coarse —
it is a first-pass triage signal, not a ground truth.

Score breakdown (weights sum to 1.0)
--------------------------------------
official_source     0.20   Government, central bank, exchange, or regulator
free_tier           0.15   No-cost access available
no_auth             0.08   No API key required (lower friction)
cadence_clarity     0.10   Update cadence is known (not "unknown")
schema_clarity      0.15   Documentation quality (1–5 scale, normalized)
financial_relevance 0.20   Category and data_types are financially useful
broad_utility       0.12   Covers multiple asset classes or data types

Usage::

    from data.scout.schema import normalize_source_candidate
    from data.scout.scorer import score_source_candidate

    candidate = normalize_source_candidate({...})
    score = score_source_candidate(candidate)
    print(score)  # e.g. 0.73
"""
from __future__ import annotations

from data.scout.schema import SourceCandidate

# ── Weights ───────────────────────────────────────────────────────────────────

_W_OFFICIAL    = 0.20
_W_FREE        = 0.15
_W_NO_AUTH     = 0.08
_W_CADENCE     = 0.10
_W_SCHEMA      = 0.15
_W_RELEVANCE   = 0.20
_W_BROAD       = 0.12

# Cadences that indicate a "known" update schedule
_KNOWN_CADENCES = {
    "realtime", "daily", "weekly", "monthly",
    "quarterly", "annual", "hourly",
}

# High-value data types for financial modelling
_HIGH_VALUE_TYPES = {
    "ohlcv", "yields", "inflation", "employment", "gdp", "earnings",
    "fundamentals", "balance_sheet", "cash_flow", "volatility",
    "macro_series", "exchange_rates", "market_cap",
}

# Financial categories (higher relevance)
_FINANCIAL_CATS = {
    "price", "macro", "fundamental", "crypto",
    "forex", "commodity", "alternative",
}


def score_source_candidate(candidate: SourceCandidate) -> float:
    """Return a usefulness/reliability score in [0.0, 1.0].

    The score is a weighted sum of practical quality signals.  It is
    deterministic and reproducible given the same input.

    Args:
        candidate: A normalized SourceCandidate.

    Returns:
        Float in [0.0, 1.0], rounded to 4 decimal places.
    """
    score = 0.0

    # official_source
    if candidate.official_source:
        score += _W_OFFICIAL

    # free_tier
    if candidate.free_tier:
        score += _W_FREE

    # no auth required (lower friction)
    if not candidate.auth_required:
        score += _W_NO_AUTH

    # update cadence is known
    if candidate.update_cadence in _KNOWN_CADENCES:
        score += _W_CADENCE

    # schema_clarity (1–5, normalized to 0–1)
    score += _W_SCHEMA * (candidate.schema_clarity - 1) / 4.0

    # financial relevance: category + high-value data types
    relevance = 0.0
    if candidate.category in _FINANCIAL_CATS:
        relevance += 0.5
    hv_match = len(set(candidate.data_types) & _HIGH_VALUE_TYPES)
    if hv_match >= 3:
        relevance += 0.5
    elif hv_match >= 1:
        relevance += 0.25 * min(hv_match, 2)
    score += _W_RELEVANCE * min(relevance, 1.0)

    # broad utility: multiple asset classes or data types
    breadth = len(candidate.asset_types) + len(candidate.data_types)
    broad_factor = min(breadth / 8.0, 1.0)
    score += _W_BROAD * broad_factor

    return round(min(score, 1.0), 4)


def score_breakdown(candidate: SourceCandidate) -> dict[str, float]:
    """Return per-factor score contributions for inspection.

    Args:
        candidate: A normalized SourceCandidate.

    Returns:
        Dict mapping factor name to contribution value.
    """
    official  = _W_OFFICIAL if candidate.official_source else 0.0
    free      = _W_FREE     if candidate.free_tier        else 0.0
    no_auth   = _W_NO_AUTH  if not candidate.auth_required else 0.0
    cadence   = _W_CADENCE  if candidate.update_cadence in _KNOWN_CADENCES else 0.0

    schema_c  = _W_SCHEMA * (candidate.schema_clarity - 1) / 4.0

    relevance = 0.0
    if candidate.category in _FINANCIAL_CATS:
        relevance += 0.5
    hv_match = len(set(candidate.data_types) & _HIGH_VALUE_TYPES)
    if hv_match >= 3:
        relevance += 0.5
    elif hv_match >= 1:
        relevance += 0.25 * min(hv_match, 2)
    fin_rel = _W_RELEVANCE * min(relevance, 1.0)

    breadth = len(candidate.asset_types) + len(candidate.data_types)
    broad   = _W_BROAD * min(breadth / 8.0, 1.0)

    return {
        "official_source":    round(official,  4),
        "free_tier":          round(free,      4),
        "no_auth":            round(no_auth,   4),
        "cadence_clarity":    round(cadence,   4),
        "schema_clarity":     round(schema_c,  4),
        "financial_relevance":round(fin_rel,   4),
        "broad_utility":      round(broad,     4),
        "total":              round(min(
            official + free + no_auth + cadence + schema_c + fin_rel + broad, 1.0
        ), 4),
    }
