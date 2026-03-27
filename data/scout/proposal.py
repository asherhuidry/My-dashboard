"""Connector proposal/spec generation for source candidates.

Produces a lightweight structured proposal describing how a connector for a
candidate source should be built.  This is a *specification layer* — not a
full connector implementation.  The proposal is meant to be consumed by a
developer (or a future automation pass) to guide implementation.

Usage::

    from data.scout.schema import normalize_source_candidate
    from data.scout.scorer import score_source_candidate
    from data.scout.proposal import propose_connector_spec

    candidate = normalize_source_candidate({...})
    score     = score_source_candidate(candidate)
    spec      = propose_connector_spec(candidate, score)
    print(spec.priority)           # "high" / "medium" / "low"
    print(spec.implementation_notes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.scout.schema import SourceCandidate

# ── Priority thresholds ───────────────────────────────────────────────────────

_HIGH_THRESHOLD   = 0.65
_MEDIUM_THRESHOLD = 0.40


# ── ConnectorProposal ─────────────────────────────────────────────────────────

@dataclass
class ConnectorProposal:
    """A lightweight structured proposal for implementing a data connector.

    This is informational only.  No connector is created by producing this
    object.

    Attributes:
        source_id:                  Stable identifier from the candidate.
        acquisition_method:         How to fetch data (api, sdk, file_download, …).
        expected_payload_shape:     List of expected top-level field names or a
                                    description of the payload structure.
        refresh_frequency_suggestion: Recommended ingest cadence.
        auth_notes:                 Notes about authentication requirements.
        validation_strategy_hint:   Suggested data-quality checks.
        priority:                   "high" / "medium" / "low" based on score.
        implementation_notes:       Free-text guidance for the implementer.
        score:                      The source score used to derive priority.
    """
    source_id:                     str
    acquisition_method:            str
    expected_payload_shape:        list[str]
    refresh_frequency_suggestion:  str
    auth_notes:                    str
    validation_strategy_hint:      str
    priority:                      str
    implementation_notes:          str
    score:                         float
    extra:                         dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "source_id":                    self.source_id,
            "acquisition_method":           self.acquisition_method,
            "expected_payload_shape":       self.expected_payload_shape,
            "refresh_frequency_suggestion": self.refresh_frequency_suggestion,
            "auth_notes":                   self.auth_notes,
            "validation_strategy_hint":     self.validation_strategy_hint,
            "priority":                     self.priority,
            "implementation_notes":         self.implementation_notes,
            "score":                        self.score,
            "extra":                        self.extra,
        }


# ── propose_connector_spec ────────────────────────────────────────────────────

def propose_connector_spec(
    candidate: SourceCandidate,
    score:     float,
) -> ConnectorProposal:
    """Generate a connector proposal from a scored SourceCandidate.

    All logic is deterministic — no network calls, no LLM.

    Args:
        candidate: A normalized SourceCandidate.
        score:     The usefulness score (0–1) from ``score_source_candidate``.

    Returns:
        A ConnectorProposal describing how to build the connector.
    """
    priority              = _derive_priority(score)
    payload_shape         = _infer_payload_shape(candidate)
    refresh_suggestion    = _infer_refresh(candidate)
    auth_notes            = _build_auth_notes(candidate)
    validation_hint       = _build_validation_hint(candidate)
    implementation_notes  = _build_implementation_notes(candidate, score)

    return ConnectorProposal(
        source_id                    = candidate.source_id,
        acquisition_method           = candidate.acquisition_method,
        expected_payload_shape       = payload_shape,
        refresh_frequency_suggestion = refresh_suggestion,
        auth_notes                   = auth_notes,
        validation_strategy_hint     = validation_hint,
        priority                     = priority,
        implementation_notes         = implementation_notes,
        score                        = score,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _derive_priority(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _infer_payload_shape(candidate: SourceCandidate) -> list[str]:
    """Infer expected top-level fields from data_types."""
    _DT_FIELDS: dict[str, list[str]] = {
        "ohlcv":          ["date", "open", "high", "low", "close", "volume"],
        "yields":         ["date", "maturity", "yield_pct"],
        "inflation":      ["date", "series_id", "value"],
        "gdp":            ["date", "country", "value", "unit"],
        "employment":     ["date", "series_id", "value"],
        "wages":          ["date", "series_id", "value"],
        "cpi":            ["date", "series_id", "value"],
        "exchange_rates": ["date", "base", "target", "rate"],
        "macro_series":   ["date", "series_id", "value", "unit"],
        "fundamentals":   ["ticker", "period", "metric", "value"],
        "earnings":       ["ticker", "period", "eps", "revenue"],
        "balance_sheet":  ["ticker", "period", "assets", "liabilities", "equity"],
        "cash_flow":      ["ticker", "period", "operating", "investing", "financing"],
        "volatility":     ["date", "index", "value"],
        "news":           ["published_at", "title", "url", "source"],
        "sentiment":      ["date", "entity", "score", "source"],
        "market_cap":     ["date", "ticker", "market_cap", "circulating_supply"],
        "money_supply":   ["date", "series_id", "value", "unit"],
        "trade":          ["date", "country", "export_value", "import_value"],
    }
    fields: list[str] = []
    seen: set[str] = set()
    for dt in candidate.data_types:
        for f in _DT_FIELDS.get(dt, [f"field_{dt}"]):
            if f not in seen:
                seen.add(f)
                fields.append(f)
    return fields or ["date", "value"]


def _infer_refresh(candidate: SourceCandidate) -> str:
    """Map update_cadence to a concrete refresh schedule."""
    _CADENCE_TO_SCHEDULE: dict[str, str] = {
        "realtime": "Every 1–5 minutes via streaming or polling",
        "hourly":   "Hourly cron (e.g. 0 * * * *)",
        "daily":    "Daily cron after market close (e.g. 0 20 * * 1-5)",
        "weekly":   "Weekly cron (e.g. 0 6 * * 1)",
        "monthly":  "Monthly cron on the 2nd of each month",
        "quarterly":"Quarterly after earnings season (Feb/May/Aug/Nov)",
        "annual":   "Annual, triggered manually or on a fixed date",
    }
    return _CADENCE_TO_SCHEDULE.get(
        candidate.update_cadence,
        f"Cadence unknown — check source docs and set manually",
    )


def _build_auth_notes(candidate: SourceCandidate) -> str:
    if not candidate.auth_required:
        return "No authentication required. Use anonymous HTTP requests."
    if candidate.free_tier:
        return (
            "API key required. Free tier available — register at the source URL "
            "and store the key as an environment variable (e.g. "
            f"{candidate.source_id.upper()}_API_KEY)."
        )
    return (
        "Paid access required. Check pricing at the source URL before implementing. "
        f"Store credentials as {candidate.source_id.upper()}_API_KEY."
    )


def _build_validation_hint(candidate: SourceCandidate) -> str:
    hints: list[str] = []

    if "ohlcv" in candidate.data_types:
        hints.append("Check open <= high and low <= close; reject rows with zero volume.")
    if any(dt in candidate.data_types for dt in ("yields", "inflation", "gdp", "cpi")):
        hints.append("Validate that numeric values are within historically plausible ranges.")
    if "news" in candidate.data_types or "sentiment" in candidate.data_types:
        hints.append("Deduplicate by URL; validate published_at is a parseable timestamp.")
    if candidate.acquisition_method == "file_download":
        hints.append("Verify file checksum or size; check for truncated downloads.")
    if candidate.acquisition_method in ("api", "sdk"):
        hints.append("Check HTTP status; surface rate-limit headers and back off on 429.")

    hints.append("Run noise_filter.py on first ingest; quarantine sources with >5% null rate.")

    return " | ".join(hints) if hints else "Run standard null-rate and range checks."


def _build_implementation_notes(candidate: SourceCandidate, score: float) -> str:
    parts: list[str] = [
        f"Score: {score:.2f} ({_derive_priority(score)} priority).",
    ]
    if candidate.acquisition_method == "api":
        parts.append(
            f"Implement as a class in data/ingest/{candidate.source_id}_connector.py "
            f"following the pattern in yfinance_connector.py or fred_connector.py."
        )
    elif candidate.acquisition_method == "file_download":
        parts.append(
            f"Implement a download helper in data/ingest/{candidate.source_id}_connector.py "
            "that fetches the file, validates it, and writes to TimescaleDB."
        )
    elif candidate.acquisition_method == "sdk":
        parts.append(f"Wrap the SDK in data/ingest/{candidate.source_id}_connector.py.")
    elif candidate.acquisition_method == "feed":
        parts.append(f"Parse the feed in data/ingest/{candidate.source_id}_connector.py.")

    if candidate.official_source:
        parts.append("Official source — high reliability expected once connector is stable.")
    if not candidate.free_tier:
        parts.append("WARNING: Paid source. Confirm budget approval before implementing.")

    return " ".join(parts)
