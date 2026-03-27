"""Source candidate schema and normalization for the source scout.

A SourceCandidate is a lightweight intake form for a data source that has
not yet been validated or approved.  It accepts partial, informal, or
under-specified input and normalizes it into a consistent structure that
can be scored, proposed as a connector spec, and eventually inserted into
the source registry.

This module is deliberately thin: no network calls, no inference, no
LLM output.  Everything is deterministic and local.

Usage::

    from data.scout.schema import SourceCandidate, normalize_source_candidate

    raw = {
        "name": "ECB Statistical Data Warehouse",
        "url":  "https://sdw-wsrest.ecb.europa.eu/service",
        "category": "macro",
        "acquisition_method": "api",
        "auth_required": False,
        "official_source": True,
        "asset_types": ["macro"],
        "data_types":  ["yields", "inflation", "money_supply"],
        "update_cadence": "daily",
        "notes": "Euro area macro series from the European Central Bank.",
    }
    candidate = normalize_source_candidate(raw)
    print(candidate.source_id)  # "ecb_statistical_data_warehouse"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Valid category values (mirrors SourceRecord convention) ──────────────────

_KNOWN_CATEGORIES = {
    "price", "macro", "fundamental", "alternative",
    "news", "sentiment", "crypto", "forex", "commodity",
}

# Known acquisition methods
_KNOWN_METHODS = {"api", "sdk", "file_download", "scrape", "feed"}

# Fallback category inferred from asset_types
_ASSET_TO_CATEGORY: dict[str, str] = {
    "equity":    "price",
    "etf":       "price",
    "crypto":    "crypto",
    "forex":     "forex",
    "commodity": "commodity",
    "macro":     "macro",
    "bond":      "macro",
    "futures":   "price",
}


# ── SourceCandidate ───────────────────────────────────────────────────────────

@dataclass
class SourceCandidate:
    """Normalized intake record for a candidate data source.

    All fields are optional at intake; ``normalize_source_candidate`` fills
    in sensible defaults.  Only ``name`` and ``url`` are required inputs.

    Attributes:
        source_id:          Stable snake_case identifier (auto-generated if
                            not provided).
        name:               Human-readable name.
        url:                Base or documentation URL.
        category:           Data category (price, macro, crypto, …).
        acquisition_method: How to fetch data (api, sdk, file_download, …).
        auth_required:      Whether an API key / login is required.
        free_tier:          Whether a no-cost tier exists.
        official_source:    True for government, central bank, exchange, or
                            regulatory bodies.  Boosts trust score.
        update_cadence:     How frequently the source updates.
        asset_types:        Asset classes covered.
        data_types:         Data fields provided (ohlcv, yields, news, …).
        schema_clarity:     Subjective 1–5 rating of documentation quality.
        notes:              Free-text notes about the source.
        discovered_at:      ISO-8601 timestamp of when this candidate was
                            recorded.
    """
    source_id:          str
    name:               str
    url:                str
    category:           str            = "unknown"
    acquisition_method: str            = "api"
    auth_required:      bool           = False
    free_tier:          bool           = True
    official_source:    bool           = False
    update_cadence:     str            = "unknown"
    asset_types:        list[str]      = field(default_factory=list)
    data_types:         list[str]      = field(default_factory=list)
    schema_clarity:     int            = 3          # 1 (opaque) – 5 (excellent)
    notes:              str            = ""
    discovered_at:      str            = ""

    def __post_init__(self) -> None:
        if not self.discovered_at:
            self.discovered_at = datetime.now(tz=timezone.utc).isoformat()
        self.schema_clarity = max(1, min(5, self.schema_clarity))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "source_id":          self.source_id,
            "name":               self.name,
            "url":                self.url,
            "category":           self.category,
            "acquisition_method": self.acquisition_method,
            "auth_required":      self.auth_required,
            "free_tier":          self.free_tier,
            "official_source":    self.official_source,
            "update_cadence":     self.update_cadence,
            "asset_types":        list(self.asset_types),
            "data_types":         list(self.data_types),
            "schema_clarity":     self.schema_clarity,
            "notes":              self.notes,
            "discovered_at":      self.discovered_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceCandidate":
        """Reconstruct from a dict produced by ``to_dict()``."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ── normalize_source_candidate ───────────────────────────────────────────────

def normalize_source_candidate(raw: dict[str, Any]) -> SourceCandidate:
    """Normalize a raw candidate dict into a SourceCandidate.

    Applies the following rules in order:
    1. ``source_id`` is auto-generated from ``name`` via slugification if not
       supplied.
    2. ``category`` is inferred from ``asset_types`` if not supplied.
    3. ``acquisition_method`` is lower-cased and validated against known values;
       unknown values are kept as-is.
    4. ``schema_clarity`` is clamped to [1, 5].
    5. All list fields are deduplicated and lower-cased.

    Args:
        raw: Dict with at minimum ``name`` and ``url``.

    Returns:
        A normalized SourceCandidate.

    Raises:
        ValueError: If ``name`` or ``url`` is missing or empty.
    """
    raw = {k: v for k, v in raw.items() if v is not None}

    name = str(raw.get("name", "")).strip()
    url  = str(raw.get("url",  "")).strip()
    if not name:
        raise ValueError("SourceCandidate requires a non-empty 'name'.")
    if not url:
        raise ValueError("SourceCandidate requires a non-empty 'url'.")

    # source_id
    source_id = raw.get("source_id") or _slugify(name)

    # category
    raw_cat    = str(raw.get("category", "")).strip().lower()
    asset_types = _normalize_list(raw.get("asset_types") or raw.get("asset_classes") or [])
    category   = raw_cat if raw_cat in _KNOWN_CATEGORIES else _infer_category(asset_types)

    # acquisition_method
    method_raw = str(raw.get("acquisition_method", "api")).strip().lower()
    method     = method_raw if method_raw in _KNOWN_METHODS else method_raw

    # cadence aliases
    cadence_raw = str(raw.get("update_cadence") or raw.get("update_frequency", "unknown")).strip().lower()
    cadence     = _normalize_cadence(cadence_raw)

    return SourceCandidate(
        source_id          = source_id,
        name               = name,
        url                = url,
        category           = category,
        acquisition_method = method,
        auth_required      = bool(raw.get("auth_required", False)),
        free_tier          = bool(raw.get("free_tier", True)),
        official_source    = bool(raw.get("official_source", False)),
        update_cadence     = cadence,
        asset_types        = asset_types,
        data_types         = _normalize_list(raw.get("data_types", [])),
        schema_clarity     = int(raw.get("schema_clarity", 3)),
        notes              = str(raw.get("notes", "")).strip(),
        discovered_at      = str(raw.get("discovered_at", "")),
    )


# ── Built-in high-signal free candidates ─────────────────────────────────────

#: A curated catalog of well-known public/free financial data sources that are
#: not yet in the seed registry.  Intended as a starting point for new runs.
CANDIDATE_CATALOG: list[dict[str, Any]] = [
    {
        "name":               "ECB Statistical Data Warehouse",
        "url":                "https://sdw-wsrest.ecb.europa.eu/service",
        "category":           "macro",
        "acquisition_method": "api",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "daily",
        "asset_types":        ["macro"],
        "data_types":         ["yields", "inflation", "money_supply", "exchange_rates"],
        "schema_clarity":     4,
        "notes":              "ECB REST API (SDMX). Euro area macro series. No key needed.",
    },
    {
        "name":               "US Treasury Yield Curve (Treasury.gov)",
        "url":                "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "category":           "macro",
        "acquisition_method": "file_download",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "daily",
        "asset_types":        ["bond", "macro"],
        "data_types":         ["yields"],
        "schema_clarity":     4,
        "notes":              "Daily Treasury par yield curve rates. CSV download. Official US government source.",
    },
    {
        "name":               "OECD Data API",
        "url":                "https://data.oecd.org/api",
        "category":           "macro",
        "acquisition_method": "api",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "monthly",
        "asset_types":        ["macro"],
        "data_types":         ["gdp", "inflation", "employment", "trade"],
        "schema_clarity":     4,
        "notes":              "OECD SDMX-JSON API. Cross-country macro data. No key needed.",
    },
    {
        "name":               "SEC EDGAR XBRL Data",
        "url":                "https://data.sec.gov/api/xbrl",
        "category":           "fundamental",
        "acquisition_method": "api",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "quarterly",
        "asset_types":        ["equity"],
        "data_types":         ["fundamentals", "earnings", "balance_sheet", "cash_flow"],
        "schema_clarity":     3,
        "notes":              "SEC EDGAR XBRL structured filings. 10-K/10-Q machine-readable. No key needed. Rate limit: 10 req/s.",
    },
    {
        "name":               "Yahoo Finance RSS / Market Summary",
        "url":                "https://finance.yahoo.com",
        "category":           "news",
        "acquisition_method": "feed",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    False,
        "update_cadence":     "realtime",
        "asset_types":        ["equity", "crypto", "forex", "commodity"],
        "data_types":         ["news", "sentiment"],
        "schema_clarity":     2,
        "notes":              "RSS feeds for market news and sentiment. Supplement to yfinance SDK.",
    },
    {
        "name":               "CBOE Volatility Index (VIX) Data",
        "url":                "https://www.cboe.com/tradable_products/vix/vix_historical_data",
        "category":           "alternative",
        "acquisition_method": "file_download",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "daily",
        "asset_types":        ["equity"],
        "data_types":         ["volatility"],
        "schema_clarity":     5,
        "notes":              "Official CBOE VIX historical data. CSV. Widely used fear gauge.",
    },
    {
        "name":               "US BLS Employment Statistics",
        "url":                "https://api.bls.gov/publicAPI/v2/timeseries/data",
        "category":           "macro",
        "acquisition_method": "api",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "monthly",
        "asset_types":        ["macro"],
        "data_types":         ["employment", "wages", "cpi"],
        "schema_clarity":     4,
        "notes":              "Bureau of Labor Statistics public API. No key needed for basic access.",
    },
    {
        "name":               "Open Exchange Rates",
        "url":                "https://openexchangerates.org/api",
        "category":           "forex",
        "acquisition_method": "api",
        "auth_required":      True,
        "free_tier":          True,
        "official_source":    False,
        "update_cadence":     "hourly",
        "asset_types":        ["forex"],
        "data_types":         ["exchange_rates"],
        "schema_clarity":     5,
        "notes":              "Free tier: 1,000 req/month, hourly updates. Requires free API key.",
    },
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert a display name to a snake_case source_id."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64]  # cap at 64 chars


def _normalize_list(items: Any) -> list[str]:
    """Deduplicate and lower-case a list of strings."""
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        s = str(item).strip().lower()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _infer_category(asset_types: list[str]) -> str:
    """Infer a category from the asset_types list, or return 'unknown'."""
    for at in asset_types:
        cat = _ASSET_TO_CATEGORY.get(at)
        if cat:
            return cat
    return "unknown"


def _normalize_cadence(raw: str) -> str:
    """Map cadence aliases to a canonical form."""
    aliases: dict[str, str] = {
        "rt":         "realtime",
        "real-time":  "realtime",
        "real_time":  "realtime",
        "live":       "realtime",
        "1d":         "daily",
        "eod":        "daily",
        "end of day": "daily",
        "1w":         "weekly",
        "1m":         "monthly",
        "mo":         "monthly",
        "q":          "quarterly",
        "qtr":        "quarterly",
    }
    return aliases.get(raw, raw)
