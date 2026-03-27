"""Source registry — lifecycle management for financial data sources.

Every piece of data FinBrain ingests should come from a registered source.
This module tracks sources through a lifecycle:

    discovered → sampled → validated → approved
                                    ↘ rejected
                  ↘ quarantined (was approved, now failing)

Persistence is local-first: a single JSON file on disk.
If Supabase is connected the registry can be synced, but it works
fully offline so CI and local development never depend on a live DB.

Usage::

    reg = SourceRegistry()
    reg.add(SourceRecord(
        source_id="fred_api",
        name="FRED — Federal Reserve Economic Data",
        category="macro",
        url="https://api.stlouisfed.org/fred",
        ...
    ))
    reg.update_status("fred_api", SourceStatus.APPROVED)
    sources = reg.filter(category="macro", status=SourceStatus.APPROVED)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default storage path — can be overridden via FINBRAIN_REGISTRY_PATH env var
_DEFAULT_PATH = Path(__file__).parent / "sources.json"
REGISTRY_PATH = Path(os.getenv("FINBRAIN_REGISTRY_PATH", str(_DEFAULT_PATH)))


# ── Status lifecycle ──────────────────────────────────────────────────────────

class SourceStatus(str, Enum):
    """Lifecycle status for a data source."""
    DISCOVERED   = "discovered"    # Known but not yet sampled
    SAMPLED      = "sampled"       # A small sample has been fetched
    VALIDATED    = "validated"     # Sample passed quality checks
    APPROVED     = "approved"      # In active use by ingest pipeline
    REJECTED     = "rejected"      # Failed validation or no longer useful
    QUARANTINED  = "quarantined"   # Was approved, now producing bad data


# Valid transitions (current_status → allowed_next_statuses)
_VALID_TRANSITIONS: dict[SourceStatus, set[SourceStatus]] = {
    SourceStatus.DISCOVERED:  {SourceStatus.SAMPLED, SourceStatus.REJECTED},
    SourceStatus.SAMPLED:     {SourceStatus.VALIDATED, SourceStatus.REJECTED, SourceStatus.QUARANTINED},
    SourceStatus.VALIDATED:   {SourceStatus.APPROVED, SourceStatus.REJECTED},
    SourceStatus.APPROVED:    {SourceStatus.QUARANTINED, SourceStatus.REJECTED},
    SourceStatus.QUARANTINED: {SourceStatus.VALIDATED, SourceStatus.REJECTED},
    SourceStatus.REJECTED:    {SourceStatus.DISCOVERED},  # allow re-evaluation
}


# ── Source record ─────────────────────────────────────────────────────────────

@dataclass
class SourceRecord:
    """A single registered data source.

    Attributes:
        source_id:          Unique stable identifier (snake_case).
        name:               Human-readable name.
        category:           Data category (macro, equity, crypto, news, alternative, ...).
        url:                Base URL or documentation URL.
        acquisition_method: How data is fetched (api, file_download, scrape, sdk).
        auth_required:      Whether an API key or auth token is required.
        free_tier:          Whether a free tier exists.
        rate_limit_notes:   Human-readable description of rate limits.
        update_frequency:   How often the source updates (realtime, daily, weekly, ...).
        asset_classes:      List of asset classes covered.
        data_types:         List of data types provided (ohlcv, fundamentals, news, ...).
        reliability_score:  Float 0–1; updated from validation history.
        status:             Current lifecycle status.
        notes:              Free-text notes.
        discovered_at:      ISO-8601 timestamp when the source was first added.
        last_checked_at:    ISO-8601 timestamp of the most recent status check.
    """
    source_id:          str
    name:               str
    category:           str
    url:                str
    acquisition_method: str                = "api"
    auth_required:      bool               = False
    free_tier:          bool               = True
    rate_limit_notes:   str                = ""
    update_frequency:   str                = "daily"
    asset_classes:      list[str]          = field(default_factory=list)
    data_types:         list[str]          = field(default_factory=list)
    reliability_score:  float              = 0.5
    status:             SourceStatus       = SourceStatus.DISCOVERED
    notes:              str                = ""
    discovered_at:      str                = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    last_checked_at:    str | None         = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SourceRecord":
        """Deserialise from a dict (e.g. loaded from JSON)."""
        d = d.copy()
        d["status"] = SourceStatus(d.get("status", "discovered"))
        return cls(**d)


# ── Registry ──────────────────────────────────────────────────────────────────

class SourceRegistry:
    """In-process registry of data sources backed by a local JSON file.

    Thread-safety: not thread-safe; designed for single-process use.
    All mutations call ``_save()`` immediately so the file stays in sync.

    Args:
        path: Path to the backing JSON file.
               Defaults to ``REGISTRY_PATH`` (``data/registry/sources.json``).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path   = Path(path) if path else REGISTRY_PATH
        self._store: dict[str, SourceRecord] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load records from the JSON file (creates empty file if missing)."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                for item in raw.get("sources", []):
                    rec = SourceRecord.from_dict(item)
                    self._store[rec.source_id] = rec
                log.debug("Loaded %d sources from %s", len(self._store), self._path)
            except Exception as exc:
                log.warning("Failed to load source registry from %s: %s", self._path, exc)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self) -> None:
        """Persist all records to the JSON file."""
        payload = {
            "version": "1",
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "sources": [rec.to_dict() for rec in self._store.values()],
        }
        try:
            self._path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.error("Failed to save source registry: %s", exc)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, record: SourceRecord, overwrite: bool = False) -> SourceRecord:
        """Add a new source to the registry.

        Args:
            record:    The SourceRecord to add.
            overwrite: If False (default), raises ValueError if source_id exists.

        Returns:
            The stored record.

        Raises:
            ValueError: If source_id already exists and overwrite is False.
        """
        if record.source_id in self._store and not overwrite:
            raise ValueError(
                f"Source '{record.source_id}' already registered. "
                "Use overwrite=True or update_status() to modify it."
            )
        self._store[record.source_id] = record
        self._save()
        log.info("Registered source: %s (%s)", record.source_id, record.status.value)
        return record

    def get(self, source_id: str) -> SourceRecord:
        """Retrieve a source by ID.

        Raises:
            KeyError: If source_id is not found.
        """
        if source_id not in self._store:
            raise KeyError(f"Source not found: {source_id!r}")
        return self._store[source_id]

    def remove(self, source_id: str) -> None:
        """Remove a source from the registry.

        Raises:
            KeyError: If source_id is not found.
        """
        if source_id not in self._store:
            raise KeyError(f"Source not found: {source_id!r}")
        del self._store[source_id]
        self._save()

    def update_status(
        self,
        source_id: str,
        new_status: SourceStatus,
        notes: str = "",
        enforce_transitions: bool = True,
    ) -> SourceRecord:
        """Update the lifecycle status of a source.

        Args:
            source_id:            Source to update.
            new_status:           Target status.
            notes:                Optional note to append.
            enforce_transitions:  If True, only valid transitions are allowed.

        Returns:
            Updated SourceRecord.

        Raises:
            KeyError:   If source_id is not found.
            ValueError: If the transition is invalid and enforce_transitions=True.
        """
        rec = self.get(source_id)
        if enforce_transitions and new_status not in _VALID_TRANSITIONS.get(rec.status, set()):
            raise ValueError(
                f"Invalid transition for {source_id!r}: "
                f"{rec.status.value!r} → {new_status.value!r}. "
                f"Allowed: {[s.value for s in _VALID_TRANSITIONS.get(rec.status, [])]}"
            )
        rec.status         = new_status
        rec.last_checked_at = datetime.now(tz=timezone.utc).isoformat()
        if notes:
            rec.notes = (rec.notes + "\n" + notes).strip()
        self._save()
        log.info("Source %s → %s", source_id, new_status.value)
        return rec

    def update_score(self, source_id: str, score: float) -> SourceRecord:
        """Update the reliability score for a source.

        Args:
            source_id: Source to update.
            score:     New reliability score in [0, 1].

        Returns:
            Updated SourceRecord.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Reliability score must be in [0, 1], got {score}")
        rec = self.get(source_id)
        rec.reliability_score = round(score, 4)
        rec.last_checked_at   = datetime.now(tz=timezone.utc).isoformat()
        self._save()
        return rec

    def update_notes(self, source_id: str, notes: str) -> SourceRecord:
        """Append notes to a source record."""
        rec = self.get(source_id)
        rec.notes = (rec.notes + "\n" + notes).strip()
        self._save()
        return rec

    # ── Query / filter ────────────────────────────────────────────────────────

    def all(self) -> list[SourceRecord]:
        """Return all registered sources."""
        return list(self._store.values())

    def filter(
        self,
        category:   str | None         = None,
        status:     SourceStatus | None = None,
        asset_class:str | None         = None,
        data_type:  str | None         = None,
        auth_required: bool | None     = None,
        min_score:  float              = 0.0,
    ) -> list[SourceRecord]:
        """Filter sources by one or more criteria.

        Args:
            category:      Filter by data category (e.g. 'macro', 'equity').
            status:        Filter by lifecycle status.
            asset_class:   Filter by asset class membership.
            data_type:     Filter by data type membership.
            auth_required: Filter by whether authentication is required.
            min_score:     Minimum reliability score to include.

        Returns:
            Filtered list of SourceRecords.
        """
        results = list(self._store.values())
        if category is not None:
            results = [r for r in results if r.category == category]
        if status is not None:
            results = [r for r in results if r.status == status]
        if asset_class is not None:
            results = [r for r in results if asset_class in r.asset_classes]
        if data_type is not None:
            results = [r for r in results if data_type in r.data_types]
        if auth_required is not None:
            results = [r for r in results if r.auth_required == auth_required]
        if min_score > 0:
            results = [r for r in results if r.reliability_score >= min_score]
        return results

    def search(self, query: str) -> list[SourceRecord]:
        """Case-insensitive substring search across id, name, and notes.

        Args:
            query: Search string.

        Returns:
            Matching SourceRecords.
        """
        q = query.lower()
        return [
            r for r in self._store.values()
            if q in r.source_id.lower()
            or q in r.name.lower()
            or q in r.notes.lower()
            or q in r.category.lower()
        ]

    def summary(self) -> dict[str, Any]:
        """Return a status-count summary of the registry."""
        counts: dict[str, int] = {s.value: 0 for s in SourceStatus}
        for rec in self._store.values():
            counts[rec.status.value] += 1
        return {
            "total":      len(self._store),
            "by_status":  counts,
            "by_category": _count_by(self._store.values(), "category"),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._store


def _count_by(records: Any, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        k = getattr(r, attr, "unknown")
        counts[k] = counts.get(k, 0) + 1
    return counts
