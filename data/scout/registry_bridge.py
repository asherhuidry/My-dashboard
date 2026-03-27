"""Source scout → source registry bridge.

Converts SourceCandidates into SourceRecords and inserts or updates them in
the SourceRegistry.  Status is always set to DISCOVERED — candidates are
never auto-approved.

Deduplication strategy:
1. Check by source_id.
2. If the source_id is not present, check all existing records for a
   matching URL (after normalization).
3. If a match is found, update the existing record rather than inserting
   a duplicate.

Usage::

    from data.registry.source_registry import SourceRegistry
    from data.scout.schema import normalize_source_candidate
    from data.scout.scorer import score_source_candidate
    from data.scout.registry_bridge import register_candidate_source

    registry  = SourceRegistry()
    candidate = normalize_source_candidate({...})
    score     = score_source_candidate(candidate)
    record    = register_candidate_source(candidate, registry, score=score)
    print(record.status)  # SourceStatus.DISCOVERED
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus
from data.scout.schema import SourceCandidate

log = logging.getLogger(__name__)


# ── RegistrationResult ────────────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    """Outcome of a single candidate registration attempt.

    Attributes:
        source_id:  The source_id of the record that was created or updated.
        action:     "created" | "updated" | "skipped".
        reason:     Why the action was taken (informational).
        record:     The SourceRecord that was created or updated, or None if
                    the candidate was skipped.
    """
    source_id: str
    action:    str
    reason:    str
    record:    SourceRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "source_id": self.source_id,
            "action":    self.action,
            "reason":    self.reason,
        }


# ── register_candidate_source ─────────────────────────────────────────────────

def register_candidate_source(
    candidate: SourceCandidate,
    registry:  SourceRegistry,
    score:     float | None = None,
    overwrite: bool = False,
) -> RegistrationResult:
    """Insert or update a candidate in the source registry.

    The source is always registered with ``status=DISCOVERED``.  If the
    source_id already exists in the registry the existing record is updated
    with any new information from the candidate (notes, score) unless
    ``overwrite=False`` and the existing status is not DISCOVERED.

    Duplicate detection:
    - First checks by source_id (exact match).
    - Then checks all existing records for a URL match (trailing-slash-
      normalized, scheme-case-insensitive).

    Args:
        candidate: Normalized SourceCandidate to register.
        registry:  Target SourceRegistry instance.
        score:     Optional pre-computed score to store as reliability_score.
        overwrite: If True, overwrite an existing record even if it has been
                   promoted beyond DISCOVERED.

    Returns:
        A RegistrationResult describing what happened.
    """
    rel_score = round(score, 4) if score is not None else 0.5

    # Check for existing record by source_id
    existing_by_id = _find_by_id(candidate.source_id, registry)

    # Check for existing record by URL
    existing_by_url = _find_by_url(candidate.url, registry)

    existing = existing_by_id or existing_by_url

    if existing and not overwrite:
        if existing.status != SourceStatus.DISCOVERED:
            # Already promoted — update notes and score but don't touch status
            _merge_notes(existing, candidate, registry)
            registry.update_score(existing.source_id, rel_score)
            return RegistrationResult(
                source_id = existing.source_id,
                action    = "updated",
                reason    = (
                    f"Existing record with status={existing.status.value}; "
                    "merged notes and updated score (status unchanged)."
                ),
                record = registry.get(existing.source_id),
            )
        # Already DISCOVERED — update freely
        _apply_candidate_to_record(existing, candidate, rel_score, registry)
        return RegistrationResult(
            source_id = existing.source_id,
            action    = "updated",
            reason    = "Existing DISCOVERED record updated with candidate data.",
            record    = registry.get(existing.source_id),
        )

    if existing and overwrite:
        _apply_candidate_to_record(existing, candidate, rel_score, registry)
        return RegistrationResult(
            source_id = existing.source_id,
            action    = "updated",
            reason    = "Overwrite=True; existing record updated.",
            record    = registry.get(existing.source_id),
        )

    # New record
    record = _candidate_to_record(candidate, rel_score)
    registry.add(record)
    return RegistrationResult(
        source_id = record.source_id,
        action    = "created",
        reason    = "New candidate registered as DISCOVERED.",
        record    = record,
    )


def register_catalog(
    candidates: list[SourceCandidate],
    registry:   SourceRegistry,
    scores:     dict[str, float] | None = None,
) -> list[RegistrationResult]:
    """Register a list of candidates into the registry.

    Args:
        candidates: List of normalized SourceCandidates.
        registry:   Target SourceRegistry.
        scores:     Optional dict mapping source_id → score.

    Returns:
        List of RegistrationResults, one per candidate.
    """
    scores = scores or {}
    results: list[RegistrationResult] = []
    for c in candidates:
        result = register_candidate_source(c, registry, score=scores.get(c.source_id))
        results.append(result)
        log.info(
            "Scout registration: %s → %s (%s)",
            c.source_id, result.action, result.reason,
        )
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _candidate_to_record(
    candidate: SourceCandidate,
    score:     float,
) -> SourceRecord:
    """Convert a SourceCandidate to a new SourceRecord (DISCOVERED status)."""
    return SourceRecord(
        source_id          = candidate.source_id,
        name               = candidate.name,
        category           = candidate.category if candidate.category != "unknown" else "alternative",
        url                = candidate.url,
        acquisition_method = candidate.acquisition_method,
        auth_required      = candidate.auth_required,
        free_tier          = candidate.free_tier,
        rate_limit_notes   = "",
        update_frequency   = candidate.update_cadence,
        asset_classes      = list(candidate.asset_types),
        data_types         = list(candidate.data_types),
        reliability_score  = score,
        status             = SourceStatus.DISCOVERED,
        notes              = candidate.notes,
        discovered_at      = candidate.discovered_at,
    )


def _apply_candidate_to_record(
    record:    SourceRecord,
    candidate: SourceCandidate,
    score:     float,
    registry:  SourceRegistry,
) -> None:
    """Merge candidate fields into an existing record and persist."""
    # Merge notes
    if candidate.notes and candidate.notes not in record.notes:
        record.notes = (record.notes + "\n" + candidate.notes).strip()

    # Update lists if candidate has more information
    for at in candidate.asset_types:
        if at not in record.asset_classes:
            record.asset_classes.append(at)
    for dt in candidate.data_types:
        if dt not in record.data_types:
            record.data_types.append(dt)

    # Update score
    record.reliability_score = score

    registry.add(record, overwrite=True)


def _merge_notes(
    record:    SourceRecord,
    candidate: SourceCandidate,
    registry:  SourceRegistry,
) -> None:
    """Append new notes from candidate without touching status."""
    if candidate.notes and candidate.notes not in record.notes:
        record.notes = (record.notes + "\n" + candidate.notes).strip()
        registry.add(record, overwrite=True)


def _find_by_id(source_id: str, registry: SourceRegistry) -> SourceRecord | None:
    """Return the record if source_id exists, else None."""
    try:
        return registry.get(source_id)
    except KeyError:
        return None


def _find_by_url(url: str, registry: SourceRegistry) -> SourceRecord | None:
    """Return the first record with a matching normalized URL, else None."""
    norm = _normalize_url(url)
    for rec in registry.all():
        if _normalize_url(rec.url) == norm:
            return rec
    return None


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison: lowercase scheme+host, strip trailing /."""
    return url.lower().rstrip("/")
