"""Local-first claim and evidence store.

Persists Claims and EvidenceItems in a single JSON file so the evidence
layer works without any database.  The store is write-through: every
mutating operation saves immediately.

Default storage path: ``data/evidence/claims.json``
Override with env var: ``FINBRAIN_CLAIMS_PATH``

Usage::

    from ml.evidence.store import ClaimStore
    from ml.evidence.schema import (
        Claim, ClaimType, ClaimStatus,
        EvidenceItem, EvidenceSourceType,
    )

    store = ClaimStore()

    ev = EvidenceItem.new(
        source_type = EvidenceSourceType.EXPERIMENT,
        source_ref  = "exp_abc123",
        summary     = "MLP achieved 0.58 accuracy on AAPL (fold 2 of 3)",
        structured_data = {"accuracy": 0.58, "fold": 2},
    )
    store.add_evidence(ev)

    claim = Claim.new(
        claim_type = ClaimType.FEATURE_USEFULNESS,
        subject    = "feature:rsi_14",
        predicate  = "improves_accuracy",
        obj        = "mlp_on_AAPL",
        confidence = 0.6,
    )
    store.add_claim(claim)
    store.link_evidence(claim.claim_id, ev.evidence_id)

    # query
    active = store.list_claims(status=ClaimStatus.PROPOSED)
    about_rsi = store.claims_by_subject("feature:rsi_14")
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.evidence.schema import (
    Claim,
    ClaimStatus,
    ClaimType,
    EvidenceItem,
)

log = logging.getLogger(__name__)

_DEFAULT_PATH = (
    Path(__file__).parent.parent.parent / "data" / "evidence" / "claims.json"
)
CLAIMS_PATH = Path(os.getenv("FINBRAIN_CLAIMS_PATH", str(_DEFAULT_PATH)))


class ClaimStore:
    """Local-first store for Claims and EvidenceItems.

    Args:
        path: Path to the JSON file.  Defaults to
              ``data/evidence/claims.json`` (or ``FINBRAIN_CLAIMS_PATH``).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = Path(path) if path else CLAIMS_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._claims:   dict[str, Claim]        = {}
        self._evidence: dict[str, EvidenceItem] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all records from disk (called once at init)."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for e in raw.get("evidence", []):
                item = EvidenceItem.from_dict(e)
                self._evidence[item.evidence_id] = item
            for c in raw.get("claims", []):
                claim = Claim.from_dict(c)
                self._claims[claim.claim_id] = claim
        except Exception as exc:
            log.warning("Could not load claim store from %s: %s", self._path, exc)

    def _save(self) -> None:
        """Persist all records to disk."""
        payload: dict[str, Any] = {
            "version":    "1",
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "evidence":   [e.to_dict() for e in self._evidence.values()],
            "claims":     [c.to_dict() for c in self._claims.values()],
        }
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Evidence operations ───────────────────────────────────────────────────

    def add_evidence(self, item: EvidenceItem) -> EvidenceItem:
        """Add a new EvidenceItem to the store.

        Args:
            item: The EvidenceItem to persist.

        Returns:
            The same item (for chaining).

        Raises:
            ValueError: If an item with the same evidence_id already exists.
        """
        if item.evidence_id in self._evidence:
            raise ValueError(
                f"EvidenceItem {item.evidence_id!r} already exists."
            )
        self._evidence[item.evidence_id] = item
        self._save()
        log.debug("Added evidence %s (%s)", item.evidence_id, item.source_type)
        return item

    def get_evidence(self, evidence_id: str) -> EvidenceItem:
        """Retrieve an EvidenceItem by ID.

        Raises:
            KeyError: If the ID is not found.
        """
        try:
            return self._evidence[evidence_id]
        except KeyError:
            raise KeyError(f"EvidenceItem {evidence_id!r} not found.") from None

    def list_evidence(
        self,
        source_type: str | None = None,
    ) -> list[EvidenceItem]:
        """Return all evidence items, optionally filtered by source_type.

        Args:
            source_type: Optional EvidenceSourceType value string to filter on.

        Returns:
            List of matching EvidenceItems in insertion order.
        """
        items = list(self._evidence.values())
        if source_type is not None:
            items = [e for e in items if e.source_type.value == source_type]
        return items

    # ── Claim operations ──────────────────────────────────────────────────────

    def add_claim(self, claim: Claim) -> Claim:
        """Add a new Claim to the store.

        Args:
            claim: The Claim to persist.

        Returns:
            The same claim (for chaining).

        Raises:
            ValueError: If a claim with the same claim_id already exists.
        """
        if claim.claim_id in self._claims:
            raise ValueError(f"Claim {claim.claim_id!r} already exists.")
        self._claims[claim.claim_id] = claim
        self._save()
        log.debug("Added claim %s [%s]", claim.claim_id, claim.claim_type)
        return claim

    def get_claim(self, claim_id: str) -> Claim:
        """Retrieve a Claim by ID.

        Raises:
            KeyError: If the ID is not found.
        """
        try:
            return self._claims[claim_id]
        except KeyError:
            raise KeyError(f"Claim {claim_id!r} not found.") from None

    def update_status(
        self,
        claim_id: str,
        status: ClaimStatus,
        notes: str = "",
    ) -> Claim:
        """Update the status of a claim.

        Args:
            claim_id: ID of the claim to update.
            status:   New ClaimStatus value.
            notes:    Optional notes to append to existing claim notes.

        Returns:
            The updated Claim.
        """
        claim = self.get_claim(claim_id)
        claim.status     = status
        claim.updated_at = datetime.now(tz=timezone.utc).isoformat()
        if notes:
            sep          = "\n" if claim.notes else ""
            claim.notes  = claim.notes + sep + notes
        self._save()
        return claim

    def update_confidence(
        self,
        claim_id:  str,
        confidence: float,
        notes:      str = "",
    ) -> Claim:
        """Update the confidence of a claim.

        Args:
            claim_id:   ID of the claim to update.
            confidence: New confidence value in [0.0, 1.0].
            notes:      Optional notes to append.

        Returns:
            The updated Claim.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {confidence}"
            )
        claim = self.get_claim(claim_id)
        claim.confidence = confidence
        claim.updated_at = datetime.now(tz=timezone.utc).isoformat()
        if notes:
            sep         = "\n" if claim.notes else ""
            claim.notes = claim.notes + sep + notes
        self._save()
        return claim

    def add_counterpoint(self, claim_id: str, counterpoint: str) -> Claim:
        """Append a counterpoint string to a claim.

        Args:
            claim_id:    ID of the claim to update.
            counterpoint: Human-readable counterargument or contradicting note.

        Returns:
            The updated Claim.
        """
        claim = self.get_claim(claim_id)
        claim.counterpoints.append(counterpoint)
        claim.updated_at = datetime.now(tz=timezone.utc).isoformat()
        self._save()
        return claim

    def link_evidence(self, claim_id: str, evidence_id: str) -> Claim:
        """Attach an EvidenceItem to a Claim.

        Args:
            claim_id:    ID of the claim.
            evidence_id: ID of the evidence to link.

        Returns:
            The updated Claim.

        Raises:
            KeyError: If either ID is not found.
        """
        claim = self.get_claim(claim_id)
        # Verify the evidence item exists
        self.get_evidence(evidence_id)
        if evidence_id not in claim.evidence_ids:
            claim.evidence_ids.append(evidence_id)
            claim.updated_at = datetime.now(tz=timezone.utc).isoformat()
            self._save()
        return claim

    def unlink_evidence(self, claim_id: str, evidence_id: str) -> Claim:
        """Remove an evidence link from a claim.

        Args:
            claim_id:    ID of the claim.
            evidence_id: ID of the evidence to unlink.

        Returns:
            The updated Claim.
        """
        claim = self.get_claim(claim_id)
        if evidence_id in claim.evidence_ids:
            claim.evidence_ids.remove(evidence_id)
            claim.updated_at = datetime.now(tz=timezone.utc).isoformat()
            self._save()
        return claim

    def get_evidence_for_claim(self, claim_id: str) -> list[EvidenceItem]:
        """Return all EvidenceItems linked to a claim.

        Missing evidence IDs are logged as warnings and skipped rather than
        raising, so partially-linked claims remain inspectable.

        Args:
            claim_id: ID of the claim.

        Returns:
            List of EvidenceItems in link order.
        """
        claim  = self.get_claim(claim_id)
        result = []
        for eid in claim.evidence_ids:
            if eid in self._evidence:
                result.append(self._evidence[eid])
            else:
                log.warning(
                    "Claim %s references missing evidence %s", claim_id, eid
                )
        return result

    # ── Query helpers ─────────────────────────────────────────────────────────

    def list_claims(
        self,
        status:     ClaimStatus | str | None = None,
        claim_type: ClaimType   | str | None = None,
        active_only: bool = False,
    ) -> list[Claim]:
        """Return claims matching optional filters.

        Args:
            status:      Filter by ClaimStatus value (or its string form).
            claim_type:  Filter by ClaimType value (or its string form).
            active_only: If True, exclude REJECTED and ARCHIVED claims.

        Returns:
            List of matching Claims in insertion order.
        """
        claims = list(self._claims.values())
        if active_only:
            claims = [c for c in claims if c.is_active]
        if status is not None:
            sv     = status if isinstance(status, str) else status.value
            claims = [c for c in claims if c.status.value == sv]
        if claim_type is not None:
            tv     = claim_type if isinstance(claim_type, str) else claim_type.value
            claims = [c for c in claims if c.claim_type.value == tv]
        return claims

    def claims_by_subject(self, subject: str) -> list[Claim]:
        """Return all claims whose subject matches exactly.

        Args:
            subject: Exact subject string (e.g. ``"feature:rsi_14"``).

        Returns:
            List of matching Claims.
        """
        return [c for c in self._claims.values() if c.subject == subject]

    def claims_by_object(self, obj: str) -> list[Claim]:
        """Return all claims whose object matches exactly.

        Args:
            obj: Exact object string (e.g. ``"mlp_on_AAPL"``).

        Returns:
            List of matching Claims.
        """
        return [c for c in self._claims.values() if c.object == obj]

    def claims_by_tag(self, tag: str) -> list[Claim]:
        """Return all claims that include the given tag.

        Args:
            tag: Tag string to search for.

        Returns:
            List of matching Claims.
        """
        return [c for c in self._claims.values() if tag in c.tags]

    # ── Aggregate stats ───────────────────────────────────────────────────────

    def summary_stats(self) -> dict[str, Any]:
        """Return a brief summary of the store contents.

        Returns:
            Dict with counts by status, by type, and totals.
        """
        from collections import Counter
        status_counts = Counter(c.status.value for c in self._claims.values())
        type_counts   = Counter(c.claim_type.value for c in self._claims.values())
        return {
            "n_claims":   len(self._claims),
            "n_evidence": len(self._evidence),
            "by_status":  dict(status_counts),
            "by_type":    dict(type_counts),
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._claims)

    def __repr__(self) -> str:
        return (
            f"ClaimStore(path={self._path}, "
            f"claims={len(self._claims)}, "
            f"evidence={len(self._evidence)})"
        )
