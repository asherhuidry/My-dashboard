"""Core evidence and claim schemas for the FinBrain evidence layer.

These are plain data structures.  No inference, no LLM, no auto-promotion.
Everything is explicit, serializable, and human-auditable.

EvidenceItem
    A single piece of structured evidence: an experiment result, a dataset
    observation, a backtest report, a manually recorded note, etc.

Claim
    A structured assertion about a source, feature, or relationship, backed
    by zero or more EvidenceItems.  A claim with no evidence is just a
    hypothesis.  A claim is only "supported" when a human (or a rule-based
    evaluator) explicitly marks it as such after reviewing the evidence.

Design principles
-----------------
- No narrative confidence is treated as truth.
- confidence is a human-set float, not an LLM output.
- Status transitions require deliberate action; nothing auto-promotes.
- counterpoints are first-class citizens, not optional afterthoughts.
- Everything round-trips through JSON losslessly.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enumerations ───────────────────────────────────────────────────────────────

class EvidenceSourceType(str, Enum):
    """Category of the original evidence source."""
    DATASET         = "dataset"          # e.g. a price series or FRED series
    BACKTEST        = "backtest"         # result from the backtest engine
    EXPERIMENT      = "experiment"       # registry experiment record
    DOCUMENT        = "document"         # research paper, article, or note
    SOURCE_REGISTRY = "source_registry"  # entry in the source registry
    MANUAL          = "manual"           # manually recorded observation


class ClaimType(str, Enum):
    """High-level category of a claim."""
    SOURCE_USEFULNESS  = "source_usefulness"   # a data source adds value
    FEATURE_USEFULNESS = "feature_usefulness"  # a feature improves model output
    RELATIONSHIP       = "relationship"         # two entities are meaningfully related
    PERFORMANCE        = "performance"          # a model or strategy meets a standard
    GENERAL            = "general"              # uncategorised


class ClaimStatus(str, Enum):
    """Lifecycle of a claim."""
    PROPOSED  = "proposed"   # stated but not yet evaluated against evidence
    SUPPORTED = "supported"  # evidence reviewed and found sufficient by a human
    WEAK      = "weak"       # some evidence, but insufficient or contradicted
    REJECTED  = "rejected"   # evidence reviewed and found to contradict the claim
    ARCHIVED  = "archived"   # no longer active; kept for historical reference


# ── EvidenceItem ───────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single, auditable piece of evidence.

    Attributes:
        evidence_id:     Unique identifier (UUID hex).
        source_type:     Which kind of source this evidence comes from.
        source_ref:      Pointer into the originating system — e.g. an
                         experiment_id, a file path, or a document title.
        summary:         One-paragraph human-readable description of what
                         this evidence shows.
        structured_data: Optional dict carrying raw values (metrics, counts,
                         p-values, etc.) for programmatic evaluation.
        citation:        Optional URL, DOI, or free-text reference string.
        created_at:      ISO-8601 UTC timestamp of when the evidence was
                         recorded.
    """
    evidence_id:     str
    source_type:     EvidenceSourceType
    source_ref:      str
    summary:         str
    structured_data: dict[str, Any] = field(default_factory=dict)
    citation:        str            = ""
    created_at:      str            = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(tz=timezone.utc).isoformat()

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "evidence_id":     self.evidence_id,
            "source_type":     self.source_type.value,
            "source_ref":      self.source_ref,
            "summary":         self.summary,
            "structured_data": self.structured_data,
            "citation":        self.citation,
            "created_at":      self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvidenceItem":
        """Reconstruct from a dictionary."""
        d = dict(d)
        d["source_type"] = EvidenceSourceType(d["source_type"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        source_type:     EvidenceSourceType | str,
        source_ref:      str,
        summary:         str,
        structured_data: dict[str, Any] | None = None,
        citation:        str = "",
    ) -> "EvidenceItem":
        """Create a new EvidenceItem with a fresh UUID and current timestamp.

        Args:
            source_type:     Category of the evidence source.
            source_ref:      Pointer to the originating record.
            summary:         Human-readable description of what this shows.
            structured_data: Optional raw metrics or values dict.
            citation:        Optional reference URL or string.

        Returns:
            A new EvidenceItem.
        """
        return cls(
            evidence_id     = uuid.uuid4().hex,
            source_type     = EvidenceSourceType(source_type),
            source_ref      = source_ref,
            summary         = summary,
            structured_data = structured_data or {},
            citation        = citation,
            created_at      = datetime.now(tz=timezone.utc).isoformat(),
        )


# ── Claim ──────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """A structured, evidence-backed assertion.

    A claim is a (subject, predicate, object) triple with attached evidence,
    a manually-set confidence level, uncertainty notes, and counterpoints.
    Confidence does not auto-update; humans set it after reviewing evidence.

    Attributes:
        claim_id:          Unique identifier (UUID hex).
        claim_type:        High-level category of the claim.
        subject:           The entity the claim is about.
                           e.g. ``"source:FRED_GDP"``, ``"feature:rsi_14"``.
        predicate:         The relationship or property being asserted.
                           e.g. ``"improves_accuracy"``, ``"is_correlated_with"``.
        object:            The target of the predicate.
                           e.g. ``"mlp_on_AAPL"``, ``"SPY"``, ``"macro_regime"``.
        evidence_ids:      IDs of attached EvidenceItems (stored separately).
        confidence:        Human-set confidence in [0.0, 1.0].  Not derived
                           from narrative or LLM output.
        uncertainty_notes: Free text describing known unknowns or caveats.
        counterpoints:     List of known counterarguments or contradicting
                           observations.
        status:            Lifecycle status; changed deliberately, never
                           automatically.
        tags:              Optional string labels for filtering.
        notes:             Free-text notes (context, next steps, etc.).
        created_at:        ISO-8601 UTC timestamp of creation.
        updated_at:        ISO-8601 UTC timestamp of last modification.
    """
    claim_id:          str
    claim_type:        ClaimType
    subject:           str
    predicate:         str
    object:            str
    evidence_ids:      list[str]
    confidence:        float
    uncertainty_notes: str       = ""
    counterpoints:     list[str] = field(default_factory=list)
    status:            ClaimStatus = ClaimStatus.PROPOSED
    tags:              list[str]   = field(default_factory=list)
    notes:             str         = ""
    created_at:        str         = ""
    updated_at:        str         = ""

    def __post_init__(self) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    # ── Convenience ────────────────────────────────────────────────────────

    @property
    def triple(self) -> tuple[str, str, str]:
        """Return the (subject, predicate, object) triple."""
        return (self.subject, self.predicate, self.object)

    @property
    def is_active(self) -> bool:
        """Return True for claims that are not rejected or archived."""
        return self.status not in (ClaimStatus.REJECTED, ClaimStatus.ARCHIVED)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "claim_id":          self.claim_id,
            "claim_type":        self.claim_type.value,
            "subject":           self.subject,
            "predicate":         self.predicate,
            "object":            self.object,
            "evidence_ids":      list(self.evidence_ids),
            "confidence":        self.confidence,
            "uncertainty_notes": self.uncertainty_notes,
            "counterpoints":     list(self.counterpoints),
            "status":            self.status.value,
            "tags":              list(self.tags),
            "notes":             self.notes,
            "created_at":        self.created_at,
            "updated_at":        self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        """Reconstruct from a dictionary."""
        d = dict(d)
        d["claim_type"] = ClaimType(d["claim_type"])
        d["status"]     = ClaimStatus(d["status"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        claim_type:        ClaimType | str,
        subject:           str,
        predicate:         str,
        obj:               str,
        confidence:        float          = 0.5,
        uncertainty_notes: str            = "",
        counterpoints:     list[str]      | None = None,
        tags:              list[str]      | None = None,
        notes:             str            = "",
    ) -> "Claim":
        """Create a new Claim in PROPOSED status with a fresh UUID.

        Args:
            claim_type:        Category of the claim.
            subject:           Entity the claim concerns.
            predicate:         Relationship being asserted.
            obj:               Target of the predicate.  (Named ``obj`` to
                               avoid shadowing the built-in ``object``.)
            confidence:        Human-set confidence in [0.0, 1.0].
            uncertainty_notes: Known caveats or unknowns.
            counterpoints:     Known counterarguments.
            tags:              String labels for filtering.
            notes:             Free-text context.

        Returns:
            A new Claim in PROPOSED status.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        return cls(
            claim_id          = uuid.uuid4().hex,
            claim_type        = ClaimType(claim_type),
            subject           = subject,
            predicate         = predicate,
            object            = obj,
            evidence_ids      = [],
            confidence        = confidence,
            uncertainty_notes = uncertainty_notes,
            counterpoints     = counterpoints or [],
            status            = ClaimStatus.PROPOSED,
            tags              = tags or [],
            notes             = notes,
            created_at        = now,
            updated_at        = now,
        )
