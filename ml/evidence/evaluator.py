"""Deterministic claim evaluation helpers.

Evaluates claims based on their attached evidence using simple, transparent
rule-based logic.  No LLM, no narrative inference, no auto-promotion.

Support levels
--------------
The evaluation produces one of four support levels:

    none     — zero evidence items attached
    weak     — 1 evidence item, or 2+ from only one source type
    moderate — 2+ items from 2+ distinct source types
    strong   — 4+ items from 3+ distinct source types

Conflict detection
------------------
Conflicts are currently identified by flag fields in ``structured_data``.
An evidence item signals conflict if ``structured_data["conflicts"]`` is
truthy.  This is a convention callers should follow when recording evidence
that contradicts the claim it is attached to.

Recommendations
---------------
The evaluator recommends one of:

    propose   — keep as PROPOSED; not enough evidence yet
    support   — sufficient evidence to mark SUPPORTED (human must decide)
    flag_weak — some evidence but insufficient or conflicted
    reject    — majority of evidence conflicts with the claim

None of these recommendations are applied automatically.  They inform a
human decision.

Usage::

    from ml.evidence.evaluator import evaluate_claim
    from ml.evidence.store import ClaimStore

    store = ClaimStore()
    claim = store.get_claim(some_id)
    items = store.get_evidence_for_claim(some_id)
    ev = evaluate_claim(claim, items)
    print(ev.support_level, ev.recommendation)
    print(ev.notes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml.evidence.schema import Claim, EvidenceItem


# ── Support thresholds ────────────────────────────────────────────────────────

_STRONG_MIN_ITEMS      = 4   # minimum evidence items for "strong"
_STRONG_MIN_SRC_TYPES  = 3   # minimum distinct source types for "strong"
_MODERATE_MIN_ITEMS    = 2   # minimum items for "moderate"
_MODERATE_MIN_SRC_TYPES = 2  # minimum distinct source types for "moderate"
_MAJORITY_CONFLICT     = 0.5 # fraction of evidence that triggers "reject" recommendation


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ClaimEvaluation:
    """Structured result of evaluating a claim against its evidence.

    All fields are computed deterministically from the attached evidence.
    Nothing here triggers an automatic status change on the claim.

    Attributes:
        claim_id:            ID of the evaluated claim.
        evidence_count:      Total attached evidence items.
        source_type_diversity: Number of distinct EvidenceSourceType values
                             among attached items.
        conflicting_count:   Evidence items that signal conflict
                             (``structured_data["conflicts"] == True``).
        support_level:       One of ``'none'``, ``'weak'``, ``'moderate'``,
                             ``'strong'``.
        support_score:       Numeric approximation of support strength in
                             [0.0, 1.0].  For display only — do not threshold
                             this score for hard decisions.
        recommendation:      One of ``'propose'``, ``'support'``,
                             ``'flag_weak'``, ``'reject'``.
        notes:               Human-readable explanation of the evaluation.
    """
    claim_id:              str
    evidence_count:        int
    source_type_diversity: int
    conflicting_count:     int
    support_level:         str         # "none" / "weak" / "moderate" / "strong"
    support_score:         float       # 0.0–1.0 indicative only
    recommendation:        str         # "propose" / "support" / "flag_weak" / "reject"
    notes:                 list[str]   = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "claim_id":              self.claim_id,
            "evidence_count":        self.evidence_count,
            "source_type_diversity": self.source_type_diversity,
            "conflicting_count":     self.conflicting_count,
            "support_level":         self.support_level,
            "support_score":         round(self.support_score, 4),
            "recommendation":        self.recommendation,
            "notes":                 list(self.notes),
        }


# ── Evaluation logic ──────────────────────────────────────────────────────────

def evaluate_claim(
    claim:          "Claim",
    evidence_items: "list[EvidenceItem]",
) -> ClaimEvaluation:
    """Evaluate a claim against a list of evidence items.

    This function is pure and deterministic: given the same inputs it always
    produces the same output.  It never reads from disk or mutates anything.

    Args:
        claim:          The Claim being evaluated.
        evidence_items: Evidence items linked to this claim.  Obtain via
                        ``ClaimStore.get_evidence_for_claim(claim_id)``.

    Returns:
        A ``ClaimEvaluation`` describing the evidence support level,
        diversity, conflict count, a numeric score, and a recommendation.
    """
    n    = len(evidence_items)
    notes: list[str] = []

    # ── Source type diversity ──────────────────────────────────────────────
    src_types = {e.source_type for e in evidence_items}
    diversity = len(src_types)

    # ── Conflict detection ─────────────────────────────────────────────────
    conflicting = [
        e for e in evidence_items
        if e.structured_data.get("conflicts") is True
    ]
    n_conflicts = len(conflicting)

    # ── Support level ──────────────────────────────────────────────────────
    if n == 0:
        support_level = "none"
    elif (
        n >= _STRONG_MIN_ITEMS
        and diversity >= _STRONG_MIN_SRC_TYPES
        and n_conflicts == 0
    ):
        support_level = "strong"
    elif n >= _MODERATE_MIN_ITEMS and diversity >= _MODERATE_MIN_SRC_TYPES:
        support_level = "moderate"
    else:
        support_level = "weak"

    # If majority of evidence conflicts, override to "none" effective
    if n > 0 and (n_conflicts / n) >= _MAJORITY_CONFLICT:
        support_level = "none"
        notes.append(
            f"{n_conflicts}/{n} evidence items flag a conflict — "
            "support level overridden to 'none'."
        )

    # ── Numeric score ──────────────────────────────────────────────────────
    _level_scores = {"none": 0.0, "weak": 0.3, "moderate": 0.6, "strong": 0.9}
    base_score  = _level_scores[support_level]

    # Small bonus for extra diversity beyond minimum; cap at 0.95
    diversity_bonus = min(0.05, (diversity - 1) * 0.01) if diversity > 1 else 0.0
    # Conflict penalty
    conflict_penalty = min(0.2, n_conflicts * 0.05)
    support_score = max(0.0, min(0.95, base_score + diversity_bonus - conflict_penalty))

    # ── Recommendation ─────────────────────────────────────────────────────
    if n == 0:
        recommendation = "propose"
        notes.append("No evidence attached — keep as PROPOSED until evidence is added.")
    elif n > 0 and (n_conflicts / n) >= _MAJORITY_CONFLICT:
        recommendation = "reject"
        notes.append(
            "Majority of attached evidence contradicts the claim — consider REJECTED."
        )
    elif support_level in ("moderate", "strong"):
        recommendation = "support"
        notes.append(
            f"Support level '{support_level}' with {diversity} distinct source "
            f"type(s) — sufficient to consider SUPPORTED after human review."
        )
    else:
        recommendation = "flag_weak"
        notes.append(
            f"{n} evidence item(s) from {diversity} source type(s) — "
            "add more diverse evidence before considering SUPPORTED."
        )

    # ── Informational notes ────────────────────────────────────────────────
    if n_conflicts > 0 and recommendation != "reject":
        notes.append(
            f"{n_conflicts} conflicting evidence item(s) noted — review carefully."
        )
    if diversity == 1 and n >= _MODERATE_MIN_ITEMS:
        notes.append(
            "All evidence is from the same source type — "
            "adding a second source type would strengthen the claim."
        )
    if claim.counterpoints:
        notes.append(
            f"{len(claim.counterpoints)} recorded counterpoint(s) on the claim "
            "— factor these into promotion decisions."
        )

    return ClaimEvaluation(
        claim_id              = claim.claim_id,
        evidence_count        = n,
        source_type_diversity = diversity,
        conflicting_count     = n_conflicts,
        support_level         = support_level,
        support_score         = round(support_score, 4),
        recommendation        = recommendation,
        notes                 = notes,
    )


def batch_evaluate(
    store: "Any",
    claim_ids: "list[str] | None" = None,
) -> "list[ClaimEvaluation]":
    """Evaluate multiple claims from a ClaimStore.

    Args:
        store:     A ClaimStore instance.
        claim_ids: Optional list of claim IDs to evaluate.  If None, all
                   claims in the store are evaluated.

    Returns:
        List of ClaimEvaluation results in the same order as the input IDs
        (or insertion order if claim_ids is None).
    """
    from ml.evidence.store import ClaimStore as _CS
    if not isinstance(store, _CS):
        raise TypeError(f"store must be a ClaimStore, got {type(store)}")

    if claim_ids is None:
        claims = store.list_claims()
    else:
        claims = [store.get_claim(cid) for cid in claim_ids]

    return [
        evaluate_claim(c, store.get_evidence_for_claim(c.claim_id))
        for c in claims
    ]
