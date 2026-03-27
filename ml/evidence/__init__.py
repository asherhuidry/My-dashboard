"""Evidence and claim layer for the FinBrain research loop.

Provides structured, serializable, deterministically-evaluable representations
of claims about data sources, features, and relationships.  This layer does
NOT auto-promote anything and does NOT use narrative confidence as truth.

Public API::

    from ml.evidence import (
        EvidenceItem, EvidenceSourceType,
        Claim, ClaimType, ClaimStatus,
        ClaimStore,
        evaluate_claim, ClaimEvaluation,
        source_usefulness_claim, feature_usefulness_claim, relationship_claim,
    )
"""
from ml.evidence.schema import (
    ClaimStatus,
    ClaimType,
    Claim,
    EvidenceItem,
    EvidenceSourceType,
)
from ml.evidence.store import ClaimStore
from ml.evidence.evaluator import ClaimEvaluation, evaluate_claim
from ml.evidence.templates import (
    feature_usefulness_claim,
    relationship_claim,
    source_usefulness_claim,
)
from ml.evidence.bridge import (
    ClaimBundle,
    SaveResult,
    bridge_comparison,
    bridge_wf_comparison,
    claims_from_comparison,
    claims_from_wf_comparison,
    evidence_from_fold_result,
    evidence_from_pipeline_result,
    evidence_from_wf_result,
    save_evidence_bundle,
)

__all__ = [
    # schema
    "EvidenceSourceType",
    "ClaimStatus",
    "ClaimType",
    "EvidenceItem",
    "Claim",
    # store
    "ClaimStore",
    # evaluator
    "ClaimEvaluation",
    "evaluate_claim",
    # templates
    "source_usefulness_claim",
    "feature_usefulness_claim",
    "relationship_claim",
    # bridge
    "ClaimBundle",
    "SaveResult",
    "evidence_from_fold_result",
    "evidence_from_wf_result",
    "evidence_from_pipeline_result",
    "claims_from_comparison",
    "claims_from_wf_comparison",
    "save_evidence_bundle",
    "bridge_wf_comparison",
    "bridge_comparison",
]
