"""Bounded source scout — candidate intake, scoring, and connector proposals.

Converts raw candidate source definitions into normalized records, scores
them for usefulness, generates connector proposals, and optionally registers
them in the source registry or creates evidence/claim stubs.

Nothing in this module auto-approves sources, writes to production
connectors, or calls external services.

Public API::

    from data.scout import (
        SourceCandidate,
        normalize_source_candidate,
        CANDIDATE_CATALOG,
        score_source_candidate,
        score_breakdown,
        ConnectorProposal,
        propose_connector_spec,
        RegistrationResult,
        register_candidate_source,
        register_catalog,
        evidence_from_candidate,
        source_claim_from_candidate,
    )
"""
from data.scout.schema import (
    SourceCandidate,
    normalize_source_candidate,
    CANDIDATE_CATALOG,
)
from data.scout.scorer import score_source_candidate, score_breakdown
from data.scout.proposal import ConnectorProposal, propose_connector_spec
from data.scout.registry_bridge import (
    RegistrationResult,
    register_candidate_source,
    register_catalog,
)
from data.scout.evidence_hooks import (
    evidence_from_candidate,
    source_claim_from_candidate,
)

__all__ = [
    # schema
    "SourceCandidate",
    "normalize_source_candidate",
    "CANDIDATE_CATALOG",
    # scorer
    "score_source_candidate",
    "score_breakdown",
    # proposal
    "ConnectorProposal",
    "propose_connector_spec",
    # registry bridge
    "RegistrationResult",
    "register_candidate_source",
    "register_catalog",
    # evidence hooks
    "evidence_from_candidate",
    "source_claim_from_candidate",
]
