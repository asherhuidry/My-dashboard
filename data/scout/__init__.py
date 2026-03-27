"""Bounded source scout — candidate intake, scoring, connector proposals, and probing.

Converts raw candidate source definitions into normalized records, scores
them for usefulness, generates connector proposals, optionally registers them
in the source registry, and can perform lightweight HTTP reachability probes.

Nothing in this module auto-approves sources, writes to production
connectors, or scrapes arbitrary content.

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
        # probe
        ProbeResult,
        probe_source_url,
        probe_url,
        ProbeRegistryResult,
        apply_probe_to_registry,
        probe_and_register,
        evidence_from_probe,
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
from data.scout.probe import ProbeResult, probe_source_url, probe_url
from data.scout.probe_registry import (
    ProbeRegistryResult,
    apply_probe_to_registry,
    probe_and_register,
)
from data.scout.probe_evidence import evidence_from_probe

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
    # probe
    "ProbeResult",
    "probe_source_url",
    "probe_url",
    # probe registry
    "ProbeRegistryResult",
    "apply_probe_to_registry",
    "probe_and_register",
    # probe evidence
    "evidence_from_probe",
]
