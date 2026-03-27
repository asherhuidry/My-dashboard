"""Claim-to-graph conversion helpers.

Translates Claim objects into GraphNodes and GraphEdges.  Nodes are derived
from the subject and object fields of a claim using the standard evidence
naming conventions.  Edges carry the claim's confidence, evidence IDs, and
uncertainty metadata.

Supported claim types
---------------------
- RELATIONSHIP       — e.g. asset:AAPL is_correlated_with asset:QQQ
- SOURCE_USEFULNESS  — e.g. source:FRED_GDP is_useful_for domain:macro_regime
- FEATURE_USEFULNESS — e.g. feature:rsi_14 improves_accuracy_on model:mlp_dataset:abc
- PERFORMANCE        — e.g. model:mlp_on_AAPL meets_performance_bar metric:beat_bm

Functions
---------
nodes_from_claim(claim)          -> list[GraphNode]
edge_from_claim(claim, status)   -> GraphEdge
"""
from __future__ import annotations

from typing import Any

from ml.evidence.schema import Claim, ClaimStatus, ClaimType
from ml.graph.schema import GraphEdge, GraphEdgeStatus, GraphNode

# ── Status mapping ────────────────────────────────────────────────────────────

_CLAIM_TO_EDGE_STATUS: dict[ClaimStatus, GraphEdgeStatus] = {
    ClaimStatus.SUPPORTED: GraphEdgeStatus.ACTIVE,
    ClaimStatus.PROPOSED:  GraphEdgeStatus.PROPOSED,
    ClaimStatus.WEAK:      GraphEdgeStatus.WEAK,
    ClaimStatus.REJECTED:  GraphEdgeStatus.REJECTED,
    ClaimStatus.ARCHIVED:  GraphEdgeStatus.ARCHIVED,
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def nodes_from_claim(claim: Claim) -> list[GraphNode]:
    """Derive the subject and object GraphNodes from a claim.

    Both the ``subject`` and ``object`` fields are parsed using the evidence
    naming conventions (``source:``, ``feature:``, ``model:``, etc.).  The
    result is always exactly two nodes — one for the subject and one for the
    object.  Callers should upsert them into a GraphStore rather than blindly
    adding them, since the same entity may appear in many claims.

    Args:
        claim: A Claim whose subject and object are namespaced entity strings.

    Returns:
        ``[subject_node, object_node]``.
    """
    subj_props = _claim_subject_props(claim)
    obj_props  = _claim_object_props(claim)

    subject_node = GraphNode.from_entity_id(claim.subject, properties=subj_props)
    object_node  = GraphNode.from_entity_id(claim.object,  properties=obj_props)
    return [subject_node, object_node]


def edge_from_claim(
    claim:  Claim,
    status: GraphEdgeStatus | None = None,
) -> GraphEdge:
    """Create a GraphEdge from a claim.

    The edge carries the claim's confidence, evidence IDs, and uncertainty
    metadata.  Counterpoints are stored in ``properties["counterpoints"]`` so
    they travel with the edge.

    Args:
        claim:  The source Claim.
        status: Override the edge status.  If not given, it is derived from
                ``claim.status`` via the standard mapping.

    Returns:
        A new GraphEdge (UUID assigned).
    """
    if status is None:
        status = _CLAIM_TO_EDGE_STATUS.get(claim.status, GraphEdgeStatus.PROPOSED)

    props: dict[str, Any] = {}
    if claim.uncertainty_notes:
        props["uncertainty_notes"] = claim.uncertainty_notes
    if claim.counterpoints:
        props["counterpoints"] = list(claim.counterpoints)
    if claim.tags:
        props["tags"] = list(claim.tags)
    if claim.notes:
        props["notes"] = claim.notes
    props["claim_type"] = claim.claim_type.value

    return GraphEdge.new(
        source_node_id = claim.subject,
        target_node_id = claim.object,
        relation       = claim.predicate,
        status         = status,
        confidence     = claim.confidence,
        claim_id       = claim.claim_id,
        evidence_ids   = list(claim.evidence_ids),
        properties     = props,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _claim_subject_props(claim: Claim) -> dict[str, Any]:
    """Extract subject-side properties from a claim for node metadata."""
    props: dict[str, Any] = {"claim_type": claim.claim_type.value}
    if claim.claim_type == ClaimType.PERFORMANCE:
        # e.g. subject = "model:mlp_on_AAPL"
        props["role"] = "model"
    elif claim.claim_type == ClaimType.FEATURE_USEFULNESS:
        props["role"] = "feature"
    elif claim.claim_type == ClaimType.SOURCE_USEFULNESS:
        props["role"] = "source"
    elif claim.claim_type == ClaimType.RELATIONSHIP:
        props["role"] = "asset_or_entity"
    return props


def _claim_object_props(claim: Claim) -> dict[str, Any]:
    """Extract object-side properties from a claim for node metadata."""
    props: dict[str, Any] = {}
    if claim.claim_type == ClaimType.PERFORMANCE:
        props["role"] = "metric_target"
    elif claim.claim_type == ClaimType.FEATURE_USEFULNESS:
        props["role"] = "model_target"
    elif claim.claim_type == ClaimType.SOURCE_USEFULNESS:
        props["role"] = "domain"
    elif claim.claim_type == ClaimType.RELATIONSHIP:
        props["role"] = "asset_or_entity"
    return props
