"""Graph enrichment: load eligible claims and convert them to graph structures.

``enrich_graph_from_claims`` is the main entry point.  It reads claims from a
ClaimStore, selects eligible ones, converts them to nodes and edges, and
persists everything into a GraphStore.

Eligibility rules
-----------------
- By default only SUPPORTED claims are enriched (safe, reviewed evidence).
- Pass ``include_proposed=True`` to also enrich PROPOSED claims.
- WEAK, REJECTED, and ARCHIVED claims are always skipped (never silently treated
  as valid graph truths).

Skip reasons are recorded so callers can audit what was left out.

Usage::

    from ml.evidence.store import ClaimStore
    from ml.graph.store import GraphStore
    from ml.graph.enrichment import enrich_graph_from_claims

    claim_store = ClaimStore()
    graph_store = GraphStore()

    report = enrich_graph_from_claims(claim_store, graph_store)
    print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.evidence.schema import ClaimStatus, ClaimType
from ml.graph.converter import edge_from_claim, nodes_from_claim
from ml.graph.schema import GraphEdgeStatus

import logging

log = logging.getLogger(__name__)

# Claim types that map cleanly to graph edges
_ENRICHABLE_TYPES = {
    ClaimType.RELATIONSHIP,
    ClaimType.SOURCE_USEFULNESS,
    ClaimType.FEATURE_USEFULNESS,
    ClaimType.PERFORMANCE,
}

# Statuses that are always skipped
_SKIP_STATUSES = {ClaimStatus.WEAK, ClaimStatus.REJECTED, ClaimStatus.ARCHIVED}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrichmentReport:
    """Summary of a single enrichment pass.

    Attributes:
        nodes_created: Number of new nodes added.
        nodes_updated: Number of existing nodes updated.
        edges_created: Number of new edges added.
        skipped:       List of (claim_id, reason) for skipped claims.
    """
    nodes_created: int = 0
    nodes_updated: int = 0
    edges_created: int = 0
    skipped:       list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        """Total number of claims that were skipped."""
        return len(self.skipped)

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        lines = [
            "EnrichmentReport",
            f"  nodes created : {self.nodes_created}",
            f"  nodes updated : {self.nodes_updated}",
            f"  edges created : {self.edges_created}",
            f"  claims skipped: {self.n_skipped}",
        ]
        if self.skipped:
            lines.append("  skip details:")
            for claim_id, reason in self.skipped[:10]:
                lines.append(f"    {claim_id[:12]}... — {reason}")
            if self.n_skipped > 10:
                lines.append(f"    ... and {self.n_skipped - 10} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "n_skipped":     self.n_skipped,
            "skipped":       [
                {"claim_id": cid, "reason": r} for cid, r in self.skipped
            ],
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def enrich_graph_from_claims(
    claim_store:      "ClaimStore",  # type: ignore[name-defined]
    graph_store:      "GraphStore",  # type: ignore[name-defined]
    include_proposed: bool = False,
    claim_ids:        list[str] | None = None,
) -> EnrichmentReport:
    """Convert eligible claims into graph nodes and edges.

    Loads claims from ``claim_store``, filters by eligibility, converts each
    to a subject node, object node, and one directed edge, then persists them
    to ``graph_store``.

    Args:
        claim_store:      Source of claim records.
        graph_store:      Destination for graph nodes and edges.
        include_proposed: If True, PROPOSED claims are enriched in addition to
                          SUPPORTED ones.  WEAK/REJECTED/ARCHIVED are always
                          skipped regardless.
        claim_ids:        If given, only these claim IDs are considered.  If
                          None, all active claims are evaluated.

    Returns:
        An EnrichmentReport describing what was created and what was skipped.
    """
    report = EnrichmentReport()

    if claim_ids is not None:
        claims = [
            c for cid in claim_ids
            if (c := claim_store.get_claim(cid)) is not None
        ]
    else:
        claims = claim_store.list_claims(active_only=True)

    for claim in claims:
        skip_reason = _skip_reason(claim, include_proposed)
        if skip_reason:
            report.skipped.append((claim.claim_id, skip_reason))
            log.debug("Skipping claim %s: %s", claim.claim_id[:12], skip_reason)
            continue

        # --- nodes ---
        try:
            subject_node, object_node = nodes_from_claim(claim)
            is_new_s = graph_store.upsert_node(subject_node)
            is_new_o = graph_store.upsert_node(object_node)
            if is_new_s:
                report.nodes_created += 1
            else:
                report.nodes_updated += 1
            if is_new_o:
                report.nodes_created += 1
            else:
                report.nodes_updated += 1
        except Exception as exc:
            reason = f"node conversion error: {exc}"
            report.skipped.append((claim.claim_id, reason))
            log.warning("Node conversion failed for claim %s: %s", claim.claim_id[:12], exc)
            continue

        # --- edge ---
        try:
            edge = edge_from_claim(claim)
            graph_store.add_edge(edge)
            report.edges_created += 1
        except Exception as exc:
            reason = f"edge creation error: {exc}"
            report.skipped.append((claim.claim_id, reason))
            log.warning("Edge creation failed for claim %s: %s", claim.claim_id[:12], exc)

    return report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _skip_reason(
    claim:            "Claim",  # type: ignore[name-defined]
    include_proposed: bool,
) -> str:
    """Return a skip reason string, or '' if the claim is eligible."""
    if claim.status in _SKIP_STATUSES:
        return f"status={claim.status.value} — not enrichable"
    if claim.status == ClaimStatus.PROPOSED and not include_proposed:
        return "status=proposed — pass include_proposed=True to include"
    if claim.claim_type not in _ENRICHABLE_TYPES:
        return f"claim_type={claim.claim_type.value} — not a supported enrichment type"
    return ""
