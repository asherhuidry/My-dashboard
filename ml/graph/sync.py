"""Differential graph-lifecycle sync.

Keeps graph edges aligned with claim statuses as claims are reviewed, updated,
or archived over time.  Unlike full enrichment (which creates everything from
scratch), sync only touches edges that need to change.

``sync_graph_from_store`` is the main entry point.

Sync behaviour
--------------
For each eligible claim:

1. Look up existing edges in the graph store by ``claim_id``.
2. If edges exist:
   - Compare the current claim status to the current edge status.
   - If they differ, call ``update_edge_status`` on every matching edge.
   - If they match, record as ``unchanged``.
3. If no edges exist and ``create_missing=True``:
   - Run node upsert + edge creation exactly as enrichment does.
   - Record as ``edges_created``.
4. If no edges exist and ``create_missing=False``:
   - Record as ``missing_claim_edges`` (informational only, no action taken).

Eligibility for sync
--------------------
All claim statuses (including WEAK, REJECTED, ARCHIVED) are considered because
sync is about propagating status changes to *existing* edges, not about
deciding whether a claim deserves an edge.  The ``create_missing`` path applies
the same enrichment eligibility rules as ``enrich_graph_from_claims``.

Usage::

    from ml.evidence.store import ClaimStore
    from ml.graph.store import GraphStore
    from ml.graph.sync import sync_graph_from_store

    report = sync_graph_from_store(claim_store, graph_store)
    print(report.summary())

    # Also create edges for SUPPORTED claims that have no edge yet:
    report = sync_graph_from_store(
        claim_store, graph_store, create_missing=True
    )

    # Sync only specific claims:
    report = sync_graph_from_store(
        claim_store, graph_store, claim_ids=["abc...", "def..."]
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ml.evidence.schema import ClaimStatus, ClaimType
from ml.graph.converter import (
    claim_status_to_edge_status,
    edge_from_claim,
    nodes_from_claim,
)

if TYPE_CHECKING:
    from ml.evidence.store import ClaimStore
    from ml.graph.store import GraphStore

log = logging.getLogger(__name__)

# Claim types eligible for edge creation when create_missing=True
_ENRICHABLE_TYPES = {
    ClaimType.RELATIONSHIP,
    ClaimType.SOURCE_USEFULNESS,
    ClaimType.FEATURE_USEFULNESS,
    ClaimType.PERFORMANCE,
}

# Statuses that block create_missing creation (same rules as enrichment)
_NO_CREATE_STATUSES = {
    ClaimStatus.WEAK,
    ClaimStatus.REJECTED,
    ClaimStatus.ARCHIVED,
}


# ── SyncReport ────────────────────────────────────────────────────────────────

@dataclass
class SyncReport:
    """Summary of a single sync pass.

    Attributes:
        edges_updated:       Number of edge statuses updated.
        edges_created:       Number of new edges created (create_missing=True).
        unchanged:           Number of claims whose edges were already current.
        claims_skipped:      List of (claim_id, reason) for skipped claims.
        missing_claim_edges: List of claim_ids that had no graph edge and
                             ``create_missing`` was False.
    """
    edges_updated:       int = 0
    edges_created:       int = 0
    unchanged:           int = 0
    claims_skipped:      list[tuple[str, str]] = field(default_factory=list)
    missing_claim_edges: list[str]             = field(default_factory=list)

    @property
    def n_skipped(self) -> int:
        """Total number of skipped claims."""
        return len(self.claims_skipped)

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        lines = [
            "SyncReport",
            f"  edges updated       : {self.edges_updated}",
            f"  edges created       : {self.edges_created}",
            f"  unchanged           : {self.unchanged}",
            f"  missing (no edge)   : {len(self.missing_claim_edges)}",
            f"  claims skipped      : {self.n_skipped}",
        ]
        if self.claims_skipped:
            lines.append("  skip details:")
            for claim_id, reason in self.claims_skipped[:10]:
                lines.append(f"    {claim_id[:12]}... — {reason}")
            if self.n_skipped > 10:
                lines.append(f"    ... and {self.n_skipped - 10} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "edges_updated":       self.edges_updated,
            "edges_created":       self.edges_created,
            "unchanged":           self.unchanged,
            "missing_claim_edges": self.missing_claim_edges,
            "n_skipped":           self.n_skipped,
            "claims_skipped":      [
                {"claim_id": cid, "reason": r}
                for cid, r in self.claims_skipped
            ],
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def sync_graph_from_store(
    claim_store:      "ClaimStore",
    graph_store:      "GraphStore",
    claim_ids:        list[str] | None = None,
    create_missing:   bool = False,
    include_proposed: bool = False,
) -> SyncReport:
    """Synchronise graph edge statuses with current claim statuses.

    Inspects claims from ``claim_store`` and updates any graph edges whose
    status no longer matches the source claim's status.  Optionally creates
    edges for eligible claims that have no graph edge yet.

    Args:
        claim_store:      Source of claim records.
        graph_store:      Destination graph store to update.
        claim_ids:        If given, only these claim IDs are considered.  If
                          None, all claims are evaluated (active and inactive).
        create_missing:   If True, eligible claims without any graph edge will
                          have nodes and an edge created, as in enrichment.
                          Defaults to False (safe, update-only mode).
        include_proposed: When ``create_missing=True``, also create edges for
                          PROPOSED claims.  Ignored when ``create_missing=False``.

    Returns:
        A ``SyncReport`` describing what changed.
    """
    report = SyncReport()

    claims = _load_claims(claim_store, claim_ids)

    for claim in claims:
        existing_edges = graph_store.edges_by_claim(claim.claim_id)

        if existing_edges:
            _sync_existing_edges(claim, existing_edges, graph_store, report)
        else:
            _handle_missing_edge(
                claim, graph_store, report, create_missing, include_proposed
            )

    return report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_claims(
    claim_store: "ClaimStore",
    claim_ids:   list[str] | None,
) -> list[Any]:
    """Fetch the relevant claims from the store."""
    if claim_ids is not None:
        return [
            c for cid in claim_ids
            if (c := claim_store.get_claim(cid)) is not None
        ]
    # Include all claims — active and inactive — because sync must propagate
    # REJECTED and ARCHIVED status changes to existing edges.
    return claim_store.list_claims(active_only=False)


def _sync_existing_edges(
    claim:          Any,
    existing_edges: list[Any],
    graph_store:    "GraphStore",
    report:         SyncReport,
) -> None:
    """Update edge statuses for a claim that already has edges."""
    target_status = claim_status_to_edge_status(claim.status)
    any_updated = False

    for edge in existing_edges:
        if edge.status == target_status:
            continue  # already current
        notes = (
            f"synced from claim status={claim.status.value} "
            f"on claim_id={claim.claim_id[:12]}"
        )
        try:
            graph_store.update_edge_status(edge.edge_id, target_status, notes=notes)
            any_updated = True
            log.debug(
                "Edge %s: %s -> %s (claim %s)",
                edge.edge_id[:12],
                edge.status.value,
                target_status.value,
                claim.claim_id[:12],
            )
        except KeyError:
            log.warning(
                "Edge %s not found during sync; skipping.", edge.edge_id[:12]
            )

    if any_updated:
        report.edges_updated += len(existing_edges)
    else:
        report.unchanged += 1


def _handle_missing_edge(
    claim:            Any,
    graph_store:      "GraphStore",
    report:           SyncReport,
    create_missing:   bool,
    include_proposed: bool,
) -> None:
    """Handle a claim that has no graph edge yet."""
    if not create_missing:
        report.missing_claim_edges.append(claim.claim_id)
        return

    skip = _create_missing_skip_reason(claim, include_proposed)
    if skip:
        report.claims_skipped.append((claim.claim_id, skip))
        return

    # Create nodes + edge, mirroring enrichment logic
    try:
        subject_node, object_node = nodes_from_claim(claim)
        graph_store.upsert_node(subject_node)
        graph_store.upsert_node(object_node)
    except Exception as exc:
        reason = f"node conversion error: {exc}"
        report.claims_skipped.append((claim.claim_id, reason))
        log.warning("Node conversion failed for claim %s: %s", claim.claim_id[:12], exc)
        return

    try:
        edge = edge_from_claim(claim)
        graph_store.add_edge(edge)
        report.edges_created += 1
        log.debug("Created missing edge for claim %s", claim.claim_id[:12])
    except Exception as exc:
        reason = f"edge creation error: {exc}"
        report.claims_skipped.append((claim.claim_id, reason))
        log.warning("Edge creation failed for claim %s: %s", claim.claim_id[:12], exc)


def _create_missing_skip_reason(claim: Any, include_proposed: bool) -> str:
    """Return a skip reason for create_missing mode, or '' if eligible."""
    if claim.status in _NO_CREATE_STATUSES:
        return f"status={claim.status.value} — not eligible for edge creation"
    if claim.status == ClaimStatus.PROPOSED and not include_proposed:
        return "status=proposed — pass include_proposed=True to create proposed edges"
    if claim.claim_type not in _ENRICHABLE_TYPES:
        return f"claim_type={claim.claim_type.value} — not a supported enrichment type"
    return ""
