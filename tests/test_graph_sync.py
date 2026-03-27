"""Tests for the ml/graph sync layer.

Covers:
- GraphStore.update_edge_status
- GraphStore.has_edge_for_claim
- claim_status_to_edge_status (centralized mapping)
- sync_graph_from_store: status propagation, unchanged detection,
  create_missing flag, include_proposed flag, idempotency,
  duplicate-edge prevention, SyncReport structure
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ml.evidence.schema import Claim, ClaimStatus, ClaimType
from ml.evidence.store import ClaimStore
from ml.graph.converter import claim_status_to_edge_status, edge_from_claim
from ml.graph.enrichment import enrich_graph_from_claims
from ml.graph.schema import GraphEdge, GraphEdgeStatus, GraphNode, NodeType
from ml.graph.store import GraphStore
from ml.graph.sync import SyncReport, sync_graph_from_store


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    return GraphStore(path=tmp_path / "graph.json")


@pytest.fixture
def claim_store(tmp_path: Path) -> ClaimStore:
    return ClaimStore(path=tmp_path / "claims.json")


def _claim(
    claim_type: ClaimType  = ClaimType.RELATIONSHIP,
    subject:    str        = "asset:AAPL",
    predicate:  str        = "is_correlated_with",
    obj:        str        = "asset:QQQ",
    status:     ClaimStatus = ClaimStatus.SUPPORTED,
    confidence: float      = 0.7,
) -> Claim:
    c = Claim.new(
        claim_type = claim_type,
        subject    = subject,
        predicate  = predicate,
        obj        = obj,
        confidence = confidence,
    )
    object.__setattr__(c, "status", status)
    return c


def _add_edge_for_claim(graph_store: GraphStore, claim: Claim) -> GraphEdge:
    """Helper: enrich a single claim into the graph store and return the edge."""
    edge = edge_from_claim(claim)
    graph_store.upsert_node(GraphNode.from_entity_id(claim.subject))
    graph_store.upsert_node(GraphNode.from_entity_id(claim.object))
    graph_store.add_edge(edge)
    return edge


# ── Phase 1: GraphStore lifecycle helpers ─────────────────────────────────────

class TestUpdateEdgeStatus:
    def test_updates_status(self, graph_store):
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.PROPOSED, 0.6, "c1")
        graph_store.add_edge(edge)
        graph_store.update_edge_status(edge.edge_id, GraphEdgeStatus.ACTIVE)
        updated = graph_store.get_edge(edge.edge_id)
        assert updated.status == GraphEdgeStatus.ACTIVE

    def test_updates_timestamp(self, graph_store):
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.PROPOSED, 0.6, "c1")
        graph_store.add_edge(edge)
        old_ts = edge.updated_at
        graph_store.update_edge_status(edge.edge_id, GraphEdgeStatus.ACTIVE)
        updated = graph_store.get_edge(edge.edge_id)
        assert updated.updated_at >= old_ts

    def test_stores_notes_in_properties(self, graph_store):
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.PROPOSED, 0.6, "c1")
        graph_store.add_edge(edge)
        graph_store.update_edge_status(
            edge.edge_id, GraphEdgeStatus.ACTIVE, notes="synced"
        )
        updated = graph_store.get_edge(edge.edge_id)
        assert updated.properties.get("sync_notes") == "synced"

    def test_raises_for_unknown_edge(self, graph_store):
        with pytest.raises(KeyError):
            graph_store.update_edge_status("nonexistent", GraphEdgeStatus.ACTIVE)

    def test_persisted_to_disk(self, tmp_path):
        path = tmp_path / "g.json"
        gs1 = GraphStore(path=path)
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.PROPOSED, 0.5, "c1")
        gs1.add_edge(edge)
        gs1.update_edge_status(edge.edge_id, GraphEdgeStatus.ACTIVE)

        gs2 = GraphStore(path=path)
        assert gs2.get_edge(edge.edge_id).status == GraphEdgeStatus.ACTIVE


class TestHasEdgeForClaim:
    def test_false_when_no_edge(self, graph_store):
        assert graph_store.has_edge_for_claim("nonexistent") is False

    def test_true_when_edge_exists(self, graph_store):
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, 0.7, "claim_x")
        graph_store.add_edge(edge)
        assert graph_store.has_edge_for_claim("claim_x") is True

    def test_false_for_different_claim(self, graph_store):
        edge = GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, 0.7, "claim_x")
        graph_store.add_edge(edge)
        assert graph_store.has_edge_for_claim("claim_y") is False


# ── Phase 2: Centralized status mapping ───────────────────────────────────────

class TestClaimStatusToEdgeStatus:
    def test_supported_maps_to_active(self):
        assert claim_status_to_edge_status(ClaimStatus.SUPPORTED) == GraphEdgeStatus.ACTIVE

    def test_proposed_maps_to_proposed(self):
        assert claim_status_to_edge_status(ClaimStatus.PROPOSED) == GraphEdgeStatus.PROPOSED

    def test_weak_maps_to_weak(self):
        assert claim_status_to_edge_status(ClaimStatus.WEAK) == GraphEdgeStatus.WEAK

    def test_rejected_maps_to_rejected(self):
        assert claim_status_to_edge_status(ClaimStatus.REJECTED) == GraphEdgeStatus.REJECTED

    def test_archived_maps_to_archived(self):
        assert claim_status_to_edge_status(ClaimStatus.ARCHIVED) == GraphEdgeStatus.ARCHIVED

    def test_edge_from_claim_uses_same_mapping(self):
        """edge_from_claim must produce the same status as claim_status_to_edge_status."""
        for cs, expected_es in [
            (ClaimStatus.SUPPORTED, GraphEdgeStatus.ACTIVE),
            (ClaimStatus.PROPOSED,  GraphEdgeStatus.PROPOSED),
            (ClaimStatus.WEAK,      GraphEdgeStatus.WEAK),
            (ClaimStatus.REJECTED,  GraphEdgeStatus.REJECTED),
            (ClaimStatus.ARCHIVED,  GraphEdgeStatus.ARCHIVED),
        ]:
            c = _claim(status=cs)
            e = edge_from_claim(c)
            assert e.status == expected_es, f"Mismatch for {cs}: {e.status} != {expected_es}"


# ── Phase 3: sync_graph_from_store — core behaviour ──────────────────────────

class TestSyncReport:
    def test_summary_contains_all_fields(self):
        r = SyncReport(edges_updated=2, edges_created=1, unchanged=3)
        r.missing_claim_edges.append("abc")
        r.claims_skipped.append(("def", "reason"))
        s = r.summary()
        assert "edges updated" in s
        assert "edges created" in s
        assert "unchanged" in s
        assert "missing" in s
        assert "skipped" in s

    def test_to_dict(self):
        r = SyncReport(edges_updated=1, unchanged=2)
        r.missing_claim_edges.append("cid1")
        r.claims_skipped.append(("cid2", "some reason"))
        d = r.to_dict()
        assert d["edges_updated"]       == 1
        assert d["unchanged"]           == 2
        assert d["missing_claim_edges"] == ["cid1"]
        assert d["n_skipped"]           == 1
        assert d["claims_skipped"][0]["claim_id"] == "cid2"


class TestSyncStatusPropagation:
    def test_proposed_edge_updated_to_active_when_claim_supported(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)
        edge = _add_edge_for_claim(graph_store, c)
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.PROPOSED

        # Promote claim
        claim_store.update_status(c.claim_id, ClaimStatus.SUPPORTED)

        report = sync_graph_from_store(claim_store, graph_store)
        assert report.edges_updated == 1
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.ACTIVE

    def test_supported_edge_updated_to_rejected_when_claim_rejected(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        edge = _add_edge_for_claim(graph_store, c)

        claim_store.update_status(c.claim_id, ClaimStatus.REJECTED)

        report = sync_graph_from_store(claim_store, graph_store)
        assert report.edges_updated >= 1
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.REJECTED

    def test_supported_edge_updated_to_archived(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        edge = _add_edge_for_claim(graph_store, c)

        claim_store.update_status(c.claim_id, ClaimStatus.ARCHIVED)

        report = sync_graph_from_store(claim_store, graph_store)
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.ARCHIVED

    def test_already_current_counted_as_unchanged(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        _add_edge_for_claim(graph_store, c)

        report = sync_graph_from_store(claim_store, graph_store)
        assert report.unchanged    == 1
        assert report.edges_updated == 0

    def test_sync_note_written_to_edge_properties(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)
        edge = _add_edge_for_claim(graph_store, c)

        claim_store.update_status(c.claim_id, ClaimStatus.SUPPORTED)
        sync_graph_from_store(claim_store, graph_store)

        updated = graph_store.get_edge(edge.edge_id)
        assert "sync_notes" in updated.properties

    def test_multiple_edges_for_one_claim_all_updated(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)
        # Manually add two edges for the same claim
        e1 = GraphEdge.new("asset:AAPL", "asset:QQQ", "r",
                           GraphEdgeStatus.PROPOSED, 0.6, c.claim_id)
        e2 = GraphEdge.new("asset:AAPL", "asset:SPY", "r",
                           GraphEdgeStatus.PROPOSED, 0.6, c.claim_id)
        graph_store.add_edge(e1)
        graph_store.add_edge(e2)

        claim_store.update_status(c.claim_id, ClaimStatus.SUPPORTED)
        report = sync_graph_from_store(claim_store, graph_store)

        assert report.edges_updated == 2
        assert graph_store.get_edge(e1.edge_id).status == GraphEdgeStatus.ACTIVE
        assert graph_store.get_edge(e2.edge_id).status == GraphEdgeStatus.ACTIVE


class TestSyncMissingEdges:
    def test_missing_edge_recorded_when_create_missing_false(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        report = sync_graph_from_store(claim_store, graph_store, create_missing=False)
        assert c.claim_id in report.missing_claim_edges
        assert report.edges_created == 0

    def test_missing_edge_created_when_flag_set(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        report = sync_graph_from_store(claim_store, graph_store, create_missing=True)
        assert report.edges_created      == 1
        assert len(report.missing_claim_edges) == 0
        assert graph_store.has_edge_for_claim(c.claim_id)

    def test_proposed_not_created_without_include_proposed(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)

        report = sync_graph_from_store(
            claim_store, graph_store, create_missing=True, include_proposed=False
        )
        assert report.edges_created == 0
        assert report.n_skipped     == 1

    def test_proposed_created_with_include_proposed(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)

        report = sync_graph_from_store(
            claim_store, graph_store, create_missing=True, include_proposed=True
        )
        assert report.edges_created == 1

    def test_weak_not_created_even_with_create_missing(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.WEAK)
        claim_store.add_claim(c)

        report = sync_graph_from_store(
            claim_store, graph_store,
            create_missing=True, include_proposed=True,
            claim_ids=[c.claim_id],
        )
        assert report.edges_created == 0
        assert report.n_skipped     == 1
        assert "weak" in report.claims_skipped[0][1]

    def test_general_claim_type_not_created(self, claim_store, graph_store):
        c = _claim(claim_type=ClaimType.GENERAL, status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        report = sync_graph_from_store(
            claim_store, graph_store, create_missing=True
        )
        assert report.edges_created == 0
        assert report.n_skipped     == 1


class TestSyncClaimIdsFilter:
    def test_only_specified_claims_synced(self, claim_store, graph_store):
        c1 = _claim(subject="asset:AAPL", obj="asset:QQQ")
        c2 = _claim(
            claim_type=ClaimType.FEATURE_USEFULNESS,
            subject="feature:rsi_14",
            predicate="improves_accuracy_on",
            obj="model:mlp_on_AAPL",
        )
        claim_store.add_claim(c1)
        claim_store.add_claim(c2)

        edge1 = _add_edge_for_claim(graph_store, c1)
        edge2 = _add_edge_for_claim(graph_store, c2)

        # Change both claim statuses
        claim_store.update_status(c1.claim_id, ClaimStatus.REJECTED)
        claim_store.update_status(c2.claim_id, ClaimStatus.REJECTED)

        # Sync only c1
        report = sync_graph_from_store(
            claim_store, graph_store, claim_ids=[c1.claim_id]
        )
        assert report.edges_updated == 1
        assert graph_store.get_edge(edge1.edge_id).status == GraphEdgeStatus.REJECTED
        # c2 not touched
        assert graph_store.get_edge(edge2.edge_id).status == GraphEdgeStatus.ACTIVE


# ── Phase 4: Idempotency and duplicate prevention ─────────────────────────────

class TestSyncIdempotency:
    def test_repeated_sync_no_status_change_is_stable(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        _add_edge_for_claim(graph_store, c)

        r1 = sync_graph_from_store(claim_store, graph_store)
        r2 = sync_graph_from_store(claim_store, graph_store)
        r3 = sync_graph_from_store(claim_store, graph_store)

        assert r1.edges_updated == 0 and r1.unchanged == 1
        assert r2.edges_updated == 0 and r2.unchanged == 1
        assert r3.edges_updated == 0 and r3.unchanged == 1

    def test_repeated_sync_after_status_change_stable(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)
        _add_edge_for_claim(graph_store, c)

        claim_store.update_status(c.claim_id, ClaimStatus.SUPPORTED)

        r1 = sync_graph_from_store(claim_store, graph_store)
        r2 = sync_graph_from_store(claim_store, graph_store)

        assert r1.edges_updated == 1
        assert r2.edges_updated == 0 and r2.unchanged == 1  # already synced

    def test_create_missing_idempotent(self, claim_store, graph_store):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        r1 = sync_graph_from_store(claim_store, graph_store, create_missing=True)
        r2 = sync_graph_from_store(claim_store, graph_store, create_missing=True)

        assert r1.edges_created == 1
        # Second call: edge now exists, so unchanged (not created again)
        assert r2.edges_created == 0
        assert r2.unchanged     == 1
        # Only one edge in the store
        assert len(graph_store.edges_by_claim(c.claim_id)) == 1

    def test_no_duplicate_edges_after_enrichment_then_sync(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        # Enrich first
        enrich_graph_from_claims(claim_store, graph_store)
        assert len(graph_store.edges_by_claim(c.claim_id)) == 1

        # Sync with create_missing should not add another edge
        report = sync_graph_from_store(
            claim_store, graph_store, create_missing=True
        )
        assert report.edges_created == 0
        assert len(graph_store.edges_by_claim(c.claim_id)) == 1


# ── Integration ───────────────────────────────────────────────────────────────

class TestSyncIntegration:
    def test_full_lifecycle_proposed_to_supported_to_rejected(
        self, claim_store, graph_store
    ):
        c = _claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)
        edge = _add_edge_for_claim(graph_store, c)

        # Step 1: still proposed — unchanged
        r = sync_graph_from_store(claim_store, graph_store)
        assert r.unchanged == 1

        # Step 2: promote to supported
        claim_store.update_status(c.claim_id, ClaimStatus.SUPPORTED)
        r = sync_graph_from_store(claim_store, graph_store)
        assert r.edges_updated == 1
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.ACTIVE

        # Step 3: reject
        claim_store.update_status(c.claim_id, ClaimStatus.REJECTED)
        r = sync_graph_from_store(claim_store, graph_store)
        assert r.edges_updated == 1
        assert graph_store.get_edge(edge.edge_id).status == GraphEdgeStatus.REJECTED

        # Step 4: sync again — already rejected, no more updates
        r = sync_graph_from_store(claim_store, graph_store)
        assert r.edges_updated == 0 and r.unchanged == 1

    def test_graph_persisted_after_sync(self, tmp_path):
        c_path = tmp_path / "claims.json"
        g_path = tmp_path / "graph.json"

        cs = ClaimStore(path=c_path)
        gs = GraphStore(path=g_path)

        c = _claim(status=ClaimStatus.PROPOSED)
        cs.add_claim(c)
        edge = _add_edge_for_claim(gs, c)

        cs.update_status(c.claim_id, ClaimStatus.SUPPORTED)
        sync_graph_from_store(cs, gs)

        gs2 = GraphStore(path=g_path)
        assert gs2.get_edge(edge.edge_id).status == GraphEdgeStatus.ACTIVE

    def test_mixed_batch_update_and_unchanged(self, claim_store, graph_store):
        # c1: already current
        c1 = _claim(subject="asset:AAPL", obj="asset:QQQ",
                    status=ClaimStatus.SUPPORTED)
        # c2: needs update
        c2 = _claim(
            claim_type=ClaimType.FEATURE_USEFULNESS,
            subject="feature:rsi_14", predicate="improves_accuracy_on",
            obj="model:mlp_on_AAPL",
            status=ClaimStatus.PROPOSED,
        )
        # c3: no edge, create_missing=False → missing
        c3 = _claim(
            claim_type=ClaimType.SOURCE_USEFULNESS,
            subject="source:FRED_GDP", predicate="is_useful_for",
            obj="domain:macro_regime",
            status=ClaimStatus.SUPPORTED,
        )
        claim_store.add_claim(c1)
        claim_store.add_claim(c2)
        claim_store.add_claim(c3)

        _add_edge_for_claim(graph_store, c1)
        _add_edge_for_claim(graph_store, c2)
        # c3 gets no edge

        claim_store.update_status(c2.claim_id, ClaimStatus.SUPPORTED)

        report = sync_graph_from_store(claim_store, graph_store)

        assert report.unchanged           == 1  # c1
        assert report.edges_updated       == 1  # c2
        assert len(report.missing_claim_edges) == 1  # c3
        assert c3.claim_id in report.missing_claim_edges
