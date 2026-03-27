"""Tests for the ml/graph layer.

Covers:
- GraphNode and GraphEdge schema (serialisation, factory, validation)
- GraphStore (CRUD, upsert, query, persistence)
- Converter (nodes_from_claim, edge_from_claim)
- Enrichment (enrich_graph_from_claims, skip logic, report)
- Integration (claim store -> graph store round-trip)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.graph.schema import GraphEdge, GraphEdgeStatus, GraphNode, NodeType
from ml.graph.store import GraphStore
from ml.graph.converter import edge_from_claim, nodes_from_claim
from ml.graph.enrichment import EnrichmentReport, enrich_graph_from_claims
from ml.evidence.schema import Claim, ClaimStatus, ClaimType
from ml.evidence.store import ClaimStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph_store(tmp_path: Path) -> GraphStore:
    return GraphStore(path=tmp_path / "graph.json")


@pytest.fixture
def claim_store(tmp_path: Path) -> ClaimStore:
    return ClaimStore(path=tmp_path / "claims.json")


def _make_claim(
    claim_type: ClaimType = ClaimType.RELATIONSHIP,
    subject:    str = "asset:AAPL",
    predicate:  str = "is_correlated_with",
    obj:        str = "asset:QQQ",
    status:     ClaimStatus = ClaimStatus.SUPPORTED,
    confidence: float = 0.7,
    tags:       list[str] | None = None,
    uncertainty_notes: str = "",
    counterpoints: list[str] | None = None,
) -> Claim:
    c = Claim.new(
        claim_type        = claim_type,
        subject           = subject,
        predicate         = predicate,
        obj               = obj,
        confidence        = confidence,
        uncertainty_notes = uncertainty_notes,
        counterpoints     = counterpoints,
        tags              = tags,
    )
    # Manually override status (Claim.new always starts PROPOSED)
    object.__setattr__(c, "status", status)
    return c


# ── Phase 1: GraphNode schema ─────────────────────────────────────────────────

class TestGraphNodeSchema:
    def test_from_entity_id_feature(self):
        n = GraphNode.from_entity_id("feature:rsi_14")
        assert n.node_id   == "feature:rsi_14"
        assert n.node_type == NodeType.FEATURE
        assert n.label     == "rsi_14"

    def test_from_entity_id_source(self):
        n = GraphNode.from_entity_id("source:FRED_GDP")
        assert n.node_type == NodeType.SOURCE
        assert n.label     == "FRED_GDP"

    def test_from_entity_id_asset(self):
        n = GraphNode.from_entity_id("asset:AAPL")
        assert n.node_type == NodeType.ASSET
        assert n.label     == "AAPL"

    def test_from_entity_id_model(self):
        n = GraphNode.from_entity_id("model:mlp_on_AAPL")
        assert n.node_type == NodeType.MODEL
        assert n.label     == "mlp_on_AAPL"

    def test_from_entity_id_dataset(self):
        n = GraphNode.from_entity_id("dataset:abc123")
        assert n.node_type == NodeType.DATASET

    def test_from_entity_id_domain(self):
        n = GraphNode.from_entity_id("domain:macro_regime")
        assert n.node_type == NodeType.DOMAIN

    def test_from_entity_id_unknown_namespace(self):
        n = GraphNode.from_entity_id("widget:foo")
        assert n.node_type == NodeType.UNKNOWN
        assert n.label     == "foo"

    def test_from_entity_id_no_colon(self):
        n = GraphNode.from_entity_id("plain_id")
        assert n.node_type == NodeType.UNKNOWN
        assert n.label     == "plain_id"

    def test_roundtrip(self):
        n = GraphNode.from_entity_id("feature:macd", {"foo": 1})
        d = n.to_dict()
        n2 = GraphNode.from_dict(d)
        assert n2.node_id    == n.node_id
        assert n2.node_type  == n.node_type
        assert n2.label      == n.label
        assert n2.properties == n.properties

    def test_timestamps_set_on_construction(self):
        n = GraphNode.from_entity_id("asset:SPY")
        assert n.created_at
        assert n.updated_at


# ── Phase 1: GraphEdge schema ─────────────────────────────────────────────────

class TestGraphEdgeSchema:
    def test_new_creates_uuid(self):
        e1 = GraphEdge.new("a", "b", "rel", GraphEdgeStatus.ACTIVE, 0.7, "cid")
        e2 = GraphEdge.new("a", "b", "rel", GraphEdgeStatus.ACTIVE, 0.7, "cid")
        assert e1.edge_id != e2.edge_id

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, 1.1, "c")
        with pytest.raises(ValueError):
            GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, -0.1, "c")

    def test_roundtrip(self):
        e = GraphEdge.new(
            "asset:AAPL", "asset:QQQ", "is_correlated_with",
            GraphEdgeStatus.ACTIVE, 0.75, "claim_abc",
            evidence_ids=["ev1", "ev2"],
            properties={"note": "60d rolling"},
        )
        d = e.to_dict()
        e2 = GraphEdge.from_dict(d)
        assert e2.edge_id        == e.edge_id
        assert e2.relation       == e.relation
        assert e2.status         == e.status
        assert e2.confidence     == e.confidence
        assert e2.evidence_ids   == e.evidence_ids
        assert e2.properties     == e.properties

    def test_evidence_ids_default_empty(self):
        e = GraphEdge.new("a", "b", "r", GraphEdgeStatus.PROPOSED, 0.5, "c")
        assert e.evidence_ids == []


# ── Phase 2: GraphStore ───────────────────────────────────────────────────────

class TestGraphStore:
    def test_add_and_get_node(self, graph_store):
        n = GraphNode.from_entity_id("feature:rsi_14")
        graph_store.add_node(n)
        retrieved = graph_store.get_node("feature:rsi_14")
        assert retrieved is not None
        assert retrieved.node_id == "feature:rsi_14"

    def test_add_node_duplicate_raises(self, graph_store):
        n = GraphNode.from_entity_id("asset:AAPL")
        graph_store.add_node(n)
        with pytest.raises(ValueError):
            graph_store.add_node(n)

    def test_upsert_node_creates_new(self, graph_store):
        n = GraphNode.from_entity_id("asset:SPY")
        is_new = graph_store.upsert_node(n)
        assert is_new is True
        assert graph_store.get_node("asset:SPY") is not None

    def test_upsert_node_updates_existing(self, graph_store):
        n = GraphNode.from_entity_id("asset:SPY")
        graph_store.upsert_node(n)
        n2 = GraphNode.from_entity_id("asset:SPY", {"added": True})
        is_new = graph_store.upsert_node(n2)
        assert is_new is False
        updated = graph_store.get_node("asset:SPY")
        assert updated.properties.get("added") is True

    def test_list_nodes_unfiltered(self, graph_store):
        graph_store.upsert_node(GraphNode.from_entity_id("asset:AAPL"))
        graph_store.upsert_node(GraphNode.from_entity_id("feature:rsi_14"))
        assert len(graph_store.list_nodes()) == 2

    def test_list_nodes_filtered_by_type(self, graph_store):
        graph_store.upsert_node(GraphNode.from_entity_id("asset:AAPL"))
        graph_store.upsert_node(GraphNode.from_entity_id("feature:rsi_14"))
        assets = graph_store.list_nodes(node_type=NodeType.ASSET)
        assert len(assets) == 1
        assert assets[0].node_id == "asset:AAPL"

    def test_add_and_get_edge(self, graph_store):
        e = GraphEdge.new("asset:AAPL", "asset:QQQ", "corr", GraphEdgeStatus.ACTIVE, 0.8, "c1")
        graph_store.add_edge(e)
        retrieved = graph_store.get_edge(e.edge_id)
        assert retrieved is not None
        assert retrieved.relation == "corr"

    def test_add_edge_duplicate_raises(self, graph_store):
        e = GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, 0.5, "c")
        graph_store.add_edge(e)
        with pytest.raises(ValueError):
            graph_store.add_edge(e)

    def test_list_edges_by_relation(self, graph_store):
        e1 = GraphEdge.new("a", "b", "corr", GraphEdgeStatus.ACTIVE, 0.7, "c1")
        e2 = GraphEdge.new("b", "c", "perf", GraphEdgeStatus.ACTIVE, 0.6, "c2")
        graph_store.add_edge(e1)
        graph_store.add_edge(e2)
        assert len(graph_store.list_edges(relation="corr")) == 1

    def test_list_edges_by_status(self, graph_store):
        e1 = GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE,   0.7, "c1")
        e2 = GraphEdge.new("b", "c", "r", GraphEdgeStatus.PROPOSED, 0.5, "c2")
        graph_store.add_edge(e1)
        graph_store.add_edge(e2)
        active = graph_store.list_edges(status=GraphEdgeStatus.ACTIVE)
        assert len(active) == 1

    def test_edges_by_claim(self, graph_store):
        e1 = GraphEdge.new("a", "b", "r", GraphEdgeStatus.ACTIVE, 0.7, "claim_X")
        e2 = GraphEdge.new("b", "c", "r", GraphEdgeStatus.ACTIVE, 0.6, "claim_Y")
        graph_store.add_edge(e1)
        graph_store.add_edge(e2)
        result = graph_store.edges_by_claim("claim_X")
        assert len(result) == 1
        assert result[0].claim_id == "claim_X"

    def test_edges_from(self, graph_store):
        e1 = GraphEdge.new("asset:AAPL", "asset:QQQ", "corr", GraphEdgeStatus.ACTIVE, 0.7, "c1")
        e2 = GraphEdge.new("asset:QQQ",  "asset:SPY",  "corr", GraphEdgeStatus.ACTIVE, 0.6, "c2")
        graph_store.add_edge(e1)
        graph_store.add_edge(e2)
        outgoing = graph_store.edges_from("asset:AAPL")
        assert len(outgoing) == 1
        assert outgoing[0].target_node_id == "asset:QQQ"

    def test_edges_to(self, graph_store):
        e = GraphEdge.new("asset:AAPL", "asset:QQQ", "corr", GraphEdgeStatus.ACTIVE, 0.7, "c1")
        graph_store.add_edge(e)
        incoming = graph_store.edges_to("asset:QQQ")
        assert len(incoming) == 1

    def test_neighbors(self, graph_store):
        graph_store.upsert_node(GraphNode.from_entity_id("asset:AAPL"))
        graph_store.upsert_node(GraphNode.from_entity_id("asset:QQQ"))
        e = GraphEdge.new("asset:AAPL", "asset:QQQ", "corr", GraphEdgeStatus.ACTIVE, 0.7, "c1")
        graph_store.add_edge(e)
        nbrs = graph_store.neighbors("asset:AAPL")
        assert any(n.node_id == "asset:QQQ" for n in nbrs)

    def test_neighbors_returns_only_known_nodes(self, graph_store):
        # Edge exists but target node not in store
        e = GraphEdge.new("asset:AAPL", "asset:GHOST", "r", GraphEdgeStatus.ACTIVE, 0.5, "c1")
        graph_store.add_edge(e)
        nbrs = graph_store.neighbors("asset:AAPL")
        assert len(nbrs) == 0

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "graph.json"
        store1 = GraphStore(path=path)
        store1.upsert_node(GraphNode.from_entity_id("feature:macd"))
        e = GraphEdge.new("feature:macd", "model:mlp_on_AAPL", "improves",
                          GraphEdgeStatus.ACTIVE, 0.65, "claim_1")
        store1.add_edge(e)

        store2 = GraphStore(path=path)
        assert store2.get_node("feature:macd") is not None
        assert store2.get_edge(e.edge_id) is not None

    def test_summary_stats(self, graph_store):
        graph_store.upsert_node(GraphNode.from_entity_id("asset:AAPL"))
        graph_store.upsert_node(GraphNode.from_entity_id("feature:rsi_14"))
        e = GraphEdge.new("feature:rsi_14", "asset:AAPL", "r",
                          GraphEdgeStatus.ACTIVE, 0.6, "c1")
        graph_store.add_edge(e)
        stats = graph_store.summary_stats()
        assert stats["n_nodes"] == 2
        assert stats["n_edges"] == 1
        assert stats["by_node_type"]["asset"]   == 1
        assert stats["by_node_type"]["feature"] == 1
        assert stats["by_edge_status"]["active"] == 1


# ── Phase 3: Converter ────────────────────────────────────────────────────────

class TestNodesFromClaim:
    def test_returns_two_nodes(self):
        c = _make_claim()
        nodes = nodes_from_claim(c)
        assert len(nodes) == 2

    def test_subject_node_id(self):
        c = _make_claim(subject="feature:rsi_14", obj="model:mlp_on_AAPL")
        subj, _ = nodes_from_claim(c)
        assert subj.node_id   == "feature:rsi_14"
        assert subj.node_type == NodeType.FEATURE

    def test_object_node_id(self):
        c = _make_claim(subject="feature:rsi_14", obj="model:mlp_on_AAPL")
        _, obj_node = nodes_from_claim(c)
        assert obj_node.node_id   == "model:mlp_on_AAPL"
        assert obj_node.node_type == NodeType.MODEL

    def test_source_usefulness_claim(self):
        c = _make_claim(
            claim_type = ClaimType.SOURCE_USEFULNESS,
            subject    = "source:FRED_GDP",
            predicate  = "is_useful_for",
            obj        = "domain:macro_regime",
        )
        subj, obj_node = nodes_from_claim(c)
        assert subj.node_type    == NodeType.SOURCE
        assert obj_node.node_type == NodeType.DOMAIN


class TestEdgeFromClaim:
    def test_returns_graph_edge(self):
        c = _make_claim()
        e = edge_from_claim(c)
        assert isinstance(e, GraphEdge)

    def test_edge_carries_claim_fields(self):
        c = _make_claim(
            subject="asset:AAPL", predicate="is_correlated_with", obj="asset:QQQ",
            confidence=0.75, status=ClaimStatus.SUPPORTED,
        )
        e = edge_from_claim(c)
        assert e.source_node_id == "asset:AAPL"
        assert e.target_node_id == "asset:QQQ"
        assert e.relation       == "is_correlated_with"
        assert e.confidence     == pytest.approx(0.75)
        assert e.claim_id       == c.claim_id
        assert e.status         == GraphEdgeStatus.ACTIVE

    def test_proposed_claim_becomes_proposed_edge(self):
        c = _make_claim(status=ClaimStatus.PROPOSED)
        e = edge_from_claim(c)
        assert e.status == GraphEdgeStatus.PROPOSED

    def test_weak_claim_becomes_weak_edge(self):
        c = _make_claim(status=ClaimStatus.WEAK)
        e = edge_from_claim(c)
        assert e.status == GraphEdgeStatus.WEAK

    def test_rejected_claim_becomes_rejected_edge(self):
        c = _make_claim(status=ClaimStatus.REJECTED)
        e = edge_from_claim(c)
        assert e.status == GraphEdgeStatus.REJECTED

    def test_status_override(self):
        c = _make_claim(status=ClaimStatus.PROPOSED)
        e = edge_from_claim(c, status=GraphEdgeStatus.ACTIVE)
        assert e.status == GraphEdgeStatus.ACTIVE

    def test_uncertainty_notes_in_properties(self):
        c = _make_claim(uncertainty_notes="Only tested on 2y window")
        e = edge_from_claim(c)
        assert e.properties["uncertainty_notes"] == "Only tested on 2y window"

    def test_counterpoints_in_properties(self):
        c = _make_claim(counterpoints=["Noisy in low-vol regimes"])
        e = edge_from_claim(c)
        assert "Noisy in low-vol regimes" in e.properties["counterpoints"]

    def test_evidence_ids_carried(self):
        c = _make_claim()
        object.__setattr__(c, "evidence_ids", ["ev1", "ev2"])
        e = edge_from_claim(c)
        assert e.evidence_ids == ["ev1", "ev2"]


# ── Phase 4: Enrichment ───────────────────────────────────────────────────────

class TestEnrichmentReport:
    def test_summary_string(self):
        r = EnrichmentReport(nodes_created=3, nodes_updated=1, edges_created=2)
        s = r.summary()
        assert "nodes created : 3" in s
        assert "edges created : 2" in s
        assert "claims skipped: 0" in s

    def test_summary_shows_skip_details(self):
        r = EnrichmentReport()
        r.skipped.append(("abc123", "status=weak"))
        s = r.summary()
        assert "abc123" in s

    def test_to_dict(self):
        r = EnrichmentReport(nodes_created=2, edges_created=1)
        r.skipped.append(("cid1", "reason1"))
        d = r.to_dict()
        assert d["nodes_created"] == 2
        assert d["edges_created"] == 1
        assert d["n_skipped"]     == 1
        assert d["skipped"][0]["claim_id"] == "cid1"


class TestEnrichGraphFromClaims:
    def test_supported_claim_creates_nodes_and_edge(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)

        report = enrich_graph_from_claims(claim_store, graph_store)
        assert report.edges_created == 1
        assert report.nodes_created == 2
        assert report.n_skipped     == 0

    def test_proposed_claim_skipped_by_default(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)

        report = enrich_graph_from_claims(claim_store, graph_store)
        assert report.edges_created == 0
        assert report.n_skipped     == 1

    def test_proposed_claim_included_with_flag(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.PROPOSED)
        claim_store.add_claim(c)

        report = enrich_graph_from_claims(
            claim_store, graph_store, include_proposed=True
        )
        assert report.edges_created == 1

    def test_weak_claim_always_skipped(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.WEAK)
        claim_store.add_claim(c)

        report = enrich_graph_from_claims(
            claim_store, graph_store, include_proposed=True
        )
        assert report.edges_created == 0
        assert report.n_skipped     == 1
        assert "weak" in report.skipped[0][1]

    def test_rejected_claim_always_skipped(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.REJECTED)
        claim_store.add_claim(c)
        # Pass claim_ids explicitly so the enricher sees it despite active_only filter
        report = enrich_graph_from_claims(
            claim_store, graph_store, claim_ids=[c.claim_id]
        )
        assert report.n_skipped == 1
        assert "rejected" in report.skipped[0][1]

    def test_archived_claim_always_skipped(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.ARCHIVED)
        claim_store.add_claim(c)
        report = enrich_graph_from_claims(
            claim_store, graph_store, claim_ids=[c.claim_id]
        )
        assert report.n_skipped == 1
        assert "archived" in report.skipped[0][1]

    def test_general_claim_type_skipped(self, claim_store, graph_store):
        c = _make_claim(
            claim_type = ClaimType.GENERAL,
            status     = ClaimStatus.SUPPORTED,
        )
        claim_store.add_claim(c)
        report = enrich_graph_from_claims(claim_store, graph_store)
        assert report.n_skipped == 1
        assert "general" in report.skipped[0][1]

    def test_duplicate_nodes_are_updated_not_created(self, claim_store, graph_store):
        # Two claims sharing the same subject
        c1 = _make_claim(
            subject="asset:AAPL", obj="asset:QQQ",
            predicate="is_correlated_with", status=ClaimStatus.SUPPORTED,
        )
        c2 = _make_claim(
            subject="asset:AAPL", obj="asset:SPY",
            predicate="is_correlated_with", status=ClaimStatus.SUPPORTED,
        )
        claim_store.add_claim(c1)
        claim_store.add_claim(c2)

        report = enrich_graph_from_claims(claim_store, graph_store)
        # AAPL appears in both claims — second time it's an update
        assert report.edges_created == 2
        assert report.nodes_created + report.nodes_updated == 4

    def test_specific_claim_ids_filter(self, claim_store, graph_store):
        c1 = _make_claim(
            subject="asset:AAPL", obj="asset:QQQ",
            status=ClaimStatus.SUPPORTED,
        )
        c2 = _make_claim(
            subject="feature:rsi_14", obj="model:mlp_on_AAPL",
            claim_type=ClaimType.FEATURE_USEFULNESS,
            predicate="improves_accuracy_on",
            status=ClaimStatus.SUPPORTED,
        )
        claim_store.add_claim(c1)
        claim_store.add_claim(c2)

        report = enrich_graph_from_claims(
            claim_store, graph_store, claim_ids=[c1.claim_id]
        )
        assert report.edges_created == 1

    def test_edge_has_claim_id(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        enrich_graph_from_claims(claim_store, graph_store)

        edges = graph_store.list_edges()
        assert len(edges) == 1
        assert edges[0].claim_id == c.claim_id

    def test_graph_store_persisted_to_disk(self, tmp_path):
        c_path = tmp_path / "claims.json"
        g_path = tmp_path / "graph.json"
        cs = ClaimStore(path=c_path)
        gs = GraphStore(path=g_path)

        c = _make_claim(status=ClaimStatus.SUPPORTED)
        cs.add_claim(c)
        enrich_graph_from_claims(cs, gs)

        gs2 = GraphStore(path=g_path)
        assert len(gs2.list_edges()) == 1
        assert len(gs2.list_nodes()) == 2


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_round_trip_all_claim_types(self, claim_store, graph_store):
        claims = [
            _make_claim(
                claim_type=ClaimType.RELATIONSHIP,
                subject="asset:AAPL", predicate="is_correlated_with", obj="asset:QQQ",
                status=ClaimStatus.SUPPORTED, confidence=0.70,
            ),
            _make_claim(
                claim_type=ClaimType.SOURCE_USEFULNESS,
                subject="source:FRED_GDP", predicate="is_useful_for",
                obj="domain:macro_regime",
                status=ClaimStatus.SUPPORTED, confidence=0.60,
            ),
            _make_claim(
                claim_type=ClaimType.FEATURE_USEFULNESS,
                subject="feature:rsi_14", predicate="improves_accuracy_on",
                obj="model:mlp_dataset:abc123",
                status=ClaimStatus.SUPPORTED, confidence=0.65,
            ),
            _make_claim(
                claim_type=ClaimType.PERFORMANCE,
                subject="model:mlp_on_AAPL", predicate="meets_performance_bar",
                obj="metric:beat_bm_majority",
                status=ClaimStatus.SUPPORTED, confidence=0.55,
            ),
        ]
        for c in claims:
            claim_store.add_claim(c)

        report = enrich_graph_from_claims(claim_store, graph_store)

        assert report.edges_created == 4
        assert report.n_skipped     == 0

        stats = graph_store.summary_stats()
        assert stats["n_edges"] == 4
        # 8 distinct entity IDs (no overlap in this set)
        assert stats["n_nodes"] == 8

    def test_neighbors_reachable_after_enrichment(self, claim_store, graph_store):
        c = _make_claim(
            subject="feature:rsi_14", predicate="improves_accuracy_on",
            obj="model:mlp_on_AAPL",
            claim_type=ClaimType.FEATURE_USEFULNESS,
            status=ClaimStatus.SUPPORTED,
        )
        claim_store.add_claim(c)
        enrich_graph_from_claims(claim_store, graph_store)

        nbrs = graph_store.neighbors("feature:rsi_14")
        assert any(n.node_id == "model:mlp_on_AAPL" for n in nbrs)

    def test_edge_linked_back_to_claim(self, claim_store, graph_store):
        c = _make_claim(status=ClaimStatus.SUPPORTED)
        claim_store.add_claim(c)
        enrich_graph_from_claims(claim_store, graph_store)

        edges = graph_store.edges_by_claim(c.claim_id)
        assert len(edges) == 1
        assert edges[0].confidence == pytest.approx(c.confidence)
