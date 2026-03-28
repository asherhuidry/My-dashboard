"""Tests for the graph visualization API routes.

Covers: knowledge graph endpoint (Neo4j live + fallback),
graph stats endpoint, node detail endpoint, and correlation endpoint structure.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    from api.routes.graph import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _mock_neo4j_session(rows: list[dict]) -> MagicMock:
    """Build a mock Neo4j session that returns the given rows."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = rows
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    return mock_session


_SAMPLE_ROW = {
    "src_id": 1, "src_label": "Asset", "src_name": "AAPL", "src_class": "equity",
    "rel": "SENSITIVE_TO", "rel_props": {"beta": 0.45, "p_value": 0.01, "regime": "all"},
    "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "VIXCLS", "tgt_class": None,
}


# ── Knowledge graph endpoint ─────────────────────────────────────────────────

class TestKnowledgeGraph:
    """Tests for GET /api/graph/knowledge."""

    @patch("db.neo4j.client.get_driver")
    def test_returns_neo4j_data(self, mock_driver) -> None:
        """When Neo4j is available, returns real graph with source='neo4j'."""
        mock_session = _mock_neo4j_session([_SAMPLE_ROW])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        resp = client.get("/api/graph/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "neo4j"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    @patch("db.neo4j.client.get_driver")
    def test_edge_includes_properties(self, mock_driver) -> None:
        """Edges should carry beta, regime, etc. from Neo4j relationship properties."""
        row = {
            **_SAMPLE_ROW,
            "rel_props": {"beta": -0.30, "p_value": 0.005, "regime": "stress", "factor_group": "volatility"},
        }
        mock_session = _mock_neo4j_session([row])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        edge = data["edges"][0]
        assert edge["label"] == "SENSITIVE_TO"
        assert edge["beta"] == -0.30
        assert edge["regime"] == "stress"
        assert edge["factor_group"] == "volatility"
        assert edge["width"] > 0

    @patch("db.neo4j.client.get_driver")
    def test_node_includes_asset_class(self, mock_driver) -> None:
        mock_session = _mock_neo4j_session([{
            "src_id": 1, "src_label": "Asset", "src_name": "AAPL", "src_class": "equity",
            "rel": "SENSITIVE_TO", "rel_props": {},
            "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "VIX", "tgt_class": None,
        }])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        aapl = [n for n in data["nodes"] if n["label"] == "AAPL"][0]
        assert aapl["class"] == "equity"
        assert aapl["type"] == "Asset"
        assert aapl["color"] == "#3b82f6"

    def test_neo4j_unavailable_returns_empty(self) -> None:
        """When Neo4j is down, returns honest empty graph, not static fallback."""
        client = _make_client()
        with patch("db.neo4j.client.get_driver", side_effect=RuntimeError("Neo4j unavailable")):
            resp = client.get("/api/graph/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "unavailable"
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["meta"]["node_count"] == 0

    @patch("db.neo4j.client.get_driver")
    def test_empty_graph_returns_zero_counts(self, mock_driver) -> None:
        mock_session = _mock_neo4j_session([])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        assert data["source"] == "neo4j"
        assert data["meta"]["node_count"] == 0
        assert data["meta"]["edge_count"] == 0

    @patch("db.neo4j.client.get_driver")
    def test_edge_color_by_relationship_type(self, mock_driver) -> None:
        mock_session = _mock_neo4j_session([
            {
                "src_id": 1, "src_label": "Asset", "src_name": "A", "src_class": None,
                "rel": "SENSITIVE_TO", "rel_props": {},
                "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "B", "tgt_class": None,
            },
            {
                "src_id": 3, "src_label": "Asset", "src_name": "C", "src_class": None,
                "rel": "CORRELATED_WITH", "rel_props": {},
                "tgt_id": 4, "tgt_label": "Asset", "tgt_name": "D", "tgt_class": None,
            },
        ])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        colors = {e["label"]: e["color"] for e in data["edges"]}
        assert colors["SENSITIVE_TO"] == "#f59e0b"
        assert colors["CORRELATED_WITH"] == "#10b981"

    @patch("db.neo4j.client.get_driver")
    def test_node_degree_and_val(self, mock_driver) -> None:
        """Nodes should have degree count and val computed from degree."""
        mock_session = _mock_neo4j_session([
            {
                "src_id": 1, "src_label": "Asset", "src_name": "HUB", "src_class": None,
                "rel": "SENSITIVE_TO", "rel_props": {},
                "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "X", "tgt_class": None,
            },
            {
                "src_id": 1, "src_label": "Asset", "src_name": "HUB", "src_class": None,
                "rel": "CORRELATED_WITH", "rel_props": {},
                "tgt_id": 3, "tgt_label": "Asset", "tgt_name": "Y", "tgt_class": None,
            },
        ])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        hub = [n for n in data["nodes"] if n["label"] == "HUB"][0]
        assert hub["degree"] == 2
        assert hub["val"] >= 4

    @patch("db.neo4j.client.get_driver")
    def test_meta_includes_rel_types(self, mock_driver) -> None:
        """Meta should include rel_type counts."""
        mock_session = _mock_neo4j_session([_SAMPLE_ROW])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        assert "rel_types" in data["meta"]
        assert data["meta"]["rel_types"]["SENSITIVE_TO"] == 1

    @patch("db.neo4j.client.get_driver")
    def test_meta_includes_active_filters(self, mock_driver) -> None:
        """Meta should echo back the active filters."""
        mock_session = _mock_neo4j_session([])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge", params={"rel_type": "SENSITIVE_TO"}).json()
        assert data["meta"]["filters"]["rel_type"] == "SENSITIVE_TO"
        assert data["meta"]["filters"]["asset_class"] is None

    @patch("db.neo4j.client.get_driver")
    def test_filter_params_passed_to_cypher(self, mock_driver) -> None:
        """Filter params should appear in the Cypher query params."""
        mock_session = _mock_neo4j_session([])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        client.get("/api/graph/knowledge", params={
            "rel_type": "SENSITIVE_TO",
            "asset_class": "equity",
            "regime": "stress",
        })

        # Verify session.run was called with params containing our filters
        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params.get("rel_type") == "SENSITIVE_TO"
        assert params.get("asset_class") == "equity"
        assert params.get("regime") == "stress"


# ── Node detail endpoint ─────────────────────────────────────────────────────

class TestNodeDetail:
    """Tests for GET /api/graph/node/{node_name}."""

    @patch("db.neo4j.client.get_driver")
    def test_returns_node_info(self, mock_driver) -> None:
        """Should return node metadata and grouped relationships."""
        mock_session = _mock_neo4j_session([{
            "node_label": "Asset", "node_name": "AAPL",
            "node_class": "equity", "node_sector": "Technology",
            "rel": "SENSITIVE_TO",
            "rel_props": {"beta": 0.45, "p_value": 0.01, "regime": "all"},
            "neighbor_label": "MacroIndicator", "neighbor_name": "VIXCLS",
            "neighbor_class": None, "direction": "outgoing",
        }])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/node/AAPL").json()
        assert data["found"] is True
        assert data["node"]["name"] == "AAPL"
        assert data["node"]["type"] == "Asset"
        assert data["node"]["class"] == "equity"
        assert data["node"]["sector"] == "Technology"
        assert data["total_edges"] == 1
        assert "SENSITIVE_TO" in data["relationships"]
        rel = data["relationships"]["SENSITIVE_TO"][0]
        assert rel["neighbor"] == "VIXCLS"
        assert rel["direction"] == "outgoing"
        assert rel["beta"] == 0.45

    @patch("db.neo4j.client.get_driver")
    def test_multiple_rel_types_grouped(self, mock_driver) -> None:
        """Relationships should be grouped by type."""
        mock_session = _mock_neo4j_session([
            {
                "node_label": "Asset", "node_name": "AAPL",
                "node_class": "equity", "node_sector": None,
                "rel": "SENSITIVE_TO", "rel_props": {"beta": 0.3},
                "neighbor_label": "MacroIndicator", "neighbor_name": "VIX",
                "neighbor_class": None, "direction": "outgoing",
            },
            {
                "node_label": "Asset", "node_name": "AAPL",
                "node_class": "equity", "node_sector": None,
                "rel": "CORRELATED_WITH", "rel_props": {"correlation": 0.8},
                "neighbor_label": "Asset", "neighbor_name": "MSFT",
                "neighbor_class": "equity", "direction": "outgoing",
            },
        ])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/node/AAPL").json()
        assert len(data["relationships"]) == 2
        assert len(data["relationships"]["SENSITIVE_TO"]) == 1
        assert len(data["relationships"]["CORRELATED_WITH"]) == 1
        assert data["total_edges"] == 2

    @patch("db.neo4j.client.get_driver")
    def test_not_found_returns_found_false(self, mock_driver) -> None:
        """Unknown node returns found=False."""
        mock_session = _mock_neo4j_session([])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/node/UNKNOWN").json()
        assert data["found"] is False
        assert data["node_name"] == "UNKNOWN"

    def test_neo4j_unavailable_returns_error(self) -> None:
        """When Neo4j is down, returns found=False with error."""
        client = _make_client()
        with patch("db.neo4j.client.get_driver", side_effect=RuntimeError("down")):
            data = client.get("/api/graph/node/AAPL").json()
        assert data["found"] is False
        assert "error" in data

    @patch("db.neo4j.client.get_driver")
    def test_excludes_updated_at_from_props(self, mock_driver) -> None:
        """updated_at should not leak into relationship properties."""
        mock_session = _mock_neo4j_session([{
            "node_label": "Asset", "node_name": "AAPL",
            "node_class": "equity", "node_sector": None,
            "rel": "SENSITIVE_TO",
            "rel_props": {"beta": 0.2, "updated_at": "2025-01-01T00:00:00"},
            "neighbor_label": "MacroIndicator", "neighbor_name": "VIX",
            "neighbor_class": None, "direction": "outgoing",
        }])
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/node/AAPL").json()
        rel = data["relationships"]["SENSITIVE_TO"][0]
        assert "updated_at" not in rel
        assert rel["beta"] == 0.2


# ── Graph stats endpoint ─────────────────────────────────────────────────────

class TestGraphStats:
    """Tests for GET /api/graph/stats."""

    @patch("db.neo4j.client.get_graph_stats")
    def test_returns_neo4j_stats(self, mock_stats) -> None:
        mock_stats.return_value = {
            "nodes": {"Asset": 40, "MacroIndicator": 10},
            "edges": {"SENSITIVE_TO": 200},
            "total_nodes": 50,
            "total_edges": 200,
        }
        client = _make_client()
        data = client.get("/api/graph/stats").json()
        assert data["source"] == "neo4j"
        assert data["total_nodes"] == 50

    def test_neo4j_unavailable_returns_zeros(self) -> None:
        with patch("db.neo4j.client.get_graph_stats", side_effect=RuntimeError("down")):
            client = _make_client()
            data = client.get("/api/graph/stats").json()
        assert data["source"] == "unavailable"
        assert data["total_nodes"] == 0
