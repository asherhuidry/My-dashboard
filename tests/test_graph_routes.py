"""Tests for the graph visualization API routes.

Covers: knowledge graph endpoint (Neo4j live + fallback),
graph stats endpoint, and correlation endpoint structure.
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


# ── Knowledge graph endpoint ─────────────────────────────────────────────────

class TestKnowledgeGraph:
    """Tests for GET /api/graph/knowledge."""

    @patch("db.neo4j.client.get_driver")
    def test_returns_neo4j_data(self, mock_driver) -> None:
        """When Neo4j is available, returns real graph with source='neo4j'."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
            {
                "src_id": 1, "src_label": "Asset", "src_name": "AAPL", "src_class": "equity",
                "rel": "SENSITIVE_TO", "rel_props": {"beta": 0.45, "p_value": 0.01, "regime": "all"},
                "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "VIXCLS", "tgt_class": None,
            },
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
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
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
            {
                "src_id": 1, "src_label": "Asset", "src_name": "AAPL", "src_class": "equity",
                "rel": "SENSITIVE_TO", "rel_props": {"beta": -0.30, "p_value": 0.005, "regime": "stress", "factor_group": "volatility"},
                "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "VIXCLS", "tgt_class": None,
            },
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
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
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
            {
                "src_id": 1, "src_label": "Asset", "src_name": "AAPL", "src_class": "equity",
                "rel": "SENSITIVE_TO", "rel_props": {},
                "tgt_id": 2, "tgt_label": "MacroIndicator", "tgt_name": "VIX", "tgt_class": None,
            },
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
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
        # get_driver import will fail since no Neo4j → exception path
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
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        assert data["source"] == "neo4j"
        assert data["meta"]["node_count"] == 0
        assert data["meta"]["edge_count"] == 0

    @patch("db.neo4j.client.get_driver")
    def test_edge_color_by_relationship_type(self, mock_driver) -> None:
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [
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
        ]
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.return_value.session.return_value = mock_session

        client = _make_client()
        data = client.get("/api/graph/knowledge").json()
        colors = {e["label"]: e["color"] for e in data["edges"]}
        assert colors["SENSITIVE_TO"] == "#f59e0b"
        assert colors["CORRELATED_WITH"] == "#10b981"


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
