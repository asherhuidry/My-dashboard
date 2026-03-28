"""Tests for the graph export (Neo4j → tensor) pipeline."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch, MagicMock

import torch
import pytest

from ml.neural.graph_export import export_graph_tensors, _REL_TYPE_MAP


class _DictRecord(dict):
    """Dict that also supports attribute-style access like Neo4j Record."""
    def __getattr__(self, key: str) -> Any:
        return self[key]


def _mock_session():
    """Build a mock Neo4j session with sample graph data."""
    session = MagicMock()

    asset_records = [_DictRecord(id="AAPL"), _DictRecord(id="MSFT"), _DictRecord(id="NVDA")]
    macro_records = [_DictRecord(id="FEDFUNDS"), _DictRecord(id="CPIAUCSL")]
    sector_records = [_DictRecord(id="Technology")]
    event_records = [_DictRecord(id="FOMC_2026-01-28")]

    edge_records = [
        _DictRecord(src="AAPL", tgt="MSFT", rel_type="CORRELATED_WITH",
                    confidence=0.75, pearson_r=0.6, evidence_count=3, age_days=0),
        _DictRecord(src="FEDFUNDS", tgt="AAPL", rel_type="CAUSES",
                    confidence=0.85, pearson_r=0.5, evidence_count=2, age_days=0),
        _DictRecord(src="NVDA", tgt="CPIAUCSL", rel_type="SENSITIVE_TO",
                    confidence=0.45, pearson_r=-0.3, evidence_count=1, age_days=0),
    ]

    call_count = {"n": 0}

    def mock_run(cypher, **kwargs):
        result = MagicMock()
        if "Asset" in cypher:
            result.__iter__ = lambda _: iter(asset_records)
        elif "MacroIndicator" in cypher:
            result.__iter__ = lambda _: iter(macro_records)
        elif "Sector" in cypher:
            result.__iter__ = lambda _: iter(sector_records)
        elif "Event" in cypher:
            result.__iter__ = lambda _: iter(event_records)
        else:
            result.__iter__ = lambda _: iter(edge_records)
        return result

    session.run = mock_run
    return session


@patch("db.neo4j.client.get_driver")
class TestExportGraphTensors:
    """Tests for the Neo4j → tensor export pipeline."""

    def _setup_driver(self, mock_get_driver: MagicMock) -> None:
        driver = MagicMock()
        sess = _mock_session()
        driver.session.return_value.__enter__ = lambda _: sess
        driver.session.return_value.__exit__ = lambda *_: None
        mock_get_driver.return_value = driver

    def test_returns_all_keys(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        assert "node_ids" in result
        assert "edge_index" in result
        assert "edge_types" in result
        assert "edge_weights" in result
        assert "n_nodes" in result
        assert "n_edges" in result
        assert "stats" in result

    def test_node_count(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        # 3 assets + 2 macro + 1 sector + 1 event = 7
        assert result["n_nodes"] == 7

    def test_edge_count(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        assert result["n_edges"] == 3

    def test_edge_index_shape(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        assert result["edge_index"].shape[0] == 2
        assert result["edge_index"].shape[1] == 3

    def test_confidence_as_weights(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors(use_confidence_weights=True)
        weights = result["edge_weights"].tolist()
        # Float32 precision: check with tolerance
        assert any(abs(w - 0.75) < 0.01 for w in weights)
        assert any(abs(w - 0.85) < 0.01 for w in weights)

    def test_pearson_r_fallback_weights(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors(use_confidence_weights=False)
        weights = result["edge_weights"].tolist()
        assert any(abs(w - 0.6) < 0.01 for w in weights)

    def test_edge_types_mapped(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        etypes = result["edge_types"].tolist()
        assert _REL_TYPE_MAP["CORRELATED_WITH"] in etypes
        assert _REL_TYPE_MAP["CAUSES"] in etypes

    def test_node_ids_in_order(self, mock_get_driver: MagicMock) -> None:
        self._setup_driver(mock_get_driver)
        result = export_graph_tensors()
        assert len(result["node_ids"]) == result["n_nodes"]
        assert "AAPL" in result["node_ids"]
        assert "FEDFUNDS" in result["node_ids"]

    def test_decay_applied_to_aged_edges(self, mock_get_driver: MagicMock) -> None:
        """Verify that old edges get decayed weights, not raw confidence."""
        driver = MagicMock()
        sess = MagicMock()

        asset_records = [_DictRecord(id="AAPL"), _DictRecord(id="MSFT")]
        edge_records = [
            # 365-day-old edge with 0.80 confidence, single evidence
            _DictRecord(src="AAPL", tgt="MSFT", rel_type="CORRELATED_WITH",
                        confidence=0.80, pearson_r=0.5, evidence_count=1, age_days=365),
        ]

        def mock_run(cypher, **kwargs):
            result = MagicMock()
            if "Asset" in cypher:
                result.__iter__ = lambda _: iter(asset_records)
            elif "MacroIndicator" in cypher or "Sector" in cypher or "Event" in cypher:
                result.__iter__ = lambda _: iter([])
            else:
                result.__iter__ = lambda _: iter(edge_records)
            return result

        sess.run = mock_run
        driver.session.return_value.__enter__ = lambda _: sess
        driver.session.return_value.__exit__ = lambda *_: None
        mock_get_driver.return_value = driver

        result = export_graph_tensors(use_confidence_weights=True)
        weight = result["edge_weights"][0].item()
        # With 365 days age and half_life=90, decay ≈ exp(-365*ln2/90) ≈ 0.058
        # Weight should be well below the raw 0.80
        assert weight < 0.40, f"Expected decayed weight < 0.40, got {weight}"
        # Floor is 10% of raw (0.80 * 0.10 = 0.08), decayed value sits at floor
        assert weight >= 0.07, f"Expected weight >= 0.07 (near floor), got {weight}"

    def test_min_confidence_filters_on_effective(self, mock_get_driver: MagicMock) -> None:
        """Verify that min_confidence filters against effective (decayed) weight."""
        driver = MagicMock()
        sess = MagicMock()

        asset_records = [_DictRecord(id="A"), _DictRecord(id="B"), _DictRecord(id="C")]
        edge_records = [
            # Fresh edge — confidence 0.60, will pass 0.50 filter
            _DictRecord(src="A", tgt="B", rel_type="CORRELATED_WITH",
                        confidence=0.60, pearson_r=0.5, evidence_count=2, age_days=0),
            # Very old edge — raw 0.60 but effective ≪ 0.50 after decay
            _DictRecord(src="B", tgt="C", rel_type="CORRELATED_WITH",
                        confidence=0.60, pearson_r=0.5, evidence_count=1, age_days=400),
        ]

        def mock_run(cypher, **kwargs):
            result = MagicMock()
            if "Asset" in cypher:
                result.__iter__ = lambda _: iter(asset_records)
            elif "MacroIndicator" in cypher or "Sector" in cypher or "Event" in cypher:
                result.__iter__ = lambda _: iter([])
            else:
                result.__iter__ = lambda _: iter(edge_records)
            return result

        sess.run = mock_run
        driver.session.return_value.__enter__ = lambda _: sess
        driver.session.return_value.__exit__ = lambda *_: None
        mock_get_driver.return_value = driver

        result = export_graph_tensors(min_confidence=0.50, use_confidence_weights=True)
        # Only the fresh edge should survive the filter
        assert result["n_edges"] == 1

    def test_empty_graph(self, mock_get_driver: MagicMock) -> None:
        driver = MagicMock()
        sess = MagicMock()
        sess.run.return_value.__iter__ = lambda _: iter([])
        driver.session.return_value.__enter__ = lambda _: sess
        driver.session.return_value.__exit__ = lambda *_: None
        mock_get_driver.return_value = driver

        result = export_graph_tensors()
        assert result["n_nodes"] == 0
        assert result["n_edges"] == 0
        assert result["edge_index"].shape == (2, 0)
