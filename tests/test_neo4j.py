"""Tests for the Neo4j client and schema module.

All tests mock the Neo4j driver so they run without live credentials.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_driver() -> MagicMock:
    """Return a mock Neo4j Driver with a chainable session context manager."""
    mock_result = MagicMock()
    mock_result.single.return_value = {"props": {"ticker": "AAPL"}}
    mock_result.__iter__ = lambda s: iter([])

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

class TestSchemaConstants:
    """Tests for NodeLabel and RelType constants."""

    def test_node_labels_defined(self) -> None:
        """All 7 node label constants are defined."""
        from db.neo4j.schema import NodeLabel
        assert NodeLabel.ASSET == "Asset"
        assert NodeLabel.SECTOR == "Sector"
        assert NodeLabel.INDUSTRY == "Industry"
        assert NodeLabel.MACRO_INDICATOR == "MacroIndicator"
        assert NodeLabel.EVENT == "Event"
        assert NodeLabel.SIGNAL == "Signal"
        assert NodeLabel.MODEL == "Model"

    def test_rel_types_defined(self) -> None:
        """All 8 relationship type constants are defined."""
        from db.neo4j.schema import RelType
        assert RelType.CORRELATED_WITH == "CORRELATED_WITH"
        assert RelType.CAUSES == "CAUSES"
        assert RelType.BELONGS_TO == "BELONGS_TO"
        assert RelType.GENERATES == "GENERATES"
        assert RelType.TRAINED_ON == "TRAINED_ON"
        assert RelType.TRIGGERED_BY == "TRIGGERED_BY"
        assert RelType.IMPACTS == "IMPACTS"
        assert RelType.PART_OF == "PART_OF"

    def test_constraints_list_not_empty(self) -> None:
        """CONSTRAINTS list has at least one entry per node label."""
        from db.neo4j.schema import CONSTRAINTS
        assert len(CONSTRAINTS) >= 6

    def test_indexes_list_not_empty(self) -> None:
        """INDEXES list has at least 4 entries."""
        from db.neo4j.schema import INDEXES
        assert len(INDEXES) >= 4

    def test_all_constraints_have_if_not_exists(self) -> None:
        """Every constraint Cypher uses IF NOT EXISTS for idempotency."""
        from db.neo4j.schema import CONSTRAINTS
        for name, cypher in CONSTRAINTS:
            assert "IF NOT EXISTS" in cypher, f"Constraint '{name}' missing IF NOT EXISTS"

    def test_all_indexes_have_if_not_exists(self) -> None:
        """Every index Cypher uses IF NOT EXISTS for idempotency."""
        from db.neo4j.schema import INDEXES
        for name, cypher in INDEXES:
            assert "IF NOT EXISTS" in cypher, f"Index '{name}' missing IF NOT EXISTS"


# ---------------------------------------------------------------------------
# apply_schema
# ---------------------------------------------------------------------------

class TestApplySchema:
    """Tests for apply_schema()."""

    def test_runs_all_constraints_and_indexes(self) -> None:
        """apply_schema runs Cypher for every constraint and index."""
        from db.neo4j.client import apply_schema
        from db.neo4j.schema import CONSTRAINTS, INDEXES

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            result = apply_schema()

        total_calls = mock_driver.session().run.call_count
        assert total_calls == len(CONSTRAINTS) + len(INDEXES)

    def test_returns_counts(self) -> None:
        """apply_schema returns dict with constraint and index counts."""
        from db.neo4j.client import apply_schema
        from db.neo4j.schema import CONSTRAINTS, INDEXES

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            result = apply_schema()

        assert result["constraints"] == len(CONSTRAINTS)
        assert result["indexes"] == len(INDEXES)


# ---------------------------------------------------------------------------
# Node merge helpers
# ---------------------------------------------------------------------------

class TestMergeAsset:
    """Tests for merge_asset()."""

    def test_merges_asset_node(self) -> None:
        """merge_asset runs a MERGE query against the Asset label."""
        from db.neo4j.client import AssetNode, merge_asset

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.single.return_value = {
            "props": {"ticker": "AAPL", "asset_class": "equity"}
        }

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            result = merge_asset(AssetNode(ticker="AAPL", name="Apple Inc.",
                                           asset_class="equity"))

        cypher = mock_driver.session().run.call_args[0][0]
        assert "MERGE" in cypher
        assert "Asset" in cypher

    def test_returns_node_properties(self) -> None:
        """merge_asset returns the node property dict."""
        from db.neo4j.client import AssetNode, merge_asset

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.single.return_value = {
            "props": {"ticker": "TSLA", "asset_class": "equity"}
        }

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            result = merge_asset(AssetNode(ticker="TSLA", name="Tesla", asset_class="equity"))

        assert result["ticker"] == "TSLA"


class TestMergeSector:
    """Tests for merge_sector()."""

    def test_merges_sector_node(self) -> None:
        """merge_sector runs a MERGE query against the Sector label."""
        from db.neo4j.client import SectorNode, merge_sector

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.single.return_value = {
            "props": {"name": "Technology"}
        }

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            result = merge_sector(SectorNode(name="Technology"))

        cypher = mock_driver.session().run.call_args[0][0]
        assert "Sector" in cypher


class TestMergeModel:
    """Tests for merge_model()."""

    def test_merges_model_node(self) -> None:
        """merge_model runs a MERGE query against the Model label."""
        from db.neo4j.client import ModelNode, merge_model

        mid = str(uuid.uuid4())
        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.single.return_value = {
            "props": {"model_id": mid}
        }

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            merge_model(ModelNode(model_id=mid, name="lstm_v1",
                                  model_type="lstm", version="1.0.0"))

        cypher = mock_driver.session().run.call_args[0][0]
        assert "Model" in cypher


# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------

class TestCreateBelongsTo:
    """Tests for create_belongs_to()."""

    def test_runs_merge_cypher(self) -> None:
        """create_belongs_to runs MERGE with BELONGS_TO."""
        from db.neo4j.client import create_belongs_to

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            create_belongs_to("AAPL", "Technology")

        cypher = mock_driver.session().run.call_args[0][0]
        assert "BELONGS_TO" in cypher


class TestCreateCorrelatedWith:
    """Tests for create_correlated_with()."""

    def test_runs_merge_cypher(self) -> None:
        """create_correlated_with runs MERGE with CORRELATED_WITH."""
        from db.neo4j.client import create_correlated_with

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            create_correlated_with("AAPL", "MSFT", correlation=0.75)

        cypher = mock_driver.session().run.call_args[0][0]
        assert "CORRELATED_WITH" in cypher

    def test_canonical_ordering(self) -> None:
        """create_correlated_with sorts tickers lexicographically to avoid duplicates."""
        from db.neo4j.client import create_correlated_with

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            create_correlated_with("MSFT", "AAPL", correlation=0.75)

        kwargs = mock_driver.session().run.call_args[1]
        assert kwargs["ticker_a"] == "AAPL"
        assert kwargs["ticker_b"] == "MSFT"


class TestCreateGenerates:
    """Tests for create_generates()."""

    def test_runs_merge_cypher(self) -> None:
        """create_generates runs MERGE with GENERATES."""
        from db.neo4j.client import create_generates

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            create_generates("model-1", "signal-1")

        cypher = mock_driver.session().run.call_args[0][0]
        assert "GENERATES" in cypher


class TestCreateTrainedOn:
    """Tests for create_trained_on()."""

    def test_runs_merge_cypher(self) -> None:
        """create_trained_on runs MERGE with TRAINED_ON."""
        from db.neo4j.client import create_trained_on

        mock_driver = _mock_driver()
        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            create_trained_on("model-1", "AAPL")

        cypher = mock_driver.session().run.call_args[0][0]
        assert "TRAINED_ON" in cypher


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestGetAssetNeighbours:
    """Tests for get_asset_neighbours()."""

    def test_returns_list_of_dicts(self) -> None:
        """get_asset_neighbours returns a list of relationship dicts."""
        from db.neo4j.client import get_asset_neighbours

        mock_row = MagicMock()
        mock_row.keys.return_value = ["source", "target", "relationship", "properties"]
        mock_row.__iter__ = lambda s: iter([
            ("source", "AAPL"), ("target", "MSFT"),
            ("relationship", "CORRELATED_WITH"), ("properties", {})
        ])
        mock_row.items = lambda: [
            ("source", "AAPL"), ("target", "MSFT"),
            ("relationship", "CORRELATED_WITH"), ("properties", {})
        ]

        mock_result = MagicMock()
        mock_result.__iter__ = lambda s: iter([mock_row])

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value = mock_result

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            rows = get_asset_neighbours("AAPL")

        assert isinstance(rows, list)

    def test_query_includes_ticker_param(self) -> None:
        """get_asset_neighbours passes the ticker as a Cypher parameter."""
        from db.neo4j.client import get_asset_neighbours

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.__iter__ = lambda s: iter([])

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            get_asset_neighbours("BTC")

        kwargs = mock_driver.session().run.call_args[1]
        assert kwargs.get("ticker") == "BTC"


class TestRunReadQuery:
    """Tests for run_read_query()."""

    def test_runs_arbitrary_cypher(self) -> None:
        """run_read_query passes the given Cypher to the session."""
        from db.neo4j.client import run_read_query

        mock_driver = _mock_driver()
        mock_driver.session().run.return_value.__iter__ = lambda s: iter([])

        with patch("db.neo4j.client.get_driver", return_value=mock_driver):
            run_read_query("MATCH (n) RETURN n LIMIT 1")

        called_cypher = mock_driver.session().run.call_args[0][0]
        assert "MATCH" in called_cypher


# ---------------------------------------------------------------------------
# Driver lifecycle
# ---------------------------------------------------------------------------

class TestDriverLifecycle:
    """Tests for get_driver() and close_driver()."""

    def test_raises_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_driver raises RuntimeError when NEO4J_URI is not set."""
        import db.neo4j.client as mod
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        mod._driver = None

        import sys
        for key in list(sys.modules.keys()):
            if key == "skills.env":
                del sys.modules[key]

        with pytest.raises(RuntimeError):
            mod.get_driver()

    def test_close_driver_resets_singleton(self) -> None:
        """close_driver sets _driver back to None."""
        import db.neo4j.client as mod

        mock_drv = MagicMock()
        mod._driver = mock_drv

        mod.close_driver()

        mock_drv.close.assert_called_once()
        assert mod._driver is None
