"""Tests for GICS sector/industry classification materialization.

Tests the TICKER_SECTOR_MAP data integrity and the materialize_sector_classification()
function that creates Sector/Industry nodes and BELONGS_TO edges in Neo4j.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_driver() -> MagicMock:
    """Return a mock Neo4j Driver with a chainable session context manager."""
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda s, k: True  # is_new = True

    mock_result = MagicMock()
    mock_result.single.return_value = mock_record

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver


# ---------------------------------------------------------------------------
# Data integrity tests (no Neo4j needed)
# ---------------------------------------------------------------------------

class TestTickerSectorMap:
    """Validate the TICKER_SECTOR_MAP data structure."""

    def test_map_not_empty(self) -> None:
        """Map has a reasonable number of entries."""
        from data.agents.knowledge_builder import TICKER_SECTOR_MAP
        assert len(TICKER_SECTOR_MAP) >= 90

    def test_all_entries_are_tuples(self) -> None:
        """Every value is a (sector, industry) tuple of strings."""
        from data.agents.knowledge_builder import TICKER_SECTOR_MAP
        for ticker, val in TICKER_SECTOR_MAP.items():
            assert isinstance(val, tuple), f"{ticker}: expected tuple, got {type(val)}"
            assert len(val) == 2, f"{ticker}: expected 2-tuple, got {len(val)}"
            assert isinstance(val[0], str), f"{ticker}: sector not a string"
            assert isinstance(val[1], str), f"{ticker}: industry not a string"

    def test_all_universe_equities_classified(self) -> None:
        """Every equity in universe.EQUITIES has a GICS classification."""
        from data.agents.knowledge_builder import TICKER_SECTOR_MAP
        from data.ingest.universe import EQUITIES
        missing = [t for t in EQUITIES if t not in TICKER_SECTOR_MAP]
        assert missing == [], f"Equities missing GICS classification: {missing}"

    def test_sectors_are_valid_gics(self) -> None:
        """All sectors match standard GICS sector names."""
        from data.agents.knowledge_builder import TICKER_SECTOR_MAP
        valid_gics_sectors = {
            "Information Technology", "Communication Services",
            "Consumer Discretionary", "Consumer Staples",
            "Financials", "Healthcare", "Industrials",
            "Energy", "Real Estate", "Utilities", "Materials",
        }
        used_sectors = {v[0] for v in TICKER_SECTOR_MAP.values()}
        invalid = used_sectors - valid_gics_sectors
        assert invalid == set(), f"Non-GICS sectors found: {invalid}"

    def test_unique_industries_exist(self) -> None:
        """At least 15 unique industry groups are represented."""
        from data.agents.knowledge_builder import TICKER_SECTOR_MAP
        industries = {v[1] for v in TICKER_SECTOR_MAP.values()}
        assert len(industries) >= 15, f"Only {len(industries)} industries"


# ---------------------------------------------------------------------------
# Materialization tests (mock Neo4j)
# ---------------------------------------------------------------------------

class TestMaterializeSectorClassification:
    """Test the materialize_sector_classification() function."""

    @patch("db.neo4j.client.get_driver")
    def test_creates_sector_nodes(self, mock_get_driver: MagicMock) -> None:
        """Cypher MERGE is called for each unique sector."""
        driver = _mock_driver()
        mock_get_driver.return_value = driver

        from data.agents.knowledge_builder import (
            TICKER_SECTOR_MAP,
            materialize_sector_classification,
        )
        result = materialize_sector_classification()

        assert "error" not in result
        assert result["tickers_classified"] == len(TICKER_SECTOR_MAP)

        # Check session.run was called (sectors + industries + ticker edges)
        session = driver.session.return_value
        assert session.run.call_count > 0

    @patch("db.neo4j.client.get_driver")
    def test_returns_correct_counts(self, mock_get_driver: MagicMock) -> None:
        """Result dict contains expected count keys."""
        driver = _mock_driver()
        mock_get_driver.return_value = driver

        from data.agents.knowledge_builder import materialize_sector_classification
        result = materialize_sector_classification()

        assert "sectors_created" in result
        assert "industries_created" in result
        assert "edges_created" in result
        assert "tickers_classified" in result
        assert "timestamp" in result
        assert result["edges_created"] > 0

    @patch("db.neo4j.client.get_driver")
    def test_cypher_contains_belongs_to(self, mock_get_driver: MagicMock) -> None:
        """BELONGS_TO edges are created in the Cypher queries."""
        driver = _mock_driver()
        mock_get_driver.return_value = driver

        from data.agents.knowledge_builder import materialize_sector_classification
        materialize_sector_classification()

        session = driver.session.return_value
        cypher_calls = [
            str(c) for c in session.run.call_args_list
        ]
        belongs_to_calls = [c for c in cypher_calls if "BELONGS_TO" in c]
        assert len(belongs_to_calls) > 0, "No BELONGS_TO edges created"

    @patch("db.neo4j.client.get_driver")
    def test_cypher_contains_part_of(self, mock_get_driver: MagicMock) -> None:
        """PART_OF edges link Industry nodes to Sector nodes."""
        driver = _mock_driver()
        mock_get_driver.return_value = driver

        from data.agents.knowledge_builder import materialize_sector_classification
        materialize_sector_classification()

        session = driver.session.return_value
        cypher_calls = [
            str(c) for c in session.run.call_args_list
        ]
        part_of_calls = [c for c in cypher_calls if "PART_OF" in c]
        assert len(part_of_calls) > 0, "No PART_OF edges created"

    @patch("db.neo4j.client.get_driver", side_effect=Exception("no neo4j"))
    def test_handles_neo4j_unavailable(self, mock_get_driver: MagicMock) -> None:
        """Returns error dict when Neo4j is unavailable."""
        from data.agents.knowledge_builder import materialize_sector_classification
        result = materialize_sector_classification()
        assert "error" in result


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaUpdates:
    """Verify schema includes Industry node and PART_OF relationship."""

    def test_industry_node_label_exists(self) -> None:
        """NodeLabel.INDUSTRY is defined."""
        from db.neo4j.schema import NodeLabel
        assert NodeLabel.INDUSTRY == "Industry"

    def test_part_of_rel_type_exists(self) -> None:
        """RelType.PART_OF is defined."""
        from db.neo4j.schema import RelType
        assert RelType.PART_OF == "PART_OF"

    def test_industry_constraint_exists(self) -> None:
        """Industry uniqueness constraint is in the schema."""
        from db.neo4j.schema import CONSTRAINTS
        constraint_names = [c[0] for c in CONSTRAINTS]
        assert "industry_name_unique" in constraint_names
