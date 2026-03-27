"""Tests for the graph materializer agent.

Tests classification logic, node/edge building, and the full materialize flow
with mocked Supabase + Neo4j dependencies.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data.agents.graph_materializer import (
    classify_series,
    _build_asset_node,
    _build_macro_node,
    _build_edge,
    _build_series_lookup,
    materialize,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

ASSET_LOOKUP = {"SPY": "etf", "AAPL": "equity", "BTC-USD": "crypto", "GLD": "etf"}
MACRO_LOOKUP = {
    "VIXCLS": ("CBOE VIX", "daily"),
    "GS10": ("10Y Treasury Yield", "daily"),
    "DFF": ("Fed Funds Rate Daily", "daily"),
}

SAMPLE_DISCOVERY_ASSET_ASSET = {
    "series_a": "SPY",
    "series_b": "AAPL",
    "pearson_r": 0.85,
    "lag_days": 0,
    "granger_p": 0.12,
    "mutual_info": 0.3,
    "strength": "strong",
    "regime": "all",
    "relationship_type": "discovered",
}

SAMPLE_DISCOVERY_MACRO_ASSET = {
    "series_a": "VIXCLS",
    "series_b": "SPY",
    "pearson_r": -0.82,
    "lag_days": 1,
    "granger_p": 0.003,  # Granger-causal
    "mutual_info": 0.5,
    "strength": "strong",
    "regime": "all",
    "relationship_type": "volatility_equity",
}

SAMPLE_DISCOVERY_WEAK = {
    "series_a": "GLD",
    "series_b": "BTC-USD",
    "pearson_r": 0.38,
    "lag_days": 5,
    "granger_p": 0.45,
    "mutual_info": 0.1,
    "strength": "weak",
    "regime": "all",
    "relationship_type": "discovered",
}


# ── Classification tests ─────────────────────────────────────────────────────

class TestClassifySeries:
    def test_known_asset(self) -> None:
        assert classify_series("SPY", ASSET_LOOKUP, MACRO_LOOKUP) == "Asset"

    def test_known_macro(self) -> None:
        assert classify_series("VIXCLS", ASSET_LOOKUP, MACRO_LOOKUP) == "MacroIndicator"

    def test_unknown_defaults_to_asset(self) -> None:
        assert classify_series("UNKNOWN_TICKER", ASSET_LOOKUP, MACRO_LOOKUP) == "Asset"

    def test_crypto_ticker(self) -> None:
        assert classify_series("BTC-USD", ASSET_LOOKUP, MACRO_LOOKUP) == "Asset"


# ── Node builder tests ────────────────────────────────────────────────────────

class TestBuildNodes:
    def test_asset_node_known(self) -> None:
        node = _build_asset_node("SPY", ASSET_LOOKUP)
        assert node["ticker"] == "SPY"
        assert node["asset_class"] == "etf"
        assert node["name"] == "SPY"

    def test_asset_node_unknown_class(self) -> None:
        node = _build_asset_node("ZZZZ", ASSET_LOOKUP)
        assert node["asset_class"] == "unknown"

    def test_macro_node_known(self) -> None:
        node = _build_macro_node("VIXCLS", MACRO_LOOKUP)
        assert node["series_id"] == "VIXCLS"
        assert node["name"] == "CBOE VIX"
        assert node["source"] == "fred"
        assert node["frequency"] == "daily"

    def test_macro_node_unknown(self) -> None:
        node = _build_macro_node("UNKNOWN", MACRO_LOOKUP)
        assert node["series_id"] == "UNKNOWN"
        assert node["name"] == "UNKNOWN"  # Falls back to series_id


# ── Edge builder tests ────────────────────────────────────────────────────────

class TestBuildEdge:
    def test_asset_asset_no_granger(self) -> None:
        """Asset↔Asset with non-significant Granger → only CORRELATED_WITH."""
        edges = _build_edge(SAMPLE_DISCOVERY_ASSET_ASSET, ASSET_LOOKUP, MACRO_LOOKUP)
        assert len(edges) == 1
        assert edges[0]["rel_type"] == "CORRELATED_WITH"
        assert edges[0]["source_label"] == "Asset"
        assert edges[0]["target_label"] == "Asset"
        assert edges[0]["pearson_r"] == 0.85

    def test_macro_asset_with_granger(self) -> None:
        """Macro→Asset with significant Granger → CORRELATED_WITH + CAUSES."""
        edges = _build_edge(SAMPLE_DISCOVERY_MACRO_ASSET, ASSET_LOOKUP, MACRO_LOOKUP)
        assert len(edges) == 2

        corr = next(e for e in edges if e["rel_type"] == "CORRELATED_WITH")
        assert corr["strength"] == "strong"

        causes = next(e for e in edges if e["rel_type"] == "CAUSES")
        assert causes["source_id"] == "VIXCLS"
        assert causes["target_id"] == "SPY"
        assert causes["source_label"] == "MacroIndicator"
        assert causes["target_label"] == "Asset"

    def test_canonical_ordering(self) -> None:
        """CORRELATED_WITH should use canonical (alphabetical) ordering."""
        row = {**SAMPLE_DISCOVERY_ASSET_ASSET, "series_a": "SPY", "series_b": "AAPL"}
        edges = _build_edge(row, ASSET_LOOKUP, MACRO_LOOKUP)
        corr = edges[0]
        # AAPL < SPY alphabetically
        assert corr["source_id"] == "AAPL"
        assert corr["target_id"] == "SPY"

    def test_no_granger_when_none(self) -> None:
        """Missing granger_p → no CAUSES edge."""
        row = {**SAMPLE_DISCOVERY_ASSET_ASSET, "granger_p": None}
        edges = _build_edge(row, ASSET_LOOKUP, MACRO_LOOKUP)
        assert len(edges) == 1
        assert edges[0]["rel_type"] == "CORRELATED_WITH"

    def test_weak_discovery_still_creates_edge(self) -> None:
        """Weak discoveries still get CORRELATED_WITH edges."""
        edges = _build_edge(SAMPLE_DISCOVERY_WEAK, ASSET_LOOKUP, MACRO_LOOKUP)
        assert len(edges) == 1
        assert edges[0]["strength"] == "weak"


# ── Full materialize flow (mocked) ───────────────────────────────────────────

class TestMaterialize:
    @patch("data.agents.graph_materializer.get_logger")
    def test_materialize_end_to_end(self, mock_logger: MagicMock) -> None:
        """Full materialize flow with mocked Supabase + Neo4j."""
        mock_logger.return_value = MagicMock()

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_query.order.return_value.limit.return_value.execute.return_value.data = [
            SAMPLE_DISCOVERY_ASSET_ASSET,
            SAMPLE_DISCOVERY_MACRO_ASSET,
        ]
        mock_table.select.return_value = mock_query
        # For in_ filter, chain through
        mock_query.in_.return_value = mock_query
        mock_supabase.table.return_value = mock_table

        with patch("data.agents.graph_materializer._build_series_lookup",
                    return_value=(ASSET_LOOKUP, MACRO_LOOKUP)), \
             patch("db.neo4j.client.apply_schema", return_value={"constraints": 6, "indexes": 4}), \
             patch("db.neo4j.client.batch_merge_assets", return_value=2) as mock_ba, \
             patch("db.neo4j.client.batch_merge_macro_indicators", return_value=1) as mock_bm, \
             patch("db.neo4j.client.batch_merge_edges", return_value=3) as mock_be, \
             patch("db.neo4j.client.get_graph_stats", return_value={
                 "nodes": {"Asset": 2, "MacroIndicator": 1},
                 "edges": {"CORRELATED_WITH": 2, "CAUSES": 1},
                 "total_nodes": 3, "total_edges": 3,
             }), \
             patch("db.supabase.client.get_client", return_value=mock_supabase), \
             patch("db.supabase.client.log_evolution"):

            result = materialize(min_strength="moderate")

        assert result["discoveries_fetched"] == 2
        assert result["asset_nodes_merged"] == 2
        assert result["macro_nodes_merged"] == 1
        assert result["edges_merged"] == 3
        # VIXCLS→SPY has Granger p=0.003 < 0.05, so generates CAUSES edge
        assert result["causes_edges"] == 1
        assert result["correlated_with_edges"] == 2

        # Verify batch_merge_assets was called with correct nodes
        asset_call_args = mock_ba.call_args[0][0]
        tickers = {n["ticker"] for n in asset_call_args}
        assert "SPY" in tickers
        assert "AAPL" in tickers

        # Verify batch_merge_macro_indicators was called
        macro_call_args = mock_bm.call_args[0][0]
        series_ids = {n["series_id"] for n in macro_call_args}
        assert "VIXCLS" in series_ids

    @patch("data.agents.graph_materializer.get_logger")
    def test_materialize_empty(self, mock_logger: MagicMock) -> None:
        """Materialize with no discoveries returns zeros."""
        mock_logger.return_value = MagicMock()

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_query.order.return_value.limit.return_value.execute.return_value.data = []
        mock_query.in_.return_value = mock_query
        mock_table.select.return_value = mock_query
        mock_supabase.table.return_value = mock_table

        with patch("data.agents.graph_materializer._build_series_lookup",
                    return_value=(ASSET_LOOKUP, MACRO_LOOKUP)), \
             patch("db.neo4j.client.apply_schema", return_value={"constraints": 6, "indexes": 4}), \
             patch("db.supabase.client.get_client", return_value=mock_supabase):

            result = materialize()

        assert result["discoveries_fetched"] == 0
        assert result["nodes_merged"] == 0
        assert result["edges_merged"] == 0
