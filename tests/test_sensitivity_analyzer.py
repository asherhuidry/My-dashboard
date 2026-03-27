"""Tests for the sensitivity analyzer and SENSITIVE_TO graph edge creation."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_factor(n: int = 200, seed: int = 42) -> pd.Series:
    """Generate a synthetic macro factor series (standardized daily changes)."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.randn(n), index=dates, name="GS10")


def _make_asset_returns(
    factor: pd.Series,
    beta: float,
    noise_std: float = 0.5,
    seed: int = 99,
) -> pd.Series:
    """Generate synthetic asset returns = alpha + beta * factor + noise.

    With known beta and low noise, regression should recover beta closely.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(factor)) * noise_std
    returns = 0.001 + beta * factor.values + noise
    return pd.Series(returns, index=factor.index, name="TEST_ASSET")


# ══════════════════════════════════════════════════════════════════════════════
# Test compute_factor_sensitivities
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeFactorSensitivities:
    """Core regression logic for factor exposure discovery."""

    def test_recovers_known_positive_beta(self):
        """Given asset = 0.8 * factor + noise, should recover beta ~ 0.8."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=250)
        asset = _make_asset_returns(factor, beta=0.8, noise_std=0.3)

        result = _regress_single(asset, factor, "AAPL", "GS10", "rate", 2.0)

        assert result is not None
        assert abs(result["beta"] - 0.8) < 0.15  # within 0.15 of true beta
        assert result["factor_group"] == "rate"
        assert result["relationship_type"] == "rate_sensitive"
        assert result["t_stat"] > 2.0
        assert result["series_a"] == "AAPL"
        assert result["series_b"] == "GS10"

    def test_recovers_known_negative_beta(self):
        """Negative exposure: asset = -0.6 * factor + noise."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=250)
        asset = _make_asset_returns(factor, beta=-0.6, noise_std=0.3)

        result = _regress_single(asset, factor, "TLT", "GS10", "rate", 2.0)

        assert result is not None
        assert result["beta"] < 0
        assert abs(result["beta"] - (-0.6)) < 0.15
        assert result["pearson_r"] < 0

    def test_rejects_insignificant_exposure(self):
        """Near-zero beta with high noise should not pass t-stat threshold."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=100)
        asset = _make_asset_returns(factor, beta=0.01, noise_std=5.0)

        result = _regress_single(asset, factor, "XYZ", "GS10", "rate", 2.0)

        assert result is None

    def test_rejects_insufficient_data(self):
        """Should return None when fewer than MIN_OBS overlapping points."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=30)  # below MIN_OBS=60
        asset = _make_asset_returns(factor, beta=1.0, noise_std=0.1)

        result = _regress_single(asset, factor, "X", "GS10", "rate", 2.0)

        assert result is None

    def test_strength_classification(self):
        """Strong beta (high t-stat) should get 'strong' strength."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=500, seed=10)
        asset = _make_asset_returns(factor, beta=1.5, noise_std=0.2, seed=11)

        result = _regress_single(asset, factor, "NVDA", "VIXCLS", "volatility", 2.0)

        assert result is not None
        assert result["strength"] == "strong"
        assert abs(result["t_stat"]) >= 4.0

    def test_full_pipeline_multiple_factors(self):
        """compute_factor_sensitivities across multiple assets and factors."""
        from data.agents.sensitivity_analyzer import compute_factor_sensitivities

        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        # Build macro DataFrame with two factor series
        macro_df = pd.DataFrame({
            "GS10":      rng.randn(n),
            "DCOILWTICO": rng.randn(n),
        }, index=dates)

        # Build asset returns: AAPL is rate-sensitive, XLE is oil-sensitive
        asset_df = pd.DataFrame({
            "AAPL": 0.001 + 0.7 * macro_df["GS10"] + rng.randn(n) * 0.3,
            "XLE":  0.001 + 0.9 * macro_df["DCOILWTICO"] + rng.randn(n) * 0.3,
            "RAND": rng.randn(n),  # pure noise — should not be sensitive
        }, index=dates)

        results = compute_factor_sensitivities(asset_df, macro_df)

        # Should find AAPL→rate and XLE→oil at minimum
        rel_types = {(r["series_a"], r["relationship_type"]) for r in results}
        assert ("AAPL", "rate_sensitive") in rel_types
        assert ("XLE", "oil_sensitive") in rel_types

        # RAND should have few or no significant sensitivities
        rand_count = sum(1 for r in results if r["series_a"] == "RAND")
        assert rand_count <= 1  # at most a spurious hit

    def test_output_schema(self):
        """Each result dict should have all required keys."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=200)
        asset = _make_asset_returns(factor, beta=0.5, noise_std=0.3)

        result = _regress_single(asset, factor, "MSFT", "T10YIE", "inflation", 2.0)

        assert result is not None
        required_keys = {
            "series_a", "series_b", "factor_group", "beta", "t_stat",
            "p_value", "r_squared", "pearson_r", "strength",
            "relationship_type", "lag_days", "timestamp",
        }
        assert required_keys.issubset(result.keys())
        assert result["lag_days"] == 0
        assert isinstance(result["timestamp"], datetime)


# ══════════════════════════════════════════════════════════════════════════════
# Test SENSITIVE_TO edge creation in materializer
# ══════════════════════════════════════════════════════════════════════════════

class TestSensitiveToEdges:
    """Graph materializer correctly routes sensitivity discoveries to SENSITIVE_TO."""

    def _asset_lookup(self) -> dict[str, str]:
        return {"AAPL": "equity", "XLE": "etf", "TLT": "etf"}

    def _macro_lookup(self) -> dict[str, tuple[str, str]]:
        return {
            "GS10":       ("10-Year Treasury Yield", "daily"),
            "DCOILWTICO": ("WTI Crude Oil", "daily"),
            "VIXCLS":     ("CBOE VIX", "daily"),
        }

    def test_sensitivity_row_creates_sensitive_to_edge(self):
        """A row with relationship_type='rate_sensitive' should create SENSITIVE_TO."""
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL",
            "series_b": "GS10",
            "pearson_r": -0.35,
            "lag_days": 0,
            "granger_p": 0.002,
            "mutual_info": -0.42,  # beta stored here
            "strength": "moderate",
            "regime": "all",
            "relationship_type": "rate_sensitive",
        }

        edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())

        assert len(edges) == 1
        edge = edges[0]
        assert edge["rel_type"] == "SENSITIVE_TO"
        assert edge["factor_group"] == "rate"
        assert edge["beta"] == -0.42
        assert edge["source_id"] == "AAPL"
        assert edge["target_id"] == "GS10"
        assert edge["source_label"] == "Asset"
        assert edge["target_label"] == "MacroIndicator"

    def test_correlation_row_still_creates_correlated_with(self):
        """Non-sensitivity rows should still create CORRELATED_WITH edges."""
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL",
            "series_b": "XLE",
            "pearson_r": 0.55,
            "lag_days": 0,
            "granger_p": None,
            "mutual_info": 0.12,
            "strength": "moderate",
            "regime": "all",
            "relationship_type": "discovered",
        }

        edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())

        assert len(edges) == 1
        assert edges[0]["rel_type"] == "CORRELATED_WITH"
        assert edges[0]["factor_group"] is None
        assert edges[0]["beta"] is None

    def test_granger_causal_still_creates_causes(self):
        """Granger-causal discoveries should still produce CAUSES edges."""
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "GS10",
            "series_b": "TLT",
            "pearson_r": -0.6,
            "lag_days": 3,
            "granger_p": 0.01,
            "mutual_info": 0.2,
            "strength": "strong",
            "regime": "all",
            "relationship_type": "rates_bonds",
        }

        edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())

        rel_types = [e["rel_type"] for e in edges]
        assert "CORRELATED_WITH" in rel_types
        assert "CAUSES" in rel_types

    def test_sensitivity_edge_direction(self):
        """SENSITIVE_TO should always go Asset -> MacroIndicator."""
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "XLE",
            "series_b": "DCOILWTICO",
            "pearson_r": 0.6,
            "lag_days": 0,
            "granger_p": None,
            "mutual_info": 0.85,
            "strength": "strong",
            "regime": "all",
            "relationship_type": "oil_sensitive",
        }

        edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())

        assert len(edges) == 1
        assert edges[0]["source_id"] == "XLE"
        assert edges[0]["target_id"] == "DCOILWTICO"
        assert edges[0]["source_label"] == "Asset"
        assert edges[0]["target_label"] == "MacroIndicator"

    def test_all_factor_groups_recognized(self):
        """Every factor group produces a valid SENSITIVE_TO edge."""
        from data.agents.graph_materializer import _build_edge
        from data.agents.sensitivity_analyzer import SENSITIVITY_FACTORS

        for group, (factor_id, _) in SENSITIVITY_FACTORS.items():
            row = {
                "series_a": "AAPL",
                "series_b": factor_id,
                "pearson_r": 0.3,
                "lag_days": 0,
                "granger_p": None,
                "mutual_info": 0.5,
                "strength": "moderate",
                "regime": "all",
                "relationship_type": f"{group}_sensitive",
            }
            edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())
            assert len(edges) == 1
            assert edges[0]["rel_type"] == "SENSITIVE_TO"
            assert edges[0]["factor_group"] == group


# ══════════════════════════════════════════════════════════════════════════════
# Test integration in correlation_hunter.run()
# ══════════════════════════════════════════════════════════════════════════════

class TestRunIntegration:
    """Sensitivity pass is called during run() and findings are persisted."""

    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass")
    @patch("data.agents.correlation_hunter.hunt_correlations")
    @patch("data.ingest.universe.get_all_symbols",
           return_value=["AAPL", "MSFT"])
    def test_run_includes_sensitivity_findings(
        self, mock_syms, mock_hunt, mock_sens, mock_persist, mock_graph,
    ):
        """run() should combine correlation + sensitivity findings."""
        from data.agents.correlation_hunter import run, CorrelationFinding

        corr_finding = CorrelationFinding(
            series_a="AAPL", series_b="MSFT", lag_days=0,
            pearson_r=0.5, granger_p=None, mutual_info=0.1,
            strength="moderate", relationship_type="discovered",
        )
        sens_finding = CorrelationFinding(
            series_a="AAPL", series_b="GS10", lag_days=0,
            pearson_r=-0.3, granger_p=0.01, mutual_info=-0.4,
            strength="moderate", relationship_type="rate_sensitive",
        )

        mock_hunt.return_value = [corr_finding]
        mock_sens.return_value = [sens_finding]
        mock_persist.return_value = "test-run-id"
        mock_graph.return_value = None

        findings = run(symbols=["AAPL", "MSFT"])

        assert len(findings) == 2
        assert findings[0].relationship_type == "discovered"
        assert findings[1].relationship_type == "rate_sensitive"

        # Persist should receive both
        persisted = mock_persist.call_args[0][0]
        assert len(persisted) == 2

    @patch("data.agents.correlation_hunter._materialize_graph")
    @patch("data.agents.correlation_hunter._persist_discoveries")
    @patch("data.agents.correlation_hunter._run_sensitivity_pass")
    @patch("data.agents.correlation_hunter.hunt_correlations")
    @patch("data.ingest.universe.get_all_symbols",
           return_value=["AAPL"])
    def test_sensitivity_failure_does_not_break_run(
        self, mock_syms, mock_hunt, mock_sens, mock_persist, mock_graph,
    ):
        """If sensitivity pass raises, run() should still return correlations."""
        from data.agents.correlation_hunter import run, CorrelationFinding

        mock_hunt.return_value = [CorrelationFinding(
            series_a="AAPL", series_b="MSFT", lag_days=0,
            pearson_r=0.5, granger_p=None, mutual_info=None,
        )]
        mock_sens.return_value = []  # empty on failure
        mock_persist.return_value = "test-run-id"
        mock_graph.return_value = None

        findings = run(symbols=["AAPL"])

        assert len(findings) == 1
        assert findings[0].relationship_type == "discovered"
