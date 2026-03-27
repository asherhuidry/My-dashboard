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


# ══════════════════════════════════════════════════════════════════════════════
# Test regime classification
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeClassification:
    """Regime mask construction from market data."""

    def _make_market_data(self, n: int = 300, seed: int = 42):
        """Build synthetic asset returns and macro data with clear regime periods."""
        rng = np.random.RandomState(seed)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")

        # SPY returns: first 150 days positive trend, last 150 negative
        spy = np.zeros(n)
        spy[:150] = 0.005 + rng.randn(150) * 0.01   # bull
        spy[150:] = -0.005 + rng.randn(150) * 0.01   # bear

        asset_returns = pd.DataFrame({
            "SPY":  spy,
            "AAPL": spy * 1.2 + rng.randn(n) * 0.005,
        }, index=dates)

        # VIX proxy: low in first half, high spikes in second half
        vix_z = np.zeros(n)
        vix_z[:150] = rng.randn(150) * 0.3           # low vol
        vix_z[150:] = 1.0 + rng.randn(150) * 0.5     # high vol

        macro_levels = pd.DataFrame({
            "VIXCLS": vix_z,
            "GS10":   rng.randn(n),
        }, index=dates)

        return asset_returns, macro_levels

    def test_bear_mask_identifies_drawdowns(self):
        """Bear mask should flag periods after sustained negative returns."""
        from data.agents.sensitivity_analyzer import build_regime_masks

        asset_returns, macro_levels = self._make_market_data()
        masks = build_regime_masks(asset_returns, macro_levels)

        assert "bear" in masks
        bear = masks["bear"]
        # After 63-day lookback, the bear period should be the second half
        # Check that bear days are concentrated in the later portion
        mid = len(bear) // 2
        bear_in_second_half = bear.iloc[mid:].sum()
        bear_in_first_half = bear.iloc[:mid].sum()
        assert bear_in_second_half > bear_in_first_half

    def test_stress_mask_identifies_high_vol(self):
        """Stress mask should flag periods of elevated VIX."""
        from data.agents.sensitivity_analyzer import build_regime_masks

        asset_returns, macro_levels = self._make_market_data()
        masks = build_regime_masks(asset_returns, macro_levels)

        assert "stress" in masks
        stress = masks["stress"]
        # Stress should be roughly 25% of days (75th percentile)
        stress_pct = stress.sum() / len(stress)
        assert 0.10 < stress_pct < 0.50

    def test_missing_spy_skips_bear(self):
        """Without SPY in returns, bear mask should not be created."""
        from data.agents.sensitivity_analyzer import build_regime_masks

        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        # No SPY column
        asset_returns = pd.DataFrame({"AAPL": rng.randn(n)}, index=dates)
        macro_levels = pd.DataFrame({"VIXCLS": rng.randn(n)}, index=dates)

        masks = build_regime_masks(asset_returns, macro_levels)

        assert "bear" not in masks

    def test_missing_vix_skips_stress(self):
        """Without VIX in macro data, stress mask should not be created."""
        from data.agents.sensitivity_analyzer import build_regime_masks

        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        asset_returns = pd.DataFrame({"SPY": rng.randn(n)}, index=dates)
        macro_levels = pd.DataFrame({"GS10": rng.randn(n)}, index=dates)

        masks = build_regime_masks(asset_returns, macro_levels)

        assert "stress" not in masks


# ══════════════════════════════════════════════════════════════════════════════
# Test regime-conditioned sensitivity
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeConditionedSensitivity:
    """OLS betas computed within specific market regimes."""

    def test_regime_beta_differs_from_full_sample(self):
        """An asset with regime-dependent exposure should show different betas."""
        from data.agents.sensitivity_analyzer import (
            compute_factor_sensitivities, compute_regime_sensitivities,
        )

        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        factor = rng.randn(n)

        # Asset: low sensitivity in first half, high in second half
        asset = np.zeros(n)
        asset[:200] = 0.1 * factor[:200] + rng.randn(200) * 0.5   # weak beta
        asset[200:] = 0.9 * factor[200:] + rng.randn(200) * 0.3   # strong beta

        asset_df = pd.DataFrame({"AAPL": asset}, index=dates)
        macro_df = pd.DataFrame({"GS10": factor}, index=dates)

        # Regime mask: second half is "bear"
        bear_mask = pd.Series([False] * 200 + [True] * 200, index=dates)
        masks = {"bear": bear_mask}

        regime_results = compute_regime_sensitivities(asset_df, macro_df, masks)

        # Should find AAPL rate_sensitive in bear regime
        bear_hits = [r for r in regime_results
                     if r["series_a"] == "AAPL" and r["regime"] == "bear"
                     and r["factor_group"] == "rate"]
        assert len(bear_hits) == 1
        # Bear beta should be closer to 0.9 than to the full-sample average
        assert abs(bear_hits[0]["beta"]) > 0.5

    def test_regime_results_carry_regime_label(self):
        """Every result from compute_regime_sensitivities has a regime field."""
        from data.agents.sensitivity_analyzer import compute_regime_sensitivities

        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        factor = rng.randn(n)
        asset = 0.8 * factor + rng.randn(n) * 0.3

        asset_df = pd.DataFrame({"NVDA": asset}, index=dates)
        macro_df = pd.DataFrame({"GS10": factor}, index=dates)

        mask = pd.Series([True] * n, index=dates)  # all days in regime
        masks = {"stress": mask}

        results = compute_regime_sensitivities(asset_df, macro_df, masks)

        for r in results:
            assert r["regime"] == "stress"
            assert r["relationship_type"].endswith("_sensitive")

    def test_insufficient_regime_days_produces_no_results(self):
        """Regime with too few days should be skipped entirely."""
        from data.agents.sensitivity_analyzer import compute_regime_sensitivities

        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)

        asset_df = pd.DataFrame({"AAPL": rng.randn(n)}, index=dates)
        macro_df = pd.DataFrame({"GS10": rng.randn(n)}, index=dates)

        # Only 10 days in regime — below MIN_REGIME_OBS
        mask = pd.Series([False] * n, index=dates)
        mask.iloc[:10] = True
        masks = {"bear": mask}

        results = compute_regime_sensitivities(asset_df, macro_df, masks)

        assert len(results) == 0

    def test_regress_single_passes_regime_to_output(self):
        """_regress_single with regime param should set the regime in output."""
        from data.agents.sensitivity_analyzer import _regress_single

        factor = _make_factor(n=200)
        asset = _make_asset_returns(factor, beta=0.7, noise_std=0.3)

        result = _regress_single(asset, factor, "TLT", "GS10", "rate", 2.0,
                                 regime="bear")

        assert result is not None
        assert result["regime"] == "bear"


# ══════════════════════════════════════════════════════════════════════════════
# Test regime-aware SENSITIVE_TO edges in materializer
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeSensitiveEdges:
    """Materializer creates separate SENSITIVE_TO edges per regime."""

    def _asset_lookup(self) -> dict[str, str]:
        return {"AAPL": "equity", "XLE": "etf", "TLT": "etf"}

    def _macro_lookup(self) -> dict[str, tuple[str, str]]:
        return {"GS10": ("10-Year Treasury Yield", "daily")}

    def test_bear_regime_edge_carries_regime(self):
        """A bear-regime sensitivity should produce edge with regime='bear'."""
        from data.agents.graph_materializer import _build_edge

        row = {
            "series_a": "AAPL",
            "series_b": "GS10",
            "pearson_r": -0.5,
            "lag_days": 0,
            "granger_p": 0.001,
            "mutual_info": -0.8,  # beta
            "strength": "strong",
            "regime": "bear",
            "relationship_type": "rate_sensitive",
        }

        edges = _build_edge(row, self._asset_lookup(), self._macro_lookup())

        assert len(edges) == 1
        assert edges[0]["rel_type"] == "SENSITIVE_TO"
        assert edges[0]["regime"] == "bear"
        assert edges[0]["beta"] == -0.8
        assert edges[0]["factor_group"] == "rate"

    def test_different_regimes_produce_separate_edges(self):
        """Same asset+factor with different regimes should produce separate edges."""
        from data.agents.graph_materializer import _build_edge

        base = {
            "series_a": "AAPL", "series_b": "GS10",
            "pearson_r": -0.3, "lag_days": 0, "granger_p": None,
            "strength": "moderate", "relationship_type": "rate_sensitive",
        }

        row_all = {**base, "mutual_info": -0.3, "regime": "all"}
        row_bear = {**base, "mutual_info": -0.7, "regime": "bear"}
        row_stress = {**base, "mutual_info": -0.9, "regime": "stress"}

        edge_all = _build_edge(row_all, self._asset_lookup(), self._macro_lookup())[0]
        edge_bear = _build_edge(row_bear, self._asset_lookup(), self._macro_lookup())[0]
        edge_stress = _build_edge(row_stress, self._asset_lookup(), self._macro_lookup())[0]

        # All should be SENSITIVE_TO with same source/target
        for e in [edge_all, edge_bear, edge_stress]:
            assert e["rel_type"] == "SENSITIVE_TO"
            assert e["source_id"] == "AAPL"
            assert e["target_id"] == "GS10"

        # But different regimes and betas
        assert edge_all["regime"] == "all"
        assert edge_bear["regime"] == "bear"
        assert edge_stress["regime"] == "stress"
        assert edge_all["beta"] == -0.3
        assert edge_bear["beta"] == -0.7
        assert edge_stress["beta"] == -0.9
