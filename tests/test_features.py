"""Tests for ml/patterns/features.py — feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.patterns.features import (
    build_features,
    calendar_features,
    cross_asset_features,
    feature_names,
    macro_lag_features,
    momentum_features,
    price_return_features,
    regime_features,
    trend_features,
    volatility_features,
    volume_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = rng.uniform(0.005, 0.02, n)
    high   = close * (1 + spread)
    low    = close * (1 - spread)
    open_  = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_macro(n: int = 300) -> pd.DataFrame:
    """Return a synthetic macro DataFrame (monthly-ish updates)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    gdp  = pd.Series(2.0 + rng.normal(0, 0.2, n), index=dates)
    cpi  = pd.Series(3.0 + rng.normal(0, 0.1, n), index=dates)
    gs2  = pd.Series(2.5 + rng.normal(0, 0.1, n), index=dates)
    gs10 = pd.Series(3.5 + rng.normal(0, 0.1, n), index=dates)
    return pd.DataFrame({"GDP": gdp, "CPIAUCSL": cpi, "GS2": gs2, "GS10": gs10})


# ─────────────────────────────────────────────────────────────────────────────
# price_return_features
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceReturnFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = price_return_features(df)
        expected = {"ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
                    "log_ret_1d", "log_ret_5d", "cum_ret_21d", "cum_ret_63d"}
        assert expected.issubset(set(out.columns))

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = price_return_features(df)
        pd.testing.assert_index_equal(out.index, df.index)

    def test_ret_1d_is_pct_change(self) -> None:
        df = _make_ohlcv()
        out = price_return_features(df)
        expected = df["close"].pct_change()
        pd.testing.assert_series_equal(out["ret_1d"], expected, check_names=False)

    def test_log_ret_1d_sign_matches_ret_1d(self) -> None:
        """log return and simple return should have the same sign."""
        df = _make_ohlcv()
        out = price_return_features(df)
        valid = out[["ret_1d", "log_ret_1d"]].dropna()
        signs_match = (np.sign(valid["ret_1d"]) == np.sign(valid["log_ret_1d"])).all()
        assert signs_match

    def test_cum_ret_21d_nan_at_start(self) -> None:
        df = _make_ohlcv()
        out = price_return_features(df)
        # Cumulative return needs 21 daily returns — first 21 rows should be NaN
        assert out["cum_ret_21d"].iloc[:21].isna().all()

    def test_has_nine_columns(self) -> None:
        df = _make_ohlcv()
        out = price_return_features(df)
        assert len(out.columns) == 9


# ─────────────────────────────────────────────────────────────────────────────
# volatility_features
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = volatility_features(df)
        expected = {"realized_vol_5d", "realized_vol_21d",
                    "garman_klass_vol", "atr_14", "atr_ratio"}
        assert expected.issubset(set(out.columns))

    def test_realized_vol_non_negative(self) -> None:
        df = _make_ohlcv()
        out = volatility_features(df)
        valid = out["realized_vol_5d"].dropna()
        assert (valid >= 0).all()

    def test_atr_ratio_is_atr_over_close(self) -> None:
        df = _make_ohlcv()
        out = volatility_features(df)
        valid = out[["atr_14", "atr_ratio"]].dropna()
        # atr_ratio = atr_14 / close
        ratio = (valid["atr_14"] / df["close"].loc[valid.index]).round(8)
        pd.testing.assert_series_equal(
            ratio, valid["atr_ratio"].round(8), check_names=False
        )

    def test_garman_klass_non_negative(self) -> None:
        df = _make_ohlcv()
        out = volatility_features(df)
        valid = out["garman_klass_vol"].dropna()
        assert (valid >= 0).all()

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = volatility_features(df)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# momentum_features
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        expected = {"rsi_14", "rsi_28", "macd", "macd_signal", "macd_hist",
                    "stoch_k", "stoch_d", "roc_10", "roc_21", "williams_r_14"}
        assert expected.issubset(set(out.columns))

    def test_rsi_bounded(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        valid = out["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_stoch_k_bounded(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        valid = out["stoch_k"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_hist_equals_macd_minus_signal(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        diff = (out["macd"] - out["macd_signal"] - out["macd_hist"]).dropna()
        assert (diff.abs() < 1e-10).all()

    def test_williams_r_bounded(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        valid = out["williams_r_14"].dropna()
        assert (valid <= 0).all() and (valid >= -100).all()

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = momentum_features(df)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# trend_features
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = trend_features(df)
        expected = {"bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct",
                    "adx_14", "ema_9", "ema_21", "ema_50", "ema_200",
                    "ema_9_21_cross", "ema_21_50_cross",
                    "ichimoku_conv", "ichimoku_base",
                    "ichimoku_span_a", "ichimoku_span_b"}
        assert expected.issubset(set(out.columns))

    def test_bb_upper_above_lower(self) -> None:
        df = _make_ohlcv()
        out = trend_features(df)
        valid = out[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_ema_cross_is_binary(self) -> None:
        df = _make_ohlcv()
        out = trend_features(df)
        vals = out["ema_9_21_cross"].dropna().unique()
        assert set(vals).issubset({0, 1})

    def test_adx_non_negative(self) -> None:
        df = _make_ohlcv()
        out = trend_features(df)
        valid = out["adx_14"].dropna()
        assert (valid >= 0).all()

    def test_ema_200_smoother_than_ema_9(self) -> None:
        """EMA-200 should have lower variance than EMA-9."""
        df = _make_ohlcv()
        out = trend_features(df)
        valid = out[["ema_9", "ema_200"]].dropna()
        assert valid["ema_200"].std() < valid["ema_9"].std()

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = trend_features(df)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# volume_features
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = volume_features(df)
        expected = {"obv", "obv_ema_21", "vwap_21", "volume_ratio_10", "cmf_20"}
        assert expected == set(out.columns)

    def test_obv_monotone_on_trending_asset(self) -> None:
        """OBV should increase on a steadily rising asset."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = pd.Series(np.linspace(100, 200, n), index=dates)
        df = pd.DataFrame({
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": np.ones(n) * 1_000_000,
        })
        out = volume_features(df)
        assert out["obv"].iloc[-1] > out["obv"].iloc[0]

    def test_volume_ratio_near_one_on_flat_volume(self) -> None:
        df = _make_ohlcv()
        # Replace volume with constant
        df["volume"] = 1_000_000.0
        out = volume_features(df)
        valid = out["volume_ratio_10"].dropna()
        assert (valid.round(4) == 1.0).all()

    def test_cmf_bounded(self) -> None:
        df = _make_ohlcv()
        out = volume_features(df)
        valid = out["cmf_20"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = volume_features(df)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# regime_features
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeFeatures:
    def test_base_columns_no_benchmark(self) -> None:
        df = _make_ohlcv()
        out = regime_features(df)
        expected = {"rolling_sharpe_21", "rolling_sharpe_63",
                    "max_drawdown_63", "rolling_vol_21"}
        assert expected == set(out.columns)

    def test_benchmark_columns_present(self) -> None:
        df = _make_ohlcv()
        bench = _make_ohlcv(seed=99)["close"]
        out = regime_features(df, benchmark=bench)
        assert "beta_21" in out.columns
        assert "corr_benchmark_21" in out.columns
        assert "corr_benchmark_63" in out.columns

    def test_max_drawdown_non_positive(self) -> None:
        df = _make_ohlcv()
        out = regime_features(df)
        valid = out["max_drawdown_63"].dropna()
        assert (valid <= 0).all()

    def test_correlation_bounded(self) -> None:
        df = _make_ohlcv()
        bench = _make_ohlcv(seed=77)["close"]
        out = regime_features(df, benchmark=bench)
        valid = out["corr_benchmark_21"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_rolling_vol_non_negative(self) -> None:
        df = _make_ohlcv()
        out = regime_features(df)
        valid = out["rolling_vol_21"].dropna()
        assert (valid >= 0).all()

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = regime_features(df)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# macro_lag_features
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroLagFeatures:
    def test_lag_columns_created(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()
        out = macro_lag_features(df, macro)
        for col in ["GDP", "CPIAUCSL"]:
            for lag in [1, 5, 21]:
                assert f"{col}_lag{lag}" in out.columns

    def test_yield_curve_slope_present(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()  # includes GS2 and GS10
        out = macro_lag_features(df, macro)
        assert "yield_curve_slope" in out.columns

    def test_no_yield_curve_without_gs_columns(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()[["GDP", "CPIAUCSL"]]
        out = macro_lag_features(df, macro)
        assert "yield_curve_slope" not in out.columns

    def test_lag1_is_previous_day(self) -> None:
        df = _make_ohlcv(n=50)
        dates = df.index
        # Constant macro series so shift is easy to verify
        macro = pd.DataFrame({"INDICATOR": np.ones(50)}, index=dates)
        macro.iloc[10] = 2.0   # one spike at row 10
        out = macro_lag_features(df, macro)
        # lag1 at row 11 should be 2.0
        assert out["INDICATOR_lag1"].iloc[11] == 2.0

    def test_index_matches_price_df(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()
        out = macro_lag_features(df, macro)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# cross_asset_features
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossAssetFeatures:
    def test_columns_created_for_each_peer(self) -> None:
        df = _make_ohlcv()
        peers = {
            "BTC":  _make_ohlcv(seed=1)["close"],
            "GOLD": _make_ohlcv(seed=2)["close"],
        }
        out = cross_asset_features(df, peers)
        for label in ["BTC", "GOLD"]:
            assert f"corr_{label}_21" in out.columns
            assert f"corr_{label}_63" in out.columns

    def test_correlation_bounded(self) -> None:
        df = _make_ohlcv()
        peers = {"SPY": _make_ohlcv(seed=5)["close"]}
        out = cross_asset_features(df, peers)
        valid = out["corr_SPY_21"].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_perfect_correlation_with_self(self) -> None:
        df = _make_ohlcv()
        peers = {"SELF": df["close"]}
        out = cross_asset_features(df, peers)
        valid = out["corr_SELF_21"].dropna()
        assert (valid.round(10) == 1.0).all()

    def test_empty_peers_returns_empty_df(self) -> None:
        df = _make_ohlcv()
        out = cross_asset_features(df, {})
        assert out.empty or len(out.columns) == 0

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        peers = {"BTC": _make_ohlcv(seed=3)["close"]}
        out = cross_asset_features(df, peers)
        pd.testing.assert_index_equal(out.index, df.index)


# ─────────────────────────────────────────────────────────────────────────────
# calendar_features
# ─────────────────────────────────────────────────────────────────────────────

class TestCalendarFeatures:
    def test_returns_expected_columns(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        expected = {"day_of_week", "month", "quarter",
                    "is_month_end", "is_quarter_end",
                    "fomc_window", "earnings_season"}
        assert expected == set(out.columns)

    def test_day_of_week_range(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        assert out["day_of_week"].between(0, 4).all()  # business days

    def test_month_range(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        assert out["month"].between(1, 12).all()

    def test_quarter_range(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        assert out["quarter"].between(1, 4).all()

    def test_binary_flags(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        for col in ["is_month_end", "is_quarter_end", "fomc_window", "earnings_season"]:
            assert set(out[col].unique()).issubset({0, 1})

    def test_fomc_window_in_expected_months(self) -> None:
        df = _make_ohlcv()
        out = calendar_features(df)
        fomc_months = out.loc[out["fomc_window"] == 1, "month"].unique()
        assert set(fomc_months).issubset({1, 3, 5, 6, 7, 9, 10, 11, 12})


# ─────────────────────────────────────────────────────────────────────────────
# build_features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_minimal_call_returns_dataframe(self) -> None:
        df = _make_ohlcv()
        out = build_features(df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(df)

    def test_all_base_column_groups_present(self) -> None:
        df = _make_ohlcv()
        out = build_features(df)
        assert "ret_1d" in out.columns           # price returns
        assert "realized_vol_5d" in out.columns  # volatility
        assert "rsi_14" in out.columns            # momentum
        assert "bb_upper" in out.columns          # trend
        assert "obv" in out.columns               # volume
        assert "rolling_sharpe_21" in out.columns # regime
        assert "day_of_week" in out.columns       # calendar

    def test_with_macro_adds_lag_columns(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()
        out = build_features(df, macro=macro)
        assert "GDP_lag1" in out.columns

    def test_with_benchmark_adds_regime_columns(self) -> None:
        df = _make_ohlcv()
        bench = _make_ohlcv(seed=20)["close"]
        out = build_features(df, benchmark=bench)
        assert "beta_21" in out.columns

    def test_with_peers_adds_cross_asset_columns(self) -> None:
        df = _make_ohlcv()
        peers = {"BTC": _make_ohlcv(seed=30)["close"]}
        out = build_features(df, peers=peers)
        assert "corr_BTC_21" in out.columns

    def test_no_duplicate_columns(self) -> None:
        df = _make_ohlcv()
        macro = _make_macro()
        bench = _make_ohlcv(seed=40)["close"]
        peers = {"SPY": _make_ohlcv(seed=50)["close"]}
        out = build_features(df, macro=macro, benchmark=bench, peers=peers)
        assert len(out.columns) == len(set(out.columns))

    def test_minimum_70_features(self) -> None:
        """Pipeline should produce at least 70 features with macro + benchmark + peers."""
        df = _make_ohlcv()
        macro = _make_macro()
        bench = _make_ohlcv(seed=60)["close"]
        peers = {"SPY": _make_ohlcv(seed=70)["close"],
                 "BTC": _make_ohlcv(seed=71)["close"],
                 "GOLD": _make_ohlcv(seed=72)["close"]}
        out = build_features(df, macro=macro, benchmark=bench, peers=peers)
        assert len(out.columns) >= 70

    def test_index_preserved(self) -> None:
        df = _make_ohlcv()
        out = build_features(df)
        pd.testing.assert_index_equal(out.index, df.index)

    def test_empty_macro_skipped(self) -> None:
        df = _make_ohlcv()
        out = build_features(df, macro=pd.DataFrame())
        assert "GDP_lag1" not in out.columns

    def test_none_peers_skipped(self) -> None:
        df = _make_ohlcv()
        out = build_features(df, peers=None)
        assert not any(c.startswith("corr_") for c in out.columns)


# ─────────────────────────────────────────────────────────────────────────────
# feature_names
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureNames:
    def test_base_names_no_extras(self) -> None:
        names = feature_names()
        assert "ret_1d" in names
        assert "rsi_14" in names
        assert "bb_upper" in names
        assert "day_of_week" in names

    def test_macro_cols_expand_list(self) -> None:
        names = feature_names(macro_cols=["GDP", "CPI"])
        assert "GDP_lag1" in names
        assert "GDP_lag5" in names
        assert "GDP_lag21" in names
        assert "CPI_lag1" in names

    def test_yield_curve_added_when_gs_present(self) -> None:
        names = feature_names(macro_cols=["GS2", "GS10"])
        assert "yield_curve_slope" in names

    def test_no_yield_curve_without_gs(self) -> None:
        names = feature_names(macro_cols=["GDP"])
        assert "yield_curve_slope" not in names

    def test_peer_labels_expand_list(self) -> None:
        names = feature_names(peer_labels=["BTC", "GOLD"])
        assert "corr_BTC_21" in names
        assert "corr_BTC_63" in names
        assert "corr_GOLD_21" in names

    def test_no_duplicates(self) -> None:
        names = feature_names(
            macro_cols=["GDP", "GS2", "GS10"],
            peer_labels=["BTC", "SPY"],
        )
        assert len(names) == len(set(names))

    def test_returns_list(self) -> None:
        assert isinstance(feature_names(), list)

    def test_names_match_build_features_base(self) -> None:
        """Base feature_names() should be a subset of build_features() columns."""
        df = _make_ohlcv()
        bench = _make_ohlcv(seed=99)["close"]
        out = build_features(df, benchmark=bench)
        expected = set(feature_names())
        actual = set(out.columns)
        missing = expected - actual
        assert not missing, f"feature_names() lists columns not in build_features(): {missing}"
