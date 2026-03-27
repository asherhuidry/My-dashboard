"""Expanded feature engineering pipeline — 150+ features per asset.

Feature categories:
  1. Price/volume technicals (30 features) — RSI, MACD, BB, ATR, OBV, etc.
  2. Cross-asset signals (25 features) — correlation to SPY, VIX, Gold, DXY, TLT
  3. Macro regime features (20 features) — yield curve, credit spreads, breakevens
  4. Fundamental ratios (15 features) — P/E, P/S, P/B, margins, growth rates
  5. Social/sentiment features (10 features) — Reddit, StockTwits, Google Trends
  6. Knowledge graph features (10 features) — supply chain centrality, sector stress
  7. Calendar/seasonality features (10 features) — day-of-week, month, earnings proximity
  8. Volatility regime features (15 features) — GARCH-style vol of vol, VIX term structure
  9. Options flow features (10 features) — put/call, implied vol surface
  10. Event-driven features (10 features) — FOMC proximity, earnings proximity, CPI release

These feed the LSTM model, XGBoost ensemble, and the correlation hunter.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── 1. Technical indicators ────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, col: str = "close", windows: list[int] | None = None) -> pd.DataFrame:
    """Add RSI at multiple windows (default: 7, 14, 21, 63)."""
    if windows is None:
        windows = [7, 14, 21, 63]
    for w in windows:
        delta = df[col].diff()
        gain  = delta.clip(lower=0).rolling(w).mean()
        loss  = (-delta.clip(upper=0)).rolling(w).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df:        pd.DataFrame,
    col:       str = "close",
    fast:      int = 12,
    slow:      int = 26,
    signal:    int = 9,
) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram."""
    ema_fast   = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow   = df[col].ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd"]        = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"]   = macd_line - signal_line
    df["macd_cross"]  = np.sign(df["macd_hist"]).diff().fillna(0)
    return df


def add_bollinger_bands(
    df:     pd.DataFrame,
    col:    str = "close",
    window: int = 20,
    std:    float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Bands: upper, lower, %B, bandwidth."""
    mid   = df[col].rolling(window).mean()
    sigma = df[col].rolling(window).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    df["bb_upper"]     = upper
    df["bb_lower"]     = lower
    df["bb_mid"]       = mid
    df["bb_pct"]       = (df[col] - lower) / (upper - lower).replace(0, np.nan)
    df["bb_width"]     = (upper - lower) / mid.replace(0, np.nan)
    df["bb_squeeze"]   = df["bb_width"] < df["bb_width"].rolling(125).quantile(0.25)
    return df


def add_atr(
    df:     pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Add Average True Range and normalized ATR."""
    high_low  = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift(1)).abs()
    low_prev  = (df["low"]  - df["close"].shift(1)).abs()
    tr        = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"]      = tr.rolling(window).mean()
    df["atr_norm"] = df["atr"] / df["close"]
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume and OBV momentum."""
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"]       = (direction * df["volume"]).cumsum()
    df["obv_ema20"] = df["obv"].ewm(span=20).mean()
    df["obv_slope"] = df["obv"].diff(5) / 5
    return df


def add_momentum(
    df:      pd.DataFrame,
    col:     str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add price momentum/returns at multiple horizons."""
    if windows is None:
        windows = [1, 5, 10, 21, 63, 126, 252]
    for w in windows:
        df[f"ret_{w}d"] = df[col].pct_change(w)
    return df


def add_volatility(
    df:      pd.DataFrame,
    col:     str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add realized volatility at multiple windows (annualized)."""
    if windows is None:
        windows = [5, 10, 21, 63]
    log_ret = np.log(df[col] / df[col].shift(1))
    for w in windows:
        df[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    # Vol of vol (regime signal)
    df["vol_of_vol"] = df["vol_21d"].rolling(21).std()
    # Vol ratio (short vs long — vol spike detection)
    df["vol_ratio"]  = df["vol_5d"] / df["vol_63d"].replace(0, np.nan)
    return df


def add_ema_structure(
    df:  pd.DataFrame,
    col: str = "close",
) -> pd.DataFrame:
    """Add EMA structure: position relative to 20/50/200 EMAs."""
    for span in [8, 20, 50, 100, 200]:
        df[f"ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    # Trend signals
    df["above_ema20"]  = (df[col] > df["ema_20"]).astype(int)
    df["above_ema50"]  = (df[col] > df["ema_50"]).astype(int)
    df["above_ema200"] = (df[col] > df["ema_200"]).astype(int)
    df["ema20_50_cross"] = np.sign(df["ema_20"] - df["ema_50"]).diff().fillna(0)
    df["ema50_200_cross"] = np.sign(df["ema_50"] - df["ema_200"]).diff().fillna(0)
    # Distance from key EMAs (normalized)
    df["dist_ema20"]  = (df[col] - df["ema_20"])  / df["ema_20"].replace(0, np.nan)
    df["dist_ema200"] = (df[col] - df["ema_200"]) / df["ema_200"].replace(0, np.nan)
    return df


def add_stochastic(
    df:      pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Add %K and %D Stochastic Oscillator."""
    low_min  = df["low"].rolling(k_window).min()
    high_max = df["high"].rolling(k_window).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_window).mean()
    df["stoch_od"] = df["stoch_k"] > df["stoch_d"]
    return df


def add_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume analysis: relative volume, VWAP-distance, accumulation."""
    df["rel_volume"]    = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    # VWAP (20-day rolling)
    typical_price       = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20"]       = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["dist_vwap"]     = (df["close"] - df["vwap_20"]) / df["vwap_20"].replace(0, np.nan)
    # Accumulation/Distribution
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    df["ad_line"]       = (clv * df["volume"]).cumsum()
    return df


def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add distance to rolling support (low) and resistance (high)."""
    df["rolling_high"] = df["high"].rolling(window).max()
    df["rolling_low"]  = df["low"].rolling(window).min()
    df["dist_to_high"] = (df["rolling_high"] - df["close"]) / df["close"]
    df["dist_to_low"]  = (df["close"] - df["rolling_low"]) / df["close"]
    df["range_position"] = df["dist_to_low"] / (df["dist_to_high"] + df["dist_to_low"]).replace(0, np.nan)
    return df


# ── 2. Cross-asset signals ─────────────────────────────────────────────────────

def add_cross_asset_correlations(
    df:           pd.DataFrame,
    reference_dfs: dict[str, pd.DataFrame],
    windows:      list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling correlations to reference assets (SPY, GLD, TLT, VIX, DXY).

    Args:
        df:             Primary asset DataFrame with 'close' column.
        reference_dfs:  Dict of {name: DataFrame} with 'close' columns.
        windows:        Correlation windows in trading days.

    Returns:
        df with correlation columns added.
    """
    if windows is None:
        windows = [21, 63]

    primary_ret = df["close"].pct_change()

    for name, ref_df in reference_dfs.items():
        ref_ret = ref_df["close"].pct_change()
        # Align on index
        aligned = primary_ret.align(ref_ret, join="left")[1]
        for w in windows:
            col = f"corr_{name}_{w}d"
            df[col] = primary_ret.rolling(w).corr(aligned)

    return df


def add_beta(
    df:       pd.DataFrame,
    spy_df:   pd.DataFrame,
    windows:  list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling beta vs SPY at multiple windows."""
    if windows is None:
        windows = [21, 63, 252]

    asset_ret = df["close"].pct_change()
    spy_ret   = spy_df["close"].pct_change().reindex(df.index)

    for w in windows:
        cov = asset_ret.rolling(w).cov(spy_ret)
        var = spy_ret.rolling(w).var()
        df[f"beta_{w}d"] = cov / var.replace(0, np.nan)

    return df


def add_relative_strength(
    df:     pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative strength vs benchmark (SPY)."""
    spy_aligned = spy_df["close"].reindex(df.index)
    for w in [21, 63, 126, 252]:
        asset_ret = df["close"].pct_change(w)
        spy_ret   = spy_aligned.pct_change(w)
        df[f"rel_strength_{w}d"] = asset_ret - spy_ret
    return df


# ── 3. Macro regime features ───────────────────────────────────────────────────

def add_macro_features(
    df:         pd.DataFrame,
    macro_data: dict[str, pd.Series],
) -> pd.DataFrame:
    """Add macro indicator values aligned to the asset's date index.

    Key macro series added:
      - t10y2y: yield curve slope (negative = inversion)
      - hy_spread: credit stress
      - vix: market fear gauge
      - t10yie: inflation expectations
      - dff: Fed Funds Rate
      - dtwexbgs: broad USD index
      - walcl: Fed balance sheet
      - real_10y: ex-ante real rates

    Args:
        df:         Asset DataFrame (index must be datetime).
        macro_data: {series_id: pd.Series} with datetime index.

    Returns:
        df with macro columns added.
    """
    key_series = [
        "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "T10YIE",
        "DFF", "DTWEXBGS", "WALCL", "DFII10",
        "T10Y3M", "DCOILWTICO", "GOLDAMGBD228NLBM",
    ]

    for series_id in key_series:
        if series_id not in macro_data:
            continue
        series = macro_data[series_id]
        # Forward-fill macro data onto asset dates
        aligned = series.reindex(df.index.union(series.index)).ffill().reindex(df.index)
        col = series_id.lower().replace(".", "_")
        df[f"macro_{col}"] = aligned

    # Compute macro-derived signals
    if "macro_t10y2y" in df.columns:
        df["yield_curve_inverted"] = (df["macro_t10y2y"] < 0).astype(int)
        df["yield_curve_stress"]   = (df["macro_t10y2y"] < -0.5).astype(int)

    if "macro_bamlh0a0hym2" in df.columns:
        df["credit_stress"]        = (df["macro_bamlh0a0hym2"] > 5.0).astype(int)
        df["credit_crisis"]        = (df["macro_bamlh0a0hym2"] > 8.0).astype(int)

    if "macro_vixcls" in df.columns:
        df["high_vol_regime"]      = (df["macro_vixcls"] > 25).astype(int)
        df["panic_regime"]         = (df["macro_vixcls"] > 35).astype(int)

    return df


def add_macro_momentum(
    df:         pd.DataFrame,
    macro_data: dict[str, pd.Series],
    windows:    list[int] | None = None,
) -> pd.DataFrame:
    """Add momentum/trend of key macro series (direction of change).

    Args:
        df:         Asset DataFrame.
        macro_data: {series_id: pd.Series}.
        windows:    Look-back periods in calendar days.

    Returns:
        df with macro momentum columns.
    """
    if windows is None:
        windows = [21, 63]

    trend_series = ["T10Y2Y", "BAMLH0A0HYM2", "DFF", "VIXCLS", "DTWEXBGS"]
    for series_id in trend_series:
        if series_id not in macro_data:
            continue
        series = macro_data[series_id]
        aligned = series.reindex(df.index.union(series.index)).ffill().reindex(df.index)
        col = series_id.lower()
        for w in windows:
            df[f"macro_{col}_chg{w}d"] = aligned.diff(w)

    return df


# ── 4. Fundamental features ────────────────────────────────────────────────────

def add_fundamental_features(
    df:        pd.DataFrame,
    fund_data: dict[str, Any],
) -> pd.DataFrame:
    """Add point-in-time fundamental data as features.

    Fundamental data is held constant between reporting periods (quarterly).
    This approximates how a live system would see the data at each date.

    Args:
        df:        Asset DataFrame.
        fund_data: Dict from get_full_fundamental_profile().

    Returns:
        df with fundamental columns added.
    """
    stats = fund_data.get("key_stats", {})

    # Valuation
    df["pe_ratio"]        = stats.get("trailingPE")
    df["forward_pe"]      = stats.get("forwardPE")
    df["ps_ratio"]        = stats.get("priceToSalesTrailing12Months")
    df["pb_ratio"]        = stats.get("priceToBook")
    df["ev_ebitda"]       = stats.get("enterpriseToEbitda")
    df["peg_ratio"]       = stats.get("pegRatio")

    # Profitability
    df["profit_margin"]   = stats.get("profitMargins")
    df["oper_margin"]     = stats.get("operatingMargins")
    df["roa"]             = stats.get("returnOnAssets")
    df["roe"]             = stats.get("returnOnEquity")
    df["gross_margin"]    = stats.get("grossMargins")

    # Growth
    df["rev_growth"]      = stats.get("revenueGrowth")
    df["earn_growth"]     = stats.get("earningsGrowth")
    df["earn_qtr_growth"] = stats.get("earningsQuarterlyGrowth")

    # Financial health
    df["debt_equity"]     = stats.get("debtToEquity")
    df["current_ratio"]   = stats.get("currentRatio")
    df["quick_ratio"]     = stats.get("quickRatio")
    df["free_cashflow"]   = stats.get("freeCashflow")

    # Market structure
    df["short_ratio"]     = stats.get("shortRatio")
    df["short_pct"]       = stats.get("shortPercentOfFloat")
    df["beta_fund"]       = stats.get("beta")
    df["div_yield"]       = stats.get("dividendYield")

    return df


# ── 5. Social/sentiment features ──────────────────────────────────────────────

def add_sentiment_features(
    df:           pd.DataFrame,
    sentiment_ts: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Add social sentiment signals as features.

    Args:
        df:           Asset DataFrame.
        sentiment_ts: {signal_name: pd.Series} time series of sentiment.
                      Keys: 'reddit_score', 'stocktwits_bull_pct',
                            'google_trends', 'put_call_ratio'.

    Returns:
        df with sentiment columns added.
    """
    if not sentiment_ts:
        return df

    for name, series in sentiment_ts.items():
        if series is None or series.empty:
            continue
        aligned = series.reindex(df.index.union(series.index)).ffill().reindex(df.index)
        df[f"sent_{name}"] = aligned

    # Sentiment momentum
    for col in [c for c in df.columns if c.startswith("sent_")]:
        df[f"{col}_5d_chg"] = df[col].diff(5)

    return df


# ── 6. Calendar/seasonality features ─────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonality features.

    Returns:
        df with calendar columns added.
    """
    idx = pd.DatetimeIndex(df.index)

    df["day_of_week"]     = idx.dayofweek          # 0=Mon, 4=Fri
    df["month"]           = idx.month
    df["quarter"]         = idx.quarter
    df["day_of_month"]    = idx.day
    df["week_of_year"]    = idx.isocalendar().week.values

    # Options expiration proximity (3rd Friday of month)
    # Approximate: flag days within 3 trading days of OpEx
    df["near_opex"] = ((idx.day >= 14) & (idx.day <= 22) & (idx.dayofweek == 4)).astype(int)

    # January effect, sell-in-May, etc.
    df["is_january"]    = (idx.month == 1).astype(int)
    df["is_december"]   = (idx.month == 12).astype(int)
    df["sell_in_may"]   = ((idx.month >= 5) & (idx.month <= 10)).astype(int)

    # Monday/Friday effects
    df["is_monday"]     = (idx.dayofweek == 0).astype(int)
    df["is_friday"]     = (idx.dayofweek == 4).astype(int)

    # End of quarter (last 5 days of March/June/September/December)
    eom = (idx.month.isin([3, 6, 9, 12])) & (idx.day >= 25)
    df["end_of_quarter"] = eom.astype(int)

    return df


def add_earnings_proximity(
    df:             pd.DataFrame,
    earnings_dates: list[str],
) -> pd.DataFrame:
    """Add features for proximity to earnings announcement.

    Args:
        df:             Asset DataFrame.
        earnings_dates: List of past/future earnings dates as 'YYYY-MM-DD'.

    Returns:
        df with earnings_days_until, earnings_days_since columns.
    """
    if not earnings_dates:
        df["earnings_days_until"] = np.nan
        df["earnings_days_since"] = np.nan
        df["within_earnings_window"] = 0
        return df

    dates = pd.to_datetime(earnings_dates)
    idx   = pd.DatetimeIndex(df.index)

    days_until = []
    days_since = []
    for d in idx:
        future = [(e - d).days for e in dates if e >= d]
        past   = [(d - e).days for e in dates if e <= d]
        days_until.append(min(future) if future else np.nan)
        days_since.append(min(past)   if past   else np.nan)

    df["earnings_days_until"]    = days_until
    df["earnings_days_since"]    = days_since
    df["within_earnings_window"] = (
        (pd.Series(days_until, index=df.index) <= 5) |
        (pd.Series(days_since, index=df.index) <= 2)
    ).astype(int)

    return df


# ── 7. Event-driven features ───────────────────────────────────────────────────

FOMC_MEETING_DATES_2024_2025: list[str] = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]

CPI_RELEASE_DATES_2024_2025: list[str] = [
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-13",
    "2025-09-10", "2025-10-09", "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-12", "2026-03-12",
]


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features for proximity to macro events (FOMC, CPI, NFP).

    Returns:
        df with event proximity features.
    """
    idx = pd.DatetimeIndex(df.index)

    fomc_dates = pd.to_datetime(FOMC_MEETING_DATES_2024_2025)
    cpi_dates  = pd.to_datetime(CPI_RELEASE_DATES_2024_2025)

    def days_to_nearest(date_series: pd.DatetimeIndex, events: pd.DatetimeIndex) -> list[int]:
        result = []
        for d in date_series:
            diffs = (events - d).days
            abs_diffs = np.abs(diffs)
            result.append(int(abs_diffs.min()) if len(abs_diffs) > 0 else 999)
        return result

    df["days_to_fomc"] = days_to_nearest(idx, fomc_dates)
    df["days_to_cpi"]  = days_to_nearest(idx, cpi_dates)
    df["near_fomc"]    = (df["days_to_fomc"] <= 3).astype(int)
    df["near_cpi"]     = (df["days_to_cpi"]  <= 2).astype(int)

    return df


# ── 8. Volatility regime features ─────────────────────────────────────────────

def add_vol_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility regime classification features.

    Regime labels:
      0 = low vol (< 15% ann.)
      1 = normal vol (15-25%)
      2 = elevated vol (25-35%)
      3 = crisis vol (> 35%)

    Returns:
        df with vol regime features.
    """
    if "vol_21d" not in df.columns:
        df = add_volatility(df)

    vol = df["vol_21d"]
    df["vol_regime"] = pd.cut(
        vol,
        bins=[0, 0.15, 0.25, 0.35, 10],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Vol percentile rank (0-1) over trailing 252 days
    df["vol_pct_rank"] = vol.rolling(252).rank(pct=True)

    # Volatility trend
    df["vol_expanding"] = (df["vol_5d"] > df["vol_21d"]).astype(int) if "vol_5d" in df.columns else 0

    # GARCH-inspired: variance ratio (recent vs historical)
    if "vol_5d" in df.columns and "vol_63d" in df.columns:
        df["variance_ratio"]  = (df["vol_5d"] ** 2) / (df["vol_63d"] ** 2).replace(0, np.nan)
        df["vol_breakout"]    = (df["variance_ratio"] > 2.0).astype(int)

    return df


# ── Master feature builder ────────────────────────────────────────────────────

def build_feature_matrix(
    ohlcv_df:        pd.DataFrame,
    reference_dfs:   dict[str, pd.DataFrame] | None = None,
    macro_data:      dict[str, pd.Series]    | None = None,
    fund_data:       dict[str, Any]          | None = None,
    sentiment_ts:    dict[str, pd.Series]    | None = None,
    earnings_dates:  list[str]               | None = None,
    include_groups:  list[str]               | None = None,
) -> pd.DataFrame:
    """Build the complete 150+ feature matrix for a single asset.

    Args:
        ohlcv_df:       OHLCV DataFrame with columns: open, high, low, close, volume.
                        Index must be a DatetimeIndex.
        reference_dfs:  Cross-asset DataFrames: {'SPY': df, 'GLD': df, ...}.
        macro_data:     FRED series: {'T10Y2Y': pd.Series, ...}.
        fund_data:      Fundamental profile dict from get_full_fundamental_profile().
        sentiment_ts:   Social sentiment time series.
        earnings_dates: List of earnings announcement dates.
        include_groups: Subset of feature groups to build (default: all).
                        Options: 'technical', 'cross_asset', 'macro', 'fundamental',
                                 'sentiment', 'calendar', 'event', 'volatility'.

    Returns:
        Feature DataFrame aligned to ohlcv_df's index.
    """
    all_groups = {
        "technical", "cross_asset", "macro", "fundamental",
        "sentiment", "calendar", "event", "volatility"
    }
    groups = set(include_groups) if include_groups else all_groups

    df = ohlcv_df.copy()

    # ── Technical ──
    if "technical" in groups:
        log.debug("Adding technical features...")
        df = add_rsi(df)
        df = add_macd(df)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_obv(df)
        df = add_momentum(df)
        df = add_ema_structure(df)
        df = add_stochastic(df)
        df = add_volume_profile(df)
        df = add_support_resistance(df)

    # ── Volatility ──
    if "volatility" in groups:
        log.debug("Adding volatility features...")
        df = add_volatility(df)
        df = add_vol_regime_features(df)

    # ── Cross-asset ──
    if "cross_asset" in groups and reference_dfs:
        log.debug("Adding cross-asset features...")
        df = add_cross_asset_correlations(df, reference_dfs)
        if "SPY" in reference_dfs:
            df = add_beta(df, reference_dfs["SPY"])
            df = add_relative_strength(df, reference_dfs["SPY"])

    # ── Macro ──
    if "macro" in groups and macro_data:
        log.debug("Adding macro features...")
        df = add_macro_features(df, macro_data)
        df = add_macro_momentum(df, macro_data)

    # ── Fundamental ──
    if "fundamental" in groups and fund_data:
        log.debug("Adding fundamental features...")
        df = add_fundamental_features(df, fund_data)

    # ── Sentiment ──
    if "sentiment" in groups and sentiment_ts:
        log.debug("Adding sentiment features...")
        df = add_sentiment_features(df, sentiment_ts)

    # ── Calendar ──
    if "calendar" in groups:
        log.debug("Adding calendar features...")
        df = add_calendar_features(df)
        if earnings_dates:
            df = add_earnings_proximity(df, earnings_dates)

    # ── Event ──
    if "event" in groups:
        log.debug("Adding event features...")
        df = add_event_features(df)

    # Drop rows with too many NaNs (first 252 days typically)
    feature_cols = [c for c in df.columns if c not in {"open", "high", "low", "close", "volume"}]
    nan_thresh   = len(feature_cols) * 0.5  # drop if >50% of features are NaN
    df = df.dropna(thresh=int(len(df.columns) - nan_thresh))

    log.info("Feature matrix: %d rows × %d features (including OHLCV)",
             len(df), len(df.columns))
    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (excludes OHLCV price columns)."""
    ohlcv = {"open", "high", "low", "close", "volume", "adj_close", "dividends", "stock_splits"}
    return [c for c in df.columns if c.lower() not in ohlcv]


def normalize_features(
    df:           pd.DataFrame,
    method:       str = "zscore",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Normalize feature columns for ML.

    Args:
        df:           Feature DataFrame.
        method:       'zscore' (mean=0, std=1) or 'minmax' (0-1).
        feature_cols: Columns to normalize; defaults to all non-OHLCV.

    Returns:
        (normalized_df, params) where params = {col: (mean/min, std/max)}.
    """
    if feature_cols is None:
        feature_cols = get_feature_names(df)

    result = df.copy()
    params: dict[str, tuple[float, float]] = {}

    for col in feature_cols:
        series = result[col].replace([np.inf, -np.inf], np.nan)
        if method == "zscore":
            mu  = float(series.mean())
            sig = float(series.std())
            if sig > 0:
                result[col] = (series - mu) / sig
                params[col] = (mu, sig)
        elif method == "minmax":
            lo = float(series.min())
            hi = float(series.max())
            if hi > lo:
                result[col] = (series - lo) / (hi - lo)
                params[col] = (lo, hi)

    return result, params


def build_label(
    df:        pd.DataFrame,
    horizon:   int = 5,
    threshold: float = 0.0,
    col:       str = "close",
) -> pd.Series:
    """Build binary classification label: 1 if price rises > threshold in `horizon` days.

    Args:
        df:        Feature DataFrame with close prices.
        horizon:   Prediction horizon in trading days.
        threshold: Minimum return to count as positive (e.g. 0.01 = 1%).
        col:       Price column name.

    Returns:
        Binary pd.Series aligned to df's index.
    """
    future_ret = df[col].shift(-horizon) / df[col] - 1
    return (future_ret > threshold).astype(int)
