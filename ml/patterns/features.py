"""Feature engineering pipeline for FinBrain.

Builds 80+ ML-ready features per asset per day from raw OHLCV and macro data.

Feature groups:
  1.  Price returns       — 1d/5d/10d/21d/63d returns, log returns
  2.  Volatility          — realized vol, Garman-Klass vol, ATR
  3.  Momentum            — RSI, MACD, Stochastic, ROC, Williams %R
  4.  Trend               — Bollinger Bands, ADX, Ichimoku, EMA ribbons
  5.  Volume              — OBV, VWAP, volume ratio, CMF
  6.  Regime              — rolling beta, rolling Sharpe, max drawdown, correlation to SPY
  7.  Macro lags          — each FRED indicator lagged 1/5/21 days, yield curve slope
  8.  Cross-asset         — BTC/SPY corr, Gold/USD corr, Oil/CPI corr
  9.  Calendar            — day of week, month, quarter, FOMC window, earnings season

All functions operate on pandas DataFrames and return a feature DataFrame
with the same index as the input. NaN values at the start of the series
(due to rolling windows) are expected — callers must decide whether to
drop or forward-fill them before feeding to a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Approximate FOMC meeting months (Jan, Mar, May, Jun, Jul, Sep, Oct, Nov, Dec)
_FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 10, 11, 12}

# Earnings seasons (quarters end): Feb, May, Aug, Nov ±2 weeks
_EARNINGS_MONTHS = {2, 5, 8, 11}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price return features
# ─────────────────────────────────────────────────────────────────────────────

def price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price return features from OHLCV data.

    Args:
        df: DataFrame with at least a 'close' column, DatetimeIndex.

    Returns:
        DataFrame with columns:
            ret_1d, ret_5d, ret_10d, ret_21d, ret_63d,
            log_ret_1d, log_ret_5d,
            cum_ret_21d, cum_ret_63d
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    for n in [1, 5, 10, 21, 63]:
        out[f"ret_{n}d"] = close.pct_change(n)

    out["log_ret_1d"] = np.log(close / close.shift(1))
    out["log_ret_5d"] = np.log(close / close.shift(5))
    out["cum_ret_21d"] = (1 + out["ret_1d"]).rolling(21).apply(
        lambda x: x.prod() - 1, raw=True
    )
    out["cum_ret_63d"] = (1 + out["ret_1d"]).rolling(63).apply(
        lambda x: x.prod() - 1, raw=True
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Volatility features
# ─────────────────────────────────────────────────────────────────────────────

def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute realised and range-based volatility features.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        DataFrame with columns:
            realized_vol_5d, realized_vol_21d,
            garman_klass_vol,
            atr_14, atr_ratio
    """
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))

    out["realized_vol_5d"]  = log_ret.rolling(5).std()  * np.sqrt(252)
    out["realized_vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)

    # Garman-Klass volatility estimator
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    out["garman_klass_vol"] = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(21).mean()

    # Average True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"]    = tr.rolling(14).mean()
    out["atr_ratio"] = out["atr_14"] / df["close"]   # normalised ATR
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Momentum features
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder RSI for a price series.

    Args:
        series: Close price series.
        period: Look-back period (default 14).

    Returns:
        RSI series in [0, 100].
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum oscillator features.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.

    Returns:
        DataFrame with columns:
            rsi_14, rsi_28,
            macd, macd_signal, macd_hist,
            stoch_k, stoch_d,
            roc_10, roc_21,
            williams_r_14
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"]

    # RSI
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_28"] = _rsi(close, 28)

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"]        = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"]   = out["macd"] - out["macd_signal"]

    # Stochastic (14/3)
    low14  = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    out["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    # Rate of Change
    out["roc_10"] = close.pct_change(10) * 100
    out["roc_21"] = close.pct_change(21) * 100

    # Williams %R
    out["williams_r_14"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trend features
# ─────────────────────────────────────────────────────────────────────────────

def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend-following indicator features.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.

    Returns:
        DataFrame with columns:
            bb_upper, bb_lower, bb_mid, bb_width, bb_pct,
            adx_14,
            ema_9, ema_21, ema_50, ema_200,
            ema_9_21_cross, ema_21_50_cross,
            ichimoku_conv, ichimoku_base, ichimoku_span_a, ichimoku_span_b
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"]

    # Bollinger Bands (20, 2)
    mid              = close.rolling(20).mean()
    std20            = close.rolling(20).std()
    out["bb_upper"]  = mid + 2 * std20
    out["bb_lower"]  = mid - 2 * std20
    out["bb_mid"]    = mid
    out["bb_width"]  = (out["bb_upper"] - out["bb_lower"]) / mid
    out["bb_pct"]    = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # ADX (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = (df["high"] - df["high"].shift(1)).clip(lower=0)
    dm_minus = (df["low"].shift(1) - df["low"]).clip(lower=0)
    atr14    = tr.rolling(14).mean()
    di_plus  = 100 * dm_plus.rolling(14).mean()  / (atr14 + 1e-9)
    di_minus = 100 * dm_minus.rolling(14).mean() / (atr14 + 1e-9)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    out["adx_14"] = dx.rolling(14).mean()

    # EMA ribbon
    for span in [9, 21, 50, 200]:
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    out["ema_9_21_cross"]  = (out["ema_9"]  > out["ema_21"]).astype(int)
    out["ema_21_50_cross"] = (out["ema_21"] > out["ema_50"]).astype(int)

    # Ichimoku (9/26/52)
    nine_high  = df["high"].rolling(9).max()
    nine_low   = df["low"].rolling(9).min()
    out["ichimoku_conv"]   = (nine_high + nine_low) / 2
    base_high  = df["high"].rolling(26).max()
    base_low   = df["low"].rolling(26).min()
    out["ichimoku_base"]   = (base_high + base_low) / 2
    out["ichimoku_span_a"] = ((out["ichimoku_conv"] + out["ichimoku_base"]) / 2).shift(26)
    span_b_high = df["high"].rolling(52).max()
    span_b_low  = df["low"].rolling(52).min()
    out["ichimoku_span_b"] = ((span_b_high + span_b_low) / 2).shift(26)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Volume features
# ─────────────────────────────────────────────────────────────────────────────

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.

    Returns:
        DataFrame with columns:
            obv, obv_ema_21,
            vwap_21,
            volume_ratio_10,
            cmf_20
    """
    out   = pd.DataFrame(index=df.index)
    close = df["close"]
    vol   = df["volume"]

    # On-Balance Volume
    direction      = np.sign(close.diff()).fillna(0)
    out["obv"]     = (direction * vol).cumsum()
    out["obv_ema_21"] = out["obv"].ewm(span=21, adjust=False).mean()

    # VWAP (rolling 21-day approximation)
    typical_price  = (df["high"] + df["low"] + close) / 3
    out["vwap_21"] = (typical_price * vol).rolling(21).sum() / (vol.rolling(21).sum() + 1e-9)

    # Volume ratio vs 10-day average
    avg_vol               = vol.rolling(10).mean()
    out["volume_ratio_10"] = vol / (avg_vol + 1e-9)

    # Chaikin Money Flow (20)
    mfv = ((close - df["low"]) - (df["high"] - close)) / (df["high"] - df["low"] + 1e-9) * vol
    out["cmf_20"] = mfv.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regime features
# ─────────────────────────────────────────────────────────────────────────────

def regime_features(
    df: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute market regime features (beta, Sharpe, drawdown, correlation).

    Args:
        df: DataFrame with 'close' column.
        benchmark: Optional close price series for a benchmark asset (e.g. SPY).
                   If None, beta and correlation features are skipped.

    Returns:
        DataFrame with columns:
            rolling_sharpe_21, rolling_sharpe_63,
            max_drawdown_63,
            rolling_vol_21,
            beta_21 (if benchmark provided),
            corr_benchmark_21 (if benchmark provided),
            corr_benchmark_63 (if benchmark provided)
    """
    out   = pd.DataFrame(index=df.index)
    ret   = df["close"].pct_change()

    # Rolling Sharpe (annualised, rf=0)
    for n in [21, 63]:
        mean_r = ret.rolling(n).mean()
        std_r  = ret.rolling(n).std()
        out[f"rolling_sharpe_{n}"] = (mean_r / (std_r + 1e-9)) * np.sqrt(252)

    # Max drawdown over 63 days
    roll_max = df["close"].rolling(63).max()
    out["max_drawdown_63"] = (df["close"] - roll_max) / (roll_max + 1e-9)

    # Rolling vol
    out["rolling_vol_21"] = ret.rolling(21).std() * np.sqrt(252)

    if benchmark is not None:
        bench_ret = benchmark.pct_change()
        # Beta
        cov   = ret.rolling(21).cov(bench_ret)
        b_var = bench_ret.rolling(21).var()
        out["beta_21"] = cov / (b_var + 1e-9)
        # Correlation
        out["corr_benchmark_21"] = ret.rolling(21).corr(bench_ret)
        out["corr_benchmark_63"] = ret.rolling(63).corr(bench_ret)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. Macro lag features
# ─────────────────────────────────────────────────────────────────────────────

def macro_lag_features(
    df: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Align macro indicator values to the price index with lags.

    Macro data is released infrequently (monthly/quarterly). We forward-fill
    values then create lagged copies to avoid look-ahead bias.

    Args:
        df: Price DataFrame with a DatetimeIndex (used only for its index).
        macro: DataFrame where each column is one macro indicator (e.g. 'GDP',
               'CPIAUCSL'), indexed by date.

    Returns:
        DataFrame with one column per (indicator × lag) combination:
            {indicator}_lag1, {indicator}_lag5, {indicator}_lag21
        Plus derived features:
            yield_curve_slope (GS10 - GS2, if both present)
    """
    out = pd.DataFrame(index=df.index)

    # Reindex macro to daily price index, forward-fill
    macro_daily = macro.reindex(df.index, method="ffill")

    for col in macro_daily.columns:
        for lag in [1, 5, 21]:
            out[f"{col}_lag{lag}"] = macro_daily[col].shift(lag)

    # Yield curve slope
    if "GS10" in macro_daily.columns and "GS2" in macro_daily.columns:
        out["yield_curve_slope"] = (
            macro_daily["GS10"].shift(1) - macro_daily["GS2"].shift(1)
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cross-asset correlation features
# ─────────────────────────────────────────────────────────────────────────────

def cross_asset_features(
    df: pd.DataFrame,
    peers: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute rolling correlations against peer assets.

    Args:
        df: DataFrame with 'close' column for the target asset.
        peers: Dict mapping label → close price Series for each peer
               (e.g. {'BTC': btc_close, 'GOLD': gold_close}).

    Returns:
        DataFrame with columns:
            corr_{label}_21, corr_{label}_63  for each peer.
    """
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()

    for label, peer_close in peers.items():
        peer_ret = peer_close.pct_change()
        out[f"corr_{label}_21"] = ret.rolling(21).corr(peer_ret)
        out[f"corr_{label}_63"] = ret.rolling(63).corr(peer_ret)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 9. Calendar features
# ─────────────────────────────────────────────────────────────────────────────

def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar and event-window features from the DatetimeIndex.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        DataFrame with columns:
            day_of_week (0=Mon … 4=Fri),
            month,
            quarter,
            is_month_end,
            is_quarter_end,
            fomc_window (1 during FOMC meeting months),
            earnings_season (1 during earnings season months)
    """
    idx = pd.DatetimeIndex(df.index)
    out = pd.DataFrame(index=df.index)

    out["day_of_week"]    = idx.dayofweek
    out["month"]          = idx.month
    out["quarter"]        = idx.quarter
    out["is_month_end"]   = idx.is_month_end.astype(int)
    out["is_quarter_end"] = idx.is_quarter_end.astype(int)
    out["fomc_window"]    = idx.month.isin(_FOMC_MONTHS).astype(int)
    out["earnings_season"] = idx.month.isin(_EARNINGS_MONTHS).astype(int)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    peers: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build the complete 80+ feature matrix for one asset.

    Runs all feature groups in sequence and concatenates the results.
    NaN rows at the start (due to rolling windows) are NOT dropped here —
    the caller decides how to handle them.

    Args:
        df: OHLCV DataFrame with columns open, high, low, close, volume
            and a DatetimeIndex sorted ascending.
        macro: Optional DataFrame of macro indicators (one column per series)
               with a DatetimeIndex. Used for macro lag features.
        benchmark: Optional close price Series for a benchmark asset (SPY).
                   Used for regime beta/correlation features.
        peers: Optional dict of {label: close_series} for cross-asset
               correlation features.

    Returns:
        DataFrame with 80+ feature columns, same index as input.
    """
    parts = [
        price_return_features(df),
        volatility_features(df),
        momentum_features(df),
        trend_features(df),
        volume_features(df),
        regime_features(df, benchmark=benchmark),
        calendar_features(df),
    ]

    if macro is not None and not macro.empty:
        parts.append(macro_lag_features(df, macro))

    if peers:
        parts.append(cross_asset_features(df, peers))

    features = pd.concat(parts, axis=1)

    # Remove any duplicate columns (e.g. atr_14 appears in vol + trend)
    features = features.loc[:, ~features.columns.duplicated()]
    return features


def feature_names(
    macro_cols: list[str] | None = None,
    peer_labels: list[str] | None = None,
) -> list[str]:
    """Return the expected list of feature column names without running the pipeline.

    Useful for schema validation and model input checks.

    Args:
        macro_cols: List of macro indicator column names (e.g. ['GDP','CPIAUCSL']).
        peer_labels: List of peer asset labels (e.g. ['BTC','GOLD']).

    Returns:
        Ordered list of feature column names.
    """
    names: list[str] = []

    # Price returns (9)
    names += [f"ret_{n}d" for n in [1, 5, 10, 21, 63]]
    names += ["log_ret_1d", "log_ret_5d", "cum_ret_21d", "cum_ret_63d"]

    # Volatility (5)
    names += ["realized_vol_5d", "realized_vol_21d", "garman_klass_vol",
              "atr_14", "atr_ratio"]

    # Momentum (11)
    names += ["rsi_14", "rsi_28",
              "macd", "macd_signal", "macd_hist",
              "stoch_k", "stoch_d",
              "roc_10", "roc_21",
              "williams_r_14"]

    # Trend (16)
    names += ["bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct",
              "adx_14",
              "ema_9", "ema_21", "ema_50", "ema_200",
              "ema_9_21_cross", "ema_21_50_cross",
              "ichimoku_conv", "ichimoku_base",
              "ichimoku_span_a", "ichimoku_span_b"]

    # Volume (5)
    names += ["obv", "obv_ema_21", "vwap_21", "volume_ratio_10", "cmf_20"]

    # Regime (4 base + 3 with benchmark)
    names += ["rolling_sharpe_21", "rolling_sharpe_63",
              "max_drawdown_63", "rolling_vol_21"]
    names += ["beta_21", "corr_benchmark_21", "corr_benchmark_63"]

    # Calendar (7)
    names += ["day_of_week", "month", "quarter",
              "is_month_end", "is_quarter_end",
              "fomc_window", "earnings_season"]

    # Macro lags (3 per indicator + yield curve slope)
    if macro_cols:
        for col in macro_cols:
            for lag in [1, 5, 21]:
                names.append(f"{col}_lag{lag}")
        if "GS10" in macro_cols and "GS2" in macro_cols:
            names.append("yield_curve_slope")

    # Cross-asset (2 per peer)
    if peer_labels:
        for label in peer_labels:
            names += [f"corr_{label}_21", f"corr_{label}_63"]

    return names
