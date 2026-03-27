"""Walk-forward backtest engine.

Strategy: generate a signal at every bar using the same rules as the
real-time analysis endpoint, then simulate a long-only strategy:
  - Long (1 unit) when signal == bullish
  - Flat (0 units) when bearish or neutral
  - 0.1% round-trip transaction cost
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from ml.patterns.features import build_features

router = APIRouter()
log    = logging.getLogger(__name__)

COMMISSION = 0.001   # 0.1% per side
MIN_BARS   = 120


# ── Signal generator (vectorised) ────────────────────────────────────────────

def _generate_signals(feat_df: pd.DataFrame) -> pd.Series:
    """Generate a +1/0 signal at every bar.

    Logic mirrors _classify_signal in analysis.py but is vectorised over
    the full DataFrame so we never look ahead.

    Args:
        feat_df: Feature DataFrame from build_features().

    Returns:
        Series of 1 (long) or 0 (flat), indexed like feat_df.
    """
    bull = pd.Series(0, index=feat_df.index, dtype=float)
    bear = pd.Series(0, index=feat_df.index, dtype=float)

    # RSI 14
    if "rsi_14" in feat_df.columns:
        bull += (feat_df["rsi_14"] < 30).astype(float)
        bear += (feat_df["rsi_14"] > 70).astype(float)

    # MACD vs signal
    if "macd" in feat_df.columns and "macd_signal" in feat_df.columns:
        bull += (feat_df["macd"] > feat_df["macd_signal"]).astype(float)
        bear += (feat_df["macd"] < feat_df["macd_signal"]).astype(float)

    # Bollinger %B
    if "bb_pct" in feat_df.columns:
        bull += (feat_df["bb_pct"] < 0.2).astype(float)
        bear += (feat_df["bb_pct"] > 0.8).astype(float)

    # EMA 9/21 cross
    if "ema_9_21_cross" in feat_df.columns:
        bull += (feat_df["ema_9_21_cross"] == 1).astype(float)
        bear += (feat_df["ema_9_21_cross"] == -1).astype(float)

    overall_bull = bull > bear + 1
    return overall_bull.astype(int)


# ── Backtest simulation ────────────────────────────────────────────────────────

def _run_backtest(
    close:   pd.Series,
    signals: pd.Series,
    initial: float = 10_000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate a long-only strategy from pre-computed signals.

    Args:
        close:   Daily close price series (aligned with signals).
        signals: 1 = long, 0 = flat.
        initial: Starting portfolio value in USD.

    Returns:
        (equity_df, metrics_dict)
    """
    # Daily return of the asset
    asset_ret = close.pct_change().fillna(0)

    # Detect trade entries/exits to apply commission
    prev_pos = signals.shift(1).fillna(0)
    trades   = (signals != prev_pos).astype(float)
    cost     = trades * COMMISSION

    # Strategy daily return
    strat_ret = signals.shift(1).fillna(0) * asset_ret - cost

    # Benchmark: buy and hold
    bench_ret = asset_ret

    # Equity curves
    equity    = initial * (1 + strat_ret).cumprod()
    benchmark = initial * (1 + bench_ret).cumprod()

    equity_df = pd.DataFrame({
        "time":      close.index,
        "equity":    equity.round(2),
        "benchmark": benchmark.round(2),
        "signal":    signals,
    }).reset_index(drop=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    trading_days = 252
    n_years      = max(len(strat_ret) / trading_days, 0.01)

    total_ret  = float((equity.iloc[-1] / initial) - 1)
    bench_ret_  = float((benchmark.iloc[-1] / initial) - 1)
    ann_ret    = float((1 + total_ret) ** (1 / n_years) - 1)

    # Sharpe
    excess = strat_ret - 0.05 / trading_days          # 5% risk-free
    sharpe = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(trading_days))

    # Sortino (only downside deviation)
    down   = strat_ret[strat_ret < 0]
    sortino = float(excess.mean() / (down.std() + 1e-9) * np.sqrt(trading_days))

    # Max drawdown
    peak    = equity.cummax()
    dd      = (equity - peak) / (peak + 1e-9)
    max_dd  = float(dd.min())

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Trade stats
    entry_mask  = (signals == 1) & (prev_pos == 0)
    exit_mask   = (signals == 0) & (prev_pos == 1)
    n_trades    = int(entry_mask.sum())

    # Win rate: per-trade return between entry and next exit
    trade_rets = []
    entry_idx  = None
    sig_list   = list(signals)
    ret_list   = list(asset_ret)
    for i, s in enumerate(sig_list):
        if s == 1 and (i == 0 or sig_list[i-1] == 0):
            entry_idx = i
        elif s == 0 and entry_idx is not None and i > 0 and sig_list[i-1] == 1:
            trade_sum = sum(ret_list[entry_idx:i])
            trade_rets.append(trade_sum)
            entry_idx = None
    win_rate = float(sum(r > 0 for r in trade_rets) / max(len(trade_rets), 1))
    avg_trade = float(np.mean(trade_rets)) if trade_rets else 0.0

    metrics = {
        "total_return_pct":     round(total_ret * 100, 2),
        "benchmark_return_pct": round(bench_ret_ * 100, 2),
        "alpha_pct":            round((total_ret - bench_ret_) * 100, 2),
        "annualised_return_pct":round(ann_ret * 100, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "sortino_ratio":        round(sortino, 3),
        "calmar_ratio":         round(calmar, 3),
        "max_drawdown_pct":     round(max_dd * 100, 2),
        "win_rate_pct":         round(win_rate * 100, 1),
        "total_trades":         n_trades,
        "avg_trade_pct":        round(avg_trade * 100, 3),
        "days_backtested":      len(close),
        "years_backtested":     round(n_years, 2),
    }
    return equity_df, metrics


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/backtest/{symbol}")
def backtest_symbol(
    symbol: str,
    days:   int = Query(default=730, ge=120, le=2000),
) -> dict:
    """Walk-forward backtest of signal consensus strategy.

    Args:
        symbol: Ticker symbol.
        days:   Number of trading days to backtest (min 120, max 2000).
    """
    symbol = symbol.upper()

    # ── Fetch ──────────────────────────────────────────────────────────────────
    try:
        period = "5y" if days > 730 else "3y" if days > 365 else "2y"
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=period)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        hist.index   = pd.to_datetime(hist.index, utc=True)
        hist.columns = [c.lower() for c in hist.columns]
        df           = hist[["open", "high", "low", "close", "volume"]].copy().tail(days)

        if len(df) < MIN_BARS:
            raise HTTPException(status_code=422, detail="Not enough history for backtest")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Features & signals ────────────────────────────────────────────────────
    try:
        feat_df = build_features(df)
        close   = df["close"].loc[feat_df.index]
        signals = _generate_signals(feat_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature computation failed: {exc}")

    # ── Simulate ───────────────────────────────────────────────────────────────
    equity_df, metrics = _run_backtest(close, signals)

    # ── Equity curve for chart ─────────────────────────────────────────────────
    curve = []
    for _, row in equity_df.iterrows():
        ts = row["time"]
        curve.append({
            "time":      int(ts.timestamp()),
            "equity":    float(row["equity"]),
            "benchmark": float(row["benchmark"]),
            "signal":    int(row["signal"]),
        })

    # ── Recent trades (last 20) ────────────────────────────────────────────────
    trades     = []
    entry_i    = None
    entry_price = None
    sig_arr    = list(signals)
    close_arr  = list(close)
    idx_arr    = list(close.index)

    for i, s in enumerate(sig_arr):
        prev = sig_arr[i-1] if i > 0 else 0
        if s == 1 and prev == 0:
            entry_i     = i
            entry_price = close_arr[i]
        elif s == 0 and prev == 1 and entry_i is not None:
            exit_price = close_arr[i]
            pnl_pct    = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "entry_date":  idx_arr[entry_i].strftime("%Y-%m-%d"),
                "exit_date":   idx_arr[i].strftime("%Y-%m-%d"),
                "entry_price": round(float(entry_price), 4),
                "exit_price":  round(float(exit_price),  4),
                "return_pct":  round(pnl_pct, 2),
                "bars_held":   i - entry_i,
                "win":         pnl_pct > 0,
            })
            entry_i = None

    # Latest price context
    latest_close = float(close.iloc[-1])
    first_close  = float(close.iloc[0])
    price_change = round((latest_close / first_close - 1) * 100, 2)

    return {
        "symbol":        symbol,
        "days":          days,
        "latest_price":  round(latest_close, 4),
        "price_change_pct": price_change,
        "metrics":       metrics,
        "equity_curve":  curve,
        "trades":        trades[-20:],
        "total_trades":  len(trades),
    }
