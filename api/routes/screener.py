"""Stock screener route — compute key features for all universe assets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, Query

from ml.patterns.features import build_features

router = APIRouter()

UNIVERSE = {
    "equity":    ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","MA",
                  "UNH","JNJ","XOM","WMT","BAC","HD","NFLX","ORCL","ADBE","CRM"],
    "crypto":    ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","AVAX-USD","LINK-USD","DOGE-USD"],
    "etf":       ["SPY","QQQ","IWM","GLD","TLT"],
    "forex":     ["EURUSD=X","GBPUSD=X","JPYUSD=X"],
}

def _safe(v) -> float | None:
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None

def _signal(features: dict) -> str:
    bull = 0
    bear = 0
    rsi = features.get("rsi_14")
    if rsi is not None:
        if rsi < 35:   bull += 1
        elif rsi > 65: bear += 1
    macd     = features.get("macd", 0) or 0
    macd_sig = features.get("macd_signal", 0) or 0
    if macd > macd_sig: bull += 1
    else:               bear += 1
    cross = features.get("ema_9_21_cross")
    if cross is not None:
        if cross == 1: bull += 1
        else:          bear += 1
    if bull > bear: return "bullish"
    if bear > bull: return "bearish"
    return "neutral"

def _fetch_one(symbol: str, asset_class: str) -> dict | None:
    try:
        df = yf.Ticker(symbol).history(period="6mo")
        if len(df) < 60:
            return None
        df.index   = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        ohlcv      = df[["open","high","low","close","volume"]].copy()
        feat_df    = build_features(ohlcv)
        latest     = feat_df.iloc[-1].to_dict()
        feats      = {k: _safe(v) for k, v in latest.items()}

        close      = ohlcv["close"]
        price      = _safe(close.iloc[-1])
        ret_1d     = _safe((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else None
        ret_5d     = _safe((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else None
        ret_21d    = _safe((close.iloc[-1] - close.iloc[-22])/ close.iloc[-22]* 100) if len(close) > 21 else None
        high_52w   = _safe(ohlcv["high"].tail(252).max())
        low_52w    = _safe(ohlcv["low"].tail(252).min())
        pct_from_high = _safe((price - high_52w) / high_52w * 100) if price and high_52w else None

        return {
            "symbol":         symbol.replace("-USD","").replace("=X",""),
            "full_symbol":    symbol,
            "asset_class":    asset_class,
            "price":          price,
            "ret_1d":         ret_1d,
            "ret_5d":         ret_5d,
            "ret_21d":        ret_21d,
            "signal":         _signal(feats),
            "rsi_14":         feats.get("rsi_14"),
            "rsi_28":         feats.get("rsi_28"),
            "adx_14":         feats.get("adx_14"),
            "bb_pct":         feats.get("bb_pct"),
            "macd_hist":      feats.get("macd_hist"),
            "realized_vol":   feats.get("realized_vol_21d"),
            "volume_ratio":   feats.get("volume_ratio_10"),
            "sharpe_21":      feats.get("rolling_sharpe_21"),
            "max_drawdown":   feats.get("max_drawdown_63"),
            "ema_cross":      feats.get("ema_9_21_cross"),
            "high_52w":       high_52w,
            "low_52w":        low_52w,
            "pct_from_high":  pct_from_high,
        }
    except Exception:
        return None


@router.get("/screener")
def screener(
    asset_class: str  = Query(default="all"),
    signal:      str  = Query(default="all"),
    min_rsi:     float = Query(default=0,   ge=0,   le=100),
    max_rsi:     float = Query(default=100, ge=0,   le=100),
    min_adx:     float = Query(default=0,   ge=0),
    sort_by:     str   = Query(default="ret_1d"),
    sort_desc:   bool  = Query(default=True),
    limit:       int   = Query(default=50,  ge=1, le=100),
) -> dict:
    """Run the screener and return filtered, sorted asset list."""
    # Build symbol list
    symbols: list[tuple[str, str]] = []
    for cls, syms in UNIVERSE.items():
        if asset_class == "all" or cls == asset_class:
            for s in syms:
                symbols.append((s, cls))

    # Fetch concurrently
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, sym, cls): sym for sym, cls in symbols}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)

    # Filter
    if signal != "all":
        results = [r for r in results if r["signal"] == signal]
    results = [r for r in results if
               (r["rsi_14"] is None or min_rsi <= (r["rsi_14"] or 0) <= max_rsi) and
               (r["adx_14"] is None or (r["adx_14"] or 0) >= min_adx)]

    # Sort
    def sort_key(r):
        v = r.get(sort_by)
        return (v is None, -(v or 0) if sort_desc else (v or 0))

    results.sort(key=sort_key)
    return {"results": results[:limit], "total": len(results)}
