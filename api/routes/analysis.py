"""Full stock analysis route — OHLCV + 74 features + signal consensus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from ml.patterns.features import build_features

router = APIRouter()


def _classify_signal(features: dict) -> dict:
    """Derive a bullish/bearish/neutral consensus from key indicators."""
    signals: list[str] = []

    rsi = features.get("rsi_14")
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:   signals.append("bullish")
        elif rsi > 70: signals.append("bearish")
        else:          signals.append("neutral")

    macd      = features.get("macd", 0)
    macd_sig  = features.get("macd_signal", 0)
    if macd is not None and macd_sig is not None:
        signals.append("bullish" if macd > macd_sig else "bearish")

    bb_pct = features.get("bb_pct")
    if bb_pct is not None and not np.isnan(bb_pct):
        if bb_pct < 0.2:   signals.append("bullish")
        elif bb_pct > 0.8: signals.append("bearish")
        else:              signals.append("neutral")

    adx = features.get("adx_14")
    if adx is not None and not np.isnan(adx):
        signals.append("trending" if adx > 25 else "ranging")

    ema_cross = features.get("ema_9_21_cross")
    if ema_cross is not None:
        signals.append("bullish" if ema_cross == 1 else "bearish")

    bull = signals.count("bullish")
    bear = signals.count("bearish")

    if bull > bear + 1:     overall = "bullish"
    elif bear > bull + 1:   overall = "bearish"
    else:                   overall = "neutral"

    score = round((bull - bear) / max(len(signals), 1) * 100, 1)
    return {
        "overall":    overall,
        "score":      score,
        "bull_count": bull,
        "bear_count": bear,
        "signals":    signals,
    }


def _safe(v) -> float | None:
    """Return float or None, stripping NaN/inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


@router.get("/analyze/{symbol}")
def analyze_symbol(
    symbol: str,
    days: int = Query(default=365, ge=30, le=1825),
) -> dict:
    """Full analysis: OHLCV + computed features + signal consensus."""
    symbol = symbol.upper()

    # ── Fetch data ────────────────────────────────────────────────────────────
    period = "2y" if days > 365 else "1y"
    ticker = yf.Ticker(symbol)
    hist   = ticker.history(period=period)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    hist.index = pd.to_datetime(hist.index, utc=True)
    hist.columns = [c.lower() for c in hist.columns]
    df = hist[["open", "high", "low", "close", "volume"]].copy().tail(days)

    # ── Company info ──────────────────────────────────────────────────────────
    try:
        info      = ticker.info
        long_name = info.get("longName", symbol)
        sector    = info.get("sector", "")
        industry  = info.get("industry", "")
        market_cap = info.get("marketCap")
        pe_ratio  = info.get("trailingPE")
        description = info.get("longBusinessSummary", "")[:400]
    except Exception:
        long_name = symbol
        sector = industry = description = ""
        market_cap = pe_ratio = None

    # ── Feature engineering ───────────────────────────────────────────────────
    feat_df    = build_features(df)
    latest_row = feat_df.iloc[-1].to_dict()
    features   = {k: _safe(v) for k, v in latest_row.items()}

    # ── Returns summary ───────────────────────────────────────────────────────
    close = df["close"]
    def ret(n: int) -> float | None:
        return _safe((close.iloc[-1] - close.iloc[-n]) / close.iloc[-n] * 100) if len(close) >= n else None

    # ── Candles (last 365 rows) ───────────────────────────────────────────────
    candles = []
    for ts, row in df.tail(365).iterrows():
        candles.append({
            "time":   int(ts.timestamp()),
            "date":   ts.strftime("%Y-%m-%d"),
            "open":   round(float(row["open"]),   4),
            "high":   round(float(row["high"]),   4),
            "low":    round(float(row["low"]),    4),
            "close":  round(float(row["close"]),  4),
            "volume": int(row["volume"]),
        })

    # ── Indicator series (for multi-panel charts) ─────────────────────────────
    def series(col: str) -> list[dict]:
        s = feat_df[col].dropna() if col in feat_df.columns else pd.Series(dtype=float)
        return [{"time": int(ts.timestamp()), "value": _safe(v)}
                for ts, v in s.tail(365).items() if _safe(v) is not None]

    signal = _classify_signal(features)

    return {
        "symbol":      symbol,
        "name":        long_name,
        "sector":      sector,
        "industry":    industry,
        "description": description,
        "market_cap":  market_cap,
        "pe_ratio":    _safe(pe_ratio),
        "price":       _safe(close.iloc[-1]),
        "returns": {
            "1d":  ret(2),
            "5d":  ret(6),
            "21d": ret(22),
            "63d": ret(64),
            "ytd": ret(min(len(close), 252)),
        },
        "candles": candles,
        "indicators": {
            "rsi_14":      series("rsi_14"),
            "rsi_28":      series("rsi_28"),
            "macd":        series("macd"),
            "macd_signal": series("macd_signal"),
            "macd_hist":   series("macd_hist"),
            "bb_upper":    series("bb_upper"),
            "bb_lower":    series("bb_lower"),
            "bb_mid":      series("bb_mid"),
            "ema_9":       series("ema_9"),
            "ema_21":      series("ema_21"),
            "ema_50":      series("ema_50"),
            "ema_200":     series("ema_200"),
            "stoch_k":     series("stoch_k"),
            "stoch_d":     series("stoch_d"),
            "adx_14":      series("adx_14"),
            "williams_r":  series("williams_r_14"),
            "obv":         series("obv"),
            "cmf_20":      series("cmf_20"),
        },
        "features":    features,
        "signal":      signal,
        "vol_52w":     _safe(feat_df["realized_vol_21d"].dropna().iloc[-1] if "realized_vol_21d" in feat_df.columns and not feat_df["realized_vol_21d"].dropna().empty else None),
        "atr":         _safe(feat_df["atr_14"].dropna().iloc[-1] if "atr_14" in feat_df.columns and not feat_df["atr_14"].dropna().empty else None),
    }
