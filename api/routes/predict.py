"""LSTM prediction endpoint — directional probability for next-bar move.

Falls back to a rules-based probability when no trained checkpoint exists.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from ml.patterns.features import build_features

router = APIRouter()
log    = logging.getLogger(__name__)

CHECKPOINT_PATH = Path("checkpoints/finbrain_lstm.pt")
SEQ_LEN         = 30          # rolling window the LSTM expects
MIN_BARS        = SEQ_LEN + 5


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(v: Any) -> float | None:
    """Strip NaN/inf, return float or None."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def _rules_probability(features: dict) -> tuple[float, list[str]]:
    """Derive a 0–1 directional probability from rule signals.

    Each positive signal adds weight; total is normalised to [0,1].
    Probability > 0.55 → bullish lean, < 0.45 → bearish lean.

    Args:
        features: Latest feature dict from build_features().

    Returns:
        (probability, list_of_contributing_signals)
    """
    contrib: list[str] = []
    votes   = []

    rsi = features.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            votes.append(1.0); contrib.append("RSI oversold")
        elif rsi > 70:
            votes.append(0.0); contrib.append("RSI overbought")
        else:
            votes.append(0.5); contrib.append("RSI neutral")

    macd     = features.get("macd", 0) or 0
    macd_sig = features.get("macd_signal", 0) or 0
    if macd > macd_sig:
        votes.append(0.65); contrib.append("MACD bullish cross")
    else:
        votes.append(0.35); contrib.append("MACD bearish cross")

    bb = features.get("bb_pct")
    if bb is not None:
        if bb < 0.2:
            votes.append(0.72); contrib.append("Near BB lower (oversold)")
        elif bb > 0.8:
            votes.append(0.28); contrib.append("Near BB upper (overbought)")
        else:
            votes.append(0.5); contrib.append("BB mid-range")

    ema_cross = features.get("ema_9_21_cross")
    if ema_cross == 1:
        votes.append(0.68); contrib.append("EMA 9/21 bullish cross")
    elif ema_cross == -1:
        votes.append(0.32); contrib.append("EMA 9/21 bearish cross")

    adx = features.get("adx_14")
    if adx and adx > 25:
        contrib.append(f"Strong trend ADX={adx:.1f}")

    sharpe = features.get("rolling_sharpe_21")
    if sharpe is not None:
        if sharpe > 0.5:
            votes.append(0.62); contrib.append("Positive Sharpe momentum")
        elif sharpe < -0.5:
            votes.append(0.38); contrib.append("Negative Sharpe momentum")

    vol_pct = features.get("vol_regime")
    if vol_pct == 1:
        contrib.append("High volatility regime")
    elif vol_pct == 0:
        contrib.append("Low volatility regime")

    prob = float(np.mean(votes)) if votes else 0.5
    return round(prob, 4), contrib


def _lstm_probability(df: pd.DataFrame, feat_df: pd.DataFrame) -> tuple[float, str] | None:
    """Run LSTM inference if a checkpoint is available.

    Args:
        df:      Raw OHLCV DataFrame.
        feat_df: Feature DataFrame from build_features().

    Returns:
        (probability, model_version) or None if checkpoint unavailable.
    """
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        import torch
        from ml.patterns.lstm import load_model, predict as lstm_predict

        model, ckpt = load_model(CHECKPOINT_PATH, device="cpu")
        feature_cols: list[str] = ckpt["feature_cols"]
        scaler_mean: np.ndarray = ckpt["scaler_mean"]
        scaler_std:  np.ndarray = ckpt["scaler_std"]
        epoch = ckpt.get("epoch", "?")

        # Build (SEQ_LEN, n_features) window from the tail of feat_df
        available = [c for c in feature_cols if c in feat_df.columns]
        if len(available) < len(feature_cols) * 0.8:
            log.warning("Feature mismatch: %d/%d cols available", len(available), len(feature_cols))
            return None

        window = feat_df[available].tail(SEQ_LEN).values.astype(np.float32)
        if window.shape[0] < SEQ_LEN:
            return None

        # Align with scaler (handle feature count mismatch gracefully)
        if window.shape[1] != len(scaler_mean):
            return None

        prob = lstm_predict(model, window, scaler_mean, scaler_std)
        return round(prob, 4), f"epoch-{epoch}"
    except Exception as exc:
        log.warning("LSTM inference failed: %s", exc)
        return None


def _top_features(features: dict, n: int = 8) -> list[dict]:
    """Return the most informative features for display.

    Args:
        features: Latest feature dict.
        n:        How many features to return.

    Returns:
        List of {name, value, category} dicts.
    """
    showcase = [
        ("rsi_14",            "RSI 14",           "momentum"),
        ("rsi_28",            "RSI 28",           "momentum"),
        ("macd",              "MACD",             "trend"),
        ("macd_signal",       "MACD Signal",      "trend"),
        ("adx_14",            "ADX 14",           "trend"),
        ("bb_pct",            "BB Position",      "volatility"),
        ("realized_vol_21d",  "Realized Vol 21D", "volatility"),
        ("rolling_sharpe_21", "Sharpe 21D",       "risk"),
        ("max_drawdown_63",   "Max DD 63D",       "risk"),
        ("obv",               "OBV",              "volume"),
        ("cmf_20",            "CMF 20",           "volume"),
        ("williams_r_14",     "Williams %R",      "momentum"),
        ("stoch_k_14",        "Stoch %K",         "momentum"),
        ("ema_9_21_cross",    "EMA Cross 9/21",   "trend"),
        ("ema_50_200_cross",  "EMA Cross 50/200", "trend"),
    ]
    out = []
    for key, label, cat in showcase:
        v = features.get(key)
        if v is not None:
            out.append({"name": label, "key": key, "value": round(float(v), 4), "category": cat})
        if len(out) >= n:
            break
    return out


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/predict/{symbol}")
def predict_symbol(
    symbol: str,
    days:   int = Query(default=365, ge=60, le=1825),
) -> dict:
    """Return directional probability and ML signal for a symbol.

    Uses trained LSTM checkpoint when available; falls back to rule-based
    probability scoring so the endpoint is always useful.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        days:   History window for feature computation.
    """
    symbol = symbol.upper()

    # ── Fetch & feature-engineer ───────────────────────────────────────────────
    try:
        ticker  = yf.Ticker(symbol)
        period  = "2y" if days > 365 else "1y"
        hist    = ticker.history(period=period)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        hist.index   = pd.to_datetime(hist.index, utc=True)
        hist.columns = [c.lower() for c in hist.columns]
        df           = hist[["open", "high", "low", "close", "volume"]].copy().tail(days)

        if len(df) < MIN_BARS:
            raise HTTPException(status_code=422, detail=f"Not enough data for {symbol}")

        feat_df    = build_features(df)
        latest_row = feat_df.iloc[-1].to_dict()
        features   = {k: _safe(v) for k, v in latest_row.items() if _safe(v) is not None}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── LSTM inference ─────────────────────────────────────────────────────────
    lstm_result = _lstm_probability(df, feat_df)
    if lstm_result:
        probability, model_version = lstm_result
        model_type = "lstm"
        confidence = "high" if abs(probability - 0.5) > 0.15 else "medium" if abs(probability - 0.5) > 0.07 else "low"
        signals    = []
    else:
        probability, signals = _rules_probability(features)
        model_type    = "rules"
        model_version = "v1-signals"
        confidence = "high" if abs(probability - 0.5) > 0.15 else "medium" if abs(probability - 0.5) > 0.07 else "low"

    # ── Direction label ────────────────────────────────────────────────────────
    direction = "bullish" if probability > 0.55 else "bearish" if probability < 0.45 else "neutral"

    # ── Price context ──────────────────────────────────────────────────────────
    close = df["close"]
    price = _safe(close.iloc[-1])
    ret1d = _safe((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else None

    # ── Historical probability series (walk-forward rules, last 90 bars) ───────
    prob_series = []
    window_size = 60
    tail_df     = feat_df.tail(90 + window_size)
    for i in range(window_size, len(tail_df)):
        row_feats = {k: _safe(v) for k, v in tail_df.iloc[i].to_dict().items() if _safe(v) is not None}
        p, _      = _rules_probability(row_feats)
        ts        = tail_df.index[i]
        prob_series.append({"time": int(ts.timestamp()), "value": p})

    return {
        "symbol":        symbol,
        "price":         price,
        "return_1d_pct": ret1d,
        "probability":   probability,
        "direction":     direction,
        "confidence":    confidence,
        "model_type":    model_type,
        "model_version": model_version,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "signals":       signals,
        "top_features":  _top_features(features),
        "prob_series":   prob_series,
    }
