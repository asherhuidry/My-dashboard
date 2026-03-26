"""OHLCV price data route."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _fetch_yfinance(symbol: str, days: int) -> list[dict]:
    """Fetch OHLCV from yfinance and return as list of dicts."""
    period = f"{max(days // 365 + 1, 1)}y" if days > 365 else ("1y" if days > 180 else "6mo")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        return []
    df = df.tail(days)
    df.index = pd.to_datetime(df.index, utc=True)
    records = []
    for ts, row in df.iterrows():
        records.append({
            "time":   int(ts.timestamp()),
            "date":   ts.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),   4),
            "high":   round(float(row["High"]),   4),
            "low":    round(float(row["Low"]),    4),
            "close":  round(float(row["Close"]),  4),
            "volume": int(row["Volume"]),
        })
    return records


@router.get("/prices/{symbol}")
def get_prices(
    symbol: str,
    days: int = Query(default=365, ge=7, le=1825),
) -> dict:
    """Return OHLCV candlestick data for a symbol."""
    symbol = symbol.upper()
    data = _fetch_yfinance(symbol, days)
    if not data:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

    closes = [r["close"] for r in data]
    latest = closes[-1]
    prev   = closes[-2] if len(closes) > 1 else latest
    change_pct = round((latest - prev) / prev * 100, 2) if prev else 0

    return {
        "symbol":     symbol,
        "candles":    data,
        "latest":     latest,
        "change_pct": change_pct,
        "high_52w":   round(max(r["high"] for r in data[-252:] if r), 4),
        "low_52w":    round(min(r["low"]  for r in data[-252:] if r), 4),
    }
