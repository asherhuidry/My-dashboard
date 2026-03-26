"""Symbol search route — queries yfinance for ticker info."""
from __future__ import annotations

import yfinance as yf
from fastapi import APIRouter, Query

router = APIRouter()

# Popular symbols for quick suggestions
_POPULAR = [
    {"symbol": "AAPL",  "name": "Apple Inc.",               "type": "equity"},
    {"symbol": "MSFT",  "name": "Microsoft Corporation",    "type": "equity"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",            "type": "equity"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",          "type": "equity"},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation",       "type": "equity"},
    {"symbol": "META",  "name": "Meta Platforms Inc.",      "type": "equity"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",               "type": "equity"},
    {"symbol": "BTC-USD","name": "Bitcoin USD",             "type": "crypto"},
    {"symbol": "ETH-USD","name": "Ethereum USD",            "type": "crypto"},
    {"symbol": "SOL-USD","name": "Solana USD",              "type": "crypto"},
    {"symbol": "SPY",   "name": "S&P 500 ETF",              "type": "etf"},
    {"symbol": "QQQ",   "name": "Nasdaq 100 ETF",           "type": "etf"},
    {"symbol": "GLD",   "name": "Gold ETF",                 "type": "commodity"},
    {"symbol": "EURUSD=X","name": "EUR/USD",                "type": "forex"},
]

@router.get("/search")
def search_symbols(q: str = Query(default="", min_length=0)) -> dict:
    """Search for ticker symbols."""
    if not q:
        return {"results": _POPULAR[:8]}

    q_upper = q.upper()
    # Filter popular list first
    matches = [s for s in _POPULAR if q_upper in s["symbol"] or q_upper in s["name"].upper()]

    # Try yfinance for unknown tickers
    if not matches:
        try:
            ticker = yf.Ticker(q_upper)
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            if price:
                matches = [{
                    "symbol": q_upper,
                    "name": getattr(ticker.info, "longName", q_upper) if hasattr(ticker, "info") else q_upper,
                    "type": "equity",
                    "price": round(float(price), 2),
                }]
        except Exception:
            pass

    return {"results": matches[:10]}
