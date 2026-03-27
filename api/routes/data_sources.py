"""Thin REST wrappers around data/sources/ for direct frontend access."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from data.sources.news         import fetch_all_news
from data.sources.fundamentals import get_full_fundamental_profile

router = APIRouter()


@router.get("/news/{symbol}")
def get_news(
    symbol:      str,
    include_web: bool = Query(default=True),
    max_yf:      int  = Query(default=8, ge=1, le=20),
) -> dict:
    """Aggregate news + sentiment for a ticker symbol.

    Args:
        symbol:      Ticker (e.g. AAPL).
        include_web: Include a DuckDuckGo web search alongside yfinance news.
        max_yf:      Max yfinance articles.
    """
    symbol = symbol.upper()
    try:
        return fetch_all_news(symbol, include_web_search=include_web, max_yf=max_yf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fundamentals/{symbol}")
def get_fundamentals(symbol: str) -> dict:
    """Comprehensive fundamental profile: earnings, insiders, options, SEC filings.

    Args:
        symbol: Ticker (e.g. AAPL).
    """
    symbol = symbol.upper()
    try:
        return get_full_fundamental_profile(symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
