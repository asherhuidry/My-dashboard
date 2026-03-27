"""Social & alternative data pipeline.

Sources (all free, no paid subscriptions):
  - Reddit: WSB, r/stocks, r/investing, r/SecurityAnalysis
  - Google Trends via pytrends (search interest as proxy for retail attention)
  - StockTwits public API (Twitter-like financial messages)
  - Fear & Greed composite (CNN + VIX + put/call + breadth)
  - Options sentiment (put/call ratio from CBOE public data)

These signals are alpha sources unavailable in traditional quant data.
Retail attention, social sentiment, and search trends often lead price
moves by 1-3 days for high-profile assets.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "FinBrain/1.0 research@finbrain.ai"}


# ── Reddit sentiment ──────────────────────────────────────────────────────────

REDDIT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "options",
    "StockMarket",
]

def fetch_reddit_sentiment(
    symbol: str,
    subreddits: list[str] | None = None,
    hours_back: int = 24,
    max_posts: int = 50,
) -> dict[str, Any]:
    """Fetch and score Reddit posts mentioning a ticker.

    Uses Reddit's public JSON API (no credentials required for read-only access).

    Args:
        symbol:     Ticker symbol to search for.
        subreddits: List of subreddits to search.
        hours_back: How many hours back to look.
        max_posts:  Max posts to analyse.

    Returns:
        Dict with: symbol, mentions, sentiment_score, label, top_posts.
    """
    if subreddits is None:
        subreddits = REDDIT_SUBREDDITS[:3]  # be conservative

    sym_variants = {symbol.upper(), f"${symbol.upper()}", symbol.lower()}
    cutoff_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)).timestamp()

    all_posts: list[dict] = []

    for sub in subreddits:
        try:
            # Reddit public JSON API — rate-limited but free
            url  = f"https://www.reddit.com/r/{sub}/search.json"
            params = {"q": symbol.upper(), "sort": "new", "limit": 25, "t": "day"}
            r = requests.get(url, params=params, headers={
                "User-Agent": "FinBrain:v1.0 (financial research tool)"
            }, timeout=8)
            if r.status_code == 429:
                log.warning("Reddit rate limited on r/%s", sub)
                time.sleep(2)
                continue
            if r.status_code != 200:
                continue
            posts = r.json().get("data", {}).get("children", [])
            for post in posts:
                d   = post.get("data", {})
                ts  = d.get("created_utc", 0)
                if ts < cutoff_ts:
                    continue
                title = d.get("title", "")
                body  = d.get("selftext", "")
                # Check that it actually mentions the symbol
                text  = f"{title} {body}".upper()
                if not any(v.upper() in text for v in sym_variants):
                    continue
                all_posts.append({
                    "subreddit": sub,
                    "title":     title[:120],
                    "score":     d.get("score", 0),
                    "comments":  d.get("num_comments", 0),
                    "created":   datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "url":       f"https://reddit.com{d.get('permalink','')}",
                })
            time.sleep(0.3)
        except Exception as exc:
            log.debug("Reddit fetch error r/%s: %s", sub, exc)
            continue

    # Score sentiment based on post scores (upvotes) and keywords
    from data.sources.news import score_sentiment
    if all_posts:
        scores = []
        for p in all_posts:
            s = score_sentiment(p["title"])
            weight = max(1, p["score"])  # weight by upvotes
            scores.append(s["score"] * weight)
        total_w = sum(max(1, p["score"]) for p in all_posts)
        avg     = sum(scores) / total_w if total_w > 0 else 0
        label   = "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral"
    else:
        avg, label = 0.0, "neutral"

    return {
        "symbol":           symbol,
        "mentions":         len(all_posts),
        "sentiment_score":  round(avg, 3),
        "sentiment_label":  label,
        "hours_back":       hours_back,
        "top_posts":        sorted(all_posts, key=lambda x: x["score"], reverse=True)[:5],
    }


# ── Google Trends ─────────────────────────────────────────────────────────────

def fetch_google_trends(
    symbol: str,
    timeframe: str = "today 3-m",
    geo:       str = "US",
) -> dict[str, Any]:
    """Fetch Google Trends search interest for a ticker.

    Search interest is a powerful retail attention signal — spikes often
    precede retail-driven price moves.

    Args:
        symbol:    Ticker symbol.
        timeframe: pytrends timeframe string (e.g. 'today 3-m', 'today 12-m').
        geo:       Geographic region (e.g. 'US', 'GB', '' for worldwide).

    Returns:
        Dict with: symbol, trend_data (list of {date, value}), avg_interest,
                   current_vs_avg_pct, trend (rising/falling/flat).
    """
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(5, 20))
        # Search for both ticker and company name-like query
        keywords = [symbol.upper()]
        pt.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
        df = pt.interest_over_time()
        if df.empty or symbol.upper() not in df.columns:
            return {"symbol": symbol, "error": "No Google Trends data"}

        series = df[symbol.upper()].dropna()
        values = list(series.values)
        dates  = [str(d.date()) for d in series.index]

        avg     = float(sum(values) / len(values)) if values else 0
        current = float(values[-1]) if values else 0
        prev_4w = float(sum(values[-4:])/4) if len(values) >= 4 else avg
        change  = (current - avg) / max(avg, 1) * 100

        trend = "rising" if current > prev_4w * 1.1 else "falling" if current < prev_4w * 0.9 else "flat"

        return {
            "symbol":               symbol,
            "trend_data":           [{"date": d, "value": int(v)} for d, v in zip(dates, values)],
            "avg_interest":         round(avg, 1),
            "current_interest":     round(current, 1),
            "current_vs_avg_pct":   round(change, 1),
            "trend":                trend,
            "timeframe":            timeframe,
        }
    except ImportError:
        return {"symbol": symbol, "error": "pytrends not installed"}
    except Exception as exc:
        log.warning("Google Trends error for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}


# ── StockTwits public sentiment ────────────────────────────────────────────────

def fetch_stocktwits_sentiment(symbol: str) -> dict[str, Any]:
    """Fetch StockTwits message stream and compute bullish/bearish ratio.

    StockTwits users explicitly tag messages as bullish or bearish.
    This gives a clean sentiment signal without keyword parsing.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict with: symbol, bull_pct, bear_pct, total_messages, label.
    """
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
            headers=_HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return {"symbol": symbol, "error": f"HTTP {r.status_code}"}
        messages = r.json().get("messages", [])
        if not messages:
            return {"symbol": symbol, "error": "No messages"}

        bull  = sum(1 for m in messages if (m.get("entities", {}).get("sentiment", {}) or {}).get("basic") == "Bullish")
        bear  = sum(1 for m in messages if (m.get("entities", {}).get("sentiment", {}) or {}).get("basic") == "Bearish")
        total = len(messages)
        bull_pct = round(bull / total * 100, 1) if total else 50
        bear_pct = round(bear / total * 100, 1) if total else 50
        label    = "bullish" if bull > bear * 1.3 else "bearish" if bear > bull * 1.3 else "neutral"

        return {
            "symbol":         symbol,
            "bull_pct":       bull_pct,
            "bear_pct":       bear_pct,
            "total_messages": total,
            "label":          label,
            "tagged_pct":     round((bull + bear) / total * 100, 1) if total else 0,
        }
    except Exception as exc:
        log.warning("StockTwits error for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)}


# ── CBOE options flow ─────────────────────────────────────────────────────────

def fetch_cboe_put_call_ratio() -> dict[str, Any]:
    """Fetch current and historical CBOE equity put/call ratio.

    The put/call ratio is a contrarian sentiment indicator:
    - Ratio > 1.0: excessive fear → potential bottom
    - Ratio < 0.6: excessive complacency → potential top

    Returns:
        Dict with: equity_pc_ratio, total_pc_ratio, interpretation.
    """
    try:
        # CBOE provides delayed data at no cost
        r = requests.get(
            "https://cdn.cboe.com/data/us/options/market_statistics/daily_market_statistics.json",
            headers=_HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        data = r.json()

        equity_ratio = data.get("equity_put_call_ratio")
        total_ratio  = data.get("total_put_call_ratio")

        if equity_ratio:
            er = float(equity_ratio)
            interpretation = (
                "extreme_fear" if er > 1.0
                else "fear" if er > 0.8
                else "neutral" if er > 0.65
                else "complacency" if er > 0.55
                else "extreme_complacency"
            )
        else:
            er = None; interpretation = "unknown"

        return {
            "equity_pc_ratio": er,
            "total_pc_ratio":  float(total_ratio) if total_ratio else None,
            "interpretation":  interpretation,
            "contrarian_signal": (
                "bullish" if (er or 0) > 0.9
                else "bearish" if (er or 1) < 0.55
                else "neutral"
            ),
        }
    except Exception as exc:
        log.debug("CBOE put/call fetch failed: %s", exc)
        return {"error": str(exc)}


# ── Composite social intelligence ─────────────────────────────────────────────

def fetch_social_intelligence(
    symbol: str,
    include_trends: bool = True,
) -> dict[str, Any]:
    """Aggregate all social/alternative data for a symbol.

    Args:
        symbol:         Ticker symbol.
        include_trends: Whether to fetch Google Trends (adds ~2s).

    Returns:
        Comprehensive social intelligence dict.
    """
    result: dict[str, Any] = {
        "symbol":    symbol,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Reddit
    result["reddit"]      = fetch_reddit_sentiment(symbol, hours_back=48, max_posts=30)

    # StockTwits
    result["stocktwits"]  = fetch_stocktwits_sentiment(symbol)

    # Google Trends (optional, slower)
    if include_trends:
        result["google_trends"] = fetch_google_trends(symbol, timeframe="today 1-m")

    # CBOE put/call (market-wide)
    result["put_call"] = fetch_cboe_put_call_ratio()

    # Composite score
    scores = []
    reddit_label = result["reddit"].get("sentiment_label", "neutral")
    st_label     = result["stocktwits"].get("label", "neutral")
    for label in [reddit_label, st_label]:
        scores.append(1.0 if label == "bullish" else -1.0 if label == "bearish" else 0.0)

    avg  = sum(scores) / max(len(scores), 1)
    comp = "bullish" if avg > 0.2 else "bearish" if avg < -0.2 else "neutral"

    result["composite"] = {
        "label": comp,
        "score": round(avg, 3),
        "reddit_label":     reddit_label,
        "stocktwits_label": st_label,
        "reddit_mentions":  result["reddit"].get("mentions", 0),
    }

    return result
