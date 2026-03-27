"""News + web-search + sentiment for FinBrain intelligence layer.

Sources:
  - yfinance .news   (stock-specific headlines, always available)
  - DuckDuckGo HTML search (free, no API key)
  - CNN Fear & Greed API  (free, no key)
  - Basic NLP sentiment scoring
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Sentiment keyword lists ────────────────────────────────────────────────────

_BULL_WORDS = {
    "beat", "beats", "surges", "soars", "jumps", "rallies", "gains", "record",
    "upgrades", "upgrade", "buy", "outperform", "strong", "bullish", "growth",
    "positive", "profit", "higher", "rises", "rose", "climbs", "breakout",
    "partnership", "deal", "revenue", "earnings beat", "raised guidance",
    "expanding", "acquisition", "approved", "launches", "breakthrough",
}
_BEAR_WORDS = {
    "misses", "miss", "falls", "drops", "plunges", "slides", "slumps", "cuts",
    "downgrade", "downgrades", "sell", "underperform", "weak", "bearish",
    "loss", "lower", "decline", "declines", "layoffs", "lawsuit", "fine",
    "recall", "bankruptcy", "fraud", "investigation", "warning", "risk",
    "missed guidance", "guidance cut", "disappointing", "slowing",
}

_HEADERS = {
    "User-Agent": "FinBrain/1.0 (financial research tool; +https://github.com/finbrain)",
}


# ── Public helpers ─────────────────────────────────────────────────────────────

def score_sentiment(text: str) -> dict[str, Any]:
    """Score sentiment of a text snippet using keyword matching.

    Args:
        text: Headline or body text to score.

    Returns:
        Dict with 'label' (bullish/bearish/neutral), 'score' (-1..1),
        'bull_hits', 'bear_hits'.
    """
    words = set(re.findall(r"\b\w+\b", text.lower()))
    bull  = len(words & _BULL_WORDS)
    bear  = len(words & _BEAR_WORDS)
    total = bull + bear
    if total == 0:
        return {"label": "neutral", "score": 0.0, "bull_hits": 0, "bear_hits": 0}
    score = round((bull - bear) / total, 3)
    label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    return {"label": label, "score": score, "bull_hits": bull, "bear_hits": bear}


def fetch_yf_news(symbol: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetch recent news for a symbol via yfinance.

    Args:
        symbol:      Ticker symbol (e.g. 'AAPL').
        max_results: Maximum number of articles to return.

    Returns:
        List of news item dicts with keys: title, url, publisher, published_at,
        summary, sentiment.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception as exc:
        log.warning("yfinance news fetch failed for %s: %s", symbol, exc)
        return []

    results = []
    for item in raw[:max_results]:
        # yfinance v0.2+ returns nested content dict
        content = item.get("content", item)
        title     = content.get("title") or item.get("title", "")
        url       = (content.get("canonicalUrl", {}).get("url") or
                     content.get("clickThroughUrl", {}).get("url") or
                     item.get("link", ""))
        publisher = (content.get("provider", {}).get("displayName") or
                     item.get("publisher", ""))
        pub_ts    = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub_ts, int):
            pub_at = datetime.fromtimestamp(pub_ts, tz=timezone.utc).isoformat()
        elif isinstance(pub_ts, str):
            pub_at = pub_ts
        else:
            pub_at = None

        summary   = content.get("summary") or content.get("description") or ""
        sentiment = score_sentiment(f"{title} {summary}")

        results.append({
            "title":        title,
            "url":          url,
            "publisher":    publisher,
            "published_at": pub_at,
            "summary":      summary[:300] if summary else "",
            "sentiment":    sentiment,
        })

    return results


def web_search(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    """Search the web for financial information using DuckDuckGo.

    Falls back to an empty list if the request fails.

    Args:
        query:       Search query string.
        max_results: Max results to return.

    Returns:
        List of dicts with keys: title, url, snippet, sentiment.
    """
    # Try duckduckgo_search package first (cleaner, more reliable)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title":    r.get("title", ""),
                "url":      r.get("href", ""),
                "snippet":  r.get("body", "")[:300],
                "sentiment": score_sentiment(
                    f"{r.get('title', '')} {r.get('body', '')}"
                ),
                "source": "ddg",
            }
            for r in raw
        ]
    except ImportError:
        pass
    except Exception as exc:
        log.warning("duckduckgo_search failed: %s", exc)

    # Fallback: DuckDuckGo Instant Answer API (topic summaries, no full results)
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            headers=_HEADERS,
            timeout=8,
        )
        data     = resp.json()
        abstract = data.get("Abstract", "")
        results  = []
        if abstract:
            results.append({
                "title":    data.get("Heading", query),
                "url":      data.get("AbstractURL", ""),
                "snippet":  abstract[:300],
                "sentiment": score_sentiment(abstract),
                "source":   "ddg_instant",
            })
        for r in data.get("RelatedTopics", [])[:max_results - 1]:
            text = r.get("Text", "")
            url  = r.get("FirstURL", "")
            if text:
                results.append({
                    "title":    text[:80],
                    "url":      url,
                    "snippet":  text[:300],
                    "sentiment": score_sentiment(text),
                    "source":   "ddg_related",
                })
        return results
    except Exception as exc:
        log.warning("DDG instant API failed: %s", exc)
        return []


def fetch_fear_greed() -> dict[str, Any] | None:
    """Fetch CNN Fear & Greed Index (free, no API key).

    Returns:
        Dict with 'value' (0-100), 'label', 'timestamp' or None on failure.
    """
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers=_HEADERS,
            timeout=6,
        )
        data  = resp.json()
        score = data.get("fear_and_greed", {}).get("score")
        rating = data.get("fear_and_greed", {}).get("rating", "")
        ts     = data.get("fear_and_greed", {}).get("timestamp", "")
        if score is None:
            return None
        return {
            "value":     round(float(score), 1),
            "label":     rating,
            "timestamp": ts,
        }
    except Exception as exc:
        log.debug("Fear & Greed fetch failed: %s", exc)
        return None


def fetch_all_news(
    symbol: str,
    include_web_search: bool = True,
    max_yf: int = 8,
    max_web: int = 4,
) -> dict[str, Any]:
    """Aggregate news from all sources for a symbol.

    Args:
        symbol:             Ticker symbol.
        include_web_search: Whether to run a web search (slower).
        max_yf:             Max yfinance articles.
        max_web:            Max web search results.

    Returns:
        Dict with 'articles', 'fear_greed', 'overall_sentiment'.
    """
    articles: list[dict] = []

    # yfinance news
    yf_news = fetch_yf_news(symbol, max_results=max_yf)
    articles.extend(yf_news)

    # Web search
    if include_web_search:
        query  = f"{symbol} stock news analysis {datetime.now().strftime('%Y')}"
        web_results = web_search(query, max_results=max_web)
        articles.extend(web_results)

    # Compute overall sentiment
    if articles:
        scores = [a["sentiment"]["score"] for a in articles]
        avg    = sum(scores) / len(scores)
        label  = "bullish" if avg > 0.05 else "bearish" if avg < -0.05 else "neutral"
    else:
        avg, label = 0.0, "neutral"

    return {
        "articles":          articles,
        "fear_greed":        fetch_fear_greed(),
        "overall_sentiment": {"label": label, "score": round(avg, 3)},
        "article_count":     len(articles),
    }
