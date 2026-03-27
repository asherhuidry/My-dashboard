"""Seed the source registry with known high-value public financial data sources.

Run once to populate the registry::

    python -m data.registry.seed_sources

Or call ``seed(registry)`` programmatically.
All seeded sources start with status=APPROVED because they are proven
connectors with existing code in the repo.  Newly discovered sources
start as DISCOVERED.
"""
from __future__ import annotations

from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus


SEED_SOURCES: list[SourceRecord] = [
    # ── Price / market data ───────────────────────────────────────────────────
    SourceRecord(
        source_id          = "yfinance",
        name               = "Yahoo Finance (yfinance)",
        category           = "price",
        url                = "https://finance.yahoo.com",
        acquisition_method = "sdk",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "~2,000 req/hour; 1 req/sec recommended",
        update_frequency   = "realtime",
        asset_classes      = ["equity", "etf", "crypto", "forex", "commodity"],
        data_types         = ["ohlcv", "fundamentals", "news", "options"],
        reliability_score  = 0.90,
        status             = SourceStatus.APPROVED,
        notes              = "Primary OHLCV source. Free. No key required. "
                             "Implemented in data/ingest/yfinance_connector.py.",
    ),
    SourceRecord(
        source_id          = "coingecko",
        name               = "CoinGecko Public API",
        category           = "price",
        url                = "https://api.coingecko.com/api/v3",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "10–30 req/min on free tier; Demo key raises to 30 req/min",
        update_frequency   = "realtime",
        asset_classes      = ["crypto"],
        data_types         = ["ohlcv", "market_cap", "volume"],
        reliability_score  = 0.88,
        status             = SourceStatus.APPROVED,
        notes              = "Primary crypto OHLCV source. "
                             "Implemented in data/ingest/coingecko_connector.py.",
    ),

    # ── Macro / economic data ─────────────────────────────────────────────────
    SourceRecord(
        source_id          = "fred_api",
        name               = "FRED — Federal Reserve Economic Data",
        category           = "macro",
        url                = "https://api.stlouisfed.org/fred",
        acquisition_method = "api",
        auth_required      = True,
        free_tier          = True,
        rate_limit_notes   = "120 req/min; API key required (free registration)",
        update_frequency   = "daily",
        asset_classes      = ["macro"],
        data_types         = ["macro_series", "yields", "inflation", "employment"],
        reliability_score  = 0.97,
        status             = SourceStatus.APPROVED,
        notes              = "65 series defined in universe.py. "
                             "Implemented in data/ingest/fred_connector.py. "
                             "Requires FRED_API_KEY.",
    ),
    SourceRecord(
        source_id          = "world_bank_api",
        name               = "World Bank Open Data API",
        category           = "macro",
        url                = "https://api.worldbank.org/v2",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "No official limit; be respectful (~5 req/sec)",
        update_frequency   = "annual",
        asset_classes      = ["macro"],
        data_types         = ["gdp", "inflation", "debt", "trade"],
        reliability_score  = 0.92,
        status             = SourceStatus.VALIDATED,
        notes              = "Fetch helper in data/ingest/macro_expanded.py. "
                             "Covers 200+ countries. Annual frequency limits real-time use.",
    ),

    # ── Fundamental / corporate ───────────────────────────────────────────────
    SourceRecord(
        source_id          = "sec_edgar",
        name               = "SEC EDGAR — Electronic Data Gathering",
        category           = "fundamental",
        url                = "https://data.sec.gov",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "10 req/sec; User-Agent header required",
        update_frequency   = "quarterly",
        asset_classes      = ["equity"],
        data_types         = ["xbrl_financials", "filings", "insider_transactions", "13f"],
        reliability_score  = 0.95,
        status             = SourceStatus.VALIDATED,
        notes              = "Implemented in data/ingest/edgar_full.py. "
                             "Covers all US public companies. No auth key needed.",
    ),
    SourceRecord(
        source_id          = "stock_act_house",
        name               = "HouseStockWatcher — Congressional Trades",
        category           = "alternative",
        url                = "https://house-stock-watcher-data.s3-us-east-2.amazonaws.com",
        acquisition_method = "file_download",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "S3 public bucket; effectively unlimited",
        update_frequency   = "weekly",
        asset_classes      = ["equity"],
        data_types         = ["congressional_trades"],
        reliability_score  = 0.85,
        status             = SourceStatus.VALIDATED,
        notes              = "STOCK Act disclosures for House members. "
                             "Implemented in data/ingest/edgar_full.py.",
    ),
    SourceRecord(
        source_id          = "stock_act_senate",
        name               = "SenateSockWatcher — Congressional Trades",
        category           = "alternative",
        url                = "https://efts.sec.gov/LATEST/search-index?q=%22senate%22",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "Same EDGAR limits as sec_edgar",
        update_frequency   = "weekly",
        asset_classes      = ["equity"],
        data_types         = ["congressional_trades"],
        reliability_score  = 0.82,
        status             = SourceStatus.VALIDATED,
        notes              = "STOCK Act disclosures for Senate members. "
                             "Implemented in data/ingest/edgar_full.py.",
    ),

    # ── Sentiment / social ────────────────────────────────────────────────────
    SourceRecord(
        source_id          = "stocktwits_api",
        name               = "StockTwits Public Stream API",
        category           = "sentiment",
        url                = "https://api.stocktwits.com/api/2",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "200 req/hour unauthenticated",
        update_frequency   = "realtime",
        asset_classes      = ["equity", "crypto"],
        data_types         = ["social_sentiment", "bullish_bearish_tags"],
        reliability_score  = 0.75,
        status             = SourceStatus.VALIDATED,
        notes              = "Explicit bullish/bearish user tags. "
                             "Implemented in data/ingest/social_data.py.",
    ),
    SourceRecord(
        source_id          = "reddit_public_api",
        name               = "Reddit Public JSON API",
        category           = "sentiment",
        url                = "https://www.reddit.com/r/{subreddit}/search.json",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "60 req/min; User-Agent required",
        update_frequency   = "realtime",
        asset_classes      = ["equity", "crypto"],
        data_types         = ["social_sentiment", "discussion"],
        reliability_score  = 0.70,
        status             = SourceStatus.VALIDATED,
        notes              = "No OAuth required for read-only search. "
                             "Covers WSB, stocks, investing, SecurityAnalysis, options. "
                             "Implemented in data/ingest/social_data.py.",
    ),
    SourceRecord(
        source_id          = "cboe_put_call",
        name               = "CBOE Daily Market Statistics (Put/Call Ratio)",
        category           = "options",
        url                = "https://cdn.cboe.com/data/us/options/market_statistics/daily_market_statistics.json",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "CDN — effectively unlimited",
        update_frequency   = "daily",
        asset_classes      = ["equity"],
        data_types         = ["put_call_ratio", "options_flow"],
        reliability_score  = 0.90,
        status             = SourceStatus.APPROVED,
        notes              = "Equity and total put/call ratios. "
                             "Contrarian sentiment indicator. "
                             "Implemented in data/ingest/social_data.py.",
    ),

    # ── News ──────────────────────────────────────────────────────────────────
    SourceRecord(
        source_id          = "duckduckgo_search",
        name               = "DuckDuckGo Instant Answer / Search",
        category           = "news",
        url                = "https://duckduckgo.com",
        acquisition_method = "sdk",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "Informal; use duckduckgo-search package; ~60 req/min safe",
        update_frequency   = "realtime",
        asset_classes      = ["equity", "macro", "crypto"],
        data_types         = ["news_headlines", "web_search"],
        reliability_score  = 0.65,
        status             = SourceStatus.APPROVED,
        notes              = "Used in Research endpoint for live web context. "
                             "Implemented in data/sources/news.py.",
    ),
    SourceRecord(
        source_id          = "cnn_fear_greed",
        name               = "CNN Fear & Greed Index",
        category           = "sentiment",
        url                = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        acquisition_method = "api",
        auth_required      = False,
        free_tier          = True,
        rate_limit_notes   = "Undocumented production API; treat as fragile",
        update_frequency   = "daily",
        asset_classes      = ["equity"],
        data_types         = ["market_sentiment"],
        reliability_score  = 0.60,
        status             = SourceStatus.SAMPLED,
        notes              = "Composite of 7 signals. URL may change without notice. "
                             "Implemented in data/sources/news.py.",
    ),

    # ── Volatility ────────────────────────────────────────────────────────────
    SourceRecord(
        source_id          = "cboe_vix",
        name               = "CBOE VIX via FRED (VIXCLS)",
        category           = "volatility",
        url                = "https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS",
        acquisition_method = "api",
        auth_required      = True,
        free_tier          = True,
        rate_limit_notes   = "Same as fred_api",
        update_frequency   = "daily",
        asset_classes      = ["equity"],
        data_types         = ["volatility_index"],
        reliability_score  = 0.97,
        status             = SourceStatus.APPROVED,
        notes              = "Part of the 65-series FRED universe. Requires FRED_API_KEY.",
    ),
]


def seed(registry: SourceRegistry | None = None, overwrite: bool = False) -> SourceRegistry:
    """Seed the registry with the default high-value source list.

    Args:
        registry:  Registry instance to seed.  Creates a new one if None.
        overwrite: If True, overwrite existing records.

    Returns:
        The populated SourceRegistry.
    """
    if registry is None:
        registry = SourceRegistry()

    added = 0
    skipped = 0
    for src in SEED_SOURCES:
        try:
            registry.add(src, overwrite=overwrite)
            added += 1
        except ValueError:
            skipped += 1

    print(f"Seeded {added} sources ({skipped} skipped — already registered).")
    print(registry.summary())
    return registry


if __name__ == "__main__":
    seed()
