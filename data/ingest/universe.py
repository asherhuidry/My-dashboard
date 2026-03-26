"""FinBrain asset universe definitions.

Defines the full set of assets to ingest, organised by asset class.
All connectors import from here so the universe is defined in one place.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Top 50 US stocks by S&P 500 market cap weighting
# ─────────────────────────────────────────────────────────────────────────────
EQUITIES: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B",
    "LLY",  "AVGO", "JPM",  "TSLA", "UNH",  "V",     "XOM",  "MA",
    "JNJ",  "PG",   "HD",   "COST", "MRK",  "ABBV",  "CVX",  "BAC",
    "KO",   "PEP",  "ADBE", "WMT",  "CRM",  "MCD",   "ACN",  "TMO",
    "ORCL", "AMD",  "LIN",  "CSCO", "ABT",  "DHR",   "NKE",  "INTC",
    "DIS",  "PM",   "TXN",  "VZ",   "NEE",  "RTX",   "AMGN", "MS",
    "UPS",  "INTU",
]

# ─────────────────────────────────────────────────────────────────────────────
# Top 20 crypto (CoinGecko coin_id → ticker)
# ─────────────────────────────────────────────────────────────────────────────
CRYPTO: list[tuple[str, str]] = [
    ("bitcoin",         "BTC"),
    ("ethereum",        "ETH"),
    ("tether",          "USDT"),
    ("binancecoin",     "BNB"),
    ("solana",          "SOL"),
    ("usd-coin",        "USDC"),
    ("ripple",          "XRP"),
    ("staked-ether",    "STETH"),
    ("dogecoin",        "DOGE"),
    ("tron",            "TRX"),
    ("the-open-network","TON"),
    ("cardano",         "ADA"),
    ("avalanche-2",     "AVAX"),
    ("shiba-inu",       "SHIB"),
    ("wrapped-bitcoin", "WBTC"),
    ("bitcoin-cash",    "BCH"),
    ("chainlink",       "LINK"),
    ("polkadot",        "DOT"),
    ("near",            "NEAR"),
    ("litecoin",        "LTC"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Major forex pairs (Yahoo Finance format: BASE/QUOTE=X)
# ─────────────────────────────────────────────────────────────────────────────
FOREX: list[str] = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
]

# ─────────────────────────────────────────────────────────────────────────────
# Commodities (Yahoo Finance futures tickers)
# ─────────────────────────────────────────────────────────────────────────────
COMMODITIES: list[str] = [
    "GC=F",   # Gold
    "SI=F",   # Silver
    "CL=F",   # WTI Crude Oil
    "NG=F",   # Natural Gas
    "HG=F",   # Copper
]

# ─────────────────────────────────────────────────────────────────────────────
# FRED macro series: (series_id, name, frequency)
# ─────────────────────────────────────────────────────────────────────────────
MACRO_SERIES: list[tuple[str, str, str]] = [
    ("GDP",      "Gross Domestic Product",              "quarterly"),
    ("CPIAUCSL", "Consumer Price Index (All Urban)",    "monthly"),
    ("FEDFUNDS", "Federal Funds Effective Rate",        "monthly"),
    ("UNRATE",   "Unemployment Rate",                   "monthly"),
    ("M2SL",     "M2 Money Supply",                    "monthly"),
    ("GS10",     "10-Year Treasury Constant Maturity", "daily"),
    ("GS2",      "2-Year Treasury Constant Maturity",  "daily"),
    ("VIXCLS",   "CBOE Volatility Index (VIX)",         "daily"),
    ("RSAFS",    "Retail Sales",                        "monthly"),
    ("MANEMP",   "ISM Manufacturing Employment",        "monthly"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Combined yfinance ticker map (ticker → asset_class)
# ─────────────────────────────────────────────────────────────────────────────

def get_yfinance_universe() -> dict[str, str]:
    """Return a flat {ticker: asset_class} dict for all yfinance-sourced assets.

    Returns:
        Dict mapping ticker symbol to asset_class string.
    """
    universe: dict[str, str] = {}
    for ticker in EQUITIES:
        universe[ticker] = "equity"
    for ticker in FOREX:
        universe[ticker] = "forex"
    for ticker in COMMODITIES:
        universe[ticker] = "commodity"
    return universe
