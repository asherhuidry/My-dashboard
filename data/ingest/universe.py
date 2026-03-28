"""FinBrain asset universe — expanded to 200+ assets, 65+ macro series.

Every data connector imports from here so the universe lives in one place.
"""
from __future__ import annotations

# ── Equities: S&P 500 top 80 + sector leaders ─────────────────────────────────
EQUITIES: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "ADBE", "CRM",  "ORCL", "AMD",  "INTC", "QCOM",  "AVGO", "TXN",
    "MU",   "AMAT", "LRCX", "KLAC",
    # Financials
    "JPM",  "BAC",  "WFC",  "GS",   "MS",   "BLK",   "C",    "AXP",
    "V",    "MA",   "COF",  "SCHW", "USB",  "BX",    "KKR",
    # Healthcare & Biotech
    "JNJ",  "UNH",  "LLY",  "ABT",  "TMO",  "MRK",   "ABBV", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA",  "PFE",
    # Consumer
    "WMT",  "COST", "HD",   "TGT",  "LOW",  "NKE",   "MCD",  "SBUX",
    "PG",   "KO",   "PEP",  "PM",   "MO",   "CL",
    # Industrials & Defense
    "LMT",  "RTX",  "NOC",  "GD",   "BA",   "CAT",   "DE",   "HON",
    "UPS",  "FDX",  "GE",   "MMM",
    # Energy
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC",   "VLO",  "PSX",
    # Real Estate / Infrastructure
    "AMT",  "PLD",  "EQIX", "CCI",  "SPG",
    # Diversified
    "BRK-B","LIN",  "APD",  "ECL",  "INTU", "DHR",   "NEE",
]

# ── Crypto: top 20 by market cap (Yahoo Finance format) ───────────────────────
CRYPTO_YF: list[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD",  "SOL-USD",  "XRP-USD",
    "ADA-USD", "AVAX-USD","DOGE-USD", "DOT-USD",  "MATIC-USD",
    "LINK-USD","ATOM-USD","LTC-USD",  "BCH-USD",  "NEAR-USD",
    "UNI-USD", "ICP-USD", "FIL-USD",  "APT-USD",  "ARB-USD",
]

# ── Crypto: CoinGecko format — (coin_id, ticker) tuples ─────────────────────
CRYPTO: list[tuple[str, str]] = [
    ("bitcoin",      "BTC"),  ("ethereum",     "ETH"),  ("binancecoin",  "BNB"),
    ("solana",       "SOL"),  ("ripple",       "XRP"),  ("cardano",      "ADA"),
    ("avalanche-2",  "AVAX"), ("dogecoin",     "DOGE"), ("polkadot",     "DOT"),
    ("matic-network","MATIC"),("chainlink",    "LINK"), ("cosmos",       "ATOM"),
    ("litecoin",     "LTC"),  ("bitcoin-cash", "BCH"),  ("near",         "NEAR"),
    ("uniswap",      "UNI"),  ("internet-computer","ICP"),("filecoin",   "FIL"),
    ("aptos",        "APT"),  ("arbitrum",     "ARB"),
]

# ── ETFs: sector, factor, volatility, rates, international ────────────────────
ETFS: list[str] = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "MDY", "IJR",
    # Sector SPDRs
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC",
    # Factor ETFs
    "MTUM", "VLUE", "QUAL", "USMV", "IWF", "IWD", "SIZE",
    # Fixed income
    "AGG", "BND", "TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "BKLN",
    # Volatility & alternatives
    "GLD", "SLV", "USO", "UNG", "DBC", "PDBC",
    # International
    "EEM", "EFA", "FXI", "EWJ", "EWZ", "MCHI", "INDA",
    # Thematic
    "ARKK", "SOXX", "IBB", "XBI", "ICLN",
]

# ── Forex: major + emerging ────────────────────────────────────────────────────
FOREX: list[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "USDCAD=X", "USDCHF=X", "NZDUSD=X", "USDCNY=X",
    "USDINR=X", "USDMXN=X", "USDBRL=X", "USDKRW=X",
    "DX-Y.NYB",  # US Dollar Index
]

# ── Commodities: futures + spot ────────────────────────────────────────────────
COMMODITIES: list[str] = [
    "GC=F",  "SI=F",  "CL=F",  "BZ=F",  "NG=F",
    "HG=F",  "ZC=F",  "ZW=F",  "ZS=F",  "KC=F",
    "LBS=F", "PL=F",  "PA=F",  "ALI=F",
]

# ── FRED macro series (65 series) — (series_id, label, frequency) ─────────────
MACRO_SERIES: list[tuple[str, str, str]] = [
    # Interest rates & yield curve
    ("FEDFUNDS",     "Fed Funds Rate",                   "monthly"),
    ("DFF",          "Fed Funds Rate Daily",              "daily"),
    ("GS1",          "1Y Treasury Yield",                 "daily"),
    ("GS2",          "2Y Treasury Yield",                 "daily"),
    ("GS5",          "5Y Treasury Yield",                 "daily"),
    ("GS10",         "10Y Treasury Yield",                "daily"),
    ("GS30",         "30Y Treasury Yield",                "daily"),
    ("T10Y2Y",       "10Y-2Y Yield Spread",              "daily"),
    ("T10Y3M",       "10Y-3M Yield Spread",              "daily"),
    ("DFII10",       "10Y Real Treasury Yield (TIPS)",    "daily"),
    ("T10YIE",       "10Y Breakeven Inflation Rate",      "daily"),
    ("T5YIE",        "5Y Breakeven Inflation Rate",       "daily"),
    # Credit spreads
    ("BAMLH0A0HYM2", "US HY Option-Adjusted Spread",     "daily"),
    ("BAMLC0A0CM",   "US IG Corp Option-Adjusted Spread","daily"),
    # Inflation
    ("CPIAUCSL",     "CPI All Urban",                    "monthly"),
    ("CPILFESL",     "Core CPI (ex Food & Energy)",       "monthly"),
    ("PCEPI",        "PCE Price Index",                   "monthly"),
    ("PCEPILFE",     "Core PCE Price Index",              "monthly"),
    ("PPIFIS",       "PPI Final Demand",                  "monthly"),
    # GDP & Growth
    ("GDP",          "Nominal GDP",                       "quarterly"),
    ("GDPC1",        "Real GDP (Chained 2017$)",          "quarterly"),
    ("INDPRO",       "Industrial Production Index",       "monthly"),
    ("TCU",          "Total Capacity Utilization",        "monthly"),
    ("DGORDER",      "Durable Goods Orders",              "monthly"),
    # Employment
    ("UNRATE",       "Unemployment Rate",                 "monthly"),
    ("PAYEMS",       "Total Nonfarm Payrolls",            "monthly"),
    ("ICSA",         "Initial Jobless Claims",            "weekly"),
    ("CCSA",         "Continued Jobless Claims",          "weekly"),
    ("CIVPART",      "Labor Force Participation Rate",    "monthly"),
    ("JTSJOL",       "JOLTS Job Openings",                "monthly"),
    # Consumer
    ("RSAFS",        "Retail Sales",                      "monthly"),
    ("PCE",          "Personal Consumption Expenditures","monthly"),
    ("PSAVERT",      "Personal Savings Rate",             "monthly"),
    ("UMCSENT",      "Univ. Michigan Consumer Sentiment", "monthly"),
    # Housing
    ("HOUST",        "Housing Starts",                    "monthly"),
    ("PERMIT",       "Building Permits",                  "monthly"),
    ("MORTGAGE30US", "30Y Mortgage Rate",                 "weekly"),
    # Money supply & Fed balance sheet
    ("M2SL",         "M2 Money Supply",                  "monthly"),
    ("WALCL",        "Fed Balance Sheet Total Assets",    "weekly"),
    ("RRPONTSYD",    "Overnight Reverse Repos",           "daily"),
    # Volatility & Financial Stress
    ("VIXCLS",       "CBOE VIX",                         "daily"),
    ("OVXCLS",       "CBOE Crude Oil Volatility (OVX)",  "daily"),
    ("GVZCLS",       "CBOE Gold Volatility (GVZ)",       "daily"),
    ("STLFSI4",      "St. Louis Fed Financial Stress",   "weekly"),
    ("NFCI",         "Chicago Fed Financial Conditions", "weekly"),
    # Manufacturing
    ("MANEMP",       "Manufacturing Employment",         "monthly"),
    ("NEWORDER",     "Manufacturers New Orders",          "monthly"),
    # Trade & Dollar
    ("BOPGSTB",      "Trade Balance",                    "monthly"),
    ("DTWEXBGS",     "Broad USD Index",                  "daily"),
    ("DTWEXEMEGS",   "Emerging Markets USD Index",       "daily"),
    # Credit
    ("TOTALNS",      "Consumer Credit Outstanding",      "monthly"),
    ("DRSFRMACBS",   "Credit Card Delinquency Rate",     "quarterly"),
    # Energy & Commodities via FRED
    ("DCOILWTICO",   "WTI Crude Oil Spot",               "daily"),
    ("DCOILBRENTEU", "Brent Crude Oil Spot",             "daily"),
    ("DHHNGSP",      "Henry Hub Natural Gas Spot",       "daily"),
    ("GOLDAMGBD228NLBM","Gold London Fix PM",            "daily"),
    # Global
    ("GEPUCURRENT",  "Global Economic Policy Uncertainty","monthly"),
    ("USEPUINDXD",   "US Economic Policy Uncertainty",  "daily"),
]

# ── Sector mapping ────────────────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "AAPL":"Technology",     "MSFT":"Technology",    "NVDA":"Technology",
    "AMD":"Technology",      "INTC":"Technology",    "QCOM":"Technology",
    "AVGO":"Technology",     "TXN":"Technology",     "MU":"Technology",
    "AMAT":"Technology",     "LRCX":"Technology",    "KLAC":"Technology",
    "ADBE":"Technology",     "CRM":"Technology",     "ORCL":"Technology",
    "GOOGL":"Technology",    "GOOG":"Technology",    "META":"Technology",
    "INTU":"Technology",
    "JPM":"Financials",      "BAC":"Financials",     "WFC":"Financials",
    "GS":"Financials",       "MS":"Financials",      "BLK":"Financials",
    "C":"Financials",        "AXP":"Financials",     "V":"Financials",
    "MA":"Financials",       "COF":"Financials",     "BX":"Financials",
    "KKR":"Financials",
    "JNJ":"Healthcare",      "UNH":"Healthcare",     "LLY":"Healthcare",
    "ABT":"Healthcare",      "TMO":"Healthcare",     "MRK":"Healthcare",
    "ABBV":"Healthcare",     "BMY":"Healthcare",     "AMGN":"Healthcare",
    "GILD":"Healthcare",     "REGN":"Healthcare",    "VRTX":"Healthcare",
    "BIIB":"Healthcare",     "MRNA":"Healthcare",    "PFE":"Healthcare",
    "DHR":"Healthcare",
    "XOM":"Energy",          "CVX":"Energy",         "COP":"Energy",
    "SLB":"Energy",          "EOG":"Energy",         "MPC":"Energy",
    "VLO":"Energy",          "PSX":"Energy",
    "WMT":"Consumer Staples","COST":"Consumer Staples","PG":"Consumer Staples",
    "KO":"Consumer Staples", "PEP":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples", "CL":"Consumer Staples",
    "HD":"Consumer Discretionary","TGT":"Consumer Discretionary",
    "LOW":"Consumer Discretionary","NKE":"Consumer Discretionary",
    "MCD":"Consumer Discretionary","SBUX":"Consumer Discretionary",
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary",
    "LMT":"Industrials",     "RTX":"Industrials",    "NOC":"Industrials",
    "GD":"Industrials",      "BA":"Industrials",     "CAT":"Industrials",
    "DE":"Industrials",      "HON":"Industrials",    "UPS":"Industrials",
    "FDX":"Industrials",     "GE":"Industrials",     "MMM":"Industrials",
    "AMT":"Real Estate",     "PLD":"Real Estate",    "EQIX":"Real Estate",
    "CCI":"Real Estate",     "SPG":"Real Estate",
    "NEE":"Utilities",
    "LIN":"Materials",       "APD":"Materials",      "ECL":"Materials",
    "BRK-B":"Conglomerates",
}

# ── Known causal relationships to test (Granger causality) ────────────────────
KNOWN_RELATIONSHIPS: list[tuple[str, str, int, str]] = [
    ("T10Y2Y",       "SPY",        30, "yield_curve_equity"),
    ("BAMLH0A0HYM2", "SPY",        14, "credit_spread_equity"),
    ("VIXCLS",       "SPY",         1, "volatility_equity"),
    ("DFF",          "TLT",         5, "rates_bonds"),
    ("DFF",          "GLD",        10, "rates_gold"),
    ("DCOILWTICO",   "XLE",         1, "oil_energy"),
    ("DCOILWTICO",   "EURUSD=X",   3, "oil_dollar"),
    ("DCOILWTICO",   "BTC-USD",     3, "oil_crypto_risk"),
    ("DFF",          "BTC-USD",    10, "rates_crypto"),
    ("VIXCLS",       "BTC-USD",     1, "volatility_crypto"),
    ("DTWEXBGS",     "EEM",         5, "dollar_emerging"),
    ("UMCSENT",      "XLY",        21, "sentiment_consumer_disc"),
    ("WALCL",        "SPY",        14, "fed_balance_equity"),
    ("PAYEMS",       "SPY",         5, "jobs_equity"),
    ("T10YIE",       "GLD",         3, "inflation_expect_gold"),
    ("INDPRO",       "XLI",        10, "industrial_production"),
    ("HOUST",        "XLB",        15, "housing_materials"),
    ("ICSA",         "SPY",         3, "claims_equity"),
    ("STLFSI4",      "HYG",         7, "fin_stress_credit"),
    ("NFCI",         "QQQ",        10, "fin_conditions_growth"),
    ("GEPUCURRENT",  "EEM",        21, "policy_uncertainty_em"),
]

def get_yfinance_universe(extended: bool = True) -> dict[str, str]:
    """Return {ticker: asset_class} dict for all yfinance-sourced assets."""
    u: dict[str, str] = {}
    for t in EQUITIES:
        u[t] = "equity"
    for t in CRYPTO_YF:
        u[t] = "crypto"
    if extended:
        for t in ETFS:
            u[t] = "etf"
    for t in FOREX:
        u[t] = "forex"
    for t in COMMODITIES:
        u[t] = "commodity"
    return u

def get_all_symbols() -> list[str]:
    """Return every symbol across all asset classes as a flat list."""
    return list(get_yfinance_universe().keys())
