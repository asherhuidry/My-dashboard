"""Knowledge Graph Builder — extracts entities and relationships from all data sources.

Builds and continuously enriches the Neo4j knowledge graph with:

  - Company → Supplier/Customer relationships (from SEC 10-K risk factors)
  - Company → Competitor relationships (mentioned in filings)
  - Company → Executive connections (shared board members)
  - Asset → Macro indicator relationships (correlation-validated)
  - Event → Price impact mappings (earnings misses, rate decisions, etc.)
  - Sector → Sector contagion links (when one sector's stress spreads)
  - Company → Industry classification (SIC codes + GICS)

The graph enables queries like:
  "What companies are exposed to TSMC as a supplier?"
  "Which stocks moved together during the 2022 rate shock?"
  "What's the typical price impact of a 25bp rate hike on XLF?"

This makes FinBrain's analysis qualitatively different from any pure
quant system — it reasons about relationships, not just numbers.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

AGENT_ID = "knowledge_builder"

# ── Industry/sector hierarchies (SIC code → GICS sector) ─────────────────────
SIC_TO_SECTOR: dict[str, str] = {
    "3674": "Technology",       # Semiconductors
    "7372": "Technology",       # Software
    "7371": "Technology",       # Computer Programming
    "3577": "Technology",       # Computer Peripherals
    "3669": "Technology",       # Communications Equipment
    "6022": "Financials",       # State commercial banks
    "6021": "Financials",       # National commercial banks
    "6153": "Financials",       # Credit institutions
    "6199": "Financials",       # Finance services
    "6211": "Financials",       # Security brokers/dealers
    "2836": "Healthcare",       # Pharmaceutical preparations
    "8000": "Healthcare",       # Health services
    "3841": "Healthcare",       # Surgical instruments
    "1311": "Energy",           # Crude petroleum and natural gas
    "1382": "Energy",           # Oil and gas field services
    "2911": "Energy",           # Petroleum refining
    "5411": "Consumer Staples", # Grocery stores
    "2000": "Consumer Staples", # Food and kindred products
    "5331": "Consumer Discretionary", # Variety stores
    "7011": "Consumer Discretionary", # Hotels and motels
    "3711": "Consumer Discretionary", # Motor vehicles
    "3714": "Consumer Discretionary", # Motor vehicle parts
    "3728": "Industrials",      # Aircraft parts
    "3812": "Industrials",      # Defense electronics
    "4911": "Utilities",        # Electric services
    "4924": "Utilities",        # Natural gas distribution
    "1500": "Real Estate",      # Building construction
    "6512": "Real Estate",      # Operators of apartment buildings
}

# ── GICS sector classification for all equities ──────────────────────────────
# Deterministic mapping: ticker → (GICS sector, GICS industry group)
# Enables BELONGS_TO edges from Asset nodes to Sector and Industry nodes.
TICKER_SECTOR_MAP: dict[str, tuple[str, str]] = {
    # ── Information Technology ──
    "AAPL": ("Information Technology", "Technology Hardware"),
    "MSFT": ("Information Technology", "Software"),
    "NVDA": ("Information Technology", "Semiconductors"),
    "ADBE": ("Information Technology", "Software"),
    "CRM":  ("Information Technology", "Software"),
    "ORCL": ("Information Technology", "Software"),
    "AMD":  ("Information Technology", "Semiconductors"),
    "INTC": ("Information Technology", "Semiconductors"),
    "QCOM": ("Information Technology", "Semiconductors"),
    "AVGO": ("Information Technology", "Semiconductors"),
    "TXN":  ("Information Technology", "Semiconductors"),
    "MU":   ("Information Technology", "Semiconductors"),
    "AMAT": ("Information Technology", "Semiconductors"),
    "LRCX": ("Information Technology", "Semiconductors"),
    "KLAC": ("Information Technology", "Semiconductors"),
    "INTU": ("Information Technology", "Software"),
    "NOW":  ("Information Technology", "Software"),
    "SNOW": ("Information Technology", "Software"),
    "PLTR": ("Information Technology", "Software"),
    "PANW": ("Information Technology", "Software"),
    "CRWD": ("Information Technology", "Software"),
    "ZS":   ("Information Technology", "Software"),
    "NET":  ("Information Technology", "Software"),
    "DDOG": ("Information Technology", "Software"),
    # ── Communication Services ──
    "GOOGL": ("Communication Services", "Interactive Media"),
    "GOOG":  ("Communication Services", "Interactive Media"),
    "META":  ("Communication Services", "Interactive Media"),
    "DIS":   ("Communication Services", "Entertainment"),
    "CMCSA": ("Communication Services", "Cable & Satellite"),
    "NFLX":  ("Communication Services", "Entertainment"),
    "T":     ("Communication Services", "Integrated Telecom"),
    "VZ":    ("Communication Services", "Integrated Telecom"),
    "TMUS":  ("Communication Services", "Wireless Telecom"),
    "CHTR":  ("Communication Services", "Cable & Satellite"),
    # ── Consumer Discretionary ──
    "AMZN": ("Consumer Discretionary", "Broadline Retail"),
    "TSLA": ("Consumer Discretionary", "Automobiles"),
    "HD":   ("Consumer Discretionary", "Home Improvement Retail"),
    "LOW":  ("Consumer Discretionary", "Home Improvement Retail"),
    "NKE":  ("Consumer Discretionary", "Apparel"),
    "MCD":  ("Consumer Discretionary", "Restaurants"),
    "SBUX": ("Consumer Discretionary", "Restaurants"),
    "TGT":  ("Consumer Discretionary", "General Merchandise"),
    "LULU": ("Consumer Discretionary", "Apparel"),
    "CMG":  ("Consumer Discretionary", "Restaurants"),
    "YUM":  ("Consumer Discretionary", "Restaurants"),
    "DPZ":  ("Consumer Discretionary", "Restaurants"),
    "SHOP": ("Consumer Discretionary", "Broadline Retail"),
    "SQ":   ("Consumer Discretionary", "Broadline Retail"),
    "UBER": ("Consumer Discretionary", "Ground Transportation"),
    "ABNB": ("Consumer Discretionary", "Hotels & Resorts"),
    # ── Consumer Staples ──
    "WMT":  ("Consumer Staples", "Consumer Staples Distribution"),
    "COST": ("Consumer Staples", "Consumer Staples Distribution"),
    "PG":   ("Consumer Staples", "Household Products"),
    "KO":   ("Consumer Staples", "Beverages"),
    "PEP":  ("Consumer Staples", "Beverages"),
    "PM":   ("Consumer Staples", "Tobacco"),
    "MO":   ("Consumer Staples", "Tobacco"),
    "CL":   ("Consumer Staples", "Household Products"),
    "EL":   ("Consumer Staples", "Personal Care Products"),
    "MNST": ("Consumer Staples", "Beverages"),
    # ── Financials ──
    "JPM":  ("Financials", "Diversified Banks"),
    "BAC":  ("Financials", "Diversified Banks"),
    "WFC":  ("Financials", "Diversified Banks"),
    "GS":   ("Financials", "Capital Markets"),
    "MS":   ("Financials", "Capital Markets"),
    "BLK":  ("Financials", "Asset Management"),
    "C":    ("Financials", "Diversified Banks"),
    "AXP":  ("Financials", "Consumer Finance"),
    "V":    ("Financials", "Transaction Processing"),
    "MA":   ("Financials", "Transaction Processing"),
    "COF":  ("Financials", "Consumer Finance"),
    "SCHW": ("Financials", "Capital Markets"),
    "USB":  ("Financials", "Regional Banks"),
    "BX":   ("Financials", "Asset Management"),
    "KKR":  ("Financials", "Asset Management"),
    "PNC":  ("Financials", "Regional Banks"),
    "TFC":  ("Financials", "Regional Banks"),
    "AIG":  ("Financials", "Insurance"),
    "MET":  ("Financials", "Insurance"),
    "PRU":  ("Financials", "Insurance"),
    "ICE":  ("Financials", "Capital Markets"),
    "CME":  ("Financials", "Capital Markets"),
    # ── Healthcare ──
    "JNJ":  ("Healthcare", "Pharmaceuticals"),
    "UNH":  ("Healthcare", "Managed Healthcare"),
    "LLY":  ("Healthcare", "Pharmaceuticals"),
    "ABT":  ("Healthcare", "Healthcare Equipment"),
    "TMO":  ("Healthcare", "Life Sciences Tools"),
    "MRK":  ("Healthcare", "Pharmaceuticals"),
    "ABBV": ("Healthcare", "Pharmaceuticals"),
    "BMY":  ("Healthcare", "Pharmaceuticals"),
    "AMGN": ("Healthcare", "Biotechnology"),
    "GILD": ("Healthcare", "Biotechnology"),
    "REGN": ("Healthcare", "Biotechnology"),
    "VRTX": ("Healthcare", "Biotechnology"),
    "BIIB": ("Healthcare", "Biotechnology"),
    "MRNA": ("Healthcare", "Biotechnology"),
    "PFE":  ("Healthcare", "Pharmaceuticals"),
    "DHR":  ("Healthcare", "Life Sciences Tools"),
    "ISRG": ("Healthcare", "Healthcare Equipment"),
    "MDT":  ("Healthcare", "Healthcare Equipment"),
    "SYK":  ("Healthcare", "Healthcare Equipment"),
    "EW":   ("Healthcare", "Healthcare Equipment"),
    "ZTS":  ("Healthcare", "Pharmaceuticals"),
    # ── Industrials ──
    "LMT":  ("Industrials", "Aerospace & Defense"),
    "RTX":  ("Industrials", "Aerospace & Defense"),
    "NOC":  ("Industrials", "Aerospace & Defense"),
    "GD":   ("Industrials", "Aerospace & Defense"),
    "BA":   ("Industrials", "Aerospace & Defense"),
    "CAT":  ("Industrials", "Construction Machinery"),
    "DE":   ("Industrials", "Agricultural Machinery"),
    "HON":  ("Industrials", "Industrial Conglomerates"),
    "UPS":  ("Industrials", "Air Freight & Logistics"),
    "FDX":  ("Industrials", "Air Freight & Logistics"),
    "GE":   ("Industrials", "Industrial Conglomerates"),
    "MMM":  ("Industrials", "Industrial Conglomerates"),
    "WM":   ("Industrials", "Environmental Services"),
    "RSG":  ("Industrials", "Environmental Services"),
    "EMR":  ("Industrials", "Electrical Equipment"),
    "ITW":  ("Industrials", "Industrial Machinery"),
    "ETN":  ("Industrials", "Electrical Equipment"),
    "ROK":  ("Industrials", "Industrial Machinery"),
    # ── Energy ──
    "XOM":  ("Energy", "Integrated Oil & Gas"),
    "CVX":  ("Energy", "Integrated Oil & Gas"),
    "COP":  ("Energy", "Exploration & Production"),
    "SLB":  ("Energy", "Oil & Gas Equipment"),
    "EOG":  ("Energy", "Exploration & Production"),
    "MPC":  ("Energy", "Oil & Gas Refining"),
    "VLO":  ("Energy", "Oil & Gas Refining"),
    "PSX":  ("Energy", "Oil & Gas Refining"),
    "OXY":  ("Energy", "Exploration & Production"),
    "DVN":  ("Energy", "Exploration & Production"),
    "HAL":  ("Energy", "Oil & Gas Equipment"),
    "FANG": ("Energy", "Exploration & Production"),
    # ── Real Estate ──
    "AMT":  ("Real Estate", "Telecom Tower REITs"),
    "PLD":  ("Real Estate", "Industrial REITs"),
    "EQIX": ("Real Estate", "Data Center REITs"),
    "CCI":  ("Real Estate", "Telecom Tower REITs"),
    "SPG":  ("Real Estate", "Retail REITs"),
    "O":    ("Real Estate", "Retail REITs"),
    "DLR":  ("Real Estate", "Data Center REITs"),
    "WELL": ("Real Estate", "Healthcare REITs"),
    "PSA":  ("Real Estate", "Self-Storage REITs"),
    # ── Utilities ──
    "NEE":  ("Utilities", "Electric Utilities"),
    "DUK":  ("Utilities", "Electric Utilities"),
    "SO":   ("Utilities", "Electric Utilities"),
    "D":    ("Utilities", "Electric Utilities"),
    "AEP":  ("Utilities", "Electric Utilities"),
    "SRE":  ("Utilities", "Multi-Utilities"),
    # ── Materials ──
    "LIN":  ("Materials", "Industrial Gases"),
    "APD":  ("Materials", "Industrial Gases"),
    "ECL":  ("Materials", "Specialty Chemicals"),
    "NEM":  ("Materials", "Gold Mining"),
    "FCX":  ("Materials", "Copper Mining"),
    "DOW":  ("Materials", "Commodity Chemicals"),
    "DD":   ("Materials", "Specialty Chemicals"),
    # ── Diversified / Conglomerates ──
    "BRK-B": ("Financials", "Multi-Sector Holdings"),
}

# ── Known supply chain relationships ─────────────────────────────────────────
# Format: (supplier_symbol, customer_symbol, product_category, revenue_pct_est)
SUPPLY_CHAIN_MAP: list[tuple[str, str, str, float | None]] = [
    # TSMC is the foundry for most fabless chips
    ("TSM",   "NVDA",  "semiconductor_manufacturing",  0.10),
    ("TSM",   "AMD",   "semiconductor_manufacturing",  0.09),
    ("TSM",   "AAPL",  "semiconductor_manufacturing",  0.25),
    ("TSM",   "QCOM",  "semiconductor_manufacturing",  0.08),
    ("TSM",   "AVGO",  "semiconductor_manufacturing",  0.06),
    # ASML makes machines that TSMC, Intel, Samsung need
    ("ASML",  "TSM",   "lithography_equipment",        None),
    ("ASML",  "INTC",  "lithography_equipment",        None),
    # AMAT/LRCX sell process equipment
    ("AMAT",  "TSM",   "process_equipment",            None),
    ("LRCX",  "TSM",   "process_equipment",            None),
    # AAPL supply chain
    ("FOXCONN","AAPL", "contract_manufacturing",       None),
    ("QCOM",  "AAPL",  "modems",                       None),
    # Cloud customers
    ("NVDA",  "MSFT",  "gpu_infrastructure",           None),
    ("NVDA",  "GOOGL", "gpu_infrastructure",           None),
    ("NVDA",  "META",  "gpu_infrastructure",           None),
    ("NVDA",  "AMZN",  "gpu_infrastructure",           None),
    # Energy
    ("SLB",   "XOM",   "oilfield_services",            None),
    ("SLB",   "CVX",   "oilfield_services",            None),
    # Healthcare
    ("TMO",   "LLY",   "research_services",            None),
    ("TMO",   "MRK",   "research_services",            None),
]

# ── Key macro → sector relationships ─────────────────────────────────────────
MACRO_SECTOR_IMPACTS: list[tuple[str, str, str, str]] = [
    # (macro_indicator, sector_etf, direction, mechanism)
    ("T10Y2Y",       "XLF", "positive", "banks profit from steeper yield curve"),
    ("T10Y2Y",       "XLU", "negative", "utilities are bond proxies, hurt by rising rates"),
    ("T10Y2Y",       "XLK", "negative", "long-duration growth stocks hurt by rising rates"),
    ("DFF",          "XLF", "positive", "higher rates boost net interest margin"),
    ("DFF",          "XLRE","negative", "higher rates hurt REITs via cap rate expansion"),
    ("CPIAUCSL",     "GLD", "positive", "gold as inflation hedge"),
    ("CPIAUCSL",     "TLT", "negative", "inflation erodes real returns on long bonds"),
    ("DCOILWTICO",   "XLE", "positive", "oil price drives energy company revenue"),
    ("DCOILWTICO",   "XLI", "negative", "higher energy costs compress industrial margins"),
    ("DCOILWTICO",   "XLY", "negative", "energy costs reduce consumer discretionary spending"),
    ("DCOILWTICO",   "EEM", "positive", "many EM economies are oil exporters"),
    ("DCOILWTICO",   "GLD", "positive", "oil shocks drive inflation → gold as hedge"),
    ("DCOILWTICO",   "TLT", "negative", "oil inflation raises rate expectations → bonds fall"),
    ("VIXCLS",       "XLV", "positive", "healthcare defensive in volatility spikes"),
    ("VIXCLS",       "XLK", "negative", "tech growth stocks hit hardest in risk-off"),
    ("VIXCLS",       "XLP", "positive", "staples are defensive in risk-off"),
    ("PAYEMS",       "XLY", "positive", "employment drives consumer spending"),
    ("UMCSENT",      "XLY", "positive", "consumer sentiment drives discretionary"),
    ("WALCL",        "QQQ", "positive", "Fed balance sheet expansion boosts risk assets"),
    ("BAMLH0A0HYM2", "HYG", "negative", "wider spreads = falling HY bond prices"),
    ("BAMLH0A0HYM2", "SPY", "negative", "credit stress leads equity stress"),
    ("DTWEXBGS",     "EEM", "negative", "stronger dollar hurts EM assets"),
    ("DTWEXBGS",     "GLD", "negative", "stronger dollar compresses gold price"),
]


# ── Entity extraction from text ────────────────────────────────────────────────

def extract_company_mentions(
    text: str,
    known_symbols: list[str] | None = None,
) -> list[str]:
    """Extract stock ticker mentions and company names from text.

    Args:
        text:          Raw text (news article, SEC filing, etc.)
        known_symbols: List of known tickers to look for.

    Returns:
        List of unique ticker symbols mentioned.
    """
    found = set()

    # $ prefixed tickers (StockTwits/Twitter style)
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text):
        found.add(m.group(1))

    # All-caps 1-5 letter sequences preceded by common words
    for m in re.finditer(
        r"\b(shares?|stock|ticker|symbol|NYSE|NASDAQ)\s+([A-Z]{1,5})\b", text
    ):
        found.add(m.group(2))

    # Filter against known symbols if provided
    if known_symbols:
        sym_set = set(s.upper() for s in known_symbols)
        found   = found & sym_set

    # Remove common false positives
    false_pos = {"CEO","CFO","COO","CTO","IPO","ETF","SEC","FED","GDP","CPI",
                 "AI","ML","US","EU","UK","NEW","THE","FOR","AND","BUT"}
    return sorted(found - false_pos)


def extract_relationships_from_text(text: str, subject_symbol: str) -> list[dict[str, Any]]:
    """Extract relationship mentions from a text block.

    Looks for patterns like "X is a supplier of Y", "X competes with Y",
    "X acquired Y", "X is a customer of Y".

    Args:
        text:           Text to parse (e.g. 10-K risk factors, news article).
        subject_symbol: The primary company's ticker.

    Returns:
        List of relationship dicts: {source, target, type, evidence}.
    """
    relationships = []
    text_lower    = text.lower()

    relationship_patterns = [
        (r"supplier[s]? (?:to|of|for)",     "supplier_of"),
        (r"customer[s]? (?:of|for)",         "customer_of"),
        (r"compet(?:es?|itor[s]?) with",     "competes_with"),
        (r"partner(?:ed|ship)? with",        "partner_of"),
        (r"acqui(?:red|ring|sition of)",     "acquired"),
        (r"joint venture with",              "joint_venture_with"),
        (r"licens(?:es?|or[s]?) (?:to|from)","licenses_to"),
        (r"manufactur(?:es?|er[s]?) for",    "manufactures_for"),
        (r"distribut(?:es?|or[s]?) for",     "distributes_for"),
    ]

    for pattern, rel_type in relationship_patterns:
        for m in re.finditer(pattern, text_lower):
            # Get surrounding context (50 chars)
            start   = max(0, m.start() - 50)
            end     = min(len(text), m.end() + 100)
            context = text[start:end]
            relationships.append({
                "source":   subject_symbol,
                "rel_type": rel_type,
                "evidence": context.strip(),
            })

    return relationships


# ── Graph writing ─────────────────────────────────────────────────────────────

def build_supply_chain_graph() -> dict[str, Any]:
    """Write the known supply chain relationships to Neo4j.

    Returns:
        Summary dict with counts of nodes and relationships created.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
    except Exception as exc:
        log.warning("Neo4j unavailable, skipping graph write: %s", exc)
        return {"error": str(exc)}

    nodes_created = 0
    rels_created  = 0

    with driver.session() as session:
        for supplier, customer, product, rev_pct in SUPPLY_CHAIN_MAP:
            try:
                # Ensure both company nodes exist
                for sym in [supplier, customer]:
                    session.run(
                        "MERGE (c:Company {ticker: $sym}) "
                        "ON CREATE SET c.created_at = $ts",
                        sym=sym, ts=datetime.now(tz=timezone.utc).isoformat()
                    )
                    nodes_created += 1

                # Create supply chain edge
                props: dict[str, Any] = {
                    "product_category": product,
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                if rev_pct is not None:
                    props["estimated_revenue_pct"] = rev_pct

                session.run(
                    "MATCH (a:Company {ticker:$sup}), (b:Company {ticker:$cus}) "
                    "MERGE (a)-[r:SUPPLIES_TO {product_category:$product}]->(b) "
                    "SET r += $props",
                    sup=supplier, cus=customer, product=product, props=props,
                )
                rels_created += 1
            except Exception as exc:
                log.debug("Graph write failed for %s→%s: %s", supplier, customer, exc)

        # Macro → Sector relationships
        for macro, etf, direction, mechanism in MACRO_SECTOR_IMPACTS:
            try:
                session.run(
                    "MERGE (m:MacroIndicator {id: $macro}) "
                    "MERGE (e:Asset {ticker: $etf}) "
                    "MERGE (m)-[r:IMPACTS {direction:$dir, mechanism:$mech}]->(e) "
                    "SET r.updated_at = $ts",
                    macro=macro, etf=etf, dir=direction, mech=mechanism,
                    ts=datetime.now(tz=timezone.utc).isoformat(),
                )
                rels_created += 1
            except Exception as exc:
                log.debug("Macro→sector write failed: %s", exc)

    result = {
        "nodes_touched": nodes_created,
        "relationships_created": rels_created,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    log.info("Knowledge graph built: %s", result)

    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id    = AGENT_ID,
            action      = "build_supply_chain_graph",
            after_state = result,
        ))
    except Exception:
        pass

    return result


def materialize_sector_classification() -> dict[str, Any]:
    """Create Sector/Industry nodes and BELONGS_TO edges from TICKER_SECTOR_MAP.

    Graph structure created:
      (Asset)-[:BELONGS_TO]->(Industry)-[:PART_OF]->(Sector)
      (Asset)-[:BELONGS_TO]->(Sector)

    Returns:
        Summary dict with counts of nodes and edges created.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
    except Exception as exc:
        log.warning("Neo4j unavailable, skipping sector classification: %s", exc)
        return {"error": str(exc)}

    sectors_created = 0
    industries_created = 0
    edges_created = 0
    ts = datetime.now(tz=timezone.utc).isoformat()

    # Collect unique sectors and industries
    sector_industries: dict[str, set[str]] = {}
    for _ticker, (sector, industry) in TICKER_SECTOR_MAP.items():
        sector_industries.setdefault(sector, set()).add(industry)

    with driver.session() as session:
        # 1. Create Sector nodes
        for sector in sector_industries:
            try:
                result = session.run(
                    "MERGE (s:Sector {name: $name}) "
                    "ON CREATE SET s.created_at = $ts, s.classification = 'GICS' "
                    "RETURN s.created_at = $ts AS is_new",
                    name=sector, ts=ts,
                )
                record = result.single()
                if record and record["is_new"]:
                    sectors_created += 1
            except Exception as exc:
                log.debug("Sector node create failed for %s: %s", sector, exc)

        # 2. Create Industry nodes + PART_OF edges to Sector
        for sector, industries in sector_industries.items():
            for industry in industries:
                try:
                    result = session.run(
                        "MERGE (i:Industry {name: $industry}) "
                        "ON CREATE SET i.created_at = $ts, i.classification = 'GICS' "
                        "WITH i "
                        "MATCH (s:Sector {name: $sector}) "
                        "MERGE (i)-[r:PART_OF]->(s) "
                        "SET r.updated_at = $ts "
                        "RETURN i.created_at = $ts AS is_new",
                        industry=industry, sector=sector, ts=ts,
                    )
                    record = result.single()
                    if record and record["is_new"]:
                        industries_created += 1
                    edges_created += 1
                except Exception as exc:
                    log.debug("Industry node create failed for %s: %s", industry, exc)

        # 3. Create Asset→Industry and Asset→Sector BELONGS_TO edges
        for ticker, (sector, industry) in TICKER_SECTOR_MAP.items():
            try:
                session.run(
                    "MERGE (a:Asset {ticker: $ticker}) "
                    "ON CREATE SET a.created_at = $ts "
                    "WITH a "
                    "MATCH (i:Industry {name: $industry}) "
                    "MATCH (s:Sector {name: $sector}) "
                    "MERGE (a)-[ri:BELONGS_TO]->(i) "
                    "SET ri.updated_at = $ts "
                    "MERGE (a)-[rs:BELONGS_TO]->(s) "
                    "SET rs.updated_at = $ts",
                    ticker=ticker, industry=industry, sector=sector, ts=ts,
                )
                edges_created += 2
            except Exception as exc:
                log.debug("BELONGS_TO edge failed for %s: %s", ticker, exc)

    result_summary = {
        "sectors_created": sectors_created,
        "industries_created": industries_created,
        "edges_created": edges_created,
        "tickers_classified": len(TICKER_SECTOR_MAP),
        "timestamp": ts,
    }
    log.info("Sector classification materialized: %s", result_summary)

    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id=AGENT_ID,
            action="materialize_sector_classification",
            after_state=result_summary,
        ))
    except Exception:
        pass

    return result_summary


def build_correlation_edges(findings: list) -> dict[str, Any]:
    """Write correlation hunter findings to Neo4j as CORRELATES_WITH edges.

    Args:
        findings: List of CorrelationFinding objects from correlation_hunter.

    Returns:
        Summary of edges written.
    """
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
    except Exception as exc:
        return {"error": str(exc)}

    written = 0
    with driver.session() as session:
        for f in findings:
            try:
                session.run(
                    "MERGE (a:Asset {ticker: $a}) "
                    "MERGE (b:Asset {ticker: $b}) "
                    "MERGE (a)-[r:CORRELATES_WITH {lag_days: $lag}]->(b) "
                    "SET r.pearson_r = $r, r.strength = $s, r.granger_p = $g, "
                    "    r.type = $t, r.updated_at = $ts",
                    a   = f.series_a,
                    b   = f.series_b,
                    lag = f.lag_days,
                    r   = f.pearson_r,
                    s   = f.strength,
                    g   = f.granger_p,
                    t   = f.relationship_type,
                    ts  = datetime.now(tz=timezone.utc).isoformat(),
                )
                written += 1
            except Exception:
                pass

    return {"edges_written": written}


# ── Run ────────────────────────────────────────────────────────────────────────

def run() -> dict[str, Any]:
    """Run full knowledge graph build: supply chain + macro + sector classification.

    Returns:
        Summary of everything written.
    """
    supply_result = build_supply_chain_graph()
    sector_result = materialize_sector_classification()

    # Also build from correlation hunter findings
    try:
        from data.agents.correlation_hunter import hunt_correlations
        from data.ingest.universe import ETFS, MACRO_SERIES
        macro_daily = [s[0] for s in MACRO_SERIES if s[2] == "daily"][:15]
        findings    = hunt_correlations(ETFS[:20], macro_series=macro_daily, max_pairs=200)
        corr_result = build_correlation_edges(findings)
    except Exception as exc:
        corr_result = {"error": str(exc)}

    return {
        "supply_chain":          supply_result,
        "sector_classification": sector_result,
        "correlations":          corr_result,
        "timestamp":             datetime.now(tz=timezone.utc).isoformat(),
    }
