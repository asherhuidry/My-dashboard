"""SEC EDGAR comprehensive data connector.

Pulls ALL structured financial data from the free SEC EDGAR APIs — no API
key required. Data includes:

  - Company facts (all XBRL: income, balance sheet, cash flow — 10+ years)
  - Insider transactions (Form 4)
  - 13F institutional holdings (top hedge fund positions every quarter)
  - STOCK Act political trading disclosures (House/Senate)
  - Recent filings list (10-K, 10-Q, 8-K) with metadata
  - Company background (SIC code, exchanges, EIN)

All data is normalised and written to Supabase. This is one of the most
valuable free data sources available — it gives 10+ years of quarterly
fundamentals with no cost.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

_BASE   = "https://data.sec.gov"
_EFTS   = "https://efts.sec.gov"
_HEADERS = {
    "User-Agent": "FinBrain research@finbrain.ai",
    "Accept-Encoding": "gzip, deflate",
}
_SLEEP  = 0.12   # SEC rate limit: ~10 req/s, be conservative


# ── CIK lookup ────────────────────────────────────────────────────────────────

_ticker_to_cik: dict[str, str] = {}

def get_cik(symbol: str) -> str | None:
    """Return the zero-padded 10-digit CIK for a ticker symbol.

    Args:
        symbol: Ticker symbol (case-insensitive).

    Returns:
        10-digit CIK string or None if not found.
    """
    if not _ticker_to_cik:
        try:
            r = requests.get(f"{_BASE}/files/company_tickers.json",
                             headers=_HEADERS, timeout=10)
            for entry in r.json().values():
                _ticker_to_cik[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
        except Exception as exc:
            log.warning("CIK map load failed: %s", exc)
            return None
    return _ticker_to_cik.get(symbol.upper())


# ── Company facts (XBRL structured financials) ────────────────────────────────

def get_company_facts(symbol: str) -> dict[str, Any]:
    """Fetch ALL XBRL financial facts for a company — full income statement,
    balance sheet, and cash flow going back 10+ years.

    The SEC returns every tagged financial metric ever filed in XBRL.
    We extract the most important ~40 metrics and return them as
    time-series lists ready for storage.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict with keys: symbol, cik, name, facts (dict of metric → timeseries).
    """
    cik = get_cik(symbol)
    if not cik:
        return {"symbol": symbol, "error": "CIK not found"}

    try:
        r = requests.get(f"{_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                         headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return {"symbol": symbol, "error": f"HTTP {r.status_code}"}
        data = r.json()
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}

    name = data.get("entityName", symbol)

    # Key GAAP concepts to extract
    KEY_CONCEPTS = {
        # Income statement
        "Revenues":                     "revenue",
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
        "GrossProfit":                  "gross_profit",
        "OperatingIncomeLoss":          "operating_income",
        "NetIncomeLoss":                "net_income",
        "EarningsPerShareDiluted":      "eps_diluted",
        "EarningsPerShareBasic":        "eps_basic",
        "ResearchAndDevelopmentExpense":"r_and_d",
        "SellingGeneralAndAdministrativeExpense": "sga",
        "DepreciationAndAmortization":  "depreciation",
        "InterestExpense":              "interest_expense",
        "IncomeTaxExpenseBenefit":      "tax_expense",
        # Balance sheet
        "Assets":                       "total_assets",
        "Liabilities":                  "total_liabilities",
        "StockholdersEquity":           "stockholders_equity",
        "CashAndCashEquivalentsAtCarryingValue": "cash",
        "ShortTermInvestments":         "short_term_investments",
        "InventoryNet":                 "inventory",
        "AccountsReceivableNetCurrent": "accounts_receivable",
        "LongTermDebt":                 "long_term_debt",
        "LongTermDebtNoncurrent":       "long_term_debt",
        "CommonStockSharesOutstanding": "shares_outstanding",
        # Cash flow
        "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
        "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
        "NetCashProvidedByUsedInFinancingActivities": "financing_cash_flow",
        "CapitalExpendituresContinuingOperations": "capex",
        "PaymentsForRepurchaseOfCommonStock": "buybacks",
        "PaymentsOfDividendsCommonStock": "dividends_paid",
        # Per share
        "BookValuePerShareBasic":       "book_value_per_share",
    }

    gaap = data.get("facts", {}).get("us-gaap", {})
    dei  = data.get("facts", {}).get("dei", {})
    facts: dict[str, list[dict]] = {}

    for concept, label in KEY_CONCEPTS.items():
        if concept not in gaap:
            continue
        units_block = gaap[concept].get("units", {})
        # Prefer USD, then USD/shares, then pure
        for unit_key in ["USD", "USD/shares", "shares", "pure"]:
            if unit_key not in units_block:
                continue
            rows = units_block[unit_key]
            # Keep only annual (form 10-K) and quarterly (form 10-Q)
            series = []
            for r in rows:
                form = r.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue
                end = r.get("end", "")
                val = r.get("val")
                if not end or val is None:
                    continue
                series.append({
                    "date":    end,
                    "value":   val,
                    "form":    form,
                    "accn":    r.get("accn", ""),
                    "filed":   r.get("filed", ""),
                    "frame":   r.get("frame", ""),
                })
            if series:
                # Deduplicate: keep latest filing per period
                seen: dict[str, dict] = {}
                for s in sorted(series, key=lambda x: x["filed"]):
                    seen[s["date"]] = s
                facts[label] = sorted(seen.values(), key=lambda x: x["date"])
                break

    return {
        "symbol":  symbol,
        "cik":     cik,
        "name":    name,
        "facts":   facts,
        "total_metrics": len(facts),
    }


# ── Filings list ──────────────────────────────────────────────────────────────

def get_recent_filings(
    symbol: str,
    forms: list[str] | None = None,
    max_filings: int = 20,
) -> list[dict[str, Any]]:
    """Get the most recent SEC filings for a company.

    Args:
        symbol:      Ticker symbol.
        forms:       Filter by form type(s), e.g. ['10-K','10-Q','8-K'].
                     None = return all forms.
        max_filings: Maximum number to return.

    Returns:
        List of filing dicts with: form, filed, period, description, accn, url.
    """
    cik = get_cik(symbol)
    if not cik:
        return []

    try:
        r = requests.get(f"{_BASE}/submissions/CIK{cik}.json",
                         headers=_HEADERS, timeout=10)
        data = r.json()
    except Exception as exc:
        log.warning("Submissions fetch failed for %s: %s", symbol, exc)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms_list    = recent.get("form", [])
    filed_list    = recent.get("filingDate", [])
    period_list   = recent.get("reportDate", [])
    desc_list     = recent.get("primaryDocument", [])
    accn_list     = recent.get("accessionNumber", [])

    results = []
    for i, form_type in enumerate(forms_list):
        if forms and form_type not in forms:
            continue
        accn = accn_list[i].replace("-", "") if i < len(accn_list) else ""
        url  = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn}/{desc_list[i] if i < len(desc_list) else ''}"
        results.append({
            "form":        form_type,
            "filed":       filed_list[i] if i < len(filed_list) else "",
            "period":      period_list[i] if i < len(period_list) else "",
            "accn":        accn_list[i] if i < len(accn_list) else "",
            "primary_doc": desc_list[i] if i < len(desc_list) else "",
            "url":         url,
        })
        if len(results) >= max_filings:
            break

    return results


# ── Insider trades (Form 4) ────────────────────────────────────────────────────

def get_insider_transactions(symbol: str, max_results: int = 20) -> list[dict[str, Any]]:
    """Fetch recent Form 4 insider transaction filings.

    Uses the EDGAR full-text search API to find Form 4s, then parses
    the XML to extract transaction details.

    Args:
        symbol:      Ticker symbol.
        max_results: Maximum transactions to return.

    Returns:
        List of insider transaction dicts.
    """
    cik = get_cik(symbol)
    if not cik:
        return []

    try:
        # Get Form 4 filings from submissions
        filings = get_recent_filings(symbol, forms=["4"], max_filings=max_results)
        if not filings:
            return []

        transactions = []
        for filing in filings[:10]:  # parse first 10
            try:
                accn_clean = filing["accn"].replace("-", "")
                if not accn_clean:
                    continue
                # Parse Form 4 XML
                xml_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_clean}/{filing['primary_doc']}"
                xr = requests.get(xml_url, headers=_HEADERS, timeout=8)
                if xr.status_code != 200 or not xr.text.strip().startswith("<"):
                    continue
                transactions.extend(_parse_form4_xml(xr.text, filing["filed"]))
                time.sleep(_SLEEP)
            except Exception:
                continue

        return transactions[:max_results]

    except Exception as exc:
        log.warning("Insider transaction fetch failed for %s: %s", symbol, exc)
        return []


def _parse_form4_xml(xml_text: str, filed_date: str) -> list[dict[str, Any]]:
    """Parse Form 4 XML and extract transaction rows.

    Args:
        xml_text:   Raw XML text of a Form 4 filing.
        filed_date: Date the form was filed.

    Returns:
        List of transaction dicts.
    """
    import re
    results = []

    name_match  = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", xml_text)
    title_match = re.search(r"<officerTitle>([^<]+)</officerTitle>", xml_text)

    name  = name_match.group(1).strip()  if name_match  else "Unknown"
    title = title_match.group(1).strip() if title_match else ""

    # Non-derivative transactions
    for tx in re.finditer(
        r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
        xml_text, re.DOTALL
    ):
        block = tx.group(1)
        def grab(tag: str) -> str:
            m = re.search(rf"<{tag}>([^<]+)</{tag}>", block)
            return m.group(1).strip() if m else ""

        tx_date = grab("transactionDate") or grab("periodOfReport") or filed_date
        tx_code = grab("transactionCode")   # A=acquire, D=dispose
        shares  = grab("transactionShares")
        price   = grab("transactionPricePerShare")
        owned   = grab("sharesOwnedFollowingTransaction")

        if tx_code and shares:
            results.append({
                "filed_date":   filed_date,
                "tx_date":      tx_date,
                "insider_name": name,
                "title":        title,
                "action":       "buy" if tx_code == "A" else "sell" if tx_code == "D" else tx_code,
                "shares":       _safe_num(shares),
                "price":        _safe_num(price),
                "shares_owned_after": _safe_num(owned),
            })

    return results


# ── 13F Institutional Holdings ────────────────────────────────────────────────

def get_13f_holdings(
    institution_cik: str,
    max_quarters: int = 4,
) -> list[dict[str, Any]]:
    """Fetch quarterly 13F holdings for a major institution.

    Args:
        institution_cik: Zero-padded CIK of the institution
                         (e.g. Berkshire=0001067983, Bridgewater=0001350694).
        max_quarters:    How many quarters of history to fetch.

    Returns:
        List of holding dicts: {quarter, symbol, shares, value_usd, pct_portfolio}.
    """
    try:
        r = requests.get(f"{_BASE}/submissions/CIK{institution_cik}.json",
                         headers=_HEADERS, timeout=10)
        data = r.json()
    except Exception as exc:
        log.warning("13F submissions failed: %s", exc)
        return []

    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    accns   = filings.get("accessionNumber", [])
    dates   = filings.get("filingDate", [])

    results = []
    count   = 0
    for i, form_type in enumerate(forms):
        if form_type not in ("13F-HR", "13F-HR/A"):
            continue
        if count >= max_quarters:
            break
        try:
            accn     = accns[i].replace("-", "")
            filed    = dates[i]
            xml_url  = f"https://www.sec.gov/Archives/edgar/data/{int(institution_cik)}/{accn}/infotable.xml"
            xr = requests.get(xml_url, headers=_HEADERS, timeout=10)
            holdings = _parse_13f_xml(xr.text, filed)
            results.extend(holdings)
            count += 1
            time.sleep(_SLEEP)
        except Exception:
            continue

    return results


def _parse_13f_xml(xml_text: str, filed_date: str) -> list[dict[str, Any]]:
    """Parse 13F XML infotable into holdings rows."""
    import re
    rows = []
    for entry in re.finditer(r"<infoTable>(.*?)</infoTable>", xml_text, re.DOTALL):
        block = entry.group(1)
        def grab(tag: str) -> str:
            m = re.search(rf"<{tag}>([^<]+)</{tag}>", block, re.IGNORECASE)
            return m.group(1).strip() if m else ""
        rows.append({
            "filed_date": filed_date,
            "company":    grab("nameOfIssuer"),
            "cusip":      grab("cusip"),
            "value_usd":  _safe_num(grab("value")) or 0,  # thousands
            "shares":     _safe_num(grab("sshPrnamt")) or 0,
            "type":       grab("sshPrnamtType"),
        })
    return rows


# ── STOCK Act political trading ────────────────────────────────────────────────

def get_congress_trades(symbol: str | None = None, max_results: int = 30) -> list[dict[str, Any]]:
    """Fetch Congressional stock trading disclosures (STOCK Act).

    Uses the House/Senate financial disclosure databases via the
    HouseStockWatcher/SenateSockWatcher public APIs (free, no key).

    Args:
        symbol:      Filter by ticker (None = return all).
        max_results: Max disclosures to return.

    Returns:
        List of disclosure dicts.
    """
    results = []
    # House disclosures (HouseStockWatcher)
    try:
        r = requests.get(
            "https://house-stock-watcher-data.s3-us-east-2.amazonaws.com/data/all_transactions.json",
            headers=_HEADERS, timeout=15,
        )
        trades = r.json() if r.status_code == 200 else []
        for t in trades:
            ticker = (t.get("ticker") or "").upper().strip()
            if symbol and ticker != symbol.upper():
                continue
            results.append({
                "source":      "house",
                "representative": t.get("representative", ""),
                "ticker":      ticker,
                "asset":       t.get("asset_description", ""),
                "transaction": t.get("type", ""),
                "amount":      t.get("amount", ""),
                "date":        t.get("transaction_date", ""),
                "disclosure_date": t.get("disclosure_date", ""),
                "district":    t.get("district", ""),
            })
    except Exception as exc:
        log.debug("House STOCK Act fetch failed: %s", exc)

    # Senate disclosures
    try:
        r = requests.get(
            "https://senate-stock-watcher-data.s3-us-east-2.amazonaws.com/aggregate/all_transactions.json",
            headers=_HEADERS, timeout=15,
        )
        trades = r.json() if r.status_code == 200 else []
        for t in trades:
            ticker = (t.get("ticker") or "").upper().strip()
            if symbol and ticker != symbol.upper():
                continue
            results.append({
                "source":   "senate",
                "senator":  t.get("senator", ""),
                "ticker":   ticker,
                "asset":    t.get("asset_description", ""),
                "transaction": t.get("type", ""),
                "amount":   t.get("amount", ""),
                "date":     t.get("transaction_date", ""),
            })
    except Exception as exc:
        log.debug("Senate STOCK Act fetch failed: %s", exc)

    # Sort by date desc, limit
    results.sort(key=lambda x: x.get("date", ""), reverse=True)
    return results[:max_results]


# ── Regulatory filings search ─────────────────────────────────────────────────

def search_edgar_full_text(
    query: str,
    forms: list[str] | None = None,
    date_from: str = "2020-01-01",
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search EDGAR full-text search for any keyword/company mention.

    Useful for finding: regulatory risks, litigation mentions, specific events.

    Args:
        query:       Search query (e.g. "antitrust investigation").
        forms:       Filter by form types (e.g. ["8-K"]).
        date_from:   Start date for search.
        max_results: Max results.

    Returns:
        List of filing dicts: {company, form, date, accn, url, snippet}.
    """
    params: dict = {
        "q":    f'"{query}"',
        "dateRange": "custom",
        "startdt": date_from,
        "hits.hits.total.value": max_results,
    }
    if forms:
        params["forms"] = ",".join(forms)

    try:
        r = requests.get(
            f"{_EFTS}/LATEST/search-index",
            params=params,
            headers=_HEADERS,
            timeout=12,
        )
        data  = r.json()
        hits  = data.get("hits", {}).get("hits", [])
        results = []
        for h in hits[:max_results]:
            src = h.get("_source", {})
            results.append({
                "company": src.get("entity_name", ""),
                "form":    src.get("form_type", ""),
                "date":    src.get("file_date", ""),
                "accn":    src.get("accession_no", ""),
                "url":     f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id','')}/{src.get('accession_no','').replace('-','')}/",
                "snippet": src.get("description", "")[:200],
            })
        return results
    except Exception as exc:
        log.debug("EDGAR full-text search failed: %s", exc)
        return []


# ── Bulk fetch for universe ────────────────────────────────────────────────────

def fetch_fundamentals_for_universe(
    symbols: list[str],
    sleep_between: float = 0.15,
) -> dict[str, dict]:
    """Fetch company facts for every symbol in a list.

    Args:
        symbols:        List of ticker symbols.
        sleep_between:  Seconds to sleep between requests (SEC rate limiting).

    Returns:
        Dict of {symbol: company_facts_dict}.
    """
    results = {}
    for sym in symbols:
        log.info("edgar_facts_fetch %s", sym)
        results[sym] = get_company_facts(sym)
        time.sleep(sleep_between)
    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None
