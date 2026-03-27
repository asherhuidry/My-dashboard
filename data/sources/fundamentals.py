"""Fundamental data layer — earnings, insiders, options, SEC EDGAR.

Sources (all free, no additional API keys required beyond yfinance):
  - yfinance: earnings calendar, options chain, institutional holders,
               insider transactions, balance sheet, income statement
  - SEC EDGAR (data.sec.gov): filings, ownership, CIK lookups
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

_EDGAR_BASE = "https://data.sec.gov"
_EDGAR_HEADERS = {
    "User-Agent": "FinBrain research@finbrain.ai",
    "Accept-Encoding": "gzip, deflate",
}


# ── yfinance helpers ──────────────────────────────────────────────────────────

def _ticker(symbol: str):
    """Return a yfinance Ticker object."""
    import yfinance as yf
    return yf.Ticker(symbol)


def get_earnings_calendar(symbol: str) -> dict[str, Any]:
    """Fetch upcoming and recent earnings dates.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict with 'next_earnings', 'eps_history' (last 4 quarters).
    """
    try:
        t = _ticker(symbol)
        info = t.info or {}
        next_date = info.get("earningsTimestamp") or info.get("earningsDate")
        if isinstance(next_date, (int, float)):
            next_date = datetime.fromtimestamp(next_date, tz=timezone.utc).strftime("%Y-%m-%d")

        # Earnings history
        try:
            hist = t.earnings_history
            if hist is not None and not hist.empty:
                eps_rows = []
                for _, row in hist.head(4).iterrows():
                    eps_rows.append({
                        "date":         str(row.get("Earnings Date", ""))[:10],
                        "eps_estimate": _safe_float(row.get("EPS Estimate")),
                        "eps_actual":   _safe_float(row.get("Reported EPS")),
                        "surprise_pct": _safe_float(row.get("Surprise(%)")),
                    })
            else:
                eps_rows = []
        except Exception:
            eps_rows = []

        return {
            "next_earnings": next_date,
            "eps_history":   eps_rows,
            "pe_forward":    _safe_float(info.get("forwardPE")),
            "pe_trailing":   _safe_float(info.get("trailingPE")),
            "eps_forward":   _safe_float(info.get("forwardEps")),
            "eps_trailing":  _safe_float(info.get("trailingEps")),
            "peg_ratio":     _safe_float(info.get("pegRatio")),
        }
    except Exception as exc:
        log.warning("Earnings fetch failed for %s: %s", symbol, exc)
        return {}


def get_insider_trades(symbol: str, max_rows: int = 10) -> list[dict[str, Any]]:
    """Fetch recent insider trading activity.

    Args:
        symbol:   Ticker symbol.
        max_rows: Maximum insider transactions to return.

    Returns:
        List of insider trade dicts.
    """
    try:
        t = _ticker(symbol)
        insiders = t.insider_transactions
        if insiders is None or insiders.empty:
            return []
        rows = []
        for _, row in insiders.head(max_rows).iterrows():
            rows.append({
                "date":       str(row.get("Start Date", ""))[:10],
                "insider":    str(row.get("Insider", "")),
                "title":      str(row.get("Relationship", "")),
                "transaction":str(row.get("Transaction", "")),
                "shares":     int(row.get("Shares", 0) or 0),
                "value":      _safe_float(row.get("Value")),
            })
        return rows
    except Exception as exc:
        log.warning("Insider trades failed for %s: %s", symbol, exc)
        return []


def get_options_summary(symbol: str) -> dict[str, Any]:
    """Summarise options chain: put/call ratio, max pain, implied move.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict with put_call_ratio, max_pain, implied_move, expirations.
    """
    try:
        t    = _ticker(symbol)
        exps = t.options
        if not exps:
            return {}
        # Use nearest expiration
        near_exp = exps[0]
        chain    = t.option_chain(near_exp)
        calls    = chain.calls
        puts     = chain.puts

        call_vol  = calls["volume"].sum() if "volume" in calls.columns else 0
        put_vol   = puts["volume"].sum()  if "volume" in puts.columns  else 0
        pc_ratio  = round(put_vol / max(call_vol, 1), 3)

        # Implied move = avg ATM implied vol / sqrt(365/days_to_exp)
        try:
            import numpy as np
            from datetime import date
            days = max((datetime.strptime(near_exp, "%Y-%m-%d").date() - date.today()).days, 1)
            atm_iv = calls["impliedVolatility"].median() if "impliedVolatility" in calls.columns else None
            implied_move = round(float(atm_iv) * (days / 365) ** 0.5 * 100, 2) if atm_iv else None
        except Exception:
            implied_move = None

        return {
            "nearest_expiry": near_exp,
            "put_call_ratio":  pc_ratio,
            "call_volume":     int(call_vol),
            "put_volume":      int(put_vol),
            "implied_move_pct": implied_move,
            "expirations":     list(exps[:6]),
        }
    except Exception as exc:
        log.warning("Options summary failed for %s: %s", symbol, exc)
        return {}


def get_institutional_holders(symbol: str, top_n: int = 8) -> list[dict[str, Any]]:
    """Fetch top institutional shareholders.

    Args:
        symbol: Ticker symbol.
        top_n:  Number of top holders to return.

    Returns:
        List of holder dicts.
    """
    try:
        t     = _ticker(symbol)
        inst  = t.institutional_holders
        if inst is None or inst.empty:
            return []
        rows = []
        for _, row in inst.head(top_n).iterrows():
            rows.append({
                "holder":      str(row.get("Holder", "")),
                "shares":      int(row.get("Shares", 0) or 0),
                "date":        str(row.get("Date Reported", ""))[:10],
                "pct_out":     _safe_float(row.get("% Out")),
                "value":       _safe_float(row.get("Value")),
            })
        return rows
    except Exception as exc:
        log.warning("Institutional holders failed for %s: %s", symbol, exc)
        return []


def get_key_stats(symbol: str) -> dict[str, Any]:
    """Fetch comprehensive fundamental stats from yfinance.info.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dict of key fundamental metrics.
    """
    try:
        info = _ticker(symbol).info or {}
        return {
            "market_cap":         info.get("marketCap"),
            "enterprise_value":   info.get("enterpriseValue"),
            "revenue_ttm":        info.get("totalRevenue"),
            "gross_profit_ttm":   info.get("grossProfits"),
            "ebitda":             info.get("ebitda"),
            "net_income":         info.get("netIncomeToCommon"),
            "free_cash_flow":     info.get("freeCashflow"),
            "debt_to_equity":     _safe_float(info.get("debtToEquity")),
            "current_ratio":      _safe_float(info.get("currentRatio")),
            "return_on_equity":   _safe_float(info.get("returnOnEquity")),
            "return_on_assets":   _safe_float(info.get("returnOnAssets")),
            "profit_margin":      _safe_float(info.get("profitMargins")),
            "revenue_growth_yoy": _safe_float(info.get("revenueGrowth")),
            "earnings_growth_yoy":_safe_float(info.get("earningsGrowth")),
            "short_ratio":        _safe_float(info.get("shortRatio")),
            "short_pct_float":    _safe_float(info.get("shortPercentOfFloat")),
            "beta":               _safe_float(info.get("beta")),
            "52w_high":           _safe_float(info.get("fiftyTwoWeekHigh")),
            "52w_low":            _safe_float(info.get("fiftyTwoWeekLow")),
            "avg_volume_10d":     info.get("averageVolume10days"),
            "dividend_yield":     _safe_float(info.get("dividendYield")),
            "price_to_book":      _safe_float(info.get("priceToBook")),
            "ev_to_revenue":      _safe_float(info.get("enterpriseToRevenue")),
            "ev_to_ebitda":       _safe_float(info.get("enterpriseToEbitda")),
        }
    except Exception as exc:
        log.warning("Key stats failed for %s: %s", symbol, exc)
        return {}


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

def _get_cik(symbol: str) -> str | None:
    """Look up CIK for a ticker symbol from EDGAR company tickers JSON.

    Args:
        symbol: Ticker symbol.

    Returns:
        Zero-padded 10-digit CIK string or None.
    """
    try:
        resp = requests.get(
            f"{_EDGAR_BASE}/files/company_tickers.json",
            headers=_EDGAR_HEADERS,
            timeout=8,
        )
        tickers = resp.json()
        sym = symbol.upper()
        for entry in tickers.values():
            if entry.get("ticker", "").upper() == sym:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        log.debug("CIK lookup failed for %s: %s", symbol, exc)
    return None


def get_sec_filings(symbol: str, max_filings: int = 5) -> list[dict[str, Any]]:
    """Fetch recent SEC filings for a company.

    Args:
        symbol:      Ticker symbol.
        max_filings: Max number of filings to return.

    Returns:
        List of filing dicts with type, date, description, url.
    """
    cik = _get_cik(symbol)
    if not cik:
        return []
    try:
        resp = requests.get(
            f"{_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
            f"&type=&dateb=&owner=include&count=10&search_text=&output=atom",
            headers=_EDGAR_HEADERS,
            timeout=8,
        )
        # Parse minimal XML for filing entries
        import re
        filings = []
        for match in re.finditer(
            r"<category[^>]*term=\"([^\"]+)\".*?<updated>([^<]+)</updated>.*?<filing-href>([^<]+)</filing-href>",
            resp.text, re.DOTALL
        )[:max_filings]:
            filings.append({
                "type":  match.group(1),
                "date":  match.group(2)[:10],
                "url":   match.group(3).strip(),
            })
        return filings
    except Exception as exc:
        log.debug("SEC filings failed for %s: %s", symbol, exc)
        return []


def get_full_fundamental_profile(symbol: str) -> dict[str, Any]:
    """Aggregate all fundamental data for a symbol.

    Args:
        symbol: Ticker symbol.

    Returns:
        Complete fundamental profile dict.
    """
    return {
        "key_stats":      get_key_stats(symbol),
        "earnings":       get_earnings_calendar(symbol),
        "insider_trades": get_insider_trades(symbol, max_rows=8),
        "institutions":   get_institutional_holders(symbol, top_n=6),
        "options":        get_options_summary(symbol),
        "sec_filings":    get_sec_filings(symbol, max_filings=4),
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    """Convert to float, returning None for invalid values."""
    if v is None:
        return None
    try:
        import math
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except Exception:
        return None
