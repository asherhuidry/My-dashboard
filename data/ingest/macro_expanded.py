"""Expanded FRED macro data connector — all 65 series from universe.py.

Handles:
  - Bulk fetching all FRED series with rate limiting
  - Frequency-aware storage (daily/weekly/monthly/quarterly)
  - Derived series: yield curve slopes, spread differentials, Z-scores
  - World Bank supplemental data (global GDP, debt-to-GDP)
  - BLS supplemental data (CPI components, PPI by industry)
  - Cross-series computations after fetch

All data is stored to Supabase in a unified macro_data table with columns:
  series_id, date, value, frequency, label, z_score_1y, z_score_3y
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

FRED_BASE   = "https://api.stlouisfed.org/fred"
FRED_KEY    = os.getenv("FRED_API_KEY", "")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "FinBrain/1.0 (financial-research)"})


# ── Core FRED fetcher ─────────────────────────────────────────────────────────

def fetch_fred_series(
    series_id: str,
    observation_start: str = "2000-01-01",
    observation_end:   str | None = None,
    frequency:         str | None = None,
    units:             str = "lin",
    vintage_dates:     str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a single FRED series.

    Args:
        series_id:         FRED series identifier (e.g. 'T10Y2Y').
        observation_start: Start date as YYYY-MM-DD string.
        observation_end:   End date (defaults to today).
        frequency:         Aggregation frequency override ('d', 'w', 'm', 'q', 'a').
        units:             Transformation: 'lin', 'chg', 'pch', 'log'.
        vintage_dates:     Comma-separated vintage dates for real-time data.

    Returns:
        List of {date, value, series_id} dicts, most recent first.
    """
    if not FRED_KEY:
        log.warning("FRED_API_KEY not set — using demo key (limited to 1000 obs)")
        key = "demo"
    else:
        key = FRED_KEY

    params: dict[str, Any] = {
        "series_id":         series_id,
        "api_key":           key,
        "file_type":         "json",
        "observation_start": observation_start,
        "units":             units,
        "sort_order":        "desc",
        "limit":             10000,
    }
    if observation_end:
        params["observation_end"] = observation_end
    if frequency:
        params["frequency"] = frequency
    if vintage_dates:
        params["vintage_dates"] = vintage_dates

    try:
        r = _SESSION.get(f"{FRED_BASE}/series/observations", params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        result = []
        for o in obs:
            val = o.get("value", ".")
            if val == ".":
                continue
            result.append({
                "series_id": series_id,
                "date":      o["date"],
                "value":     float(val),
            })
        return result
    except Exception as exc:
        log.warning("FRED fetch failed for %s: %s", series_id, exc)
        return []


def fetch_fred_series_info(series_id: str) -> dict[str, Any]:
    """Fetch metadata for a FRED series (title, units, frequency, last_updated)."""
    if not FRED_KEY:
        return {}
    try:
        r = _SESSION.get(
            f"{FRED_BASE}/series",
            params={"series_id": series_id, "api_key": FRED_KEY, "file_type": "json"},
            timeout=10,
        )
        r.raise_for_status()
        slist = r.json().get("seriess", [{}])
        return slist[0] if slist else {}
    except Exception as exc:
        log.debug("FRED info failed for %s: %s", series_id, exc)
        return {}


# ── Derived / computed series ─────────────────────────────────────────────────

def compute_z_score(values: list[float], window: int) -> list[float | None]:
    """Rolling Z-score over `window` observations."""
    import statistics
    result: list[float | None] = []
    for i, v in enumerate(values):
        if i < window - 1:
            result.append(None)
            continue
        window_vals = values[i - window + 1 : i + 1]
        mu  = statistics.mean(window_vals)
        sig = statistics.stdev(window_vals)
        result.append((v - mu) / sig if sig > 0 else 0.0)
    return result


def compute_yield_curve_slope(
    long_series: list[dict],
    short_series: list[dict],
) -> list[dict[str, Any]]:
    """Compute spread between two yield series (long - short).

    Useful for constructing curves not directly in FRED:
      e.g. 30Y-10Y steepener, 10Y-1Y slope.

    Args:
        long_series:  List of {date, value} for the longer maturity.
        short_series: List of {date, value} for the shorter maturity.

    Returns:
        List of {date, value} spread observations.
    """
    short_map = {o["date"]: o["value"] for o in short_series}
    result = []
    for obs in long_series:
        d = obs["date"]
        if d in short_map:
            result.append({"date": d, "value": round(obs["value"] - short_map[d], 4)})
    return result


def compute_real_rate(
    nominal_series: list[dict],
    inflation_series: list[dict],
) -> list[dict[str, Any]]:
    """Compute ex-ante real rate = nominal yield - inflation breakeven."""
    inf_map = {o["date"]: o["value"] for o in inflation_series}
    result = []
    for obs in nominal_series:
        d = obs["date"]
        if d in inf_map:
            result.append({"date": d, "value": round(obs["value"] - inf_map[d], 4)})
    return result


# ── Bulk ingestion pipeline ───────────────────────────────────────────────────

def fetch_all_macro_series(
    series_list: list[tuple[str, str, str]],
    start_date:  str = "2005-01-01",
    rate_limit:  float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all macro series from universe.MACRO_SERIES.

    Args:
        series_list: List of (series_id, label, frequency) tuples.
        start_date:  Historical start for all series.
        rate_limit:  Seconds to sleep between FRED API calls.

    Returns:
        Dict mapping series_id → list of {date, value} observations.
    """
    results: dict[str, list[dict]] = {}

    # FRED frequency codes
    freq_map = {"daily": "d", "weekly": "w", "monthly": "m", "quarterly": "q"}

    total = len(series_list)
    for i, (series_id, label, frequency) in enumerate(series_list):
        log.info("Fetching FRED %s (%s) [%d/%d]", series_id, label, i + 1, total)
        fred_freq = freq_map.get(frequency)
        obs = fetch_fred_series(series_id, observation_start=start_date, frequency=fred_freq)
        results[series_id] = obs
        if obs:
            log.debug("  → %d observations (most recent: %s = %.4f)",
                      len(obs), obs[0]["date"], obs[0]["value"])
        time.sleep(rate_limit)

    return results


def enrich_with_derived_series(
    data: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Add derived series computed from fetched data.

    New series added:
      - GS30_GS10: 30Y-10Y spread (long-end steepener)
      - GS10_GS1:  10Y-1Y slope (broader inversion signal)
      - REAL_10Y:  10Y nominal - 10Y breakeven (ex-ante real rate)
      - HY_IG_SPREAD: HY spread - IG spread (quality spread)
      - OIL_GAS_RATIO: WTI / Henry Hub ratio

    Returns:
        Extended data dict with new series.
    """
    derived = {}

    # 30Y-10Y steepener
    if "GS30" in data and "GS10" in data:
        derived["GS30_GS10"] = compute_yield_curve_slope(data["GS30"], data["GS10"])
        log.debug("Computed GS30_GS10: %d observations", len(derived["GS30_GS10"]))

    # 10Y-1Y slope
    if "GS10" in data and "GS1" in data:
        derived["GS10_GS1"] = compute_yield_curve_slope(data["GS10"], data["GS1"])
        log.debug("Computed GS10_GS1: %d observations", len(derived["GS10_GS1"]))

    # Ex-ante real rate (10Y nominal - 10Y breakeven)
    if "GS10" in data and "T10YIE" in data:
        derived["REAL_10Y"] = compute_real_rate(data["GS10"], data["T10YIE"])
        log.debug("Computed REAL_10Y: %d observations", len(derived["REAL_10Y"]))

    # HY-IG quality spread
    if "BAMLH0A0HYM2" in data and "BAMLC0A0CM" in data:
        derived["HY_IG_SPREAD"] = compute_yield_curve_slope(
            data["BAMLH0A0HYM2"], data["BAMLC0A0CM"]
        )
        log.debug("Computed HY_IG_SPREAD: %d observations", len(derived["HY_IG_SPREAD"]))

    # Oil/Gas ratio (energy sector signal)
    if "DCOILWTICO" in data and "DHHNGSP" in data:
        gas_map = {o["date"]: o["value"] for o in data["DHHNGSP"]}
        ratio = []
        for obs in data["DCOILWTICO"]:
            d = obs["date"]
            if d in gas_map and gas_map[d] > 0:
                ratio.append({"date": d, "value": round(obs["value"] / gas_map[d], 2)})
        derived["OIL_GAS_RATIO"] = ratio
        log.debug("Computed OIL_GAS_RATIO: %d observations", len(ratio))

    data.update(derived)
    return data


def add_z_scores(
    data: dict[str, list[dict]],
    window_1y: int = 252,
    window_3y: int = 756,
) -> dict[str, list[dict]]:
    """Add Z-score fields to each observation.

    Modifies each observation dict in-place to add:
      - z_1y: 1-year rolling Z-score
      - z_3y: 3-year rolling Z-score
      - percentile_1y: empirical percentile vs 1Y window

    Args:
        data:      Series data dict.
        window_1y: Trading day window for 1Y Z-score.
        window_3y: Trading day window for 3Y Z-score.

    Returns:
        Modified data dict.
    """
    for series_id, obs_list in data.items():
        if not obs_list:
            continue
        # obs_list is sorted descending; reverse for chronological
        chron = list(reversed(obs_list))
        values = [o["value"] for o in chron]

        z1 = compute_z_score(values, min(window_1y, len(values)))
        z3 = compute_z_score(values, min(window_3y, len(values)))

        for i, obs in enumerate(chron):
            obs["z_1y"] = round(z1[i], 3) if z1[i] is not None else None
            obs["z_3y"] = round(z3[i], 3) if z3[i] is not None else None

    return data


# ── Supabase storage ──────────────────────────────────────────────────────────

def store_macro_to_supabase(
    data: dict[str, list[dict]],
    labels: dict[str, str],
    freqs:  dict[str, str],
    upsert: bool = True,
) -> dict[str, int]:
    """Upsert macro observations to Supabase macro_data table.

    Args:
        data:   {series_id: [{date, value, z_1y?, z_3y?}, ...]}
        labels: {series_id: human_readable_label}
        freqs:  {series_id: frequency_string}
        upsert: If True, upsert (insert or update); else insert only.

    Returns:
        {series_id: rows_written} summary.
    """
    try:
        from db.supabase.client import get_client
        sb = get_client()
    except Exception as exc:
        log.warning("Supabase unavailable: %s", exc)
        return {}

    summary: dict[str, int] = {}
    BATCH_SIZE = 500

    for series_id, obs_list in data.items():
        if not obs_list:
            continue
        rows = []
        for obs in obs_list:
            row: dict[str, Any] = {
                "series_id": series_id,
                "date":      obs["date"],
                "value":     obs["value"],
                "label":     labels.get(series_id, series_id),
                "frequency": freqs.get(series_id, "unknown"),
            }
            if "z_1y" in obs:
                row["z_score_1y"] = obs["z_1y"]
            if "z_3y" in obs:
                row["z_score_3y"] = obs["z_3y"]
            rows.append(row)

        written = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            try:
                if upsert:
                    sb.table("macro_data").upsert(
                        batch, on_conflict="series_id,date"
                    ).execute()
                else:
                    sb.table("macro_data").insert(batch, ignore_duplicates=True).execute()
                written += len(batch)
            except Exception as exc:
                log.debug("Supabase write error for %s batch %d: %s", series_id, i, exc)

        summary[series_id] = written
        log.debug("Stored %d rows for %s", written, series_id)

    return summary


# ── Macro regime detection ────────────────────────────────────────────────────

def detect_macro_regime(latest: dict[str, float]) -> dict[str, Any]:
    """Classify the current macro regime from latest series values.

    Regimes:
      - Growth: expansion vs contraction (yield curve, employment, PMI proxy)
      - Inflation: hot vs cold vs deflation (CPI, PCE, breakevens)
      - Credit: tight vs loose (HY spreads, IG spreads, financial stress)
      - Liquidity: easy vs tight (Fed balance sheet, M2, overnight repos)
      - Risk: risk-on vs risk-off (VIX, HY spreads, USD, EEM)

    Args:
        latest: Dict of {series_id: latest_value} for key indicators.

    Returns:
        Dict with regime labels and sub-scores.
    """
    scores: dict[str, float] = {}

    # --- Growth regime ---
    t10y2y  = latest.get("T10Y2Y", 0)
    t10y3m  = latest.get("T10Y3M", 0)
    # Inverted yield curve = contraction signal
    curve_score = 1.0 if t10y2y > 0.5 else -1.0 if t10y2y < -0.25 else 0.0
    scores["growth"] = curve_score

    # --- Inflation regime ---
    cpi_yoy = latest.get("CPIAUCSL", 2.0)
    breakeven_10y = latest.get("T10YIE", 2.0)
    if cpi_yoy > 4.0 or breakeven_10y > 3.0:
        scores["inflation"] = 1.0   # hot
    elif cpi_yoy < 1.5 or breakeven_10y < 1.5:
        scores["inflation"] = -1.0  # cold
    else:
        scores["inflation"] = 0.0   # normal

    # --- Credit regime ---
    hy_spread = latest.get("BAMLH0A0HYM2", 4.0)
    fin_stress = latest.get("STLFSI4", 0.0)
    if hy_spread > 6.0 or fin_stress > 1.5:
        scores["credit"] = -1.0  # tight / stressed
    elif hy_spread < 3.0 and fin_stress < -0.5:
        scores["credit"] = 1.0   # loose / benign
    else:
        scores["credit"] = 0.0

    # --- Liquidity regime ---
    vix = latest.get("VIXCLS", 20)
    hy_pct_change = latest.get("BAMLH0A0HYM2", 4.0)
    scores["risk"] = (
        -1.0 if vix > 30 else
         1.0 if vix < 15 else
         0.0
    )

    # Composite macro quadrant (growth x inflation matrix)
    g = scores["growth"]
    inf = scores["inflation"]

    if g >= 0 and inf > 0:
        quadrant = "reflation"       # growing, high inflation
    elif g >= 0 and inf <= 0:
        quadrant = "goldilocks"      # growing, low inflation
    elif g < 0 and inf > 0:
        quadrant = "stagflation"     # contracting, high inflation (worst)
    else:
        quadrant = "deflation_risk"  # contracting, low inflation

    return {
        "quadrant":    quadrant,
        "scores":      scores,
        "key_levels": {
            "yield_curve":  round(t10y2y, 3),
            "hy_spread":    round(hy_spread, 2),
            "vix":          round(vix, 1),
            "breakeven_10y": round(breakeven_10y, 2),
        },
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def get_latest_macro_snapshot(
    series_ids: list[str] | None = None,
) -> dict[str, float]:
    """Get the latest value for each macro series from Supabase.

    Args:
        series_ids: Specific series to fetch; None = all.

    Returns:
        {series_id: latest_value} dict.
    """
    try:
        from db.supabase.client import get_client
        sb = get_client()
        q = sb.table("macro_data").select("series_id, date, value").order("date", desc=True)
        if series_ids:
            q = q.in_("series_id", series_ids)
        result = q.limit(5000).execute()
        # Deduplicate: keep latest per series_id
        seen: dict[str, float] = {}
        for row in result.data:
            sid = row["series_id"]
            if sid not in seen:
                seen[sid] = row["value"]
        return seen
    except Exception as exc:
        log.warning("Failed to get macro snapshot: %s", exc)
        return {}


# ── Supplemental: World Bank ──────────────────────────────────────────────────

def fetch_world_bank_series(
    indicator: str,
    countries: list[str] | None = None,
    start_year: int = 2000,
) -> list[dict[str, Any]]:
    """Fetch World Bank indicator data.

    Args:
        indicator: WB indicator code (e.g. 'NY.GDP.MKTP.KD.ZG' for real GDP growth).
        countries: ISO-3 country codes; defaults to major economies.
        start_year: Historical start year.

    Returns:
        List of {country, year, value} dicts.
    """
    if countries is None:
        countries = ["USA", "CHN", "JPN", "DEU", "GBR", "IND", "FRA", "CAN", "BRA", "KOR"]

    results = []
    for country in countries:
        try:
            url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
            r = _SESSION.get(url, params={
                "format": "json",
                "per_page": 100,
                "date": f"{start_year}:{datetime.now().year}",
            }, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            if len(data) < 2 or not data[1]:
                continue
            for point in data[1]:
                if point.get("value") is not None:
                    results.append({
                        "country":   country,
                        "year":      point["date"],
                        "value":     float(point["value"]),
                        "indicator": indicator,
                    })
            time.sleep(0.1)
        except Exception as exc:
            log.debug("World Bank fetch failed for %s/%s: %s", country, indicator, exc)

    return results


WORLD_BANK_INDICATORS = [
    ("NY.GDP.MKTP.KD.ZG", "real_gdp_growth"),
    ("FP.CPI.TOTL.ZG",    "inflation_cpi"),
    ("GC.DOD.TOTL.GD.ZS", "debt_to_gdp"),
    ("BN.CAB.XOKA.GD.ZS", "current_account_pct_gdp"),
    ("NE.EXP.GNFS.ZS",    "exports_pct_gdp"),
    ("BX.KLT.DINV.WD.GD.ZS", "fdi_inflow_pct_gdp"),
]


def fetch_global_macro_context() -> dict[str, Any]:
    """Fetch global macro context from World Bank for major economies.

    Returns:
        Dict with global growth, inflation, and debt metrics by country.
    """
    context: dict[str, Any] = {}
    for indicator_code, label in WORLD_BANK_INDICATORS[:4]:  # limit for speed
        log.info("Fetching World Bank: %s", label)
        data = fetch_world_bank_series(indicator_code)
        context[label] = data
        time.sleep(0.2)
    return context


# ── Main entrypoint ────────────────────────────────────────────────────────────

def run(
    start_date: str = "2005-01-01",
    store: bool = True,
    include_z_scores: bool = True,
    include_derived: bool = True,
) -> dict[str, Any]:
    """Run the full expanded macro data pipeline.

    Steps:
      1. Fetch all 65 FRED series with rate limiting
      2. Compute derived series (spreads, real rates, ratios)
      3. Add Z-scores for regime detection
      4. Store to Supabase
      5. Detect current macro regime
      6. Return summary

    Args:
        start_date:       Historical start date (YYYY-MM-DD).
        store:            Whether to persist to Supabase.
        include_z_scores: Whether to compute and store Z-scores.
        include_derived:  Whether to compute derived series.

    Returns:
        Summary dict with series counts, regime, and storage results.
    """
    from data.ingest.universe import MACRO_SERIES

    labels = {s[0]: s[1] for s in MACRO_SERIES}
    freqs  = {s[0]: s[2] for s in MACRO_SERIES}

    log.info("Starting expanded macro ingest: %d series, start=%s", len(MACRO_SERIES), start_date)

    # Step 1: Fetch all FRED series
    data = fetch_all_macro_series(MACRO_SERIES, start_date=start_date, rate_limit=0.25)

    total_obs = sum(len(v) for v in data.values())
    log.info("Fetched %d total observations across %d series", total_obs, len(data))

    # Step 2: Derived series
    if include_derived:
        data = enrich_with_derived_series(data)
        # Add labels/freqs for derived series
        derived_labels = {
            "GS30_GS10":    "30Y-10Y Treasury Spread",
            "GS10_GS1":     "10Y-1Y Treasury Slope",
            "REAL_10Y":     "10Y Real Rate (ex-ante)",
            "HY_IG_SPREAD": "HY-IG Quality Spread",
            "OIL_GAS_RATIO":"WTI/Henry Hub Ratio",
        }
        labels.update(derived_labels)
        for k in derived_labels:
            freqs[k] = "daily"

    # Step 3: Z-scores
    if include_z_scores:
        data = add_z_scores(data)

    # Step 4: Store
    storage_summary: dict[str, int] = {}
    if store:
        log.info("Storing to Supabase...")
        storage_summary = store_macro_to_supabase(data, labels, freqs)
        rows_stored = sum(storage_summary.values())
        log.info("Stored %d rows to Supabase", rows_stored)

    # Step 5: Regime detection
    latest = {sid: obs[0]["value"] for sid, obs in data.items() if obs}
    regime = detect_macro_regime(latest)
    log.info("Current macro regime: %s (growth=%.1f, credit=%.1f, risk=%.1f)",
             regime["quadrant"],
             regime["scores"].get("growth", 0),
             regime["scores"].get("credit", 0),
             regime["scores"].get("risk", 0))

    result = {
        "series_fetched":    len(data),
        "total_observations": total_obs,
        "storage":           storage_summary,
        "regime":            regime,
        "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
    }

    # Log to evolution
    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id    = "macro_ingest",
            action      = "fetch_all_macro_series",
            after_state = {
                "series_count": len(data),
                "observations": total_obs,
                "regime":       regime["quadrant"],
            },
        ))
    except Exception:
        pass

    return result
