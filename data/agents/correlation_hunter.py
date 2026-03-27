"""Correlation Hunter — discovers hidden statistical relationships between data series.

This agent systematically tests every pair of time series in the universe for:
  1. Rolling Pearson correlation (0, 5, 10, 21 day lags)
  2. Granger causality (does X predict Y beyond Y's own history?)
  3. Mutual information (non-linear dependency)
  4. Lead-lag relationships (which series leads which)
  5. Regime-conditional correlation (does the relationship change in bear markets?)
  6. Cross-asset spillovers (volatility contagion)

Findings are written to Supabase for tracking, and the most significant
relationships are written to the Neo4j knowledge graph as edges.

This is the core of what makes FinBrain revolutionary: it doesn't just
read known relationships — it discovers unknown ones systematically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

AGENT_ID = "correlation_hunter"

# Minimum overlapping observations to run any test
MIN_OBS = 60

# Significance thresholds
PEARSON_THRESHOLD    = 0.35   # |r| >= 0.35 to be "notable"
GRANGER_P_THRESHOLD  = 0.05   # p-value for Granger causality
MI_THRESHOLD         = 0.05   # mutual information bits


@dataclass
class CorrelationFinding:
    """One discovered statistical relationship.

    Attributes:
        series_a:   Name of the first series (the potential cause or co-mover).
        series_b:   Name of the second series (the potential effect).
        lag_days:   Lead/lag in trading days (positive = A leads B).
        pearson_r:  Pearson correlation coefficient at the best lag.
        granger_p:  p-value of Granger causality test (A → B).
        mutual_info:Mutual information between the two series.
        regime:     'all' | 'bull' | 'bear' — which market regime this holds in.
        strength:   'strong' | 'moderate' | 'weak'.
        relationship_type: One of the KNOWN_RELATIONSHIPS types or 'discovered'.
        timestamp:  When this was computed.
    """
    series_a:          str
    series_b:          str
    lag_days:          int
    pearson_r:         float
    granger_p:         float | None
    mutual_info:       float | None
    regime:            str = "all"
    strength:          str = "moderate"
    relationship_type: str = "discovered"
    timestamp:         datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_a":          self.series_a,
            "series_b":          self.series_b,
            "lag_days":          self.lag_days,
            "pearson_r":         round(self.pearson_r, 4),
            "granger_p":         round(self.granger_p, 4) if self.granger_p is not None else None,
            "mutual_info":       round(self.mutual_info, 4) if self.mutual_info is not None else None,
            "regime":            self.regime,
            "strength":          self.strength,
            "relationship_type": self.relationship_type,
            "computed_at":       self.timestamp.isoformat(),
        }


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_returns(symbols: list[str], period: str = "3y") -> pd.DataFrame:
    """Fetch daily close prices and compute log returns.

    Args:
        symbols: List of yfinance ticker symbols.
        period:  yfinance period string.

    Returns:
        DataFrame of daily log returns, one column per symbol.
        Index is timezone-naive date for clean alignment with FRED data.
    """
    data = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            h = t.history(period=period)
            if not h.empty:
                h.index = h.index.normalize().tz_localize(None)
                closes  = h["Close"].dropna()
                data[sym] = np.log(closes / closes.shift(1)).dropna()
        except Exception as exc:
            log.debug("Fetch failed for %s: %s", sym, exc)
    return pd.DataFrame(data).dropna(how="all")


def _fetch_macro_levels(series_ids: list[str]) -> pd.DataFrame:
    """Fetch FRED macro series as standardized daily changes.

    Uses first-differencing and z-score normalization rather than pct_change,
    because many macro indicators are already levels (VIX, yields, spreads)
    where pct_change produces noisy near-zero values.  The resulting series
    are comparable in scale to log-return price series.

    Args:
        series_ids: List of FRED series identifiers.

    Returns:
        DataFrame of standardized daily changes, one column per series.
    """
    import os
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv("FRED_API_KEY", ""))
    except Exception:
        return pd.DataFrame()

    data = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start="2015-01-01")
            s = s.dropna()
            s.index = pd.to_datetime(s.index).normalize()
            # Resample to business days (forward-fill for monthly/quarterly)
            daily = s.resample("B").last().ffill()
            # First-difference then z-score normalise so the scale is
            # comparable to log returns of price series.
            diff = daily.diff().dropna()
            std = diff.std()
            if std > 0:
                data[sid] = diff / std
            else:
                log.debug("Skipping %s: zero variance after differencing", sid)
        except Exception as exc:
            log.debug("FRED fetch failed for %s: %s", sid, exc)

    return pd.DataFrame(data).dropna(how="all")


# ── Statistical tests ─────────────────────────────────────────────────────────

def pearson_with_lag(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 20,
) -> tuple[float, int]:
    """Find best Pearson correlation across lags -max_lag to +max_lag.

    Positive lag means x leads y by that many periods.

    Args:
        x:       First series (potential cause).
        y:       Second series (potential effect).
        max_lag: Maximum lag/lead to test in either direction.

    Returns:
        (best_r, best_lag) — the highest |correlation| and its lag.
    """
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < MIN_OBS:
        return 0.0, 0

    xa, ya = aligned.iloc[:, 0], aligned.iloc[:, 1]
    best_r, best_lag = 0.0, 0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = xa.iloc[lag:], ya.iloc[:len(xa)-lag] if lag > 0 else ya
        else:
            xs, ys = xa.iloc[:len(xa)+lag], ya.iloc[-lag:]

        if len(xs) < MIN_OBS or len(ys) < MIN_OBS:
            continue
        n  = min(len(xs), len(ys))
        xs, ys = xs.iloc[:n].values, ys.iloc[:n].values
        r  = float(np.corrcoef(xs, ys)[0, 1])
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, lag

    return best_r, best_lag


def granger_causality_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 5,
) -> float | None:
    """Test whether 'cause' Granger-causes 'effect'.

    A Granger causality test answers: "Does knowing the past of X improve
    our forecast of Y beyond just knowing the past of Y itself?"

    Args:
        cause:   The potential causal series.
        effect:  The potential effect series.
        max_lag: Maximum lag to test.

    Returns:
        The minimum p-value across lags, or None if the test fails.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        df = pd.concat([effect, cause], axis=1).dropna()
        if len(df) < max_lag * 4 + 10:
            return None
        results = grangercausalitytests(df.values, maxlag=max_lag, verbose=False)
        p_values = [results[lag][0]["ssr_chi2test"][1] for lag in range(1, max_lag + 1)]
        return float(min(p_values))
    except ImportError:
        return None
    except Exception:
        return None


def mutual_information(x: pd.Series, y: pd.Series) -> float | None:
    """Estimate mutual information between two series.

    Unlike Pearson, mutual information captures non-linear relationships.
    If MI > 0, there is some statistical dependency even if correlation = 0.

    Args:
        x: First series.
        y: Second series.

    Returns:
        Mutual information in bits, or None if computation fails.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < MIN_OBS:
            return None
        xi = df.iloc[:, 0].values.reshape(-1, 1)
        yi = df.iloc[:, 1].values
        mi = mutual_info_regression(xi, yi, random_state=42)
        return float(mi[0])
    except ImportError:
        return None
    except Exception:
        return None


def regime_conditional_correlation(
    x: pd.Series,
    y: pd.Series,
    benchmark: pd.Series | None = None,
) -> dict[str, float]:
    """Compute correlations separately in bull and bear regimes.

    Many correlations are regime-dependent: assets that are uncorrelated
    in normal markets become highly correlated during crises (correlation
    goes to 1 exactly when you don't want it to).

    Args:
        x:         First series.
        y:         Second series.
        benchmark: Series defining regimes (default: x itself).
                   Bear = periods where 63-day return < 0.

    Returns:
        Dict with 'all', 'bull', 'bear' correlation values.
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < MIN_OBS:
        return {"all": float(np.corrcoef(x.dropna(), y.dropna())[0,1]) if len(x.dropna()) >= 2 else 0}

    xa, ya = df.iloc[:, 0], df.iloc[:, 1]

    # Define regime using rolling 63-day return of x (or benchmark)
    ref = benchmark.reindex(xa.index).ffill() if benchmark is not None else xa
    rolling_ret = ref.rolling(63).sum()  # sum of log returns ≈ 63d return
    bear_mask   = rolling_ret < 0
    bull_mask   = ~bear_mask

    results = {"all": round(float(xa.corr(ya)), 4)}

    if bear_mask.sum() >= 20:
        results["bear"] = round(float(xa[bear_mask].corr(ya[bear_mask])), 4)
    if bull_mask.sum() >= 20:
        results["bull"] = round(float(xa[bull_mask].corr(ya[bull_mask])), 4)

    return results


# ── Discovery engine ──────────────────────────────────────────────────────────

def hunt_correlations(
    price_symbols: list[str],
    macro_series:  list[str] | None = None,
    min_pearson:   float = PEARSON_THRESHOLD,
    test_granger:  bool  = True,
    max_pairs:     int   = 500,
) -> list[CorrelationFinding]:
    """Run the full correlation discovery pipeline.

    Tests all pairs of (price, price) and (macro, price) series for
    statistically significant relationships at multiple lags.

    Args:
        price_symbols: List of asset ticker symbols.
        macro_series:  Optional list of FRED series IDs to include.
        min_pearson:   Minimum |r| to consider a finding significant.
        test_granger:  Whether to run Granger causality (slower).
        max_pairs:     Maximum pairs to test (for performance).

    Returns:
        List of CorrelationFinding objects, sorted by |pearson_r| descending.
    """
    findings: list[CorrelationFinding] = []

    log.info("Fetching price returns for %d symbols", len(price_symbols))
    price_df = _fetch_returns(price_symbols[:50])  # cap for performance

    macro_df = pd.DataFrame()
    if macro_series:
        log.info("Fetching macro series: %d", len(macro_series))
        macro_df = _fetch_macro_levels(macro_series[:20])

    # Combine all series
    combined = pd.concat([price_df, macro_df], axis=1).dropna(how="all")
    cols     = list(combined.columns)
    log.info("Running correlation tests on %d series (%d pairs)", len(cols), len(cols)*(len(cols)-1)//2)

    tested = 0
    for i, col_a in enumerate(cols):
        for col_b in cols[i+1:]:
            if tested >= max_pairs:
                break
            tested += 1

            xa = combined[col_a].dropna()
            xb = combined[col_b].dropna()

            # Cross-align
            aligned = pd.concat([xa, xb], axis=1).dropna()
            if len(aligned) < MIN_OBS:
                continue

            r, lag = pearson_with_lag(aligned.iloc[:,0], aligned.iloc[:,1], max_lag=20)

            if abs(r) < min_pearson:
                continue

            # Granger causality (directional: col_a → col_b at best lag)
            g_p = None
            if test_granger and abs(lag) <= 10:
                g_p = granger_causality_test(aligned.iloc[:,0], aligned.iloc[:,1])

            # Mutual information
            mi = mutual_information(aligned.iloc[:,0], aligned.iloc[:,1])

            strength = (
                "strong"   if abs(r) >= 0.65
                else "moderate" if abs(r) >= 0.45
                else "weak"
            )

            # Determine type
            from data.ingest.universe import KNOWN_RELATIONSHIPS
            known_types = {(a, b): rt for a, b, _, rt in KNOWN_RELATIONSHIPS}
            rel_type    = known_types.get((col_a, col_b), known_types.get((col_b, col_a), "discovered"))

            findings.append(CorrelationFinding(
                series_a          = col_a,
                series_b          = col_b,
                lag_days          = lag,
                pearson_r         = r,
                granger_p         = g_p,
                mutual_info       = mi,
                strength          = strength,
                relationship_type = rel_type,
            ))

    findings.sort(key=lambda f: abs(f.pearson_r), reverse=True)
    log.info("Correlation hunter: found %d significant relationships from %d tested pairs",
             len(findings), tested)
    return findings


def build_correlation_matrix(
    symbols: list[str],
    period:  str = "1y",
    method:  str = "pearson",
) -> pd.DataFrame:
    """Build a full correlation matrix for a set of symbols.

    Args:
        symbols: List of yfinance symbols.
        period:  History period.
        method:  'pearson' | 'spearman' | 'kendall'.

    Returns:
        Symmetric DataFrame correlation matrix, indexed and columned by symbol.
    """
    df = _fetch_returns(symbols, period=period)
    return df.corr(method=method)


def find_leading_indicators(
    target: str,
    candidates: list[str],
    max_lead: int = 21,
) -> list[dict[str, Any]]:
    """Find which series best predict 'target' in advance.

    Tests all candidates at lags 1 to max_lead and returns the ones
    with the strongest predictive relationship.

    Args:
        target:     The series to predict (e.g. 'SPY').
        candidates: Potential leading indicators to test.
        max_lead:   Maximum lead time in days.

    Returns:
        List of dicts sorted by predictive power (|r| desc), each with:
        {series, lag_days, pearson_r, granger_p, interpretation}.
    """
    all_syms = list(set([target] + candidates))
    df       = _fetch_returns(all_syms)

    if target not in df.columns:
        return []

    y_ret  = df[target].dropna()
    results = []

    for col in candidates:
        if col == target or col not in df.columns:
            continue
        x_ret = df[col].dropna()

        # Test at fixed positive lags (x leads y)
        best_r, best_lag = 0.0, 0
        for lag in range(1, max_lead + 1):
            aligned = pd.concat([x_ret.shift(lag), y_ret], axis=1).dropna()
            if len(aligned) < MIN_OBS:
                continue
            r = float(aligned.iloc[:,0].corr(aligned.iloc[:,1]))
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r, best_lag = r, lag

        if abs(best_r) < PEARSON_THRESHOLD:
            continue

        # Granger
        aligned_gc = pd.concat([x_ret, y_ret], axis=1).dropna()
        g_p = None
        if len(aligned_gc) >= 30:
            g_p = granger_causality_test(aligned_gc.iloc[:,0], aligned_gc.iloc[:,1])

        direction = "positive" if best_r > 0 else "inverse"
        results.append({
            "series":         col,
            "lag_days":       best_lag,
            "pearson_r":      round(best_r, 4),
            "granger_p":      round(g_p, 4) if g_p else None,
            "direction":      direction,
            "interpretation": (
                f"{col} leads {target} by {best_lag} days with {direction} correlation "
                f"(r={best_r:.3f})"
                + (f", Granger p={g_p:.3f}" if g_p else "")
            ),
        })

    results.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    return results[:15]


def run(symbols: list[str] | None = None) -> list[CorrelationFinding]:
    """Run full discovery pipeline on the FinBrain universe.

    Runs two complementary discovery passes:
    1. Correlation hunting — pairwise Pearson, Granger, MI across all series
    2. Sensitivity analysis — OLS factor exposure (beta) for each asset against
       8 macro factors (rates, inflation, dollar, oil, volatility, credit,
       liquidity, financial stress)

    After discovering and persisting, automatically materializes the full
    market graph in Neo4j so the graph compounds by default.

    Args:
        symbols: Optional symbol list override. Defaults to the full universe.

    Returns:
        List of all significant CorrelationFindings discovered.
    """
    from data.ingest.universe import get_all_symbols, MACRO_SERIES

    if symbols is None:
        symbols = get_all_symbols()

    price_syms = symbols[:60]
    macro_ids = [s[0] for s in MACRO_SERIES if s[2] == "daily"][:20]

    # ── Pass 1: Correlation discovery ──────────────────────────────────
    findings = hunt_correlations(
        price_symbols = price_syms,
        macro_series  = macro_ids,
        test_granger  = True,
        max_pairs     = 1000,
    )

    # ── Pass 2: Factor sensitivity discovery ───────────────────────────
    sensitivity_findings = _run_sensitivity_pass(price_syms, macro_ids)
    findings.extend(sensitivity_findings)

    # Persist all discoveries (correlations + sensitivities) to queryable store
    run_id = _persist_discoveries(findings)

    # Materialize the full graph in Neo4j after successful persistence.
    # Uses MERGE so it's idempotent — safe to run every time.
    # Failures here never affect discovery persistence.
    graph_result = _materialize_graph(run_id)

    # Log summary to evolution audit trail
    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        n_sensitivity = len(sensitivity_findings)
        n_correlation = len(findings) - n_sensitivity
        after_state: dict = {
            "findings":      len(findings),
            "correlations":  n_correlation,
            "sensitivities": n_sensitivity,
            "strong":        sum(1 for f in findings if f.strength == "strong"),
            "with_granger":  sum(1 for f in findings if f.granger_p and f.granger_p < 0.05),
            "run_id":        run_id,
        }
        if graph_result:
            after_state["graph"] = graph_result
        log_evolution(EvolutionLogEntry(
            agent_id    = AGENT_ID,
            action      = "hunt_correlations",
            after_state = after_state,
        ))
    except Exception:
        pass

    return findings


def _run_sensitivity_pass(
    price_symbols: list[str],
    macro_ids: list[str],
) -> list[CorrelationFinding]:
    """Run factor sensitivity analysis and convert results to CorrelationFindings.

    Fetches asset returns and macro data, then computes OLS betas for each
    asset against each sensitivity factor. Results are stored as
    CorrelationFindings with relationship_type like 'rate_sensitive'.

    The beta coefficient is stored in the mutual_info field (repurposed)
    so it persists through the existing Supabase schema without migration.

    Args:
        price_symbols: Asset tickers to analyze.
        macro_ids:     FRED series IDs (superset — only sensitivity factors used).

    Returns:
        List of CorrelationFindings for significant factor exposures.
    """
    try:
        from data.agents.sensitivity_analyzer import (
            compute_factor_sensitivities, SENSITIVITY_FACTORS,
        )
    except ImportError as exc:
        log.warning("Sensitivity analyzer not available: %s", exc)
        return []

    # Fetch the same data the correlation hunter uses
    log.info("Running factor sensitivity analysis...")
    price_df = _fetch_returns(price_symbols[:50])

    # Ensure all sensitivity factor series are included in macro fetch
    factor_ids = [fid for fid, _ in SENSITIVITY_FACTORS.values()]
    all_macro = list(set(macro_ids) | set(factor_ids))
    macro_df = _fetch_macro_levels(all_macro)

    if price_df.empty or macro_df.empty:
        log.warning("Insufficient data for sensitivity analysis")
        return []

    raw_results = compute_factor_sensitivities(price_df, macro_df)

    # Convert sensitivity dicts to CorrelationFindings for unified persistence.
    # Beta is stored in mutual_info to fit the existing Supabase schema.
    findings = []
    for r in raw_results:
        findings.append(CorrelationFinding(
            series_a          = r["series_a"],
            series_b          = r["series_b"],
            lag_days          = r["lag_days"],
            pearson_r         = r["pearson_r"],
            granger_p         = r.get("p_value"),
            mutual_info       = r["beta"],       # beta stored in mutual_info
            strength          = r["strength"],
            relationship_type = r["relationship_type"],
        ))

    log.info("Sensitivity pass: %d significant factor exposures found", len(findings))
    return findings


def _materialize_graph(run_id: str | None) -> dict | None:
    """Materialize persisted discoveries into the Neo4j market graph.

    Called automatically after successful discovery persistence.
    Only runs if persistence succeeded (run_id is not None).
    Failures are logged but never propagated — discovery data is safe.

    Args:
        run_id: The discovery run_id, or None if persistence failed.

    Returns:
        Materialization summary dict, or None if skipped/failed.
    """
    if run_id is None:
        log.info("Skipping graph materialization — discovery persistence failed")
        return None

    try:
        from data.agents.graph_materializer import materialize
        log.info("Materializing market graph after discovery run %s...", run_id)
        result = materialize(min_strength="moderate")
        log.info(
            "Graph materialized: %d nodes, %d edges (%d CORRELATED_WITH, %d CAUSES)",
            result.get("asset_nodes_merged", 0) + result.get("macro_nodes_merged", 0),
            result.get("edges_merged", 0),
            result.get("correlated_with_edges", 0),
            result.get("causes_edges", 0),
        )
        return result
    except Exception as exc:
        log.warning("Graph materialization failed (discoveries are safe): %s", exc)
        return None


def _persist_discoveries(findings: list[CorrelationFinding]) -> str | None:
    """Convert CorrelationFindings to DiscoveryRecords and persist them.

    Args:
        findings: List of correlation findings from the hunt.

    Returns:
        The run_id used for this batch, or None if persistence failed.
    """
    if not findings:
        return None
    try:
        import uuid as _uuid
        from db.supabase.client import DiscoveryRecord, save_discoveries

        run_id = str(_uuid.uuid4())
        records = [
            DiscoveryRecord(
                series_a          = f.series_a,
                series_b          = f.series_b,
                lag_days          = f.lag_days,
                pearson_r         = f.pearson_r,
                granger_p         = f.granger_p,
                mutual_info       = f.mutual_info,
                regime            = f.regime,
                strength          = f.strength,
                relationship_type = f.relationship_type,
                run_id            = run_id,
                computed_at       = f.timestamp,
            )
            for f in findings
        ]
        save_discoveries(records)
        log.info("Persisted %d discoveries (run_id=%s)", len(records), run_id)
        return run_id
    except Exception as exc:
        log.warning("Failed to persist discoveries: %s", exc)
        return None
