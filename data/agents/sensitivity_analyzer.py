"""Sensitivity Analyzer — discovers factor exposure edges for the market graph.

Instead of just finding "what moves together" (correlation), this module
discovers "what drives behavior" — which macro factors each asset is
meaningfully exposed to, and how those exposures change across regimes.

For each (asset, macro factor) pair, runs an OLS regression:
    asset_return_t = alpha + beta * factor_change_t + epsilon_t

A statistically significant beta creates a SENSITIVE_TO edge in the graph:
    Asset -[SENSITIVE_TO {factor_group: "rate", regime: "all", beta: -0.42}]-> MacroIndicator

Regime-conditioned analysis runs the same regression on subsets of the data
defined by market conditions:
    - "bear":   rolling 63-day benchmark return < 0
    - "stress": VIX level above its 75th percentile

This captures how exposures shift when conditions change — e.g. an equity
might have beta=+0.2 to credit spreads in normal markets but beta=+0.8
during stress, revealing hidden crisis sensitivity.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Factor definitions
#
# Each factor group maps to a single representative FRED series — the one
# most liquid, most responsive, and most commonly used as the benchmark
# for that type of macro exposure.
# ─────────────────────────────────────────────────────────────────────────────

SENSITIVITY_FACTORS: dict[str, tuple[str, str]] = {
    "rate":             ("GS10",          "10-Year Treasury Yield"),
    "inflation":        ("T10YIE",        "10-Year Breakeven Inflation"),
    "dollar":           ("DTWEXBGS",      "Broad USD Index"),
    "oil":              ("DCOILWTICO",    "WTI Crude Oil"),
    "volatility":       ("VIXCLS",        "CBOE VIX"),
    "credit":           ("BAMLH0A0HYM2",  "US High-Yield Spread"),
    "liquidity":        ("WALCL",         "Fed Balance Sheet Total Assets"),
    "financial_stress": ("STLFSI4",       "St. Louis Fed Financial Stress"),
}

# Minimum overlapping observations for regression
MIN_OBS = 60

# Minimum observations in a regime sub-sample to run regression
MIN_REGIME_OBS = 40

# Minimum |t-stat| to consider an exposure significant
MIN_T_STAT = 2.0

# Rolling window for bear/bull regime classification (trading days)
REGIME_LOOKBACK = 63


def compute_factor_sensitivities(
    asset_returns: pd.DataFrame,
    macro_levels: pd.DataFrame,
    min_t_stat: float = MIN_T_STAT,
) -> list[dict[str, Any]]:
    """Compute factor exposure (beta) for each asset against each macro factor.

    For each (asset, factor) pair with sufficient overlapping data, runs OLS:
        asset_return = alpha + beta * factor_change + epsilon

    Only returns pairs where the beta is statistically significant
    (|t-stat| >= min_t_stat, roughly p < 0.05).

    Args:
        asset_returns: DataFrame of daily log returns, one column per asset.
        macro_levels:  DataFrame of standardized daily changes for macro series.
        min_t_stat:    Minimum |t-statistic| to keep a sensitivity finding.

    Returns:
        List of sensitivity dicts, each containing:
        - series_a (asset), series_b (factor), factor_group, beta, t_stat,
          p_value, r_squared, pearson_r, strength, relationship_type
    """
    findings: list[dict[str, Any]] = []

    for factor_group, (factor_id, _factor_label) in SENSITIVITY_FACTORS.items():
        if factor_id not in macro_levels.columns:
            log.debug("Factor %s (%s) not in macro data — skipping", factor_group, factor_id)
            continue

        factor = macro_levels[factor_id].dropna()

        for asset in asset_returns.columns:
            result = _regress_single(
                asset_returns[asset], factor, asset, factor_id, factor_group,
                min_t_stat,
            )
            if result is not None:
                findings.append(result)

    log.info(
        "Sensitivity analysis: %d significant exposures across %d factor groups",
        len(findings),
        len({f["factor_group"] for f in findings}),
    )
    return findings


def _regress_single(
    asset_series: pd.Series,
    factor_series: pd.Series,
    asset_name: str,
    factor_id: str,
    factor_group: str,
    min_t_stat: float,
    regime: str = "all",
) -> dict[str, Any] | None:
    """Run OLS regression for a single (asset, factor) pair.

    Args:
        asset_series:  Daily log returns for the asset.
        factor_series: Standardized daily changes for the factor.
        asset_name:    Ticker symbol.
        factor_id:     FRED series ID.
        factor_group:  Semantic factor label (e.g., "rates").
        min_t_stat:    Minimum |t-stat| threshold.
        regime:        Market regime label ("all", "bear", "stress").

    Returns:
        Sensitivity dict if significant, None otherwise.
    """
    aligned = pd.concat([asset_series, factor_series], axis=1).dropna()
    if len(aligned) < MIN_OBS:
        return None

    y = aligned.iloc[:, 0].values  # asset returns
    x = aligned.iloc[:, 1].values  # factor changes

    # OLS via numpy: y = alpha + beta * x
    # Design matrix with intercept
    X = np.column_stack([np.ones(len(x)), x])

    try:
        # Solve normal equations: (X'X)^-1 X'y
        XtX = X.T @ X
        Xty = X.T @ y
        params = np.linalg.solve(XtX, Xty)

        alpha, beta = params[0], params[1]

        # Residuals and standard errors
        residuals = y - X @ params
        n = len(y)
        k = 2  # intercept + one regressor
        dof = n - k

        if dof <= 0:
            return None

        mse = (residuals @ residuals) / dof
        # Variance of beta: (X'X)^-1 * MSE, take [1,1] element
        var_beta = mse * np.linalg.inv(XtX)[1, 1]

        if var_beta <= 0:
            return None

        se_beta = np.sqrt(var_beta)
        t_stat = beta / se_beta

        # R-squared
        ss_res = residuals @ residuals
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # p-value from t-distribution
        from scipy.stats import t as t_dist
        p_value = float(2.0 * t_dist.sf(abs(t_stat), dof))

    except (np.linalg.LinAlgError, ValueError):
        return None

    if abs(t_stat) < min_t_stat:
        return None

    # Pearson correlation (for reference — stored alongside beta)
    corr = float(np.corrcoef(y, x)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    # Strength based on |t-stat|: how confident we are in this exposure
    strength = (
        "strong"   if abs(t_stat) >= 4.0
        else "moderate" if abs(t_stat) >= 2.5
        else "weak"
    )

    return {
        "series_a":          asset_name,
        "series_b":          factor_id,
        "factor_group":      factor_group,
        "beta":              round(float(beta), 6),
        "t_stat":            round(float(t_stat), 4),
        "p_value":           round(float(p_value), 6),
        "r_squared":         round(float(r_squared), 6),
        "pearson_r":         round(corr, 6),
        "strength":          strength,
        "relationship_type": f"{factor_group}_sensitive",
        "regime":            regime,
        "lag_days":          0,
        "timestamp":         datetime.now(tz=timezone.utc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_masks(
    asset_returns: pd.DataFrame,
    macro_levels: pd.DataFrame,
    benchmark: str = "SPY",
    vix_series: str = "VIXCLS",
) -> dict[str, pd.Series]:
    """Build boolean regime masks from market data.

    Two complementary regime definitions:
    - "bear":   rolling 63-day benchmark return < 0 (market drawdown)
    - "stress": VIX level above its 75th percentile (fear spike)

    Using the 75th percentile rather than a fixed VIX threshold makes the
    definition adaptive to the sample — works for any historical window.

    Args:
        asset_returns: DataFrame of daily log returns (must contain benchmark).
        macro_levels:  DataFrame of standardized daily changes (must contain VIX).
        benchmark:     Ticker for bear/bull classification (default SPY).
        vix_series:    FRED series for stress classification (default VIXCLS).

    Returns:
        Dict of {"bear": bool_series, "stress": bool_series} aligned to
        asset_returns index. Missing regimes are omitted.
    """
    masks: dict[str, pd.Series] = {}
    idx = asset_returns.index

    # ── Bear regime: rolling 63-day benchmark return < 0 ───────────────
    if benchmark in asset_returns.columns:
        rolling_ret = asset_returns[benchmark].rolling(REGIME_LOOKBACK).sum()
        bear_mask = (rolling_ret < 0).reindex(idx).fillna(False)
        if bear_mask.sum() >= MIN_REGIME_OBS:
            masks["bear"] = bear_mask
            log.info("Bear regime: %d / %d days (%.0f%%)",
                     bear_mask.sum(), len(bear_mask),
                     100 * bear_mask.sum() / len(bear_mask))
        else:
            log.info("Bear regime: insufficient days (%d < %d)",
                     bear_mask.sum(), MIN_REGIME_OBS)
    else:
        log.info("Bear regime: benchmark %s not in returns data", benchmark)

    # ── Stress regime: VIX above 75th percentile ───────────────────────
    if vix_series in macro_levels.columns:
        # macro_levels contains standardized changes, but for VIX level
        # we need the raw cumulative level. Reconstruct approximate VIX
        # level from cumulative sum of changes (relative, not absolute).
        # Since macro_levels are z-scored first-differences, we use the
        # z-score itself as a proxy: high z-score = elevated VIX.
        vix_z = macro_levels[vix_series].reindex(idx).fillna(0).rolling(21).mean()
        vix_75 = vix_z.quantile(0.75)
        stress_mask = (vix_z > vix_75).fillna(False)
        if stress_mask.sum() >= MIN_REGIME_OBS:
            masks["stress"] = stress_mask
            log.info("Stress regime: %d / %d days (%.0f%%)",
                     stress_mask.sum(), len(stress_mask),
                     100 * stress_mask.sum() / len(stress_mask))
        else:
            log.info("Stress regime: insufficient days (%d < %d)",
                     stress_mask.sum(), MIN_REGIME_OBS)
    else:
        log.info("Stress regime: VIX series %s not in macro data", vix_series)

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Regime-conditioned sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_sensitivities(
    asset_returns: pd.DataFrame,
    macro_levels: pd.DataFrame,
    regime_masks: dict[str, pd.Series],
    min_t_stat: float = MIN_T_STAT,
) -> list[dict[str, Any]]:
    """Compute factor sensitivities within each market regime.

    Runs the same OLS as compute_factor_sensitivities, but on subsets of data
    defined by regime masks. This reveals how exposures change when market
    conditions change — the key insight that static betas miss.

    Args:
        asset_returns: DataFrame of daily log returns, one column per asset.
        macro_levels:  DataFrame of standardized daily changes for macro series.
        regime_masks:  Dict of {"regime_name": bool_series} from build_regime_masks.
        min_t_stat:    Minimum |t-statistic| to keep a finding.

    Returns:
        List of sensitivity dicts with regime field set to the regime name.
    """
    findings: list[dict[str, Any]] = []

    for regime_name, mask in regime_masks.items():
        # Subset data to this regime's days
        regime_idx = mask.index[mask]
        regime_returns = asset_returns.reindex(regime_idx).dropna(how="all")
        regime_macro = macro_levels.reindex(regime_idx).dropna(how="all")

        if len(regime_returns) < MIN_REGIME_OBS:
            log.info("Regime '%s': too few observations (%d) — skipping",
                     regime_name, len(regime_returns))
            continue

        regime_count = 0
        for factor_group, (factor_id, _) in SENSITIVITY_FACTORS.items():
            if factor_id not in regime_macro.columns:
                continue

            factor = regime_macro[factor_id].dropna()

            for asset in regime_returns.columns:
                result = _regress_single(
                    regime_returns[asset], factor,
                    asset, factor_id, factor_group,
                    min_t_stat, regime=regime_name,
                )
                if result is not None:
                    findings.append(result)
                    regime_count += 1

        log.info("Regime '%s': %d significant exposures", regime_name, regime_count)

    return findings
