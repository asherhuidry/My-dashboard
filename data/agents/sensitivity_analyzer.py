"""Sensitivity Analyzer — discovers factor exposure edges for the market graph.

Instead of just finding "what moves together" (correlation), this module
discovers "what drives behavior" — which macro factors each asset is
meaningfully exposed to.

For each (asset, macro factor) pair, runs an OLS regression:
    asset_return_t = alpha + beta * factor_change_t + epsilon_t

A statistically significant beta creates a SENSITIVE_TO edge in the graph:
    Asset -[SENSITIVE_TO {factor_group: "rates", beta: -0.42}]-> MacroIndicator

This is more useful than raw correlation because it describes the *nature*
of the relationship — rate-sensitive, inflation-sensitive, dollar-sensitive —
rather than just "these two series co-move."
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

# Minimum |t-stat| to consider an exposure significant
MIN_T_STAT = 2.0


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
) -> dict[str, Any] | None:
    """Run OLS regression for a single (asset, factor) pair.

    Args:
        asset_series:  Daily log returns for the asset.
        factor_series: Standardized daily changes for the factor.
        asset_name:    Ticker symbol.
        factor_id:     FRED series ID.
        factor_group:  Semantic factor label (e.g., "rates").
        min_t_stat:    Minimum |t-stat| threshold.

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
        "lag_days":          0,
        "timestamp":         datetime.now(tz=timezone.utc),
    }
