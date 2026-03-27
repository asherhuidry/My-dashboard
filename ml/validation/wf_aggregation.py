"""Fold-level aggregation and variance-aware promotion for walk-forward results.

Aggregates per-fold metrics into mean/std/min/max summaries and evaluates
whether a model meets walk-forward promotion criteria.  All logic is
transparent and deterministic — no auto-promotion is performed here.

Walk-forward promotion criteria
---------------------------------
All gate criteria must pass for ``overall_recommended = True``:

    mean_accuracy    ≥ 0.55      — average accuracy across folds
    mean_hit_rate    ≥ 0.52      — average backtest hit rate across folds
    n_folds_beat_bm  ≥ ceil(n/2) — beats buy-and-hold on majority of folds

Advisory (inform the score but do not gate):

    mean_auc         ≥ 0.52      — average discrimination ability
    std_accuracy     ≤ 0.08      — consistency penalty flag
    mean_sharpe      > 0.00      — positive risk-adjusted return on average

Usage::

    from ml.validation.wf_runner import run_walk_forward
    from ml.validation.wf_aggregation import (
        aggregate_folds, wf_promotion_recommend,
        WalkForwardPromotionConfig,
    )

    results = run_walk_forward("AAPL", df=ohlcv_df, models=("baseline", "mlp"))
    for model_type, wf in results.items():
        agg   = aggregate_folds(wf.fold_results, model_type)
        promo = wf_promotion_recommend(agg)
        print(promo.summary)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.patterns.train_mlp import PROMO_MIN_ACCURACY, PROMO_MIN_HIT_RATE
from ml.validation.wf_runner import FoldResult


# ── Promotion configuration ────────────────────────────────────────────────────

@dataclass
class WalkForwardPromotionConfig:
    """Thresholds for walk-forward promotion criteria.

    Attributes:
        min_mean_accuracy:   Gate — mean accuracy across folds must reach this.
        min_mean_hit_rate:   Gate — mean backtest hit rate must reach this.
        min_folds_beat_bm:   Gate — number of folds where the model beats the
                             buy-and-hold benchmark.  ``None`` defaults to the
                             ceiling of half the available folds.
        max_std_accuracy:    Advisory — flags high variance in accuracy.  Not a
                             hard gate by default; set ``std_gate=True`` to make
                             it one.
        min_mean_auc:        Advisory — minimum mean AUC.
        min_mean_sharpe:     Advisory — mean Sharpe must be positive.
        std_is_gate:         When ``True``, the ``max_std_accuracy`` check
                             becomes a hard gate (blocks promotion).
    """
    min_mean_accuracy: float = PROMO_MIN_ACCURACY   # 0.55
    min_mean_hit_rate: float = PROMO_MIN_HIT_RATE   # 0.52
    min_folds_beat_bm: int | None = None            # defaults to ceil(n_folds / 2)
    max_std_accuracy:  float = 0.08
    min_mean_auc:      float = 0.52
    min_mean_sharpe:   float = 0.0
    std_is_gate:       bool  = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_mean_accuracy": self.min_mean_accuracy,
            "min_mean_hit_rate": self.min_mean_hit_rate,
            "min_folds_beat_bm": self.min_folds_beat_bm,
            "max_std_accuracy":  self.max_std_accuracy,
            "min_mean_auc":      self.min_mean_auc,
            "min_mean_sharpe":   self.min_mean_sharpe,
            "std_is_gate":       self.std_is_gate,
        }


# ── Aggregate result ───────────────────────────────────────────────────────────

@dataclass
class FoldAggregate:
    """Aggregated statistics over all completed walk-forward folds.

    All float fields are rounded to 4 decimal places for readability.

    Attributes:
        model_type:              Model identifier.
        n_folds:                 Number of successfully completed folds.
        mean_accuracy:           Mean out-of-sample accuracy.
        std_accuracy:            Standard deviation of accuracy across folds.
        min_accuracy:            Minimum accuracy (worst fold).
        max_accuracy:            Maximum accuracy (best fold).
        mean_auc:                Mean AUC across folds.
        std_auc:                 Standard deviation of AUC.
        mean_hit_rate:           Mean backtest hit rate across folds.
        std_hit_rate:            Standard deviation of hit rate.
        mean_sharpe:             Mean Sharpe ratio across folds.
        std_sharpe:              Standard deviation of Sharpe.
        mean_cumulative_return:  Mean cumulative return across folds.
        std_cumulative_return:   Standard deviation of cumulative return.
        mean_composite_score:    Mean composite ranking score across folds.
        std_composite_score:     Standard deviation of composite score.
        n_folds_beat_benchmark:  Folds where strategy beat buy-and-hold.
        n_folds_promo_recommended: Folds where single-fold promotion gates passed.
        fold_results:            Raw per-fold results (for deeper inspection).
    """
    model_type:              str
    n_folds:                 int
    mean_accuracy:           float
    std_accuracy:            float
    min_accuracy:            float
    max_accuracy:            float
    mean_auc:                float
    std_auc:                 float
    mean_hit_rate:           float
    std_hit_rate:            float
    mean_sharpe:             float
    std_sharpe:              float
    mean_cumulative_return:  float
    std_cumulative_return:   float
    mean_composite_score:    float
    std_composite_score:     float
    n_folds_beat_benchmark:  int
    n_folds_promo_recommended: int
    fold_results:            list[FoldResult] = field(default_factory=list)

    def to_dict(self, include_fold_results: bool = True) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Args:
            include_fold_results: Whether to embed the full per-fold results.
                                  Set ``False`` for compact serialisation.
        """
        d: dict[str, Any] = {
            "model_type":               self.model_type,
            "n_folds":                  self.n_folds,
            "mean_accuracy":            self.mean_accuracy,
            "std_accuracy":             self.std_accuracy,
            "min_accuracy":             self.min_accuracy,
            "max_accuracy":             self.max_accuracy,
            "mean_auc":                 self.mean_auc,
            "std_auc":                  self.std_auc,
            "mean_hit_rate":            self.mean_hit_rate,
            "std_hit_rate":             self.std_hit_rate,
            "mean_sharpe":              self.mean_sharpe,
            "std_sharpe":               self.std_sharpe,
            "mean_cumulative_return":   self.mean_cumulative_return,
            "std_cumulative_return":    self.std_cumulative_return,
            "mean_composite_score":     self.mean_composite_score,
            "std_composite_score":      self.std_composite_score,
            "n_folds_beat_benchmark":   self.n_folds_beat_benchmark,
            "n_folds_promo_recommended": self.n_folds_promo_recommended,
        }
        if include_fold_results:
            d["fold_results"] = [fr.to_dict() for fr in self.fold_results]
        return d


def _composite_from_fold(fr: FoldResult) -> float:
    """Compute composite score for a single fold result."""
    from ml.comparison.ranking import composite_score_from_result

    class _Proxy:
        """Thin proxy so composite_score_from_result works on FoldResult."""
        def __init__(self, fr: FoldResult) -> None:
            self.metrics          = fr.metrics
            self.backtest_summary = fr.backtest_summary

    return composite_score_from_result(_Proxy(fr))


def aggregate_folds(
    fold_results: list[FoldResult],
    model_type:   str,
) -> FoldAggregate:
    """Aggregate walk-forward fold results into summary statistics.

    Args:
        fold_results: List of ``FoldResult`` for one model (from one
                      ``WalkForwardResult``).
        model_type:   Model identifier string (used for labelling).

    Returns:
        ``FoldAggregate`` with mean/std/min/max statistics and pass/fail counts.

    Raises:
        ValueError: If ``fold_results`` is empty.
    """
    if not fold_results:
        raise ValueError("fold_results is empty — nothing to aggregate.")

    def _metric(key: str) -> np.ndarray:
        return np.array([fr.metrics.get(key, 0.0) for fr in fold_results], dtype=float)

    def _bt(key: str) -> np.ndarray:
        return np.array([fr.backtest_summary.get(key, 0.0) for fr in fold_results], dtype=float)

    accs       = _metric("accuracy")
    aucs       = _metric("auc")
    hit_rates  = _bt("hit_rate")
    sharpes    = _bt("sharpe")
    cum_rets   = _bt("cumulative_return")
    bm_rets    = _bt("benchmark_return")
    composites = np.array([_composite_from_fold(fr) for fr in fold_results], dtype=float)

    n_beat_bm = int(np.sum(cum_rets > bm_rets))
    n_promo   = sum(1 for fr in fold_results if fr.promotion_recommended)

    def _r4(x: float) -> float:
        return round(float(x), 4)

    return FoldAggregate(
        model_type              = model_type,
        n_folds                 = len(fold_results),
        mean_accuracy           = _r4(accs.mean()),
        std_accuracy            = _r4(accs.std(ddof=0)),
        min_accuracy            = _r4(accs.min()),
        max_accuracy            = _r4(accs.max()),
        mean_auc                = _r4(aucs.mean()),
        std_auc                 = _r4(aucs.std(ddof=0)),
        mean_hit_rate           = _r4(hit_rates.mean()),
        std_hit_rate            = _r4(hit_rates.std(ddof=0)),
        mean_sharpe             = _r4(sharpes.mean()),
        std_sharpe              = _r4(sharpes.std(ddof=0)),
        mean_cumulative_return  = _r4(cum_rets.mean()),
        std_cumulative_return   = _r4(cum_rets.std(ddof=0)),
        mean_composite_score    = _r4(composites.mean()),
        std_composite_score     = _r4(composites.std(ddof=0)),
        n_folds_beat_benchmark  = n_beat_bm,
        n_folds_promo_recommended = n_promo,
        fold_results            = list(fold_results),
    )


# ── Walk-forward promotion ─────────────────────────────────────────────────────

@dataclass
class WalkForwardPromotion:
    """Walk-forward promotion recommendation for one model.

    Attributes:
        model_type:          Model identifier.
        overall_recommended: ``True`` only when **all** gate criteria pass.
        criteria:            Per-criterion evaluation dicts.
        summary:             One-line human-readable verdict.
        n_folds:             Number of folds evaluated.
    """
    model_type:          str
    overall_recommended: bool
    criteria:            list[dict[str, Any]]
    summary:             str
    n_folds:             int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type":          self.model_type,
            "overall_recommended": self.overall_recommended,
            "criteria":            self.criteria,
            "summary":             self.summary,
            "n_folds":             self.n_folds,
        }


def wf_promotion_recommend(
    aggregate: FoldAggregate,
    config:    WalkForwardPromotionConfig | None = None,
) -> WalkForwardPromotion:
    """Evaluate walk-forward promotion criteria for an aggregated result.

    Gate criteria (all must pass for ``overall_recommended = True``):

    - ``mean_accuracy    ≥ config.min_mean_accuracy``
    - ``mean_hit_rate    ≥ config.min_mean_hit_rate``
    - ``n_folds_beat_bm  ≥ config.min_folds_beat_bm``  (defaults to ceil(n/2))
    - ``std_accuracy     ≤ config.max_std_accuracy``    (only if std_is_gate)

    Advisory criteria (reported but do not block promotion):

    - ``mean_auc    ≥ config.min_mean_auc``
    - ``mean_sharpe ≥ config.min_mean_sharpe``
    - ``std_accuracy ≤ config.max_std_accuracy``        (if not a gate)

    Args:
        aggregate: Output of ``aggregate_folds``.
        config:    Promotion thresholds; defaults used if ``None``.

    Returns:
        ``WalkForwardPromotion`` with a clear pass/fail verdict.
    """
    cfg = config or WalkForwardPromotionConfig()
    n   = aggregate.n_folds

    min_beat = cfg.min_folds_beat_bm if cfg.min_folds_beat_bm is not None else math.ceil(n / 2)

    criteria: list[dict[str, Any]] = []

    def _crit(
        name:        str,
        value:       float,
        threshold:   float,
        passed:      bool,
        gate:        bool,
        description: str,
        fmt:         str = ".3f",
    ) -> None:
        cmp = "≥" if passed else "<"
        criteria.append({
            "criterion":   name,
            "value":       round(value, 4),
            "threshold":   threshold,
            "passed":      passed,
            "gate":        gate,
            "description": description,
            "message":     f"{value:{fmt}} {cmp} {threshold}",
        })

    # ── Gate criteria ──────────────────────────────────────────────────────
    _crit(
        "mean_accuracy",
        aggregate.mean_accuracy,
        cfg.min_mean_accuracy,
        aggregate.mean_accuracy >= cfg.min_mean_accuracy,
        gate=True,
        description=(
            f"Mean out-of-sample accuracy across {n} folds must be "
            f"≥ {cfg.min_mean_accuracy}"
        ),
    )

    _crit(
        "mean_hit_rate",
        aggregate.mean_hit_rate,
        cfg.min_mean_hit_rate,
        aggregate.mean_hit_rate >= cfg.min_mean_hit_rate,
        gate=True,
        description=(
            f"Mean backtest hit rate across {n} folds must be "
            f"≥ {cfg.min_mean_hit_rate}"
        ),
    )

    beat_pass = aggregate.n_folds_beat_benchmark >= min_beat
    criteria.append({
        "criterion":   "folds_beat_benchmark",
        "value":       aggregate.n_folds_beat_benchmark,
        "threshold":   min_beat,
        "passed":      beat_pass,
        "gate":        True,
        "description": (
            f"Strategy must beat buy-and-hold benchmark on at least "
            f"{min_beat} of {n} folds"
        ),
        "message": (
            f"{aggregate.n_folds_beat_benchmark}/{n} folds "
            f"{'beat' if beat_pass else 'failed to beat'} benchmark "
            f"(need ≥ {min_beat})"
        ),
    })

    std_pass = aggregate.std_accuracy <= cfg.max_std_accuracy
    _crit(
        "std_accuracy",
        aggregate.std_accuracy,
        cfg.max_std_accuracy,
        std_pass,
        gate=cfg.std_is_gate,
        description=(
            "Standard deviation of accuracy across folds — "
            f"high variance (> {cfg.max_std_accuracy}) signals instability"
        ),
        fmt=".4f",
    )
    # Fix the comparison symbol for std (lower is better)
    criteria[-1]["message"] = (
        f"{aggregate.std_accuracy:.4f} "
        f"{'≤' if std_pass else '>'} {cfg.max_std_accuracy}"
    )

    # ── Advisory criteria ──────────────────────────────────────────────────
    _crit(
        "mean_auc",
        aggregate.mean_auc,
        cfg.min_mean_auc,
        aggregate.mean_auc >= cfg.min_mean_auc,
        gate=False,
        description=f"Mean AUC across folds should be ≥ {cfg.min_mean_auc}",
    )

    sharpe_pass = aggregate.mean_sharpe > cfg.min_mean_sharpe
    _crit(
        "mean_sharpe",
        aggregate.mean_sharpe,
        cfg.min_mean_sharpe,
        sharpe_pass,
        gate=False,
        description="Mean Sharpe ratio across folds should be positive",
        fmt=".3f",
    )

    # ── Overall verdict ───────────────────────────────────────────────────
    gate_results = [c for c in criteria if c["gate"]]
    overall      = all(c["passed"] for c in gate_results)
    failed_gates = [c["criterion"] for c in gate_results if not c["passed"]]
    n_passed     = sum(1 for c in criteria if c["passed"])

    if overall:
        summary = (
            f"WALK-FORWARD PROMOTION RECOMMENDED  "
            f"({n_passed}/{len(criteria)} criteria passed, {n} folds)"
        )
    else:
        summary = (
            f"NOT RECOMMENDED — failing gate criteria: "
            f"{', '.join(failed_gates)}"
        )

    return WalkForwardPromotion(
        model_type          = aggregate.model_type,
        overall_recommended = overall,
        criteria            = criteria,
        summary             = summary,
        n_folds             = n,
    )
