"""Experiment ranking, promotion explanation, and review utilities.

All functions here are transparent and auditable: they read from the
``ExperimentRegistry``, compute deterministic scores, and explain their
reasoning in human-readable terms.  No auto-promotion logic is executed here;
outputs are purely informational.

Promotion criteria (all must pass for a RECOMMENDED flag)::

    accuracy   ≥ 0.55   — above random for direction
    hit_rate   ≥ 0.52   — backtest: fraction of profitable trades
    beat_bench > 0.00   — cumulative return exceeds buy-and-hold

Advisory criteria (contribute to composite score but not hard gates)::

    auc        ≥ 0.52   — discrimination ability
    sharpe     > 0.00   — positive risk-adjusted return

Usage::

    from ml.comparison.ranking import rank_experiments, explain_promotion, top_n

    ranked = rank_experiments(registry, dataset_version="abc123def456")
    for rank, rec in ranked:
        print(rank, rec.model_type, rec.metrics.get("accuracy"))

    detail = explain_promotion(registry.get(some_experiment_id))
    print(detail["summary"])
"""
from __future__ import annotations

import logging
from typing import Any

from ml.patterns.train_mlp import (
    PROMO_MIN_ACCURACY,
    PROMO_MIN_BEAT_BM,
    PROMO_MIN_HIT_RATE,
)
from ml.registry.experiment_registry import ExperimentRecord, ExperimentRegistry, ExperimentStatus

log = logging.getLogger(__name__)

# ── Promotion thresholds ──────────────────────────────────────────────────────

PROMOTION_CRITERIA: dict[str, dict[str, Any]] = {
    "accuracy": {
        "threshold":   PROMO_MIN_ACCURACY,
        "higher":      True,
        "gate":        True,   # must pass for overall recommendation
        "source":      "metrics",
        "description": "Out-of-sample accuracy (fraction of correct direction calls)",
    },
    "auc": {
        "threshold":   0.52,
        "higher":      True,
        "gate":        False,  # advisory only
        "source":      "metrics",
        "description": "Area under the ROC curve (discrimination ability)",
    },
    "hit_rate": {
        "threshold":   PROMO_MIN_HIT_RATE,
        "higher":      True,
        "gate":        True,
        "source":      "backtest",
        "description": "Fraction of profitable trades in the backtest window",
    },
    "beat_benchmark": {
        "threshold":   PROMO_MIN_BEAT_BM,
        "higher":      True,
        "gate":        True,
        "source":      "backtest_derived",
        "description": "Cumulative return exceeds buy-and-hold benchmark return",
    },
    "sharpe": {
        "threshold":   0.0,
        "higher":      True,
        "gate":        False,  # advisory only
        "source":      "backtest",
        "description": "Annualised Sharpe ratio (positive = risk-adjusted profit)",
    },
}

# Default weights for composite_score
DEFAULT_WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "auc":      0.25,
    "hit_rate": 0.25,
    "sharpe":   0.20,
}


# ── Composite scoring ─────────────────────────────────────────────────────────

def composite_score(
    record:  ExperimentRecord,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted composite score from an ``ExperimentRecord``.

    Combines accuracy, AUC, hit-rate, and Sharpe ratio into a single
    scalar in approximately ``[0, 1]`` for ranking.  Sharpe is clamped to
    ``[-3, 3]`` and rescaled to ``[0, 1]``; the other metrics are already
    in ``[0, 1]``.

    Args:
        record:  ``ExperimentRecord`` from the registry.
        weights: Optional weight override dict.  Keys: ``accuracy``, ``auc``,
                 ``hit_rate``, ``sharpe``.  Defaults if not provided.

    Returns:
        Float in approximately ``[0, 1]``.  Higher is better.
        Returns ``0.0`` if the record has no metrics.
    """
    w  = weights or DEFAULT_WEIGHTS
    m  = record.metrics or {}
    bt = record.backtest

    accuracy = float(m.get("accuracy", 0.0))
    auc      = float(m.get("auc",      0.0))
    hit_rate = float(bt.hit_rate if bt else 0.0)
    sharpe_norm = _normalise_sharpe(float(bt.sharpe if bt else 0.0))

    return round(
        w.get("accuracy", 0) * accuracy
        + w.get("auc",     0) * auc
        + w.get("hit_rate",0) * hit_rate
        + w.get("sharpe",  0) * sharpe_norm,
        6,
    )


def composite_score_from_result(
    result:  Any,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute composite score directly from a ``PipelineResult``.

    Used during comparison runs before results are persisted to the registry
    (``PipelineResult`` is not an ``ExperimentRecord`` but has the same
    metric and backtest fields).

    Args:
        result:  ``PipelineResult`` from any pipeline.
        weights: Optional weight override.

    Returns:
        Float in approximately ``[0, 1]``.  Higher is better.
    """
    w  = weights or DEFAULT_WEIGHTS
    m  = getattr(result, "metrics", {}) or {}
    bt = getattr(result, "backtest_summary", {}) or {}

    accuracy = float(m.get("accuracy", 0.0))
    auc      = float(m.get("auc",      0.0))
    hit_rate = float(bt.get("hit_rate", 0.0))
    sharpe_norm = _normalise_sharpe(float(bt.get("sharpe", 0.0)))

    return round(
        w.get("accuracy", 0) * accuracy
        + w.get("auc",     0) * auc
        + w.get("hit_rate",0) * hit_rate
        + w.get("sharpe",  0) * sharpe_norm,
        6,
    )


def _normalise_sharpe(sharpe: float) -> float:
    """Rescale Sharpe from [-3, 3] to [0, 1]."""
    return (min(max(sharpe, -3.0), 3.0) + 3.0) / 6.0


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_experiments(
    registry:         ExperimentRegistry,
    dataset_version:  str | None   = None,
    model_type:       str | None   = None,
    require_backtest: bool         = True,
    n:                int | None   = None,
    weights:          dict[str, float] | None = None,
) -> list[tuple[int, ExperimentRecord]]:
    """Rank completed experiments by composite score.

    Queries the registry for ``COMPLETED`` and ``PROMOTED`` experiments,
    applies optional filters, sorts by composite score, and returns a
    ranked list.

    Args:
        registry:         ``ExperimentRegistry`` to query.
        dataset_version:  Filter to a specific 12-char dataset version hash.
                          Only experiments whose ``dataset_info`` contains a
                          matching ``dataset_version`` key are included.
        model_type:       Optional model type filter (``'mlp'``, ``'logistic'``,
                          ``'lstm'``).
        require_backtest: Exclude experiments without an attached backtest.
        n:                Return at most *n* results.
        weights:          Composite score weight override.

    Returns:
        List of ``(rank, ExperimentRecord)`` tuples sorted best-first.
        Rank starts at 1.  Empty list if no experiments match.
    """
    records: list[ExperimentRecord] = []
    for status in (ExperimentStatus.COMPLETED, ExperimentStatus.PROMOTED):
        records += registry.filter(model_type=model_type, status=status)

    if dataset_version:
        records = [
            r for r in records
            if r.dataset_info.get("dataset_version") == dataset_version
            or r.dataset_info.get("comparison_group_version") == dataset_version
        ]

    if require_backtest:
        records = [r for r in records if r.backtest is not None]

    sorted_records = sorted(
        records,
        key=lambda r: composite_score(r, weights),
        reverse=True,
    )
    if n is not None:
        sorted_records = sorted_records[:n]

    return [(i + 1, r) for i, r in enumerate(sorted_records)]


# ── Promotion explanation ─────────────────────────────────────────────────────

def explain_promotion(record: ExperimentRecord) -> dict[str, Any]:
    """Explain in detail why an experiment is or is not promotion-worthy.

    Checks every promotion criterion independently — both gate criteria
    (required for promotion) and advisory criteria (contribute to score) —
    and produces a human-readable summary.

    Args:
        record: ``ExperimentRecord`` to evaluate.

    Returns:
        Dict with keys:
        - ``experiment_id``
        - ``model_type``
        - ``overall_pass`` — True only when all *gate* criteria pass
        - ``criteria``     — list of per-criterion dicts
        - ``summary``      — one-line human-readable verdict
        - ``composite_score``
    """
    m  = record.metrics or {}
    bt = record.backtest
    criteria_results: list[dict[str, Any]] = []

    # ── Accuracy ──────────────────────────────────────────────────────────
    _check_metric(criteria_results, "accuracy",
                  float(m.get("accuracy", 0.0)),
                  PROMOTION_CRITERIA["accuracy"])

    # ── AUC ───────────────────────────────────────────────────────────────
    _check_metric(criteria_results, "auc",
                  float(m.get("auc", 0.0)),
                  PROMOTION_CRITERIA["auc"])

    if bt is not None:
        # ── Hit rate ──────────────────────────────────────────────────────
        _check_metric(criteria_results, "hit_rate",
                      float(bt.hit_rate),
                      PROMOTION_CRITERIA["hit_rate"])

        # ── Beat benchmark ────────────────────────────────────────────────
        excess = float(bt.cumulative_return) - float(bt.benchmark_return)
        beat_pass = excess > PROMO_MIN_BEAT_BM
        criteria_results.append({
            "criterion":   "beat_benchmark",
            "value":       round(excess, 4),
            "threshold":   PROMO_MIN_BEAT_BM,
            "passed":      beat_pass,
            "gate":        True,
            "description": PROMOTION_CRITERIA["beat_benchmark"]["description"],
            "message": (
                f"return {bt.cumulative_return*100:+.1f}% vs benchmark "
                f"{bt.benchmark_return*100:+.1f}% "
                f"({'beats' if beat_pass else 'does not beat'})"
            ),
        })

        # ── Sharpe ────────────────────────────────────────────────────────
        _check_metric(criteria_results, "sharpe",
                      float(bt.sharpe),
                      PROMOTION_CRITERIA["sharpe"],
                      fmt=".2f")
    else:
        for criterion in ("hit_rate", "beat_benchmark", "sharpe"):
            criteria_results.append({
                "criterion":   criterion,
                "value":       None,
                "threshold":   PROMOTION_CRITERIA[criterion]["threshold"],
                "passed":      False,
                "gate":        PROMOTION_CRITERIA[criterion]["gate"],
                "description": PROMOTION_CRITERIA[criterion]["description"],
                "message":     "No backtest attached — criterion cannot be evaluated",
            })

    # Overall: all gate criteria must pass
    gate_results = [cr for cr in criteria_results if cr.get("gate", False)]
    overall_pass = all(cr["passed"] for cr in gate_results)
    failed_gates = [cr["criterion"] for cr in gate_results if not cr["passed"]]
    passed_count = sum(1 for cr in criteria_results if cr["passed"])

    summary = (
        f"PROMOTION RECOMMENDED ({passed_count}/{len(criteria_results)} criteria passed)"
        if overall_pass
        else f"NOT RECOMMENDED — failing gate criteria: {', '.join(failed_gates)}"
    )

    return {
        "experiment_id":   record.experiment_id,
        "model_type":      record.model_type,
        "overall_pass":    overall_pass,
        "criteria":        criteria_results,
        "summary":         summary,
        "composite_score": composite_score(record),
    }


def _check_metric(
    out:       list[dict[str, Any]],
    name:      str,
    value:     float,
    spec:      dict[str, Any],
    fmt:       str = ".3f",
) -> None:
    """Append a criterion evaluation dict to ``out``."""
    threshold = spec["threshold"]
    passed    = value >= threshold if spec["higher"] else value <= threshold
    cmp_sym   = "≥" if spec["higher"] else "≤"
    out.append({
        "criterion":   name,
        "value":       round(value, 4),
        "threshold":   threshold,
        "passed":      passed,
        "gate":        spec.get("gate", False),
        "description": spec["description"],
        "message":     f"{value:{fmt}} {cmp_sym if passed else ('< ' if spec['higher'] else '> ')}{threshold}",
    })


# ── Dataset version comparison ────────────────────────────────────────────────

def compare_dataset_version(
    registry:        ExperimentRegistry,
    dataset_version: str,
) -> list[ExperimentRecord]:
    """Return all experiments that share a given dataset version.

    The version hash is stored in ``ExperimentRecord.dataset_info`` under
    the key ``dataset_version``.  Two experiments sharing a version used
    identical features, target, and split fractions (and their results are
    directly comparable).

    LSTM experiments use a sequence dataset whose ``dataset_version`` hash
    differs from the flat dataset version.  They are included when their
    ``comparison_group_version`` matches the requested ``dataset_version``.
    This key is written by ``run_lstm_pipeline`` when a flat ``DatasetMeta``
    is provided to the comparison runner.

    Args:
        registry:        ``ExperimentRegistry`` to query.
        dataset_version: 12-character hex hash from ``DatasetMeta`` — either
                         the flat version (finds baseline, MLP, and LSTM) or
                         the sequence version (finds only LSTM).

    Returns:
        All matching records sorted most-recent-first.
    """
    return [
        r for r in registry.all()
        if r.dataset_info.get("dataset_version") == dataset_version
        or r.dataset_info.get("comparison_group_version") == dataset_version
    ]


# ── Top-N by metric ───────────────────────────────────────────────────────────

_BACKTEST_ATTRS = frozenset({
    "hit_rate", "sharpe", "cumulative_return",
    "annualised_return", "max_drawdown", "trade_count",
    "benchmark_return",
})


def top_n(
    registry:         ExperimentRegistry,
    n:                int          = 5,
    metric:           str          = "accuracy",
    model_type:       str | None   = None,
    require_backtest: bool         = True,
) -> list[ExperimentRecord]:
    """Return the top N experiments ranked by a single metric.

    Accepts both evaluation metrics (``accuracy``, ``auc``, ``f1``, …) and
    backtest attributes (``hit_rate``, ``sharpe``, ``cumulative_return``, …).
    For ``max_drawdown``, higher (less negative) is treated as better.

    Args:
        registry:         ``ExperimentRegistry`` to query.
        n:                Maximum number of results to return.
        metric:           Metric name.  Evaluation: ``'accuracy'``, ``'auc'``,
                          ``'f1'``, ``'precision'``, ``'recall'``.
                          Backtest: ``'hit_rate'``, ``'sharpe'``,
                          ``'cumulative_return'``, ``'annualised_return'``,
                          ``'max_drawdown'``, ``'trade_count'``.
        model_type:       Optional model type filter.
        require_backtest: Skip experiments without an attached backtest.

    Returns:
        List of at most *n* ``ExperimentRecord`` objects, best-first.
    """
    records: list[ExperimentRecord] = []
    for status in (ExperimentStatus.COMPLETED, ExperimentStatus.PROMOTED):
        records += registry.filter(model_type=model_type, status=status)

    if require_backtest:
        records = [r for r in records if r.backtest is not None]

    def _key(r: ExperimentRecord) -> float:
        if metric in _BACKTEST_ATTRS and r.backtest is not None:
            val = getattr(r.backtest, metric, None)
            if val is not None:
                # max_drawdown: less negative = better → use the value directly
                # (drawdown is already negative, so -inf is worst)
                return float(val) if metric != "max_drawdown" else -abs(float(val))
        return float((r.metrics or {}).get(metric, 0.0))

    return sorted(records, key=_key, reverse=True)[:n]
