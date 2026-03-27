"""Multi-model comparison runner.

Evaluates two or more models on the *same* dataset version and produces a
ranked ``ComparisonResult``.  The dataset is assembled once and passed to
each model pipeline so splits, features, and targets are identical across
all models.

Supported models:
- ``'baseline'`` — logistic regression (fast, always runs)
- ``'mlp'``      — MLP neural network
- ``'lstm'``     — LSTM sequence model (optional; requires sufficient data)

Models that fail (e.g., LSTM skipped due to insufficient sequences) are
logged as warnings and the comparison continues.  A partial comparison
(baseline + MLP) is still a useful result.

Two evaluation modes
--------------------
single-split (default):
    One train/val/test split (70/15/15).  Fast and deterministic.
    Returns a ``ComparisonResult``.

walk-forward:
    Multiple chronological folds.  More evidence of generalisation.
    Returns a ``WalkForwardComparisonResult``.

Usage::

    from ml.comparison.runner import run_comparison
    from ml.validation.walk_forward import WalkForwardConfig

    # Single-split (existing behaviour)
    result = run_comparison(
        symbol         = "AAPL",
        df             = ohlcv_df,
        models         = ("baseline", "mlp"),
        registry       = reg,
        checkpoint_dir = Path("ml/checkpoints"),
        epochs         = 20,
        patience       = 5,
    )
    result.print_summary()

    # Walk-forward mode
    wf_result = run_comparison(
        symbol       = "AAPL",
        df           = ohlcv_df,
        models       = ("baseline", "mlp"),
        registry     = reg,
        walk_forward = True,
        wf_config    = WalkForwardConfig(n_folds=5),
        epochs       = 20,
        patience     = 5,
    )
    wf_result.print_summary()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml.comparison.ranking import composite_score_from_result
from ml.data.dataset_builder import DatasetMeta, assemble_dataset
from ml.patterns.train_mlp import PipelineResult, fetch_price_df
from ml.registry.experiment_registry import ExperimentRegistry

log = logging.getLogger(__name__)

SUPPORTED_MODELS = ("baseline", "mlp", "lstm")

# Lazily imported to avoid circular imports and keep baseline runner fast
_WalkForwardConfig    = None  # ml.validation.walk_forward.WalkForwardConfig
_WalkForwardResult    = None  # ml.validation.wf_runner.WalkForwardResult


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Result of one model within a ``ComparisonResult``.

    Attributes:
        model_type:            ``'baseline'``, ``'mlp'``, or ``'lstm'``.
        experiment_id:         Registry ID of this run.
        metrics:               Evaluation metrics dict.
        backtest_summary:      Serialised backtest dict.
        promotion_recommended: Whether this run meets all promotion thresholds.
        promotion_reasons:     Per-threshold pass/fail messages.
        dataset_version:       12-char version hash of the shared dataset.
        composite_score:       Weighted ranking score (higher = better).
        rank:                  Position in this comparison (1 = best).
    """
    model_type:            str
    experiment_id:         str
    metrics:               dict[str, float]
    backtest_summary:      dict[str, Any]
    promotion_recommended: bool
    promotion_reasons:     list[str]
    dataset_version:       str
    composite_score:       float        = 0.0
    rank:                  int | None   = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "model_type":            self.model_type,
            "experiment_id":         self.experiment_id,
            "metrics":               self.metrics,
            "backtest_summary":      self.backtest_summary,
            "promotion_recommended": self.promotion_recommended,
            "promotion_reasons":     self.promotion_reasons,
            "dataset_version":       self.dataset_version,
            "composite_score":       round(self.composite_score, 4),
            "rank":                  self.rank,
        }


@dataclass
class ComparisonResult:
    """Ranked output of a multi-model comparison run.

    Attributes:
        symbol:          Ticker or dataset identifier.
        dataset_version: Shared dataset version hash (all models used this).
        dataset_meta:    Full metadata for the shared dataset.
        results:         ``ModelResult`` for each model that completed.
        winner:          ``model_type`` of the top-ranked model, or ``None``
                         if the comparison is empty.
        generated_at:    ISO-8601 timestamp.
    """
    symbol:          str
    dataset_version: str
    dataset_meta:    DatasetMeta
    results:         list[ModelResult]
    winner:          str | None
    generated_at:    str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def ranked(self) -> list[ModelResult]:
        """Return results sorted by composite score, best first.

        Returns:
            Sorted copy of ``self.results``.
        """
        return sorted(self.results, key=lambda r: r.composite_score, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "symbol":          self.symbol,
            "dataset_version": self.dataset_version,
            "dataset_meta":    self.dataset_meta.to_dict(),
            "winner":          self.winner,
            "generated_at":    self.generated_at,
            "results":         [r.to_dict() for r in self.ranked()],
        }

    def print_summary(self) -> None:
        """Print a human-readable comparison summary to stdout."""
        sep = "=" * 68
        print(sep)
        print(
            f"  Comparison  |  {self.symbol}  |  "
            f"{self.generated_at[:19].replace('T', ' ')} UTC"
        )
        print(f"  Dataset version : {self.dataset_version}")
        print(
            f"  Rows: {self.dataset_meta.n_rows}  "
            f"(train {self.dataset_meta.n_train} / "
            f"val {self.dataset_meta.n_val} / "
            f"test {self.dataset_meta.n_test})"
        )
        print(f"  Winner          : {self.winner or '(none)'}")
        print(sep)
        for r in self.ranked():
            promo = "[OK] RECOMMENDED" if r.promotion_recommended else "[--] not recommended"
            print(
                f"  #{r.rank}  {r.model_type:10s}  "
                f"score={r.composite_score:.4f}  {promo}"
            )
            m  = r.metrics
            bt = r.backtest_summary
            print(
                f"       acc={m.get('accuracy', 0):.3f}  "
                f"auc={m.get('auc', 0):.3f}  "
                f"f1={m.get('f1', 0):.3f}  "
                f"hit={bt.get('hit_rate', 0):.3f}  "
                f"sharpe={bt.get('sharpe', 0):.2f}  "
                f"ret={bt.get('cumulative_return', 0)*100:+.1f}%"
            )
            print(f"       exp_id={r.experiment_id[:8]}...")
        print(sep)

    def save_summary(self, output_dir: "Path | str | None" = None) -> Path:
        """Write the comparison result to a JSON file.

        The file is named ``comparison_{symbol}_{timestamp}.json`` and written
        to ``output_dir`` (default: ``ml/outputs/comparison/``).

        Args:
            output_dir: Directory to write the summary file.  Created if it
                        does not exist.

        Returns:
            Resolved ``Path`` of the written file.
        """
        if output_dir is None:
            repo_root = Path(__file__).parent.parent.parent
            output_dir = repo_root / "ml" / "outputs" / "comparison"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.generated_at[:19].replace(":", "").replace("T", "_")
        fname = f"comparison_{self.symbol}_{ts}.json"
        path = out_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("ComparisonResult saved: %s", path)
        return path


# ── Walk-forward comparison result ────────────────────────────────────────────

@dataclass
class WalkForwardComparisonResult:
    """Ranked output of a walk-forward multi-model comparison.

    Replaces ``ComparisonResult`` when ``run_comparison(walk_forward=True)``
    is used.  Contains per-model aggregates and variance-aware promotion
    recommendations instead of single-split metrics.

    Attributes:
        symbol:      Ticker symbol.
        wf_config:   ``WalkForwardConfig`` used for this run.
        aggregates:  Per-model ``FoldAggregate`` keyed by model type.
        promotions:  Per-model ``WalkForwardPromotion`` keyed by model type.
        winner:      Model type with the highest ``mean_composite_score``.
                     ``None`` if no models completed.
        generated_at: ISO-8601 timestamp.
    """
    symbol:       str
    wf_config:    Any   # WalkForwardConfig (typed Any to avoid import cycle)
    aggregates:   dict[str, Any]  # dict[str, FoldAggregate]
    promotions:   dict[str, Any]  # dict[str, WalkForwardPromotion]
    winner:       str | None
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def ranked(self) -> list[tuple[str, Any]]:
        """Return ``(model_type, FoldAggregate)`` pairs sorted best-first.

        Ranking uses ``mean_composite_score`` descending.
        """
        return sorted(
            self.aggregates.items(),
            key=lambda kv: kv[1].mean_composite_score,
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "symbol":       self.symbol,
            "wf_config":    self.wf_config.to_dict(),
            "winner":       self.winner,
            "generated_at": self.generated_at,
            "aggregates":   {k: v.to_dict(include_fold_results=False)
                             for k, v in self.aggregates.items()},
            "promotions":   {k: v.to_dict() for k, v in self.promotions.items()},
        }

    def print_summary(self) -> None:
        """Print a human-readable walk-forward comparison summary."""
        sep = "=" * 72
        print(sep)
        print(
            f"  Walk-Forward Comparison  |  {self.symbol}  |  "
            f"{self.generated_at[:19].replace('T', ' ')} UTC"
        )
        cfg = self.wf_config
        print(
            f"  Folds: {cfg.n_folds}  window={cfg.window}  "
            f"min_train={cfg.min_train_size}  val_frac={cfg.val_frac}"
        )
        print(f"  Winner: {self.winner or '(none)'}")
        print(sep)
        for model_type, agg in self.ranked():
            promo = self.promotions.get(model_type)
            promo_str = (
                "[OK] RECOMMENDED" if (promo and promo.overall_recommended)
                else "[--] not recommended"
            )
            print(
                f"  {model_type:10s}  "
                f"score={agg.mean_composite_score:.4f}+/-{agg.std_composite_score:.4f}"
                f"  {promo_str}"
            )
            print(
                f"    acc={agg.mean_accuracy:.3f}+/-{agg.std_accuracy:.3f}  "
                f"auc={agg.mean_auc:.3f}+/-{agg.std_auc:.3f}  "
                f"hit={agg.mean_hit_rate:.3f}+/-{agg.std_hit_rate:.3f}  "
                f"sharpe={agg.mean_sharpe:.2f}+/-{agg.std_sharpe:.2f}"
            )
            print(
                f"    beat_bm={agg.n_folds_beat_benchmark}/{agg.n_folds}  "
                f"fold_promo={agg.n_folds_promo_recommended}/{agg.n_folds}"
            )
        print(sep)

    def save_summary(self, output_dir: "Path | str | None" = None) -> Path:
        """Write the walk-forward comparison result to a JSON file.

        The file is named ``wf_{symbol}_{timestamp}.json`` and written to
        ``output_dir`` (default: ``ml/outputs/walk_forward/`` relative to the
        repository root, or the current working directory if that path does not
        exist).

        Args:
            output_dir: Directory to write the summary file.  Created if it
                        does not exist.  Falls back to
                        ``ml/outputs/walk_forward/`` when ``None``.

        Returns:
            Resolved ``Path`` of the written file.
        """
        if output_dir is None:
            # Locate repository root relative to this file's location.
            repo_root = Path(__file__).parent.parent.parent
            output_dir = repo_root / "ml" / "outputs" / "walk_forward"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.generated_at[:19].replace(":", "").replace("T", "_")
        fname = f"wf_{self.symbol}_{ts}.json"
        path = out_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("WalkForwardComparisonResult saved: %s", path)
        return path


# ── Runner ────────────────────────────────────────────────────────────────────

def run_comparison(
    symbol:         str,
    df:             pd.DataFrame | None         = None,
    models:         tuple[str, ...] | list[str] = ("baseline", "mlp"),
    registry:       ExperimentRegistry | None   = None,
    checkpoint_dir: Path | None                 = None,
    epochs:         int                         = 20,
    patience:       int                         = 5,
    tags:           list[str] | None            = None,
    period:         str                         = "3y",
    target_horizon: int                         = 1,
    walk_forward:   bool                        = False,
    wf_config:      Any                         = None,
) -> "ComparisonResult | WalkForwardComparisonResult":
    """Evaluate multiple models on the same dataset version.

    Assembles the shared flat dataset once, then runs each requested model
    pipeline.  The MLP and logistic baseline receive the *exact same*
    feature arrays and splits.  The LSTM gets the same raw OHLCV data with
    its own sequence assembly (same time window, compatible version metadata).

    If a model fails (import error, insufficient data, training error), it is
    skipped with a warning and the comparison continues.  A partial result
    with at least one model is still returned.

    Args:
        symbol:         Ticker symbol.
        df:             Pre-built OHLCV DataFrame; fetched from yfinance
                        if ``None``.
        models:         Sequence of model keys to compare.  Supported:
                        ``'baseline'``, ``'mlp'``, ``'lstm'``.
        registry:       Shared ``ExperimentRegistry``; created locally if
                        ``None``.
        checkpoint_dir: Directory for all model checkpoints.
        epochs:         Training epochs for MLP and LSTM.
        patience:       Early-stopping patience.
        tags:           Registry tags applied to every experiment in this run.
        period:         yfinance period string (ignored when ``df`` is given).
        target_horizon: Bars ahead to predict.
        walk_forward:   When ``True``, run walk-forward cross-validation
                        instead of a single train/val/test split and return a
                        ``WalkForwardComparisonResult``.
        wf_config:      ``WalkForwardConfig`` for the walk-forward path.
                        Defaults are used if ``None`` and ``walk_forward=True``.

    Returns:
        ``ComparisonResult`` (single-split) or ``WalkForwardComparisonResult``
        (walk-forward).

    Raises:
        RuntimeError: If every requested model fails.
    """
    reg = registry if registry is not None else ExperimentRegistry()

    # ── Walk-forward mode: delegate to wf_runner ──────────────────────────
    if walk_forward:
        return _run_comparison_walk_forward(
            symbol         = symbol,
            df             = df,
            models         = list(models),
            registry       = reg,
            checkpoint_dir = checkpoint_dir,
            epochs         = epochs,
            patience       = patience,
            tags           = tags,
            period         = period,
            target_horizon = target_horizon,
            wf_config      = wf_config,
        )

    # ── 1. Assemble the shared flat dataset ───────────────────────────────
    if df is None:
        df = fetch_price_df(symbol, period=period)

    data, meta = assemble_dataset(df, symbol=symbol, target_horizon=target_horizon)
    comparison_tags = list(tags or []) + [f"cmp_v{meta.dataset_version}"]

    log.info(
        "Comparison: symbol=%s  dataset_version=%s  models=%s",
        symbol, meta.dataset_version, list(models),
    )

    # ── 2. Run each model and collect results ─────────────────────────────
    model_results: list[ModelResult] = []

    for model_key in models:
        model_key = model_key.lower()
        if model_key not in SUPPORTED_MODELS:
            log.warning("Unknown model %r — skipping", model_key)
            continue

        log.info("Running %s pipeline…", model_key)
        try:
            result = _run_model(
                model_key      = model_key,
                symbol         = symbol,
                df             = df,
                data           = data,
                meta           = meta,
                reg            = reg,
                checkpoint_dir = checkpoint_dir,
                epochs         = epochs,
                patience       = patience,
                comparison_tags = comparison_tags,
                target_horizon = target_horizon,
            )
            mr = _to_model_result(result, model_key, meta.dataset_version)
            model_results.append(mr)
            log.info(
                "%s complete: score=%.4f  promoted=%s",
                model_key, mr.composite_score, mr.promotion_recommended,
            )
        except Exception as exc:
            log.warning(
                "Model %r failed and will be excluded from comparison: %s",
                model_key, exc, exc_info=True,
            )

    if not model_results:
        raise RuntimeError(
            f"All models failed for symbol={symbol!r}. "
            "Check logs for individual failure reasons."
        )

    # ── 3. Rank results ───────────────────────────────────────────────────
    ranked = sorted(model_results, key=lambda r: r.composite_score, reverse=True)
    for i, r in enumerate(ranked, start=1):
        r.rank = i

    return ComparisonResult(
        symbol          = symbol,
        dataset_version = meta.dataset_version,
        dataset_meta    = meta,
        results         = ranked,
        winner          = ranked[0].model_type,
        generated_at    = datetime.now(tz=timezone.utc).isoformat(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_model(
    model_key:       str,
    symbol:          str,
    df:              pd.DataFrame,
    data:            dict[str, Any],
    meta:            DatasetMeta,
    reg:             ExperimentRegistry,
    checkpoint_dir:  Path | None,
    epochs:          int,
    patience:        int,
    comparison_tags: list[str],
    target_horizon:  int,
) -> PipelineResult:
    """Dispatch to the appropriate pipeline function.

    Args:
        model_key: One of ``'baseline'``, ``'mlp'``, ``'lstm'``.
        All other args forwarded to the pipeline.

    Returns:
        ``PipelineResult`` from the pipeline.
    """
    if model_key == "baseline":
        from ml.patterns.baseline import run_baseline_pipeline
        return run_baseline_pipeline(
            symbol         = symbol,
            df             = df,
            dataset        = data,
            dataset_meta   = meta,
            registry       = reg,
            checkpoint_dir = checkpoint_dir,
            tags           = comparison_tags,
        )

    if model_key == "mlp":
        from ml.patterns.train_mlp import run_pipeline as _mlp_run
        return _mlp_run(
            symbol         = symbol,
            df             = df,
            dataset        = data,
            dataset_meta   = meta,
            registry       = reg,
            checkpoint_dir = checkpoint_dir,
            epochs         = epochs,
            patience       = patience,
            tags           = comparison_tags,
            target_horizon = target_horizon,
        )

    if model_key == "lstm":
        from ml.patterns.train_lstm import LSTMConfig, run_lstm_pipeline
        return run_lstm_pipeline(
            symbol         = symbol,
            df             = df,
            dataset_meta   = meta,
            registry       = reg,
            checkpoint_dir = checkpoint_dir,
            config         = LSTMConfig(
                epochs   = epochs,
                patience = patience,
            ),
            tags           = comparison_tags,
            target_horizon = target_horizon,
        )

    raise ValueError(f"Unknown model key: {model_key!r}")


def _run_comparison_walk_forward(
    symbol:         str,
    df:             pd.DataFrame | None,
    models:         list[str],
    registry:       ExperimentRegistry,
    checkpoint_dir: Path | None,
    epochs:         int,
    patience:       int,
    tags:           list[str] | None,
    period:         str,
    target_horizon: int,
    wf_config:      Any,
) -> WalkForwardComparisonResult:
    """Internal implementation for walk-forward comparison mode.

    Calls ``run_walk_forward`` from ``ml.validation.wf_runner`` and
    wraps the results in a ``WalkForwardComparisonResult``.
    """
    from ml.validation.walk_forward import WalkForwardConfig
    from ml.validation.wf_aggregation import (
        FoldAggregate,
        WalkForwardPromotion,
        aggregate_folds,
        wf_promotion_recommend,
    )
    from ml.validation.wf_runner import run_walk_forward

    cfg = wf_config if wf_config is not None else WalkForwardConfig()

    if df is None:
        df = fetch_price_df(symbol, period=period)

    wf_results = run_walk_forward(
        symbol         = symbol,
        df             = df,
        models         = models,
        config         = cfg,
        registry       = registry,
        checkpoint_dir = checkpoint_dir,
        epochs         = epochs,
        patience       = patience,
        tags           = tags,
        target_horizon = target_horizon,
    )

    if not wf_results:
        raise RuntimeError(
            f"All walk-forward models failed for symbol={symbol!r}. "
            "Check logs for individual failure reasons."
        )

    aggregates:  dict[str, FoldAggregate]      = {}
    promotions:  dict[str, WalkForwardPromotion] = {}

    for model_type, wf in wf_results.items():
        if not wf.fold_results:
            log.warning("Model %r produced no completed folds — excluded.", model_type)
            continue
        agg   = aggregate_folds(wf.fold_results, model_type)
        promo = wf_promotion_recommend(agg)
        aggregates[model_type] = agg
        promotions[model_type] = promo

    if not aggregates:
        raise RuntimeError(
            f"No models produced valid fold results for symbol={symbol!r}."
        )

    winner = max(aggregates, key=lambda m: aggregates[m].mean_composite_score)

    return WalkForwardComparisonResult(
        symbol       = symbol,
        wf_config    = cfg,
        aggregates   = aggregates,
        promotions   = promotions,
        winner       = winner,
        generated_at = datetime.now(tz=timezone.utc).isoformat(),
    )


def _to_model_result(
    result:          PipelineResult,
    model_type:      str,
    dataset_version: str,
) -> ModelResult:
    """Convert a ``PipelineResult`` to a ``ModelResult``.

    Args:
        result:          ``PipelineResult`` from any pipeline.
        model_type:      Model identifier string.
        dataset_version: Shared dataset version hash.

    Returns:
        ``ModelResult`` with composite score computed.
    """
    return ModelResult(
        model_type            = model_type,
        experiment_id         = result.experiment_id,
        metrics               = result.metrics,
        backtest_summary      = result.backtest_summary,
        promotion_recommended = result.promotion_recommended,
        promotion_reasons     = result.promotion_reasons,
        dataset_version       = dataset_version,
        composite_score       = composite_score_from_result(result),
    )
