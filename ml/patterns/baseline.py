"""Logistic regression baseline model for directional price prediction.

Provides the same ``evaluate()`` and ``predict_proba()`` interface as the
MLP so both can participate in the same comparison flow and write comparable
metrics to the experiment registry.

Logistic regression is chosen as the baseline because it is:
- transparent (interpretable coefficients)
- fast (trains in milliseconds on 10k rows)
- a strong signal-quality sanity check — if the MLP cannot beat it, the
  features are not adding value beyond what linear separation can exploit

Usage::

    from ml.patterns.baseline import LogisticBaseline, run_baseline_pipeline

    result = run_baseline_pipeline(symbol="AAPL", df=ohlcv_df, registry=reg,
                                   checkpoint_dir=tmp / "ckpts")
    print(result.metrics)
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.backtest.engine import BacktestConfig, run_backtest
from ml.data.dataset_builder import DatasetMeta, assemble_dataset
from ml.patterns.mlp import FeatureScaler
from ml.patterns.train_mlp import (
    PROMO_MIN_ACCURACY,
    PROMO_MIN_BEAT_BM,
    PROMO_MIN_HIT_RATE,
    PipelineResult,
    fetch_price_df,
)
from ml.registry.experiment_registry import ExperimentRegistry

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "ml" / "checkpoints"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    """Hyperparameters for the logistic regression baseline.

    Attributes:
        C:            Inverse regularisation strength (sklearn convention).
                      Higher = less regularisation.  Default 1.0.
        max_iter:     Maximum solver iterations.  Default 1000.
        class_weight: ``'balanced'`` adjusts weights inversely proportional
                      to class frequencies; ``None`` treats all equally.
        solver:       sklearn solver.  ``'lbfgs'`` is robust for small-medium
                      datasets; switch to ``'saga'`` for very large datasets.
    """
    C:            float       = 1.0
    max_iter:     int         = 1000
    class_weight: str | None  = "balanced"
    solver:       str         = "lbfgs"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaselineConfig":
        """Deserialise from a dict."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Model ─────────────────────────────────────────────────────────────────────

class LogisticBaseline:
    """Logistic regression wrapper with the same interface as the MLP.

    Accepts Z-score-normalised features (via ``FeatureScaler``), fits an
    sklearn ``LogisticRegression``, and exposes ``evaluate()`` and
    ``predict_proba()`` that are API-compatible with the MLP so both can
    participate in the same comparison flow.
    """

    def __init__(self, config: BaselineConfig | None = None) -> None:
        """Initialise the baseline.

        Args:
            config: Hyperparameter config; defaults used if ``None``.
        """
        self.config = config or BaselineConfig()
        self._model: LogisticRegression | None = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, float]:
        """Fit the logistic regression and return training accuracy.

        Args:
            X_train: Feature matrix ``(n_samples, n_features)``, pre-scaled.
            y_train: Binary labels ``(n_samples,)`` as float (0.0 or 1.0).

        Returns:
            Dict with ``train_accuracy``.
        """
        self._model = LogisticRegression(
            C            = self.config.C,
            max_iter     = self.config.max_iter,
            class_weight = self.config.class_weight,
            solver       = self.config.solver,
            random_state = 42,
        )
        self._model.fit(X_train, y_train.astype(int))
        train_acc = float(self._model.score(X_train, y_train.astype(int)))
        log.info("Logistic baseline trained: train_accuracy=%.4f", train_acc)
        return {"train_accuracy": train_acc}

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class (direction up).

        Args:
            X: Feature matrix ``(n_samples, n_features)``, pre-scaled.

        Returns:
            1-D float32 array of probabilities in ``[0, 1]``.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba()")
        return self._model.predict_proba(X)[:, 1].astype(np.float32)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        X:         np.ndarray,
        y:         np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate the fitted model on held-out data.

        Returns the same metric keys as ``ml.patterns.mlp.evaluate`` so
        results are directly comparable in the registry.

        Args:
            X:         Feature matrix ``(n_samples, n_features)``, pre-scaled.
            y:         Binary labels ``(n_samples,)`` as float.
            threshold: Decision threshold for binary predictions.

        Returns:
            Dict with ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc``.
        """
        probs  = self.predict_proba(X)
        preds  = (probs >= threshold).astype(int)
        labels = y.astype(int)
        auc = (
            float(roc_auc_score(labels, probs))
            if len(np.unique(labels)) > 1
            else 0.5
        )
        return {
            "accuracy":  float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall":    float(recall_score(labels, preds, zero_division=0)),
            "f1":        float(f1_score(labels, preds, zero_division=0)),
            "auc":       auc,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Save the fitted model and config to a ``.pkl`` file.

        A ``.json`` sidecar is written alongside for human-readable config
        inspection without loading the pickle.

        Args:
            path: Destination file path (e.g. ``checkpoints/exp_baseline.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"config": self.config.to_dict(), "model": self._model}, fh)
        path.with_suffix(".json").write_text(
            json.dumps({"config": self.config.to_dict()}, indent=2),
            encoding="utf-8",
        )
        log.info("Baseline checkpoint saved: %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "LogisticBaseline":
        """Load a fitted baseline from a ``.pkl`` file written by ``save()``.

        Args:
            path: File path written by ``save()``.

        Returns:
            Fitted ``LogisticBaseline`` ready for inference.
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        instance = cls(config=BaselineConfig.from_dict(payload["config"]))
        instance._model = payload["model"]
        return instance


# ── End-to-end pipeline ───────────────────────────────────────────────────────

def run_baseline_pipeline(
    symbol:             str,
    period:             str                     = "3y",
    target_horizon:     int                     = 1,
    backtest_threshold: float                   = 0.55,
    tags:               list[str] | None        = None,
    notes:              str                     = "",
    df:                 pd.DataFrame | None     = None,
    dataset:            dict[str, Any] | None   = None,
    dataset_meta:       DatasetMeta | None      = None,
    registry:           ExperimentRegistry | None = None,
    checkpoint_dir:     Path | None             = None,
    config:             BaselineConfig | None   = None,
) -> PipelineResult:
    """Run the full logistic baseline pipeline end-to-end.

    Mirrors the structure of ``ml.patterns.train_mlp.run_pipeline`` so the
    two can be called together in a comparison flow.  When a pre-assembled
    ``dataset`` is supplied (by the comparison runner), all models share
    exactly the same features and splits.

    Args:
        symbol:             Ticker symbol.
        period:             yfinance period string (ignored when ``df`` supplied).
        target_horizon:     Bars ahead to predict.
        backtest_threshold: Signal threshold for long entry in the backtest.
        tags:               Registry tags applied to the experiment record.
        notes:              Free-text notes stored in the registry.
        df:                 Pre-built OHLCV DataFrame; fetched from yfinance
                            if both ``df`` and ``dataset`` are ``None``.
        dataset:            Pre-assembled data dict from ``assemble_dataset``.
                            If supplied, ``dataset_meta`` should also be provided.
        dataset_meta:       ``DatasetMeta`` for a pre-assembled dataset.
        registry:           ``ExperimentRegistry`` to log to; created locally
                            if ``None``.
        checkpoint_dir:     Directory for ``.pkl`` and scaler JSON; defaults
                            to ``ml/checkpoints``.
        config:             ``BaselineConfig`` hyperparameters.

    Returns:
        ``PipelineResult`` with metrics, backtest summary, checkpoint paths,
        and promotion recommendation.

    Raises:
        ValueError: If the assembled test set has fewer than 20 samples.
    """
    reg      = registry if registry is not None else ExperimentRegistry()
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    cfg      = config or BaselineConfig()

    # ── 1. Data ───────────────────────────────────────────────────────────
    if dataset is None:
        if df is None:
            df = fetch_price_df(symbol, period=period)
        dataset, dataset_meta = assemble_dataset(
            df, symbol=symbol, target_horizon=target_horizon
        )

    if dataset["n_test"] < 20:
        raise ValueError(
            f"Test set has only {dataset['n_test']} samples — need at least 20."
        )

    ds_info = (
        dataset_meta.to_dataset_info()
        if dataset_meta is not None
        else {"symbol": symbol, "n_train": dataset["n_train"],
              "n_val": dataset["n_val"], "n_test": dataset["n_test"]}
    )

    # ── 2. Scale ──────────────────────────────────────────────────────────
    scaler  = FeatureScaler()
    X_train = scaler.fit_transform(dataset["X_train"])
    X_val   = scaler.transform(dataset["X_val"])
    X_test  = scaler.transform(dataset["X_test"])

    # ── 3. Create experiment record ───────────────────────────────────────
    exp = reg.create(
        name=(
            f"baseline_{symbol.lower()}_"
            f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M')}"
        ),
        model_type   = "logistic",
        hyperparams  = cfg.to_dict(),
        dataset_info = ds_info,
        notes        = notes,
        tags         = tags or [],
    )
    log.info("Baseline experiment started: %s", exp.experiment_id)

    try:
        # ── 4. Train ──────────────────────────────────────────────────────
        model = LogisticBaseline(cfg)
        model.fit(X_train, dataset["y_train"])

        # ── 5. Evaluate on held-out test set ──────────────────────────────
        metrics = model.evaluate(X_test, dataset["y_test"])
        log.info("Baseline evaluation: %s", metrics)

        # ── 6. Save checkpoint + scaler ───────────────────────────────────
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path   = ckpt_dir / f"{exp.experiment_id}_baseline.pkl"
        scaler_path = ckpt_dir / f"{exp.experiment_id}_baseline_scaler.json"
        model.save(ckpt_path)
        scaler_path.write_text(json.dumps(scaler.to_dict()), encoding="utf-8")

        # ── 7. Finish experiment ──────────────────────────────────────────
        reg.finish(
            exp.experiment_id,
            metrics         = metrics,
            checkpoint_path = str(ckpt_path),
        )

        # ── 8. Backtest on test set ───────────────────────────────────────
        scores    = model.predict_proba(X_test)
        signals   = pd.Series(scores, index=dataset["close_test"].index)
        bt_report = run_backtest(
            prices  = dataset["close_test"],
            signals = signals,
            config  = BacktestConfig(threshold=backtest_threshold),
        )
        reg.attach_backtest(exp.experiment_id, bt_report.to_backtest_result())

        # ── 9. Promotion recommendation ───────────────────────────────────
        reasons: list[str] = []
        ok_acc  = metrics.get("accuracy", 0) >= PROMO_MIN_ACCURACY
        ok_hit  = bt_report.hit_rate          >= PROMO_MIN_HIT_RATE
        ok_beat = bt_report.cumulative_return > bt_report.benchmark_return + PROMO_MIN_BEAT_BM

        reasons.append(
            f"accuracy {metrics['accuracy']:.3f} "
            f"{'≥' if ok_acc else '<'} {PROMO_MIN_ACCURACY}"
        )
        reasons.append(
            f"hit rate {bt_report.hit_rate:.3f} "
            f"{'≥' if ok_hit else '<'} {PROMO_MIN_HIT_RATE}"
        )
        reasons.append(
            f"return {bt_report.cumulative_return*100:+.1f}% "
            f"{'beats' if ok_beat else 'does NOT beat'} benchmark "
            f"{bt_report.benchmark_return*100:+.1f}%"
        )

        return PipelineResult(
            experiment_id         = exp.experiment_id,
            symbol                = symbol,
            metrics               = metrics,
            backtest_summary      = bt_report.to_dict(),
            checkpoint_path       = str(ckpt_path),
            scaler_path           = str(scaler_path),
            promotion_recommended = ok_acc and ok_hit and ok_beat,
            promotion_reasons     = reasons,
        )

    except Exception as exc:
        reg.finish(exp.experiment_id, failed=True, failure_reason=str(exc))
        log.error("Baseline pipeline failed for %s: %s", symbol, exc, exc_info=True)
        raise
