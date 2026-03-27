"""Tests for ml.comparison — runner, ranking, and promotion utilities."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.comparison.ranking import (
    PROMOTION_CRITERIA,
    composite_score,
    composite_score_from_result,
    compare_dataset_version,
    explain_promotion,
    rank_experiments,
    top_n,
)
from ml.comparison.runner import (
    ComparisonResult,
    ModelResult,
    run_comparison,
)
from ml.data.dataset_builder import assemble_dataset
from ml.registry.experiment_registry import (
    BacktestResult,
    ExperimentRegistry,
    ExperimentStatus,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    end   = datetime.now(tz=timezone.utc)
    idx   = pd.bdate_range(end=end, periods=n, tz="UTC")
    close = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high":   close * (1 + np.random.uniform(0.001, 0.015, n)),
            "low":    close * (1 - np.random.uniform(0.001, 0.015, n)),
            "close":  close,
            "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def _make_backtest(
    cumulative_return: float = 0.12,
    benchmark_return:  float = 0.08,
    hit_rate:          float = 0.55,
    sharpe:            float = 1.2,
) -> BacktestResult:
    return BacktestResult(
        cumulative_return  = cumulative_return,
        annualised_return  = cumulative_return * 0.5,
        hit_rate           = hit_rate,
        max_drawdown       = -0.10,
        sharpe             = sharpe,
        trade_count        = 30,
        benchmark_return   = benchmark_return,
        period_start       = "2023-01-01",
        period_end         = "2024-01-01",
    )


def _make_registry_with_records(
    tmp_path: Path,
    n_records: int = 3,
    dataset_version: str = "abc123def456",
) -> ExperimentRegistry:
    """Create a registry with a mix of completed experiments."""
    reg = ExperimentRegistry(path=tmp_path / "exp.json")
    model_types = ["logistic", "mlp", "lstm"]
    for i in range(n_records):
        exp = reg.create(
            name         = f"exp_{i}",
            model_type   = model_types[i % len(model_types)],
            hyperparams  = {},
            dataset_info = {"dataset_version": dataset_version, "symbol": "AAPL"},
        )
        reg.finish(
            exp.experiment_id,
            metrics = {
                "accuracy": 0.50 + i * 0.03,
                "auc":      0.51 + i * 0.02,
                "f1":       0.48 + i * 0.02,
                "precision": 0.50,
                "recall":    0.50,
            },
        )
        reg.attach_backtest(
            exp.experiment_id,
            _make_backtest(
                cumulative_return = 0.05 + i * 0.03,
                hit_rate          = 0.51 + i * 0.02,
                sharpe            = 0.5 + i * 0.5,
            ),
        )
    return reg


# ── composite_score ───────────────────────────────────────────────────────────

class TestCompositeScore:
    def _record_with(self, accuracy=0.58, auc=0.60, hit_rate=0.55, sharpe=1.0):
        from ml.registry.experiment_registry import ExperimentRecord
        import uuid
        rec = ExperimentRecord(
            experiment_id = str(uuid.uuid4()),
            name          = "test",
            model_type    = "mlp",
            status        = ExperimentStatus.COMPLETED,
            metrics       = {"accuracy": accuracy, "auc": auc},
        )
        rec.backtest = _make_backtest(hit_rate=hit_rate, sharpe=sharpe)
        return rec

    def test_returns_float(self):
        rec = self._record_with()
        assert isinstance(composite_score(rec), float)

    def test_no_backtest_returns_reduced_score(self):
        from ml.registry.experiment_registry import ExperimentRecord
        import uuid
        rec = ExperimentRecord(
            experiment_id = str(uuid.uuid4()),
            name          = "test",
            model_type    = "mlp",
            status        = ExperimentStatus.COMPLETED,
            metrics       = {"accuracy": 0.60, "auc": 0.62},
        )
        assert composite_score(rec) < composite_score(self._record_with(accuracy=0.60, auc=0.62))

    def test_higher_metrics_higher_score(self):
        low  = self._record_with(accuracy=0.50, auc=0.51, hit_rate=0.51, sharpe=0.0)
        high = self._record_with(accuracy=0.65, auc=0.70, hit_rate=0.60, sharpe=2.0)
        assert composite_score(high) > composite_score(low)

    def test_custom_weights(self):
        rec   = self._record_with(accuracy=1.0, auc=0.0, hit_rate=0.0, sharpe=-3.0)
        score = composite_score(rec, weights={"accuracy": 1.0, "auc": 0.0, "hit_rate": 0.0, "sharpe": 0.0})
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_is_deterministic(self):
        rec = self._record_with()
        assert composite_score(rec) == composite_score(rec)


class TestCompositeScoreFromResult:
    def _result(self, accuracy=0.58, auc=0.60, hit_rate=0.55, sharpe=1.0):
        from ml.patterns.train_mlp import PipelineResult
        return PipelineResult(
            experiment_id         = "test",
            symbol                = "AAPL",
            metrics               = {"accuracy": accuracy, "auc": auc},
            backtest_summary      = {"hit_rate": hit_rate, "sharpe": sharpe},
            checkpoint_path       = "",
            scaler_path           = "",
            promotion_recommended = False,
        )

    def test_returns_float(self):
        assert isinstance(composite_score_from_result(self._result()), float)

    def test_consistent_with_record_score(self):
        """Scores from PipelineResult and ExperimentRecord should be equal."""
        result = self._result(accuracy=0.58, auc=0.60, hit_rate=0.55, sharpe=1.2)
        from ml.registry.experiment_registry import ExperimentRecord
        import uuid
        rec = ExperimentRecord(
            experiment_id = str(uuid.uuid4()),
            name          = "t",
            model_type    = "mlp",
            status        = ExperimentStatus.COMPLETED,
            metrics       = result.metrics,
        )
        rec.backtest = _make_backtest(hit_rate=0.55, sharpe=1.2)
        assert composite_score_from_result(result) == pytest.approx(
            composite_score(rec), abs=1e-5
        )


# ── rank_experiments ──────────────────────────────────────────────────────────

class TestRankExperiments:
    def test_returns_list_of_tuples(self, tmp_path):
        reg = _make_registry_with_records(tmp_path)
        ranked = rank_experiments(reg)
        assert isinstance(ranked, list)
        for item in ranked:
            assert isinstance(item, tuple) and len(item) == 2

    def test_ranks_start_at_one(self, tmp_path):
        reg = _make_registry_with_records(tmp_path)
        ranked = rank_experiments(reg)
        if ranked:
            assert ranked[0][0] == 1

    def test_sorted_by_composite_score_descending(self, tmp_path):
        reg = _make_registry_with_records(tmp_path)
        ranked = rank_experiments(reg)
        scores = [composite_score(r) for _, r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_dataset_version_filter(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, dataset_version="version_A")
        # Add one with different version
        exp = reg.create(name="other", model_type="mlp", hyperparams={},
                         dataset_info={"dataset_version": "version_B"})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.60, "auc": 0.62})
        reg.attach_backtest(exp.experiment_id, _make_backtest())
        ranked = rank_experiments(reg, dataset_version="version_A")
        for _, r in ranked:
            assert r.dataset_info.get("dataset_version") == "version_A"

    def test_model_type_filter(self, tmp_path):
        reg = _make_registry_with_records(tmp_path)
        ranked = rank_experiments(reg, model_type="mlp")
        for _, r in ranked:
            assert r.model_type == "mlp"

    def test_n_limits_results(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        ranked = rank_experiments(reg, n=2)
        assert len(ranked) <= 2

    def test_require_backtest_true(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        exp = reg.create(name="no_bt", model_type="mlp", hyperparams={}, dataset_info={})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.60})
        # No backtest attached
        ranked = rank_experiments(reg, require_backtest=True)
        assert all(r.backtest is not None for _, r in ranked)

    def test_empty_registry_returns_empty(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        assert rank_experiments(reg) == []


# ── explain_promotion ─────────────────────────────────────────────────────────

class TestExplainPromotion:
    def _make_good_record(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        exp = reg.create(name="good", model_type="mlp", hyperparams={}, dataset_info={})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.60, "auc": 0.63,
                                               "f1": 0.58, "precision": 0.59, "recall": 0.57})
        reg.attach_backtest(
            exp.experiment_id,
            _make_backtest(cumulative_return=0.15, benchmark_return=0.08,
                           hit_rate=0.57, sharpe=1.5),
        )
        return reg.get(exp.experiment_id)

    def _make_poor_record(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp2.json")
        exp = reg.create(name="poor", model_type="logistic", hyperparams={}, dataset_info={})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.48, "auc": 0.50,
                                               "f1": 0.40, "precision": 0.45, "recall": 0.42})
        reg.attach_backtest(
            exp.experiment_id,
            _make_backtest(cumulative_return=0.02, benchmark_return=0.10,
                           hit_rate=0.48, sharpe=-0.5),
        )
        return reg.get(exp.experiment_id)

    def test_returns_dict_with_required_keys(self, tmp_path):
        rec = self._make_good_record(tmp_path)
        result = explain_promotion(rec)
        for key in ("experiment_id", "model_type", "overall_pass",
                    "criteria", "summary", "composite_score"):
            assert key in result

    def test_good_record_overall_pass(self, tmp_path):
        rec = self._make_good_record(tmp_path)
        assert explain_promotion(rec)["overall_pass"] is True

    def test_poor_record_not_overall_pass(self, tmp_path):
        rec = self._make_poor_record(tmp_path)
        assert explain_promotion(rec)["overall_pass"] is False

    def test_criteria_is_list(self, tmp_path):
        rec = self._make_good_record(tmp_path)
        result = explain_promotion(rec)
        assert isinstance(result["criteria"], list)
        assert len(result["criteria"]) > 0

    def test_each_criterion_has_required_keys(self, tmp_path):
        rec = self._make_good_record(tmp_path)
        for cr in explain_promotion(rec)["criteria"]:
            assert "criterion" in cr
            assert "value"     in cr
            assert "threshold" in cr
            assert "passed"    in cr
            assert "message"   in cr

    def test_summary_contains_recommended_or_not(self, tmp_path):
        good = self._make_good_record(tmp_path)
        poor = self._make_poor_record(tmp_path)
        assert "RECOMMENDED" in explain_promotion(good)["summary"]
        assert "NOT"         in explain_promotion(poor)["summary"]

    def test_no_backtest_all_backtest_criteria_fail(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp3.json")
        exp = reg.create(name="no_bt", model_type="mlp", hyperparams={}, dataset_info={})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.60, "auc": 0.65})
        rec = reg.get(exp.experiment_id)
        result = explain_promotion(rec)
        backtest_crit = [cr for cr in result["criteria"]
                         if cr["criterion"] in ("hit_rate", "beat_benchmark", "sharpe")]
        assert all(not cr["passed"] for cr in backtest_crit)

    def test_composite_score_is_float(self, tmp_path):
        rec = self._make_good_record(tmp_path)
        assert isinstance(explain_promotion(rec)["composite_score"], float)


# ── compare_dataset_version ───────────────────────────────────────────────────

class TestCompareDatasetVersion:
    def test_returns_matching_records(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, dataset_version="target_v")
        # Add one with a different version
        exp = reg.create(name="other", model_type="mlp", hyperparams={},
                         dataset_info={"dataset_version": "other_v"})
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.60})

        results = compare_dataset_version(reg, "target_v")
        assert len(results) == 3
        for r in results:
            assert r.dataset_info.get("dataset_version") == "target_v"

    def test_returns_empty_for_unknown_version(self, tmp_path):
        reg = _make_registry_with_records(tmp_path)
        assert compare_dataset_version(reg, "unknown_version") == []

    def test_returns_list_of_experiment_records(self, tmp_path):
        from ml.registry.experiment_registry import ExperimentRecord
        reg = _make_registry_with_records(tmp_path, dataset_version="v1")
        results = compare_dataset_version(reg, "v1")
        assert all(isinstance(r, ExperimentRecord) for r in results)

    def test_lstm_found_by_comparison_group_version(self, tmp_path):
        """LSTM experiments with comparison_group_version must be returned when
        searching by the flat dataset version."""
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        # Simulate LSTM registry entry: dataset_version = seq hash, but
        # comparison_group_version = flat hash
        exp = reg.create(
            name         = "lstm_run",
            model_type   = "lstm",
            hyperparams  = {},
            dataset_info = {
                "dataset_version":        "seq_hash_abcdef",
                "comparison_group_version": "flat_hash_123456",
                "symbol":                 "AAPL",
            },
        )
        reg.finish(exp.experiment_id, metrics={"accuracy": 0.57})

        # Searching by flat version should return the LSTM experiment
        results = compare_dataset_version(reg, "flat_hash_123456")
        assert any(r.experiment_id == exp.experiment_id for r in results)

        # Searching by seq version should also still work
        results_seq = compare_dataset_version(reg, "seq_hash_abcdef")
        assert any(r.experiment_id == exp.experiment_id for r in results_seq)


# ── top_n ─────────────────────────────────────────────────────────────────────

class TestTopN:
    def test_returns_at_most_n(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        assert len(top_n(reg, n=2)) <= 2

    def test_sorted_by_accuracy_descending(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        results = top_n(reg, n=3, metric="accuracy")
        accs = [r.metrics.get("accuracy", 0) for r in results]
        assert accs == sorted(accs, reverse=True)

    def test_sorted_by_hit_rate(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        results = top_n(reg, n=3, metric="hit_rate")
        hrs = [r.backtest.hit_rate for r in results if r.backtest]
        assert hrs == sorted(hrs, reverse=True)

    def test_sorted_by_sharpe(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        results = top_n(reg, n=3, metric="sharpe")
        sharpes = [r.backtest.sharpe for r in results if r.backtest]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_model_type_filter(self, tmp_path):
        reg = _make_registry_with_records(tmp_path, n_records=3)
        results = top_n(reg, metric="accuracy", model_type="logistic")
        for r in results:
            assert r.model_type == "logistic"

    def test_empty_registry_returns_empty(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        assert top_n(reg) == []


# ── ModelResult ───────────────────────────────────────────────────────────────

class TestModelResult:
    def _make(self) -> ModelResult:
        return ModelResult(
            model_type            = "mlp",
            experiment_id         = "test-id",
            metrics               = {"accuracy": 0.58, "auc": 0.60},
            backtest_summary      = {"hit_rate": 0.55, "sharpe": 1.2,
                                     "cumulative_return": 0.10},
            promotion_recommended = True,
            promotion_reasons     = ["accuracy 0.58 ≥ 0.55"],
            dataset_version       = "abc123def456",
            composite_score       = 0.42,
            rank                  = 1,
        )

    def test_to_dict_has_all_fields(self):
        d = self._make().to_dict()
        for key in ("model_type", "experiment_id", "metrics", "backtest_summary",
                    "promotion_recommended", "promotion_reasons", "dataset_version",
                    "composite_score", "rank"):
            assert key in d


# ── ComparisonResult ──────────────────────────────────────────────────────────

class TestComparisonResult:
    def _make(self, tmp_path) -> ComparisonResult:
        ohlcv = _make_ohlcv()
        _, meta = assemble_dataset(ohlcv, symbol="AAPL")
        results = [
            ModelResult(
                model_type            = t,
                experiment_id         = f"id_{t}",
                metrics               = {"accuracy": 0.55 + i * 0.02, "auc": 0.57},
                backtest_summary      = {"hit_rate": 0.53, "sharpe": 0.8,
                                         "cumulative_return": 0.08},
                promotion_recommended = i > 0,
                promotion_reasons     = [],
                dataset_version       = meta.dataset_version,
                composite_score       = 0.30 + i * 0.05,
                rank                  = i + 1,
            )
            for i, t in enumerate(["logistic", "mlp"])
        ]
        return ComparisonResult(
            symbol          = "AAPL",
            dataset_version = meta.dataset_version,
            dataset_meta    = meta,
            results         = results,
            winner          = "mlp",
        )

    def test_ranked_returns_sorted_list(self, tmp_path):
        cr = self._make(tmp_path)
        ranked = cr.ranked()
        scores = [r.composite_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict_has_required_keys(self, tmp_path):
        d = self._make(tmp_path).to_dict()
        for key in ("symbol", "dataset_version", "dataset_meta", "winner",
                    "generated_at", "results"):
            assert key in d

    def test_print_summary_runs(self, tmp_path, capsys):
        self._make(tmp_path).print_summary()
        out = capsys.readouterr().out
        assert "AAPL" in out
        assert "mlp" in out

    def test_winner_matches_top_score(self, tmp_path):
        cr = self._make(tmp_path)
        best = cr.ranked()[0]
        assert cr.winner == best.model_type


# ── run_comparison (integration) ──────────────────────────────────────────────

class TestRunComparison:
    """Integration tests for run_comparison with baseline + MLP."""

    def test_baseline_only_returns_result(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
        )
        assert isinstance(result, ComparisonResult)
        assert len(result.results) == 1
        assert result.results[0].model_type == "baseline"

    def test_baseline_and_mlp_returns_two_results(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        assert len(result.results) == 2
        model_types = {r.model_type for r in result.results}
        assert "baseline" in model_types
        assert "mlp" in model_types

    def test_all_results_share_dataset_version(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        versions = {r.dataset_version for r in result.results}
        assert len(versions) == 1
        assert result.dataset_version in versions

    def test_winner_is_set(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        assert result.winner in ("baseline", "mlp")

    def test_ranks_are_assigned(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        ranks = [r.rank for r in result.results]
        assert sorted(ranks) == list(range(1, len(result.results) + 1))

    def test_experiments_recorded_in_registry(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        for mr in result.results:
            exp = reg.get(mr.experiment_id)
            assert exp.status == ExperimentStatus.COMPLETED
            assert exp.backtest is not None

    def test_unknown_model_skipped(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "unknown_model"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
        )
        # Should complete with just baseline
        assert len(result.results) >= 1

    def test_all_models_fail_raises(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        with patch("ml.comparison.runner._run_model",
                   side_effect=RuntimeError("always fail")):
            with pytest.raises(RuntimeError, match="All models failed"):
                run_comparison(
                    symbol   = "AAPL",
                    df       = _make_ohlcv(),
                    models   = ["baseline"],
                    registry = reg,
                )

    def test_comparison_tags_added_to_experiments(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            tags           = ["my_experiment"],
        )
        exp = reg.get(result.results[0].experiment_id)
        assert "my_experiment" in exp.tags

    def test_dataset_version_in_experiment_registry(self, tmp_path):
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
        )
        exp = reg.get(result.results[0].experiment_id)
        assert exp.dataset_info.get("dataset_version") == result.dataset_version

    def test_mlp_and_baseline_share_registry_dataset_version(self, tmp_path):
        """Both models must store the same dataset_version in the registry record."""
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "AAPL",
            df             = _make_ohlcv(),
            models         = ["baseline", "mlp"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            epochs         = 3,
            patience       = 3,
        )
        registry_versions = set()
        for mr in result.results:
            exp = reg.get(mr.experiment_id)
            registry_versions.add(exp.dataset_info.get("dataset_version"))
        assert len(registry_versions) == 1, (
            f"Expected one shared dataset_version, got: {registry_versions}"
        )
        assert result.dataset_version in registry_versions
