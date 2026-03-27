"""Tests for ml.validation — walk-forward split, evaluation, and promotion."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.validation.walk_forward import (
    FoldSpec,
    WalkForwardConfig,
    assemble_walk_forward_dataset,
    make_folds,
    slice_fold,
)
from ml.validation.wf_aggregation import (
    FoldAggregate,
    WalkForwardPromotion,
    WalkForwardPromotionConfig,
    aggregate_folds,
    wf_promotion_recommend,
)
from ml.validation.wf_runner import (
    FoldResult,
    WalkForwardResult,
    run_walk_forward,
    run_walk_forward_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 350) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def ohlcv():
    return _make_ohlcv(n=350)


@pytest.fixture(scope="module")
def full_data(ohlcv):
    return assemble_walk_forward_dataset(ohlcv, symbol="TEST")


@pytest.fixture(scope="module")
def small_cfg():
    """Fast config for integration tests: 3 folds, small train window."""
    return WalkForwardConfig(n_folds=3, min_train_size=80, val_frac=0.15)


# ── WalkForwardConfig validation ──────────────────────────────────────────────

class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.n_folds == 5
        assert cfg.min_train_size == 120
        assert cfg.val_frac == 0.15
        assert cfg.window == "expanding"
        assert cfg.gap == 0

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            WalkForwardConfig(window="random")

    def test_invalid_val_frac_raises(self):
        with pytest.raises(ValueError, match="val_frac"):
            WalkForwardConfig(val_frac=1.0)
        with pytest.raises(ValueError, match="val_frac"):
            WalkForwardConfig(val_frac=-0.1)

    def test_invalid_n_folds_raises(self):
        with pytest.raises(ValueError, match="n_folds"):
            WalkForwardConfig(n_folds=0)

    def test_invalid_min_train_size_raises(self):
        with pytest.raises(ValueError, match="min_train_size"):
            WalkForwardConfig(min_train_size=1)

    def test_invalid_gap_raises(self):
        with pytest.raises(ValueError, match="gap"):
            WalkForwardConfig(gap=-1)

    def test_to_dict(self):
        cfg = WalkForwardConfig(n_folds=3, window="rolling")
        d = cfg.to_dict()
        assert d["n_folds"] == 3
        assert d["window"] == "rolling"


# ── FoldSpec ──────────────────────────────────────────────────────────────────

class TestFoldSpec:
    def _make(self) -> FoldSpec:
        return FoldSpec(
            fold_idx=0, train_start=0, train_end=100,
            val_start=85, val_end=100,
            test_start=100, test_end=126,
            n_train=85, n_val=15, n_test=26,
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("fold_idx", "train_start", "train_end", "val_start", "val_end",
                    "test_start", "test_end", "n_train", "n_val", "n_test",
                    "train_date_start", "test_date_start"):
            assert key in d

    def test_n_train_plus_n_val_equals_raw_window(self):
        fold = self._make()
        raw_n = fold.train_end - fold.train_start
        assert fold.n_train + fold.n_val == raw_n


# ── make_folds ────────────────────────────────────────────────────────────────

class TestMakeFolds:
    def _cfg(self, **kw) -> WalkForwardConfig:
        return WalkForwardConfig(n_folds=5, min_train_size=100, **kw)

    def test_returns_list_of_fold_specs(self):
        folds = make_folds(300, self._cfg())
        assert isinstance(folds, list)
        assert all(isinstance(f, FoldSpec) for f in folds)

    def test_correct_fold_count(self):
        folds = make_folds(300, self._cfg())
        assert len(folds) == 5

    def test_fold_indices_start_at_zero(self):
        folds = make_folds(300, self._cfg())
        assert folds[0].fold_idx == 0
        assert folds[-1].fold_idx == len(folds) - 1

    def test_train_and_test_do_not_overlap(self):
        folds = make_folds(300, self._cfg())
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_test_windows_do_not_overlap(self):
        folds = make_folds(300, self._cfg())
        for i in range(len(folds) - 1):
            assert folds[i].test_end <= folds[i + 1].test_start

    def test_test_end_within_bounds(self):
        n = 300
        folds = make_folds(n, self._cfg())
        for fold in folds:
            assert fold.test_end <= n

    def test_n_train_positive(self):
        folds = make_folds(300, self._cfg())
        for fold in folds:
            assert fold.n_train > 0

    def test_n_test_positive(self):
        folds = make_folds(300, self._cfg())
        for fold in folds:
            assert fold.n_test > 0

    def test_val_frac_zero_gives_no_val(self):
        folds = make_folds(300, WalkForwardConfig(n_folds=5, min_train_size=100, val_frac=0.0))
        for fold in folds:
            assert fold.n_val == 0
            assert fold.val_start is None
            assert fold.val_end is None

    def test_val_frac_nonzero_gives_val(self):
        folds = make_folds(300, self._cfg(val_frac=0.15))
        for fold in folds:
            assert fold.n_val >= 1
            assert fold.val_start is not None

    def test_val_window_inside_train_window(self):
        folds = make_folds(300, self._cfg(val_frac=0.15))
        for fold in folds:
            assert fold.val_start >= fold.train_start
            assert fold.val_end == fold.train_end

    def test_expanding_window_grows(self):
        folds = make_folds(300, self._cfg(window="expanding"))
        train_sizes = [f.train_end - f.train_start for f in folds]
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i + 1] > train_sizes[i]

    def test_rolling_window_fixed_size(self):
        folds = make_folds(300, self._cfg(window="rolling"))
        train_sizes = [f.train_end - f.train_start for f in folds]
        assert len(set(train_sizes)) == 1

    def test_gap_applied(self):
        folds_no_gap   = make_folds(300, self._cfg(gap=0))
        folds_with_gap = make_folds(300, self._cfg(gap=3))
        for f0, fg in zip(folds_no_gap, folds_with_gap):
            assert fg.test_start - fg.train_end == 3

    def test_date_index_populates_date_strings(self):
        n   = 300
        idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        folds = make_folds(n, self._cfg(), date_index=idx)
        for fold in folds:
            assert fold.train_date_start != ""
            assert fold.test_date_start  != ""

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            make_folds(50, WalkForwardConfig(n_folds=5, min_train_size=100))

    def test_custom_test_size(self):
        folds = make_folds(300, WalkForwardConfig(n_folds=5, min_train_size=100, test_size=20))
        for fold in folds:
            assert fold.n_test == 20

    def test_fewer_folds_returned_when_data_runs_out(self):
        # With min_train_size=200, test_size=30, n_samples=250:
        # fold 0 test_end = 200+30 = 230 (OK), fold 1 test_end = 230+30 = 260 > 250 (stop)
        # So only 1 fold can be formed despite requesting 5.
        folds = make_folds(
            250, WalkForwardConfig(n_folds=5, min_train_size=200, test_size=30)
        )
        assert len(folds) < 5
        assert len(folds) >= 1


# ── assemble_walk_forward_dataset ─────────────────────────────────────────────

class TestAssembleWalkForwardDataset:
    def test_returns_dict_with_required_keys(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        for key in ("X_full", "y_full", "close_full", "feature_df",
                    "feature_cols", "date_index", "n_samples", "symbol"):
            assert key in full

    def test_x_y_same_length(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert len(full["X_full"]) == len(full["y_full"])

    def test_close_full_same_length(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert len(full["close_full"]) == full["n_samples"]

    def test_date_index_same_length(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert len(full["date_index"]) == full["n_samples"]

    def test_x_is_float32(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert full["X_full"].dtype == np.float32

    def test_y_is_binary(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert set(full["y_full"]).issubset({0.0, 1.0})

    def test_no_inf_in_x(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert np.all(np.isfinite(full["X_full"]))

    def test_symbol_uppercased(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="aapl")
        assert full["symbol"] == "AAPL"

    def test_feature_df_columns_match_feature_cols(self, ohlcv):
        full = assemble_walk_forward_dataset(ohlcv, symbol="AAPL")
        assert list(full["feature_df"].columns) == full["feature_cols"]


# ── slice_fold ────────────────────────────────────────────────────────────────

class TestSliceFold:
    def test_train_test_correct_sizes(self, full_data):
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80, val_frac=0.15)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        fd   = slice_fold(full_data, fold)
        assert len(fd["X_train"]) == fold.n_train
        assert len(fd["X_test"])  == fold.n_test

    def test_val_rows_match(self, full_data):
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80, val_frac=0.15)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        fd   = slice_fold(full_data, fold)
        assert len(fd["X_val"]) == fold.n_val

    def test_no_val_gives_empty_array(self, full_data):
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80, val_frac=0.0)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        fd   = slice_fold(full_data, fold)
        assert len(fd["X_val"]) == 0

    def test_close_test_aligned_to_x_test(self, full_data):
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        fd   = slice_fold(full_data, fold)
        assert len(fd["close_test"]) == len(fd["X_test"])

    def test_train_not_contaminated_by_test(self, full_data):
        """Train rows must all be before test rows (chronological integrity)."""
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        # Use date_index to verify no overlap
        train_end_date = full_data["date_index"][fold.train_end - 1]
        test_start_date = full_data["date_index"][fold.test_start]
        assert train_end_date <= test_start_date

    def test_feature_df_fold_covers_full_window(self, full_data):
        cfg  = WalkForwardConfig(n_folds=3, min_train_size=80)
        fold = make_folds(full_data["n_samples"], cfg)[0]
        fd   = slice_fold(full_data, fold)
        expected_len = fold.test_end - fold.train_start
        assert len(fd["feature_df_fold"]) == expected_len


# ── FoldResult ────────────────────────────────────────────────────────────────

class TestFoldResult:
    def _make(self) -> FoldResult:
        fold = FoldSpec(
            fold_idx=0, train_start=0, train_end=100,
            val_start=85, val_end=100, test_start=100, test_end=126,
            n_train=85, n_val=15, n_test=26,
        )
        return FoldResult(
            fold_idx              = 0,
            fold_spec             = fold,
            model_type            = "baseline",
            metrics               = {"accuracy": 0.56, "auc": 0.59, "f1": 0.55},
            backtest_summary      = {"hit_rate": 0.54, "sharpe": 0.8,
                                     "cumulative_return": 0.07,
                                     "benchmark_return": 0.05},
            promotion_recommended = True,
            promotion_reasons     = ["accuracy 0.56 ≥ 0.55"],
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("fold_idx", "fold_spec", "model_type", "metrics",
                    "backtest_summary", "promotion_recommended", "promotion_reasons"):
            assert key in d

    def test_fold_spec_serialised(self):
        d = self._make().to_dict()
        assert "n_train" in d["fold_spec"]


# ── WalkForwardResult ─────────────────────────────────────────────────────────

class TestWalkForwardResult:
    def _make(self) -> WalkForwardResult:
        fold = FoldSpec(
            fold_idx=0, train_start=0, train_end=100,
            val_start=85, val_end=100, test_start=100, test_end=126,
            n_train=85, n_val=15, n_test=26,
        )
        fr = FoldResult(
            fold_idx=0, fold_spec=fold, model_type="baseline",
            metrics={"accuracy": 0.56, "auc": 0.59},
            backtest_summary={"hit_rate": 0.54, "sharpe": 0.8,
                              "cumulative_return": 0.07, "benchmark_return": 0.05},
            promotion_recommended=True, promotion_reasons=[],
        )
        cfg = WalkForwardConfig(n_folds=3, min_train_size=80)
        return WalkForwardResult(
            symbol="AAPL", model_type="baseline",
            n_folds_total=3, n_folds_run=1,
            fold_results=[fr], config=cfg,
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("symbol", "model_type", "n_folds_total", "n_folds_run",
                    "fold_results", "config", "generated_at"):
            assert key in d

    def test_config_serialised(self):
        d = self._make().to_dict()
        assert "n_folds" in d["config"]


# ── aggregate_folds ───────────────────────────────────────────────────────────

class TestAggregateFolds:
    def _make_fold_results(self, n=3) -> list[FoldResult]:
        results = []
        for i in range(n):
            fold = FoldSpec(
                fold_idx=i, train_start=0, train_end=100,
                val_start=85, val_end=100, test_start=100, test_end=126,
                n_train=85, n_val=15, n_test=26,
            )
            acc = 0.54 + i * 0.02
            hr  = 0.51 + i * 0.02
            cr  = 0.05 + i * 0.03
            results.append(FoldResult(
                fold_idx=i, fold_spec=fold, model_type="baseline",
                metrics={"accuracy": acc, "auc": 0.55, "f1": 0.50},
                backtest_summary={
                    "hit_rate": hr, "sharpe": 0.5 + i * 0.3,
                    "cumulative_return": cr,
                    "benchmark_return": 0.04,
                },
                promotion_recommended=(acc >= 0.55 and hr >= 0.52),
                promotion_reasons=[],
            ))
        return results

    def test_returns_fold_aggregate(self):
        frs = self._make_fold_results()
        agg = aggregate_folds(frs, "baseline")
        assert isinstance(agg, FoldAggregate)

    def test_n_folds_correct(self):
        frs = self._make_fold_results(3)
        agg = aggregate_folds(frs, "baseline")
        assert agg.n_folds == 3

    def test_mean_accuracy_in_range(self):
        frs = self._make_fold_results(3)
        agg = aggregate_folds(frs, "baseline")
        accs = [fr.metrics["accuracy"] for fr in frs]
        assert agg.mean_accuracy == pytest.approx(sum(accs) / len(accs), abs=1e-3)

    def test_std_accuracy_positive(self):
        frs = self._make_fold_results(3)
        agg = aggregate_folds(frs, "baseline")
        assert agg.std_accuracy >= 0.0

    def test_min_max_accuracy(self):
        frs = self._make_fold_results(3)
        agg = aggregate_folds(frs, "baseline")
        accs = [fr.metrics["accuracy"] for fr in frs]
        assert agg.min_accuracy == pytest.approx(min(accs), abs=1e-3)
        assert agg.max_accuracy == pytest.approx(max(accs), abs=1e-3)

    def test_n_folds_beat_benchmark(self):
        frs = self._make_fold_results(3)
        agg = aggregate_folds(frs, "baseline")
        # cr = [0.05, 0.08, 0.11], bm = 0.04 → all three beat benchmark
        assert agg.n_folds_beat_benchmark == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_folds([], "baseline")

    def test_to_dict_without_fold_results(self):
        frs = self._make_fold_results(2)
        agg = aggregate_folds(frs, "baseline")
        d = agg.to_dict(include_fold_results=False)
        assert "fold_results" not in d
        assert "mean_accuracy" in d

    def test_to_dict_with_fold_results(self):
        frs = self._make_fold_results(2)
        agg = aggregate_folds(frs, "baseline")
        d = agg.to_dict(include_fold_results=True)
        assert "fold_results" in d
        assert len(d["fold_results"]) == 2


# ── wf_promotion_recommend ────────────────────────────────────────────────────

class TestWfPromotionRecommend:
    def _good_agg(self) -> FoldAggregate:
        fold = FoldSpec(0, 0, 100, 85, 100, 100, 126, 85, 15, 26)
        frs  = []
        for i in range(5):
            frs.append(FoldResult(
                fold_idx=i, fold_spec=fold, model_type="mlp",
                metrics={"accuracy": 0.58, "auc": 0.61},
                backtest_summary={
                    "hit_rate": 0.55, "sharpe": 1.2,
                    "cumulative_return": 0.12,
                    "benchmark_return": 0.08,
                },
                promotion_recommended=True, promotion_reasons=[],
            ))
        return aggregate_folds(frs, "mlp")

    def _poor_agg(self) -> FoldAggregate:
        fold = FoldSpec(0, 0, 100, 85, 100, 100, 126, 85, 15, 26)
        frs  = []
        for i in range(5):
            frs.append(FoldResult(
                fold_idx=i, fold_spec=fold, model_type="logistic",
                metrics={"accuracy": 0.48, "auc": 0.50},
                backtest_summary={
                    "hit_rate": 0.48, "sharpe": -0.5,
                    "cumulative_return": 0.02,
                    "benchmark_return": 0.08,
                },
                promotion_recommended=False, promotion_reasons=[],
            ))
        return aggregate_folds(frs, "logistic")

    def test_returns_walk_forward_promotion(self):
        agg = self._good_agg()
        promo = wf_promotion_recommend(agg)
        assert isinstance(promo, WalkForwardPromotion)

    def test_good_model_recommended(self):
        agg = self._good_agg()
        promo = wf_promotion_recommend(agg)
        assert promo.overall_recommended is True

    def test_poor_model_not_recommended(self):
        agg = self._poor_agg()
        promo = wf_promotion_recommend(agg)
        assert promo.overall_recommended is False

    def test_criteria_is_list_with_required_keys(self):
        promo = wf_promotion_recommend(self._good_agg())
        for cr in promo.criteria:
            for key in ("criterion", "value", "threshold", "passed", "gate", "message"):
                assert key in cr

    def test_summary_contains_recommended(self):
        assert "RECOMMENDED" in wf_promotion_recommend(self._good_agg()).summary

    def test_summary_contains_not_recommended(self):
        assert "NOT" in wf_promotion_recommend(self._poor_agg()).summary

    def test_custom_config_min_folds_beat_bm(self):
        """If min_folds_beat_bm is set high, a mediocre model fails."""
        agg = self._poor_agg()  # n_folds_beat_benchmark = 0
        cfg = WalkForwardPromotionConfig(min_folds_beat_bm=3)
        promo = wf_promotion_recommend(agg, config=cfg)
        assert promo.overall_recommended is False

    def test_std_gate(self):
        """When std_is_gate=True and std_accuracy > threshold, gate fails."""
        # Create folds with very different accuracies
        fold = FoldSpec(0, 0, 100, 85, 100, 100, 126, 85, 15, 26)
        frs  = [
            FoldResult(
                fold_idx=i, fold_spec=fold, model_type="mlp",
                metrics={"accuracy": 0.55 + i * 0.1, "auc": 0.60},  # high variance
                backtest_summary={
                    "hit_rate": 0.55, "sharpe": 1.0,
                    "cumulative_return": 0.10, "benchmark_return": 0.05,
                },
                promotion_recommended=True, promotion_reasons=[],
            )
            for i in range(5)
        ]
        agg = aggregate_folds(frs, "mlp")
        cfg = WalkForwardPromotionConfig(std_is_gate=True, max_std_accuracy=0.05)
        promo = wf_promotion_recommend(agg, config=cfg)
        # std_accuracy will be high due to spread
        std_cr = next(c for c in promo.criteria if c["criterion"] == "std_accuracy")
        assert std_cr["gate"] is True

    def test_n_folds_in_result(self):
        agg = self._good_agg()
        promo = wf_promotion_recommend(agg)
        assert promo.n_folds == 5

    def test_default_min_folds_beat_bm_is_ceiling_half(self):
        """Without explicit min_folds_beat_bm, default is ceil(n/2)."""
        agg = self._good_agg()  # 5 folds, all beat benchmark
        promo = wf_promotion_recommend(agg)
        beat_cr = next(c for c in promo.criteria if c["criterion"] == "folds_beat_benchmark")
        assert beat_cr["threshold"] == math.ceil(5 / 2)

    def test_to_dict_has_required_keys(self):
        promo = wf_promotion_recommend(self._good_agg())
        d = promo.to_dict()
        for key in ("model_type", "overall_recommended", "criteria", "summary", "n_folds"):
            assert key in d


# ── run_walk_forward_model integration ────────────────────────────────────────

class TestRunWalkForwardModelBaseline:
    """Integration tests for the baseline model only (fast, no GPU needed)."""

    def test_returns_walk_forward_result(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        assert isinstance(wf, WalkForwardResult)

    def test_model_type_is_baseline(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        assert wf.model_type == "baseline"

    def test_correct_fold_count(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        assert wf.n_folds_run == wf.n_folds_total

    def test_fold_results_non_empty(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        assert len(wf.fold_results) > 0

    def test_metrics_present_in_each_fold(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        for fr in wf.fold_results:
            assert "accuracy" in fr.metrics
            assert "auc" in fr.metrics

    def test_backtest_present_in_each_fold(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        for fr in wf.fold_results:
            assert "hit_rate" in fr.backtest_summary
            assert "cumulative_return" in fr.backtest_summary

    def test_promotion_is_bool(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        for fr in wf.fold_results:
            assert isinstance(fr.promotion_recommended, bool)

    def test_promotion_reasons_length(self, ohlcv, small_cfg):
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        for fr in wf.fold_results:
            assert len(fr.promotion_reasons) == 3

    def test_full_data_shared(self, full_data, small_cfg):
        """Passing full_data avoids re-assembling the dataset."""
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    full_data=full_data, config=small_cfg)
        assert wf.n_folds_run > 0

    def test_register_folds_creates_registry_entries(self, ohlcv, small_cfg, tmp_path):
        from ml.registry.experiment_registry import ExperimentRegistry, ExperimentStatus
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        wf  = run_walk_forward_model(
            "baseline", symbol="TEST", df=ohlcv, config=small_cfg,
            registry=reg, register_folds=True,
        )
        entries = reg.filter(status=ExperimentStatus.COMPLETED)
        assert len(entries) == wf.n_folds_run

    def test_invalid_model_key_raises(self, ohlcv):
        with pytest.raises(ValueError, match="model_key"):
            run_walk_forward_model("unknown", symbol="TEST", df=ohlcv)

    def test_to_dict_serialisable(self, ohlcv, small_cfg):
        import json
        wf = run_walk_forward_model("baseline", symbol="TEST",
                                    df=ohlcv, config=small_cfg)
        d  = wf.to_dict()
        # Should not raise
        json.dumps(d)


class TestRunWalkForwardModelMLP:
    """Integration test for MLP walk-forward (slightly slower than baseline)."""

    def test_mlp_runs_and_returns_result(self, ohlcv):
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100, val_frac=0.15)
        wf  = run_walk_forward_model(
            "mlp", symbol="TEST", df=ohlcv, config=cfg, epochs=3, patience=3
        )
        assert isinstance(wf, WalkForwardResult)
        assert wf.n_folds_run >= 1

    def test_mlp_fold_metrics_in_range(self, ohlcv):
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        wf  = run_walk_forward_model(
            "mlp", symbol="TEST", df=ohlcv, config=cfg, epochs=3, patience=3
        )
        for fr in wf.fold_results:
            assert 0.0 <= fr.metrics.get("accuracy", 0) <= 1.0


# ── run_walk_forward (multi-model) ────────────────────────────────────────────

class TestRunWalkForward:
    def test_returns_dict_of_results(self, ohlcv, small_cfg):
        results = run_walk_forward(
            symbol="TEST", df=ohlcv, models=("baseline",), config=small_cfg
        )
        assert isinstance(results, dict)
        assert "baseline" in results

    def test_both_models_returned(self, ohlcv):
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        results = run_walk_forward(
            symbol="TEST", df=ohlcv,
            models=("baseline", "mlp"),
            config=cfg, epochs=3, patience=3,
        )
        assert "baseline" in results
        assert "mlp" in results

    def test_unknown_model_excluded(self, ohlcv, small_cfg):
        results = run_walk_forward(
            symbol="TEST", df=ohlcv,
            models=("baseline", "unknown_model"),
            config=small_cfg,
        )
        assert "unknown_model" not in results
        assert "baseline" in results


# ── WalkForwardComparisonResult (Phase 5) ────────────────────────────────────

class TestWalkForwardComparisonResult:
    def test_run_comparison_walk_forward_returns_wf_result(self, ohlcv):
        from ml.comparison.runner import run_comparison, WalkForwardComparisonResult
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol       = "TEST",
            df           = ohlcv,
            models       = ["baseline"],
            walk_forward = True,
            wf_config    = cfg,
        )
        assert isinstance(result, WalkForwardComparisonResult)

    def test_winner_is_set(self, ohlcv):
        from ml.comparison.runner import run_comparison
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol       = "TEST",
            df           = ohlcv,
            models       = ["baseline"],
            walk_forward = True,
            wf_config    = cfg,
        )
        assert result.winner == "baseline"

    def test_aggregates_and_promotions_populated(self, ohlcv):
        from ml.comparison.runner import run_comparison
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol       = "TEST",
            df           = ohlcv,
            models       = ["baseline"],
            walk_forward = True,
            wf_config    = cfg,
        )
        assert "baseline" in result.aggregates
        assert "baseline" in result.promotions

    def test_ranked_returns_sorted_list(self, ohlcv):
        from ml.comparison.runner import run_comparison
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol       = "TEST",
            df           = ohlcv,
            models       = ["baseline"],
            walk_forward = True,
            wf_config    = cfg,
        )
        ranked = result.ranked()
        scores = [agg.mean_composite_score for _, agg in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict_serialisable(self, ohlcv):
        import json
        from ml.comparison.runner import run_comparison
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol="TEST", df=ohlcv, models=["baseline"],
            walk_forward=True, wf_config=cfg,
        )
        json.dumps(result.to_dict())

    def test_print_summary_runs(self, ohlcv, capsys):
        from ml.comparison.runner import run_comparison
        cfg = WalkForwardConfig(n_folds=2, min_train_size=100)
        result = run_comparison(
            symbol="TEST", df=ohlcv, models=["baseline"],
            walk_forward=True, wf_config=cfg,
        )
        result.print_summary()
        out = capsys.readouterr().out
        assert "Walk-Forward" in out

    def test_single_split_path_unchanged(self, ohlcv, tmp_path):
        """Ensure walk_forward=False still returns ComparisonResult."""
        from ml.comparison.runner import ComparisonResult, run_comparison
        from ml.registry.experiment_registry import ExperimentRegistry
        reg = ExperimentRegistry(path=tmp_path / "exp.json")
        result = run_comparison(
            symbol         = "TEST",
            df             = ohlcv,
            models         = ["baseline"],
            registry       = reg,
            checkpoint_dir = tmp_path / "ckpts",
            walk_forward   = False,
        )
        assert isinstance(result, ComparisonResult)
