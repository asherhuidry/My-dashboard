"""Tests for ml/evidence/bridge.py.

All tests use synthetic in-memory data — no network calls, no disk writes
except to tmp_path.  Real WalkForwardResult / ComparisonResult objects are
constructed with minimal fixture data so tests remain fast and deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ml.evidence.bridge import (
    ClaimBundle,
    SaveResult,
    bridge_comparison,
    bridge_wf_comparison,
    claims_from_comparison,
    claims_from_wf_comparison,
    evidence_from_fold_result,
    evidence_from_pipeline_result,
    evidence_from_wf_result,
    save_evidence_bundle,
)
from ml.evidence.schema import (
    Claim,
    ClaimStatus,
    ClaimType,
    EvidenceItem,
    EvidenceSourceType,
)
from ml.evidence.store import ClaimStore


# ── Synthetic fixture helpers ─────────────────────────────────────────────────

def _fold_spec(fold_idx: int = 0) -> Any:
    """Minimal FoldSpec-like object."""
    from ml.validation.walk_forward import FoldSpec
    return FoldSpec(
        fold_idx=fold_idx,
        train_start=0, train_end=100,
        val_start=85, val_end=100,
        test_start=100, test_end=126,
        n_train=85, n_val=15, n_test=26,
        train_date_start="2024-01-01",
        train_date_end="2024-05-01",
        val_date_start="2024-04-01",
        val_date_end="2024-05-01",
        test_date_start="2024-05-01",
        test_date_end="2024-11-01",
    )


def _fold_result(
    fold_idx:   int   = 0,
    model_type: str   = "baseline",
    acc:        float = 0.58,
    hit:        float = 0.55,
    cum_ret:    float = 0.10,
    bm_ret:     float = 0.07,
    sharpe:     float = 1.2,
    promo:      bool  = True,
    exp_id:     str   = "exp_abc123",
) -> Any:
    from ml.validation.wf_runner import FoldResult
    return FoldResult(
        fold_idx              = fold_idx,
        fold_spec             = _fold_spec(fold_idx),
        model_type            = model_type,
        metrics               = {"accuracy": acc, "auc": 0.61, "f1": 0.57},
        backtest_summary      = {
            "hit_rate":           hit,
            "sharpe":             sharpe,
            "cumulative_return":  cum_ret,
            "benchmark_return":   bm_ret,
        },
        promotion_recommended = promo,
        promotion_reasons     = ["accuracy ok", "hit_rate ok"],
        experiment_id         = exp_id,
    )


def _wf_result(
    symbol:     str         = "AAPL",
    model_type: str         = "baseline",
    n_folds:    int         = 3,
    accs:       list[float] | None = None,
) -> Any:
    from ml.validation.walk_forward import WalkForwardConfig
    from ml.validation.wf_runner import WalkForwardResult
    cfg = WalkForwardConfig(n_folds=n_folds, min_train_size=100)
    if accs is None:
        accs = [0.56, 0.58, 0.60]
    frs = [
        _fold_result(fold_idx=i, model_type=model_type, acc=accs[i % len(accs)])
        for i in range(n_folds)
    ]
    return WalkForwardResult(
        symbol        = symbol,
        model_type    = model_type,
        n_folds_total = n_folds,
        n_folds_run   = n_folds,
        fold_results  = frs,
        config        = cfg,
        generated_at  = "2026-01-01T00:00:00+00:00",
    )


def _fold_aggregate(
    model_type:  str   = "baseline",
    n_folds:     int   = 3,
    mean_acc:    float = 0.57,
    std_acc:     float = 0.02,
    mean_hit:    float = 0.54,
    beat_bm:     int   = 2,
) -> Any:
    from ml.validation.wf_aggregation import FoldAggregate
    frs = [_fold_result(i, model_type) for i in range(n_folds)]
    return FoldAggregate(
        model_type               = model_type,
        n_folds                  = n_folds,
        mean_accuracy            = mean_acc,
        std_accuracy             = std_acc,
        min_accuracy             = mean_acc - std_acc,
        max_accuracy             = mean_acc + std_acc,
        mean_auc                 = 0.60,
        std_auc                  = 0.02,
        mean_hit_rate            = mean_hit,
        std_hit_rate             = 0.05,
        mean_sharpe              = 1.1,
        std_sharpe               = 0.3,
        mean_cumulative_return   = 0.09,
        std_cumulative_return    = 0.03,
        mean_composite_score     = 0.55,
        std_composite_score      = 0.08,
        n_folds_beat_benchmark   = beat_bm,
        n_folds_promo_recommended = 2,
        fold_results             = frs,
    )


def _wf_promotion(model_type: str, overall: bool) -> Any:
    from ml.validation.wf_aggregation import WalkForwardPromotion
    return WalkForwardPromotion(
        model_type          = model_type,
        overall_recommended = overall,
        criteria            = [
            {"criterion": "mean_accuracy", "value": 0.57, "threshold": 0.55,
             "passed": overall, "gate": True, "description": "", "message": ""},
        ],
        summary = "OK" if overall else "NOT RECOMMENDED",
        n_folds = 3,
    )


def _wf_comparison_result(
    symbol:  str   = "AAPL",
    models:  tuple = ("baseline", "mlp"),
    winner:  str   = "baseline",
) -> Any:
    from ml.comparison.runner import WalkForwardComparisonResult
    from ml.validation.walk_forward import WalkForwardConfig
    cfg = WalkForwardConfig(n_folds=3, min_train_size=100)
    aggs  = {m: _fold_aggregate(m) for m in models}
    promos = {m: _wf_promotion(m, m == winner) for m in models}
    return WalkForwardComparisonResult(
        symbol       = symbol,
        wf_config    = cfg,
        aggregates   = aggs,
        promotions   = promos,
        winner       = winner,
        generated_at = "2026-01-01T00:00:00+00:00",
    )


def _dataset_meta():
    from ml.data.dataset_builder import DatasetMeta
    return DatasetMeta(
        dataset_version="abc123456789",
        symbol="AAPL",
        feature_cols=["rsi_14", "macd"],
        target_definition="1-bar direction",
        n_rows=500, n_train=350, n_val=75, n_test=75,
        time_range_start="2022-01-01", time_range_end="2024-01-01",
        train_frac=0.70, val_frac=0.15,
        target_horizon=1, seq_len=None,
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _model_result(
    model_type: str   = "baseline",
    acc:        float = 0.57,
    promo:      bool  = False,
    rank:       int   = 1,
) -> Any:
    from ml.comparison.runner import ModelResult
    return ModelResult(
        model_type            = model_type,
        experiment_id         = f"exp_{model_type}_001",
        metrics               = {"accuracy": acc, "auc": 0.60, "f1": 0.56},
        backtest_summary      = {
            "hit_rate": 0.54, "sharpe": 1.1,
            "cumulative_return": 0.09, "benchmark_return": 0.07,
        },
        promotion_recommended = promo,
        promotion_reasons     = ["accuracy ok"],
        dataset_version       = "abc123456789",
        composite_score       = 0.54,
        rank                  = rank,
    )


def _comparison_result(
    symbol:  str   = "AAPL",
    models:  tuple = ("baseline", "mlp"),
    winner:  str   = "baseline",
) -> Any:
    from ml.comparison.runner import ComparisonResult
    results = [
        _model_result(m, rank=i + 1)
        for i, m in enumerate(models)
    ]
    return ComparisonResult(
        symbol          = symbol,
        dataset_version = "abc123456789",
        dataset_meta    = _dataset_meta(),
        results         = results,
        winner          = winner,
        generated_at    = "2026-01-01T00:00:00+00:00",
    )


def _pipeline_result(
    symbol:     str   = "AAPL",
    model_type: str   = "baseline",
    promo:      bool  = False,
    acc:        float = 0.57,
) -> Any:
    """Minimal object that looks like a PipelineResult."""
    @dataclass
    class FakePipelineResult:
        experiment_id:        str
        symbol:               str
        model_type:           str
        metrics:              dict
        backtest_summary:     dict
        promotion_recommended: bool
        promotion_reasons:    list

    return FakePipelineResult(
        experiment_id         = f"exp_{model_type}_001",
        symbol                = symbol,
        model_type            = model_type,
        metrics               = {"accuracy": acc, "auc": 0.60, "f1": 0.56},
        backtest_summary      = {
            "hit_rate": 0.54, "sharpe": 1.1,
            "cumulative_return": 0.09, "benchmark_return": 0.07,
        },
        promotion_recommended = promo,
        promotion_reasons     = [],
    )


@pytest.fixture
def store(tmp_path: Path) -> ClaimStore:
    return ClaimStore(path=tmp_path / "claims.json")


# ── Phase 1: evidence_from_fold_result ────────────────────────────────────────

class TestEvidenceFromFoldResult:
    def test_returns_evidence_item(self):
        fr = _fold_result()
        ev = evidence_from_fold_result(fr, "AAPL")
        assert isinstance(ev, EvidenceItem)

    def test_source_type_experiment_when_exp_id_present(self):
        fr = _fold_result(exp_id="exp_abc123")
        ev = evidence_from_fold_result(fr, "AAPL")
        assert ev.source_type == EvidenceSourceType.EXPERIMENT
        assert ev.source_ref  == "exp_abc123"

    def test_source_type_backtest_when_no_exp_id(self):
        fr = _fold_result(exp_id="")
        ev = evidence_from_fold_result(fr, "AAPL")
        assert ev.source_type == EvidenceSourceType.BACKTEST

    def test_structured_data_has_key_metrics(self):
        fr = _fold_result(acc=0.61, hit=0.55, cum_ret=0.12, bm_ret=0.08)
        ev = evidence_from_fold_result(fr, "AAPL")
        sd = ev.structured_data
        assert sd["accuracy"]  == pytest.approx(0.61)
        assert sd["hit_rate"]  == pytest.approx(0.55)
        assert sd["beat_benchmark"] is True
        assert sd["cumulative_return"] == pytest.approx(0.12)
        assert sd["symbol"]    == "AAPL"
        assert sd["fold_idx"]  == 0

    def test_beat_benchmark_false_when_under(self):
        fr = _fold_result(cum_ret=0.03, bm_ret=0.08)
        ev = evidence_from_fold_result(fr, "AAPL")
        assert ev.structured_data["beat_benchmark"] is False

    def test_symbol_uppercased(self):
        fr = _fold_result()
        ev = evidence_from_fold_result(fr, "aapl")
        assert ev.structured_data["symbol"] == "AAPL"

    def test_summary_contains_key_info(self):
        fr = _fold_result(fold_idx=2, model_type="mlp", acc=0.58)
        ev = evidence_from_fold_result(fr, "AAPL")
        assert "Fold 2" in ev.summary
        assert "mlp"    in ev.summary
        assert "AAPL"   in ev.summary

    def test_round_trips_to_json(self):
        fr = _fold_result()
        ev = evidence_from_fold_result(fr, "AAPL")
        d  = ev.to_dict()
        ev2 = EvidenceItem.from_dict(d)
        assert ev2.evidence_id == ev.evidence_id
        assert ev2.source_type == ev.source_type


# ── Phase 1: evidence_from_wf_result ─────────────────────────────────────────

class TestEvidenceFromWfResult:
    def test_returns_one_per_fold_plus_aggregate(self):
        wf = _wf_result(n_folds=3)
        items = evidence_from_wf_result(wf)
        # 3 fold items + 1 aggregate
        assert len(items) == 4

    def test_empty_fold_results_returns_empty(self):
        from ml.validation.wf_runner import WalkForwardResult
        from ml.validation.walk_forward import WalkForwardConfig
        wf = WalkForwardResult(
            symbol="AAPL", model_type="baseline",
            n_folds_total=3, n_folds_run=0, fold_results=[],
            config=WalkForwardConfig(), generated_at="2026-01-01T00:00:00+00:00",
        )
        assert evidence_from_wf_result(wf) == []

    def test_aggregate_item_is_last(self):
        wf = _wf_result(n_folds=2)
        items = evidence_from_wf_result(wf)
        agg   = items[-1]
        assert "aggregate" in agg.summary.lower()
        assert agg.source_type == EvidenceSourceType.BACKTEST

    def test_aggregate_structured_data_has_mean_accuracy(self):
        wf = _wf_result(n_folds=3, accs=[0.56, 0.58, 0.60])
        items = evidence_from_wf_result(wf)
        agg   = items[-1].structured_data
        assert "mean_accuracy" in agg
        assert agg["n_folds_run"] == 3
        assert "wf_config" in agg

    def test_fold_items_have_fold_idx(self):
        wf    = _wf_result(n_folds=3)
        items = evidence_from_wf_result(wf)
        fold_idxs = [it.structured_data["fold_idx"] for it in items[:-1]]
        assert fold_idxs == [0, 1, 2]

    def test_all_items_are_evidence_items(self):
        wf    = _wf_result(n_folds=2)
        items = evidence_from_wf_result(wf)
        for it in items:
            assert isinstance(it, EvidenceItem)

    def test_n_folds_beat_benchmark_counted(self):
        from ml.validation.wf_runner import FoldResult, WalkForwardResult
        from ml.validation.walk_forward import WalkForwardConfig
        frs = [
            _fold_result(i, cum_ret=0.10, bm_ret=0.07)  # beats bm
            if i < 2 else
            _fold_result(i, cum_ret=0.03, bm_ret=0.08)  # misses
            for i in range(3)
        ]
        wf = WalkForwardResult(
            symbol="AAPL", model_type="baseline",
            n_folds_total=3, n_folds_run=3, fold_results=frs,
            config=WalkForwardConfig(), generated_at="2026-01-01T00:00:00+00:00",
        )
        items = evidence_from_wf_result(wf)
        agg   = items[-1].structured_data
        assert agg["n_folds_beat_benchmark"] == 2


# ── Phase 1: evidence_from_pipeline_result ────────────────────────────────────

class TestEvidenceFromPipelineResult:
    def test_returns_evidence_item(self):
        pr = _pipeline_result()
        ev = evidence_from_pipeline_result(pr)
        assert isinstance(ev, EvidenceItem)

    def test_source_type_is_experiment(self):
        pr = _pipeline_result()
        ev = evidence_from_pipeline_result(pr)
        assert ev.source_type == EvidenceSourceType.EXPERIMENT

    def test_source_ref_is_experiment_id(self):
        pr = _pipeline_result()
        ev = evidence_from_pipeline_result(pr)
        assert ev.source_ref == pr.experiment_id

    def test_structured_data_has_metrics(self):
        pr = _pipeline_result(acc=0.59)
        ev = evidence_from_pipeline_result(pr)
        assert ev.structured_data["accuracy"] == pytest.approx(0.59)
        assert ev.structured_data["symbol"] == "AAPL"

    def test_beat_benchmark_flag(self):
        pr = _pipeline_result()
        ev = evidence_from_pipeline_result(pr)
        # cum_ret=0.09 > bm_ret=0.07 → True
        assert ev.structured_data["beat_benchmark"] is True

    def test_symbol_uppercased(self):
        pr = _pipeline_result(symbol="spy")
        ev = evidence_from_pipeline_result(pr)
        assert ev.structured_data["symbol"] == "SPY"


# ── Phase 2: claims_from_comparison ──────────────────────────────────────────

class TestClaimsFromComparison:
    def test_returns_claim_bundles(self):
        result  = _comparison_result()
        bundles = claims_from_comparison(result)
        assert all(isinstance(b, ClaimBundle) for b in bundles)

    def test_one_claim_per_model_plus_winner(self):
        result  = _comparison_result(models=("baseline", "mlp"), winner="baseline")
        bundles = claims_from_comparison(result)
        # 2 model claims + 1 winner claim = 3
        assert len(bundles) == 3

    def test_single_model_no_winner_claim(self):
        result  = _comparison_result(models=("baseline",), winner="baseline")
        bundles = claims_from_comparison(result)
        # Only 1 model → no winner claim (need len > 1)
        assert len(bundles) == 1

    def test_model_claim_type_is_performance(self):
        bundles = claims_from_comparison(_comparison_result(models=("baseline",)))
        assert bundles[0].claim.claim_type == ClaimType.PERFORMANCE

    def test_model_claim_subject_format(self):
        result  = _comparison_result(models=("baseline",), winner="baseline")
        bundles = claims_from_comparison(result)
        subj    = bundles[0].claim.subject
        assert "model:baseline_on_AAPL" == subj

    def test_winner_claim_predicate(self):
        bundles = claims_from_comparison(
            _comparison_result(models=("baseline", "mlp"), winner="baseline")
        )
        winner_bundle = [b for b in bundles
                         if b.claim.predicate == "leads_single_split_comparison"]
        assert len(winner_bundle) == 1

    def test_promoted_model_higher_confidence(self):
        from ml.comparison.runner import ModelResult
        # Build one promoted, one not
        res_promo = _model_result("baseline", promo=True,  rank=1)
        res_none  = _model_result("mlp",      promo=False, rank=2)
        from ml.comparison.runner import ComparisonResult
        comp = ComparisonResult(
            symbol="AAPL", dataset_version="abc123456789",
            dataset_meta=_dataset_meta(),
            results=[res_promo, res_none], winner="baseline",
            generated_at="2026-01-01T00:00:00+00:00",
        )
        bundles = claims_from_comparison(comp)
        conf_map = {b.claim.subject.split("_on_")[0].replace("model:", ""): b.claim.confidence
                    for b in bundles if b.claim.predicate != "leads_single_split_comparison"}
        assert conf_map["baseline"] > conf_map["mlp"]

    def test_status_is_proposed(self):
        bundles = claims_from_comparison(_comparison_result())
        for b in bundles:
            assert b.claim.status == ClaimStatus.PROPOSED

    def test_uncertainty_notes_mention_single_split(self):
        bundles = claims_from_comparison(_comparison_result(models=("baseline",)))
        assert "single" in bundles[0].claim.uncertainty_notes.lower()

    def test_evidence_ids_initially_empty(self):
        bundles = claims_from_comparison(_comparison_result())
        for b in bundles:
            assert b.evidence_ids == []

    def test_to_dict_serializable(self):
        bundles = claims_from_comparison(_comparison_result())
        for b in bundles:
            json.dumps(b.claim.to_dict())


# ── Phase 2: claims_from_wf_comparison ───────────────────────────────────────

class TestClaimsFromWfComparison:
    def test_returns_claim_bundles(self):
        result  = _wf_comparison_result()
        bundles = claims_from_wf_comparison(result)
        assert all(isinstance(b, ClaimBundle) for b in bundles)

    def test_one_claim_per_model_plus_winner(self):
        result  = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        bundles = claims_from_wf_comparison(result)
        # 2 model claims + 1 winner claim = 3
        assert len(bundles) == 3

    def test_model_claim_subject_format(self):
        result  = _wf_comparison_result(models=("baseline",), winner="baseline")
        bundles = claims_from_wf_comparison(result)
        model_bundles = [b for b in bundles if b.claim.predicate == "achieves_wf_mean_accuracy"]
        assert model_bundles[0].claim.subject == "model:baseline_on_AAPL"

    def test_winner_claim_predicate(self):
        result  = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        bundles = claims_from_wf_comparison(result)
        winner_bundles = [b for b in bundles
                          if b.claim.predicate == "leads_walk_forward_comparison"]
        assert len(winner_bundles) == 1
        assert "baseline" in winner_bundles[0].claim.subject

    def test_high_variance_adds_counterpoint(self):
        from ml.comparison.runner import WalkForwardComparisonResult
        from ml.validation.walk_forward import WalkForwardConfig
        agg_high_var = _fold_aggregate(std_acc=0.09)  # > 0.06 threshold
        from ml.validation.wf_aggregation import WalkForwardPromotion
        promo = WalkForwardPromotion("baseline", False, [], "NOT OK", 3)
        result = WalkForwardComparisonResult(
            symbol="AAPL",
            wf_config=WalkForwardConfig(n_folds=3),
            aggregates={"baseline": agg_high_var},
            promotions={"baseline": promo},
            winner="baseline",
            generated_at="2026-01-01T00:00:00+00:00",
        )
        bundles = claims_from_wf_comparison(result)
        model_claim = [b for b in bundles
                       if b.claim.predicate == "achieves_wf_mean_accuracy"][0].claim
        assert any("variance" in cp.lower() for cp in model_claim.counterpoints)

    def test_promoted_model_higher_confidence(self):
        from ml.comparison.runner import WalkForwardComparisonResult
        from ml.validation.walk_forward import WalkForwardConfig
        agg_good = _fold_aggregate("baseline", mean_acc=0.57, beat_bm=2)
        agg_poor = _fold_aggregate("mlp",      mean_acc=0.50, beat_bm=1)
        from ml.validation.wf_aggregation import WalkForwardPromotion
        result = WalkForwardComparisonResult(
            symbol="AAPL",
            wf_config=WalkForwardConfig(n_folds=3),
            aggregates={"baseline": agg_good, "mlp": agg_poor},
            promotions={
                "baseline": WalkForwardPromotion("baseline", True,  [], "OK", 3),
                "mlp":      WalkForwardPromotion("mlp",      False, [], "NO", 3),
            },
            winner="baseline",
            generated_at="2026-01-01T00:00:00+00:00",
        )
        bundles = claims_from_wf_comparison(result)
        conf_map = {
            b.claim.subject.split("_on_")[0].replace("model:", ""): b.claim.confidence
            for b in bundles if b.claim.predicate == "achieves_wf_mean_accuracy"
        }
        assert conf_map["baseline"] > conf_map["mlp"]

    def test_wf_tags_present(self):
        result  = _wf_comparison_result(models=("baseline",), winner="baseline")
        bundles = claims_from_wf_comparison(result)
        model_claim = [b for b in bundles
                       if b.claim.predicate == "achieves_wf_mean_accuracy"][0].claim
        assert "walk_forward" in model_claim.tags
        assert "AAPL" in model_claim.tags

    def test_status_proposed(self):
        bundles = claims_from_wf_comparison(_wf_comparison_result())
        for b in bundles:
            assert b.claim.status == ClaimStatus.PROPOSED


# ── Phase 3: save_evidence_bundle ────────────────────────────────────────────

class TestSaveEvidenceBundle:
    def test_saves_items_and_claims(self, store):
        ev1 = EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "summary 1")
        c1  = Claim.new(ClaimType.PERFORMANCE, "model:x_on_A", "p", "o", 0.5)
        res = save_evidence_bundle(store, [ev1], [ClaimBundle(c1, [])])
        assert ev1.evidence_id in res.evidence_ids
        assert c1.claim_id     in res.claim_ids

    def test_links_evidence_to_claim(self, store):
        ev1 = EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s")
        c1  = Claim.new(ClaimType.PERFORMANCE, "s", "p", "o", 0.5)
        save_evidence_bundle(
            store, [ev1], [ClaimBundle(c1, [ev1.evidence_id])]
        )
        linked = store.get_evidence_for_claim(c1.claim_id)
        assert any(e.evidence_id == ev1.evidence_id for e in linked)

    def test_skips_duplicate_evidence(self, store):
        ev1 = EvidenceItem.new(EvidenceSourceType.MANUAL, "r", "s")
        store.add_evidence(ev1)
        res = save_evidence_bundle(store, [ev1], [])
        assert ev1.evidence_id in res.skipped_evidence
        assert ev1.evidence_id not in res.evidence_ids

    def test_skips_duplicate_claim(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "s", "p", "o", 0.5)
        store.add_claim(c1)
        res = save_evidence_bundle(store, [], [ClaimBundle(c1, [])])
        assert c1.claim_id in res.skipped_claims

    def test_returns_save_result(self, store):
        res = save_evidence_bundle(store, [], [])
        assert isinstance(res, SaveResult)
        assert res.evidence_ids     == []
        assert res.claim_ids        == []
        assert res.skipped_evidence == []
        assert res.skipped_claims   == []

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "c.json"
        s1   = ClaimStore(path=path)
        ev1  = EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s")
        c1   = Claim.new(ClaimType.PERFORMANCE, "s", "p", "o", 0.5)
        save_evidence_bundle(s1, [ev1], [ClaimBundle(c1, [ev1.evidence_id])])
        s2 = ClaimStore(path=path)
        assert len(s2.list_evidence()) == 1
        assert len(s2.list_claims())   == 1
        linked = s2.get_evidence_for_claim(c1.claim_id)
        assert linked[0].evidence_id == ev1.evidence_id


# ── Phase 3: bridge_wf_comparison ────────────────────────────────────────────

class TestBridgeWfComparison:
    def test_saves_claims_and_evidence(self, store):
        result = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        wf_results = {
            "baseline": _wf_result("AAPL", "baseline", n_folds=3),
            "mlp":      _wf_result("AAPL", "mlp",      n_folds=3),
        }
        saved = bridge_wf_comparison(result, store, wf_results=wf_results)
        assert len(saved.claim_ids)    > 0
        assert len(saved.evidence_ids) > 0

    def test_claim_count_correct(self, store):
        result = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        saved  = bridge_wf_comparison(result, store)
        # 2 model claims + 1 winner claim
        assert len(saved.claim_ids) == 3

    def test_evidence_linked_when_wf_results_provided(self, store):
        result = _wf_comparison_result(models=("baseline",), winner="baseline")
        wf_results = {"baseline": _wf_result("AAPL", "baseline", n_folds=2)}
        saved  = bridge_wf_comparison(result, store, wf_results=wf_results)
        # Should have 2 fold items + 1 agg = 3 evidence items for 1 model
        assert len(saved.evidence_ids) == 3
        # The model claim should have linked evidence
        claim_id = saved.claim_ids[0]
        linked = store.get_evidence_for_claim(claim_id)
        assert len(linked) > 0

    def test_second_call_does_not_raise(self, store):
        """Calling bridge_wf_comparison twice on the same store does not raise.

        Evidence IDs are UUIDs so each call produces new items (not skipped).
        Claims are also UUID-keyed so a second call adds new claim records.
        This test simply verifies that no exception is raised.
        """
        result = _wf_comparison_result(models=("baseline",), winner="baseline")
        wf_results = {"baseline": _wf_result("AAPL", "baseline", n_folds=2)}
        s1 = bridge_wf_comparison(result, store, wf_results=wf_results)
        s2 = bridge_wf_comparison(result, store, wf_results=wf_results)
        # Both calls succeed and return SaveResults with the same shape
        assert len(s2.evidence_ids) == len(s1.evidence_ids)
        assert len(s2.claim_ids)    == len(s1.claim_ids)

    def test_without_wf_results_still_saves_claims(self, store):
        result = _wf_comparison_result(models=("baseline",), winner="baseline")
        saved  = bridge_wf_comparison(result, store, wf_results=None)
        assert len(saved.claim_ids)    > 0
        assert len(saved.evidence_ids) == 0  # no evidence without wf_results


# ── Phase 3: bridge_comparison ────────────────────────────────────────────────

class TestBridgeComparison:
    def test_saves_claims(self, store):
        result = _comparison_result(models=("baseline", "mlp"), winner="baseline")
        saved  = bridge_comparison(result, store)
        assert len(saved.claim_ids) == 3  # 2 model + 1 winner

    def test_with_pipeline_results_saves_evidence(self, store):
        result = _comparison_result(models=("baseline",), winner="baseline")
        prs    = [_pipeline_result("AAPL", "baseline")]
        saved  = bridge_comparison(result, store, pipeline_results=prs)
        assert len(saved.evidence_ids) == 1

    def test_evidence_linked_to_model_claim(self, store):
        result = _comparison_result(models=("baseline",), winner="baseline")
        prs    = [_pipeline_result("AAPL", "baseline")]
        saved  = bridge_comparison(result, store, pipeline_results=prs)
        model_claims = store.list_claims(claim_type=ClaimType.PERFORMANCE)
        model_claim  = [c for c in model_claims
                        if c.predicate == "achieves_composite_score"][0]
        linked = store.get_evidence_for_claim(model_claim.claim_id)
        assert len(linked) == 1


# ── Integration: full round-trip ──────────────────────────────────────────────

class TestIntegration:
    def test_wf_comparison_full_round_trip(self, tmp_path):
        """End-to-end: build results → bridge → store → evaluate → reload."""
        from ml.evidence.evaluator import batch_evaluate

        result = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        wf_results = {
            "baseline": _wf_result("AAPL", "baseline", n_folds=3),
            "mlp":      _wf_result("AAPL", "mlp",      n_folds=3),
        }
        path  = tmp_path / "claims.json"
        store = ClaimStore(path=path)
        saved = bridge_wf_comparison(result, store, wf_results=wf_results)

        # Evaluate all claims
        evals = batch_evaluate(store)
        assert len(evals) > 0
        for ev in evals:
            assert ev.support_level in ("none", "weak", "moderate", "strong")
            assert ev.recommendation in ("propose", "support", "flag_weak", "reject")

        # Reload from disk — check everything persisted
        store2 = ClaimStore(path=path)
        assert len(store2) == len(store)
        assert len(store2.list_evidence()) == len(store.list_evidence())

    def test_tags_allow_filtering(self, store):
        result = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        bridge_wf_comparison(result, store)
        wf_claims = store.claims_by_tag("walk_forward")
        assert len(wf_claims) > 0
        aapl_claims = store.claims_by_tag("AAPL")
        assert len(aapl_claims) > 0

    def test_summary_stats_after_bridge(self, store):
        result = _wf_comparison_result(models=("baseline", "mlp"), winner="baseline")
        bridge_wf_comparison(result, store)
        stats = store.summary_stats()
        assert stats["n_claims"] == 3
        assert "proposed" in stats["by_status"]
        assert stats["by_type"].get("performance", 0) == 3

    def test_claim_subjects_query_by_symbol(self, store):
        result = _wf_comparison_result(symbol="SPY", models=("baseline",), winner="baseline")
        bridge_wf_comparison(result, store)
        spy_claims = store.claims_by_subject("model:baseline_on_SPY")
        assert len(spy_claims) >= 1
