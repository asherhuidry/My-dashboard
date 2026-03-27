"""Tests for the ml/evidence/ module.

Covers schema, store, evaluator, and templates.  All tests are local-first
(no network, no DB) and use temporary files for persistence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.evidence.schema import (
    Claim,
    ClaimStatus,
    ClaimType,
    EvidenceItem,
    EvidenceSourceType,
)
from ml.evidence.store import ClaimStore
from ml.evidence.evaluator import ClaimEvaluation, evaluate_claim, batch_evaluate
from ml.evidence.templates import (
    feature_usefulness_claim,
    performance_claim,
    relationship_claim,
    source_usefulness_claim,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> ClaimStore:
    """A fresh ClaimStore backed by a temp file."""
    return ClaimStore(path=tmp_path / "claims.json")


@pytest.fixture
def ev_experiment() -> EvidenceItem:
    return EvidenceItem.new(
        source_type     = EvidenceSourceType.EXPERIMENT,
        source_ref      = "exp_abc123",
        summary         = "MLP accuracy 0.58 on AAPL fold 2",
        structured_data = {"accuracy": 0.58},
    )


@pytest.fixture
def ev_backtest() -> EvidenceItem:
    return EvidenceItem.new(
        source_type     = EvidenceSourceType.BACKTEST,
        source_ref      = "wf_AAPL_2026",
        summary         = "Baseline beat buy-and-hold on 2/3 folds",
        structured_data = {"beat_bm_folds": 2, "total_folds": 3},
    )


@pytest.fixture
def simple_claim() -> Claim:
    return Claim.new(
        claim_type = ClaimType.FEATURE_USEFULNESS,
        subject    = "feature:rsi_14",
        predicate  = "improves_accuracy_on",
        obj        = "model:mlp_dataset:abc123",
        confidence = 0.6,
    )


# ── Phase 1: EvidenceItem schema ──────────────────────────────────────────────

class TestEvidenceItem:
    def test_new_assigns_id_and_timestamp(self):
        ev = EvidenceItem.new(
            source_type = EvidenceSourceType.MANUAL,
            source_ref  = "note-001",
            summary     = "Observed positive autocorrelation in AAPL returns",
        )
        assert ev.evidence_id
        assert ev.created_at
        assert ev.source_type == EvidenceSourceType.MANUAL

    def test_to_dict_round_trips(self):
        ev = EvidenceItem.new(
            source_type     = EvidenceSourceType.BACKTEST,
            source_ref      = "wf_SPY_2026",
            summary         = "SPY baseline beat benchmark 2/3",
            structured_data = {"n": 3},
            citation        = "https://example.com",
        )
        d  = ev.to_dict()
        ev2 = EvidenceItem.from_dict(d)
        assert ev2.evidence_id    == ev.evidence_id
        assert ev2.source_type    == ev.source_type
        assert ev2.source_ref     == ev.source_ref
        assert ev2.structured_data == ev.structured_data
        assert ev2.citation        == ev.citation

    def test_source_type_is_enum(self):
        ev = EvidenceItem.new(
            source_type = "experiment",
            source_ref  = "x",
            summary     = "y",
        )
        assert ev.source_type == EvidenceSourceType.EXPERIMENT

    def test_all_source_types_round_trip(self):
        for st in EvidenceSourceType:
            ev = EvidenceItem.new(source_type=st, source_ref="r", summary="s")
            d  = ev.to_dict()
            ev2 = EvidenceItem.from_dict(d)
            assert ev2.source_type == st


# ── Phase 1: Claim schema ─────────────────────────────────────────────────────

class TestClaim:
    def test_new_creates_proposed_claim(self, simple_claim: Claim):
        assert simple_claim.status     == ClaimStatus.PROPOSED
        assert simple_claim.claim_id
        assert simple_claim.evidence_ids == []
        assert simple_claim.created_at
        assert simple_claim.updated_at

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Claim.new(
                claim_type = ClaimType.GENERAL,
                subject    = "x", predicate = "y", obj = "z",
                confidence = 1.5,
            )

    def test_to_dict_round_trips(self, simple_claim: Claim):
        d  = simple_claim.to_dict()
        c2 = Claim.from_dict(d)
        assert c2.claim_id    == simple_claim.claim_id
        assert c2.claim_type  == simple_claim.claim_type
        assert c2.subject     == simple_claim.subject
        assert c2.predicate   == simple_claim.predicate
        assert c2.object      == simple_claim.object
        assert c2.confidence  == simple_claim.confidence
        assert c2.status      == simple_claim.status

    def test_triple_property(self, simple_claim: Claim):
        assert simple_claim.triple == (
            "feature:rsi_14",
            "improves_accuracy_on",
            "model:mlp_dataset:abc123",
        )

    def test_is_active(self, simple_claim: Claim):
        assert simple_claim.is_active
        simple_claim.status = ClaimStatus.REJECTED
        assert not simple_claim.is_active
        simple_claim.status = ClaimStatus.ARCHIVED
        assert not simple_claim.is_active
        simple_claim.status = ClaimStatus.SUPPORTED
        assert simple_claim.is_active

    def test_all_claim_types_round_trip(self):
        for ct in ClaimType:
            c = Claim.new(claim_type=ct, subject="s", predicate="p", obj="o",
                          confidence=0.5)
            d = c.to_dict()
            c2 = Claim.from_dict(d)
            assert c2.claim_type == ct

    def test_all_statuses_round_trip(self):
        for cs in ClaimStatus:
            c = Claim.new(claim_type=ClaimType.GENERAL, subject="s",
                          predicate="p", obj="o", confidence=0.5)
            c.status = cs
            d  = c.to_dict()
            c2 = Claim.from_dict(d)
            assert c2.status == cs


# ── Phase 2: ClaimStore ───────────────────────────────────────────────────────

class TestClaimStore:
    def test_empty_store_length(self, store: ClaimStore):
        assert len(store) == 0

    def test_add_and_get_evidence(self, store: ClaimStore, ev_experiment: EvidenceItem):
        store.add_evidence(ev_experiment)
        retrieved = store.get_evidence(ev_experiment.evidence_id)
        assert retrieved.evidence_id == ev_experiment.evidence_id

    def test_add_duplicate_evidence_raises(self, store: ClaimStore, ev_experiment: EvidenceItem):
        store.add_evidence(ev_experiment)
        with pytest.raises(ValueError, match="already exists"):
            store.add_evidence(ev_experiment)

    def test_get_missing_evidence_raises(self, store: ClaimStore):
        with pytest.raises(KeyError):
            store.get_evidence("nonexistent")

    def test_add_and_get_claim(self, store: ClaimStore, simple_claim: Claim):
        store.add_claim(simple_claim)
        retrieved = store.get_claim(simple_claim.claim_id)
        assert retrieved.claim_id == simple_claim.claim_id

    def test_add_duplicate_claim_raises(self, store: ClaimStore, simple_claim: Claim):
        store.add_claim(simple_claim)
        with pytest.raises(ValueError, match="already exists"):
            store.add_claim(simple_claim)

    def test_get_missing_claim_raises(self, store: ClaimStore):
        with pytest.raises(KeyError):
            store.get_claim("nonexistent")

    def test_link_evidence(self, store, simple_claim, ev_experiment):
        store.add_evidence(ev_experiment)
        store.add_claim(simple_claim)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        claim = store.get_claim(simple_claim.claim_id)
        assert ev_experiment.evidence_id in claim.evidence_ids

    def test_link_missing_evidence_raises(self, store, simple_claim):
        store.add_claim(simple_claim)
        with pytest.raises(KeyError):
            store.link_evidence(simple_claim.claim_id, "bad_id")

    def test_link_idempotent(self, store, simple_claim, ev_experiment):
        store.add_evidence(ev_experiment)
        store.add_claim(simple_claim)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        claim = store.get_claim(simple_claim.claim_id)
        assert claim.evidence_ids.count(ev_experiment.evidence_id) == 1

    def test_unlink_evidence(self, store, simple_claim, ev_experiment):
        store.add_evidence(ev_experiment)
        store.add_claim(simple_claim)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        store.unlink_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        claim = store.get_claim(simple_claim.claim_id)
        assert ev_experiment.evidence_id not in claim.evidence_ids

    def test_get_evidence_for_claim(self, store, simple_claim, ev_experiment, ev_backtest):
        store.add_evidence(ev_experiment)
        store.add_evidence(ev_backtest)
        store.add_claim(simple_claim)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        store.link_evidence(simple_claim.claim_id, ev_backtest.evidence_id)
        items = store.get_evidence_for_claim(simple_claim.claim_id)
        assert len(items) == 2
        ids = {e.evidence_id for e in items}
        assert ev_experiment.evidence_id in ids
        assert ev_backtest.evidence_id in ids

    def test_update_status(self, store, simple_claim):
        store.add_claim(simple_claim)
        updated = store.update_status(
            simple_claim.claim_id,
            ClaimStatus.SUPPORTED,
            notes="Evidence sufficient",
        )
        assert updated.status == ClaimStatus.SUPPORTED
        assert "Evidence sufficient" in updated.notes

    def test_update_confidence(self, store, simple_claim):
        store.add_claim(simple_claim)
        updated = store.update_confidence(simple_claim.claim_id, 0.8)
        assert updated.confidence == pytest.approx(0.8)

    def test_update_confidence_out_of_range_raises(self, store, simple_claim):
        store.add_claim(simple_claim)
        with pytest.raises(ValueError, match="confidence"):
            store.update_confidence(simple_claim.claim_id, 1.5)

    def test_add_counterpoint(self, store, simple_claim):
        store.add_claim(simple_claim)
        store.add_counterpoint(simple_claim.claim_id, "Noisy in low-vol regimes")
        claim = store.get_claim(simple_claim.claim_id)
        assert "Noisy in low-vol regimes" in claim.counterpoints

    def test_list_claims_no_filter(self, store, simple_claim):
        store.add_claim(simple_claim)
        assert len(store.list_claims()) == 1

    def test_list_claims_by_status(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "a", "p", "b", 0.5)
        c2 = Claim.new(ClaimType.GENERAL, "c", "p", "d", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        store.update_status(c1.claim_id, ClaimStatus.SUPPORTED)
        proposed = store.list_claims(status=ClaimStatus.PROPOSED)
        supported = store.list_claims(status=ClaimStatus.SUPPORTED)
        assert len(proposed)  == 1
        assert len(supported) == 1

    def test_list_claims_by_type(self, store):
        c1 = Claim.new(ClaimType.SOURCE_USEFULNESS, "a", "p", "b", 0.5)
        c2 = Claim.new(ClaimType.FEATURE_USEFULNESS, "c", "p", "d", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        src_claims = store.list_claims(claim_type=ClaimType.SOURCE_USEFULNESS)
        assert len(src_claims) == 1
        assert src_claims[0].claim_id == c1.claim_id

    def test_list_claims_active_only(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "a", "p", "b", 0.5)
        c2 = Claim.new(ClaimType.GENERAL, "c", "p", "d", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        store.update_status(c2.claim_id, ClaimStatus.REJECTED)
        active = store.list_claims(active_only=True)
        assert len(active) == 1
        assert active[0].claim_id == c1.claim_id

    def test_claims_by_subject(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "feature:rsi_14", "p", "b", 0.5)
        c2 = Claim.new(ClaimType.GENERAL, "feature:macd", "p", "d", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        results = store.claims_by_subject("feature:rsi_14")
        assert len(results) == 1
        assert results[0].claim_id == c1.claim_id

    def test_claims_by_object(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "a", "p", "model:mlp_on_AAPL", 0.5)
        c2 = Claim.new(ClaimType.GENERAL, "b", "p", "model:lstm_on_SPY", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        results = store.claims_by_object("model:mlp_on_AAPL")
        assert len(results) == 1

    def test_claims_by_tag(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "a", "p", "b", 0.5, tags=["macro"])
        c2 = Claim.new(ClaimType.GENERAL, "c", "p", "d", 0.5, tags=["equity"])
        store.add_claim(c1)
        store.add_claim(c2)
        assert len(store.claims_by_tag("macro"))  == 1
        assert len(store.claims_by_tag("equity")) == 1
        assert len(store.claims_by_tag("other"))  == 0

    def test_summary_stats(self, store, simple_claim, ev_experiment):
        store.add_evidence(ev_experiment)
        store.add_claim(simple_claim)
        stats = store.summary_stats()
        assert stats["n_claims"]   == 1
        assert stats["n_evidence"] == 1
        assert "proposed" in stats["by_status"]

    def test_list_evidence_filter_by_source_type(self, store, ev_experiment, ev_backtest):
        store.add_evidence(ev_experiment)
        store.add_evidence(ev_backtest)
        exps = store.list_evidence(source_type="experiment")
        bts  = store.list_evidence(source_type="backtest")
        assert len(exps) == 1
        assert len(bts)  == 1

    # ── Persistence ────────────────────────────────────────────────────────

    def test_persists_and_reloads(self, tmp_path, simple_claim, ev_experiment):
        path = tmp_path / "claims.json"
        s1 = ClaimStore(path=path)
        s1.add_evidence(ev_experiment)
        s1.add_claim(simple_claim)
        s1.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)

        # Create new store instance pointing to same file
        s2 = ClaimStore(path=path)
        assert len(s2) == 1
        claim2 = s2.get_claim(simple_claim.claim_id)
        assert claim2.claim_id == simple_claim.claim_id
        assert ev_experiment.evidence_id in claim2.evidence_ids

    def test_json_file_is_valid_json(self, tmp_path, simple_claim):
        path = tmp_path / "claims.json"
        s = ClaimStore(path=path)
        s.add_claim(simple_claim)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "claims" in raw
        assert "evidence" in raw
        assert raw["version"] == "1"

    def test_empty_store_loads_cleanly_from_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        s = ClaimStore(path=path)
        assert len(s) == 0


# ── Phase 3: evaluate_claim ───────────────────────────────────────────────────

class TestEvaluateClaim:
    def test_no_evidence_gives_none_level(self, simple_claim: Claim):
        ev = evaluate_claim(simple_claim, [])
        assert ev.support_level  == "none"
        assert ev.evidence_count == 0
        assert ev.recommendation == "propose"
        assert ev.support_score  == pytest.approx(0.0)

    def test_one_evidence_gives_weak(self, simple_claim, ev_experiment):
        ev = evaluate_claim(simple_claim, [ev_experiment])
        assert ev.support_level  == "weak"
        assert ev.evidence_count == 1
        assert ev.recommendation == "flag_weak"

    def test_two_same_type_gives_weak(self, simple_claim):
        items = [
            EvidenceItem.new(EvidenceSourceType.EXPERIMENT, f"e{i}", f"summary {i}")
            for i in range(2)
        ]
        ev = evaluate_claim(simple_claim, items)
        assert ev.support_level         == "weak"
        assert ev.source_type_diversity == 1

    def test_two_different_types_gives_moderate(self, simple_claim, ev_experiment, ev_backtest):
        ev = evaluate_claim(simple_claim, [ev_experiment, ev_backtest])
        assert ev.support_level         == "moderate"
        assert ev.source_type_diversity == 2
        assert ev.recommendation        == "support"

    def test_four_items_three_types_gives_strong(self, simple_claim):
        items = [
            EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s"),
            EvidenceItem.new(EvidenceSourceType.BACKTEST,   "e2", "s"),
            EvidenceItem.new(EvidenceSourceType.DOCUMENT,   "e3", "s"),
            EvidenceItem.new(EvidenceSourceType.DATASET,    "e4", "s"),
        ]
        ev = evaluate_claim(simple_claim, items)
        assert ev.support_level         == "strong"
        assert ev.source_type_diversity == 4
        assert ev.recommendation        == "support"

    def test_majority_conflict_overrides_to_none(self, simple_claim):
        items = [
            EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s",
                             structured_data={"conflicts": True}),
            EvidenceItem.new(EvidenceSourceType.BACKTEST,   "e2", "s",
                             structured_data={"conflicts": True}),
            EvidenceItem.new(EvidenceSourceType.DATASET,    "e3", "s"),
        ]
        # 2/3 = 66% conflict → majority
        ev = evaluate_claim(simple_claim, items)
        assert ev.support_level    == "none"
        assert ev.recommendation   == "reject"
        assert ev.conflicting_count == 2

    def test_single_conflict_does_not_override(self, simple_claim):
        items = [
            EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s",
                             structured_data={"conflicts": True}),
            EvidenceItem.new(EvidenceSourceType.BACKTEST,   "e2", "s"),
            EvidenceItem.new(EvidenceSourceType.DOCUMENT,   "e3", "s"),
            EvidenceItem.new(EvidenceSourceType.DATASET,    "e4", "s"),
        ]
        # 1/4 = 25% conflict → not majority
        ev = evaluate_claim(simple_claim, items)
        assert ev.support_level     in ("strong", "moderate")
        assert ev.conflicting_count == 1
        assert ev.recommendation    != "reject"

    def test_counterpoints_mentioned_in_notes(self, simple_claim):
        simple_claim.counterpoints = ["Regime dependency", "Sample size small"]
        ev = evaluate_claim(simple_claim, [])
        assert any("counterpoint" in n.lower() for n in ev.notes)

    def test_evaluation_to_dict_is_serializable(self, simple_claim, ev_experiment):
        ev = evaluate_claim(simple_claim, [ev_experiment])
        d  = ev.to_dict()
        assert isinstance(json.dumps(d), str)
        assert d["claim_id"] == simple_claim.claim_id

    def test_support_score_monotonically_improves(self, simple_claim):
        """More evidence (up to strong) should increase support_score."""
        ev0 = evaluate_claim(simple_claim, []).support_score
        items_weak = [EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e1", "s")]
        ev1 = evaluate_claim(simple_claim, items_weak).support_score
        items_mod  = [
            EvidenceItem.new(EvidenceSourceType.EXPERIMENT, "e2", "s"),
            EvidenceItem.new(EvidenceSourceType.BACKTEST,   "e3", "s"),
        ]
        ev2 = evaluate_claim(simple_claim, items_mod).support_score
        assert ev0 < ev1 < ev2

    def test_batch_evaluate(self, store, simple_claim, ev_experiment):
        store.add_evidence(ev_experiment)
        store.add_claim(simple_claim)
        store.link_evidence(simple_claim.claim_id, ev_experiment.evidence_id)
        results = batch_evaluate(store)
        assert len(results) == 1
        assert isinstance(results[0], ClaimEvaluation)

    def test_batch_evaluate_specific_ids(self, store):
        c1 = Claim.new(ClaimType.GENERAL, "a", "p", "b", 0.5)
        c2 = Claim.new(ClaimType.GENERAL, "c", "p", "d", 0.5)
        store.add_claim(c1)
        store.add_claim(c2)
        results = batch_evaluate(store, claim_ids=[c1.claim_id])
        assert len(results) == 1
        assert results[0].claim_id == c1.claim_id


# ── Phase 4: claim templates ──────────────────────────────────────────────────

class TestClaimTemplates:
    def test_source_usefulness_triple(self):
        c = source_usefulness_claim(
            source_id  = "FRED_GDP",
            useful_for = "macro_regime",
        )
        assert c.subject     == "source:FRED_GDP"
        assert c.predicate   == "is_useful_for"
        assert c.object      == "domain:macro_regime"
        assert c.claim_type  == ClaimType.SOURCE_USEFULNESS
        assert c.status      == ClaimStatus.PROPOSED

    def test_source_usefulness_custom_confidence(self):
        c = source_usefulness_claim("X", "Y", confidence=0.8)
        assert c.confidence == pytest.approx(0.8)

    def test_source_usefulness_counterpoints_and_tags(self):
        c = source_usefulness_claim(
            "FRED_GDP", "macro_regime",
            counterpoints=["lagged by 1 quarter"],
            tags=["macro", "FRED"],
        )
        assert "lagged by 1 quarter" in c.counterpoints
        assert "macro" in c.tags

    def test_feature_usefulness_triple(self):
        c = feature_usefulness_claim(
            feature_name    = "rsi_14",
            model_type      = "mlp",
            dataset_version = "abc123",
        )
        assert c.subject    == "feature:rsi_14"
        assert c.predicate  == "improves_accuracy_on"
        assert c.object     == "model:mlp_dataset:abc123"
        assert c.claim_type == ClaimType.FEATURE_USEFULNESS

    def test_feature_usefulness_improvement_note_in_notes(self):
        c = feature_usefulness_claim(
            "rsi_14", "mlp", "abc123",
            improvement_note="acc +0.03 on 3-fold WF",
        )
        assert "acc +0.03" in c.notes

    def test_relationship_claim_triple(self):
        c = relationship_claim(
            asset_a   = "AAPL",
            asset_b   = "QQQ",
            criterion = "rolling_60d_correlation > 0.85",
        )
        assert c.subject    == "asset:AAPL"
        assert c.predicate  == "is_correlated_with"
        assert "QQQ" in c.object
        assert "rolling_60d_correlation" in c.object
        assert c.claim_type == ClaimType.RELATIONSHIP

    def test_relationship_claim_custom_predicate(self):
        c = relationship_claim(
            asset_a   = "AAPL",
            asset_b   = "SPY",
            criterion = "beta",
            predicate = "is_causal_for",
        )
        assert c.predicate == "is_causal_for"

    def test_performance_claim_triple(self):
        c = performance_claim(
            model_type      = "mlp",
            symbol          = "AAPL",
            dataset_version = "v1",
            metric          = "mean_accuracy",
            threshold       = 0.55,
            observed        = 0.58,
        )
        assert "mlp_on_AAPL" in c.subject
        assert c.predicate    == "meets_performance_bar"
        assert "mean_accuracy" in c.object
        assert c.claim_type   == ClaimType.PERFORMANCE
        assert "0.58" in c.notes or "0.5800" in c.notes

    def test_all_templates_produce_proposed_status(self):
        claims = [
            source_usefulness_claim("X", "Y"),
            feature_usefulness_claim("f", "mlp", "v1"),
            relationship_claim("A", "B", "corr"),
            performance_claim("mlp", "AAPL", "v1", "acc", 0.55, 0.58),
        ]
        for c in claims:
            assert c.status == ClaimStatus.PROPOSED

    def test_all_templates_round_trip(self):
        claims = [
            source_usefulness_claim("X", "Y"),
            feature_usefulness_claim("f", "mlp", "v1"),
            relationship_claim("A", "B", "corr"),
            performance_claim("mlp", "AAPL", "v1", "acc", 0.55, 0.58),
        ]
        for c in claims:
            d  = c.to_dict()
            c2 = Claim.from_dict(d)
            assert c2.claim_id   == c.claim_id
            assert c2.claim_type == c.claim_type


# ── Integration: full workflow ─────────────────────────────────────────────────

class TestIntegrationWorkflow:
    def test_full_claim_lifecycle(self, store):
        """Complete: create → add evidence → link → evaluate → promote."""
        # 1. Create claim from template
        claim = source_usefulness_claim(
            source_id  = "FRED_UNRATE",
            useful_for = "equity_regime",
            confidence = 0.5,
        )
        store.add_claim(claim)

        # 2. Add evidence from two distinct source types
        ev1 = EvidenceItem.new(
            EvidenceSourceType.DATASET, "FRED:UNRATE",
            "Unemployment rate shows mean-reversion aligned with SPY drawdowns",
            structured_data={"pearson_r": -0.43},
        )
        ev2 = EvidenceItem.new(
            EvidenceSourceType.EXPERIMENT, "exp_xyz",
            "Adding UNRATE to feature set improved val accuracy by 0.02",
            structured_data={"accuracy_delta": 0.02},
        )
        store.add_evidence(ev1)
        store.add_evidence(ev2)
        store.link_evidence(claim.claim_id, ev1.evidence_id)
        store.link_evidence(claim.claim_id, ev2.evidence_id)

        # 3. Evaluate
        items = store.get_evidence_for_claim(claim.claim_id)
        evaluation = evaluate_claim(claim, items)
        assert evaluation.support_level  == "moderate"
        assert evaluation.recommendation == "support"
        assert evaluation.evidence_count == 2

        # 4. Human decides to promote
        store.update_status(
            claim.claim_id, ClaimStatus.SUPPORTED,
            notes="Reviewed 2026-03-27; 2-source moderate support accepted"
        )
        updated = store.get_claim(claim.claim_id)
        assert updated.status == ClaimStatus.SUPPORTED

        # 5. Summary stats reflect the change
        stats = store.summary_stats()
        assert stats["by_status"].get("supported", 0) == 1

    def test_conflicting_evidence_leads_to_reject_recommendation(self, store, simple_claim):
        store.add_claim(simple_claim)
        ev1 = EvidenceItem.new(
            EvidenceSourceType.BACKTEST, "bt1",
            "RSI-14 ablation shows no accuracy improvement (conflicts)",
            structured_data={"conflicts": True, "accuracy_delta": -0.01},
        )
        ev2 = EvidenceItem.new(
            EvidenceSourceType.EXPERIMENT, "exp1",
            "Walk-forward run without RSI-14 performed equally (conflicts)",
            structured_data={"conflicts": True},
        )
        ev3 = EvidenceItem.new(
            EvidenceSourceType.DATASET, "data1",
            "RSI-14 marginally positive in pre-2022 data",
        )
        store.add_evidence(ev1)
        store.add_evidence(ev2)
        store.add_evidence(ev3)
        for eid in [ev1.evidence_id, ev2.evidence_id, ev3.evidence_id]:
            store.link_evidence(simple_claim.claim_id, eid)

        items = store.get_evidence_for_claim(simple_claim.claim_id)
        evaluation = evaluate_claim(simple_claim, items)
        # 2/3 = 66% conflict → "reject" recommendation
        assert evaluation.recommendation == "reject"
        assert evaluation.conflicting_count == 2
