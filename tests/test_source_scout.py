"""Tests for the data/scout source-scout layer.

Covers:
- SourceCandidate normalization (slugify, category inference, dedup, cadence)
- Scorer (weights, ordering, breakdown consistency)
- ConnectorProposal (priority thresholds, payload shape, auth notes)
- Registry bridge (create, update, dedup by id, dedup by URL, overwrite)
- Evidence hooks (EvidenceItem fields, Claim fields, confidence clamping)
- Integration (full pipeline: raw → candidate → score → proposal → register
  → evidence → claim)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus
from data.scout.evidence_hooks import evidence_from_candidate, source_claim_from_candidate
from data.scout.proposal import ConnectorProposal, propose_connector_spec
from data.scout.registry_bridge import (
    RegistrationResult,
    register_candidate_source,
    register_catalog,
)
from data.scout.schema import (
    CANDIDATE_CATALOG,
    SourceCandidate,
    normalize_source_candidate,
)
from data.scout.scorer import score_breakdown, score_source_candidate
from ml.evidence.schema import (
    ClaimType,
    EvidenceSourceType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path: Path) -> SourceRegistry:
    return SourceRegistry(path=tmp_path / "sources.json")


def _raw_ecb() -> dict:
    return {
        "name":               "ECB Statistical Data Warehouse",
        "url":                "https://sdw-wsrest.ecb.europa.eu/service",
        "category":           "macro",
        "acquisition_method": "api",
        "auth_required":      False,
        "free_tier":          True,
        "official_source":    True,
        "update_cadence":     "daily",
        "asset_types":        ["macro"],
        "data_types":         ["yields", "inflation", "money_supply", "exchange_rates"],
        "schema_clarity":     4,
        "notes":              "ECB SDMX REST API.",
    }


def _raw_partial() -> dict:
    """Minimal valid raw input."""
    return {"name": "Some Data Feed", "url": "https://example.com/data"}


# ── Phase 1: normalize_source_candidate ──────────────────────────────────────

class TestNormalizeSourceCandidate:
    def test_returns_source_candidate(self):
        c = normalize_source_candidate(_raw_ecb())
        assert isinstance(c, SourceCandidate)

    def test_source_id_generated_from_name(self):
        c = normalize_source_candidate(_raw_ecb())
        assert c.source_id == "ecb_statistical_data_warehouse"

    def test_explicit_source_id_preserved(self):
        raw = {**_raw_ecb(), "source_id": "ecb_sdw"}
        c = normalize_source_candidate(raw)
        assert c.source_id == "ecb_sdw"

    def test_category_preserved_when_known(self):
        c = normalize_source_candidate(_raw_ecb())
        assert c.category == "macro"

    def test_category_inferred_from_asset_types(self):
        raw = {"name": "Crypto Feed", "url": "https://x.com", "asset_types": ["crypto"]}
        c = normalize_source_candidate(raw)
        assert c.category == "crypto"

    def test_unknown_category_when_no_assets(self):
        c = normalize_source_candidate(_raw_partial())
        assert c.category == "unknown"

    def test_cadence_alias_normalized(self):
        raw = {**_raw_ecb(), "update_cadence": "eod"}
        c = normalize_source_candidate(raw)
        assert c.update_cadence == "daily"

    def test_cadence_realtime_alias(self):
        raw = {**_raw_ecb(), "update_cadence": "real-time"}
        c = normalize_source_candidate(raw)
        assert c.update_cadence == "realtime"

    def test_data_types_lowercased_and_deduped(self):
        raw = {**_raw_ecb(), "data_types": ["Yields", "YIELDS", "inflation"]}
        c = normalize_source_candidate(raw)
        assert c.data_types == ["yields", "inflation"]

    def test_schema_clarity_clamped(self):
        raw = {**_raw_ecb(), "schema_clarity": 10}
        c = normalize_source_candidate(raw)
        assert c.schema_clarity == 5

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            normalize_source_candidate({"url": "https://x.com"})

    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            normalize_source_candidate({"name": "X"})

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            normalize_source_candidate({"name": "  ", "url": "https://x.com"})

    def test_discovered_at_set(self):
        c = normalize_source_candidate(_raw_ecb())
        assert c.discovered_at

    def test_roundtrip_to_dict(self):
        c = normalize_source_candidate(_raw_ecb())
        c2 = SourceCandidate.from_dict(c.to_dict())
        assert c2.source_id   == c.source_id
        assert c2.data_types  == c.data_types
        assert c2.update_cadence == c.update_cadence

    def test_partial_raw_gets_defaults(self):
        c = normalize_source_candidate(_raw_partial())
        assert c.acquisition_method == "api"
        assert c.free_tier           is True
        assert c.auth_required       is False
        assert c.official_source     is False
        assert c.schema_clarity      == 3

    def test_asset_classes_alias_accepted(self):
        raw = {**_raw_ecb(), "asset_classes": ["macro", "bond"]}
        c = normalize_source_candidate(raw)
        assert "macro" in c.asset_types

    def test_source_id_slug_long_name_truncated(self):
        name = "A" * 100
        c = normalize_source_candidate({"name": name, "url": "https://x.com"})
        assert len(c.source_id) <= 64

    def test_candidate_catalog_all_valid(self):
        for raw in CANDIDATE_CATALOG:
            c = normalize_source_candidate(raw)
            assert c.source_id
            assert c.url


# ── Phase 2: scorer ───────────────────────────────────────────────────────────

class TestScoreSourceCandidate:
    def test_returns_float_in_range(self):
        c = normalize_source_candidate(_raw_ecb())
        s = score_source_candidate(c)
        assert 0.0 <= s <= 1.0

    def test_official_free_scores_higher(self):
        official = normalize_source_candidate(_raw_ecb())     # official=True, free=True
        partial  = normalize_source_candidate(_raw_partial())  # official=False defaults
        assert score_source_candidate(official) > score_source_candidate(partial)

    def test_auth_required_lowers_score(self):
        raw_no_auth  = {**_raw_ecb(), "auth_required": False}
        raw_auth     = {**_raw_ecb(), "auth_required": True}
        s_no_auth = score_source_candidate(normalize_source_candidate(raw_no_auth))
        s_auth    = score_source_candidate(normalize_source_candidate(raw_auth))
        assert s_no_auth > s_auth

    def test_known_cadence_adds_score(self):
        raw_known   = {**_raw_ecb(), "update_cadence": "daily"}
        raw_unknown = {**_raw_ecb(), "update_cadence": "unknown"}
        s_known   = score_source_candidate(normalize_source_candidate(raw_known))
        s_unknown = score_source_candidate(normalize_source_candidate(raw_unknown))
        assert s_known > s_unknown

    def test_higher_schema_clarity_scores_higher(self):
        c1 = normalize_source_candidate({**_raw_ecb(), "schema_clarity": 5})
        c2 = normalize_source_candidate({**_raw_ecb(), "schema_clarity": 1})
        assert score_source_candidate(c1) > score_source_candidate(c2)

    def test_more_data_types_boosts_broad_utility(self):
        few  = normalize_source_candidate({**_raw_ecb(), "data_types": ["yields"]})
        many = normalize_source_candidate({**_raw_ecb(), "data_types": [
            "yields", "inflation", "gdp", "employment", "money_supply"
        ]})
        assert score_source_candidate(many) > score_source_candidate(few)

    def test_score_breakdown_total_matches_score(self):
        c = normalize_source_candidate(_raw_ecb())
        s = score_source_candidate(c)
        bd = score_breakdown(c)
        assert abs(bd["total"] - s) < 1e-6

    def test_score_breakdown_keys(self):
        c = normalize_source_candidate(_raw_ecb())
        bd = score_breakdown(c)
        expected = {
            "official_source", "free_tier", "no_auth", "cadence_clarity",
            "schema_clarity", "financial_relevance", "broad_utility", "total",
        }
        assert set(bd.keys()) == expected

    def test_all_catalog_candidates_scoreable(self):
        for raw in CANDIDATE_CATALOG:
            c = normalize_source_candidate(raw)
            s = score_source_candidate(c)
            assert 0.0 <= s <= 1.0


# ── Phase 2: propose_connector_spec ──────────────────────────────────────────

class TestProposeConnectorSpec:
    def test_returns_connector_proposal(self):
        c  = normalize_source_candidate(_raw_ecb())
        s  = score_source_candidate(c)
        sp = propose_connector_spec(c, s)
        assert isinstance(sp, ConnectorProposal)

    def test_high_score_yields_high_priority(self):
        raw = {**_raw_ecb(), "official_source": True, "free_tier": True,
               "schema_clarity": 5, "data_types": ["yields", "inflation", "gdp", "employment"]}
        c = normalize_source_candidate(raw)
        s = score_source_candidate(c)
        sp = propose_connector_spec(c, s)
        assert sp.priority == "high"

    def test_low_score_yields_low_priority(self):
        c  = normalize_source_candidate(_raw_partial())
        s  = score_source_candidate(c)   # minimal candidate → low score
        sp = propose_connector_spec(c, s)
        assert sp.priority in ("low", "medium")

    def test_payload_shape_populated(self):
        c  = normalize_source_candidate(_raw_ecb())
        sp = propose_connector_spec(c, 0.75)
        assert len(sp.expected_payload_shape) > 0

    def test_payload_shape_for_ohlcv(self):
        raw = {"name": "Price Feed", "url": "https://x.com", "data_types": ["ohlcv"]}
        c  = normalize_source_candidate(raw)
        sp = propose_connector_spec(c, 0.5)
        assert "close" in sp.expected_payload_shape
        assert "volume" in sp.expected_payload_shape

    def test_no_auth_note_when_auth_not_required(self):
        c  = normalize_source_candidate({**_raw_ecb(), "auth_required": False})
        sp = propose_connector_spec(c, 0.7)
        assert "No authentication" in sp.auth_notes

    def test_auth_required_note_mentions_env_var(self):
        c  = normalize_source_candidate({**_raw_ecb(), "auth_required": True})
        sp = propose_connector_spec(c, 0.7)
        assert "API_KEY" in sp.auth_notes

    def test_file_download_validation_hint(self):
        raw = {**_raw_ecb(), "acquisition_method": "file_download"}
        c  = normalize_source_candidate(raw)
        sp = propose_connector_spec(c, 0.7)
        assert "checksum" in sp.validation_strategy_hint or "download" in sp.validation_strategy_hint.lower()

    def test_implementation_notes_includes_score(self):
        c  = normalize_source_candidate(_raw_ecb())
        sp = propose_connector_spec(c, 0.82)
        assert "0.82" in sp.implementation_notes

    def test_to_dict_serializable(self):
        import json
        c  = normalize_source_candidate(_raw_ecb())
        sp = propose_connector_spec(c, 0.75)
        d  = sp.to_dict()
        assert json.dumps(d)  # must not raise


# ── Phase 3: registry_bridge ─────────────────────────────────────────────────

class TestRegisterCandidateSource:
    def test_new_candidate_creates_record(self, registry):
        c  = normalize_source_candidate(_raw_ecb())
        s  = score_source_candidate(c)
        r  = register_candidate_source(c, registry, score=s)
        assert r.action    == "created"
        assert r.record    is not None
        assert r.record.status == SourceStatus.DISCOVERED

    def test_status_always_discovered(self, registry):
        c = normalize_source_candidate(_raw_ecb())
        r = register_candidate_source(c, registry)
        assert r.record.status == SourceStatus.DISCOVERED

    def test_score_stored_as_reliability(self, registry):
        c = normalize_source_candidate(_raw_ecb())
        r = register_candidate_source(c, registry, score=0.77)
        assert r.record.reliability_score == pytest.approx(0.77)

    def test_duplicate_id_returns_updated(self, registry):
        c = normalize_source_candidate(_raw_ecb())
        register_candidate_source(c, registry)
        # Second call with same candidate
        r2 = register_candidate_source(c, registry)
        assert r2.action == "updated"
        assert len(registry) == 1

    def test_duplicate_url_detected(self, registry):
        c1 = normalize_source_candidate({
            "name": "ECB v1", "url": "https://sdw-wsrest.ecb.europa.eu/service",
            "source_id": "ecb_v1",
        })
        c2 = normalize_source_candidate({
            "name": "ECB v2", "url": "https://sdw-wsrest.ecb.europa.eu/service/",  # trailing /
            "source_id": "ecb_v2",
        })
        register_candidate_source(c1, registry)
        r2 = register_candidate_source(c2, registry)
        # Should update the existing record rather than create a duplicate
        assert r2.action == "updated"
        assert len(registry) == 1

    def test_promoted_record_notes_merged_not_status_changed(self, registry):
        c = normalize_source_candidate(_raw_ecb())
        r = register_candidate_source(c, registry)
        # Manually promote to SAMPLED
        registry.update_status(c.source_id, SourceStatus.SAMPLED)
        # Re-register with new notes
        c2 = normalize_source_candidate({**_raw_ecb(), "notes": "Extra context."})
        r2 = register_candidate_source(c2, registry)
        assert r2.action == "updated"
        assert registry.get(c.source_id).status == SourceStatus.SAMPLED
        assert "Extra context" in registry.get(c.source_id).notes

    def test_overwrite_flag_replaces_record(self, registry):
        c = normalize_source_candidate(_raw_ecb())
        register_candidate_source(c, registry)
        registry.update_status(c.source_id, SourceStatus.APPROVED,
                                enforce_transitions=False)
        r = register_candidate_source(c, registry, overwrite=True)
        assert r.action == "updated"

    def test_register_catalog_registers_all(self, registry):
        candidates = [normalize_source_candidate(raw) for raw in CANDIDATE_CATALOG]
        results = register_catalog(candidates, registry)
        assert len(results) == len(CANDIDATE_CATALOG)
        assert all(r.action in ("created", "updated") for r in results)
        assert len(registry) == len(CANDIDATE_CATALOG)

    def test_register_catalog_with_scores(self, registry):
        candidates = [normalize_source_candidate(raw) for raw in CANDIDATE_CATALOG[:2]]
        scores = {c.source_id: score_source_candidate(c) for c in candidates}
        results = register_catalog(candidates, registry, scores=scores)
        for r in results:
            assert r.record.reliability_score > 0

    def test_unknown_category_remapped_to_alternative(self, registry):
        c = normalize_source_candidate({"name": "Misc Feed", "url": "https://misc.io"})
        r = register_candidate_source(c, registry)
        assert r.record.category == "alternative"


# ── Phase 4: evidence hooks ───────────────────────────────────────────────────

class TestEvidenceFromCandidate:
    def test_returns_evidence_item(self):
        c  = normalize_source_candidate(_raw_ecb())
        ev = evidence_from_candidate(c, 0.80)
        from ml.evidence.schema import EvidenceItem
        assert isinstance(ev, EvidenceItem)

    def test_source_type_is_source_registry(self):
        c  = normalize_source_candidate(_raw_ecb())
        ev = evidence_from_candidate(c, 0.80)
        assert ev.source_type == EvidenceSourceType.SOURCE_REGISTRY

    def test_source_ref_contains_source_id(self):
        c  = normalize_source_candidate(_raw_ecb())
        ev = evidence_from_candidate(c, 0.80)
        assert c.source_id in ev.source_ref

    def test_structured_data_has_score(self):
        c  = normalize_source_candidate(_raw_ecb())
        ev = evidence_from_candidate(c, 0.75)
        assert ev.structured_data["usefulness_score"] == pytest.approx(0.75)

    def test_structured_data_has_key_fields(self):
        c  = normalize_source_candidate(_raw_ecb())
        ev = evidence_from_candidate(c, 0.80)
        sd = ev.structured_data
        assert "source_id"       in sd
        assert "category"        in sd
        assert "official_source" in sd
        assert "free_tier"       in sd


class TestSourceClaimFromCandidate:
    def test_returns_claim(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        from ml.evidence.schema import Claim
        assert isinstance(cl, Claim)

    def test_claim_type_is_source_usefulness(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert cl.claim_type == ClaimType.SOURCE_USEFULNESS

    def test_subject_is_source_prefixed(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert cl.subject == f"source:{c.source_id}"

    def test_object_is_domain_prefixed(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert cl.object.startswith("domain:")
        assert "macro" in cl.object

    def test_confidence_equals_score(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert cl.confidence == pytest.approx(0.72)

    def test_confidence_clamped_to_1(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 1.5)
        assert cl.confidence <= 1.0

    def test_custom_useful_for_overrides_category(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.7, useful_for="regime_detection")
        assert "regime_detection" in cl.object

    def test_status_is_proposed(self):
        from ml.evidence.schema import ClaimStatus
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert cl.status == ClaimStatus.PROPOSED

    def test_tags_include_scout(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert "scout" in cl.tags

    def test_uncertainty_notes_mention_heuristics(self):
        c  = normalize_source_candidate(_raw_ecb())
        cl = source_claim_from_candidate(c, 0.72)
        assert "heuristic" in cl.uncertainty_notes.lower()


# ── Integration ───────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_raw_to_evidence_and_claim(self, registry, tmp_path):
        from ml.evidence.store import ClaimStore
        claim_store = ClaimStore(path=tmp_path / "claims.json")

        # Normalize
        c = normalize_source_candidate(_raw_ecb())

        # Score
        score = score_source_candidate(c)
        assert score > 0

        # Proposal
        spec = propose_connector_spec(c, score)
        assert spec.priority in ("high", "medium", "low")

        # Register
        result = register_candidate_source(c, registry, score=score)
        assert result.action   == "created"
        assert result.record.status == SourceStatus.DISCOVERED

        # Evidence + claim
        ev    = evidence_from_candidate(c, score)
        claim = source_claim_from_candidate(c, score)

        claim_store.add_evidence(ev)
        claim_store.add_claim(claim)
        claim_store.link_evidence(claim.claim_id, ev.evidence_id)

        # Verify linkage
        items = claim_store.get_evidence_for_claim(claim.claim_id)
        assert len(items) == 1
        assert items[0].evidence_id == ev.evidence_id

    def test_full_catalog_pipeline(self, registry):
        candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
        scores = {c.source_id: score_source_candidate(c) for c in candidates}
        specs  = [propose_connector_spec(c, scores[c.source_id]) for c in candidates]

        # All high-value official sources should score above 0.55
        official = [c for c in candidates if c.official_source]
        for c in official:
            assert scores[c.source_id] >= 0.55, (
                f"{c.source_id} official source scored too low: {scores[c.source_id]}"
            )

        results = register_catalog(candidates, registry, scores=scores)
        assert len(results) == len(CANDIDATE_CATALOG)
        assert registry.summary()["total"] == len(CANDIDATE_CATALOG)
        assert all(r.record.status == SourceStatus.DISCOVERED for r in results)

        # All specs should have non-empty payload shapes
        for spec in specs:
            assert len(spec.expected_payload_shape) > 0
