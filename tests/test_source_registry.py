"""Tests for data.registry.source_registry and seed_sources."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from data.registry.source_registry import (
    SourceRecord,
    SourceRegistry,
    SourceStatus,
    _VALID_TRANSITIONS,
)
from data.registry.seed_sources import SEED_SOURCES, seed


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_registry(tmp_path: Path) -> SourceRegistry:
    """A fresh registry backed by a temp file."""
    return SourceRegistry(path=tmp_path / "sources.json")


def _make_record(source_id: str = "test_src") -> SourceRecord:
    return SourceRecord(
        source_id   = source_id,
        name        = "Test Source",
        category    = "price",
        url         = "https://example.com",
        asset_classes = ["equity"],
        data_types  = ["ohlcv"],
    )


# ── SourceRecord serialisation ────────────────────────────────────────────────

class TestSourceRecord:
    def test_default_status_is_discovered(self):
        rec = _make_record()
        assert rec.status == SourceStatus.DISCOVERED

    def test_to_dict_round_trip(self):
        rec = _make_record()
        d   = rec.to_dict()
        assert isinstance(d["status"], str)
        rec2 = SourceRecord.from_dict(d)
        assert rec2.source_id == rec.source_id
        assert rec2.status    == rec.status
        assert rec2.asset_classes == rec.asset_classes

    def test_from_dict_handles_status_string(self):
        rec = SourceRecord.from_dict({
            "source_id": "x", "name": "X", "category": "macro",
            "url": "http://x", "status": "approved",
            "asset_classes": [], "data_types": [],
        })
        assert rec.status == SourceStatus.APPROVED


# ── Registry CRUD ─────────────────────────────────────────────────────────────

class TestRegistryCRUD:
    def test_add_and_get(self, tmp_registry):
        rec = _make_record("src1")
        tmp_registry.add(rec)
        assert tmp_registry.get("src1").source_id == "src1"

    def test_add_duplicate_raises(self, tmp_registry):
        tmp_registry.add(_make_record("dup"))
        with pytest.raises(ValueError, match="already registered"):
            tmp_registry.add(_make_record("dup"))

    def test_add_overwrite(self, tmp_registry):
        tmp_registry.add(_make_record("ovr"))
        rec2 = _make_record("ovr")
        rec2.name = "Updated"
        tmp_registry.add(rec2, overwrite=True)
        assert tmp_registry.get("ovr").name == "Updated"

    def test_remove(self, tmp_registry):
        tmp_registry.add(_make_record("del_me"))
        tmp_registry.remove("del_me")
        with pytest.raises(KeyError):
            tmp_registry.get("del_me")

    def test_remove_missing_raises(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.remove("does_not_exist")

    def test_get_missing_raises(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.get("nonexistent")

    def test_contains(self, tmp_registry):
        tmp_registry.add(_make_record("in_reg"))
        assert "in_reg" in tmp_registry
        assert "out_reg" not in tmp_registry

    def test_len(self, tmp_registry):
        assert len(tmp_registry) == 0
        tmp_registry.add(_make_record("a"))
        tmp_registry.add(_make_record("b"))
        assert len(tmp_registry) == 2


# ── Status transitions ────────────────────────────────────────────────────────

class TestStatusTransitions:
    def test_valid_progression(self, tmp_registry):
        tmp_registry.add(_make_record("prog"))
        tmp_registry.update_status("prog", SourceStatus.SAMPLED)
        tmp_registry.update_status("prog", SourceStatus.VALIDATED)
        tmp_registry.update_status("prog", SourceStatus.APPROVED)
        assert tmp_registry.get("prog").status == SourceStatus.APPROVED

    def test_invalid_transition_raises(self, tmp_registry):
        tmp_registry.add(_make_record("bad"))
        # Can't go directly from DISCOVERED to APPROVED
        with pytest.raises(ValueError, match="Invalid transition"):
            tmp_registry.update_status("bad", SourceStatus.APPROVED)

    def test_skip_enforcement(self, tmp_registry):
        tmp_registry.add(_make_record("skip"))
        # Should work when enforcement is disabled
        tmp_registry.update_status("skip", SourceStatus.APPROVED, enforce_transitions=False)
        assert tmp_registry.get("skip").status == SourceStatus.APPROVED

    def test_quarantine_from_approved(self, tmp_registry):
        tmp_registry.add(_make_record("qr"), overwrite=True)
        tmp_registry.update_status("qr", SourceStatus.SAMPLED, enforce_transitions=False)
        tmp_registry.update_status("qr", SourceStatus.VALIDATED)
        tmp_registry.update_status("qr", SourceStatus.APPROVED)
        tmp_registry.update_status("qr", SourceStatus.QUARANTINED)
        assert tmp_registry.get("qr").status == SourceStatus.QUARANTINED

    def test_notes_appended_on_transition(self, tmp_registry):
        tmp_registry.add(_make_record("noted"))
        tmp_registry.update_status("noted", SourceStatus.SAMPLED, notes="First sample ok")
        assert "First sample ok" in tmp_registry.get("noted").notes

    def test_last_checked_updated(self, tmp_registry):
        tmp_registry.add(_make_record("ts"))
        rec = tmp_registry.get("ts")
        assert rec.last_checked_at is None
        tmp_registry.update_status("ts", SourceStatus.SAMPLED, enforce_transitions=False)
        assert tmp_registry.get("ts").last_checked_at is not None

    def test_all_transitions_defined(self):
        """Every SourceStatus must have an entry in _VALID_TRANSITIONS."""
        for status in SourceStatus:
            assert status in _VALID_TRANSITIONS, f"Missing transition entry for {status}"


# ── Reliability score ─────────────────────────────────────────────────────────

class TestReliabilityScore:
    def test_update_score(self, tmp_registry):
        tmp_registry.add(_make_record("scored"))
        tmp_registry.update_score("scored", 0.92)
        assert tmp_registry.get("scored").reliability_score == 0.92

    def test_score_out_of_range(self, tmp_registry):
        tmp_registry.add(_make_record("bad_score"))
        with pytest.raises(ValueError):
            tmp_registry.update_score("bad_score", 1.5)
        with pytest.raises(ValueError):
            tmp_registry.update_score("bad_score", -0.1)


# ── Filter and search ─────────────────────────────────────────────────────────

class TestFilterSearch:
    def _populated(self, path):
        reg = SourceRegistry(path=path / "sources.json")
        reg.add(SourceRecord("a", "A", "macro",     "http://a", status=SourceStatus.APPROVED, asset_classes=["macro"], data_types=["yields"]))
        reg.add(SourceRecord("b", "B", "price",     "http://b", status=SourceStatus.DISCOVERED, asset_classes=["equity"], data_types=["ohlcv"]))
        reg.add(SourceRecord("c", "C", "sentiment", "http://c", status=SourceStatus.APPROVED, asset_classes=["equity"], data_types=["sentiment"]))
        reg.add(SourceRecord("d", "D", "macro",     "http://d", status=SourceStatus.REJECTED, asset_classes=["macro"], data_types=["inflation"], reliability_score=0.2))
        return reg

    def test_filter_by_category(self, tmp_path):
        reg = self._populated(tmp_path)
        assert {r.source_id for r in reg.filter(category="macro")} == {"a", "d"}

    def test_filter_by_status(self, tmp_path):
        reg = self._populated(tmp_path)
        assert {r.source_id for r in reg.filter(status=SourceStatus.APPROVED)} == {"a", "c"}

    def test_filter_by_asset_class(self, tmp_path):
        reg = self._populated(tmp_path)
        assert {r.source_id for r in reg.filter(asset_class="equity")} == {"b", "c"}

    def test_filter_by_data_type(self, tmp_path):
        reg = self._populated(tmp_path)
        assert {r.source_id for r in reg.filter(data_type="ohlcv")} == {"b"}

    def test_filter_combined(self, tmp_path):
        reg = self._populated(tmp_path)
        result = reg.filter(category="macro", status=SourceStatus.APPROVED)
        assert [r.source_id for r in result] == ["a"]

    def test_filter_min_score(self, tmp_path):
        reg = self._populated(tmp_path)
        result = reg.filter(min_score=0.5)
        assert "d" not in {r.source_id for r in result}

    def test_search_by_name(self, tmp_path):
        reg = self._populated(tmp_path)
        result = reg.search("macro")
        assert all(r.category == "macro" for r in result)

    def test_all_returns_all(self, tmp_path):
        reg = self._populated(tmp_path)
        assert len(reg.all()) == 4


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_survives_reload(self, tmp_path):
        path = tmp_path / "sources.json"
        reg1 = SourceRegistry(path=path)
        reg1.add(_make_record("persist_me"))
        reg1.update_status("persist_me", SourceStatus.SAMPLED, enforce_transitions=False)

        reg2 = SourceRegistry(path=path)
        rec = reg2.get("persist_me")
        assert rec.source_id == "persist_me"
        assert rec.status == SourceStatus.SAMPLED

    def test_json_format(self, tmp_path):
        path = tmp_path / "s.json"
        reg = SourceRegistry(path=path)
        reg.add(_make_record("json_test"))
        payload = json.loads(path.read_text())
        assert payload["version"] == "1"
        assert len(payload["sources"]) == 1
        assert payload["sources"][0]["source_id"] == "json_test"

    def test_summary(self, tmp_path):
        reg = SourceRegistry(path=tmp_path / "s.json")
        reg.add(_make_record("s1"))
        reg.add(_make_record("s2"))
        reg.update_status("s2", SourceStatus.SAMPLED, enforce_transitions=False)
        s = reg.summary()
        assert s["total"] == 2
        assert s["by_status"]["discovered"] == 1
        assert s["by_status"]["sampled"] == 1


# ── Seed sources ──────────────────────────────────────────────────────────────

class TestSeedSources:
    def test_seed_populates_registry(self, tmp_path):
        reg = seed(registry=SourceRegistry(path=tmp_path / "s.json"))
        assert len(reg) == len(SEED_SOURCES)

    def test_seed_idempotent(self, tmp_path):
        reg = SourceRegistry(path=tmp_path / "s.json")
        seed(registry=reg)
        seed(registry=reg)  # second call should not raise, just skip
        assert len(reg) == len(SEED_SOURCES)

    def test_all_seeds_have_required_fields(self):
        for src in SEED_SOURCES:
            assert src.source_id, "source_id must not be empty"
            assert src.name,      "name must not be empty"
            assert src.url,       "url must not be empty"
            assert src.category,  "category must not be empty"
            assert 0.0 <= src.reliability_score <= 1.0

    def test_approved_seeds_have_notes(self):
        """Approved sources must explain why they are approved."""
        for src in SEED_SOURCES:
            if src.status == SourceStatus.APPROVED:
                assert src.notes, f"{src.source_id} is APPROVED but has no notes"
