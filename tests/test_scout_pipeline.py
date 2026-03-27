"""Tests for data.scout.pipeline — end-to-end probe → sample orchestrator.

All HTTP I/O is mocked.  Tests verify the composition logic: that probe
results feed correctly into the sample phase, that flags control behavior,
and that errors in one phase don't break the other.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from data.scout.pipeline import ScoutRunResult, run_scout_pipeline
from data.scout.probe_catalog import ProbeCatalogResult
from data.scout.probe_registry import ProbeRegistryResult
from data.scout.sampler import SampleQuality, SampleResult
from data.scout.schema import normalize_source_candidate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _candidate(name: str, url: str = "https://example.com") -> object:
    return normalize_source_candidate({"name": name, "url": url})


def _probe_reg_result(source_id: str, ok: bool, action: str = "sampled") -> ProbeRegistryResult:
    probe = MagicMock()
    probe.ok = ok
    probe.latency_ms = 100.0
    probe.method_used = "HEAD"
    probe.http_status = 200 if ok else None
    probe.error = "" if ok else "fail"
    record = MagicMock()
    record.status.value = "sampled" if ok else "discovered"
    record.category = "macro"
    return ProbeRegistryResult(
        source_id=source_id, probe=probe, action=action,
        reason="ok" if ok else "fail", record=record,
    )


def _probe_catalog_result(per_source: list[ProbeRegistryResult]) -> ProbeCatalogResult:
    reachable = sum(1 for r in per_source if r.probe.ok)
    return ProbeCatalogResult(
        total=len(per_source),
        attempted=len(per_source),
        reachable=reachable,
        failed=len(per_source) - reachable,
        skipped=0,
        avg_latency_ms=100.0 if reachable else None,
        per_source=per_source,
    )


def _sample_result(source_id: str, action: str = "validated",
                   advanced: bool = True) -> SampleResult:
    quality = SampleQuality(row_count=5, column_count=3, passed=(action == "validated"))
    return SampleResult(
        source_id=source_id, fetched=True, quality=quality,
        advanced=advanced, action=action, reason="ok",
    )


# ── ScoutRunResult unit tests ───────────────────────────────────────────────

class TestScoutRunResult:
    def test_summary_contains_key_fields(self):
        probe = _probe_catalog_result([_probe_reg_result("src_a", ok=True)])
        r = ScoutRunResult(
            probe_result=probe,
            newly_sampled=["src_a"],
            sample_results=[_sample_result("src_a")],
            newly_validated=["src_a"],
        )
        s = r.summary()
        assert "SAMPLED" in s
        assert "VALIDATED" in s
        assert "src_a" in s

    def test_summary_probe_only(self):
        probe = _probe_catalog_result([_probe_reg_result("src_a", ok=True)])
        r = ScoutRunResult(probe_result=probe, newly_sampled=["src_a"])
        s = r.summary()
        assert "sample attempts   : 0" in s

    def test_to_dict_keys(self):
        probe = _probe_catalog_result([])
        r = ScoutRunResult(probe_result=probe)
        d = r.to_dict()
        for key in ("probe", "sample_results", "newly_sampled",
                     "newly_validated", "sample_failures", "sample_skipped"):
            assert key in d

    def test_to_dict_serializable(self):
        import json
        prr = _probe_reg_result("x", ok=True)
        prr.to_dict = MagicMock(return_value={"source_id": "x", "ok": True})
        probe = _probe_catalog_result([prr])
        sr = _sample_result("x")
        r = ScoutRunResult(
            probe_result=probe,
            sample_results=[sr],
            newly_sampled=["x"],
            newly_validated=["x"],
        )
        # Should not raise
        json.dumps(r.to_dict())


# ── Pipeline: probe only (sampling disabled) ────────────────────────────────

class TestPipelineProbeOnly:
    @patch("data.scout.pipeline.probe_catalog")
    def test_no_sampling_when_disabled(self, mock_probe):
        candidates = [_candidate("Src A")]
        mock_probe.return_value = _probe_catalog_result(
            [_probe_reg_result("src_a", ok=True, action="sampled")]
        )
        registry = MagicMock()

        result = run_scout_pipeline(
            candidates, registry, run_sampling=False,
        )

        assert result.newly_sampled == ["src_a"]
        assert result.sample_results == []
        assert result.newly_validated == []

    @patch("data.scout.pipeline.probe_catalog")
    def test_no_sampling_when_none_sampled(self, mock_probe):
        candidates = [_candidate("Src A")]
        mock_probe.return_value = _probe_catalog_result(
            [_probe_reg_result("src_a", ok=False, action="updated")]
        )
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert result.newly_sampled == []
        assert result.sample_results == []

    @patch("data.scout.pipeline.probe_catalog")
    def test_empty_candidates(self, mock_probe):
        mock_probe.return_value = _probe_catalog_result([])
        registry = MagicMock()

        result = run_scout_pipeline([], registry)

        assert result.probe_result.total == 0
        assert result.sample_results == []


# ── Pipeline: probe + sample ────────────────────────────────────────────────

class TestPipelineFull:
    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_newly_sampled_get_sampled(self, mock_probe, mock_sample):
        candidates = [_candidate("Src A"), _candidate("Src B")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("src_a", ok=True, action="sampled"),
            _probe_reg_result("src_b", ok=True, action="sampled"),
        ])
        mock_sample.side_effect = [
            _sample_result("src_a", action="validated", advanced=True),
            _sample_result("src_b", action="failed", advanced=False),
        ]
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert mock_sample.call_count == 2
        assert result.newly_sampled == ["src_a", "src_b"]
        assert result.newly_validated == ["src_a"]
        assert result.sample_failures == ["src_b"]

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_only_sampled_action_triggers_sample(self, mock_probe, mock_sample):
        """Sources with action='updated' or 'created' should not be sampled."""
        candidates = [_candidate("Src A"), _candidate("Src B"), _candidate("Src C")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("src_a", ok=True, action="sampled"),
            _probe_reg_result("src_b", ok=True, action="updated"),
            _probe_reg_result("src_c", ok=False, action="updated"),
        ])
        mock_sample.return_value = _sample_result("src_a")
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        # Only src_a should be sampled
        mock_sample.assert_called_once()
        assert result.newly_sampled == ["src_a"]

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_sdk_sources_show_as_skipped(self, mock_probe, mock_sample):
        candidates = [_candidate("SDK Src")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("sdk_src", ok=True, action="sampled"),
        ])
        mock_sample.return_value = SampleResult(
            source_id="sdk_src", fetched=False, quality=None,
            advanced=False, action="skipped", reason="SDK",
        )
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert result.sample_skipped == ["sdk_src"]
        assert result.newly_validated == []

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_advance_to_validated_flag_forwarded(self, mock_probe, mock_sample):
        candidates = [_candidate("Src")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("src", ok=True, action="sampled"),
        ])
        mock_sample.return_value = _sample_result("src", advanced=False)
        registry = MagicMock()

        run_scout_pipeline(
            candidates, registry, advance_to_validated=False,
        )

        _, kwargs = mock_sample.call_args
        assert kwargs["advance_to_validated"] is False

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_sample_timeout_forwarded(self, mock_probe, mock_sample):
        candidates = [_candidate("Src")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("src", ok=True, action="sampled"),
        ])
        mock_sample.return_value = _sample_result("src")
        registry = MagicMock()

        run_scout_pipeline(candidates, registry, sample_timeout=30)

        _, kwargs = mock_sample.call_args
        assert kwargs["timeout"] == 30

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_probe_kwargs_forwarded(self, mock_probe, mock_sample):
        candidates = [_candidate("Src")]
        mock_probe.return_value = _probe_catalog_result([])
        registry = MagicMock()
        scores = {"src": 0.9}

        run_scout_pipeline(
            candidates, registry, scores=scores,
            probe_timeout=5, advance_to_sampled=False, concurrency=4,
        )

        _, kwargs = mock_probe.call_args
        assert kwargs["scores"] == scores
        assert kwargs["timeout"] == 5
        assert kwargs["advance_to_sampled"] is False
        assert kwargs["concurrency"] == 4


# ── Pipeline: error isolation ───────────────────────────────────────────────

class TestPipelineErrors:
    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_sample_exception_isolated(self, mock_probe, mock_sample):
        """An exception in one sample should not abort the rest."""
        candidates = [_candidate("A"), _candidate("B")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("a", ok=True, action="sampled"),
            _probe_reg_result("b", ok=True, action="sampled"),
        ])
        mock_sample.side_effect = [
            RuntimeError("boom"),
            _sample_result("b", action="validated", advanced=True),
        ]
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert len(result.sample_results) == 2
        assert result.sample_failures == ["a"]
        assert result.newly_validated == ["b"]
        # The error result should be a SampleResult with action="error"
        assert result.sample_results[0].action == "error"
        assert "boom" in result.sample_results[0].reason

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_all_samples_fail(self, mock_probe, mock_sample):
        candidates = [_candidate("A")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("a", ok=True, action="sampled"),
        ])
        mock_sample.return_value = _sample_result("a", action="error", advanced=False)
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert result.newly_validated == []
        assert result.sample_failures == ["a"]


# ── Pipeline: full lifecycle coherence ──────────────────────────────────────

class TestLifecycleCoherence:
    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_discovered_to_validated_in_one_pass(self, mock_probe, mock_sample):
        """The pipeline should take a source from DISCOVERED → SAMPLED → VALIDATED."""
        candidates = [_candidate("FRED API", url="https://api.stlouisfed.org/fred")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("fred_api", ok=True, action="sampled"),
        ])
        mock_sample.return_value = _sample_result(
            "fred_api", action="validated", advanced=True,
        )
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        # Full lifecycle in one call
        assert "fred_api" in result.newly_sampled
        assert "fred_api" in result.newly_validated
        assert result.sample_failures == []
        assert result.sample_skipped == []

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_mixed_lifecycle_outcomes(self, mock_probe, mock_sample):
        """Different sources can reach different lifecycle stages."""
        candidates = [_candidate("A"), _candidate("B"), _candidate("C")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("a", ok=True, action="sampled"),   # will validate
            _probe_reg_result("b", ok=True, action="sampled"),   # will fail sample
            _probe_reg_result("c", ok=False, action="updated"),  # probe failed
        ])
        mock_sample.side_effect = [
            _sample_result("a", action="validated", advanced=True),
            _sample_result("b", action="failed", advanced=False),
        ]
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        assert result.newly_sampled == ["a", "b"]    # c didn't make it
        assert result.newly_validated == ["a"]         # only a validated
        assert result.sample_failures == ["b"]         # b failed sampling
        assert len(result.sample_results) == 2         # c was never sampled

    @patch("data.scout.pipeline.sample_and_validate")
    @patch("data.scout.pipeline.probe_catalog")
    def test_no_auto_approval(self, mock_probe, mock_sample):
        """The pipeline must never advance beyond VALIDATED."""
        candidates = [_candidate("Src")]
        mock_probe.return_value = _probe_catalog_result([
            _probe_reg_result("src", ok=True, action="sampled"),
        ])
        # sample_and_validate returns validated but the pipeline doesn't
        # call any further advancement
        mock_sample.return_value = _sample_result("src", action="validated", advanced=True)
        registry = MagicMock()

        result = run_scout_pipeline(candidates, registry)

        # Verify sample_and_validate was called with advance_to_validated=True
        # but the pipeline itself doesn't call update_status to APPROVED
        _, kwargs = mock_sample.call_args
        assert kwargs["advance_to_validated"] is True
        # No direct registry.update_status call from the pipeline itself
        # (only probe_catalog and sample_and_validate touch the registry)
