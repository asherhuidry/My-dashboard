"""Tests for data.scout.probe_catalog — batch probe runner.

All HTTP I/O is mocked so the suite is deterministic and offline.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from data.scout.probe_catalog import (
    ProbeCatalogResult,
    _build_catalog_result,
    _error_result,
    probe_catalog,
)
from data.scout.probe_registry import ProbeRegistryResult
from data.scout.schema import normalize_source_candidate


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(name: str, url: str = "https://example.com", method: str = "http") -> Any:
    return normalize_source_candidate(
        {"name": name, "url": url, "acquisition_method": method}
    )


def _make_probe(ok: bool, latency_ms: float = 120.0, method_used: str = "HEAD",
                http_status: int | None = 200) -> MagicMock:
    probe = MagicMock()
    probe.ok = ok
    probe.latency_ms = latency_ms
    probe.method_used = method_used
    probe.http_status = http_status
    probe.error = "" if ok else "connection refused"
    return probe


def _make_registry_result(source_id: str, ok: bool, latency_ms: float = 120.0,
                           method_used: str = "HEAD", action: str = "sampled",
                           http_status: int | None = 200) -> ProbeRegistryResult:
    probe = _make_probe(ok, latency_ms=latency_ms, method_used=method_used,
                        http_status=http_status)
    record = MagicMock()
    record.status.value = "SAMPLED" if ok else "DISCOVERED"
    record.category = "macro"
    return ProbeRegistryResult(
        source_id=source_id,
        probe=probe,
        action=action,
        reason="probe ok" if ok else "probe failed",
        record=record,
    )


def _make_registry() -> MagicMock:
    return MagicMock()


# ── ProbeCatalogResult unit tests ─────────────────────────────────────────────

class TestProbeCatalogResult:
    def _result(self, reachable=3, attempted=4, total=5, failed=1, skipped=1,
                avg=150.0) -> ProbeCatalogResult:
        return ProbeCatalogResult(
            total=total,
            attempted=attempted,
            reachable=reachable,
            failed=failed,
            skipped=skipped,
            avg_latency_ms=avg,
        )

    def test_success_rate_typical(self):
        r = self._result(reachable=3, attempted=4)
        assert r.success_rate == 0.75

    def test_success_rate_zero_attempted(self):
        r = self._result(reachable=0, attempted=0)
        assert r.success_rate == 0.0

    def test_success_rate_all_reachable(self):
        r = self._result(reachable=5, attempted=5)
        assert r.success_rate == 1.0

    def test_get_existing(self):
        rr = _make_registry_result("fred_api", ok=True)
        r = ProbeCatalogResult(
            total=1, attempted=1, reachable=1, failed=0, skipped=0,
            avg_latency_ms=100.0, per_source=[rr],
        )
        assert r.get("fred_api") is rr

    def test_get_missing_returns_none(self):
        r = self._result()
        assert r.get("nonexistent") is None

    def test_summary_contains_key_fields(self):
        r = self._result()
        s = r.summary()
        assert "total" in s
        assert "reachable" in s
        assert "150ms" in s
        assert "75.0%" in s

    def test_summary_avg_none(self):
        r = ProbeCatalogResult(
            total=1, attempted=0, reachable=0, failed=0, skipped=1,
            avg_latency_ms=None,
        )
        assert "n/a" in r.summary()

    def test_summary_table_header_and_sep(self):
        r = self._result()
        rows = r.summary_table()
        assert len(rows) >= 2
        assert "source_id" in rows[0]
        assert "-" * 10 in rows[1]

    def test_summary_table_rows_sorted(self):
        rr_b = _make_registry_result("source_b", ok=True)
        rr_a = _make_registry_result("source_a", ok=True)
        r = ProbeCatalogResult(
            total=2, attempted=2, reachable=2, failed=0, skipped=0,
            avg_latency_ms=100.0, per_source=[rr_b, rr_a],
        )
        rows = r.summary_table()
        data_rows = rows[2:]
        assert data_rows[0].startswith("source_a")
        assert data_rows[1].startswith("source_b")

    def test_summary_table_ok_column(self):
        rr_ok = _make_registry_result("ok_src", ok=True)
        rr_fail = _make_registry_result("fail_src", ok=False)
        r = ProbeCatalogResult(
            total=2, attempted=2, reachable=1, failed=1, skipped=0,
            avg_latency_ms=100.0, per_source=[rr_ok, rr_fail],
        )
        rows = r.summary_table()
        row_text = "\n".join(rows[2:])
        assert "yes" in row_text
        assert "NO" in row_text

    def test_to_dict_keys(self):
        r = self._result()
        d = r.to_dict()
        for key in ("total", "attempted", "reachable", "failed", "skipped",
                    "success_rate", "avg_latency_ms", "per_source"):
            assert key in d

    def test_to_dict_avg_latency_rounded(self):
        r = ProbeCatalogResult(
            total=1, attempted=1, reachable=1, failed=0, skipped=0,
            avg_latency_ms=142.678,
        )
        d = r.to_dict()
        assert d["avg_latency_ms"] == 142.7

    def test_to_dict_avg_latency_none(self):
        r = ProbeCatalogResult(
            total=0, attempted=0, reachable=0, failed=0, skipped=0,
            avg_latency_ms=None,
        )
        assert r.to_dict()["avg_latency_ms"] is None

    def test_to_dict_per_source_calls_to_dict(self):
        rr = _make_registry_result("src", ok=True)
        rr.to_dict = MagicMock(return_value={"source_id": "src"})
        r = ProbeCatalogResult(
            total=1, attempted=1, reachable=1, failed=0, skipped=0,
            avg_latency_ms=100.0, per_source=[rr],
        )
        d = r.to_dict()
        assert d["per_source"] == [{"source_id": "src"}]
        rr.to_dict.assert_called_once()


# ── _build_catalog_result ─────────────────────────────────────────────────────

class TestBuildCatalogResult:
    def _candidates(self, n=3):
        return [_make_candidate(f"Source {i}") for i in range(n)]

    def test_empty_results(self):
        candidates = self._candidates(2)
        result = _build_catalog_result(candidates, [])
        assert result.total == 2
        assert result.attempted == 0
        assert result.skipped == 2
        assert result.avg_latency_ms is None

    def test_all_reachable(self):
        candidates = self._candidates(3)
        results = [_make_registry_result(c.source_id, ok=True, latency_ms=100.0)
                   for c in candidates]
        r = _build_catalog_result(candidates, results)
        assert r.reachable == 3
        assert r.failed == 0
        assert r.avg_latency_ms == 100.0

    def test_mixed_ok_fail(self):
        candidates = self._candidates(4)
        results = [
            _make_registry_result(candidates[0].source_id, ok=True, latency_ms=80.0),
            _make_registry_result(candidates[1].source_id, ok=True, latency_ms=120.0),
            _make_registry_result(candidates[2].source_id, ok=False),
            _make_registry_result(candidates[3].source_id, ok=False),
        ]
        r = _build_catalog_result(candidates, results)
        assert r.reachable == 2
        assert r.failed == 2
        assert r.skipped == 0
        assert r.avg_latency_ms == 100.0  # (80+120)/2

    def test_skipped_when_results_less_than_candidates(self):
        candidates = self._candidates(5)
        results = [_make_registry_result(c.source_id, ok=True) for c in candidates[:3]]
        r = _build_catalog_result(candidates, results)
        assert r.skipped == 2

    def test_sdk_probes_excluded_from_avg_latency(self):
        candidates = self._candidates(2)
        results = [
            _make_registry_result(candidates[0].source_id, ok=True,
                                  latency_ms=200.0, method_used="HEAD"),
            _make_registry_result(candidates[1].source_id, ok=True,
                                  latency_ms=0.0, method_used="NONE"),
        ]
        r = _build_catalog_result(candidates, results)
        # Only the HEAD probe's latency should count
        assert r.avg_latency_ms == 200.0

    def test_avg_latency_none_when_all_sdk(self):
        candidates = self._candidates(2)
        results = [
            _make_registry_result(c.source_id, ok=True,
                                  latency_ms=0.0, method_used="NONE")
            for c in candidates
        ]
        r = _build_catalog_result(candidates, results)
        assert r.avg_latency_ms is None

    def test_zero_latency_http_probes_excluded(self):
        """Probes with latency_ms=0 (error result) are excluded from avg."""
        candidates = self._candidates(2)
        results = [
            _make_registry_result(candidates[0].source_id, ok=True,
                                  latency_ms=300.0, method_used="HEAD"),
            _make_registry_result(candidates[1].source_id, ok=True,
                                  latency_ms=0.0, method_used="HEAD"),
        ]
        r = _build_catalog_result(candidates, results)
        assert r.avg_latency_ms == 300.0

    def test_per_source_populated(self):
        candidates = self._candidates(2)
        results = [_make_registry_result(c.source_id, ok=True) for c in candidates]
        r = _build_catalog_result(candidates, results)
        assert len(r.per_source) == 2


# ── _error_result ─────────────────────────────────────────────────────────────

class TestErrorResult:
    def test_error_result_fields(self):
        c = _make_candidate("Bad Source", url="https://bad.example.com")
        exc = RuntimeError("DNS failure")
        r = _error_result(c, exc)
        assert r.source_id == c.source_id
        assert r.action == "error"
        assert r.probe.ok is False
        assert "RuntimeError" in r.probe.error
        assert "DNS failure" in r.probe.error
        assert r.record is None

    def test_error_result_probe_method(self):
        c = _make_candidate("Src")
        r = _error_result(c, ValueError("oops"))
        assert r.probe.method_used == "HEAD"
        assert r.probe.latency_ms == 0.0


# ── probe_catalog: empty input ─────────────────────────────────────────────────

class TestProbeCatalogEmpty:
    def test_empty_candidates_returns_zeros(self):
        registry = _make_registry()
        result = probe_catalog([], registry)
        assert result.total == 0
        assert result.attempted == 0
        assert result.reachable == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.avg_latency_ms is None
        assert result.per_source == []

    def test_empty_does_not_call_probe_and_register(self):
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register") as mock_par:
            probe_catalog([], registry)
            mock_par.assert_not_called()


# ── probe_catalog: sequential ─────────────────────────────────────────────────

class TestProbeCatalogSequential:
    def _run(self, candidates, side_effects):
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=side_effects) as mock_par:
            result = probe_catalog(candidates, registry, concurrency=1)
        return result, mock_par

    def test_single_ok_probe(self):
        c = _make_candidate("FRED")
        rr = _make_registry_result(c.source_id, ok=True, latency_ms=90.0)
        result, mock_par = self._run([c], [rr])
        assert result.total == 1
        assert result.reachable == 1
        assert result.failed == 0
        mock_par.assert_called_once()

    def test_single_failed_probe(self):
        c = _make_candidate("Bad Source")
        rr = _make_registry_result(c.source_id, ok=False)
        result, _ = self._run([c], [rr])
        assert result.reachable == 0
        assert result.failed == 1

    def test_multiple_candidates_called_in_order(self):
        candidates = [_make_candidate(f"Src{i}") for i in range(3)]
        results = [_make_registry_result(c.source_id, ok=True) for c in candidates]
        result, mock_par = self._run(candidates, results)
        assert result.total == 3
        assert result.attempted == 3
        call_candidates = [call.kwargs["candidate"] for call in mock_par.call_args_list]
        assert call_candidates == candidates

    def test_scores_forwarded(self):
        c = _make_candidate("Scored Src")
        rr = _make_registry_result(c.source_id, ok=True)
        registry = _make_registry()
        scores = {c.source_id: 0.87}
        with patch("data.scout.probe_catalog.probe_and_register",
                   return_value=rr) as mock_par:
            probe_catalog([c], registry, scores=scores)
        call_kwargs = mock_par.call_args.kwargs
        assert call_kwargs["score"] == 0.87

    def test_timeout_forwarded(self):
        c = _make_candidate("T Src")
        rr = _make_registry_result(c.source_id, ok=True)
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   return_value=rr) as mock_par:
            probe_catalog([c], registry, timeout=5)
        assert mock_par.call_args.kwargs["timeout"] == 5

    def test_advance_to_sampled_forwarded(self):
        c = _make_candidate("T Src")
        rr = _make_registry_result(c.source_id, ok=True)
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   return_value=rr) as mock_par:
            probe_catalog([c], registry, advance_to_sampled=False)
        assert mock_par.call_args.kwargs["advance_to_sampled"] is False

    def test_exception_creates_error_result(self):
        c = _make_candidate("Exploding Source")
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=RuntimeError("boom")):
            result = probe_catalog([c], registry)
        assert result.total == 1
        assert result.attempted == 1
        assert result.failed == 1  # error result has ok=False
        assert result.per_source[0].action == "error"

    def test_exception_does_not_abort_remaining(self):
        candidates = [_make_candidate(f"Src{i}") for i in range(3)]
        rr_ok = _make_registry_result(candidates[2].source_id, ok=True)
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=[RuntimeError("boom"), RuntimeError("bang"), rr_ok]):
            result = probe_catalog(candidates, registry)
        assert result.attempted == 3
        assert result.reachable == 1

    def test_no_scores_arg_uses_empty_dict(self):
        c = _make_candidate("Src")
        rr = _make_registry_result(c.source_id, ok=True)
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   return_value=rr) as mock_par:
            probe_catalog([c], registry)  # no scores kwarg
        assert mock_par.call_args.kwargs["score"] is None


# ── probe_catalog: parallel ────────────────────────────────────────────────────

class TestProbeCatalogParallel:
    def _run_parallel(self, candidates, side_effects, concurrency=4):
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=side_effects):
            result = probe_catalog(candidates, registry, concurrency=concurrency)
        return result

    def test_parallel_same_counts_as_sequential(self):
        candidates = [_make_candidate(f"Src{i}") for i in range(4)]
        results = [
            _make_registry_result(candidates[0].source_id, ok=True, latency_ms=100.0),
            _make_registry_result(candidates[1].source_id, ok=True, latency_ms=200.0),
            _make_registry_result(candidates[2].source_id, ok=False),
            _make_registry_result(candidates[3].source_id, ok=True, latency_ms=150.0),
        ]
        r = self._run_parallel(candidates, results, concurrency=2)
        assert r.total == 4
        assert r.reachable == 3
        assert r.failed == 1

    def test_parallel_exception_isolated(self):
        candidates = [_make_candidate(f"Src{i}") for i in range(3)]
        rr0 = _make_registry_result(candidates[0].source_id, ok=True)
        rr2 = _make_registry_result(candidates[2].source_id, ok=True)
        registry = _make_registry()

        call_count = 0
        call_lock = threading.Lock()

        def side_effect(**kwargs):
            nonlocal call_count
            with call_lock:
                n = call_count
                call_count += 1
            if n == 1:
                raise RuntimeError("middle explodes")
            return [rr0, rr2][0 if n == 0 else 1]

        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=side_effect):
            result = probe_catalog(candidates, registry, concurrency=3)
        # All 3 candidates attempted; 1 became error result (ok=False)
        assert result.attempted == 3

    def test_concurrency_capped_at_max(self):
        """Concurrency above _MAX_SAFE_CONCURRENCY (8) is silently capped."""
        candidates = [_make_candidate(f"Src{i}") for i in range(2)]
        results = [_make_registry_result(c.source_id, ok=True) for c in candidates]
        # Should not raise even with absurdly high concurrency
        r = self._run_parallel(candidates, results, concurrency=999)
        assert r.total == 2

    def test_concurrency_zero_treated_as_one(self):
        """concurrency=0 is clamped to 1 (sequential)."""
        c = _make_candidate("Src")
        rr = _make_registry_result(c.source_id, ok=True)
        registry = _make_registry()
        with patch("data.scout.probe_catalog.probe_and_register", return_value=rr):
            result = probe_catalog([c], registry, concurrency=0)
        assert result.reachable == 1

    def test_parallel_result_order_matches_candidate_order(self):
        """Results should be in original candidate order, not completion order."""
        candidates = [_make_candidate(f"Src{i}") for i in range(4)]
        # Provide results in original order; ThreadPoolExecutor may complete out of order
        results_map = {c.source_id: _make_registry_result(c.source_id, ok=True)
                       for c in candidates}
        registry = _make_registry()

        def side_effect(**kwargs):
            return results_map[kwargs["candidate"].source_id]

        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=side_effect):
            result = probe_catalog(candidates, registry, concurrency=4)

        assert [r.source_id for r in result.per_source] == [c.source_id for c in candidates]


# ── Integration: CANDIDATE_CATALOG smoke test ──────────────────────────────────

class TestCandidateCatalogIntegration:
    def test_probe_full_catalog_with_mocks(self):
        """Smoke-test that probe_catalog handles the full CANDIDATE_CATALOG shape."""
        from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate

        candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
        registry = _make_registry()

        def fake_probe_and_register(**kwargs):
            c = kwargs["candidate"]
            return _make_registry_result(c.source_id, ok=True, latency_ms=50.0)

        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=fake_probe_and_register):
            result = probe_catalog(candidates, registry)

        assert result.total == len(candidates)
        assert result.attempted == len(candidates)
        assert result.reachable == len(candidates)
        assert result.failed == 0
        assert result.skipped == 0
        assert result.success_rate == 1.0
        assert result.avg_latency_ms is not None

    def test_summary_table_length_matches_candidates(self):
        from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate

        candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
        registry = _make_registry()

        def fake_probe_and_register(**kwargs):
            c = kwargs["candidate"]
            return _make_registry_result(c.source_id, ok=True, latency_ms=50.0)

        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=fake_probe_and_register):
            result = probe_catalog(candidates, registry)

        rows = result.summary_table()
        # header + separator + one row per candidate
        assert len(rows) == 2 + len(candidates)

    def test_to_dict_serializable(self):
        """to_dict() output must be JSON-serializable (no custom objects)."""
        import json
        from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate

        candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
        registry = _make_registry()

        def fake_probe_and_register(**kwargs):
            c = kwargs["candidate"]
            rr = _make_registry_result(c.source_id, ok=True, latency_ms=50.0)
            rr.to_dict = MagicMock(return_value={"source_id": c.source_id, "ok": True})
            return rr

        with patch("data.scout.probe_catalog.probe_and_register",
                   side_effect=fake_probe_and_register):
            result = probe_catalog(candidates, registry)

        # Should not raise
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["total"] == len(candidates)

    def test_repeated_run_does_not_raise(self):
        """Calling probe_catalog twice on same registry must not raise."""
        from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate

        candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG[:3]]
        registry = _make_registry()

        def fake(**kwargs):
            c = kwargs["candidate"]
            return _make_registry_result(c.source_id, ok=True)

        with patch("data.scout.probe_catalog.probe_and_register", side_effect=fake):
            r1 = probe_catalog(candidates, registry)

        with patch("data.scout.probe_catalog.probe_and_register", side_effect=fake):
            r2 = probe_catalog(candidates, registry)

        assert r1.total == r2.total
        assert r1.reachable == r2.reachable
