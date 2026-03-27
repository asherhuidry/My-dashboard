"""Tests for data.scout.sampler — source data sampling and validation.

All HTTP I/O is mocked so the suite is deterministic and offline.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from data.scout.sampler import (
    SampleQuality,
    SampleResult,
    _assess_quality,
    _parse_response,
    _sample_note,
    sample_and_validate,
    sample_sampled_sources,
    sample_source,
)
from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_record(
    source_id: str = "test_source",
    method: str = "api",
    url: str = "https://api.example.com/data",
    status: SourceStatus = SourceStatus.SAMPLED,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        name=f"Test {source_id}",
        category="macro",
        url=url,
        acquisition_method=method,
        status=status,
    )


def _json_body(rows: list[dict]) -> str:
    return json.dumps(rows)


def _nested_json_body(key: str, rows: list[dict]) -> str:
    return json.dumps({key: rows, "meta": {"count": len(rows)}})


def _csv_body(header: list[str], rows: list[list[str]]) -> str:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r))
    return "\n".join(lines)


def _mock_fetch(body: str = "", http_status: int = 200, content_type: str = "application/json",
                error: str = "", latency_ms: float = 50.0, raw_size: int = 0):
    """Return a mock _FetchResult."""
    from data.scout.sampler import _FetchResult
    return _FetchResult(
        body=body,
        http_status=http_status if not error else None,
        content_type=content_type,
        latency_ms=latency_ms,
        raw_size=raw_size or len(body.encode()),
        error=error,
    )


# ── SampleQuality tests ─────────────────────────────────────────────────────

class TestSampleQuality:
    def test_to_dict_keys(self):
        q = SampleQuality(row_count=5, column_count=3, passed=True)
        d = q.to_dict()
        for key in ("row_count", "column_count", "columns", "null_rate",
                     "has_timestamps", "has_numeric", "parse_format", "passed", "issues"):
            assert key in d

    def test_to_dict_null_rate_rounded(self):
        q = SampleQuality(row_count=5, column_count=3, null_rate=0.12345)
        assert q.to_dict()["null_rate"] == 0.1235

    def test_default_values(self):
        q = SampleQuality(row_count=0, column_count=0)
        assert q.passed is False
        assert q.issues == []
        assert q.parse_format == "unknown"


# ── SampleResult tests ───────────────────────────────────────────────────────

class TestSampleResult:
    def test_sampled_at_auto_populated(self):
        r = SampleResult(
            source_id="x", fetched=False, quality=None,
            advanced=False, action="skipped", reason="test",
        )
        assert r.sampled_at  # should be an ISO string

    def test_to_dict_keys(self):
        r = SampleResult(
            source_id="x", fetched=True,
            quality=SampleQuality(row_count=3, column_count=2, passed=True),
            advanced=False, action="validated", reason="ok",
            http_status=200, latency_ms=100.5, raw_size=512,
        )
        d = r.to_dict()
        for key in ("source_id", "fetched", "quality", "advanced", "action",
                     "reason", "http_status", "latency_ms", "raw_size", "sampled_at"):
            assert key in d
        assert d["quality"]["row_count"] == 3
        assert d["latency_ms"] == 100.5

    def test_to_dict_quality_none(self):
        r = SampleResult(
            source_id="x", fetched=False, quality=None,
            advanced=False, action="error", reason="fail",
        )
        assert r.to_dict()["quality"] is None


# ── _parse_response tests ────────────────────────────────────────────────────

class TestParseResponse:
    def test_json_array(self):
        body = _json_body([{"date": "2024-01-01", "value": 1.5}])
        rows, fmt = _parse_response(body, "application/json")
        assert fmt == "json_array"
        assert len(rows) == 1
        assert rows[0]["value"] == 1.5

    def test_nested_json(self):
        body = _nested_json_body("observations", [
            {"date": "2024-01-01", "value": "1.5"},
            {"date": "2024-01-02", "value": "1.6"},
        ])
        rows, fmt = _parse_response(body, "application/json")
        assert fmt == "json"
        assert len(rows) == 2

    def test_csv(self):
        body = _csv_body(["date", "value"], [
            ["2024-01-01", "1.5"],
            ["2024-01-02", "1.6"],
        ])
        rows, fmt = _parse_response(body, "text/csv")
        assert fmt == "csv"
        assert len(rows) == 2
        assert rows[0]["date"] == "2024-01-01"

    def test_ndjson(self):
        body = '{"date":"2024-01-01","val":1}\n{"date":"2024-01-02","val":2}\n'
        rows, fmt = _parse_response(body, "application/x-ndjson")
        assert fmt == "ndjson"
        assert len(rows) == 2

    def test_json_inferred_from_body(self):
        """JSON detected from body content even if Content-Type is generic."""
        body = _json_body([{"a": 1}])
        rows, fmt = _parse_response(body, "text/plain")
        assert fmt == "json_array"
        assert len(rows) == 1

    def test_unparseable(self):
        rows, fmt = _parse_response("<html>404</html>", "text/html")
        assert rows == []

    def test_empty_body(self):
        rows, fmt = _parse_response("", "application/json")
        # Should not crash
        assert rows == []

    def test_truncates_at_50_rows(self):
        big = [{"i": i} for i in range(100)]
        body = _json_body(big)
        rows, _ = _parse_response(body, "application/json")
        assert len(rows) == 50

    def test_csv_with_json_content_type_falls_through(self):
        """If JSON parse fails, CSV fallback works."""
        csv = _csv_body(["x", "y"], [["1", "2"]])
        rows, fmt = _parse_response(csv, "text/plain")
        assert fmt == "csv"
        assert len(rows) == 1


# ── _assess_quality tests ────────────────────────────────────────────────────

class TestAssessQuality:
    def test_empty_rows(self):
        q = _assess_quality([], "json")
        assert q.passed is False
        assert q.row_count == 0
        assert "No parseable rows" in q.issues[0]

    def test_single_row_fails_min(self):
        q = _assess_quality([{"date": "2024-01-01", "value": "1.0"}], "json")
        assert q.passed is False
        assert any("1 row" in i for i in q.issues)

    def test_valid_sample(self):
        rows = [
            {"date": "2024-01-01", "value": "1.5", "series": "GDP"},
            {"date": "2024-01-02", "value": "1.6", "series": "GDP"},
            {"date": "2024-01-03", "value": "1.7", "series": "GDP"},
        ]
        q = _assess_quality(rows, "json_array")
        assert q.passed is True
        assert q.row_count == 3
        assert q.column_count == 3
        assert q.has_timestamps is True
        assert q.has_numeric is True
        assert q.null_rate == 0.0

    def test_high_null_rate(self):
        rows = [
            {"date": "2024-01-01", "value": None, "x": ""},
            {"date": "2024-01-02", "value": None, "x": ""},
        ]
        q = _assess_quality(rows, "json")
        assert q.passed is False
        assert any("Null rate" in i for i in q.issues)

    def test_timestamp_detection(self):
        rows = [
            {"timestamp": "2024-01-01T00:00:00Z", "amount": 100},
            {"timestamp": "2024-01-02T00:00:00Z", "amount": 200},
        ]
        q = _assess_quality(rows, "json")
        assert q.has_timestamps is True

    def test_numeric_detection_from_values(self):
        """Numeric detected from actual values even without hint column names."""
        rows = [
            {"foo": 1.5, "bar": "text"},
            {"foo": 2.5, "bar": "text"},
        ]
        q = _assess_quality(rows, "json")
        assert q.has_numeric is True

    def test_numeric_detection_from_string_values(self):
        """Numeric detected from string values that are parseable as float."""
        rows = [
            {"foo": "1.5", "bar": "text"},
            {"foo": "2.5", "bar": "text"},
        ]
        q = _assess_quality(rows, "json")
        assert q.has_numeric is True

    def test_columns_union(self):
        """Columns are the union of all keys across rows."""
        rows = [
            {"a": 1, "b": 2},
            {"b": 3, "c": 4},
        ]
        q = _assess_quality(rows, "json")
        assert set(q.columns) == {"a", "b", "c"}
        assert q.column_count == 3


# ── sample_source tests ─────────────────────────────────────────────────────

class TestSampleSource:
    def test_sdk_skipped(self):
        record = _make_record(method="sdk")
        result = sample_source(record)
        assert result.action == "skipped"
        assert result.fetched is False
        assert result.quality is None

    def test_scrape_skipped(self):
        record = _make_record(method="scrape")
        result = sample_source(record)
        assert result.action == "skipped"

    @patch("data.scout.sampler._fetch_sample")
    def test_fetch_error(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch(error="Connection refused")
        record = _make_record()
        result = sample_source(record)
        assert result.action == "error"
        assert result.fetched is False
        assert "Connection refused" in result.reason

    @patch("data.scout.sampler._fetch_sample")
    def test_valid_json_sample(self, mock_fetch):
        body = _json_body([
            {"date": "2024-01-01", "value": "1.5"},
            {"date": "2024-01-02", "value": "1.6"},
            {"date": "2024-01-03", "value": "1.7"},
        ])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record()
        result = sample_source(record)
        assert result.action == "validated"
        assert result.fetched is True
        assert result.quality is not None
        assert result.quality.passed is True
        assert result.quality.row_count == 3

    @patch("data.scout.sampler._fetch_sample")
    def test_insufficient_rows(self, mock_fetch):
        body = _json_body([{"x": 1}])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record()
        result = sample_source(record)
        assert result.action == "failed"
        assert result.quality is not None
        assert result.quality.passed is False

    @patch("data.scout.sampler._fetch_sample")
    def test_csv_sample(self, mock_fetch):
        body = _csv_body(["date", "close", "volume"], [
            ["2024-01-01", "150.5", "1000000"],
            ["2024-01-02", "151.0", "1100000"],
            ["2024-01-03", "149.8", "900000"],
        ])
        mock_fetch.return_value = _mock_fetch(body=body, content_type="text/csv")
        record = _make_record()
        result = sample_source(record)
        assert result.action == "validated"
        assert result.quality.parse_format == "csv"

    @patch("data.scout.sampler._fetch_sample")
    def test_unparseable_html(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch(
            body="<html><body>Not Found</body></html>",
            content_type="text/html",
        )
        record = _make_record()
        result = sample_source(record)
        assert result.action == "failed"
        assert result.quality.row_count == 0

    @patch("data.scout.sampler._fetch_sample")
    def test_http_status_and_latency_propagated(self, mock_fetch):
        body = _json_body([{"a": 1}, {"a": 2}])
        mock_fetch.return_value = _mock_fetch(body=body, http_status=200, latency_ms=142.5)
        record = _make_record()
        result = sample_source(record)
        assert result.http_status == 200
        assert result.latency_ms == 142.5


# ── sample_and_validate tests ───────────────────────────────────────────────

class TestSampleAndValidate:
    def _registry_with(self, record: SourceRecord) -> MagicMock:
        reg = MagicMock(spec=SourceRegistry)
        reg.get.return_value = record
        return reg

    @patch("data.scout.sampler._fetch_sample")
    def test_advances_sampled_to_validated(self, mock_fetch):
        body = _json_body([{"date": "2024-01-01", "value": "1"}, {"date": "2024-01-02", "value": "2"}])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record(status=SourceStatus.SAMPLED)
        registry = self._registry_with(record)
        result = sample_and_validate("test_source", registry)
        assert result.advanced is True
        registry.update_status.assert_called_once_with(
            "test_source", SourceStatus.VALIDATED, notes="",
        )

    @patch("data.scout.sampler._fetch_sample")
    def test_does_not_advance_if_disabled(self, mock_fetch):
        body = _json_body([{"date": "2024-01-01", "value": "1"}, {"date": "2024-01-02", "value": "2"}])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record(status=SourceStatus.SAMPLED)
        registry = self._registry_with(record)
        result = sample_and_validate("test_source", registry, advance_to_validated=False)
        assert result.advanced is False
        registry.update_status.assert_not_called()

    @patch("data.scout.sampler._fetch_sample")
    def test_does_not_advance_if_not_sampled(self, mock_fetch):
        body = _json_body([{"date": "2024-01-01", "value": "1"}, {"date": "2024-01-02", "value": "2"}])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record(status=SourceStatus.DISCOVERED)
        registry = self._registry_with(record)
        result = sample_and_validate("test_source", registry)
        assert result.advanced is False

    @patch("data.scout.sampler._fetch_sample")
    def test_does_not_advance_on_failure(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch(error="timeout")
        record = _make_record(status=SourceStatus.SAMPLED)
        registry = self._registry_with(record)
        result = sample_and_validate("test_source", registry)
        assert result.advanced is False
        registry.update_status.assert_not_called()

    @patch("data.scout.sampler._fetch_sample")
    def test_notes_always_recorded(self, mock_fetch):
        mock_fetch.return_value = _mock_fetch(error="dns failed")
        record = _make_record(status=SourceStatus.SAMPLED)
        registry = self._registry_with(record)
        sample_and_validate("test_source", registry)
        registry.update_notes.assert_called_once()
        note_arg = registry.update_notes.call_args[0][1]
        assert "[sample" in note_arg

    @patch("data.scout.sampler._fetch_sample")
    def test_transition_error_handled(self, mock_fetch):
        body = _json_body([{"date": "2024-01-01", "value": "1"}, {"date": "2024-01-02", "value": "2"}])
        mock_fetch.return_value = _mock_fetch(body=body)
        record = _make_record(status=SourceStatus.SAMPLED)
        registry = self._registry_with(record)
        registry.update_status.side_effect = ValueError("Invalid transition")
        result = sample_and_validate("test_source", registry)
        # Should not raise, just not advance
        assert result.advanced is False


# ── sample_sampled_sources tests ─────────────────────────────────────────────

class TestSampleSampledSources:
    @patch("data.scout.sampler.sample_and_validate")
    def test_samples_all_sampled(self, mock_sav):
        registry = MagicMock(spec=SourceRegistry)
        r1 = _make_record("src_a", status=SourceStatus.SAMPLED)
        r2 = _make_record("src_b", status=SourceStatus.SAMPLED)
        registry.filter.return_value = [r1, r2]
        mock_sav.return_value = SampleResult(
            source_id="x", fetched=True, quality=None,
            advanced=False, action="validated", reason="ok",
        )
        results = sample_sampled_sources(registry)
        assert len(results) == 2
        assert mock_sav.call_count == 2

    @patch("data.scout.sampler.sample_and_validate")
    def test_empty_registry(self, mock_sav):
        registry = MagicMock(spec=SourceRegistry)
        registry.filter.return_value = []
        results = sample_sampled_sources(registry)
        assert results == []
        mock_sav.assert_not_called()

    @patch("data.scout.sampler.sample_and_validate")
    def test_exception_isolated(self, mock_sav):
        registry = MagicMock(spec=SourceRegistry)
        r1 = _make_record("src_a")
        r2 = _make_record("src_b")
        registry.filter.return_value = [r1, r2]
        mock_sav.side_effect = [RuntimeError("boom"), SampleResult(
            source_id="src_b", fetched=True, quality=None,
            advanced=False, action="validated", reason="ok",
        )]
        results = sample_sampled_sources(registry)
        assert len(results) == 2
        assert results[0].action == "error"
        assert results[1].action == "validated"


# ── _sample_note tests ───────────────────────────────────────────────────────

class TestSampleNote:
    def test_basic_note(self):
        r = SampleResult(
            source_id="x", fetched=True,
            quality=SampleQuality(
                row_count=5, column_count=3, null_rate=0.05,
                parse_format="json", passed=True,
            ),
            advanced=True, action="validated", reason="ok",
            http_status=200, latency_ms=100.0,
        )
        note = _sample_note(r)
        assert "[sample" in note
        assert "rows=5" in note
        assert "cols=3" in note
        assert "json" in note

    def test_error_note(self):
        r = SampleResult(
            source_id="x", fetched=False, quality=None,
            advanced=False, action="error", reason="fail",
        )
        note = _sample_note(r)
        assert "action=error" in note
