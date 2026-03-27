"""Source sampler — fetch a small data sample from a SAMPLED source and validate it.

Bridges the gap between "URL is reachable" (probe → SAMPLED) and "source
produces usable financial data" (SAMPLED → VALIDATED).  For each source, the
sampler:

1. Fetches a tiny payload (≤ 5 KB or ≤ 50 rows) via the source's URL.
2. Attempts to parse the response into tabular rows.
3. Runs lightweight quality checks (row count, null rate, column presence).
4. Produces a ``SampleResult`` with structured quality metrics.
5. Optionally advances the registry entry from SAMPLED → VALIDATED.

Nothing here writes to production tables or triggers full ingest.

Fetch strategies by acquisition_method:
- api / feed:      GET the URL, read ≤ 5 KB, parse JSON or CSV.
- file_download:   GET the URL, read ≤ 5 KB, parse CSV or JSON.
- sdk:             Not HTTP-fetchable → returns a skip result.
- scrape:          Not supported → returns a skip result.

Usage::

    from data.registry.source_registry import SourceRegistry
    from data.scout.sampler import sample_source, sample_and_validate

    registry = SourceRegistry()
    result   = sample_and_validate("ecb_statistical_data_warehouse", registry)
    print(result.quality)     # SampleQuality dataclass
    print(result.advanced)    # True if SAMPLED → VALIDATED
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_BYTES       = 5120        # 5 KB — enough for a sample, not a full dataset
_DEFAULT_TIMEOUT = 15          # seconds
_MIN_ROWS        = 2           # absolute minimum for a valid sample
_MAX_NULL_PCT    = 0.50        # lenient for a sample — tighten at full ingest
_USER_AGENT      = (
    "FinBrain-SourceSampler/1.0 "
    "(data quality sample; single request; no bulk collection)"
)

# Methods that can be HTTP-sampled
_FETCHABLE_METHODS = {"api", "feed", "file_download"}

# Methods that cannot be HTTP-sampled
_SKIP_METHODS = {"sdk", "scrape"}


# ── SampleQuality ───────────────────────────────────────────────────────────

@dataclass
class SampleQuality:
    """Quality metrics derived from a fetched data sample.

    Attributes:
        row_count:      Number of rows parsed.
        column_count:   Number of columns detected.
        columns:        List of column names (or field keys).
        null_rate:      Fraction of null/missing values across all cells (0–1).
        has_timestamps: Whether a plausible date/time column was found.
        has_numeric:    Whether at least one numeric column was found.
        parse_format:   What format the response was parsed as ("json", "csv",
                        "json_array", "ndjson", or "unknown").
        passed:         True if the sample meets minimum quality thresholds.
        issues:         List of human-readable quality issues found.
    """
    row_count:      int
    column_count:   int
    columns:        list[str]       = field(default_factory=list)
    null_rate:      float           = 0.0
    has_timestamps: bool            = False
    has_numeric:    bool            = False
    parse_format:   str             = "unknown"
    passed:         bool            = False
    issues:         list[str]       = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "row_count":      self.row_count,
            "column_count":   self.column_count,
            "columns":        self.columns,
            "null_rate":      round(self.null_rate, 4),
            "has_timestamps": self.has_timestamps,
            "has_numeric":    self.has_numeric,
            "parse_format":   self.parse_format,
            "passed":         self.passed,
            "issues":         self.issues,
        }


# ── SampleResult ────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Outcome of a sample-and-validate operation.

    Attributes:
        source_id:   The source that was sampled.
        fetched:     True if an HTTP fetch was attempted and returned data.
        quality:     Quality metrics (None if fetch failed or was skipped).
        advanced:    True if the registry was advanced SAMPLED → VALIDATED.
        action:      What happened: "validated", "failed", "skipped", "error".
        reason:      Human-readable explanation.
        http_status: HTTP status code from the fetch (None if skipped/error).
        latency_ms:  Fetch round-trip time in ms.
        raw_size:    Bytes read from the response.
        sampled_at:  ISO-8601 UTC timestamp.
    """
    source_id:   str
    fetched:     bool
    quality:     SampleQuality | None
    advanced:    bool
    action:      str
    reason:      str
    http_status: int | None     = None
    latency_ms:  float          = 0.0
    raw_size:    int             = 0
    sampled_at:  str             = ""

    def __post_init__(self) -> None:
        if not self.sampled_at:
            self.sampled_at = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "source_id":   self.source_id,
            "fetched":     self.fetched,
            "quality":     self.quality.to_dict() if self.quality else None,
            "advanced":    self.advanced,
            "action":      self.action,
            "reason":      self.reason,
            "http_status": self.http_status,
            "latency_ms":  round(self.latency_ms, 1),
            "raw_size":    self.raw_size,
            "sampled_at":  self.sampled_at,
        }


# ── sample_source ───────────────────────────────────────────────────────────

def sample_source(
    record:  SourceRecord,
    timeout: int = _DEFAULT_TIMEOUT,
) -> SampleResult:
    """Fetch a small data sample from a source and assess quality.

    Does NOT modify the registry.  Callers who want registry advancement
    should use ``sample_and_validate`` instead.

    Args:
        record:  A SourceRecord (should be at SAMPLED status, but not enforced).
        timeout: HTTP request timeout in seconds.

    Returns:
        A SampleResult with quality metrics.
    """
    if record.acquisition_method in _SKIP_METHODS:
        return SampleResult(
            source_id = record.source_id,
            fetched   = False,
            quality   = None,
            advanced  = False,
            action    = "skipped",
            reason    = (
                f"Acquisition method '{record.acquisition_method}' cannot be "
                "HTTP-sampled. Requires manual or SDK-based validation."
            ),
        )

    # Fetch
    fetch = _fetch_sample(record.url, timeout)
    if fetch.error:
        return SampleResult(
            source_id   = record.source_id,
            fetched     = False,
            quality     = None,
            advanced    = False,
            action      = "error",
            reason      = f"Fetch failed: {fetch.error}",
            http_status = fetch.http_status,
            latency_ms  = fetch.latency_ms,
        )

    # Parse
    rows, parse_format = _parse_response(fetch.body, fetch.content_type)

    # Assess quality
    quality = _assess_quality(rows, parse_format)

    action = "validated" if quality.passed else "failed"
    reason = (
        f"Sample OK: {quality.row_count} rows, {quality.column_count} cols, "
        f"null_rate={quality.null_rate:.1%}, format={quality.parse_format}."
        if quality.passed
        else f"Sample quality insufficient: {'; '.join(quality.issues)}"
    )

    return SampleResult(
        source_id   = record.source_id,
        fetched     = True,
        quality     = quality,
        advanced    = False,   # caller must advance registry
        action      = action,
        reason      = reason,
        http_status = fetch.http_status,
        latency_ms  = fetch.latency_ms,
        raw_size    = fetch.raw_size,
    )


# ── sample_and_validate ─────────────────────────────────────────────────────

def sample_and_validate(
    source_id:          str,
    registry:           SourceRegistry,
    timeout:            int  = _DEFAULT_TIMEOUT,
    advance_to_validated: bool = True,
) -> SampleResult:
    """Sample a source and optionally advance it to VALIDATED in the registry.

    Args:
        source_id:            Registry source_id to sample.
        registry:             The SourceRegistry instance.
        timeout:              HTTP timeout in seconds.
        advance_to_validated: If True and the sample passes, advance
                              SAMPLED → VALIDATED.

    Returns:
        A SampleResult.  ``result.advanced`` is True only if the registry
        status was actually changed.

    Raises:
        KeyError: If source_id is not in the registry.
    """
    record = registry.get(source_id)

    result = sample_source(record, timeout=timeout)

    # Record sample notes regardless of outcome
    note = _sample_note(result)
    registry.update_notes(source_id, note)

    # Advance if appropriate
    if (
        advance_to_validated
        and result.action == "validated"
        and record.status == SourceStatus.SAMPLED
    ):
        try:
            registry.update_status(source_id, SourceStatus.VALIDATED, notes="")
            result.advanced = True
            result.reason += " Status advanced SAMPLED → VALIDATED."
            log.info("Source %s advanced to VALIDATED via sampling.", source_id)
        except ValueError as exc:
            log.warning("Could not advance %s: %s", source_id, exc)

    return result


# ── Batch sampler ───────────────────────────────────────────────────────────

def sample_sampled_sources(
    registry:             SourceRegistry,
    timeout:              int  = _DEFAULT_TIMEOUT,
    advance_to_validated: bool = True,
) -> list[SampleResult]:
    """Sample all sources currently at SAMPLED status.

    Args:
        registry:             The SourceRegistry instance.
        timeout:              HTTP timeout per source.
        advance_to_validated: If True, advance passing sources to VALIDATED.

    Returns:
        List of SampleResults, one per SAMPLED source.
    """
    sampled = registry.filter(status=SourceStatus.SAMPLED)
    results: list[SampleResult] = []
    for record in sampled:
        try:
            r = sample_and_validate(
                record.source_id, registry,
                timeout=timeout,
                advance_to_validated=advance_to_validated,
            )
            results.append(r)
            log.info(
                "sample_sampled_sources [%d/%d] %s → %s",
                len(results), len(sampled), record.source_id, r.action,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Unexpected error sampling %s: %s", record.source_id, exc)
            results.append(SampleResult(
                source_id = record.source_id,
                fetched   = False,
                quality   = None,
                advanced  = False,
                action    = "error",
                reason    = f"Unexpected error: {exc}",
            ))
    return results


# ── Internal: fetch ─────────────────────────────────────────────────────────

@dataclass
class _FetchResult:
    body:         str
    http_status:  int | None
    content_type: str
    latency_ms:   float
    raw_size:     int
    error:        str

def _fetch_sample(url: str, timeout: int) -> _FetchResult:
    """GET a URL and return up to _MAX_BYTES of the response body."""
    start = time.monotonic()
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json, text/csv, */*"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency  = (time.monotonic() - start) * 1000
            raw      = resp.read(_MAX_BYTES)
            body     = raw.decode("utf-8", errors="replace")
            return _FetchResult(
                body         = body,
                http_status  = resp.status,
                content_type = resp.headers.get("Content-Type", ""),
                latency_ms   = latency,
                raw_size     = len(raw),
                error        = "",
            )
    except urllib.error.HTTPError as exc:
        latency = (time.monotonic() - start) * 1000
        return _FetchResult(
            body="", http_status=exc.code, content_type="",
            latency_ms=latency, raw_size=0,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except urllib.error.URLError as exc:
        latency = (time.monotonic() - start) * 1000
        return _FetchResult(
            body="", http_status=None, content_type="",
            latency_ms=latency, raw_size=0,
            error=f"URLError: {exc.reason}",
        )
    except Exception as exc:  # noqa: BLE001
        latency = (time.monotonic() - start) * 1000
        return _FetchResult(
            body="", http_status=None, content_type="",
            latency_ms=latency, raw_size=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Internal: parse ─────────────────────────────────────────────────────────

def _parse_response(
    body: str,
    content_type: str,
) -> tuple[list[dict[str, Any]], str]:
    """Try to parse the response body into a list of row dicts.

    Returns (rows, format_name).  On failure returns ([], "unknown").
    """
    ct = content_type.lower()

    # Try JSON first (most APIs)
    if "json" in ct or body.lstrip().startswith(("{", "[")):
        rows, fmt = _try_parse_json(body)
        if rows:
            return rows, fmt

    # Try CSV
    if "csv" in ct or "text" in ct or not body.lstrip().startswith(("{", "[")):
        rows = _try_parse_csv(body)
        if rows:
            return rows, "csv"

    return [], "unknown"


def _try_parse_json(body: str) -> tuple[list[dict[str, Any]], str]:
    """Attempt to extract row-dicts from a JSON response."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        # Try NDJSON (one JSON object per line)
        return _try_parse_ndjson(body)

    # Direct array of objects
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[:50], "json_array"

    # Nested: look for the first list-of-dicts value in the top-level object
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[:50], "json"

    return [], "unknown"


def _try_parse_ndjson(body: str) -> tuple[list[dict[str, Any]], str]:
    """Try parsing newline-delimited JSON."""
    rows: list[dict[str, Any]] = []
    for line in body.strip().splitlines()[:50]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except (json.JSONDecodeError, ValueError):
            break
    if rows:
        return rows, "ndjson"
    return [], "unknown"


def _try_parse_csv(body: str) -> list[dict[str, Any]]:
    """Try parsing the body as CSV with a header row."""
    try:
        reader = csv.DictReader(io.StringIO(body))
        rows = []
        for i, row in enumerate(reader):
            if i >= 50:
                break
            rows.append(dict(row))
        return rows
    except Exception:  # noqa: BLE001
        return []


# ── Internal: quality assessment ────────────────────────────────────────────

_TIME_HINTS = {
    "date", "time", "timestamp", "datetime", "created_at", "updated_at",
    "published_at", "period", "observation_date", "realtime_start",
}

_NUMERIC_HINTS = {
    "value", "close", "open", "high", "low", "volume", "price", "rate",
    "yield", "amount", "score", "count", "market_cap", "eps", "revenue",
    "gdp", "inflation", "yield_pct",
}


def _assess_quality(
    rows:         list[dict[str, Any]],
    parse_format: str,
) -> SampleQuality:
    """Run lightweight quality checks on parsed rows."""
    issues: list[str] = []

    if not rows:
        issues.append(f"No parseable rows (format={parse_format}).")
        return SampleQuality(
            row_count=0, column_count=0, parse_format=parse_format,
            passed=False, issues=issues,
        )

    # Columns: union of all keys across rows
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    row_count = len(rows)
    col_count = len(all_keys)
    lower_keys = {k.lower() for k in all_keys}

    if row_count < _MIN_ROWS:
        issues.append(f"Only {row_count} row(s); need at least {_MIN_ROWS}.")

    if col_count < 1:
        issues.append("No columns detected.")

    # Null rate: count None/empty across all cells
    total_cells = row_count * col_count if col_count > 0 else 1
    null_count = sum(
        1 for row in rows for k in all_keys
        if row.get(k) is None or str(row.get(k, "")).strip() == ""
    )
    null_rate = null_count / total_cells

    if null_rate > _MAX_NULL_PCT:
        issues.append(f"Null rate {null_rate:.0%} exceeds {_MAX_NULL_PCT:.0%} threshold.")

    # Timestamp detection
    has_timestamps = bool(lower_keys & _TIME_HINTS)

    # Numeric detection
    has_numeric = bool(lower_keys & _NUMERIC_HINTS)
    if not has_numeric:
        # Also try checking actual values
        for row in rows[:5]:
            for v in row.values():
                if isinstance(v, (int, float)):
                    has_numeric = True
                    break
                if isinstance(v, str):
                    try:
                        float(v)
                        has_numeric = True
                        break
                    except (ValueError, TypeError):
                        pass
            if has_numeric:
                break

    passed = len(issues) == 0

    return SampleQuality(
        row_count      = row_count,
        column_count   = col_count,
        columns        = all_keys,
        null_rate      = null_rate,
        has_timestamps = has_timestamps,
        has_numeric    = has_numeric,
        parse_format   = parse_format,
        passed         = passed,
        issues         = issues,
    )


# ── Internal: registry note ─────────────────────────────────────────────────

def _sample_note(result: SampleResult) -> str:
    """Format a SampleResult as a registry note string."""
    parts = [
        f"[sample {result.sampled_at[:10]}]",
        f"action={result.action}",
    ]
    if result.http_status is not None:
        parts.append(f"http={result.http_status}")
    if result.quality:
        q = result.quality
        parts.append(f"rows={q.row_count}")
        parts.append(f"cols={q.column_count}")
        parts.append(f"null_rate={q.null_rate:.1%}")
        parts.append(f"format={q.parse_format}")
        if q.issues:
            parts.append(f"issues={q.issues}")
    if result.latency_ms > 0:
        parts.append(f"latency={result.latency_ms:.0f}ms")
    return " | ".join(parts)
