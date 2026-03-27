"""Batch probe runner for source candidate catalogs.

Runs the existing lightweight HTTP probe across a list of SourceCandidates,
updates the source registry through the existing lifecycle bridge, and
returns a structured ``ProbeCatalogResult``.

Default behavior is sequential and safe.  An optional ``concurrency``
parameter enables simple thread-pool execution for catalogs where latency
dominates (e.g. probing 8 remote APIs sequentially would take ~80 s at a
10 s timeout each; parallel reduces that to ~10 s).  With concurrency > 1
each candidate is probed in its own thread; registry writes are serialised
via a lock so the JSON file is never corrupted.

Nothing here auto-approves sources.  A successful probe advances a
DISCOVERED entry to SAMPLED only.

Usage::

    from data.registry.source_registry import SourceRegistry
    from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate
    from data.scout.probe_catalog import probe_catalog

    registry   = SourceRegistry()
    candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
    result     = probe_catalog(candidates, registry)

    print(result.summary())
    for row in result.summary_table():
        print(row)
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from data.registry.source_registry import SourceRegistry
from data.scout.probe_registry import ProbeRegistryResult, probe_and_register
from data.scout.schema import SourceCandidate

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT     = 10
_MAX_SAFE_CONCURRENCY = 8   # hard cap — prevents accidental DoS


# ── ProbeCatalogResult ────────────────────────────────────────────────────────

@dataclass
class ProbeCatalogResult:
    """Summary of a batch probe run over a candidate catalog.

    Attributes:
        total:              Number of candidates submitted.
        attempted:          Number for which a probe was run (excludes SDK
                            candidates and any that raised unexpected errors).
        reachable:          Number with probe.ok == True.
        failed:             Number with probe.ok == False.
        skipped:            Number skipped due to unexpected errors.
        avg_latency_ms:     Mean latency across reachable HTTP probes (ms).
                            None if no reachable HTTP probes were collected.
        per_source:         List of ProbeRegistryResult, one per candidate.
    """
    total:           int
    attempted:       int
    reachable:       int
    failed:          int
    skipped:         int
    avg_latency_ms:  float | None
    per_source:      list[ProbeRegistryResult] = field(default_factory=list)

    # ── Computed helpers ───────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        """Fraction of attempted probes that were reachable (0–1)."""
        if self.attempted == 0:
            return 0.0
        return round(self.reachable / self.attempted, 4)

    def get(self, source_id: str) -> ProbeRegistryResult | None:
        """Return the ProbeRegistryResult for a specific source_id, or None."""
        for r in self.per_source:
            if r.source_id == source_id:
                return r
        return None

    # ── Text output ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        avg = f"{self.avg_latency_ms:.0f}ms" if self.avg_latency_ms is not None else "n/a"
        lines = [
            "ProbeCatalogResult",
            f"  total candidates  : {self.total}",
            f"  attempted         : {self.attempted}",
            f"  reachable         : {self.reachable}",
            f"  failed            : {self.failed}",
            f"  skipped           : {self.skipped}",
            f"  success rate      : {self.success_rate:.1%}",
            f"  avg latency (ok)  : {avg}",
        ]
        return "\n".join(lines)

    def summary_table(self) -> list[str]:
        """Return a list of one-line strings, one per source, suitable for printing.

        Each line contains: source_id, ok, registry status, latency, HTTP
        status, and acquisition method.
        """
        header = (
            f"{'source_id':<45} {'ok':<5} {'status':<12} "
            f"{'lat_ms':>7} {'http':>5} {'method':<14} category"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for r in sorted(self.per_source, key=lambda x: x.source_id):
            probe   = r.probe
            status  = r.record.status.value if r.record else "n/a"
            lat     = f"{probe.latency_ms:.0f}" if probe.latency_ms else "-"
            http    = str(probe.http_status) if probe.http_status is not None else "-"
            ok_str  = "yes" if probe.ok else "NO"
            cat     = r.record.category if r.record else "-"
            method  = probe.method_used
            rows.append(
                f"{r.source_id:<45} {ok_str:<5} {status:<12} "
                f"{lat:>7} {http:>5} {method:<14} {cat}"
            )
        return rows

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "total":          self.total,
            "attempted":      self.attempted,
            "reachable":      self.reachable,
            "failed":         self.failed,
            "skipped":        self.skipped,
            "success_rate":   self.success_rate,
            "avg_latency_ms": round(self.avg_latency_ms, 1)
                              if self.avg_latency_ms is not None else None,
            "per_source":     [r.to_dict() for r in self.per_source],
        }


# ── probe_catalog ─────────────────────────────────────────────────────────────

def probe_catalog(
    candidates:         list[SourceCandidate],
    registry:           SourceRegistry,
    scores:             dict[str, float] | None = None,
    timeout:            int  = _DEFAULT_TIMEOUT,
    advance_to_sampled: bool = True,
    concurrency:        int  = 1,
) -> ProbeCatalogResult:
    """Probe a list of source candidates and update the registry.

    Args:
        candidates:         List of normalized SourceCandidates to probe.
        registry:           The source registry to update.
        scores:             Optional dict mapping source_id → pre-computed score.
                            Forwarded to probe_and_register for each candidate.
        timeout:            Per-probe HTTP timeout in seconds (default: 10).
        advance_to_sampled: If True (default), a reachable probe on a
                            DISCOVERED entry advances it to SAMPLED.
        concurrency:        Number of threads to use.  1 (default) = sequential.
                            Capped at ``_MAX_SAFE_CONCURRENCY`` (8).

    Returns:
        A ``ProbeCatalogResult`` with per-source outcomes and aggregate stats.
    """
    if not candidates:
        return ProbeCatalogResult(
            total=0, attempted=0, reachable=0,
            failed=0, skipped=0, avg_latency_ms=None,
        )

    scores       = scores or {}
    concurrency  = max(1, min(concurrency, _MAX_SAFE_CONCURRENCY))
    results: list[ProbeRegistryResult] = []

    if concurrency == 1:
        results = _probe_sequential(
            candidates, registry, scores, timeout, advance_to_sampled
        )
    else:
        results = _probe_parallel(
            candidates, registry, scores, timeout, advance_to_sampled, concurrency
        )

    return _build_catalog_result(candidates, results)


# ── Sequential probe ──────────────────────────────────────────────────────────

def _probe_sequential(
    candidates:         list[SourceCandidate],
    registry:           SourceRegistry,
    scores:             dict[str, float],
    timeout:            int,
    advance_to_sampled: bool,
) -> list[ProbeRegistryResult]:
    results: list[ProbeRegistryResult] = []
    for c in candidates:
        try:
            r = probe_and_register(
                candidate          = c,
                registry           = registry,
                score              = scores.get(c.source_id),
                timeout            = timeout,
                advance_to_sampled = advance_to_sampled,
            )
            results.append(r)
            log.info(
                "probe_catalog [%s/%s] %s → ok=%s action=%s",
                len(results), len(candidates),
                c.source_id, r.probe.ok, r.action,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("probe_catalog: unexpected error for %s: %s", c.source_id, exc)
            results.append(_error_result(c, exc))
    return results


# ── Parallel probe ────────────────────────────────────────────────────────────

def _probe_parallel(
    candidates:         list[SourceCandidate],
    registry:           SourceRegistry,
    scores:             dict[str, float],
    timeout:            int,
    advance_to_sampled: bool,
    concurrency:        int,
) -> list[ProbeRegistryResult]:
    """Run probes in a thread pool; serialise all registry writes via a lock."""
    registry_lock  = threading.Lock()
    ordered: dict[str, ProbeRegistryResult] = {}

    def _probe_one(c: SourceCandidate) -> ProbeRegistryResult:
        try:
            with registry_lock:
                r = probe_and_register(
                    candidate          = c,
                    registry           = registry,
                    score              = scores.get(c.source_id),
                    timeout            = timeout,
                    advance_to_sampled = advance_to_sampled,
                )
            return r
        except Exception as exc:  # noqa: BLE001
            log.warning("probe_catalog: unexpected error for %s: %s", c.source_id, exc)
            return _error_result(c, exc)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_probe_one, c): c for c in candidates}
        for future in as_completed(futures):
            c = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = _error_result(c, exc)
            ordered[c.source_id] = result

    # Return in original candidate order
    return [ordered[c.source_id] for c in candidates if c.source_id in ordered]


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def _build_catalog_result(
    candidates: list[SourceCandidate],
    results:    list[ProbeRegistryResult],
) -> ProbeCatalogResult:
    reachable   = sum(1 for r in results if r.probe.ok)
    attempted   = len(results)
    failed      = sum(1 for r in results if not r.probe.ok)
    skipped     = len(candidates) - attempted

    # Average latency over reachable HTTP probes (exclude SDK synthetic probes)
    latencies = [
        r.probe.latency_ms
        for r in results
        if r.probe.ok and r.probe.method_used != "NONE" and r.probe.latency_ms > 0
    ]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    return ProbeCatalogResult(
        total          = len(candidates),
        attempted      = attempted,
        reachable      = reachable,
        failed         = failed,
        skipped        = skipped,
        avg_latency_ms = avg_latency,
        per_source     = results,
    )


def _error_result(c: SourceCandidate, exc: Exception) -> ProbeRegistryResult:
    """Create a synthetic ProbeRegistryResult for an unexpected exception."""
    from data.scout.probe import ProbeResult
    probe = ProbeResult(
        url         = c.url,
        ok          = False,
        http_status = None,
        method_used = "HEAD",
        latency_ms  = 0.0,
        error       = f"{type(exc).__name__}: {exc}",
        notes       = "Unexpected error in probe_catalog; see error field.",
    )
    return ProbeRegistryResult(
        source_id = c.source_id,
        probe     = probe,
        action    = "error",
        reason    = f"Unexpected exception: {exc}",
        record    = None,
    )
