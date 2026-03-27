"""End-to-end source scouting pipeline: probe → sample → validate.

Orchestrates the full DISCOVERED → SAMPLED → VALIDATED lifecycle in a
single call, composing existing modules without modifying them:

1. ``probe_catalog`` probes candidates and advances reachable ones to SAMPLED.
2. ``sample_and_validate`` fetches a tiny data sample from each newly-SAMPLED
   source and advances those that pass quality checks to VALIDATED.

Nothing is auto-approved beyond VALIDATED.  Each step is opt-in via flags.

Usage::

    from data.registry.source_registry import SourceRegistry
    from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate
    from data.scout.scorer import score_source_candidate
    from data.scout.pipeline import run_scout_pipeline

    registry   = SourceRegistry()
    candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
    scores     = {c.source_id: score_source_candidate(c) for c in candidates}

    result = run_scout_pipeline(candidates, registry, scores=scores)
    print(result.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from data.registry.source_registry import SourceRegistry
from data.scout.probe_catalog import ProbeCatalogResult, probe_catalog
from data.scout.sampler import SampleResult, sample_and_validate
from data.scout.schema import SourceCandidate

log = logging.getLogger(__name__)

_DEFAULT_PROBE_TIMEOUT  = 10
_DEFAULT_SAMPLE_TIMEOUT = 15


# ── ScoutRunResult ──────────────────────────────────────────────────────────

@dataclass
class ScoutRunResult:
    """Combined outcome of a probe-then-sample pipeline run.

    Attributes:
        probe_result:     The ProbeCatalogResult from the probe phase.
        sample_results:   List of SampleResults from the sample phase
                          (empty if sampling was disabled or no sources
                          reached SAMPLED).
        newly_sampled:    source_ids that were advanced to SAMPLED by probing.
        newly_validated:  source_ids that were advanced to VALIDATED by sampling.
        sample_failures:  source_ids where sampling was attempted but failed.
        sample_skipped:   source_ids skipped (e.g. SDK sources).
    """
    probe_result:    ProbeCatalogResult
    sample_results:  list[SampleResult]    = field(default_factory=list)
    newly_sampled:   list[str]             = field(default_factory=list)
    newly_validated: list[str]             = field(default_factory=list)
    sample_failures: list[str]             = field(default_factory=list)
    sample_skipped:  list[str]             = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the full pipeline run."""
        lines = [
            "ScoutRunResult",
            "── Probe phase ──",
            f"  candidates probed : {self.probe_result.total}",
            f"  reachable         : {self.probe_result.reachable}",
            f"  failed            : {self.probe_result.failed}",
            f"  newly SAMPLED     : {len(self.newly_sampled)}",
            "── Sample phase ──",
            f"  sample attempts   : {len(self.sample_results)}",
            f"  newly VALIDATED   : {len(self.newly_validated)}",
            f"  sample failures   : {len(self.sample_failures)}",
            f"  sample skipped    : {len(self.sample_skipped)}",
        ]
        if self.newly_validated:
            lines.append(f"  validated sources : {', '.join(self.newly_validated)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "probe":            self.probe_result.to_dict(),
            "sample_results":   [r.to_dict() for r in self.sample_results],
            "newly_sampled":    self.newly_sampled,
            "newly_validated":  self.newly_validated,
            "sample_failures":  self.sample_failures,
            "sample_skipped":   self.sample_skipped,
        }


# ── run_scout_pipeline ──────────────────────────────────────────────────────

def run_scout_pipeline(
    candidates:           list[SourceCandidate],
    registry:             SourceRegistry,
    scores:               dict[str, float] | None = None,
    probe_timeout:        int  = _DEFAULT_PROBE_TIMEOUT,
    sample_timeout:       int  = _DEFAULT_SAMPLE_TIMEOUT,
    advance_to_sampled:   bool = True,
    run_sampling:         bool = True,
    advance_to_validated: bool = True,
    concurrency:          int  = 1,
) -> ScoutRunResult:
    """Run the full probe → sample pipeline on a list of candidates.

    Phase 1 — Probe:
        Calls ``probe_catalog`` to check reachability and advance
        DISCOVERED sources to SAMPLED.

    Phase 2 — Sample (opt-in via ``run_sampling=True``):
        For each source that was newly advanced to SAMPLED, calls
        ``sample_and_validate`` to fetch a tiny data sample and
        optionally advance to VALIDATED.

    Args:
        candidates:           List of normalized SourceCandidates.
        registry:             The SourceRegistry to update.
        scores:               Optional dict mapping source_id → score.
        probe_timeout:        Per-probe HTTP timeout (seconds).
        sample_timeout:       Per-sample HTTP timeout (seconds).
        advance_to_sampled:   If True (default), advance reachable probes
                              to SAMPLED.
        run_sampling:         If True (default), run the sample phase on
                              newly SAMPLED sources.
        advance_to_validated: If True (default), advance passing samples
                              to VALIDATED.
        concurrency:          Thread count for probe phase (1 = sequential).

    Returns:
        A ScoutRunResult with outcomes from both phases.
    """
    # Phase 1: Probe
    log.info("Scout pipeline: probing %d candidates…", len(candidates))
    probe_result = probe_catalog(
        candidates          = candidates,
        registry            = registry,
        scores              = scores,
        timeout             = probe_timeout,
        advance_to_sampled  = advance_to_sampled,
        concurrency         = concurrency,
    )

    # Identify sources that were just advanced to SAMPLED
    newly_sampled = [
        r.source_id
        for r in probe_result.per_source
        if r.action == "sampled"
    ]
    log.info(
        "Scout pipeline: probe complete. %d reachable, %d newly SAMPLED.",
        probe_result.reachable, len(newly_sampled),
    )

    if not run_sampling or not newly_sampled:
        return ScoutRunResult(
            probe_result   = probe_result,
            newly_sampled  = newly_sampled,
        )

    # Phase 2: Sample
    log.info("Scout pipeline: sampling %d newly-SAMPLED sources…", len(newly_sampled))
    sample_results:  list[SampleResult] = []
    newly_validated: list[str] = []
    sample_failures: list[str] = []
    sample_skipped:  list[str] = []

    for source_id in newly_sampled:
        try:
            sr = sample_and_validate(
                source_id            = source_id,
                registry             = registry,
                timeout              = sample_timeout,
                advance_to_validated = advance_to_validated,
            )
            sample_results.append(sr)

            if sr.advanced:
                newly_validated.append(source_id)
            elif sr.action == "skipped":
                sample_skipped.append(source_id)
            elif sr.action in ("failed", "error"):
                sample_failures.append(source_id)

            log.info(
                "Scout pipeline: sampled %s → %s (advanced=%s)",
                source_id, sr.action, sr.advanced,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Scout pipeline: error sampling %s: %s", source_id, exc)
            sample_failures.append(source_id)
            sample_results.append(SampleResult(
                source_id = source_id,
                fetched   = False,
                quality   = None,
                advanced  = False,
                action    = "error",
                reason    = f"Pipeline error: {exc}",
            ))

    log.info(
        "Scout pipeline: sample complete. %d validated, %d failed, %d skipped.",
        len(newly_validated), len(sample_failures), len(sample_skipped),
    )

    return ScoutRunResult(
        probe_result    = probe_result,
        sample_results  = sample_results,
        newly_sampled   = newly_sampled,
        newly_validated = newly_validated,
        sample_failures = sample_failures,
        sample_skipped  = sample_skipped,
    )
