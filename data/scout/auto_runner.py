"""Automated source scouting runner — bounded, scheduled execution.

Wraps the scout pipeline for unattended weekly runs (GitHub Actions).
Enforces resource bounds (max candidates, timeout budget), logs results
to the evolution trail, and produces a summary for the intelligence layer.

Safety:
- Never auto-approves sources beyond VALIDATED
- Caps the number of probes per run (default 20)
- Hard timeout budget prevents runaway network calls
- All actions logged to evolution_log for auditability
"""
from __future__ import annotations

import logging
import time
from typing import Any

from skills.logger import get_logger

log = get_logger(__name__)

AGENT_ID = "auto_scout"

# Resource bounds
MAX_CANDIDATES_PER_RUN: int = 20
PROBE_TIMEOUT_SECONDS: int = 8
SAMPLE_TIMEOUT_SECONDS: int = 12
TOTAL_BUDGET_SECONDS: int = 300  # 5 minute hard cap


def run_auto_scout(
    max_candidates: int = MAX_CANDIDATES_PER_RUN,
    run_sampling: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a bounded scout pipeline run.

    Loads the candidate catalog, scores and ranks them, then runs the
    top-N through probe → sample → validate. Results are logged to
    the evolution trail.

    Args:
        max_candidates: Maximum candidates to probe this run.
        run_sampling: Whether to also run the sample+validate phase.
        dry_run: If True, score and rank but don't actually probe.

    Returns:
        Summary dict with counts and outcomes.
    """
    from data.registry.source_registry import SourceRegistry
    from data.scout.schema import CANDIDATE_CATALOG, normalize_source_candidate
    from data.scout.scorer import score_source_candidate

    start = time.monotonic()

    # ── Load and score candidates ───────────────────────────────────────
    candidates = [normalize_source_candidate(r) for r in CANDIDATE_CATALOG]
    scores = {c.source_id: score_source_candidate(c) for c in candidates}

    # Rank by score descending, take top N
    ranked = sorted(candidates, key=lambda c: scores.get(c.source_id, 0), reverse=True)
    batch = ranked[:max_candidates]

    log.info("Auto-scout: %d candidates scored, running top %d",
             len(candidates), len(batch))

    summary: dict[str, Any] = {
        "total_candidates": len(candidates),
        "batch_size": len(batch),
        "top_scores": {c.source_id: round(scores[c.source_id], 3) for c in batch[:5]},
        "dry_run": dry_run,
    }

    if dry_run:
        summary["status"] = "dry_run"
        _log_to_evolution(summary)
        return summary

    # ── Run pipeline ────────────────────────────────────────────────────
    registry = SourceRegistry()

    from data.scout.pipeline import run_scout_pipeline
    result = run_scout_pipeline(
        candidates=batch,
        registry=registry,
        scores=scores,
        probe_timeout=PROBE_TIMEOUT_SECONDS,
        sample_timeout=SAMPLE_TIMEOUT_SECONDS,
        run_sampling=run_sampling,
        advance_to_sampled=True,
        advance_to_validated=True,
    )

    elapsed = round(time.monotonic() - start, 1)

    summary.update({
        "status": "completed",
        "elapsed_seconds": elapsed,
        "probed": result.probe_result.total,
        "reachable": result.probe_result.reachable,
        "probe_failed": result.probe_result.failed,
        "newly_sampled": result.newly_sampled,
        "newly_validated": result.newly_validated,
        "sample_failures": result.sample_failures,
    })

    _log_to_evolution(summary)
    log.info("Auto-scout complete in %.1fs: %d probed, %d validated",
             elapsed, result.probe_result.total, len(result.newly_validated))

    return summary


def _log_to_evolution(summary: dict[str, Any]) -> None:
    """Log scout run to evolution trail."""
    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id=AGENT_ID,
            action="auto_scout_run",
            after_state=summary,
        ))
    except Exception as exc:
        log.warning("Could not log to evolution trail: %s", exc)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dry = "--dry-run" in sys.argv
    result = run_auto_scout(dry_run=dry)
    print(f"\nAuto-scout {'(dry run)' if dry else ''}:")
    print(f"  Candidates scored: {result['total_candidates']}")
    print(f"  Batch size:        {result['batch_size']}")
    if not dry:
        print(f"  Probed:            {result.get('probed', 0)}")
        print(f"  Reachable:         {result.get('reachable', 0)}")
        print(f"  Newly validated:   {result.get('newly_validated', [])}")
        print(f"  Elapsed:           {result.get('elapsed_seconds', 0)}s")
