"""Probe-to-registry integration and convenience probe flow.

Connects ProbeResult outcomes to the source registry lifecycle:

- A reachable probe on a DISCOVERED source can advance it to SAMPLED.
- An unreachable or error probe records notes but does not change status.
- Repeated probes append findings without creating duplicates.

``probe_and_register`` is the main convenience function:
it probes a candidate, updates (or creates) its registry entry, and returns
a structured result.

Usage::

    from data.registry.source_registry import SourceRegistry
    from data.scout.schema import normalize_source_candidate
    from data.scout.probe_registry import probe_and_register

    registry  = SourceRegistry()
    candidate = normalize_source_candidate({
        "name": "FRED API", "url": "https://api.stlouisfed.org/fred",
    })
    result = probe_and_register(candidate, registry)
    print(result.probe.ok)           # True
    print(result.action)             # "sampled" | "updated" | "skipped"
    print(result.record.status)      # SourceStatus.SAMPLED | DISCOVERED
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from data.registry.source_registry import SourceRecord, SourceRegistry, SourceStatus
from data.scout.probe import ProbeResult, probe_source_url
from data.scout.registry_bridge import register_candidate_source

if TYPE_CHECKING:
    from data.scout.schema import SourceCandidate

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10


# ── ProbeRegistryResult ───────────────────────────────────────────────────────

@dataclass
class ProbeRegistryResult:
    """Outcome of a probe-and-register operation.

    Attributes:
        source_id: The source_id that was operated on.
        probe:     The raw ProbeResult.
        action:    What happened to the registry entry:
                   "sampled"   — DISCOVERED → SAMPLED (reachable probe)
                   "updated"   — Notes appended; status unchanged
                   "created"   — New DISCOVERED record was created
                   "skipped"   — Source is already beyond SAMPLED; no action
        reason:    Human-readable explanation of the action.
        record:    The SourceRecord after the operation (may be None if the
                   registry entry could not be retrieved).
    """
    source_id: str
    probe:     ProbeResult
    action:    str
    reason:    str
    record:    SourceRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "source_id": self.source_id,
            "probe":     self.probe.to_dict(),
            "action":    self.action,
            "reason":    self.reason,
        }


# ── apply_probe_to_registry ───────────────────────────────────────────────────

def apply_probe_to_registry(
    source_id:   str,
    probe:       ProbeResult,
    registry:    SourceRegistry,
    advance_to_sampled: bool = True,
) -> ProbeRegistryResult:
    """Apply a ProbeResult to an existing registry entry.

    Behaviour:
    - If the source is DISCOVERED and the probe is ok → status → SAMPLED
      (only when ``advance_to_sampled=True``).
    - If the probe is not ok → append error notes; status unchanged.
    - If the source is already SAMPLED, VALIDATED, APPROVED, or QUARANTINED
      → notes updated only; status not touched.
    - If the source is REJECTED → notes updated only.

    Args:
        source_id:          Registry source_id to update.
        probe:              ProbeResult from probe_source_url.
        registry:           Target SourceRegistry.
        advance_to_sampled: If True (default), a reachable probe on a
                            DISCOVERED source advances status to SAMPLED.

    Returns:
        A ProbeRegistryResult describing the outcome.

    Raises:
        KeyError: If source_id is not found in the registry.
    """
    record = registry.get(source_id)
    note   = _probe_note(probe)

    # Sources already beyond DISCOVERED — update notes only
    _beyond_discovered = {
        SourceStatus.SAMPLED, SourceStatus.VALIDATED,
        SourceStatus.APPROVED, SourceStatus.QUARANTINED,
    }
    if record.status in _beyond_discovered:
        registry.update_notes(source_id, note)
        return ProbeRegistryResult(
            source_id = source_id,
            probe     = probe,
            action    = "updated",
            reason    = (
                f"Source already at status={record.status.value}; "
                "probe notes appended, status unchanged."
            ),
            record    = registry.get(source_id),
        )

    # REJECTED — also notes only
    if record.status == SourceStatus.REJECTED:
        registry.update_notes(source_id, note)
        return ProbeRegistryResult(
            source_id = source_id,
            probe     = probe,
            action    = "updated",
            reason    = "Source is REJECTED; probe notes appended, status unchanged.",
            record    = registry.get(source_id),
        )

    # DISCOVERED
    if probe.ok and advance_to_sampled:
        registry.update_status(source_id, SourceStatus.SAMPLED, notes=note)
        return ProbeRegistryResult(
            source_id = source_id,
            probe     = probe,
            action    = "sampled",
            reason    = (
                f"Probe ok (HTTP {probe.http_status}); "
                "status advanced DISCOVERED → SAMPLED."
            ),
            record    = registry.get(source_id),
        )

    # Probe failed or advance_to_sampled=False
    registry.update_notes(source_id, note)
    reason = (
        "Probe failed or advance_to_sampled=False; notes recorded, status unchanged."
        if not probe.ok
        else "advance_to_sampled=False; notes recorded."
    )
    return ProbeRegistryResult(
        source_id = source_id,
        probe     = probe,
        action    = "updated",
        reason    = reason,
        record    = registry.get(source_id),
    )


# ── probe_and_register ────────────────────────────────────────────────────────

def probe_and_register(
    candidate:          "SourceCandidate",
    registry:           SourceRegistry,
    score:              float | None = None,
    timeout:            int  = _DEFAULT_TIMEOUT,
    advance_to_sampled: bool = True,
) -> ProbeRegistryResult:
    """Probe a candidate URL, then create/update its registry entry.

    Steps:
    1. Ensure the candidate has a DISCOVERED registry entry (create if absent).
    2. Run probe_source_url against the candidate.
    3. Apply the probe result via apply_probe_to_registry.

    A reachable probe on a DISCOVERED entry advances status to SAMPLED.
    An unreachable probe records notes and leaves status unchanged.
    If the source already exists beyond DISCOVERED, the probe notes are
    appended and status is not changed.

    Args:
        candidate:          The SourceCandidate to probe.
        registry:           The target SourceRegistry.
        score:              Optional pre-computed score (forwarded to
                            register_candidate_source if the record is new).
        timeout:            Probe timeout in seconds.
        advance_to_sampled: If True (default), a reachable probe advances a
                            DISCOVERED entry to SAMPLED.

    Returns:
        A ProbeRegistryResult with the probe and registry outcome.
    """
    # Step 1: ensure a registry entry exists
    reg_result = register_candidate_source(candidate, registry, score=score)
    source_id  = reg_result.source_id
    log.debug(
        "probe_and_register: registry %s for %s",
        reg_result.action, source_id,
    )

    # Step 2: probe
    probe = probe_source_url(candidate, timeout=timeout)
    log.info(
        "Probed %s: ok=%s status=%s latency=%.0fms",
        source_id, probe.ok, probe.http_status, probe.latency_ms,
    )

    # Step 3: apply probe to registry
    return apply_probe_to_registry(
        source_id           = source_id,
        probe               = probe,
        registry            = registry,
        advance_to_sampled  = advance_to_sampled,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _probe_note(probe: ProbeResult) -> str:
    """Format a probe result as a registry note string."""
    status_str = str(probe.http_status) if probe.http_status is not None else "n/a"
    parts = [
        f"[probe {probe.probed_at[:10]}]",
        f"method={probe.method_used}",
        f"status={status_str}",
        f"ok={probe.ok}",
        f"latency={probe.latency_ms:.0f}ms",
    ]
    if probe.redirected:
        parts.append(f"redirected_to={probe.final_url}")
    if probe.content_type:
        parts.append(f"content_type={probe.content_type}")
    if probe.error:
        parts.append(f"error={probe.error!r}")
    return " | ".join(parts)
