"""Optional evidence hook: turn a ProbeResult into an EvidenceItem.

Produces a single SOURCE_REGISTRY EvidenceItem recording the probe outcome.
The item carries the HTTP status, latency, reachability flag, and any redirect
information as structured data.

This is intentionally narrow: one EvidenceItem per probe, no Claim generation
here.  Callers who want a Claim should use ``source_claim_from_candidate``
from ``data.scout.evidence_hooks`` and attach this evidence to it.

Usage::

    from data.scout.probe_evidence import evidence_from_probe
    from ml.evidence.store import ClaimStore

    ev = evidence_from_probe(candidate, probe_result)
    store = ClaimStore()
    store.add_evidence(ev)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ml.evidence.schema import EvidenceItem, EvidenceSourceType

if TYPE_CHECKING:
    from data.scout.probe import ProbeResult
    from data.scout.schema import SourceCandidate


def evidence_from_probe(
    candidate: "SourceCandidate",
    probe:     "ProbeResult",
) -> EvidenceItem:
    """Create an EvidenceItem from a probe result.

    Args:
        candidate: The source candidate that was probed.
        probe:     The ProbeResult from probe_source_url.

    Returns:
        A new EvidenceItem with source_type=SOURCE_REGISTRY.
    """
    reachable_str = "reachable" if probe.ok else "unreachable"
    status_str    = str(probe.http_status) if probe.http_status is not None else "n/a"

    summary = (
        f"HTTP probe of '{candidate.name}' ({candidate.url}): "
        f"{reachable_str}. "
        f"Status {status_str} via {probe.method_used}. "
        f"Latency {probe.latency_ms:.0f}ms."
    )
    if probe.redirected:
        summary += f" Redirected to {probe.final_url}."
    if probe.error:
        summary += f" Error: {probe.error}"

    structured: dict = {
        "source_id":    candidate.source_id,
        "url":          probe.url,
        "ok":           probe.ok,
        "http_status":  probe.http_status,
        "method_used":  probe.method_used,
        "latency_ms":   round(probe.latency_ms, 1),
        "content_type": probe.content_type,
        "final_url":    probe.final_url,
        "redirected":   probe.redirected,
        "probed_at":    probe.probed_at,
    }
    if probe.error:
        structured["error"] = probe.error

    return EvidenceItem.new(
        source_type     = EvidenceSourceType.SOURCE_REGISTRY,
        source_ref      = f"probe:{candidate.source_id}:{probe.probed_at[:10]}",
        summary         = summary,
        structured_data = structured,
    )
