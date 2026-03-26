"""Noise filter agent for FinBrain.

Scans TimescaleDB price rows for data quality issues and quarantines
bad records. Never deletes source data — quarantined rows are flagged
in the Supabase quarantine table for human review.

Three detection passes per asset:
  1. Z-score outlier detection — close price z-score > threshold
  2. Volume spike filter       — volume z-score > threshold
  3. Cross-source consistency  — |yfinance close - alpha_vantage close| > 0.5%

Every run is logged to agent_runs and evolution_log.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from db.supabase.client import (
    AgentRun,
    EvolutionLogEntry,
    QuarantineRecord,
    end_agent_run,
    log_evolution,
    quarantine_record,
    start_agent_run,
)
from skills.logger import get_logger

logger = get_logger(__name__)

AGENT_ID = "noise_filter"

# Tunable thresholds
Z_SCORE_THRESHOLD = 4.0          # flag close price z-scores above this
VOLUME_Z_SCORE_THRESHOLD = 6.0   # flag volume spikes above this
CROSS_SOURCE_DIFF_PCT = 0.005    # 0.5% max allowed diff between sources


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceRecord:
    """A single price record as loaded from TimescaleDB for noise analysis.

    Attributes:
        time: The bar timestamp.
        asset: Ticker symbol.
        asset_class: Asset class string.
        close: Closing price.
        volume: Trading volume.
        source: Data source label.
        raw: The original row dict for quarantine payload.
    """
    time: datetime
    asset: str
    asset_class: str
    close: float
    volume: float
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NoiseFilterResult:
    """Summary of one noise filter run for one asset.

    Attributes:
        asset: Ticker symbol processed.
        total_rows: Total rows examined.
        z_score_flags: Rows flagged by z-score outlier detection.
        volume_flags: Rows flagged by volume spike detection.
        cross_source_flags: Rows flagged by cross-source consistency check.
        quarantined: Total rows quarantined.
    """
    asset: str
    total_rows: int
    z_score_flags: int = 0
    volume_flags: int = 0
    cross_source_flags: int = 0
    quarantined: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _z_scores(values: list[float]) -> list[float]:
    """Compute z-scores for a list of values.

    Returns a list of zeros if the standard deviation is zero (constant series).

    Args:
        values: List of numeric values.

    Returns:
        List of z-score floats, same length as values.
    """
    if len(values) < 2:
        return [0.0] * len(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return [0.0] * len(values)
    return [(v - mean) / stdev for v in values]


# ─────────────────────────────────────────────────────────────────────────────
# Detection passes
# ─────────────────────────────────────────────────────────────────────────────

def detect_price_outliers(
    records: list[PriceRecord],
    threshold: float = Z_SCORE_THRESHOLD,
) -> list[PriceRecord]:
    """Flag records whose close price z-score exceeds the threshold.

    Args:
        records: List of PriceRecord objects for one asset, one source.
        threshold: Z-score above which a record is flagged.

    Returns:
        List of flagged PriceRecord objects.
    """
    if not records:
        return []
    closes = [r.close for r in records]
    zscores = _z_scores(closes)
    flagged = [r for r, z in zip(records, zscores) if abs(z) > threshold]
    logger.info("price_outlier_detection", asset=records[0].asset,
                checked=len(records), flagged=len(flagged))
    return flagged


def detect_volume_spikes(
    records: list[PriceRecord],
    threshold: float = VOLUME_Z_SCORE_THRESHOLD,
) -> list[PriceRecord]:
    """Flag records whose volume z-score exceeds the threshold.

    Args:
        records: List of PriceRecord objects for one asset, one source.
        threshold: Z-score above which a record is flagged as a volume spike.

    Returns:
        List of flagged PriceRecord objects.
    """
    if not records:
        return []
    volumes = [r.volume for r in records]
    if all(v == 0 for v in volumes):
        return []  # crypto from CoinGecko has no volume — skip
    zscores = _z_scores(volumes)
    flagged = [r for r, z in zip(records, zscores) if abs(z) > threshold]
    logger.info("volume_spike_detection", asset=records[0].asset,
                checked=len(records), flagged=len(flagged))
    return flagged


def detect_cross_source_inconsistencies(
    records_a: list[PriceRecord],
    records_b: list[PriceRecord],
    max_diff_pct: float = CROSS_SOURCE_DIFF_PCT,
) -> list[tuple[PriceRecord, PriceRecord, float]]:
    """Compare close prices for the same asset from two different sources.

    Matches records by date and flags pairs where the relative price
    difference exceeds max_diff_pct.

    Args:
        records_a: Records from source A (e.g. yfinance).
        records_b: Records from source B (e.g. alpha_vantage).
        max_diff_pct: Maximum allowed relative difference (0.005 = 0.5%).

    Returns:
        List of (record_a, record_b, diff_pct) tuples for flagged pairs.
    """
    if not records_a or not records_b:
        return []

    # Index by date string for O(1) lookup
    def _date_key(r: PriceRecord) -> str:
        return r.time.strftime("%Y-%m-%d")

    index_b: dict[str, PriceRecord] = {_date_key(r): r for r in records_b}
    flagged: list[tuple[PriceRecord, PriceRecord, float]] = []

    for r_a in records_a:
        key = _date_key(r_a)
        r_b = index_b.get(key)
        if r_b is None:
            continue
        if r_a.close == 0:
            continue
        diff_pct = abs(r_a.close - r_b.close) / r_a.close
        if diff_pct > max_diff_pct:
            flagged.append((r_a, r_b, diff_pct))

    if flagged and records_a:
        logger.warning("cross_source_inconsistency", asset=records_a[0].asset,
                       flagged_pairs=len(flagged))
    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# Main filter function
# ─────────────────────────────────────────────────────────────────────────────

def filter_asset(
    records: list[PriceRecord],
    cross_source_records: list[PriceRecord] | None = None,
) -> NoiseFilterResult:
    """Run all three detection passes on one asset's price records.

    Quarantines all flagged records to Supabase. Never deletes source data.

    Args:
        records: All price records for one asset from the primary source.
        cross_source_records: Optional records from a secondary source for
            cross-source consistency checking.

    Returns:
        A NoiseFilterResult summarising what was flagged.
    """
    if not records:
        return NoiseFilterResult(asset="unknown", total_rows=0)

    asset = records[0].asset
    result = NoiseFilterResult(asset=asset, total_rows=len(records))
    quarantined_ids: set[str] = set()

    def _quarantine(record: PriceRecord, reason: str) -> None:
        """Quarantine one record and track it to avoid duplicates."""
        record_key = f"{record.asset}_{record.time.isoformat()}_{record.source}"
        if record_key in quarantined_ids:
            return
        quarantined_ids.add(record_key)
        quarantine_record(QuarantineRecord(
            original_table="prices",
            data={**record.raw, "close": record.close, "volume": record.volume},
            reason=reason,
            quarantined_by=AGENT_ID,
            original_id=record_key,
        ))
        result.quarantined += 1

    # Pass 1: Z-score price outliers
    price_outliers = detect_price_outliers(records)
    result.z_score_flags = len(price_outliers)
    for r in price_outliers:
        _quarantine(r, f"price_outlier_z_score_gt_{Z_SCORE_THRESHOLD}")

    # Pass 2: Volume spikes
    vol_spikes = detect_volume_spikes(records)
    result.volume_flags = len(vol_spikes)
    for r in vol_spikes:
        _quarantine(r, f"volume_spike_z_score_gt_{VOLUME_Z_SCORE_THRESHOLD}")

    # Pass 3: Cross-source consistency (only if secondary records provided)
    if cross_source_records:
        inconsistencies = detect_cross_source_inconsistencies(
            records, cross_source_records
        )
        result.cross_source_flags = len(inconsistencies)
        for r_a, r_b, diff_pct in inconsistencies:
            _quarantine(
                r_a,
                f"cross_source_diff_{diff_pct:.4f}_gt_{CROSS_SOURCE_DIFF_PCT}"
            )

    logger.info("noise_filter_asset_done", asset=asset,
                total=result.total_rows, quarantined=result.quarantined)
    return result


def run(
    records_by_asset: dict[str, list[PriceRecord]],
    cross_source_by_asset: dict[str, list[PriceRecord]] | None = None,
) -> list[NoiseFilterResult]:
    """Run the noise filter across all assets and log to Supabase.

    Args:
        records_by_asset: Dict mapping asset ticker → list of PriceRecord.
        cross_source_by_asset: Optional dict mapping asset ticker → secondary
            source records for cross-source consistency checks.

    Returns:
        List of NoiseFilterResult — one per asset.
    """
    agent_run = AgentRun(agent_name=AGENT_ID, trigger="scheduled")
    start_agent_run(agent_run)

    results: list[NoiseFilterResult] = []
    for asset, records in records_by_asset.items():
        cross = (cross_source_by_asset or {}).get(asset)
        result = filter_asset(records, cross_source_records=cross)
        results.append(result)

    total_quarantined = sum(r.quarantined for r in results)
    total_rows = sum(r.total_rows for r in results)

    summary = (
        f"Scanned {total_rows} rows across {len(results)} assets. "
        f"Quarantined {total_quarantined} records."
    )
    end_agent_run(agent_run, success=True, summary=summary)

    log_evolution(EvolutionLogEntry(
        agent_id=AGENT_ID,
        action="scan_prices",
        before_state={"total_rows": total_rows},
        after_state={"quarantined": total_quarantined},
        metadata={
            "assets_scanned": len(results),
            "z_score_flags": sum(r.z_score_flags for r in results),
            "volume_flags": sum(r.volume_flags for r in results),
            "cross_source_flags": sum(r.cross_source_flags for r in results),
        },
    ))

    return results
