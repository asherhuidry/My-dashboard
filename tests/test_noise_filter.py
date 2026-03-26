"""Tests for the noise filter agent.

Uses synthetic dirty data to verify all three detection passes and
quarantine behaviour. All Supabase calls are mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from data.agents.noise_filter import (
    CROSS_SOURCE_DIFF_PCT,
    VOLUME_Z_SCORE_THRESHOLD,
    Z_SCORE_THRESHOLD,
    NoiseFilterResult,
    PriceRecord,
    _z_scores,
    detect_cross_source_inconsistencies,
    detect_price_outliers,
    detect_volume_spikes,
    filter_asset,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year: int, month: int, day: int) -> datetime:
    """Return a UTC-aware datetime."""
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_records(
    closes: list[float],
    volumes: list[float] | None = None,
    asset: str = "AAPL",
    source: str = "yfinance",
) -> list[PriceRecord]:
    """Build a list of PriceRecord objects from close/volume lists.

    Args:
        closes: List of closing prices.
        volumes: List of volumes (defaults to 1_000_000 each).
        asset: Ticker symbol.
        source: Data source label.

    Returns:
        List of PriceRecord objects with sequential dates starting 2024-01-01.
    """
    if volumes is None:
        volumes = [1_000_000.0] * len(closes)
    base = _utc(2024, 1, 1)
    return [
        PriceRecord(
            time=base + timedelta(days=i),
            asset=asset,
            asset_class="equity",
            close=c,
            volume=v,
            source=source,
            raw={"close": c, "volume": v},
        )
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


# ---------------------------------------------------------------------------
# _z_scores helper
# ---------------------------------------------------------------------------

class TestZScores:
    """Tests for the _z_scores statistical helper."""

    def test_known_values(self) -> None:
        """_z_scores returns correct z-scores for a simple series."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        zs = _z_scores(values)
        assert len(zs) == len(values)
        # Mean = 5.0, stdev ≈ 2.0 — value 9 should have positive z
        assert zs[-1] > 0
        assert zs[0] < 0

    def test_constant_series_returns_zeros(self) -> None:
        """_z_scores returns all zeros for a constant series."""
        zs = _z_scores([100.0] * 10)
        assert all(z == 0.0 for z in zs)

    def test_single_value_returns_zero(self) -> None:
        """_z_scores returns [0.0] for a single-element list."""
        assert _z_scores([42.0]) == [0.0]

    def test_empty_returns_empty(self) -> None:
        """_z_scores returns [] for an empty list."""
        assert _z_scores([]) == []


# ---------------------------------------------------------------------------
# detect_price_outliers
# ---------------------------------------------------------------------------

class TestDetectPriceOutliers:
    """Tests for detect_price_outliers()."""

    def test_flags_extreme_outlier(self) -> None:
        """detect_price_outliers flags a record with an extreme price spike."""
        # 29 normal closes + 1 extreme spike
        normal = [150.0] * 29
        records = _make_records(normal + [99999.0])
        flagged = detect_price_outliers(records, threshold=4.0)
        assert len(flagged) == 1
        assert flagged[0].close == 99999.0

    def test_no_flags_for_clean_data(self) -> None:
        """detect_price_outliers returns empty list for clean price series."""
        records = _make_records([150.0 + i * 0.1 for i in range(30)])
        flagged = detect_price_outliers(records, threshold=4.0)
        assert flagged == []

    def test_empty_records_returns_empty(self) -> None:
        """detect_price_outliers returns [] for empty input."""
        assert detect_price_outliers([]) == []

    def test_flags_negative_price_spike(self) -> None:
        """detect_price_outliers flags a large negative outlier."""
        normal = [150.0] * 29
        records = _make_records(normal + [-999.0])
        flagged = detect_price_outliers(records, threshold=4.0)
        assert len(flagged) == 1
        assert flagged[0].close == -999.0


# ---------------------------------------------------------------------------
# detect_volume_spikes
# ---------------------------------------------------------------------------

class TestDetectVolumeSpikes:
    """Tests for detect_volume_spikes()."""

    def test_flags_extreme_volume_spike(self) -> None:
        """detect_volume_spikes flags a record with an extreme volume spike.

        Uses 50 normal points so the max achievable z-score is √49 ≈ 7.0,
        which exceeds the threshold of 6.0 for a genuine spike.
        """
        normal_vol = [1_000_000.0] * 49
        spike_vol = 500_000_000.0
        records = _make_records(
            closes=[150.0] * 50,
            volumes=normal_vol + [spike_vol],
        )
        flagged = detect_volume_spikes(records, threshold=6.0)
        assert len(flagged) == 1
        assert flagged[0].volume == spike_vol

    def test_skips_all_zero_volumes(self) -> None:
        """detect_volume_spikes returns [] when all volumes are zero (crypto)."""
        records = _make_records([100.0] * 10, volumes=[0.0] * 10)
        flagged = detect_volume_spikes(records, threshold=6.0)
        assert flagged == []

    def test_no_flags_for_normal_volume(self) -> None:
        """detect_volume_spikes returns [] for normally distributed volumes."""
        vols = [1_000_000.0 + i * 10_000 for i in range(30)]
        records = _make_records([150.0] * 30, volumes=vols)
        flagged = detect_volume_spikes(records, threshold=6.0)
        assert flagged == []

    def test_empty_records_returns_empty(self) -> None:
        """detect_volume_spikes returns [] for empty input."""
        assert detect_volume_spikes([]) == []


# ---------------------------------------------------------------------------
# detect_cross_source_inconsistencies
# ---------------------------------------------------------------------------

class TestDetectCrossSourceInconsistencies:
    """Tests for detect_cross_source_inconsistencies()."""

    def test_flags_large_discrepancy(self) -> None:
        """flags a pair where the two sources differ by more than 0.5%."""
        records_yf = _make_records([150.0] * 5, source="yfinance")
        # One date has a 2% discrepancy
        records_av = _make_records([150.0, 150.0, 147.0, 150.0, 150.0],
                                   source="alpha_vantage")
        flagged = detect_cross_source_inconsistencies(records_yf, records_av,
                                                      max_diff_pct=0.005)
        assert len(flagged) == 1
        _, _, diff_pct = flagged[0]
        assert diff_pct > 0.005

    def test_no_flags_for_identical_data(self) -> None:
        """returns [] when both sources have identical close prices."""
        records_a = _make_records([150.0] * 5, source="yfinance")
        records_b = _make_records([150.0] * 5, source="alpha_vantage")
        flagged = detect_cross_source_inconsistencies(records_a, records_b)
        assert flagged == []

    def test_no_flags_for_small_diff(self) -> None:
        """returns [] when price difference is within tolerance."""
        records_a = _make_records([150.0] * 5, source="yfinance")
        # 0.2% difference — below 0.5% threshold
        records_b = _make_records([150.3] * 5, source="alpha_vantage")
        flagged = detect_cross_source_inconsistencies(records_a, records_b)
        assert flagged == []

    def test_skips_dates_not_in_both_sources(self) -> None:
        """Does not flag dates that only appear in one source."""
        records_a = _make_records([150.0, 151.0, 152.0], source="yfinance")
        # Only overlaps on first date
        records_b = [PriceRecord(
            time=_utc(2024, 1, 1), asset="AAPL", asset_class="equity",
            close=150.0, volume=1e6, source="alpha_vantage", raw={}
        )]
        flagged = detect_cross_source_inconsistencies(records_a, records_b)
        assert flagged == []

    def test_empty_inputs_return_empty(self) -> None:
        """returns [] when either input list is empty."""
        records = _make_records([150.0] * 5)
        assert detect_cross_source_inconsistencies([], records) == []
        assert detect_cross_source_inconsistencies(records, []) == []


# ---------------------------------------------------------------------------
# filter_asset
# ---------------------------------------------------------------------------

class TestFilterAsset:
    """Tests for filter_asset() — the main per-asset filter function."""

    def test_quarantines_price_outlier(self) -> None:
        """filter_asset quarantines one record with an extreme price outlier."""
        normal = [150.0] * 29
        records = _make_records(normal + [99999.0])

        with patch("data.agents.noise_filter.quarantine_record") as mock_q, \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            result = filter_asset(records)

        assert result.z_score_flags == 1
        assert result.quarantined >= 1
        mock_q.assert_called()

    def test_quarantines_volume_spike(self) -> None:
        """filter_asset quarantines a record with an extreme volume spike.

        Uses 50 points so the spike z-score exceeds the threshold of 6.0.
        """
        records = _make_records(
            closes=[150.0] * 50,
            volumes=[1_000_000.0] * 49 + [500_000_000.0],
        )

        with patch("data.agents.noise_filter.quarantine_record") as mock_q, \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            result = filter_asset(records)

        assert result.volume_flags == 1
        assert result.quarantined >= 1
        mock_q.assert_called()

    def test_quarantines_cross_source_discrepancy(self) -> None:
        """filter_asset quarantines a record with cross-source price mismatch."""
        primary = _make_records([150.0] * 5)
        secondary = _make_records([150.0, 150.0, 147.0, 150.0, 150.0],
                                  source="alpha_vantage")

        with patch("data.agents.noise_filter.quarantine_record") as mock_q, \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            result = filter_asset(primary, cross_source_records=secondary)

        assert result.cross_source_flags == 1
        assert result.quarantined >= 1

    def test_no_quarantine_for_clean_data(self) -> None:
        """filter_asset quarantines nothing for perfectly clean data."""
        records = _make_records([150.0 + i * 0.01 for i in range(30)])

        with patch("data.agents.noise_filter.quarantine_record") as mock_q, \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            result = filter_asset(records)

        assert result.quarantined == 0
        mock_q.assert_not_called()

    def test_deduplicates_quarantine(self) -> None:
        """filter_asset does not quarantine the same record twice."""
        # Construct a record that is both a price outlier AND flagged in cross-source
        extreme_val = 99999.0
        primary = _make_records([150.0] * 29 + [extreme_val])
        secondary = _make_records([150.0] * 29 + [100.0],  # large diff on last record
                                  source="alpha_vantage")

        with patch("data.agents.noise_filter.quarantine_record") as mock_q, \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            result = filter_asset(primary, cross_source_records=secondary)

        # The outlier record should appear only once in quarantine
        quarantine_ids = [
            call_args[0][0].original_id
            for call_args in mock_q.call_args_list
        ]
        assert len(quarantine_ids) == len(set(quarantine_ids))

    def test_empty_records_returns_zero_result(self) -> None:
        """filter_asset handles empty records without errors."""
        result = filter_asset([])
        assert result.total_rows == 0
        assert result.quarantined == 0


# ---------------------------------------------------------------------------
# run (orchestrator)
# ---------------------------------------------------------------------------

class TestNoiseFilterRun:
    """Tests for the run() orchestrator."""

    def test_run_processes_all_assets(self) -> None:
        """run() returns one result per asset in the input dict."""
        assets = {
            "AAPL": _make_records([150.0 + i for i in range(10)], asset="AAPL"),
            "MSFT": _make_records([300.0 + i for i in range(10)], asset="MSFT"),
        }

        with patch("data.agents.noise_filter.quarantine_record"), \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            results = run(assets)

        assert len(results) == 2
        asset_names = {r.asset for r in results}
        assert asset_names == {"AAPL", "MSFT"}

    def test_run_logs_to_supabase(self) -> None:
        """run() calls start_agent_run, end_agent_run, and log_evolution."""
        assets = {"AAPL": _make_records([150.0] * 5, asset="AAPL")}

        with patch("data.agents.noise_filter.quarantine_record"), \
             patch("data.agents.noise_filter.start_agent_run") as mock_start, \
             patch("data.agents.noise_filter.end_agent_run") as mock_end, \
             patch("data.agents.noise_filter.log_evolution") as mock_log:
            run(assets)

        mock_start.assert_called_once()
        mock_end.assert_called_once()
        mock_log.assert_called_once()

    def test_run_total_rows_counted(self) -> None:
        """run() result total_rows sums all assets correctly."""
        assets = {
            "AAPL": _make_records([150.0] * 5, asset="AAPL"),
            "BTC":  _make_records([40000.0] * 8, asset="BTC"),
        }

        with patch("data.agents.noise_filter.quarantine_record"), \
             patch("data.agents.noise_filter.start_agent_run"), \
             patch("data.agents.noise_filter.end_agent_run"), \
             patch("data.agents.noise_filter.log_evolution"):
            results = run(assets)

        total = sum(r.total_rows for r in results)
        assert total == 13
