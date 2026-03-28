"""Tests for the automated source scouting runner."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
from typing import Any

import pytest

from data.scout.auto_runner import run_auto_scout, MAX_CANDIDATES_PER_RUN


def _make_candidate(source_id: str) -> MagicMock:
    c = MagicMock()
    c.source_id = source_id
    return c


# All deferred imports in auto_runner must be patched at source
_P_CATALOG   = "data.scout.schema.CANDIDATE_CATALOG"
_P_NORMALIZE = "data.scout.schema.normalize_source_candidate"
_P_SCORE     = "data.scout.scorer.score_source_candidate"
_P_REGISTRY  = "data.registry.source_registry.SourceRegistry"
_P_PIPELINE  = "data.scout.pipeline.run_scout_pipeline"
_P_LOG       = "data.scout.auto_runner._log_to_evolution"


class TestAutoRunner:
    """Tests for run_auto_scout."""

    @patch(_P_LOG)
    @patch(_P_SCORE)
    @patch(_P_NORMALIZE)
    @patch(_P_CATALOG, new=[{"source_id": "a"}, {"source_id": "b"}])
    def test_dry_run_returns_summary(
        self,
        mock_normalize: MagicMock,
        mock_score: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        mock_normalize.side_effect = [_make_candidate("a"), _make_candidate("b")]
        mock_score.side_effect = [0.8, 0.6]

        result = run_auto_scout(dry_run=True)

        assert result["status"] == "dry_run"
        assert result["total_candidates"] == 2
        assert result["batch_size"] == 2
        mock_log.assert_called_once()

    @patch(_P_LOG)
    @patch(_P_SCORE)
    @patch(_P_NORMALIZE)
    @patch(_P_CATALOG, new=[{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}])
    def test_max_candidates_limits_batch(
        self,
        mock_normalize: MagicMock,
        mock_score: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        mock_normalize.side_effect = [_make_candidate(s) for s in ["a", "b", "c"]]
        mock_score.side_effect = [0.9, 0.7, 0.5]

        result = run_auto_scout(max_candidates=2, dry_run=True)
        assert result["batch_size"] == 2

    @patch(_P_LOG)
    @patch(_P_SCORE)
    @patch(_P_NORMALIZE)
    @patch(_P_CATALOG, new=[{"source_id": "a"}])
    def test_dry_run_does_not_probe(
        self,
        mock_normalize: MagicMock,
        mock_score: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        mock_normalize.return_value = _make_candidate("a")
        mock_score.return_value = 0.8

        result = run_auto_scout(dry_run=True)
        assert "probed" not in result
        assert result["dry_run"] is True

    @patch(_P_LOG)
    @patch(_P_PIPELINE)
    @patch(_P_REGISTRY)
    @patch(_P_SCORE)
    @patch(_P_NORMALIZE)
    @patch(_P_CATALOG, new=[{"source_id": "a"}])
    def test_full_run_calls_pipeline(
        self,
        mock_normalize: MagicMock,
        mock_score: MagicMock,
        mock_registry: MagicMock,
        mock_pipeline: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        mock_normalize.return_value = _make_candidate("a")
        mock_score.return_value = 0.8

        probe_result = MagicMock()
        probe_result.total = 1
        probe_result.reachable = 1
        probe_result.failed = 0
        pipeline_result = MagicMock()
        pipeline_result.probe_result = probe_result
        pipeline_result.newly_sampled = ["a"]
        pipeline_result.newly_validated = ["a"]
        pipeline_result.sample_failures = []
        mock_pipeline.return_value = pipeline_result

        result = run_auto_scout(dry_run=False)

        assert result["status"] == "completed"
        assert result["probed"] == 1
        assert result["newly_validated"] == ["a"]
        mock_pipeline.assert_called_once()

    @patch(_P_LOG)
    @patch(_P_SCORE)
    @patch(_P_NORMALIZE)
    @patch(_P_CATALOG, new=[{"source_id": "a"}, {"source_id": "b"}])
    def test_ranks_by_score_descending(
        self,
        mock_normalize: MagicMock,
        mock_score: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        mock_normalize.side_effect = [_make_candidate("a"), _make_candidate("b")]
        mock_score.side_effect = [0.3, 0.9]

        result = run_auto_scout(max_candidates=1, dry_run=True)
        top_ids = list(result["top_scores"].keys())
        assert top_ids[0] == "b"
