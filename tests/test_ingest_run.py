"""Tests for ECB integration into the ingestion orchestrator.

The run.py module has pre-existing import issues (CRYPTO name mismatch)
that prevent direct import. These tests verify the ECB integration logic
in isolation by testing the connector directly and verifying the wiring
pattern.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data.ingest.ecb_connector import ECB_SERIES, ECBFetchResult, ECBSeriesDef


# ── Universe / import regression tests ─────────────────────────────────────

class TestUniverseImports:
    def test_run_module_imports_cleanly(self):
        """run.py must import without errors — guards the CRYPTO name fix."""
        from data.ingest import run  # noqa: F401

    def test_crypto_is_coingecko_format(self):
        """CRYPTO must be list[tuple[str, str]] for coingecko_connector.run()."""
        from data.ingest.universe import CRYPTO
        assert isinstance(CRYPTO, list)
        assert len(CRYPTO) >= 10
        for item in CRYPTO:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2
            coin_id, ticker = item
            assert isinstance(coin_id, str) and isinstance(ticker, str)

    def test_crypto_yf_still_exists(self):
        """CRYPTO_YF (Yahoo Finance format) must still be available."""
        from data.ingest.universe import CRYPTO_YF
        assert isinstance(CRYPTO_YF, list)
        assert all(isinstance(t, str) for t in CRYPTO_YF)

    def test_crypto_and_crypto_yf_same_count(self):
        """Both crypto lists should cover the same number of assets."""
        from data.ingest.universe import CRYPTO, CRYPTO_YF
        assert len(CRYPTO) == len(CRYPTO_YF)


# ── ECB connector integration tests ─────────────────────────────────────────

class TestECBInRunAll:
    """Test that ECB connector.run() works correctly as called by run_all()."""

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_ecb_run_returns_all_series(self, mock_fetch, mock_sleep):
        """run() with defaults should attempt all 8 ECB series."""
        from data.ingest.ecb_connector import run

        mock_fetch.return_value = ECBFetchResult(
            indicator="x", name="x", rows_fetched=10,
            rows_written=10, start_date="2024-01-01", end_date="2024-03-01",
        )
        results = run()
        assert len(results) == len(ECB_SERIES)
        assert mock_fetch.call_count == len(ECB_SERIES)

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_ecb_summary_matches_run_all_pattern(self, mock_fetch, mock_sleep):
        """Verify ECB results follow the same pattern run_all() expects:
        sum(r.rows_written for r in results if r.error is None)."""
        from data.ingest.ecb_connector import run

        mock_fetch.side_effect = [
            ECBFetchResult(
                indicator="ECB_EURUSD", name="ok", rows_fetched=100,
                rows_written=100, start_date="2024-01-01", end_date="2024-03-01",
            ),
            ECBFetchResult(
                indicator="ECB_DFR", name="fail", rows_fetched=0,
                rows_written=0, start_date=None, end_date=None, error="timeout",
            ),
        ]
        results = run(series=[ECB_SERIES[0], ECB_SERIES[4]])

        # This is exactly what run_all() does to compute summary["ecb"]
        ecb_rows = sum(r.rows_written for r in results if r.error is None)
        assert ecb_rows == 100

    @patch("data.ingest.ecb_connector.time.sleep")
    @patch("data.ingest.ecb_connector.fetch_series")
    def test_all_fail_gives_zero(self, mock_fetch, mock_sleep):
        from data.ingest.ecb_connector import run

        mock_fetch.return_value = ECBFetchResult(
            indicator="x", name="x", rows_fetched=0,
            rows_written=0, start_date=None, end_date=None, error="boom",
        )
        results = run()
        ecb_rows = sum(r.rows_written for r in results if r.error is None)
        assert ecb_rows == 0


# ── Wiring verification ─────────────────────────────────────────────────────

class TestRunModuleWiring:
    def test_ecb_imported_in_run_module(self):
        """Verify run.py imports ecb_connector."""
        import importlib
        import ast
        from pathlib import Path

        run_path = Path(__file__).parent.parent / "data" / "ingest" / "run.py"
        source = run_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all imported names
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.append(alias.name)

        assert "ecb_connector" in imported_names

    def test_ecb_in_run_all_function(self):
        """Verify run_all() function body references ecb_connector.run."""
        from pathlib import Path

        run_path = Path(__file__).parent.parent / "data" / "ingest" / "run.py"
        source = run_path.read_text(encoding="utf-8")

        assert "ecb_connector.run()" in source
        assert 'summary["ecb"]' in source
        assert 'connector="ecb"' in source

    def test_ecb_in_dry_run_function(self):
        """Verify _dry_run() includes ECB series listing."""
        from pathlib import Path

        run_path = Path(__file__).parent.parent / "data" / "ingest" / "run.py"
        source = run_path.read_text(encoding="utf-8")

        assert "ecb_connector.ECB_SERIES" in source
        assert "ECB" in source


# ── Workflow verification ────────────────────────────────────────────────────

class TestWorkflowWiring:
    def test_comprehensive_ingest_has_ecb_job(self):
        """The comprehensive_ingest.yml should have an ecb-ingest job."""
        from pathlib import Path
        import yaml

        wf_path = (
            Path(__file__).parent.parent
            / ".github" / "workflows" / "comprehensive_ingest.yml"
        )
        content = wf_path.read_text(encoding="utf-8")
        wf = yaml.safe_load(content)

        assert "ecb-ingest" in wf["jobs"]
        ecb_job = wf["jobs"]["ecb-ingest"]
        assert "ECB" in ecb_job["name"]

    def test_pipeline_summary_includes_ecb(self):
        """The pipeline-summary job should report ECB status."""
        from pathlib import Path

        wf_path = (
            Path(__file__).parent.parent
            / ".github" / "workflows" / "comprehensive_ingest.yml"
        )
        content = wf_path.read_text(encoding="utf-8")
        assert "ecb-ingest" in content
        assert "ECB ingest" in content

    def test_discovery_pipeline_calls_run(self):
        """The discovery-pipeline job should call run() (which persists), not hunt_correlations()."""
        from pathlib import Path

        wf_path = (
            Path(__file__).parent.parent
            / ".github" / "workflows" / "comprehensive_ingest.yml"
        )
        content = wf_path.read_text(encoding="utf-8")
        # Must import and call the run() entrypoint (which persists discoveries)
        assert "from data.agents.correlation_hunter import run" in content
        assert "findings = run(" in content
        # Must NOT call hunt_correlations directly (bypasses persistence)
        assert "hunt_correlations(" not in content

    def test_discovery_pipeline_has_supabase_env(self):
        """The discovery-pipeline job needs SUPABASE env vars for discovery persistence."""
        import yaml
        from pathlib import Path

        wf_path = (
            Path(__file__).parent.parent
            / ".github" / "workflows" / "comprehensive_ingest.yml"
        )
        wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        # Env vars are set at workflow level, verify they exist
        top_env = wf.get("env", {})
        assert "SUPABASE_URL" in top_env, "workflow must pass SUPABASE_URL env var"
        assert "SUPABASE_KEY" in top_env, "workflow must pass SUPABASE_KEY env var"

    def test_pipeline_summary_needs_discovery_pipeline(self):
        """Pipeline summary must depend on discovery-pipeline to report its status."""
        import yaml
        from pathlib import Path

        wf_path = (
            Path(__file__).parent.parent
            / ".github" / "workflows" / "comprehensive_ingest.yml"
        )
        wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        summary_job = wf["jobs"]["pipeline-summary"]
        assert "discovery-pipeline" in summary_job["needs"]


# ── ECB series metadata ─────────────────────────────────────────────────────

class TestECBSeriesMetadata:
    def test_all_indicators_prefixed(self):
        for s in ECB_SERIES:
            assert s.indicator.startswith("ECB_"), f"{s.indicator} missing ECB_ prefix"

    def test_no_auth_required(self):
        """ECB API requires no authentication — this is a key property."""
        # Verify by checking the connector has no env var lookups
        import inspect
        from data.ingest import ecb_connector
        source = inspect.getsource(ecb_connector)
        assert "get_api_key" not in source.lower()
        assert "API_KEY" not in source  # no key references in connector code

    def test_series_cover_multiple_categories(self):
        frequencies = {s.frequency for s in ECB_SERIES}
        assert "daily" in frequencies
        assert "monthly" in frequencies or "event" in frequencies
