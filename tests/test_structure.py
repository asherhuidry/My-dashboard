"""Tests that verify the project folder structure and scaffolding are correct."""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _dirs() -> list[str]:
    """Return all required project directories."""
    return [
        "data/ingest",
        "data/agents",
        "db/timescale",
        "db/qdrant",
        "db/neo4j",
        "db/supabase",
        "db/agents",
        "ml/patterns",
        "ml/backtest",
        "ml/reasoning",
        "ml/improve",
        "ml/agents",
        "architect",
        "api/routes",
        "skills",
        "tests",
        ".github/workflows",
    ]


def _init_packages() -> list[str]:
    """Return all directories that must be Python packages (have __init__.py)."""
    return [
        "data",
        "data/ingest",
        "data/agents",
        "db",
        "db/timescale",
        "db/qdrant",
        "db/neo4j",
        "db/supabase",
        "db/agents",
        "ml",
        "ml/patterns",
        "ml/backtest",
        "ml/reasoning",
        "ml/improve",
        "ml/agents",
        "architect",
        "api",
        "api/routes",
        "skills",
        "tests",
    ]


def test_all_directories_exist() -> None:
    """Verify every required directory exists on disk."""
    for d in _dirs():
        path = ROOT / d
        assert path.is_dir(), f"Missing directory: {d}"


def test_all_packages_have_init() -> None:
    """Verify every Python package directory has an __init__.py."""
    for pkg in _init_packages():
        init = ROOT / pkg / "__init__.py"
        assert init.exists(), f"Missing __init__.py in: {pkg}"


def test_env_example_exists() -> None:
    """Verify .env.example is present and lists all required keys."""
    env_example = ROOT / ".env.example"
    assert env_example.exists(), ".env.example is missing"
    content = env_example.read_text()
    required_keys = [
        "ANTHROPIC_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "ALPHA_VANTAGE_KEY",
        "FRED_API_KEY",
    ]
    for key in required_keys:
        assert key in content, f"Missing env key in .env.example: {key}"


def test_gitignore_excludes_dotenv() -> None:
    """Verify .gitignore will not allow .env to be committed."""
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore is missing"
    content = gitignore.read_text()
    assert ".env" in content, ".env is not excluded in .gitignore"


def test_requirements_txt_exists() -> None:
    """Verify requirements.txt lists critical dependencies."""
    req = ROOT / "requirements.txt"
    assert req.exists(), "requirements.txt is missing"
    content = req.read_text()
    for pkg in ["fastapi", "torch", "supabase", "qdrant-client", "neo4j", "yfinance", "pytest"]:
        assert pkg in content, f"Missing package in requirements.txt: {pkg}"


def test_env_loader_raises_on_missing_var(monkeypatch: object) -> None:
    """Verify skills/env.py raises RuntimeError for unset variables."""
    import importlib
    import sys

    # Ensure env module is freshly imported without .env loaded
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if "skills.env" in sys.modules:
        del sys.modules["skills.env"]

    from skills.env import _require
    import pytest

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _require("ANTHROPIC_API_KEY")
