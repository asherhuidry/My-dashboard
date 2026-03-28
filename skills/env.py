"""Central environment variable loader for FinBrain.

All modules must import credentials from here — never read os.environ directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return the value of a required environment variable.

    Raises RuntimeError if the variable is not set, so misconfigured
    environments fail fast at startup rather than at runtime.

    Args:
        key: The name of the environment variable.

    Returns:
        The string value of the environment variable.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable '{key}' is not set. Check your .env file.")
    return value


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key."""
    return _require("ANTHROPIC_API_KEY")


def get_supabase_url() -> str:
    """Return the Supabase project URL."""
    return _require("SUPABASE_URL")


def get_supabase_key() -> str:
    """Return the Supabase service role key."""
    return _require("SUPABASE_KEY")


def get_qdrant_url() -> str:
    """Return the Qdrant Cloud cluster URL."""
    return _require("QDRANT_URL")


def get_qdrant_api_key() -> str:
    """Return the Qdrant API key."""
    return _require("QDRANT_API_KEY")


def get_neo4j_uri() -> str:
    """Return the Neo4j Aura connection URI."""
    return _require("NEO4J_URI")


def get_neo4j_user() -> str:
    """Return the Neo4j username."""
    return _require("NEO4J_USER")


def get_neo4j_password() -> str:
    """Return the Neo4j password."""
    return _require("NEO4J_PASSWORD")


def get_alpha_vantage_key() -> str:
    """Return the Alpha Vantage API key."""
    return _require("ALPHA_VANTAGE_KEY")


def get_fred_api_key() -> str:
    """Return the FRED API key."""
    return _require("FRED_API_KEY")


def get_finnhub_api_key() -> str:
    """Return the Finnhub API key."""
    return _require("FINNHUB_API_KEY")
