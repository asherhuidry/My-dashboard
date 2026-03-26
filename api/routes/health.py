"""Health check route."""
from __future__ import annotations
import os
from fastapi import APIRouter
from skills.env import get_supabase_url, get_supabase_key

router = APIRouter()

@router.get("/health")
def health_check() -> dict:
    """Return system health status."""
    checks = {
        "supabase": bool(os.getenv("SUPABASE_URL")),
        "qdrant": bool(os.getenv("QDRANT_URL")),
        "neo4j": bool(os.getenv("NEO4J_URI")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "fred": bool(os.getenv("FRED_API_KEY")),
    }
    all_ok = all(checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
        "version": "1.0.0",
    }
