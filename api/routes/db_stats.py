"""Database statistics route."""
from __future__ import annotations

from fastapi import APIRouter
from skills.env import get_supabase_url, get_supabase_key

router = APIRouter()

_TABLES = [
    "prices", "macro_events",
    "evolution_log", "roadmap", "signals",
    "model_registry", "api_sources", "system_health",
    "agent_runs", "quarantine",
]


@router.get("/db-stats")
def get_db_stats() -> dict:
    """Return row counts and status for all tables."""
    try:
        from supabase import create_client
        client = create_client(get_supabase_url(), get_supabase_key())
    except Exception as e:
        return {"error": str(e), "tables": []}

    tables = []
    for tbl in _TABLES:
        try:
            res   = client.table(tbl).select("*", count="exact").limit(1).execute()
            count = res.count if res.count is not None else len(res.data)
            tables.append({"table": tbl, "rows": count, "status": "ok"})
        except Exception as e:
            tables.append({"table": tbl, "rows": 0, "status": "error", "error": str(e)[:80]})

    total = sum(t["rows"] for t in tables)
    return {"tables": tables, "total_rows": total}
