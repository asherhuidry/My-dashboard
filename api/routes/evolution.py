"""Evolution log and roadmap routes."""
from __future__ import annotations

from fastapi import APIRouter, Query
from skills.env import get_supabase_url, get_supabase_key

router = APIRouter()


@router.get("/evolution-log")
def get_evolution_log(limit: int = Query(default=50, le=200)) -> dict:
    """Return recent agent activity from evolution_log."""
    try:
        from supabase import create_client
        client = create_client(get_supabase_url(), get_supabase_key())
        res = (
            client.table("evolution_log")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"logs": res.data}
    except Exception as e:
        return {"logs": [], "error": str(e)}


@router.get("/roadmap")
def get_roadmap() -> dict:
    """Return all roadmap tasks."""
    try:
        from supabase import create_client
        client = create_client(get_supabase_url(), get_supabase_key())
        res = (
            client.table("roadmap")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return {"tasks": res.data}
    except Exception as e:
        return {"tasks": [], "error": str(e)}
