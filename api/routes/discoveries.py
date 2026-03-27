"""Read-only API for persisted correlation discoveries.

Exposes the discoveries table populated by the weekly correlation hunter
so dashboards and research tools can query accumulated research state.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from skills.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/discoveries")
def list_discoveries(
    series:    str | None   = Query(default=None, description="Filter to findings involving this series (as A or B)"),
    strength:  str | None   = Query(default=None, description="Filter by strength: strong, moderate, weak"),
    run_id:    str | None   = Query(default=None, description="Filter to a specific discovery run"),
    min_abs_r: float | None = Query(default=None, ge=0, le=1, description="Minimum |pearson_r| to include"),
    limit:     int          = Query(default=50, ge=1, le=500, description="Maximum rows to return"),
) -> dict:
    """Query persisted correlation discoveries with optional filters.

    Returns discoveries ordered by |pearson_r| descending.
    """
    try:
        from db.supabase.client import get_discoveries

        rows = get_discoveries(
            series=series,
            strength=strength,
            run_id=run_id,
            min_abs_r=min_abs_r,
            limit=limit,
        )
        return {
            "discoveries": rows,
            "total": len(rows),
            "filters": {
                k: v for k, v in {
                    "series": series,
                    "strength": strength,
                    "run_id": run_id,
                    "min_abs_r": min_abs_r,
                    "limit": limit,
                }.items() if v is not None
            },
        }
    except Exception as exc:
        logger.error("discoveries_query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/discoveries/summary")
def discovery_summary() -> dict:
    """Return a high-level summary of accumulated discovery state.

    Useful for dashboards to show: how many discoveries exist,
    breakdown by strength, most recent run timestamp.
    """
    try:
        from db.supabase.client import get_discoveries

        all_rows = get_discoveries(limit=500)

        if not all_rows:
            return {
                "total_discoveries": 0,
                "by_strength": {},
                "unique_series": 0,
                "run_count": 0,
                "latest_run_id": None,
            }

        by_strength: dict[str, int] = {}
        series_set: set[str] = set()
        run_ids: set[str] = set()

        for row in all_rows:
            s = row.get("strength", "unknown")
            by_strength[s] = by_strength.get(s, 0) + 1
            series_set.add(row.get("series_a", ""))
            series_set.add(row.get("series_b", ""))
            rid = row.get("run_id")
            if rid:
                run_ids.add(rid)

        series_set.discard("")

        return {
            "total_discoveries": len(all_rows),
            "by_strength": by_strength,
            "unique_series": len(series_set),
            "run_count": len(run_ids),
            "latest_run_id": all_rows[0].get("run_id") if all_rows else None,
        }
    except Exception as exc:
        logger.error("discoveries_summary_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
