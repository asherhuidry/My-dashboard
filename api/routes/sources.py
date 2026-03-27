"""Read-only API for the source registry lifecycle.

Exposes the local source registry so dashboards can inspect what the
scout / probe / validation pipeline has discovered, sampled, and validated.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from skills.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_registry():
    """Lazy-load the source registry (avoids import-time file I/O)."""
    from data.registry.source_registry import SourceRegistry
    return SourceRegistry()


@router.get("/sources")
def list_sources(
    status:   str | None = Query(default=None, description="Filter by lifecycle status"),
    category: str | None = Query(default=None, description="Filter by category (macro, equity, crypto, ...)"),
    search:   str | None = Query(default=None, description="Substring search across id, name, notes"),
    min_score: float     = Query(default=0.0, ge=0, le=1, description="Minimum reliability score"),
    sort:     str        = Query(default="score", description="Sort by: score, name, status, discovered_at"),
    limit:    int        = Query(default=100, ge=1, le=500),
) -> dict:
    """List registered sources with optional filters.

    Returns sources sorted by the chosen field (descending for score,
    ascending for name/status).
    """
    try:
        reg = _get_registry()

        # Apply filters
        if search:
            sources = reg.search(search)
            # Still apply other filters on top of search results
            if status:
                sources = [s for s in sources if s.status.value == status]
            if category:
                sources = [s for s in sources if s.category == category]
            if min_score > 0:
                sources = [s for s in sources if s.reliability_score >= min_score]
        else:
            from data.registry.source_registry import SourceStatus
            status_enum = None
            if status:
                try:
                    status_enum = SourceStatus(status)
                except ValueError:
                    pass
            sources = reg.filter(
                status=status_enum,
                category=category,
                min_score=min_score,
            )

        # Sort
        if sort == "score":
            sources.sort(key=lambda s: s.reliability_score, reverse=True)
        elif sort == "name":
            sources.sort(key=lambda s: s.name.lower())
        elif sort == "status":
            _STATUS_ORDER = {"approved": 0, "validated": 1, "sampled": 2, "discovered": 3, "quarantined": 4, "rejected": 5}
            sources.sort(key=lambda s: _STATUS_ORDER.get(s.status.value, 9))
        elif sort == "discovered_at":
            sources.sort(key=lambda s: s.discovered_at or "", reverse=True)

        # Limit
        sources = sources[:limit]

        return {
            "sources": [s.to_dict() for s in sources],
            "total": len(sources),
            "filters": {
                k: v for k, v in {
                    "status": status,
                    "category": category,
                    "search": search,
                    "min_score": min_score if min_score > 0 else None,
                    "sort": sort,
                    "limit": limit,
                }.items() if v is not None
            },
        }
    except Exception as exc:
        logger.error("sources_list_failed", error=str(exc))
        # Return empty state rather than crash — registry file may not exist yet
        return {"sources": [], "total": 0, "filters": {}, "registry_missing": True}


@router.get("/sources/summary")
def source_summary() -> dict:
    """High-level summary of the source registry lifecycle state.

    Returns status counts, category breakdown, top validated/scored sources,
    and pipeline activity indicators.
    """
    try:
        reg = _get_registry()
        all_sources = reg.all()

        if not all_sources:
            return _empty_summary()

        # Status counts
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_acquisition: dict[str, int] = {}
        free_count = 0
        auth_count = 0

        for s in all_sources:
            st = s.status.value
            by_status[st] = by_status.get(st, 0) + 1
            by_category[s.category] = by_category.get(s.category, 0) + 1
            by_acquisition[s.acquisition_method] = by_acquisition.get(s.acquisition_method, 0) + 1
            if s.free_tier:
                free_count += 1
            if s.auth_required:
                auth_count += 1

        # Top scored (top 5 by reliability_score)
        scored = sorted(all_sources, key=lambda s: s.reliability_score, reverse=True)
        top_scored = [
            {"source_id": s.source_id, "name": s.name, "score": s.reliability_score,
             "status": s.status.value, "category": s.category}
            for s in scored[:5]
        ]

        # Top validated (validated or approved, by score)
        active = [s for s in all_sources if s.status.value in ("validated", "approved")]
        active.sort(key=lambda s: s.reliability_score, reverse=True)
        top_validated = [
            {"source_id": s.source_id, "name": s.name, "score": s.reliability_score,
             "status": s.status.value, "category": s.category}
            for s in active[:5]
        ]

        return {
            "total": len(all_sources),
            "by_status": by_status,
            "by_category": by_category,
            "by_acquisition": by_acquisition,
            "free_count": free_count,
            "auth_required_count": auth_count,
            "top_scored": top_scored,
            "top_validated": top_validated,
        }
    except Exception as exc:
        logger.error("sources_summary_failed", error=str(exc))
        return _empty_summary()


def _empty_summary() -> dict:
    """Zero-state summary when no sources exist."""
    return {
        "total": 0,
        "by_status": {},
        "by_category": {},
        "by_acquisition": {},
        "free_count": 0,
        "auth_required_count": 0,
        "top_scored": [],
        "top_validated": [],
    }
