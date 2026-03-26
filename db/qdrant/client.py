"""Qdrant client for FinBrain.

Provides a lazy singleton QdrantClient and typed helpers for:
- Creating / ensuring all 5 collections exist
- Upserting embedding points
- Searching by vector similarity
- Fetching points by ID
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    UpdateResult,
)

from db.qdrant.collections import ALL_COLLECTIONS, CollectionConfig
from skills.env import get_qdrant_api_key, get_qdrant_url
from skills.logger import get_logger

logger = get_logger(__name__)

_client: QdrantClient | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Client initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    """Return a cached QdrantClient, initialising it on first call.

    Returns:
        An authenticated QdrantClient connected to Qdrant Cloud.
    """
    global _client
    if _client is None:
        url = get_qdrant_url()
        api_key = get_qdrant_api_key()
        _client = QdrantClient(url=url, api_key=api_key)
        logger.info("qdrant_client_initialised", url=url[:40] + "...")
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Collection management
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collections() -> list[str]:
    """Create all 5 FinBrain collections if they do not already exist.

    Idempotent — safe to call on every startup.

    Returns:
        List of collection names that were acted on.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    created: list[str] = []

    for cfg in ALL_COLLECTIONS:
        if cfg.name not in existing:
            client.create_collection(
                collection_name=cfg.name,
                vectors_config=cfg.vector_params(),
            )
            logger.info("qdrant_collection_created", collection=cfg.name, dim=cfg.dim)
            created.append(cfg.name)
        else:
            logger.info("qdrant_collection_exists", collection=cfg.name)

    return created


def collection_info(name: str) -> dict[str, Any]:
    """Return basic info about a collection (point count, vector dim, status).

    Args:
        name: The collection name to query.

    Returns:
        Dict with keys: name, vectors_count, status, vector_size, distance.
    """
    client = get_client()
    info = client.get_collection(name)
    return {
        "name": name,
        "vectors_count": info.vectors_count,
        "status": str(info.status),
        "vector_size": info.config.params.vectors.size,  # type: ignore[union-attr]
        "distance": str(info.config.params.vectors.distance),  # type: ignore[union-attr]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Typed point dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingPoint:
    """One point to upsert into a Qdrant collection.

    Args:
        vector: The 768-dim embedding vector.
        payload: Arbitrary metadata stored alongside the vector.
        point_id: UUID string; auto-generated if not provided.
    """
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    point_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Validate that the vector has the expected dimension.

        Raises:
            ValueError: If the vector length does not match VECTOR_DIM.
        """
        from db.qdrant.collections import VECTOR_DIM
        if len(self.vector) != VECTOR_DIM:
            raise ValueError(
                f"Vector must be {VECTOR_DIM}-dimensional, got {len(self.vector)}"
            )

    def to_point_struct(self) -> PointStruct:
        """Convert to a Qdrant PointStruct for upsert.

        Returns:
            A PointStruct instance ready to pass to the Qdrant client.
        """
        return PointStruct(
            id=self.point_id,
            vector=self.vector,
            payload=self.payload,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def upsert_points(collection_name: str, points: list[EmbeddingPoint]) -> UpdateResult:
    """Upsert a batch of embedding points into a collection.

    Uses Qdrant's upsert which inserts new points and overwrites existing
    ones with the same ID. Safe to call repeatedly for the same points.

    Args:
        collection_name: Target collection name.
        points: List of EmbeddingPoint objects to upsert.

    Returns:
        The Qdrant UpdateResult for the operation.
    """
    if not points:
        logger.info("qdrant_upsert_skipped_empty", collection=collection_name)
        return UpdateResult(operation_id=0, status="completed")  # type: ignore[call-arg]

    client = get_client()
    structs = [p.to_point_struct() for p in points]
    result = client.upsert(collection_name=collection_name, points=structs)
    logger.info("qdrant_upserted", collection=collection_name, count=len(points))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────────────────────────────────────

def search(
    collection_name: str,
    query_vector: list[float],
    top_k: int = 10,
    score_threshold: float | None = None,
    filter_payload: dict[str, Any] | None = None,
) -> list[ScoredPoint]:
    """Search a collection for the nearest neighbours to a query vector.

    Args:
        collection_name: The collection to search.
        query_vector: The 768-dim query embedding.
        top_k: Maximum number of results to return.
        score_threshold: Minimum cosine similarity score (0–1) to include.
        filter_payload: Optional Qdrant payload filter dict.

    Returns:
        List of ScoredPoint objects ordered by descending similarity.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = get_client()
    qdrant_filter = None
    if filter_payload:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_payload.items()
        ]
        qdrant_filter = Filter(must=conditions)

    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "query_vector": query_vector,
        "limit": top_k,
    }
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    if qdrant_filter is not None:
        kwargs["query_filter"] = qdrant_filter

    results = client.search(**kwargs)
    logger.info("qdrant_searched", collection=collection_name, top_k=top_k,
                results_returned=len(results))
    return results


def fetch_by_id(collection_name: str, point_id: str) -> dict[str, Any] | None:
    """Retrieve a single point by its ID from a collection.

    Args:
        collection_name: The collection to query.
        point_id: The UUID string of the point to fetch.

    Returns:
        Dict with keys 'id', 'vector', 'payload', or None if not found.
    """
    client = get_client()
    results = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_vectors=True,
        with_payload=True,
    )
    if not results:
        return None
    p = results[0]
    return {"id": str(p.id), "vector": p.vector, "payload": p.payload}
