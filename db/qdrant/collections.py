"""Qdrant collection definitions for FinBrain.

All collections use 768-dimensional cosine similarity vectors,
matching the output dimension of sentence-transformers/all-mpnet-base-v2
and similar models used throughout the embedding pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import Distance, VectorParams


VECTOR_DIM = 768
VECTOR_DISTANCE = Distance.COSINE


@dataclass(frozen=True)
class CollectionConfig:
    """Immutable configuration for one Qdrant collection.

    Args:
        name: The collection name in Qdrant.
        description: Human-readable description of what this collection stores.
        dim: Vector dimensionality.
        distance: Distance metric for similarity search.
    """
    name: str
    description: str
    dim: int = VECTOR_DIM
    distance: Distance = VECTOR_DISTANCE

    def vector_params(self) -> VectorParams:
        """Return the Qdrant VectorParams for this collection.

        Returns:
            A VectorParams instance configured with this collection's dim and distance.
        """
        return VectorParams(size=self.dim, distance=self.distance)


# ─────────────────────────────────────────────────────────────────────────────
# The 5 FinBrain collections
# ─────────────────────────────────────────────────────────────────────────────

ASSET_EMBEDDINGS = CollectionConfig(
    name="asset_embeddings",
    description="Per-asset context vectors built from price history, fundamentals, and news.",
)

NEWS_EMBEDDINGS = CollectionConfig(
    name="news_embeddings",
    description="News article and event embeddings for semantic search and sentiment context.",
)

PATTERN_EMBEDDINGS = CollectionConfig(
    name="pattern_embeddings",
    description="Chart pattern vectors for similarity-based pattern retrieval.",
)

MACRO_EMBEDDINGS = CollectionConfig(
    name="macro_embeddings",
    description="Macro indicator regime vectors for macro state similarity search.",
)

SIGNAL_EMBEDDINGS = CollectionConfig(
    name="signal_embeddings",
    description="Generated signal context vectors for explaining signal provenance.",
)

ALL_COLLECTIONS: list[CollectionConfig] = [
    ASSET_EMBEDDINGS,
    NEWS_EMBEDDINGS,
    PATTERN_EMBEDDINGS,
    MACRO_EMBEDDINGS,
    SIGNAL_EMBEDDINGS,
]
