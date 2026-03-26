"""Tests for the Qdrant client and collections module.

All tests mock the QdrantClient so they run without live credentials.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from db.qdrant.collections import (
    ALL_COLLECTIONS,
    VECTOR_DIM,
    VECTOR_DISTANCE,
    ASSET_EMBEDDINGS,
    NEWS_EMBEDDINGS,
    PATTERN_EMBEDDINGS,
    MACRO_EMBEDDINGS,
    SIGNAL_EMBEDDINGS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_vector(dim: int = VECTOR_DIM) -> list[float]:
    """Return a zero vector of the given dimension."""
    return [0.0] * dim


def _mock_qdrant_client(existing_collections: list[str] | None = None) -> MagicMock:
    """Return a mock QdrantClient.

    Args:
        existing_collections: Collection names to report as already existing.

    Returns:
        A configured MagicMock that mimics QdrantClient.
    """
    existing = existing_collections or []

    mock_col = MagicMock()
    mock_col.name = ""

    mock_collections_response = MagicMock()
    mock_collections_response.collections = [
        _col_obj(n) for n in existing
    ]

    mock_client = MagicMock()
    mock_client.get_collections.return_value = mock_collections_response
    mock_client.create_collection.return_value = True
    mock_client.upsert.return_value = MagicMock(operation_id=1, status="completed")
    mock_client.search.return_value = []
    mock_client.retrieve.return_value = []
    return mock_client


def _col_obj(name: str) -> MagicMock:
    """Return a mock collection info object with the given name."""
    obj = MagicMock()
    obj.name = name
    return obj


# ---------------------------------------------------------------------------
# CollectionConfig
# ---------------------------------------------------------------------------

class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_all_five_collections_defined(self) -> None:
        """ALL_COLLECTIONS contains exactly 5 collections."""
        assert len(ALL_COLLECTIONS) == 5

    def test_all_collections_are_768_dim(self) -> None:
        """Every collection is configured with 768-dimensional vectors."""
        for cfg in ALL_COLLECTIONS:
            assert cfg.dim == VECTOR_DIM, f"{cfg.name} has wrong dim: {cfg.dim}"

    def test_all_collections_use_cosine(self) -> None:
        """Every collection uses cosine distance."""
        from qdrant_client.models import Distance
        for cfg in ALL_COLLECTIONS:
            assert cfg.distance == Distance.COSINE, f"{cfg.name} uses wrong distance"

    def test_collection_names(self) -> None:
        """All 5 expected collection names are present."""
        names = {cfg.name for cfg in ALL_COLLECTIONS}
        assert names == {
            "asset_embeddings",
            "news_embeddings",
            "pattern_embeddings",
            "macro_embeddings",
            "signal_embeddings",
        }

    def test_vector_params_returns_correct_type(self) -> None:
        """vector_params() returns a VectorParams with correct size."""
        from qdrant_client.models import VectorParams
        vp = ASSET_EMBEDDINGS.vector_params()
        assert isinstance(vp, VectorParams)
        assert vp.size == VECTOR_DIM


# ---------------------------------------------------------------------------
# EmbeddingPoint
# ---------------------------------------------------------------------------

class TestEmbeddingPoint:
    """Tests for EmbeddingPoint dataclass."""

    def test_auto_generates_uuid(self) -> None:
        """EmbeddingPoint generates a unique UUID if point_id not provided."""
        from db.qdrant.client import EmbeddingPoint
        p1 = EmbeddingPoint(vector=_zero_vector())
        p2 = EmbeddingPoint(vector=_zero_vector())
        assert p1.point_id != p2.point_id

    def test_wrong_dim_raises(self) -> None:
        """EmbeddingPoint raises ValueError for wrong vector dimension."""
        from db.qdrant.client import EmbeddingPoint
        with pytest.raises(ValueError, match="768-dimensional"):
            EmbeddingPoint(vector=[0.0] * 10)

    def test_to_point_struct(self) -> None:
        """to_point_struct() returns a PointStruct with matching fields."""
        from db.qdrant.client import EmbeddingPoint
        from qdrant_client.models import PointStruct
        pid = str(uuid.uuid4())
        payload = {"asset": "AAPL", "source": "news"}
        p = EmbeddingPoint(vector=_zero_vector(), payload=payload, point_id=pid)
        struct = p.to_point_struct()
        assert isinstance(struct, PointStruct)
        assert struct.id == pid
        assert struct.payload == payload


# ---------------------------------------------------------------------------
# ensure_collections
# ---------------------------------------------------------------------------

class TestEnsureCollections:
    """Tests for ensure_collections()."""

    def test_creates_all_five_when_none_exist(self) -> None:
        """ensure_collections creates all 5 collections when starting fresh."""
        import db.qdrant.client as mod
        mock_client = _mock_qdrant_client(existing_collections=[])
        mod._client = None

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            created = mod.ensure_collections()

        assert len(created) == 5
        assert mock_client.create_collection.call_count == 5

    def test_skips_existing_collections(self) -> None:
        """ensure_collections does not recreate collections that already exist."""
        import db.qdrant.client as mod
        existing = ["asset_embeddings", "news_embeddings"]
        mock_client = _mock_qdrant_client(existing_collections=existing)

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            created = mod.ensure_collections()

        assert len(created) == 3
        assert mock_client.create_collection.call_count == 3

    def test_idempotent_when_all_exist(self) -> None:
        """ensure_collections does nothing when all 5 collections already exist."""
        import db.qdrant.client as mod
        all_names = [cfg.name for cfg in ALL_COLLECTIONS]
        mock_client = _mock_qdrant_client(existing_collections=all_names)

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            created = mod.ensure_collections()

        assert created == []
        mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# upsert_points
# ---------------------------------------------------------------------------

class TestUpsertPoints:
    """Tests for upsert_points()."""

    def test_upserts_batch(self) -> None:
        """upsert_points calls client.upsert with the correct collection name."""
        from db.qdrant.client import EmbeddingPoint, upsert_points

        mock_client = _mock_qdrant_client()
        points = [
            EmbeddingPoint(vector=_zero_vector(), payload={"asset": "AAPL"}),
            EmbeddingPoint(vector=_zero_vector(), payload={"asset": "MSFT"}),
        ]

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            result = upsert_points("asset_embeddings", points)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "asset_embeddings"
        assert len(call_kwargs["points"]) == 2

    def test_empty_list_skips_upsert(self) -> None:
        """upsert_points does not call client.upsert for an empty list."""
        from db.qdrant.client import upsert_points

        mock_client = _mock_qdrant_client()

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            upsert_points("asset_embeddings", [])

        mock_client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests for search()."""

    def test_calls_client_search_with_params(self) -> None:
        """search() passes collection name, query vector and limit to client."""
        from db.qdrant.client import search

        mock_client = _mock_qdrant_client()
        query = _zero_vector()

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            results = search("news_embeddings", query, top_k=5)

        mock_client.search.assert_called_once()
        kwargs = mock_client.search.call_args[1]
        assert kwargs["collection_name"] == "news_embeddings"
        assert kwargs["limit"] == 5
        assert kwargs["query_vector"] == query

    def test_returns_empty_list_on_no_results(self) -> None:
        """search() returns an empty list when client returns no results."""
        from db.qdrant.client import search

        mock_client = _mock_qdrant_client()
        mock_client.search.return_value = []

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            results = search("macro_embeddings", _zero_vector())

        assert results == []

    def test_score_threshold_passed(self) -> None:
        """search() includes score_threshold in the client call when provided."""
        from db.qdrant.client import search

        mock_client = _mock_qdrant_client()

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            search("signal_embeddings", _zero_vector(), score_threshold=0.8)

        kwargs = mock_client.search.call_args[1]
        assert kwargs["score_threshold"] == 0.8

    def test_payload_filter_builds_qdrant_filter(self) -> None:
        """search() constructs a Qdrant Filter when filter_payload is provided."""
        from db.qdrant.client import search

        mock_client = _mock_qdrant_client()

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            search("asset_embeddings", _zero_vector(), filter_payload={"asset_class": "equity"})

        kwargs = mock_client.search.call_args[1]
        assert "query_filter" in kwargs


# ---------------------------------------------------------------------------
# fetch_by_id
# ---------------------------------------------------------------------------

class TestFetchById:
    """Tests for fetch_by_id()."""

    def test_returns_none_when_not_found(self) -> None:
        """fetch_by_id returns None when the client returns empty results."""
        from db.qdrant.client import fetch_by_id

        mock_client = _mock_qdrant_client()
        mock_client.retrieve.return_value = []

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            result = fetch_by_id("asset_embeddings", str(uuid.uuid4()))

        assert result is None

    def test_returns_dict_when_found(self) -> None:
        """fetch_by_id returns a dict with id, vector, and payload keys."""
        from db.qdrant.client import fetch_by_id

        pid = str(uuid.uuid4())
        mock_point = MagicMock()
        mock_point.id = pid
        mock_point.vector = _zero_vector()
        mock_point.payload = {"asset": "BTC"}

        mock_client = _mock_qdrant_client()
        mock_client.retrieve.return_value = [mock_point]

        with patch("db.qdrant.client.get_client", return_value=mock_client):
            result = fetch_by_id("asset_embeddings", pid)

        assert result is not None
        assert result["id"] == pid
        assert result["payload"] == {"asset": "BTC"}


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

class TestClientSingleton:
    """Tests for get_client() lazy initialisation."""

    def test_raises_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_client raises RuntimeError when QDRANT_URL is not set."""
        import db.qdrant.client as mod
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        mod._client = None

        import sys
        for key in list(sys.modules.keys()):
            if key == "skills.env":
                del sys.modules[key]

        with pytest.raises(RuntimeError):
            mod.get_client()
