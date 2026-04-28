"""
Qdrant vector store.

Wraps the Qdrant Python client with our domain types.
Collection is created with cosine distance — standard for sentence embeddings.
"""
from __future__ import annotations

import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.core.config import settings
from src.core.models import ProcessedChunk

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384   # all-MiniLM-L6-v2 output dimension


class QdrantStore:
    """
    Async Qdrant client wrapper.

    Handles collection init, upsert, and semantic search.
    Each chunk is stored as a point with its full metadata payload —
    no need to join back to Postgres for display purposes.
    """

    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
         url=settings.qdrant_url,
         api_key=settings.qdrant_api_key,
         timeout=30,
        )
        self._collection = settings.qdrant_collection
        
        

    async def ensure_collection(self) -> None:
        """Create collection if it doesn't exist. Idempotent."""
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self._collection)

    async def upsert_chunks(self, chunks: list[ProcessedChunk]) -> None:
        """Upsert chunk vectors with metadata payload."""
        points = [
            PointStruct(
                id=str(chunk.id),
                vector=chunk.embedding,
                payload={
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
            if chunk.embedding is not None
        ]
        if points:
            await self._client.upsert(collection_name=self._collection, points=points)
            logger.debug("Upserted %d points", len(points))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of payloads with scores.

        In a regulated context you'd also log: who searched, when, what vector.
        """
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {"score": r.score, **r.payload}
            for r in results
        ]
