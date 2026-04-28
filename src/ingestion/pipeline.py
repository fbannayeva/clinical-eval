"""
Ingestion pipeline.

Orchestrates: fetch → chunk → embed → store in Qdrant + Postgres.
This is Phase 1 core — everything downstream depends on clean data here.
"""
from __future__ import annotations

import logging
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.models import ProcessedChunk, RawDocument
from src.ingestion.clinicaltrials_client import ClinicalTrialsClient
from src.ingestion.pubmed_client import PubMedClient
from src.storage.qdrant_store import QdrantStore
from src.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline:
        1. Fetch from PubMed + ClinicalTrials.gov
        2. Split into semantic chunks
        3. Embed with SentenceTransformers
        4. Store vectors in Qdrant
        5. Store metadata in Postgres (with audit log)

    Designed to be idempotent — re-running with same external_id is a no-op.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        postgres: PostgresStore,
    ) -> None:
        self.qdrant = qdrant
        self.postgres = postgres
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " "],
        )

    async def ingest_query(self, query: str, max_per_source: int = 10) -> dict:
        """
        Fetch and ingest documents matching a query from all sources.
        Returns a summary of what was ingested.
        """
        logger.info("Starting ingestion for query: %s", query)

        pubmed_docs, ct_docs = [], []

        async with PubMedClient() as pubmed:
            pubmed_docs = await pubmed.search(query, max_results=max_per_source)

        async with ClinicalTrialsClient() as ct:
            ct_docs = await ct.search(query, max_results=max_per_source)

        all_docs = pubmed_docs + ct_docs
        logger.info("Fetched %d documents total", len(all_docs))

        ingested, skipped = 0, 0
        for doc in all_docs:
            already_exists = await self.postgres.document_exists(doc.external_id, doc.source)
            if already_exists:
                skipped += 1
                continue

            chunks = self._chunk_document(doc)
            chunks = self._embed_chunks(chunks)

            await self.qdrant.upsert_chunks(chunks)
            await self.postgres.save_document(doc)
            await self.postgres.save_chunks(chunks)
            ingested += 1

        logger.info("Ingestion complete: %d new, %d skipped", ingested, skipped)
        return {
            "query": query,
            "fetched": len(all_docs),
            "ingested": ingested,
            "skipped": skipped,
        }

    def _chunk_document(self, doc: RawDocument) -> list[ProcessedChunk]:
        """Split document text into overlapping chunks."""
        text = "\n\n".join(filter(None, [doc.title, doc.abstract, doc.full_text]))
        if not text.strip():
            return []

        raw_chunks = self._splitter.split_text(text)
        return [
            ProcessedChunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk,
                metadata={
                    "source": doc.source,
                    "external_id": doc.external_id,
                    "title": doc.title,
                    "url": str(doc.url) if doc.url else None,
                    **doc.raw_metadata,
                },
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def _embed_chunks(self, chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        """Compute embeddings in batch for efficiency."""
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, batch_size=32, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        return chunks
