"""
PostgreSQL/SQLite async store.

Uses SQLAlchemy 2.0 async. Works with both PostgreSQL (production)
and SQLite (local development without Docker).
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Integer, JSON,
    String, Text, select, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings
from src.core.models import DocumentSource, RawDocument, ProcessedChunk

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id           = Column(String, primary_key=True)
    source       = Column(String, nullable=False, index=True)
    external_id  = Column(String, nullable=False, index=True)
    title        = Column(Text, nullable=False)
    abstract     = Column(Text)
    url          = Column(String)
    published_at = Column(DateTime)
    raw_metadata = Column(JSON, default={})
    created_at   = Column(DateTime, default=datetime.utcnow)


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id             = Column(String, primary_key=True)
    document_id    = Column(String, nullable=False, index=True)
    chunk_index    = Column(Integer, nullable=False)
    text           = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, default={})
    created_at     = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    entity_id  = Column(String)
    details    = Column(JSON, default={})
    success    = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PostgresStore:

    def __init__(self) -> None:
        is_sqlite = settings.postgres_dsn.startswith("sqlite")

        if is_sqlite:
            self._engine = create_async_engine(
                settings.postgres_dsn,
                echo=settings.environment == "development",
                connect_args={"check_same_thread": False},
            )
        else:
            self._engine = create_async_engine(
                settings.postgres_dsn,
                echo=settings.environment == "development",
                pool_size=10,
                max_overflow=20,
            )

        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialised")

    async def close(self) -> None:
        await self._engine.dispose()

    async def document_exists(self, external_id: str, source: DocumentSource) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentRecord.id).where(
                    DocumentRecord.external_id == external_id,
                    DocumentRecord.source == source.value,
                )
            )
            return result.scalar_one_or_none() is not None

    async def save_document(self, doc: RawDocument) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(DocumentRecord(
                    id=str(doc.id),
                    source=doc.source.value,
                    external_id=doc.external_id,
                    title=doc.title,
                    abstract=doc.abstract,
                    url=str(doc.url) if doc.url else None,
                    published_at=doc.published_at,
                    raw_metadata=doc.raw_metadata,
                    created_at=doc.created_at,
                ))
                session.add(AuditLog(
                    event_type="INGEST",
                    entity_id=doc.external_id,
                    details={
                        "source": doc.source.value,
                        "title": doc.title[:100],
                        "document_id": str(doc.id),
                    },
                ))
        logger.debug("Saved document %s (%s)", doc.external_id, doc.source)

    async def save_chunks(self, chunks: list[ProcessedChunk]) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session.add_all([
                    ChunkRecord(
                        id=str(c.id),
                        document_id=str(c.document_id),
                        chunk_index=c.chunk_index,
                        text=c.text,
                        chunk_metadata=c.metadata,
                    )
                    for c in chunks
                ])

    async def log_query(self, request_id: str, query: str, success: bool, details: dict) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                session.add(AuditLog(
                    event_type="QUERY",
                    entity_id=request_id,
                    details={"query": query[:200], **details},
                    success=success,
                ))

    async def get_ingestion_stats(self) -> dict:
        async with self._session_factory() as session:
            doc_count   = await session.scalar(text("SELECT COUNT(*) FROM documents"))
            chunk_count = await session.scalar(text("SELECT COUNT(*) FROM chunks"))
            last_ingest = await session.scalar(
                text("SELECT MAX(created_at) FROM audit_log WHERE event_type = 'INGEST'")
            )
            return {
                "total_documents": doc_count or 0,
                "total_chunks": chunk_count or 0,
                "last_ingestion": str(last_ingest) if last_ingest else None,
            }