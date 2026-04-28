"""
FastAPI application entry point.

Demonstrates production patterns:
- Lifespan context for resource management (no global state)
- Structured audit logging on every request
- Typed request/response with Pydantic v2
- Proper HTTP error codes, not bare 500s
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agents.orchestrator import AgentOrchestrator
from src.core.config import settings
from src.core.models import QueryRequest, QueryResponse
from src.ingestion.pipeline import IngestionPipeline
from src.storage.postgres_store import PostgresStore
from src.storage.qdrant_store import QdrantStore

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown logic.
    Resources are initialised once and injected — no singletons, no globals.
    """
    qdrant   = QdrantStore()
    postgres = PostgresStore()

    await qdrant.ensure_collection()
    await postgres.init_schema()

    app.state.qdrant    = qdrant
    app.state.postgres  = postgres
    app.state.ingestion = IngestionPipeline(qdrant, postgres)
    app.state.agents    = AgentOrchestrator(qdrant)

    logger.info("Services initialised. Environment: %s", settings.environment)
    yield

    # Shutdown — close connections
    await postgres.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Clinical Trial Intelligence Platform",
    description="AI-powered clinical trial analysis. RAG + multi-agent pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Middleware: request audit log ────────────────────────────────────────────

@app.middleware("http")
async def audit_log(request: Request, call_next):
    """
    Logs every request with a correlation ID.
    In a regulated environment this goes to an immutable audit store.
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%d",
        request_id, request.method, request.url.path,
        response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(request: Request, query: str, max_per_source: int = 10):
    """
    Trigger ingestion for a search query.
    Fetches from PubMed + ClinicalTrials.gov, chunks, embeds, stores.
    """
    pipeline: IngestionPipeline = request.app.state.ingestion
    result = await pipeline.ingest_query(query, max_per_source=max_per_source)
    return result


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Run the multi-agent pipeline on a natural language query.

    Returns a structured TrialReport with citations and confidence scores.
    """
    orchestrator: AgentOrchestrator = request.app.state.agents
    start = time.perf_counter()

    try:
        report = await orchestrator.run(body)
    except Exception as exc:
        logger.exception("Agent pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution failed. Check logs for details.",
        ) from exc

    duration_ms = int((time.perf_counter() - start) * 1000)
    return QueryResponse(report=report, processing_time_ms=duration_ms)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all — never leak stack traces to clients."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
