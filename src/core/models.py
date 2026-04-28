"""
Domain models for the Clinical Trial Intelligence Platform.

All models use Pydantic v2 with strict typing — this is what you'd see
in a regulated environment where schema drift causes downstream failures.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


# ─── Enums ────────────────────────────────────────────────────────────────────

class TrialPhase(StrEnum):
    PHASE_1   = "Phase 1"
    PHASE_2   = "Phase 2"
    PHASE_3   = "Phase 3"
    PHASE_4   = "Phase 4"
    NOT_APPLICABLE = "N/A"


class TrialStatus(StrEnum):
    RECRUITING         = "RECRUITING"
    COMPLETED          = "COMPLETED"
    TERMINATED         = "TERMINATED"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    WITHDRAWN          = "WITHDRAWN"
    UNKNOWN            = "UNKNOWN"


class DocumentSource(StrEnum):
    PUBMED           = "pubmed"
    CLINICALTRIALS   = "clinicaltrials"
    FDA              = "fda"


# ─── Base ─────────────────────────────────────────────────────────────────────

class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Ingestion models ─────────────────────────────────────────────────────────

class RawDocument(TimestampedModel):
    """Raw document as fetched from external source, before processing."""
    id: UUID = Field(default_factory=uuid4)
    source: DocumentSource
    external_id: str              # e.g. PubMed PMID or NCT number
    title: str
    abstract: str | None = None
    full_text: str | None = None
    url: HttpUrl | None = None
    published_at: datetime | None = None
    raw_metadata: dict = Field(default_factory=dict)


class ProcessedChunk(BaseModel):
    """Chunked + embedded document fragment ready for vector storage."""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    text: str
    embedding: list[float] | None = None   # populated after embedding step
    metadata: dict = Field(default_factory=dict)


# ─── Extraction models (structured output from LLM) ───────────────────────────

class Endpoint(BaseModel):
    name: str
    type: str   # "primary" | "secondary" | "exploratory"
    measurement: str | None = None
    timeframe: str | None = None


class PopulationCriteria(BaseModel):
    inclusion: list[str] = Field(default_factory=list)
    exclusion: list[str] = Field(default_factory=list)
    age_range: str | None = None
    sample_size: int | None = None


class ExtractedTrialData(BaseModel):
    """
    Structured data extracted by the Extraction Agent.
    Pydantic enforces schema — if the LLM hallucinates a field, it fails loudly.
    """
    nct_number: str | None = None
    title: str
    phase: TrialPhase = TrialPhase.NOT_APPLICABLE
    status: TrialStatus = TrialStatus.UNKNOWN
    sponsor: str | None = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    population: PopulationCriteria = Field(default_factory=PopulationCriteria)
    start_date: str | None = None
    completion_date: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    source_citations: list[str] = Field(default_factory=list)


# ─── Report model ─────────────────────────────────────────────────────────────

class TrialReport(TimestampedModel):
    """Final output from the Summary Agent."""
    id: UUID = Field(default_factory=uuid4)
    query: str
    executive_summary: str
    key_findings: list[str]
    trials_analyzed: list[ExtractedTrialData]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "LLM outputs are not a substitute for clinical expert review.",
            "Source data accuracy depends on upstream ClinicalTrials.gov / PubMed data.",
        ]
    )
    generated_by_model: str
    langfuse_trace_id: str | None = None


# ─── API request / response ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(min_length=5, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    report: TrialReport
    processing_time_ms: int
