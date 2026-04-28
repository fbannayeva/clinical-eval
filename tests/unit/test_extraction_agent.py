"""
Unit tests for ExtractionAgent.

Tests the JSON parsing and Pydantic validation in isolation —
no real LLM calls, no infrastructure needed.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator import ExtractionAgent
from src.core.models import ExtractedTrialData, TrialPhase, TrialStatus


def make_mock_llm(response_json: dict) -> AsyncMock:
    """Build a mock Anthropic client that returns given JSON."""
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_json)

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    mock_llm = AsyncMock()
    mock_llm.messages.create = AsyncMock(return_value=mock_response)
    return mock_llm


VALID_EXTRACTION = {
    "nct_number": "NCT04702958",
    "title": "A Phase 3 Study of Sotorasib in KRAS G12C NSCLC",
    "phase": "Phase 3",
    "status": "COMPLETED",
    "sponsor": "Amgen",
    "conditions": ["Non-Small Cell Lung Cancer"],
    "interventions": ["Sotorasib 960mg QD"],
    "endpoints": [
        {
            "name": "Progression-Free Survival",
            "type": "primary",
            "measurement": "Months",
            "timeframe": "24 months",
        }
    ],
    "population": {
        "inclusion": ["KRAS G12C mutation confirmed", "ECOG PS 0-2"],
        "exclusion": ["Prior KRAS inhibitor treatment"],
        "age_range": "18+",
        "sample_size": 345,
    },
    "start_date": "2021-01",
    "completion_date": "2023-06",
    "confidence_score": 0.92,
    "source_citations": [],
}


@pytest.mark.asyncio
async def test_extraction_agent_happy_path():
    agent = ExtractionAgent(llm=make_mock_llm(VALID_EXTRACTION), trace=None)
    chunks = [{"text": "Dummy context", "external_id": "NCT04702958"}]

    result = await agent.extract(chunks)

    assert isinstance(result, ExtractedTrialData)
    assert result.nct_number == "NCT04702958"
    assert result.phase == TrialPhase.PHASE_3
    assert result.status == TrialStatus.COMPLETED
    assert result.confidence_score == 0.92
    assert result.population.sample_size == 345


@pytest.mark.asyncio
async def test_extraction_agent_handles_malformed_json():
    """If the LLM returns garbage, we return None — never crash."""
    mock_content = MagicMock()
    mock_content.text = "This is not JSON at all"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_llm = AsyncMock()
    mock_llm.messages.create = AsyncMock(return_value=mock_response)

    agent = ExtractionAgent(llm=mock_llm, trace=None)
    result = await agent.extract([{"text": "some text", "external_id": "UNKNOWN"}])

    assert result is None


@pytest.mark.asyncio
async def test_extraction_agent_confidence_score_bounds():
    """Pydantic should reject confidence_score > 1.0."""
    bad_data = {**VALID_EXTRACTION, "confidence_score": 1.5}
    agent = ExtractionAgent(llm=make_mock_llm(bad_data), trace=None)

    result = await agent.extract([{"text": "text", "external_id": "X"}])
    assert result is None   # Pydantic validation error → caught → None


@pytest.mark.asyncio
async def test_extraction_agent_source_citations_injected():
    """Source IDs from chunks should always be injected into the result."""
    agent = ExtractionAgent(llm=make_mock_llm(VALID_EXTRACTION), trace=None)
    chunks = [
        {"text": "chunk 1", "external_id": "NCT04702958"},
        {"text": "chunk 2", "external_id": "NCT04702958"},
        {"text": "chunk 3", "external_id": "PMD12345678"},
    ]

    result = await agent.extract(chunks)
    assert result is not None
    assert "NCT04702958" in result.source_citations
    assert "PMD12345678" in result.source_citations
