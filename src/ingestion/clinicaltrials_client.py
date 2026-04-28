"""
ClinicalTrials.gov API v2 client.

The CTG API v2 (2023+) returns JSON — much cleaner than the old XML API.
We map NCT studies to RawDocument for uniform downstream processing.
"""
from __future__ import annotations

import logging

import httpx

from src.core.config import settings
from src.core.models import DocumentSource, RawDocument, TrialStatus

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    """
    Async client for ClinicalTrials.gov API v2.

    Usage:
        async with ClinicalTrialsClient() as client:
            docs = await client.search("KRAS non-small cell lung cancer", max_results=10)
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        self._client = httpx.AsyncClient(
            base_url=settings.clinicaltrials_base_url,
            timeout=30.0,
            headers={"User-Agent": "ClinicalTrialIntelligencePlatform/1.0"},
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        status_filter: list[TrialStatus] | None = None,
    ) -> list[RawDocument]:
        """Search trials and return typed RawDocument list."""
        params: dict = {
            "query.term": query,
            "pageSize": min(max_results, 100),
            "format": "json",
            "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
                      "OverallStatus,Phase,StartDate,CompletionDate,LeadSponsorName,"
                      "Condition,InterventionName,EnrollmentCount",
        }
        if status_filter:
            params["filter.overallStatus"] = "|".join(s.value for s in status_filter)

        try:
            resp = await self._client.get("/studies", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("ClinicalTrials API error: %s", exc)
            return []

        studies = data.get("studies", [])
        logger.info("Fetched %d trials for query: %s", len(studies), query)
        return [self._parse_study(s) for s in studies]

    def _parse_study(self, study: dict) -> RawDocument:
        proto = study.get("protocolSection", {})
        id_module       = proto.get("identificationModule", {})
        desc_module     = proto.get("descriptionModule", {})
        status_module   = proto.get("statusModule", {})
        design_module   = proto.get("designModule", {})
        sponsor_module  = proto.get("sponsorCollaboratorsModule", {})
        conditions_module = proto.get("conditionsModule", {})
        interventions_module = proto.get("armsInterventionsModule", {})

        nct_id = id_module.get("nctId", "UNKNOWN")
        title  = id_module.get("officialTitle") or id_module.get("briefTitle", "Untitled")
        abstract = desc_module.get("briefSummary") or desc_module.get("detailedDescription")

        # Build rich metadata — extracted by the Extraction Agent later
        metadata = {
            "nct_id":        nct_id,
            "status":        status_module.get("overallStatus"),
            "phase":         design_module.get("phases", []),
            "sponsor":       sponsor_module.get("leadSponsor", {}).get("name"),
            "conditions":    conditions_module.get("conditions", []),
            "interventions": [
                i.get("name") for i in interventions_module.get("interventions", [])
            ],
            "enrollment":    design_module.get("enrollmentInfo", {}).get("count"),
            "start_date":    status_module.get("startDateStruct", {}).get("date"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date"),
        }

        return RawDocument(
            source=DocumentSource.CLINICALTRIALS,
            external_id=nct_id,
            title=title,
            abstract=abstract,
            url=f"https://clinicaltrials.gov/study/{nct_id}",  # type: ignore[arg-type]
            raw_metadata=metadata,
        )
