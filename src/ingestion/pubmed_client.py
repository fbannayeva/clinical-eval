"""
PubMed ingestion client.

Uses NCBI E-utilities API. Handles pagination, rate limiting (3 req/s without
API key, 10/s with). Returns typed RawDocument objects — no dict soup.
"""
from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from src.core.config import settings
from src.core.models import DocumentSource, RawDocument

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedClient:
    """
    Async client for PubMed E-utilities.

    Usage:
        async with PubMedClient() as client:
            docs = await client.search("KRAS G12C inhibitor clinical trial", max_results=20)
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._rate_limit = asyncio.Semaphore(3 if not settings.pubmed_api_key else 10)

    async def __aenter__(self) -> "PubMedClient":
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def search(self, query: str, max_results: int = 20) -> list[RawDocument]:
        """Search PubMed and return typed RawDocument list."""
        pmids = await self._esearch(query, max_results)
        if not pmids:
            logger.info("No results for query: %s", query)
            return []

        logger.info("Fetching %d articles for query: %s", len(pmids), query)
        docs = await asyncio.gather(*[self._efetch(pmid) for pmid in pmids])
        return [d for d in docs if d is not None]

    async def _esearch(self, query: str, max_results: int) -> list[str]:
        """Return list of PMIDs matching query."""
        async with self._rate_limit:
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "usehistory": "y",
            }
            if settings.pubmed_api_key:
                params["api_key"] = settings.pubmed_api_key

            resp = await self._client.get(ESEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data["esearchresult"]["idlist"]

    async def _efetch(self, pmid: str) -> RawDocument | None:
        """Fetch and parse a single article by PMID."""
        async with self._rate_limit:
            await asyncio.sleep(0.1)   # polite rate limiting

            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "rettype": "abstract",
            }
            if settings.pubmed_api_key:
                params["api_key"] = settings.pubmed_api_key

            try:
                resp = await self._client.get(EFETCH_URL, params=params)
                resp.raise_for_status()
                return self._parse_article_xml(pmid, resp.text)
            except Exception as exc:
                logger.warning("Failed to fetch PMID %s: %s", pmid, exc)
                return None

    def _parse_article_xml(self, pmid: str, xml_text: str) -> RawDocument | None:
        """Parse PubMed XML into a RawDocument."""
        try:
            root = ET.fromstring(xml_text)
            article = root.find(".//PubmedArticle")
            if article is None:
                return None

            title = self._get_text(article, ".//ArticleTitle") or "Untitled"
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.get("Label", "") + ": " if p.get("Label") else "") + (p.text or "")
                for p in abstract_parts
            ).strip()

            pub_date = self._extract_date(article)

            return RawDocument(
                source=DocumentSource.PUBMED,
                external_id=pmid,
                title=title,
                abstract=abstract or None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",  # type: ignore[arg-type]
                published_at=pub_date,
                raw_metadata={"pmid": pmid},
            )
        except ET.ParseError as exc:
            logger.warning("XML parse error for PMID %s: %s", pmid, exc)
            return None

    @staticmethod
    def _get_text(element: ET.Element, path: str) -> str | None:
        node = element.find(path)
        return node.text if node is not None else None

    @staticmethod
    def _extract_date(article: ET.Element) -> datetime | None:
        year  = article.findtext(".//PubDate/Year")
        month = article.findtext(".//PubDate/Month") or "01"
        day   = article.findtext(".//PubDate/Day") or "01"
        if year:
            try:
                return datetime.strptime(f"{year}-{month[:3]}-{day}", "%Y-%b-%d")
            except ValueError:
                try:
                    return datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                except ValueError:
                    return datetime(int(year), 1, 1)
        return None
