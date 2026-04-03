import re
from typing import Any, Dict, List

import httpx

from app.core.config import settings
from app.models.context import DocumentSection
from app.utils.logger import logger


class WebResearchService:
    """Best-effort web research helper for evidence-aware review."""

    @staticmethod
    async def gather_supporting_sources(
        content: str,
        sections: List[DocumentSection],
        max_queries: int | None = None,
    ) -> Dict[str, Any]:
        queries = WebResearchService._build_queries(content, sections)
        limit = max_queries or settings.WEB_RESEARCH_MAX_QUERIES
        query_results: List[Dict[str, Any]] = []

        if not settings.ENABLE_WEB_RESEARCH:
            return {"enabled": False, "queries": queries[:limit], "results": []}

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            for query in queries[:limit]:
                result = await WebResearchService._search_query(client, query)
                query_results.append(result)

        return {"enabled": True, "queries": queries[:limit], "results": query_results}

    @staticmethod
    async def _search_query(client: httpx.AsyncClient, query: str) -> Dict[str, Any]:
        sources: List[Dict[str, str]] = []
        errors: List[str] = []

        try:
            ddg_response = await client.get(
                settings.DUCKDUCKGO_API_URL,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            ddg_response.raise_for_status()
            payload = ddg_response.json()

            abstract_url = payload.get("AbstractURL")
            if abstract_url:
                sources.append(
                    {
                        "title": payload.get("Heading") or query,
                        "url": abstract_url,
                        "snippet": payload.get("AbstractText") or "",
                        "source": "duckduckgo",
                    }
                )

            for topic in payload.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("FirstURL"):
                    sources.append(
                        {
                            "title": topic.get("Text", query),
                            "url": topic["FirstURL"],
                            "snippet": topic.get("Text", ""),
                            "source": "duckduckgo",
                        }
                    )
        except Exception as exc:
            errors.append(f"duckduckgo: {exc}")

        try:
            crossref_response = await client.get(
                settings.CROSSREF_API_URL,
                params={"query.title": query, "rows": 3, "select": "DOI,title,URL,published"},
                headers={"User-Agent": settings.RESEARCH_USER_AGENT},
            )
            crossref_response.raise_for_status()
            items = crossref_response.json().get("message", {}).get("items", [])
            for item in items:
                title = " ".join(item.get("title", [])[:1]).strip() or query
                sources.append(
                    {
                        "title": title,
                        "url": item.get("URL") or "",
                        "snippet": item.get("DOI", ""),
                        "source": "crossref",
                    }
                )
        except Exception as exc:
            errors.append(f"crossref: {exc}")

        deduped: List[Dict[str, str]] = []
        seen_urls = set()
        for source in sources:
            url = source.get("url") or source.get("title")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(source)

        return {"query": query, "sources": deduped[:5], "errors": errors}

    @staticmethod
    def _build_queries(content: str, sections: List[DocumentSection]) -> List[str]:
        heading_queries = [
            section.heading
            for section in sections
            if section.heading and section.heading != "Document" and len(section.heading.split()) >= 2
        ]

        sentence_queries: List[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            cleaned = sentence.strip()
            if len(cleaned.split()) < 6 or len(cleaned) > 140:
                continue
            if re.search(r"\b(studies|research|data|evidence|survey|according to)\b", cleaned, re.IGNORECASE):
                sentence_queries.append(cleaned)

        combined = heading_queries + sentence_queries
        if not combined and content.strip():
            combined = [" ".join(content.split()[:12])]

        unique: List[str] = []
        seen = set()
        for query in combined:
            normalized = query.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(query)
        return unique
