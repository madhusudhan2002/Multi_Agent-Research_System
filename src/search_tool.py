"""
Web search tool used by the Researcher agent.
Uses Tavily (https://tavily.com) if TAVILY_API_KEY is set, otherwise
returns deterministic mock results so the pipeline runs fully offline.
"""
import requests
from typing import List

from .config import settings
from .models import SearchResult


class SearchTool:
    def __init__(self):
        self.live = settings.LIVE_SEARCH

    def search(self, query: str, max_results: int = 4) -> List[SearchResult]:
        if not self.live:
            return self._mock_search(query, max_results)

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.TAVILY_API_KEY,
                    "query": query,
                    "max_results": max_results,
                },
                timeout=settings.REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                SearchResult(
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")[:400],
                )
                for r in data.get("results", [])[:max_results]
            ]
        except Exception:
            # Fail soft to mock results rather than crashing the pipeline
            return self._mock_search(query, max_results)

    @staticmethod
    def _mock_search(query: str, max_results: int) -> List[SearchResult]:
        return [
            SearchResult(
                title=f"[MOCK] Source {i+1} on: {query[:60]}",
                url=f"https://example.com/mock-source-{i+1}",
                snippet=(
                    "This is a simulated search result used because no "
                    "TAVILY_API_KEY was configured. Add one in .env for real "
                    "web search results."
                ),
            )
            for i in range(min(max_results, 3))
        ]
