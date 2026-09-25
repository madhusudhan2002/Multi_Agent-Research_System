"""
Web search tool used by the Researcher agent.

Uses DuckDuckGo for live web search without requiring
a Tavily API key.

Falls back to deterministic mock results if the live
search fails.
"""

from typing import List

from .models import SearchResult


class SearchTool:
    """
    Web search tool.

    Primary:
        DuckDuckGo via the ddgs package

    Fallback:
        Deterministic mock results
    """

    def __init__(self):
        self.live = True

    def search(
        self,
        query: str,
        max_results: int = 4
    ) -> List[SearchResult]:

        try:
            from ddgs import DDGS

            results = []

            with DDGS() as ddgs:

                search_results = ddgs.text(
                    query,
                    max_results=max_results
                )

                for result in search_results:

                    results.append(
                        SearchResult(
                            title=result.get(
                                "title",
                                "Untitled"
                            ),
                            url=result.get(
                                "href",
                                ""
                            ),
                            snippet=result.get(
                                "body",
                                ""
                            )[:400]
                        )
                    )

            if results:
                return results

            return self._mock_search(
                query,
                max_results
            )

        except Exception as e:

            print(
                f"Web search failed: {e}"
            )

            return self._mock_search(
                query,
                max_results
            )

    @staticmethod
    def _mock_search(
        query: str,
        max_results: int
    ) -> List[SearchResult]:

        return [
            SearchResult(
                title=f"[MOCK] Source {i + 1} on: {query[:60]}",
                url=f"https://example.com/mock-source-{i + 1}",
                snippet=(
                    "This is a simulated search result because "
                    "live web search was unavailable."
                ),
            )
            for i in range(
                min(max_results, 3)
            )
        ]