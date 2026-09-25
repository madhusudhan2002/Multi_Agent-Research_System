"""
Web search tool used by the Researcher agent.

Uses DuckDuckGo for live web search without requiring
a Tavily API key.

If DuckDuckGo temporarily fails, the search is retried
with a simplified query. If no results are available,
an empty result list is returned rather than fabricated
sources.
"""

import time
import re
from typing import List

from .models import SearchResult


class SearchTool:
    """
    Web search tool.

    Primary:
        DuckDuckGo via the ddgs package

    Fallback:
        Retry with a simplified query.

    Important:
        No fabricated/mock sources are returned.
    """

    def __init__(self):
        self.live = True

    def search(
        self,
        query: str,
        max_results: int = 4
    ) -> List[SearchResult]:

        queries = [
            query,
            self._simplify_query(query),
        ]

        last_error = None

        for attempt, search_query in enumerate(queries, start=1):

            if not search_query.strip():
                continue

            try:
                from ddgs import DDGS

                results = []

                with DDGS() as ddgs:

                    search_results = ddgs.text(
                        search_query,
                        max_results=max_results
                    )

                    for result in search_results:

                        title = result.get(
                            "title",
                            "Untitled"
                        )

                        url = result.get(
                            "href",
                            ""
                        )

                        snippet = result.get(
                            "body",
                            ""
                        )

                        if not url and not snippet:
                            continue

                        results.append(
                            SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet[:400]
                            )
                        )

                if results:
                    return results

                print(
                    f"Web search returned no results "
                    f"(attempt {attempt}/{len(queries)}): "
                    f"{search_query}"
                )

            except Exception as e:

                last_error = e

                print(
                    f"Web search failed "
                    f"(attempt {attempt}/{len(queries)}): "
                    f"{type(e).__name__}: {e}"
                )

                if attempt < len(queries):
                    time.sleep(1)

        if last_error:
            print(
                "Web search unavailable after retries. "
                "No fabricated sources will be returned."
            )
        else:
            print(
                "Web search returned no results after retries."
            )

        return []

    @staticmethod
    def _simplify_query(query: str) -> str:
        """
        Create a shorter search query for retrying
        when the original query returns no results.
        """

        cleaned = re.sub(
            r"[^\w\s-]",
            " ",
            query
        )

        words = cleaned.split()

        # Keep the most useful portion of a long query.
        if len(words) > 12:
            words = words[:12]

        return " ".join(words)