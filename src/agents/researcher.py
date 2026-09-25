"""
Researcher Agent.

Uses the external SearchTool for web research and
Gemini only for analyzing the search results.

The LLM itself does NOT have browser access.
"""

from ..llm_client import LLMClient, LLMResponse
from ..search_tool import SearchTool
from ..models import ResearchFinding


SYSTEM_PROMPT = """
You are a research assistant.

Your job is to analyze web search results that are
provided to you.

IMPORTANT:

You DO NOT have browser access.

You DO NOT have access to browser.open.

You DO NOT have access to external tools.

You MUST NOT attempt to call any tool.

You MUST NOT invent URLs.

You MUST use only the search results provided
inside the user prompt.

Write a concise research summary based on those
search results.

Clearly distinguish between:

- information directly supported by sources
- claims that appear uncertain
- missing information

Do not invent statistics.

Do not invent case studies.

Do not invent sources.

Return a factual research summary.
"""


class ResearcherAgent:

    def __init__(
        self,
        llm: LLMClient,
        search: SearchTool
    ):

        self.llm = llm
        self.search = search

    # =============================================================
    # RESEARCH
    # =============================================================

    def research(
        self,
        subtask: str
    ) -> tuple[ResearchFinding, LLMResponse]:

        # ---------------------------------------------------------
        # Search web
        # ---------------------------------------------------------

        search_results = self.search.search(
            subtask,
            max_results=5
        )

        # ---------------------------------------------------------
        # Build search context
        # ---------------------------------------------------------

        search_parts = []

        for i, result in enumerate(
            search_results,
            start=1
        ):

            search_parts.append(
                f"""
SOURCE {i}

Title:
{result.title}

URL:
{result.url}

Snippet:
{result.snippet}
"""
            )

        search_context = "\n".join(
            search_parts
        )

        # ---------------------------------------------------------
        # LLM prompt
        # ---------------------------------------------------------

        prompt = f"""
Research sub-question:

{subtask}


Web search results:

{search_context}


TASK:

Analyze the search results and produce a concise
research finding for the sub-question.

Important:

- Do not browse the web yourself.
- Do not call browser.open.
- Do not call any tool.
- Use only the sources supplied above.
- Do not invent information.
- Do not invent statistics.
- Do not invent URLs.
- If the sources are insufficient, explicitly say so.

Return only the research summary.
"""

        # ---------------------------------------------------------
        # Generate research summary
        # ---------------------------------------------------------

        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=900
        )

        summary = response.text.strip()

        # ---------------------------------------------------------
        # Empty response fallback
        # ---------------------------------------------------------

        if not summary:

            summary = (
                "The language model returned an empty "
                "research response. The available search "
                "results could not be summarized."
            )

        # ---------------------------------------------------------
        # Build ResearchFinding
        # ---------------------------------------------------------

        finding = ResearchFinding(
            subtask=subtask,
            summary=summary,
            sources=search_results
        )

        return finding, response