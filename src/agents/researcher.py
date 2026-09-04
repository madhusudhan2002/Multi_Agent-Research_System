"""Researcher Agent: searches the web and summarizes findings with sources."""
from ..llm_client import LLMClient, LLMResponse
from ..search_tool import SearchTool
from ..models import ResearchFinding

SYSTEM_PROMPT = """You are a research assistant. You are given a sub-question
and a set of web search results. Write a concise, factual 3-5 sentence
summary that directly answers the sub-question, based ONLY on the provided
sources. If sources disagree, say so explicitly. Do not invent facts that
are not supported by the sources."""


class ResearcherAgent:
    def __init__(self, llm: LLMClient, search: SearchTool):
        self.llm = llm
        self.search = search

    def research(self, subtask: str) -> tuple[ResearchFinding, LLMResponse]:
        sources = self.search.search(subtask)
        sources_block = "\n".join(
            f"- {s.title} ({s.url}): {s.snippet}" for s in sources
        ) or "No sources found."

        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=f"Sub-question: {subtask}\n\nSources:\n{sources_block}",
            max_tokens=500,
        )
        finding = ResearchFinding(subtask=subtask, summary=resp.text, sources=sources)
        return finding, resp
