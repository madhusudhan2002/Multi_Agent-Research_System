"""Planner Agent: decomposes a research question into focused sub-tasks."""
import json
import re
from typing import List

from ..llm_client import LLMClient, LLMResponse
from ..config import settings

SYSTEM_PROMPT = """You are the Planner in a multi-agent research system.
Break the user's research question into 3-4 focused, non-overlapping
sub-questions that, together, fully cover the topic. Respond with ONLY a
JSON array of strings, nothing else."""


class PlannerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, query: str) -> tuple[List[str], LLMResponse]:
        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=f"Research question: {query}",
            max_tokens=400,
        )
        subtasks = self._parse_json_array(resp.text)
        subtasks = subtasks[: settings.MAX_RESEARCH_SUBTASKS]
        if not subtasks:
            # Fallback so the pipeline never dies on a parsing hiccup
            subtasks = [query]
        return subtasks, resp

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            return [str(item) for item in parsed if isinstance(item, str)]
        except json.JSONDecodeError:
            return []
