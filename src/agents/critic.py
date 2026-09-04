"""Critic Agent: reviews research findings for contradictions, weak sourcing,
and gaps, and can request one additional round of research."""
import json
import re

from ..llm_client import LLMClient, LLMResponse
from ..models import ResearchFinding, CriticReport

SYSTEM_PROMPT = """You are a rigorous fact-checking critic reviewing a set of
research findings before they go into a report. Check for: (1) internal
contradictions between findings, (2) claims backed by only one weak source,
(3) gaps in coverage of the original question. Respond with ONLY a JSON
object of the form:
{"issues": ["..."], "needs_more_research": true/false,
 "additional_questions": ["..."], "confidence": 0.0-1.0}"""


class CriticAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, query: str, findings: list[ResearchFinding]) -> tuple[CriticReport, LLMResponse]:
        findings_block = "\n\n".join(
            f"Sub-question: {f.subtask}\nFinding: {f.summary}\n"
            f"Source count: {len(f.sources)}"
            for f in findings
        )
        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=f"Original question: {query}\n\nFindings:\n{findings_block}",
            max_tokens=500,
        )
        report = self._parse(resp.text)
        return report, resp

    @staticmethod
    def _parse(text: str) -> CriticReport:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return CriticReport()
        try:
            data = json.loads(match.group(0))
            return CriticReport(
                issues=data.get("issues", []),
                needs_more_research=bool(data.get("needs_more_research", False)),
                additional_questions=data.get("additional_questions", []),
                confidence=float(data.get("confidence", 1.0)),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return CriticReport()
