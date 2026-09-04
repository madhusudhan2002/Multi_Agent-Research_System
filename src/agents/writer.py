"""Writer Agent: synthesizes all findings + critic feedback into a final,
cited Markdown report."""
from ..llm_client import LLMClient, LLMResponse
from ..models import ResearchFinding, CriticReport

SYSTEM_PROMPT = """You are a professional research report writer. Combine the
provided findings into a well-structured Markdown report with:
- A one-paragraph Executive Summary
- A section per sub-topic
- A closing "Caveats & Confidence" section reflecting the critic's feedback
- A "Sources" section listing all URLs
Be concise, factual, and do not add claims beyond what's in the findings."""


class WriterAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def write(
        self, query: str, findings: list[ResearchFinding], critic: CriticReport
    ) -> tuple[str, LLMResponse]:
        findings_block = "\n\n".join(
            f"### {f.subtask}\n{f.summary}\n"
            + "\n".join(f"- [{s.title}]({s.url})" for s in f.sources)
            for f in findings
        )
        critic_block = (
            f"Critic confidence: {critic.confidence}\n"
            f"Issues flagged: {critic.issues or 'None'}"
        )
        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=(
                f"Original research question: {query}\n\n"
                f"Findings:\n{findings_block}\n\n"
                f"Critic review:\n{critic_block}"
            ),
            max_tokens=1200,
        )
        return resp.text, resp
