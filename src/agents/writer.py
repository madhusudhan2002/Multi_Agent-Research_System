"""
Writer Agent.

Combines research findings and critic feedback into
a clean, professional Markdown research report.
"""

import re

from ..llm_client import LLMClient, LLMResponse
from ..models import ResearchFinding, CriticReport


SYSTEM_PROMPT = """
You are a professional research report writer.

Your job is to convert research findings into a clear,
accurate and professional Markdown report.

IMPORTANT MARKDOWN RULES:

1. Return VALID MARKDOWN only.

2. Use normal Markdown headings:

# Main Title

## Section

### Subsection

3. NEVER write headings like:

**# Heading**
**## Heading**

4. NEVER wrap the entire report in bold.

5. Use normal Markdown paragraphs.

6. Use bullet lists like:

- Item one
- Item two

7. Use numbered lists like:

1. First
2. Second

8. Use normal Markdown tables when useful.

Example:

| Topic | Finding |
|---|---|
| Example | Example result |

9. Use normal Markdown links:

[Source Name](https://example.com)

10. Do NOT escape Markdown characters unnecessarily.

11. Do NOT put the entire report inside a code block.

12. Do NOT use triple backticks around the report.

13. Do not invent sources.

14. Do not invent statistics.

15. Clearly distinguish research findings from uncertain claims.

16. When sources disagree, mention the disagreement.

17. Keep the report factual and evidence-based.

Recommended structure:

# Title

## Executive Summary

## Current State

## Key Findings

## Risks and Limitations

## Analysis

## Conclusion

## Sources

The report should be concise but sufficiently detailed.
"""


class WriterAgent:

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def write(
        self,
        query: str,
        findings: list[ResearchFinding],
        critic_report: CriticReport
    ) -> tuple[str, LLMResponse]:

        # ---------------------------------------------------------
        # Build research block
        # ---------------------------------------------------------

        research_sections = []

        for finding in findings:

            source_lines = []

            for source in finding.sources:

                source_lines.append(
                    f"- {source.title} — {source.url}\n"
                    f"  {source.snippet}"
                )

            sources_text = "\n".join(source_lines)

            research_sections.append(
                f"""
SUB-QUESTION:
{finding.subtask}

RESEARCH SUMMARY:
{finding.summary}

SOURCES:
{sources_text if sources_text else "No sources available."}
"""
            )

        research_block = "\n".join(
            research_sections
        )

        # ---------------------------------------------------------
        # Critic information
        # ---------------------------------------------------------

        issues_text = "\n".join(
            f"- {issue}"
            for issue in critic_report.issues
        )

        questions_text = "\n".join(
            f"- {question}"
            for question in critic_report.additional_questions
        )

        # ---------------------------------------------------------
        # Writer prompt
        # ---------------------------------------------------------

        prompt = f"""
Original research question:

{query}


RESEARCH FINDINGS:

{research_block}


CRITIC REVIEW:

Issues:
{issues_text if issues_text else "None identified."}

Needs more research:
{critic_report.needs_more_research}

Additional questions:
{questions_text if questions_text else "None."}

Critic confidence:
{critic_report.confidence:.2f}


TASK:

Create a professional research report answering the original
research question.

Use the available research findings and sources.

Do not invent facts.

Do not invent statistics.

Do not invent URLs.

If a statistic or claim comes from a source, attribute it
appropriately.

Return ONLY the final Markdown report.

Remember:

- Use # and ## headings directly.
- Never bold headings.
- Never wrap the whole report in **.
- Use normal Markdown tables.
- Use normal Markdown links.
- Do not use code fences around the report.
"""

        # ---------------------------------------------------------
        # Generate report
        # ---------------------------------------------------------

        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=prompt,
            max_tokens=2500
        )

        report = response.text.strip()

        # ---------------------------------------------------------
        # Clean accidental Markdown formatting
        # ---------------------------------------------------------

        report = self._clean_markdown(report)

        return report, response

    # -------------------------------------------------------------
    # Markdown cleanup
    # -------------------------------------------------------------

    @staticmethod
    def _clean_markdown(text: str) -> str:

        # Remove accidental code fences surrounding entire report
        text = re.sub(
            r"^\s*```markdown\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^\s*```\s*$",
            "",
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------------------------------
        # Fix:
        # **# Heading**
        # **## Heading**
        # **### Heading**
        # ---------------------------------------------------------

        text = re.sub(
            r"^\s*\*\*(#{1,6})\s*(.*?)\*\*\s*$",
            r"\1 \2",
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------------------------------
        # Fix headings where only the beginning was bolded:
        #
        # **# Heading
        # ---------------------------------------------------------

        text = re.sub(
            r"^\s*\*\*(#{1,6})\s*(.*?)\s*$",
            r"\1 \2",
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------------------------------
        # Fix headings where trailing bold remains:
        #
        # # Heading**
        # ---------------------------------------------------------

        text = re.sub(
            r"^(#{1,6}\s+.*?)\*\*\s*$",
            r"\1",
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------------------------------
        # Fix bold horizontal separators:
        #
        # **---**
        # ---------------------------------------------------------

        text = re.sub(
            r"^\s*\*\*\s*---+\s*\*\*\s*$",
            "---",
            text,
            flags=re.MULTILINE
        )

        # ---------------------------------------------------------
        # Remove excessive blank lines
        # ---------------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()