"""
Critic Agent.

Performs deterministic quality checks on research findings.

Checks:
1. Empty findings
2. Missing sources
3. Source coverage
4. Research coverage
5. Potentially unsupported claims
6. Whether additional research may be needed

This version intentionally does NOT call the LLM.
"""

import re

from ..llm_client import LLMResponse
from ..models import ResearchFinding, CriticReport


class CriticAgent:

    def __init__(self, llm=None):
        """
        The LLM parameter is kept for compatibility with
        the existing Orchestrator.

        The current Critic does not need an LLM.
        """
        self.llm = llm

    # =============================================================
    # MAIN REVIEW
    # =============================================================

    def review(
        self,
        query: str,
        findings: list[ResearchFinding]
    ) -> tuple[CriticReport, LLMResponse]:

        issues = []
        additional_questions = []

        # ---------------------------------------------------------
        # Basic research statistics
        # ---------------------------------------------------------

        total_findings = len(findings)

        usable_findings = [
            finding
            for finding in findings
            if finding.summary
            and finding.summary.strip()
        ]

        findings_with_sources = [
            finding
            for finding in usable_findings
            if finding.sources
        ]

        total_sources = sum(
            len(finding.sources)
            for finding in usable_findings
        )

        # ---------------------------------------------------------
        # Check 1: Missing findings
        # ---------------------------------------------------------

        if total_findings == 0:

            issues.append(
                "No research findings were returned."
            )

            additional_questions.append(
                query
            )

        # ---------------------------------------------------------
        # Check 2: Empty findings
        # ---------------------------------------------------------

        empty_count = (
            total_findings
            - len(usable_findings)
        )

        if empty_count > 0:

            issues.append(
                f"{empty_count} research sub-task(s) "
                f"returned empty findings."
            )

        # ---------------------------------------------------------
        # Check 3: Missing sources
        # ---------------------------------------------------------

        findings_without_sources = (
            len(usable_findings)
            - len(findings_with_sources)
        )

        if findings_without_sources > 0:

            issues.append(
                f"{findings_without_sources} finding(s) "
                f"have no source references."
            )

        # ---------------------------------------------------------
        # Check 4: Source coverage
        # ---------------------------------------------------------

        if (
            len(usable_findings) > 0
            and total_sources == 0
        ):

            issues.append(
                "No source references were available "
                "for the research findings."
            )

        # ---------------------------------------------------------
        # Check 5: Very short findings
        # ---------------------------------------------------------

        short_findings = []

        for finding in usable_findings:

            if len(finding.summary.strip()) < 100:

                short_findings.append(
                    finding.subtask
                )

        if short_findings:

            issues.append(
                f"{len(short_findings)} finding(s) "
                f"contain limited supporting detail."
            )

        # ---------------------------------------------------------
        # Check 6: Potentially unsupported quantitative claims
        # ---------------------------------------------------------

        quantitative_pattern = re.compile(
            r"""
            (
                \b\d+(?:\.\d+)?\s*%
                |
                \$\s*\d+(?:\.\d+)?
                |
                \b\d+(?:\.\d+)?\s*
                (?:million|billion|trillion)
            )
            """,
            re.IGNORECASE | re.VERBOSE
        )

        quantitative_findings = 0

        for finding in usable_findings:

            if quantitative_pattern.search(
                finding.summary
            ):

                quantitative_findings += 1

        if (
            quantitative_findings > 0
            and total_sources > 0
        ):

            issues.append(
                "Some findings contain quantitative "
                "claims that should be checked against "
                "their cited sources."
            )

        # ---------------------------------------------------------
        # Check 7: Additional research questions
        # ---------------------------------------------------------

        if findings_without_sources > 0:

            additional_questions.append(
                "What primary sources can verify the "
                "unsupported or weakly sourced findings?"
            )

        if quantitative_findings > 0:

            additional_questions.append(
                "Which cited sources directly support "
                "the quantitative claims?"
            )

        # ---------------------------------------------------------
        # Limit issues
        # ---------------------------------------------------------

        issues = issues[:3]

        additional_questions = additional_questions[:2]

        # ---------------------------------------------------------
        # Determine whether more research is needed
        # ---------------------------------------------------------

        needs_more_research = False

        if total_findings == 0:
            needs_more_research = True

        elif len(usable_findings) < total_findings:
            needs_more_research = True

        elif findings_without_sources > 0:
            needs_more_research = True

        # ---------------------------------------------------------
        # Confidence calculation
        # ---------------------------------------------------------

        confidence = self._calculate_confidence(
            total_findings=total_findings,
            usable_findings=len(usable_findings),
            findings_with_sources=len(
                findings_with_sources
            ),
            total_sources=total_sources,
            quantitative_findings=quantitative_findings
        )

        # ---------------------------------------------------------
        # If there are no serious issues
        # ---------------------------------------------------------

        if not issues:

            issues = []

        # ---------------------------------------------------------
        # Create CriticReport
        # ---------------------------------------------------------

        report = CriticReport(
            issues=issues,
            needs_more_research=needs_more_research,
            additional_questions=additional_questions,
            confidence=confidence
        )

        # ---------------------------------------------------------
        # Create compatible LLMResponse
        #
        # Orchestrator expects a response object so that it
        # can track token usage.
        #
        # Since this Critic does not use an LLM, tokens = 0.
        # ---------------------------------------------------------

        response_text = self._format_report(
            report
        )

        response = LLMResponse(
            text=response_text,
            input_tokens=0,
            output_tokens=0
        )

        return report, response

    # =============================================================
    # CONFIDENCE
    # =============================================================

    @staticmethod
    def _calculate_confidence(
        total_findings: int,
        usable_findings: int,
        findings_with_sources: int,
        total_sources: int,
        quantitative_findings: int
    ) -> float:

        # No findings
        if total_findings == 0:

            return 0.0

        # ---------------------------------------------------------
        # Start with base confidence
        # ---------------------------------------------------------

        confidence = 0.50

        # ---------------------------------------------------------
        # Finding completeness
        # ---------------------------------------------------------

        finding_ratio = (
            usable_findings / total_findings
        )

        confidence += (
            0.20 * finding_ratio
        )

        # ---------------------------------------------------------
        # Source coverage
        # ---------------------------------------------------------

        if usable_findings > 0:

            source_ratio = (
                findings_with_sources
                / usable_findings
            )

            confidence += (
                0.20 * source_ratio
            )

        # ---------------------------------------------------------
        # Source quantity
        # ---------------------------------------------------------

        if total_sources >= 6:

            confidence += 0.10

        elif total_sources >= 3:

            confidence += 0.05

        # ---------------------------------------------------------
        # Quantitative claims require additional caution
        # ---------------------------------------------------------

        if quantitative_findings > 0:

            confidence -= 0.05

        # ---------------------------------------------------------
        # Clamp value
        # ---------------------------------------------------------

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return round(
            confidence,
            2
        )

    # =============================================================
    # FORMAT CRITIC REPORT
    # =============================================================

    @staticmethod
    def _format_report(
        report: CriticReport
    ) -> str:

        if report.issues:

            issues_text = "\n".join(
                f"- {issue}"
                for issue in report.issues
            )

        else:

            issues_text = "- None"

        if report.additional_questions:

            questions_text = "\n".join(
                f"- {question}"
                for question
                in report.additional_questions
            )

        else:

            questions_text = "- None"

        return (
            "ISSUES:\n"
            f"{issues_text}\n\n"
            "NEEDS_MORE_RESEARCH:\n"
            f"{str(report.needs_more_research).lower()}\n\n"
            "ADDITIONAL_QUESTIONS:\n"
            f"{questions_text}\n\n"
            "CONFIDENCE:\n"
            f"{report.confidence:.2f}"
        )