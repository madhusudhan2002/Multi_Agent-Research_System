"""
Orchestrator.

Coordinates:

Planner
   ↓
Researcher(s)
   ↓
Critic
   ↓
Writer

Research tasks run in parallel.

The critic loop is configurable through
MAX_CRITIC_LOOPS.
"""

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from .config import settings
from .llm_client import LLMClient
from .search_tool import SearchTool

from .agents.planner import (
    PlannerAgent
)

from .agents.researcher import (
    ResearcherAgent
)

from .agents.critic import (
    CriticAgent
)

from .agents.writer import (
    WriterAgent
)

from .models import (
    PipelineResult,
    AgentTraceEvent
)

from .utils.cost_tracker import (
    CostTracker
)

from .utils.logger import (
    log_step,
    log_info
)


class Orchestrator:

    def __init__(self):

        self.llm = LLMClient()

        self.search = SearchTool()

        self.planner = PlannerAgent(
            self.llm
        )

        self.researcher = ResearcherAgent(
            self.llm,
            self.search
        )

        self.critic = CriticAgent(
            self.llm
        )

        self.writer = WriterAgent(
            self.llm
        )

        self.cost = CostTracker()

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def run(
        self,
        query: str,
        verbose: bool = True
    ) -> PipelineResult:

        result = PipelineResult(
            query=query
        )

        # -----------------------------------------------------
        # MODE
        # -----------------------------------------------------

        if not settings.LIVE_LLM:

            log_info(
                "No GEMINI_API_KEY found — "
                "running in MOCK MODE."
            )

        # -----------------------------------------------------
        # PLANNER
        # -----------------------------------------------------

        subtasks, resp = (
            self.planner.plan(
                query
            )
        )

        # Limit subtasks
        # according to configuration.

        max_tasks = max(
            1,
            settings.MAX_RESEARCH_SUBTASKS
        )

        subtasks = subtasks[
            :max_tasks
        ]

        self.cost.add(
            resp.input_tokens,
            resp.output_tokens
        )

        result.subtasks = subtasks

        result.trace.append(
            AgentTraceEvent(
                "Planner",
                "decompose_query",
                (
                    f"{len(subtasks)} "
                    f"sub-tasks: {subtasks}"
                ),
                resp.input_tokens,
                resp.output_tokens,
            )
        )

        if verbose:

            log_step(
                "Planner",
                "Decomposed query into "
                "sub-tasks",

                "\n".join(
                    f"- {task}"
                    for task in subtasks
                )
            )

        # -----------------------------------------------------
        # RESEARCH
        # -----------------------------------------------------

        findings = self._research_parallel(
            subtasks,
            result,
            verbose
        )

        result.findings = findings

        # -----------------------------------------------------
        # CRITIC
        # -----------------------------------------------------

        loops = 0

        while True:

            critic_report, resp = (
                self.critic.review(
                    query,
                    findings
                )
            )

            self.cost.add(
                resp.input_tokens,
                resp.output_tokens
            )

            result.critic_reports.append(
                critic_report
            )

            result.trace.append(
                AgentTraceEvent(

                    "Critic",

                    "review_findings",

                    (
                        f"confidence="
                        f"{critic_report.confidence}, "

                        f"issues="
                        f"{critic_report.issues}"
                    ),

                    resp.input_tokens,
                    resp.output_tokens,
                )
            )

            if verbose:

                log_step(

                    "Critic",

                    "Reviewed findings",

                    (
                        f"Confidence: "
                        f"{critic_report.confidence}\n"

                        f"Issues: "
                        f"{critic_report.issues or 'None'}\n"

                        f"Needs more research: "
                        f"{critic_report.needs_more_research}"
                    )
                )

            # -------------------------------------------------
            # STOP CONDITIONS
            # -------------------------------------------------

            # No more research required
            if not critic_report.needs_more_research:
                break

            # Maximum critic loops reached
            if (
                loops
                >= settings.MAX_CRITIC_LOOPS
            ):
                break

            # No additional questions
            if not critic_report.additional_questions:
                break

            # -------------------------------------------------
            # ADDITIONAL RESEARCH
            # -------------------------------------------------

            extra_findings = (
                self._research_parallel(
                    critic_report.additional_questions,
                    result,
                    verbose
                )
            )

            findings.extend(
                extra_findings
            )

            result.findings = findings

            loops += 1

        # -----------------------------------------------------
        # WRITER
        # -----------------------------------------------------

        report_md, resp = (
            self.writer.write(
                query,
                findings,
                result.critic_reports[-1]
            )
        )

        self.cost.add(
            resp.input_tokens,
            resp.output_tokens
        )

        result.final_report_markdown = (
            report_md
        )

        result.trace.append(
            AgentTraceEvent(

                "Writer",

                "compile_report",

                (
                    f"{len(report_md)} "
                    f"chars generated"
                ),

                resp.input_tokens,
                resp.output_tokens,
            )
        )

        if verbose:

            preview = report_md[:400]

            if len(report_md) > 400:
                preview += "..."

            log_step(
                "Writer",
                "Compiled final report",
                preview
            )

        # -----------------------------------------------------
        # TOKEN / COST INFORMATION
        # -----------------------------------------------------

        result.total_input_tokens = (
            self.cost.input_tokens
        )

        result.total_output_tokens = (
            self.cost.output_tokens
        )

        result.estimated_cost_usd = (
            self.cost.estimated_cost_usd
        )

        return result

    # =========================================================
    # PARALLEL RESEARCH
    # =========================================================

    def _research_parallel(
        self,
        subtasks,
        result: PipelineResult,
        verbose: bool
    ):

        findings = []

        if not subtasks:
            return findings

        worker_count = min(
            3,
            max(
                1,
                len(subtasks)
            )
        )

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as pool:

            futures = {

                pool.submit(
                    self.researcher.research,
                    subtask
                ): subtask

                for subtask in subtasks
            }

            for future in as_completed(
                futures
            ):

                subtask = futures[
                    future
                ]

                try:

                    finding, resp = (
                        future.result()
                    )

                    self.cost.add(
                        resp.input_tokens,
                        resp.output_tokens
                    )

                    findings.append(
                        finding
                    )

                    result.trace.append(
                        AgentTraceEvent(

                            "Researcher",

                            "research_subtask",

                            (
                                f"["
                                f"{finding.subtask}"
                                f"] "

                                f"{len(finding.sources)} "
                                f"sources found"
                            ),

                            resp.input_tokens,
                            resp.output_tokens,
                        )
                    )

                    if verbose:

                        log_step(
                            "Researcher",

                            finding.subtask,

                            finding.summary
                        )

                except Exception as e:

                    print(
                        f"Researcher failed "
                        f"for subtask: "
                        f"{subtask}"
                    )

                    print(
                        f"Error: {e}"
                    )

        return findings