"""
Orchestrator: coordinates Planner -> Researcher(s) -> Critic -> Writer.

Design notes (worth reading if you're reviewing this for a resume/interview):
- Research sub-tasks run in PARALLEL via a thread pool, since each is an
  independent I/O-bound LLM + search call.
- The Critic can trigger exactly one additional research round
  (MAX_CRITIC_LOOPS) to keep cost/latency bounded while still allowing
  self-correction -- a common real-world tradeoff in agentic systems.
- All token usage is tracked centrally for cost observability.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import settings
from .llm_client import LLMClient
from .search_tool import SearchTool
from .agents.planner import PlannerAgent
from .agents.researcher import ResearcherAgent
from .agents.critic import CriticAgent
from .agents.writer import WriterAgent
from .models import PipelineResult, AgentTraceEvent
from .utils.cost_tracker import CostTracker
from .utils.logger import log_step, log_info


class Orchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.search = SearchTool()
        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearcherAgent(self.llm, self.search)
        self.critic = CriticAgent(self.llm)
        self.writer = WriterAgent(self.llm)
        self.cost = CostTracker()

    def run(self, query: str, verbose: bool = True) -> PipelineResult:
        result = PipelineResult(query=query)

        if not settings.LIVE_LLM:
            log_info(
                "No ANTHROPIC_API_KEY found — running in MOCK MODE. "
                "Output structure is real; text content is simulated. "
                "Add a key to .env for live results."
            )

        # 1. PLAN ---------------------------------------------------------
        subtasks, resp = self.planner.plan(query)
        self.cost.add(resp.input_tokens, resp.output_tokens)
        result.subtasks = subtasks
        result.trace.append(AgentTraceEvent(
            "Planner", "decompose_query", f"{len(subtasks)} sub-tasks: {subtasks}",
            resp.input_tokens, resp.output_tokens,
        ))
        if verbose:
            log_step("Planner", "Decomposed query into sub-tasks", "\n".join(f"- {s}" for s in subtasks))

        # 2. RESEARCH (parallel) -------------------------------------------
        findings = self._research_parallel(subtasks, result, verbose)
        result.findings = findings

        # 3. CRITIC (with one bounded self-correction loop) ----------------
        loops = 0
        while loops <= settings.MAX_CRITIC_LOOPS:
            critic_report, resp = self.critic.review(query, findings)
            self.cost.add(resp.input_tokens, resp.output_tokens)
            result.critic_reports.append(critic_report)
            result.trace.append(AgentTraceEvent(
                "Critic", "review_findings",
                f"confidence={critic_report.confidence}, issues={critic_report.issues}",
                resp.input_tokens, resp.output_tokens,
            ))
            if verbose:
                log_step(
                    "Critic", "Reviewed findings",
                    f"Confidence: {critic_report.confidence}\n"
                    f"Issues: {critic_report.issues or 'None'}\n"
                    f"Needs more research: {critic_report.needs_more_research}",
                )

            if not critic_report.needs_more_research or loops == settings.MAX_CRITIC_LOOPS:
                break

            extra_findings = self._research_parallel(
                critic_report.additional_questions, result, verbose
            )
            findings = findings + extra_findings
            result.findings = findings
            loops += 1

        # 4. WRITE ----------------------------------------------------------
        report_md, resp = self.writer.write(query, findings, result.critic_reports[-1])
        self.cost.add(resp.input_tokens, resp.output_tokens)
        result.final_report_markdown = report_md
        result.trace.append(AgentTraceEvent(
            "Writer", "compile_report", f"{len(report_md)} chars generated",
            resp.input_tokens, resp.output_tokens,
        ))
        if verbose:
            log_step("Writer", "Compiled final report", report_md[:400] + ("..." if len(report_md) > 400 else ""))

        result.total_input_tokens = self.cost.input_tokens
        result.total_output_tokens = self.cost.output_tokens
        result.estimated_cost_usd = self.cost.estimated_cost_usd
        return result

    def _research_parallel(self, subtasks, result: PipelineResult, verbose: bool):
        findings = []
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(subtasks)))) as pool:
            futures = {pool.submit(self.researcher.research, st): st for st in subtasks}
            for future in as_completed(futures):
                finding, resp = future.result()
                self.cost.add(resp.input_tokens, resp.output_tokens)
                findings.append(finding)
                result.trace.append(AgentTraceEvent(
                    "Researcher", "research_subtask",
                    f"[{finding.subtask}] {len(finding.sources)} sources found",
                    resp.input_tokens, resp.output_tokens,
                ))
                if verbose:
                    log_step("Researcher", finding.subtask, finding.summary)
        return findings
