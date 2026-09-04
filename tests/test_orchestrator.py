"""
Unit tests for the multi-agent pipeline. These run entirely in MOCK mode
(no API keys needed) so they're safe to run in CI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator
from src.agents.planner import PlannerAgent
from src.agents.critic import CriticAgent
from src.llm_client import LLMClient
from src.models import ResearchFinding, SearchResult


def test_planner_parses_json_array():
    llm = LLMClient()
    planner = PlannerAgent(llm)
    subtasks, resp = planner.plan("What is the impact of AI on software jobs?")
    assert isinstance(subtasks, list)
    assert len(subtasks) >= 1
    assert all(isinstance(s, str) for s in subtasks)


def test_planner_falls_back_gracefully_on_bad_json():
    assert PlannerAgent._parse_json_array("not json at all") == []
    assert PlannerAgent._parse_json_array('["a", "b"]') == ["a", "b"]
    assert PlannerAgent._parse_json_array('[1, "a"]') == ["a"]  # non-strings dropped


def test_critic_parses_valid_json():
    report = CriticAgent._parse(
        '{"issues": ["x"], "needs_more_research": true, '
        '"additional_questions": ["y"], "confidence": 0.5}'
    )
    assert report.issues == ["x"]
    assert report.needs_more_research is True
    assert report.additional_questions == ["y"]
    assert report.confidence == 0.5


def test_critic_falls_back_gracefully_on_bad_json():
    report = CriticAgent._parse("garbage, no json here")
    assert report.issues == []
    assert report.needs_more_research is False
    assert report.confidence == 1.0


def test_full_pipeline_runs_end_to_end_in_mock_mode():
    orchestrator = Orchestrator()
    result = orchestrator.run("What is retrieval-augmented generation?", verbose=False)

    assert result.query
    assert len(result.subtasks) >= 1
    assert len(result.findings) >= 1
    assert len(result.critic_reports) >= 1
    assert result.final_report_markdown
    assert result.total_input_tokens > 0
    assert result.total_output_tokens > 0
    assert result.estimated_cost_usd >= 0
    # every research subtask should produce a matching trace event
    researcher_events = [e for e in result.trace if e.agent == "Researcher"]
    assert len(researcher_events) == len(result.subtasks)


def test_research_finding_holds_sources():
    finding = ResearchFinding(
        subtask="test?",
        summary="summary text",
        sources=[SearchResult(title="t", url="https://x.com", snippet="s")],
    )
    assert len(finding.sources) == 1
    assert finding.sources[0].url == "https://x.com"
