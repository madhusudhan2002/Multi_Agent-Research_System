"""Shared data structures passed between agents."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class ResearchFinding:
    subtask: str
    summary: str
    sources: List[SearchResult] = field(default_factory=list)


@dataclass
class CriticReport:
    issues: List[str] = field(default_factory=list)
    needs_more_research: bool = False
    additional_questions: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class AgentTraceEvent:
    agent: str
    action: str
    detail: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PipelineResult:
    query: str
    subtasks: List[str] = field(default_factory=list)
    findings: List[ResearchFinding] = field(default_factory=list)
    critic_reports: List[CriticReport] = field(default_factory=list)
    final_report_markdown: str = ""
    trace: List[AgentTraceEvent] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
