"""
Optional FastAPI server exposing the multi-agent pipeline as an HTTP API.

Run with:
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from src.orchestrator import Orchestrator

app = FastAPI(
    title="Multi-Agent Research API",
    description="Planner -> Researcher -> Critic -> Writer pipeline",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    subtasks: list[str]
    report_markdown: str
    confidence: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    orchestrator = Orchestrator()
    result = orchestrator.run(req.query, verbose=False)
    return ResearchResponse(
        query=result.query,
        subtasks=result.subtasks,
        report_markdown=result.final_report_markdown,
        confidence=result.critic_reports[-1].confidence if result.critic_reports else 0.0,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )
