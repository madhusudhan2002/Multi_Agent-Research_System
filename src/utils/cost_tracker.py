"""Tracks token usage and estimated $ cost across all agent calls."""
from dataclasses import dataclass, field
from ..llm_client import estimate_cost_usd


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost_usd(self.input_tokens, self.output_tokens)

    def summary(self) -> str:
        return (
            f"{self.calls} LLM calls | "
            f"{self.input_tokens:,} input tokens | "
            f"{self.output_tokens:,} output tokens | "
            f"~${self.estimated_cost_usd:.4f} estimated cost"
        )
