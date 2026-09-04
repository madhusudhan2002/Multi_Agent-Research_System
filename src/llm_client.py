"""
Thin wrapper around the Anthropic API with:
  - automatic retry/backoff on transient errors
  - token usage tracking
  - a MOCK fallback so the whole pipeline runs offline with zero API keys
    (useful for demos, CI, and grading without burning API credits)
"""
import time
import random
from dataclasses import dataclass

from .config import settings

# Approx pricing for cost estimation (USD per 1M tokens). Update if your
# model/pricing differs -- this is only an estimate for the demo's cost tracker.
PRICE_PER_1M_INPUT = 3.00
PRICE_PER_1M_OUTPUT = 15.00


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """Calls Claude if ANTHROPIC_API_KEY is set, otherwise returns
    deterministic mock text so the pipeline is fully runnable offline."""

    def __init__(self):
        self.live = settings.LIVE_LLM
        self._client = None
        if self.live:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        if not self.live:
            return self._mock_complete(system, prompt)

        last_err = None
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                resp = self._client.messages.create(
                    model=settings.MODEL_NAME,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
                text = "".join(
                    block.text for block in resp.content if block.type == "text"
                )
                return LLMResponse(
                    text=text,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
            except Exception as e:  # broad on purpose: network/API errors, rate limits
                last_err = e
                sleep_s = min(2 ** attempt + random.random(), 20)
                time.sleep(sleep_s)
        raise RuntimeError(f"LLM call failed after {settings.MAX_RETRIES} attempts: {last_err}")

    # ------------------------------------------------------------------
    # Mock mode: keeps output structure realistic so the demo is legible
    # without requiring API keys.
    # ------------------------------------------------------------------
    def _mock_complete(self, system: str, prompt: str) -> LLMResponse:
        mock_notice = "MOCK MODE (no ANTHROPIC_API_KEY set) — this text is simulated.\n"

        if "Break the user's research question" in system:
            # No prose prefix here: the caller parses this as raw JSON.
            text = (
                '["What is the current state of the technology?", '
                '"What are the main risks or limitations?", '
                '"Who are the key players/companies involved?", '
                '"What does the near-term outlook look like?"]'
            )
        elif "research assistant" in system:
            text = mock_notice + (
                "Based on the (simulated) sources gathered, this sub-topic shows "
                "steady development with several credible sources confirming the "
                "core trend. Multiple independent sources agree on the general "
                "direction, though exact figures vary by source."
            )
        elif "fact-checking critic" in system:
            # No prose prefix here either: the caller parses this as raw JSON.
            text = (
                '{"issues": ["One finding cites a single source only"], '
                '"needs_more_research": false, "additional_questions": [], '
                '"confidence": 0.82}'
            )
        elif "professional research report" in system:
            text = mock_notice + (
                "## Executive Summary\nThis is a mock report generated without an "
                "API key so you can see the pipeline's structure end-to-end. "
                "Add your ANTHROPIC_API_KEY to .env for a real, LLM-written report "
                "with full citations."
            )
        else:
            text = mock_notice + "Mock response."

        # Rough fake token counts so the cost tracker still demonstrates its logic
        return LLMResponse(text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (
        output_tokens / 1_000_000
    ) * PRICE_PER_1M_OUTPUT
