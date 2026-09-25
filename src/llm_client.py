"""
LLM client using Groq.

Groq is used only for text generation.

Important:
This client does NOT provide browser/search tools to the LLM.
Web searching is handled separately by SearchTool.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import settings


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:

    def __init__(self):

        self.live = settings.LIVE_LLM

        self._client = None

        if self.live:

            from groq import Groq

            self._client = Groq(
                api_key=settings.GROQ_API_KEY
            )

    # =============================================================
    # COMPLETE
    # =============================================================

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:

        if not self.live:

            return self._mock_complete(
                system,
                prompt
            )

        last_err = None

        for attempt in range(
            1,
            settings.MAX_RETRIES + 1
        ):

            try:

                request_args = {

                    "model": settings.MODEL_NAME,

                    "messages": [

                        {
                            "role": "system",
                            "content": system
                        },

                        {
                            "role": "user",
                            "content": prompt
                        }

                    ],

                    "max_completion_tokens": max_tokens,

                    "temperature": 0.2,

                    # IMPORTANT:
                    # Do not allow automatic tool calls.
                    "tool_choice": "none"
                }

                # -------------------------------------------------
                # Response format is optional.
                # -------------------------------------------------

                if response_format is not None:

                    request_args[
                        "response_format"
                    ] = response_format

                # -------------------------------------------------
                # Groq request
                # -------------------------------------------------

                response = (
                    self._client
                    .chat
                    .completions
                    .create(
                        **request_args
                    )
                )

                # -------------------------------------------------
                # Extract text
                # -------------------------------------------------

                text = ""

                if response.choices:

                    message = response.choices[
                        0
                    ].message

                    text = (
                        message.content
                        or ""
                    )

                # -------------------------------------------------
                # Token usage
                # -------------------------------------------------

                usage = response.usage

                input_tokens = 0
                output_tokens = 0

                if usage:

                    input_tokens = (
                        getattr(
                            usage,
                            "prompt_tokens",
                            0
                        )
                        or 0
                    )

                    output_tokens = (
                        getattr(
                            usage,
                            "completion_tokens",
                            0
                        )
                        or 0
                    )

                # -------------------------------------------------
                # Empty response handling
                # -------------------------------------------------

                if not text.strip():

                    print(
                        "Groq returned an empty response."
                    )

                    if attempt < settings.MAX_RETRIES:

                        time.sleep(1)

                        continue

                return LLMResponse(

                    text=text,

                    input_tokens=input_tokens,

                    output_tokens=output_tokens

                )

            except Exception as e:

                last_err = e

                error_name = type(e).__name__
                error_text = str(e)

                print(
                    f"Groq API error "
                    f"(attempt {attempt}/"
                    f"{settings.MAX_RETRIES}): "
                    f"{error_name}: "
                    f"{error_text}"
                )

                # -------------------------------------------------
                # JSON validation errors should not be retried
                # repeatedly.
                # -------------------------------------------------

                if (
                    "json_validate_failed"
                    in error_text
                    or
                    "Failed to generate JSON"
                    in error_text
                    or
                    "Failed to validate JSON"
                    in error_text
                ):

                    break

                # -------------------------------------------------
                # Tool-use errors should not be repeatedly retried.
                # -------------------------------------------------

                if (
                    "tool_use_failed"
                    in error_text
                    or
                    "Tool choice is none"
                    in error_text
                ):

                    break

                # -------------------------------------------------
                # Normal retry
                # -------------------------------------------------

                sleep_s = min(
                    (2 ** attempt)
                    + random.random(),
                    15
                )

                time.sleep(sleep_s)

        raise RuntimeError(
            "LLM call failed after "
            f"{settings.MAX_RETRIES} attempts: "
            f"{last_err}"
        )

    # =============================================================
    # MOCK RESPONSE
    # =============================================================

    def _mock_complete(
        self,
        system: str,
        prompt: str
    ) -> LLMResponse:

        if (
            "Break the user's research question"
            in system
        ):

            text = (
                '["What is the current state of '
                'the technology?", '
                '"What are the main risks or '
                'limitations?", '
                '"Who are the key players involved?", '
                '"What does the near-term outlook '
                'look like?"]'
            )

        elif "research assistant" in system:

            text = (
                "Mock research result. Configure "
                "GROQ_API_KEY to generate a real "
                "LLM response."
            )

        elif "fact-checking critic" in system:

            text = (
                "ISSUES:\n"
                "- None\n\n"
                "NEEDS_MORE_RESEARCH:\n"
                "false\n\n"
                "ADDITIONAL_QUESTIONS:\n"
                "- None\n\n"
                "CONFIDENCE:\n"
                "0.80"
            )

        elif (
            "professional research report"
            in system
        ):

            text = (
                "# Research Report\n\n"
                "Mock report. Configure "
                "GROQ_API_KEY to generate "
                "a real research report."
            )

        else:

            text = "Mock response."

        return LLMResponse(

            text=text,

            input_tokens=len(prompt) // 4,

            output_tokens=len(text) // 4

        )


# =============================================================
# COST
# =============================================================

def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int
) -> float:

    return 0.0