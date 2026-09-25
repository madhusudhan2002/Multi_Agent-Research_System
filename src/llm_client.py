"""
LLM client using Google Gemini.

Gemini is used only for text generation.
Web searching is handled separately by SearchTool.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

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
            self._client = genai.Client(
                api_key=settings.GEMINI_API_KEY
            )

    # =========================================================
    # COMPLETE
    # =========================================================

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:

        # -----------------------------------------------------
        # MOCK MODE
        # -----------------------------------------------------

        if not self.live:
            return self._mock_complete(system, prompt)

        last_err = None

        max_attempts = max(
            1,
            settings.MAX_RETRIES
        )

        for attempt in range(1, max_attempts + 1):

            try:

                # -------------------------------------------------
                # Gemini generation configuration
                # -------------------------------------------------

                config_kwargs = {
                    "system_instruction": system,
                    "temperature": 0.2,
                    "max_output_tokens": max_tokens,
                }

                # Request JSON when the caller expects JSON.
                if response_format is not None:
                    config_kwargs["response_mime_type"] = (
                        "application/json"
                    )

                config = types.GenerateContentConfig(
                    **config_kwargs
                )

                # -------------------------------------------------
                # Gemini API call
                # -------------------------------------------------

                response = self._client.models.generate_content(
                    model=settings.MODEL_NAME,
                    contents=prompt,
                    config=config,
                )

                # -------------------------------------------------
                # Extract response text
                # -------------------------------------------------

                text = getattr(
                    response,
                    "text",
                    ""
                ) or ""

                # -------------------------------------------------
                # Token usage
                # -------------------------------------------------

                input_tokens = 0
                output_tokens = 0

                usage = getattr(
                    response,
                    "usage_metadata",
                    None
                )

                if usage:

                    input_tokens = (
                        getattr(
                            usage,
                            "prompt_token_count",
                            0
                        )
                        or 0
                    )

                    output_tokens = (
                        getattr(
                            usage,
                            "candidates_token_count",
                            0
                        )
                        or 0
                    )

                # -------------------------------------------------
                # Empty response
                # -------------------------------------------------

                if not text.strip():

                    if attempt < max_attempts:

                        print(
                            "Gemini returned an empty response. "
                            "Retrying..."
                        )

                        time.sleep(
                            2 ** attempt
                        )

                        continue

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return LLMResponse(
                    text=text.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            # -----------------------------------------------------
            # API ERROR
            # -----------------------------------------------------

            except Exception as e:

                last_err = e

                error_name = type(e).__name__
                error_text = str(e)

                print(
                    f"Gemini API error "
                    f"(attempt {attempt}/{max_attempts}): "
                    f"{error_name}: {error_text}"
                )

                # -------------------------------------------------
                # Authentication errors
                # -------------------------------------------------

                authentication_error = (
                    "401" in error_text
                    or "403" in error_text
                    or "API key" in error_text
                    or "authentication" in error_text.lower()
                )

                if authentication_error:
                    break

                # -------------------------------------------------
                # Temporary/retryable errors
                # -------------------------------------------------

                retryable = (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "502" in error_text
                    or "503" in error_text
                    or "504" in error_text
                    or "UNAVAILABLE" in error_text
                    or "TIMEOUT" in error_text.upper()
                )

                if not retryable:
                    break

                if attempt < max_attempts:

                    wait_seconds = 2 ** attempt

                    print(
                        f"Retrying Gemini in "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(wait_seconds)

        # ---------------------------------------------------------
        # All attempts failed
        # ---------------------------------------------------------

        raise RuntimeError(
            "Gemini call failed after "
            f"{max_attempts} attempts: "
            f"{last_err}"
        )

    # =========================================================
    # MOCK MODE
    # =========================================================

    def _mock_complete(
        self,
        system: str,
        prompt: str
    ) -> LLMResponse:

        if "Break the user's research question" in system:

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
                "GEMINI_API_KEY to generate "
                "a real Gemini response."
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

        elif "professional research report" in system:

            text = (
                "# Research Report\n\n"
                "Mock report. Configure "
                "GEMINI_API_KEY to generate "
                "a real Gemini report."
            )

        else:

            text = "Mock response."

        return LLMResponse(
            text=text,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
        )


# =============================================================
# COST
# =============================================================

def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Cost calculation placeholder.

    Kept at zero because the application currently
    does not maintain Gemini pricing data.
    """

    return 0.0