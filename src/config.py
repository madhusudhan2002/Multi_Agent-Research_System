"""
Centralized configuration.

Loads values from .env locally and Streamlit secrets
when deployed.
"""

import os

from dotenv import load_dotenv


# Load .env file locally
load_dotenv()


def get_secret(name: str, default: str = "") -> str:
    """
    Get a configuration value.

    Priority:
    1. Environment variable
    2. Streamlit secrets
    3. Default value
    """

    value = os.getenv(name)

    if value:
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets.get(name, default)

        return str(value).strip()

    except Exception:
        return default


class Settings:

    # ---------------------------------------------------------
    # API KEYS
    # ---------------------------------------------------------

    GROQ_API_KEY: str = get_secret(
        "GROQ_API_KEY"
    )

    TAVILY_API_KEY: str = get_secret(
        "TAVILY_API_KEY"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    MODEL_NAME: str = get_secret(
        "MODEL_NAME",
        "openai/gpt-oss-20b"
    )

    # ---------------------------------------------------------
    # RESEARCH SETTINGS
    # ---------------------------------------------------------

    # Keep the number of initial research tasks small
    # to reduce Groq token consumption.
    MAX_RESEARCH_SUBTASKS: int = int(
        get_secret(
            "MAX_RESEARCH_SUBTASKS",
            "3"
        )
    )

    # Disable additional critic/research rounds.
    # This prevents repeated Groq API calls.
    MAX_CRITIC_LOOPS: int = int(
        get_secret(
            "MAX_CRITIC_LOOPS",
            "0"
        )
    )

    # ---------------------------------------------------------
    # API SETTINGS
    # ---------------------------------------------------------

    REQUEST_TIMEOUT_SECONDS: int = int(
        get_secret(
            "REQUEST_TIMEOUT_SECONDS",
            "60"
        )
    )

    MAX_RETRIES: int = int(
        get_secret(
            "MAX_RETRIES",
            "2"
        )
    )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    @property
    def LIVE_LLM(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def LIVE_SEARCH(self) -> bool:
        # DuckDuckGo is used by SearchTool.
        # No API key is required.
        return True


settings = Settings()