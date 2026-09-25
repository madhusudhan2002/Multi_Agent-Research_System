"""
Centralized configuration.

Uses Google Gemini API for LLM generation.

Loads values from .env locally and
Streamlit secrets when deployed.
"""

import os

from dotenv import load_dotenv


load_dotenv()


def get_secret(name: str, default: str = "") -> str:
    """
    Get configuration value.

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
    GEMINI_API_KEY: str = get_secret("GEMINI_API_KEY")
    MODEL_NAME: str = get_secret(
        "MODEL_NAME",
        "gemini-3.8-flash"
    )

    MAX_RESEARCH_SUBTASKS: int = int(
        get_secret("MAX_RESEARCH_SUBTASKS", "2")
    )

    MAX_CRITIC_LOOPS: int = int(
        get_secret("MAX_CRITIC_LOOPS", "0")
    )

    REQUEST_TIMEOUT_SECONDS: int = int(
        get_secret("REQUEST_TIMEOUT_SECONDS", "60")
    )

    MAX_RETRIES: int = int(
        get_secret("MAX_RETRIES", "2")
    )

    @property
    def LIVE_LLM(self) -> bool:
        return bool(self.GEMINI_API_KEY)

    @property
    def LIVE_SEARCH(self) -> bool:
        return True


settings = Settings()