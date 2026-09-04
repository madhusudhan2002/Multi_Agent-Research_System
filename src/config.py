"""
Centralized configuration. Loads from .env if present, otherwise falls
back to sane defaults so the project runs out of the box in MOCK mode.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
    MODEL_NAME: str = os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929")

    MAX_RESEARCH_SUBTASKS: int = int(os.getenv("MAX_RESEARCH_SUBTASKS", 4))
    MAX_CRITIC_LOOPS: int = int(os.getenv("MAX_CRITIC_LOOPS", 1))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 60))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

    @property
    def LIVE_LLM(self) -> bool:
        """True if a real Anthropic API key is configured."""
        return bool(self.ANTHROPIC_API_KEY)

    @property
    def LIVE_SEARCH(self) -> bool:
        """True if a real Tavily API key is configured."""
        return bool(self.TAVILY_API_KEY)


settings = Settings()
