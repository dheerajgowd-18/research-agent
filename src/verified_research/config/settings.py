"""Configuration settings for Verified Research Agent."""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Search Configuration
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    search_max_results: int = Field(default=5, alias="SEARCH_MAX_RESULTS")
    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")
    max_human_research_cycles: int = Field(default=2, alias="MAX_HUMAN_RESEARCH_CYCLES")

    # LLM Configuration
    llm_provider: Literal["groq", "openai", "fake"] = Field(
        default="groq", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="openai/gpt-oss-20b", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # LangSmith / Observability (Optional)
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(
        default="verified-research-agent", alias="LANGCHAIN_PROJECT"
    )

    # Persistence / Checkpoint Configuration
    checkpoint_db_path: str = Field(default="checkpoints.db", alias="CHECKPOINT_DB_PATH")


DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_HUMAN_RESEARCH_CYCLES = 2
DEFAULT_CHECKPOINT_DB_PATH = "checkpoints.db"




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings."""
    return Settings()

