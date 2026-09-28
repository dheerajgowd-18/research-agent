"""LangSmith and observability configuration management."""

import logging
import os
from typing import Any
from langsmith.utils import get_env_var
from verified_research.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def is_tracing_enabled(settings: Settings | None = None) -> bool:
    """Determine whether LangSmith tracing is currently enabled.

    Evaluates:
        1. Explicit settings.langsmith_tracing / settings.langchain_tracing_v2
        2. Environment variables (LANGSMITH_TRACING, LANGCHAIN_TRACING_V2, LANGCHAIN_TRACING)
        3. LangSmith runtime context

    Returns:
        True if tracing is enabled, False otherwise.
    """
    cfg = settings or get_settings()
    if cfg.langsmith_tracing:
        return True

    # Check environment variables directly
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("true", "1", "yes", "on"):
            return True

    return False


def get_tracing_config(settings: Settings | None = None) -> dict[str, Any]:
    """Retrieve the active tracing configuration with credentials masked for safe inspection.

    Args:
        settings: Application settings. If None, uses cached settings.

    Returns:
        Dictionary detailing tracing status, project, endpoint, and masked credential status.
    """
    cfg = settings or get_settings()
    api_key = cfg.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    enabled = is_tracing_enabled(cfg)

    masked_key = None
    if api_key:
        clean_key = api_key.strip()
        if len(clean_key) > 8:
            masked_key = f"{clean_key[:4]}...{clean_key[-4:]}"
        else:
            masked_key = "***"

    project = cfg.langsmith_project or os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "verified-research-agent"
    endpoint = cfg.langsmith_endpoint or os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"

    return {
        "tracing_enabled": enabled,
        "project": project,
        "endpoint": endpoint,
        "has_api_key": bool(api_key),
        "masked_api_key": masked_key,
    }


def setup_langsmith_environment(settings: Settings | None = None) -> None:
    """Synchronize application settings to environment variables for LangSmith/LangChain.

    Sets standard LangSmith environment variables when tracing is enabled.
    Clears LangSmith's internal LRU cache for get_env_var to ensure immediate effect.
    Never logs raw credentials.

    Args:
        settings: Application settings. If None, uses cached settings.
    """
    cfg = settings or get_settings()
    if not is_tracing_enabled(cfg):
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if cfg.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project

    if cfg.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = cfg.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = cfg.langsmith_endpoint

    if cfg.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key

    # Invalidate LangSmith's LRU cache so the new environment is recognized
    try:
        get_env_var.cache_clear()
    except Exception:
        pass

    logger.debug("[Observability] LangSmith environment synchronized for project '%s'", cfg.langsmith_project)
