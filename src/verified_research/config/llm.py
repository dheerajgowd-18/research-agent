"""LLM provider abstraction and factory."""

import logging
from langchain_core.language_models import BaseChatModel
from verified_research.config.settings import Settings, get_settings
from verified_research.tools.search import MissingApiKeyError

logger = logging.getLogger(__name__)


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Instantiate a chat model based on application settings.

    Args:
        settings: Application settings. If None, uses cached settings.

    Returns:
        A BaseChatModel instance.

    Raises:
        MissingApiKeyError: If required API key for the chosen provider is absent.
        ValueError: If an unsupported LLM provider is configured.
    """
    cfg = settings or get_settings()
    provider = cfg.llm_provider.lower()

    if provider == "groq":
        if not cfg.groq_api_key:
            raise MissingApiKeyError(
                "Groq API key is missing. Set GROQ_API_KEY in your environment or .env file."
            )
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
                api_key=cfg.groq_api_key,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error("[LLM] Failed to initialize ChatGroq: %s", e)
            raise RuntimeError(f"Failed to initialize ChatGroq: {e}") from e

    elif provider == "fake":
        from langchain_core.language_models.fake_chat_models import FakeChatModel

        return FakeChatModel()

    else:
        raise ValueError(f"Unsupported LLM provider: '{cfg.llm_provider}'. Supported: 'groq', 'fake'.")
