"""Search tool abstraction and Tavily client implementation."""

import logging
from typing import Protocol, runtime_checkable
from verified_research.config.settings import Settings, get_settings
from verified_research.models.research import Source
from verified_research.reliability.models import ErrorInfo
from verified_research.reliability.policy import RetryPolicy, execute_with_retry

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Raised when an error occurs during web search execution."""

    def __init__(self, message: str, error_info: ErrorInfo | None = None) -> None:
        super().__init__(message)
        self.error_info = error_info


class MissingApiKeyError(ValueError):
    """Raised when an external API key is required but missing."""


@runtime_checkable
class SearchService(Protocol):
    """Protocol defining the search service contract."""

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """Execute a search query and return normalized Source objects."""
        ...


class TavilySearchClient:
    """Tavily search service implementation that normalizes results into Source models."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = get_settings().tavily_api_key
        self.max_results = max_results
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=get_settings().retry_max_attempts,
            base_delay=get_settings().retry_base_delay,
            max_delay=get_settings().retry_max_delay,
            backoff_factor=get_settings().retry_backoff_factor,
            jitter=get_settings().retry_jitter,
        )
        self._client = None

    def _get_client(self):
        if not self.api_key or not self.api_key.strip():
            raise MissingApiKeyError(
                "Tavily API key is missing. Set TAVILY_API_KEY in your environment or .env file."
            )
        if self._client is None:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                raise SearchError(f"Failed to initialize Tavily client: {e}") from e
        return self._client

    def search(self, query: str, max_results: int | None = None) -> list[Source]:
        """Execute search via Tavily and convert the raw response into normalized Source models.

        Args:
            query: The search query string.
            max_results: Maximum number of results to fetch (defaults to self.max_results).

        Returns:
            A list of normalized Source objects. Returns empty list if no results found.

        Raises:
            MissingApiKeyError: If API key is not configured.
            SearchError: If Tavily API request fails.
        """
        if not query or not query.strip():
            logger.warning("[SEARCH] Empty query provided, returning 0 sources.")
            return []

        limit = max_results or self.max_results
        client = self._get_client()

        def _do_search():
            return client.search(
                query=query.strip(),
                max_results=limit,
                search_depth="basic",
            )

        from verified_research.observability.metadata import sanitize_text
        from verified_research.observability.tracer import record_metadata, trace_span

        with trace_span(
            name="search",
            run_type="tool",
            metadata={"query": sanitize_text(query), "max_results": limit},
            tags=["search", "tavily"],
        ):
            try:
                raw_response = execute_with_retry(
                    operation=_do_search,
                    policy=self.retry_policy,
                    component="search",
                    operation_name="tavily_search",
                    reraise_original=True,
                )
            except Exception as e:
                logger.error("[SEARCH] Tavily search failed for query '%s': %s", query, e)
                error_info = getattr(e, "error_info", None)
                raise SearchError(f"Tavily search request failed: {e}", error_info=error_info) from e

            raw_results = raw_response.get("results", []) if isinstance(raw_response, dict) else []
            sources = self._normalize_results(raw_results)
            record_metadata(sources_count=len(sources))
            return sources

    @staticmethod
    def _normalize_results(raw_results: list[dict]) -> list[Source]:
        """Convert raw Tavily result dicts into typed Source instances."""
        sources: list[Source] = []
        for index, item in enumerate(raw_results, start=1):
            source_id = f"src_{index:03d}"
            title = (item.get("title") or "Untitled Source").strip()
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()

            # Ensure minimal content and url validity before building Source
            if not url:
                url = f"https://unknown-source.local/{source_id}"
            if not content:
                content = "(No content snippet available)"

            sources.append(
                Source(
                    source_id=source_id,
                    title=title,
                    url=url,
                    content=content,
                )
            )

        return sources
