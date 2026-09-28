"""Core tracing utilities and context managers for LangSmith observability."""

from contextlib import contextmanager
import functools
import logging
from typing import Any, Callable, Generator
from langsmith.run_helpers import get_current_run_tree, trace
from verified_research.observability.config import is_tracing_enabled
from verified_research.observability.metadata import sanitize_metadata

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(
    name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
    client: Any | None = None,
) -> Generator[Any, None, None]:
    """Execute code within a structured LangSmith trace span if tracing is enabled.

    When tracing is disabled, gracefully yields None with near-zero overhead.
    Automatically scrubs secrets from metadata and tags.

    Args:
        name: Name of the span (e.g. 'search:tavily', 'verify_claim:claim_001').
        run_type: LangSmith run type ('chain', 'tool', 'llm', etc.).
        metadata: Structured operational metadata dictionary.
        tags: List of search/filtering tags.
        inputs: Optional operational inputs to record on the span.
        client: Optional LangSmith client override (useful for isolated testing).

    Yields:
        The active RunTree if tracing is enabled, otherwise None.
    """
    clean_meta = sanitize_metadata(metadata or {})
    clean_tags = list(tags or [])
    should_trace = is_tracing_enabled() or client is not None

    if should_trace:
        try:
            ctx = trace(
                name=name,
                run_type=run_type,
                metadata=clean_meta,
                tags=clean_tags,
                inputs=inputs or {},
                client=client,
            )
        except Exception as e:
            logger.debug("[Observability] Failed to initialize trace span '%s': %s", name, e)
            yield None
            return

        with ctx as run_tree:
            yield run_tree
    else:
        yield None


def record_metadata(**metadata: Any) -> None:
    """Attach structured operational metadata to the active LangSmith run tree.

    If tracing is disabled or no run tree is active, performs a safe no-op.
    Automatically redacts sensitive keys and credential patterns.

    Args:
        **metadata: Key-value pairs to merge into the active span's metadata.
    """
    if not is_tracing_enabled():
        return

    clean_meta = sanitize_metadata(metadata)
    try:
        rt = get_current_run_tree()
        if rt is not None:
            if hasattr(rt, "metadata") and isinstance(rt.metadata, dict):
                rt.metadata.update(clean_meta)
            if hasattr(rt, "extra") and isinstance(rt.extra, dict):
                if "metadata" not in rt.extra or not isinstance(rt.extra["metadata"], dict):
                    rt.extra["metadata"] = {}
                rt.extra["metadata"].update(clean_meta)
    except Exception as e:
        logger.debug("[Observability] Failed to record metadata: %s", e)


def record_event(name: str, **data: Any) -> None:
    """Record an operational lifecycle event on the active span."""
    if not is_tracing_enabled():
        return

    clean_data = sanitize_metadata(data)
    try:
        rt = get_current_run_tree()
        if rt is not None and hasattr(rt, "events") and isinstance(rt.events, list):
            import datetime
            rt.events.append({
                "name": name,
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "data": clean_data,
            })
    except Exception as e:
        logger.debug("[Observability] Failed to record event '%s': %s", name, e)


def observe(
    name: str | None = None,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata_extractor: Callable[..., dict[str, Any]] | None = None,
) -> Callable:
    """Decorator to automatically wrap a function execution in a trace span."""
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            meta = metadata_extractor(*args, **kwargs) if metadata_extractor else {}
            with trace_span(name=span_name, run_type=run_type, metadata=meta, tags=tags):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
