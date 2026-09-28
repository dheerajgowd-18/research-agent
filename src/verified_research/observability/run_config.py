"""Configuration factory for graph execution runs with LangSmith observability."""

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from verified_research.observability.collector import ObservabilityCallbackHandler
from verified_research.observability.config import is_tracing_enabled
from verified_research.observability.metadata import build_graph_session_metadata

DEFAULT_TAGS = ["verified-research", "phase-12"]


def build_trace_run_config(
    thread_id: str | None = None,
    session_id: str | None = None,
    question: str | None = None,
    question_id: str | None = None,
    additional_tags: list[str] | None = None,
    additional_metadata: dict[str, Any] | None = None,
    collector: ObservabilityCallbackHandler | None = None,
    tracer_client: Any | None = None,
    extra_callbacks: list[BaseCallbackHandler] | None = None,
) -> RunnableConfig:
    """Construct a complete RunnableConfig for graph execution with tracing, tags, and metadata.

    Args:
        thread_id: Optional thread identifier for checkpoint persistence.
        session_id: Optional research session identifier.
        question: Substantive research question for session metadata.
        question_id: Optional explicit identifier for the question.
        additional_tags: Extra tags for categorization and filtering.
        additional_metadata: Extra operational metadata to attach to the trace.
        collector: Optional ObservabilityCallbackHandler instance to gather metrics.
        tracer_client: Optional custom LangSmith client (useful for hermetic tests).
        extra_callbacks: Any additional callback handlers.

    Returns:
        A RunnableConfig dictionary compatible with LangGraph invoke() and stream().
    """
    tags = list(DEFAULT_TAGS)
    if additional_tags:
        for t in additional_tags:
            if t not in tags:
                tags.append(t)

    session_meta = build_graph_session_metadata(
        thread_id=thread_id,
        session_id=session_id,
        phase="12",
        question=question,
        question_id=question_id,
    )
    if additional_metadata:
        session_meta.update(additional_metadata)

    callbacks: list[BaseCallbackHandler] = []

    # Attach operational metrics collector
    active_collector = collector or ObservabilityCallbackHandler()
    callbacks.append(active_collector)

    # Attach LangChainTracer if client is injected or tracing is enabled
    if tracer_client is not None:
        callbacks.append(LangChainTracer(client=tracer_client))
    elif is_tracing_enabled():
        callbacks.append(LangChainTracer())

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    config: RunnableConfig = {
        "tags": tags,
        "metadata": session_meta,
        "callbacks": callbacks,
    }

    if thread_id:
        config["configurable"] = {"thread_id": str(thread_id)}

    return config
