"""Observability and LangSmith tracing package for Verified Research Agent."""

from verified_research.observability.collector import ObservabilityCallbackHandler
from verified_research.observability.config import (
    get_tracing_config,
    is_tracing_enabled,
    setup_langsmith_environment,
)
from verified_research.observability.metadata import (
    build_graph_session_metadata,
    build_hitl_metadata,
    build_reliability_metadata,
    build_research_metadata,
    build_supervisor_metadata,
    build_verifier_claim_metadata,
    sanitize_metadata,
    sanitize_text,
)
from verified_research.observability.run_config import build_trace_run_config
from verified_research.observability.tracer import (
    observe,
    record_event,
    record_metadata,
    trace_span,
)

__all__ = [
    "is_tracing_enabled",
    "get_tracing_config",
    "setup_langsmith_environment",
    "trace_span",
    "record_metadata",
    "record_event",
    "observe",
    "ObservabilityCallbackHandler",
    "build_trace_run_config",
    "sanitize_metadata",
    "sanitize_text",
    "build_supervisor_metadata",
    "build_research_metadata",
    "build_verifier_claim_metadata",
    "build_hitl_metadata",
    "build_reliability_metadata",
    "build_graph_session_metadata",
]
