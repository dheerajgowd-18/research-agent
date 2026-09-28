"""Structured metadata builders and security sanitization for observability."""

import re
from typing import Any
from verified_research.models.research import Critique

# Regex patterns identifying sensitive tokens or keys
SECRET_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9]{30,}", re.IGNORECASE),
    re.compile(r"tvly-[a-zA-Z0-9\-_]{20,}", re.IGNORECASE),
    re.compile(r"lsv2_[a-zA-Z0-9_]{30,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9\-_.]+", re.IGNORECASE),
]

SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "groq_api_key",
    "tavily_api_key",
    "langchain_api_key",
    "langsmith_api_key",
    "openai_api_key",
}


def sanitize_text(text: str) -> str:
    """Scrub sensitive credentials, tokens, and authorization headers from arbitrary text."""
    if not isinstance(text, str):
        return str(text)

    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def sanitize_metadata(data: dict[str, Any], max_string_len: int = 500) -> dict[str, Any]:
    """Recursively scrub secrets and bound string lengths in operational metadata.

    Args:
        data: Dictionary of metadata key-value pairs.
        max_string_len: Maximum length for string values to prevent bloating metadata.

    Returns:
        Sanitized metadata dictionary safe for telemetry.
    """
    clean: dict[str, Any] = {}
    for key, value in data.items():
        k_lower = str(key).lower()
        if any(s in k_lower for s in SENSITIVE_KEY_NAMES):
            clean[key] = "[REDACTED]"
            continue

        if isinstance(value, str):
            sanitized_val = sanitize_text(value)
            if len(sanitized_val) > max_string_len:
                sanitized_val = sanitized_val[:max_string_len] + "..."
            clean[key] = sanitized_val
        elif isinstance(value, dict):
            clean[key] = sanitize_metadata(value, max_string_len=max_string_len)
        elif isinstance(value, (list, tuple)):
            clean_list = []
            for item in value:
                if isinstance(item, str):
                    s_item = sanitize_text(item)
                    if len(s_item) > max_string_len:
                        s_item = s_item[:max_string_len] + "..."
                    clean_list.append(s_item)
                elif isinstance(item, dict):
                    clean_list.append(sanitize_metadata(item, max_string_len=max_string_len))
                else:
                    clean_list.append(item)
            clean[key] = clean_list
        else:
            clean[key] = value

    return clean


def build_supervisor_metadata(
    step: int,
    selected_worker: str,
    decision: str,
    termination_reason: str | None = None,
    reasoning: str | None = None,
    unverified_claims_count: int | None = None,
) -> dict[str, Any]:
    """Construct structured operational metadata for supervisor executions."""
    data: dict[str, Any] = {
        "supervisor_step": step,
        "selected_worker": selected_worker,
        "decision": decision,
        "termination_reason": termination_reason,
    }
    if reasoning:
        data["reasoning"] = reasoning[:200]
    if unverified_claims_count is not None:
        data["unverified_claims_count"] = unverified_claims_count
    return sanitize_metadata(data)


def build_research_metadata(
    iteration: int,
    search_queries: list[str] | None = None,
    new_sources_count: int = 0,
    total_sources_count: int = 0,
    critique: Critique | None = None,
    expansion_events: list[str] | None = None,
) -> dict[str, Any]:
    """Construct structured operational metadata for research worker iterations."""
    queries = [sanitize_text(q)[:150] for q in (search_queries or [])]
    data: dict[str, Any] = {
        "research_iteration": iteration,
        "search_operation_count": len(queries),
        "search_queries": queries,
        "new_sources_count": new_sources_count,
        "total_sources_count": total_sources_count,
    }
    if critique is not None:
        data["critic_quality_score"] = float(critique.quality_score)
        data["critic_should_research_again"] = bool(critique.should_research_again)
        data["critic_missing_topics_count"] = len(critique.missing_topics)
        data["critic_citation_gaps_count"] = len(critique.citation_gaps)
    if expansion_events:
        data["expansion_events"] = expansion_events[:5]
    return sanitize_metadata(data)


def build_verifier_claim_metadata(
    claim_id: str,
    verdict: str,
    confidence: float,
    evidence_count: int,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Construct structured operational metadata for claim verification."""
    data: dict[str, Any] = {
        "claim_id": claim_id,
        "verdict": verdict,
        "confidence": round(float(confidence), 3),
        "evidence_count": evidence_count,
    }
    if reasoning:
        data["reasoning"] = reasoning[:200]
    return sanitize_metadata(data)


def build_hitl_metadata(
    event: str,
    human_action: str | None = None,
    human_research_cycle: int = 0,
    claims_count: int = 0,
    edited_claims_count: int | None = None,
    feedback_present: bool = False,
) -> dict[str, Any]:
    """Construct structured operational metadata for Human-in-the-Loop review events."""
    data: dict[str, Any] = {
        "hitl_event": event,
        "human_action": human_action,
        "human_research_cycle": human_research_cycle,
        "claims_count": claims_count,
        "feedback_present": feedback_present,
    }
    if edited_claims_count is not None:
        data["edited_claims_count"] = edited_claims_count
    return sanitize_metadata(data)


def build_reliability_metadata(
    component: str,
    operation: str,
    attempt_number: int,
    retry_count: int,
    error_category: str | None = None,
    final_status: str = "success",
    delay_seconds: float | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    """Construct structured operational metadata for reliability and retry tracking."""
    data: dict[str, Any] = {
        "component": component,
        "operation": operation,
        "attempt_number": attempt_number,
        "retry_count": retry_count,
        "final_status": final_status,
    }
    if error_category:
        data["error_category"] = str(error_category)
    if delay_seconds is not None:
        data["delay_seconds"] = round(float(delay_seconds), 3)
    if status_code is not None:
        data["status_code"] = status_code
    return sanitize_metadata(data)


def build_graph_session_metadata(
    thread_id: str | None = None,
    session_id: str | None = None,
    phase: str = "12",
    question: str | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Construct top-level graph execution metadata."""
    data: dict[str, Any] = {
        "phase": phase,
    }
    if thread_id:
        data["thread_id"] = str(thread_id)
    if session_id:
        data["session_id"] = str(session_id)
    if question:
        clean_q = sanitize_text(question.strip())
        data["question_preview"] = clean_q[:150]
        if not question_id:
            import hashlib
            data["question_id"] = f"q_{hashlib.md5(clean_q.encode('utf-8')).hexdigest()[:8]}"
    if question_id:
        data["question_id"] = question_id

    return sanitize_metadata(data)
