"""Analyst node implementation for Phase 1 research pipeline."""

import logging
from typing import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.config.llm import get_chat_model
from verified_research.graph.state import ResearchState
from verified_research.models.research import AnalystOutput, Finding, Source

logger = logging.getLogger(__name__)


class InvalidSourceReferenceError(ValueError):
    """Raised when a finding references a source_id not present in graph state."""


def validate_finding_sources(findings: list[Finding], valid_sources: list[Source]) -> None:
    """Verify that every source_id referenced in findings exists in the provided sources.

    Args:
        findings: List of synthesized findings to validate.
        valid_sources: Ground-truth sources present in the current graph state.

    Raises:
        InvalidSourceReferenceError: If any finding cites an unrecognized source_id.
    """
    valid_ids = {s.source_id for s in valid_sources}

    for finding in findings:
        for sid in finding.source_ids:
            if sid not in valid_ids:
                raise InvalidSourceReferenceError(
                    f"Finding '{finding.finding_id}' referenced invalid source_id '{sid}'. "
                    f"Available source IDs in state: {sorted(valid_ids)}"
                )


def _format_sources_for_prompt(sources: list[Source]) -> str:
    """Format source models into a clean textual representation for LLM context."""
    formatted_blocks = []
    for s in sources:
        formatted_blocks.append(
            f"[{s.source_id}] {s.title}\nURL: {s.url}\nContent: {s.content}"
        )
    return "\n\n".join(formatted_blocks)


def create_analyst_node(
    llm: BaseChatModel | None = None,
) -> Callable[[ResearchState], dict[str, list[Finding]]]:
    """Factory to create an analyst node with an optional injected LLM.

    Contract:
        Input: state['question'] (str), state['sources'] (list[Source])
        Output: {'findings': list[Finding]}

    Args:
        llm: Optional BaseChatModel instance. If None, initialized via get_chat_model().

    Returns:
        A callable node function conforming to LangGraph node specification.
    """

    def analyst_node(state: ResearchState) -> dict[str, list[Finding]]:
        """LangGraph node that synthesizes findings from question and sources.

        Contract:
            INPUT:
                question: str (required)
                sources: list[Source] (required)
            OUTPUT:
                findings: list[Finding]
        """
        question = state.get("question")
        if not question or not question.strip():
            raise ValueError("Analyst node requires a non-empty 'question' in state.")

        sources = state.get("sources", [])
        logger.info("[ANALYST] received %d sources for question: '%s'", len(sources), question)

        # Handle empty sources cleanly
        if not sources:
            logger.warning("[ANALYST] No sources available in state; returning 0 findings.")
            return {"findings": []}

        # Resolve LLM instance (lazy initialization if not injected)
        model = llm or get_chat_model()

        system_instruction = (
            "You are a rigorous, domain-agnostic research analyst.\n"
            "Your task is to synthesize clear, factual findings answering the research question "
            "based SOLELY on the provided sources.\n\n"
            "CRITICAL TRACEABILITY RULES:\n"
            "1. Each finding MUST cite at least one source_id from the provided sources.\n"
            "2. NEVER invent, hallucinate, or reference a source_id that does not appear in the sources list.\n"
            "3. Do not generate generic commentary. Every finding must represent a substantiated insight."
        )

        sources_context = _format_sources_for_prompt(sources)
        user_prompt = (
            f"Research Question: {question}\n\n"
            f"Retrieved Sources:\n"
            f"{sources_context}\n\n"
            "Synthesize structured findings answering the question. Associate each finding "
            "with its supporting source_ids."
        )

        try:
            structured_model = model.with_structured_output(AnalystOutput)
            result = structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as e:
            logger.error("[ANALYST] LLM execution or structured output parsing failed: %s", e)
            raise RuntimeError(f"Analyst LLM synthesis failed: {e}") from e

        if not isinstance(result, AnalystOutput):
            raise ValueError(
                f"Analyst expected structured AnalystOutput, got {type(result).__name__}"
            )

        findings = result.findings

        # Enforce source traceability and reject invented source IDs
        validate_finding_sources(findings, sources)

        logger.info("[Analyst] findings=%d", len(findings))
        return {"findings": findings}

    return analyst_node


# Default node instance using standard configuration
analyst_node = create_analyst_node()
