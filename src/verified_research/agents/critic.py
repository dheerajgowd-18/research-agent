"""Critic node implementation for research evaluation and refinement recommendations."""

import logging
from typing import Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.config.llm import get_chat_model
from verified_research.graph.state import ResearchState
from verified_research.models.research import Critique, Finding, Source

logger = logging.getLogger(__name__)


def _format_findings_for_critic(findings: list[Finding]) -> str:
    """Format finding models into readable text for the critic prompt."""
    blocks = []
    for f in findings:
        sources_str = ", ".join(f.source_ids)
        blocks.append(f"[{f.finding_id}] {f.text}\n  Cited Sources: [{sources_str}]")
    return "\n\n".join(blocks)


def _format_sources_summary_for_critic(sources: list[Source]) -> str:
    """Format source headers for the critic prompt."""
    blocks = []
    for s in sources:
        blocks.append(f"[{s.source_id}] {s.title} ({s.url})")
    return "\n".join(blocks)


def create_critic_node(
    llm: BaseChatModel | None = None,
    retry_policy: Any | None = None,
) -> Callable[[ResearchState], dict[str, Critique]]:
    """Factory to create a critic node with an optional injected LLM and retry policy.

    Contract:
        Input: state['question'] (str), state['sources'] (list[Source]), state['findings'] (list[Finding])
        Output: {'critique': Critique}

    Args:
        llm: Optional BaseChatModel instance. If None, initialized via get_chat_model().
        retry_policy: Optional RetryPolicy for LLM invocation reliability.

    Returns:
        A callable node function conforming to LangGraph node specification.
    """
    from verified_research.reliability.policy import RetryPolicy, execute_with_retry

    active_policy = retry_policy or RetryPolicy()

    def critic_node(state: ResearchState) -> dict[str, Critique]:
        """LangGraph node that evaluates findings and determines if more research is required.

        Contract:
            INPUT:
                question: str (required)
                sources: list[Source] (optional, default [])
                findings: list[Finding] (optional, default [])
            OUTPUT:
                critique: Critique
        """
        question = state.get("question")
        follow_up = state.get("follow_up_question")
        active_question = (
            follow_up.strip()
            if follow_up and follow_up.strip()
            else (question.strip() if question else "")
        )
        if not active_question:
            raise ValueError("Critic node requires a non-empty 'question' in state.")

        clean_question = active_question
        sources = state.get("sources", [])
        findings = state.get("findings", [])

        # If no findings were produced, automatically flag need for research
        if not findings:
            critique = Critique(
                quality_score=0.0,
                missing_topics=["No findings were produced for the question."],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=[clean_question],
                should_research_again=True,
            )
            logger.info(
                "[Critic] score=%.2f continue=%s",
                critique.quality_score,
                critique.should_research_again,
            )
            return {"critique": critique}

        model = llm or get_chat_model()

        system_instruction = (
            "You are a critical research auditor and fact-checking evaluator.\n"
            "Your objective is to evaluate whether current findings adequately, accurately, "
            "and thoroughly answer the research question.\n\n"
            "CRITIQUE CRITERIA:\n"
            "1. Answer Completeness: Do the findings answer the core question?\n"
            "2. Missing Topics: Note important angles, counterpoints, or nuances omitted.\n"
            "3. Weak Findings: Identify vague, repetitive, or poorly supported statements.\n"
            "4. Citation Gaps: Identify claims with insufficient or questionable source attribution.\n"
            "5. Recommended Queries: If research is deficient, suggest 1-3 targeted search queries.\n"
            "6. Iteration Decision: Set 'should_research_again' to True ONLY if major gaps exist and "
            "further research would meaningfully improve quality. If findings are solid, set to False.\n"
            "7. Quality Score: Assign a score from 0.0 (inadequate) to 1.0 (exemplary)."
        )

        user_content = (
            f"Original Question:\n{clean_question}\n\n"
            f"Retrieved Sources ({len(sources)}):\n{_format_sources_summary_for_critic(sources)}\n\n"
            f"Synthesized Findings ({len(findings)}):\n{_format_findings_for_critic(findings)}\n\n"
            "Evaluate the quality of the findings and provide your structured critique."
        )

        def _do_critique() -> Any:
            structured_model = model.with_structured_output(Critique)
            return structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content),
                ]
            )

        try:
            result = execute_with_retry(
                operation=_do_critique,
                policy=active_policy,
                component="critic",
                operation_name="llm_critique",
                reraise_original=True,
            )
        except Exception as e:
            logger.error("[CRITIC] LLM structured evaluation failed: %s", e)
            raise RuntimeError(f"Critic evaluation failed: {e}") from e

        if not isinstance(result, Critique):
            raise ValueError(f"Critic expected Critique model, got {type(result).__name__}")

        logger.info(
            "[Critic] score=%.2f continue=%s",
            result.quality_score,
            result.should_research_again,
        )
        return {"critique": result}

    return critic_node


# Default node instance
critic_node = create_critic_node()
