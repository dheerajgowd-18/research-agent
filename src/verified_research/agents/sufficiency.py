"""Sufficiency evaluation and reuse synthesis agents for follow-up research sessions."""

import logging
import re
from typing import Any, Callable, Protocol, runtime_checkable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.config.llm import get_chat_model
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    AnalystOutput,
    Claim,
    Evidence,
    Finding,
    ResearchReuseDecision,
    Source,
)
from verified_research.models.traceability import (
    generate_claims_and_evidence_from_findings,
    validate_traceability,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class SufficiencyService(Protocol):
    """Protocol for research sufficiency evaluation."""

    def evaluate_sufficiency(
        self,
        question: str,
        sources: list[Source],
        findings: list[Finding],
        claims: list[Claim],
        evidence: list[Evidence],
        previous_questions: list[str] | None = None,
    ) -> ResearchReuseDecision:
        """Evaluate whether existing research is sufficient to answer the follow-up question."""
        ...


class HeuristicSufficiencyService:
    """Deterministic, rules-based sufficiency evaluator.

    Evaluates:
        1. Temporal mismatch: If the question requests a specific year/decade absent from evidence.
        2. Entity/topic mismatch: If the core subject matter is absent from existing evidence/sources.
        3. Sufficiency: If temporal constraints and entity coverage are satisfied by existing evidence.
    """

    def evaluate_sufficiency(
        self,
        question: str,
        sources: list[Source],
        findings: list[Finding],
        claims: list[Claim],
        evidence: list[Evidence],
        previous_questions: list[str] | None = None,
    ) -> ResearchReuseDecision:
        clean_q = question.strip()
        all_text = " ".join(
            [e.text for e in evidence]
            + [s.content for s in sources]
            + [s.title for s in sources]
            + [f.text for f in findings]
            + [c.text for c in claims]
        ).lower()

        # 1. Temporal Check: extract 4-digit years (e.g. 2024, 2026)
        query_years = re.findall(r"\b(19\d\d|20\d\d)\b", clean_q)
        for year in query_years:
            if year not in all_text:
                logger.info("[Sufficiency] Temporal mismatch: year '%s' not in existing research", year)
                return ResearchReuseDecision(
                    decision="RESEARCH_MORE",
                    reasoning=f"Follow-up requests information from year '{year}', which is not present in existing evidence.",
                    missing_topics=[f"Temporal coverage for year {year}"],
                )

        # 2. Entity / Keyword Coverage Check
        # Extract substantive words (length >= 4, ignoring common stop words)
        stop_words = {
            "what", "which", "when", "where", "whom", "whose", "why", "how",
            "those", "these", "that", "this", "from", "with", "about", "into",
            "more", "most", "some", "such", "than", "then", "there", "their",
            "matter", "explain", "detail", "compare", "versus", "against",
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", clean_q.lower())
        substantive_words = [w for w in words if w not in stop_words]

        if not substantive_words:
            # Short / generic query referencing prior context -> REUSE
            return ResearchReuseDecision(
                decision="REUSE",
                reasoning="Generic follow-up referencing existing research findings.",
                missing_topics=[],
            )

        matched_words = [w for w in substantive_words if w in all_text]
        coverage_ratio = len(matched_words) / len(substantive_words)

        if coverage_ratio < 0.35:
            missing = [w for w in substantive_words if w not in all_text]
            logger.info(
                "[Sufficiency] Entity mismatch: coverage %.2f < 0.35. Missing: %s",
                coverage_ratio,
                missing,
            )
            return ResearchReuseDecision(
                decision="RESEARCH_MORE",
                reasoning=f"Follow-up introduces topics or entities not covered in existing research ({', '.join(missing[:3])}).",
                missing_topics=missing[:3],
            )

        # 3. Sufficient: Evidence covers the requested scope
        logger.info("[Sufficiency] Existing research is sufficient (coverage %.2f)", coverage_ratio)
        return ResearchReuseDecision(
            decision="REUSE",
            reasoning="Existing findings and evidence excerpts comprehensively address the follow-up question.",
            missing_topics=[],
        )


class LLMSufficiencyService:
    """LLM-based sufficiency evaluator with structured output validation."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def _get_model(self) -> BaseChatModel:
        if self.llm is not None:
            return self.llm
        return get_chat_model()

    def evaluate_sufficiency(
        self,
        question: str,
        sources: list[Source],
        findings: list[Finding],
        claims: list[Claim],
        evidence: list[Evidence],
        previous_questions: list[str] | None = None,
    ) -> ResearchReuseDecision:
        model = self._get_model()

        system_instruction = (
            "You are a strict, objective research sufficiency auditor.\n"
            "Your task is to determine whether existing research, findings, and evidence are strictly "
            "sufficient to answer a new follow-up question, or if additional web research is required.\n\n"
            "DECISION CRITERIA:\n"
            "1. REUSE: Set decision to 'REUSE' ONLY if the existing evidence excerpts and synthesized findings "
            "directly, factually, and completely substantiate the answer to the follow-up question without speculation.\n"
            "2. RESEARCH_MORE: Set decision to 'RESEARCH_MORE' if:\n"
            "   - Temporal Scope Mismatch: The follow-up requests a newer timeframe (e.g., 2026 vs 2024).\n"
            "   - Entity/Subject Mismatch: The follow-up introduces entities or concepts not in evidence.\n"
            "   - Incomplete Coverage: Answering would require external world knowledge or guessing.\n\n"
            "Provide explicit reasoning and list any missing topics."
        )

        evidence_summary = "\n".join(
            f"[{e.evidence_id}] {e.text}" for e in evidence[:10]
        )
        findings_summary = "\n".join(
            f"[{f.finding_id}] {f.text}" for f in findings[:10]
        )
        prev_q_str = ", ".join(f"'{q}'" for q in (previous_questions or []))

        user_content = (
            f"FOLLOW-UP QUESTION:\n{question}\n\n"
            f"PREVIOUS QUESTIONS IN THREAD:\n{prev_q_str}\n\n"
            f"EXISTING FINDINGS ({len(findings)} items):\n{findings_summary}\n\n"
            f"EXISTING EVIDENCE ({len(evidence)} items):\n{evidence_summary}\n\n"
            "Evaluate whether existing research is sufficient (REUSE) or if new research is required (RESEARCH_MORE)."
        )

        try:
            structured_model = model.with_structured_output(ResearchReuseDecision)
            result = structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content),
                ]
            )
            if not isinstance(result, ResearchReuseDecision):
                raise ValueError(
                    f"Expected ResearchReuseDecision, got {type(result).__name__}"
                )
            return result
        except Exception as e:
            logger.warning("[Sufficiency] LLM evaluation failed (%s); falling back to heuristic", e)
            fallback = HeuristicSufficiencyService()
            return fallback.evaluate_sufficiency(
                question=question,
                sources=sources,
                findings=findings,
                claims=claims,
                evidence=evidence,
                previous_questions=previous_questions,
            )


def create_evaluate_sufficiency_node(
    sufficiency_service: SufficiencyService | None = None,
    llm: BaseChatModel | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Factory to create an evaluate_sufficiency node.

    Contract:
        Input: state['question'], optional state['follow_up_question'], state.get('sources', []), state.get('evidence', [])
        Output:
            - question: updated active question
            - previous_questions: updated list of questions in session
            - reuse_decision: ResearchReuseDecision
            - research_iteration: reset to 0 if RESEARCH_MORE
    """
    if sufficiency_service is not None:
        service = sufficiency_service
    elif llm is not None:
        service = LLMSufficiencyService(llm=llm)
    else:
        service = HeuristicSufficiencyService()

    def evaluate_sufficiency_node(state: ResearchState) -> dict[str, Any]:
        follow_up = state.get("follow_up_question")
        if follow_up and follow_up.strip():
            active_question = follow_up.strip()
        else:
            active_question = state.get("question", "").strip()

        if not active_question:
            raise ValueError("Sufficiency evaluation requires a non-empty 'question' in state.")

        sources = state.get("sources", [])
        evidence = state.get("evidence", [])
        findings = state.get("findings", [])
        claims = state.get("claims", [])
        previous_questions = list(state.get("previous_questions", []))

        # Initial research: no prior sources exist
        if not sources or not evidence:
            decision = ResearchReuseDecision(
                decision="RESEARCH_MORE",
                reasoning="Initial research query; no prior research exists in session.",
                missing_topics=[],
            )
            if active_question not in previous_questions:
                previous_questions.append(active_question)
            return {
                "question": active_question,
                "previous_questions": previous_questions,
                "reuse_decision": decision,
            }

        # Follow-up session: evaluate existing research sufficiency
        decision = service.evaluate_sufficiency(
            question=active_question,
            sources=sources,
            findings=findings,
            claims=claims,
            evidence=evidence,
            previous_questions=previous_questions,
        )

        logger.info(
            "[Sufficiency] Evaluated question '%s': decision=%s reasoning='%s'",
            active_question,
            decision.decision,
            decision.reasoning,
        )

        if active_question not in previous_questions:
            previous_questions.append(active_question)

        updates: dict[str, Any] = {
            "question": active_question,
            "previous_questions": previous_questions,
            "reuse_decision": decision,
        }

        # If more research is needed, reset internal research iteration for the new pass
        if decision.decision == "RESEARCH_MORE":
            updates["research_iteration"] = 0
            updates["human_research_cycles"] = 0

        return updates

    return evaluate_sufficiency_node


def create_reuse_analyst_node(
    llm: BaseChatModel | None = None,
    custom_reuse_fn: Callable[[str, list[Source], list[Evidence], list[Finding]], tuple[list[Finding], list[Claim]]] | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Factory creating an analyst node specialized in synthesizing follow-up claims from existing evidence without web search.

    Contract:
        Input: state['question'], state['sources'], state['evidence'], state['findings']
        Output:
            - findings: list[Finding]
            - claims: list[Claim]
    """

    def reuse_analyst_node(state: ResearchState) -> dict[str, Any]:
        question = state.get("question", "")
        sources = state.get("sources", [])
        evidence = state.get("evidence", [])
        existing_findings = state.get("findings", [])

        if not sources or not evidence:
            return {"findings": [], "claims": []}

        if custom_reuse_fn is not None:
            new_findings, new_claims = custom_reuse_fn(
                question, sources, evidence, existing_findings
            )
            validate_traceability(new_claims, evidence, sources)
            return {"findings": new_findings, "claims": new_claims}

        # Default synthesis using LLM or deterministic fallback
        if llm is not None:
            system_instruction = (
                "You are an analytical research synthesis expert.\n"
                "Your task is to synthesize findings and testable claims to answer the follow-up question "
                "SOLEY using the provided sources and existing evidence.\n"
                "Do NOT reference external information. Associate each finding with supporting source IDs."
            )
            prompt = (
                f"FOLLOW-UP QUESTION:\n{question}\n\n"
                f"AVAILABLE SOURCES:\n"
                + "\n".join(f"[{s.source_id}] {s.title}: {s.content}" for s in sources)
            )
            structured_model = llm.with_structured_output(AnalystOutput)
            result = structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=prompt),
                ]
            )
            findings = result.findings
            _, claims = generate_claims_and_evidence_from_findings(findings, sources)
            validate_traceability(claims, evidence, sources)
            return {"findings": findings, "claims": claims}

        # Deterministic fallback: filter existing findings and generate scoped follow-up claims
        # matching the follow-up question
        scoped_claims: list[Claim] = []
        scoped_findings: list[Finding] = []

        for idx, f in enumerate(existing_findings, start=1):
            claim_id = f"reuse_claim_{idx:03d}"
            # Resolve all evidence matching this finding's sources
            matched_ev_ids = [
                e.evidence_id for e in evidence if e.source_id in f.source_ids
            ]
            if matched_ev_ids:
                scoped_findings.append(f)
                scoped_claims.append(
                    Claim(
                        claim_id=claim_id,
                        text=f"Regarding '{question}': {f.text}",
                        evidence_ids=matched_ev_ids,
                    )
                )

        if not scoped_claims and evidence:
            # If no findings matched, create claim directly from first available evidence
            scoped_claims.append(
                Claim(
                    claim_id="reuse_claim_001",
                    text=f"Direct synthesis for '{question}': {evidence[0].text}",
                    evidence_ids=[evidence[0].evidence_id],
                )
            )

        validate_traceability(scoped_claims, evidence, sources)
        return {"findings": scoped_findings, "claims": scoped_claims}

    return reuse_analyst_node


# Default node instances
evaluate_sufficiency_node = create_evaluate_sufficiency_node()
reuse_analyst_node = create_reuse_analyst_node()
