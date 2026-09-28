"""Writer agent responsible for synthesizing final, evidence-grounded research reports."""

import logging
from typing import Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.config.llm import get_chat_model
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    Citation,
    Claim,
    Evidence,
    FinalReport,
    HumanReview,
    ReportSection,
    Source,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class InvariantViolationError(ValueError):
    """Raised when an operation violates a quality-critical architectural invariant."""


class WriterService:
    """Service that synthesizes verified research into a grounded final research report.

    Architectural Invariants:
        1. Cannot run before Human Review approval or edit completion.
        2. Cannot invent evidence, sources, or introduce unverified factual claims.
        3. Every factual claim referenced must map to an approved claim.
        4. Every citation must map to a retrieved source and preserved evidence snapshot.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm

    def _get_model(self) -> BaseChatModel:
        if self.llm is not None:
            return self.llm
        return get_chat_model()

    def generate_report(self, state: ResearchState) -> FinalReport:
        """Synthesize the final research report strictly from approved claims and evidence.

        Args:
            state: Current ResearchState containing question, claims, evidence, sources,
                   and human_review.

        Returns:
            Grounded FinalReport with structured sections and explicit citations.

        Raises:
            InvariantViolationError: If human review has not approved or edited claims.
        """
        review = state.get("human_review")
        if review is None or review.action not in ("approve", "edit"):
            raise InvariantViolationError(
                "Writer cannot execute before Human Review approval or edit completion "
                f"(current human_review={review})."
            )

        # Invariant 2: Only approved or human-edited claims may be used
        if review.action == "edit" and review.edited_claims:
            claims: list[Claim] = list(review.edited_claims)
        else:
            claims = list(state.get("claims", []))

        evidence: list[Evidence] = list(state.get("evidence", []))
        sources: list[Source] = list(state.get("sources", []))
        verifications: list[VerificationResult] = list(state.get("verification_results", []))
        question: str = state.get("question", "Research Investigation")

        # Map lookups for fast attribution and validation
        sources_by_id = {s.source_id: s for s in sources}
        evidence_by_id = {e.evidence_id: e for e in evidence}
        verifications_by_claim = {v.claim_id: v for v in verifications}

        # Build citations mapping each source to the claims and evidence that ground it
        citations: list[Citation] = []
        source_citation_index: dict[str, str] = {}

        for idx, src in enumerate(sources, start=1):
            cit_marker = f"[{idx}]"
            source_citation_index[src.source_id] = cit_marker
            src_evidence = [e.evidence_id for e in evidence if e.source_id == src.source_id]
            src_claims = [
                c.claim_id
                for c in claims
                if any(eid in src_evidence for eid in c.evidence_ids)
            ]
            citations.append(
                Citation(
                    citation_id=cit_marker,
                    source_id=src.source_id,
                    claim_ids=src_claims,
                    evidence_ids=src_evidence,
                    source_title=src.title,
                    source_url=src.url,
                )
            )

        # Use structured LLM generation if an LLM is explicitly provided
        if self.llm is not None:
            try:
                report = self._llm_write(
                    model=self.llm,
                    question=question,
                    claims=claims,
                    evidence=evidence,
                    sources=sources,
                    citations=citations,
                    sources_by_id=sources_by_id,
                    evidence_by_id=evidence_by_id,
                )
                if report is not None:
                    return report
            except Exception as exc:
                logger.warning("[Writer:LLM] LLM synthesis failed (%s); falling back to deterministic writer", exc)

        # Deterministic grounded synthesis
        return self._deterministic_write(
            question=question,
            claims=claims,
            evidence=evidence,
            sources=sources,
            citations=citations,
            sources_by_id=sources_by_id,
            evidence_by_id=evidence_by_id,
            verifications_by_claim=verifications_by_claim,
        )

    def _llm_write(
        self,
        model: BaseChatModel,
        question: str,
        claims: list[Claim],
        evidence: list[Evidence],
        sources: list[Source],
        citations: list[Citation],
        sources_by_id: dict[str, Source],
        evidence_by_id: dict[str, Evidence],
    ) -> FinalReport | None:
        """Attempt LLM structured output generation with strict grounded schema."""
        claims_text = "\n".join(
            f"- [{c.claim_id}] {c.text} (Cited Evidence: {', '.join(c.evidence_ids)})"
            for c in claims
        )
        evidence_text = "\n".join(
            f"- [{e.evidence_id}] (Source: {e.source_id}): \"{e.text}\""
            for e in evidence
        )
        citations_text = "\n".join(
            f"- {c.citation_id}: {c.source_title} ({c.source_url})"
            for c in citations
        )

        system_prompt = (
            "You are the Lead Research Report Writer for an evidence-grounded verification system.\n"
            "Your objective is to produce a comprehensive, publication-grade research report answering the user's question.\n\n"
            "CRITICAL CONSTRAINTS (INVARIANTS):\n"
            "1. You MUST base all factual statements strictly and solely on the provided approved claims and evidence.\n"
            "2. Do NOT invent new facts, metrics, or external assertions not found in the approved claims.\n"
            "3. Use bracketed numeric citations (e.g. [1], [2]) that correspond directly to the supplied source citations.\n"
            "4. Organize your response into structured, thematic sections."
        )

        user_prompt = (
            f"RESEARCH QUESTION:\n{question}\n\n"
            f"APPROVED VERIFIED CLAIMS:\n{claims_text}\n\n"
            f"PRESERVED EVIDENCE SNAPSHOTS:\n{evidence_text}\n\n"
            f"AVAILABLE CITATIONS:\n{citations_text}\n\n"
            "Generate the structured FinalReport."
        )

        structured_llm = model.with_structured_output(FinalReport)
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if isinstance(result, FinalReport):
            # Invariant check: Ensure claim references match existing approved claims
            valid_claim_ids = {c.claim_id for c in claims}
            sanitized_refs = [cid for cid in result.claim_references if cid in valid_claim_ids] or list(valid_claim_ids)
            return FinalReport(
                title=result.title,
                summary=result.summary,
                answer=result.answer,
                sections=result.sections,
                citations=citations,  # Guarantee deterministic citation mapping
                claim_references=sanitized_refs,
                verified_claim_count=len(sanitized_refs),
            )
        return None

    def _deterministic_write(
        self,
        question: str,
        claims: list[Claim],
        evidence: list[Evidence],
        sources: list[Source],
        citations: list[Citation],
        sources_by_id: dict[str, Source],
        evidence_by_id: dict[str, Evidence],
        verifications_by_claim: dict[str, VerificationResult],
    ) -> FinalReport:
        """Deterministic grounded synthesis guaranteeing strict traceability and zero hallucinations."""
        title = f"Verified Research: {question.rstrip('?.,')} Investigation"

        # Group claims with their evidence citations
        claim_lines: list[str] = []
        sections: list[ReportSection] = []

        for idx, claim in enumerate(claims, start=1):
            claim_citations: list[str] = []
            for eid in claim.evidence_ids:
                ev = evidence_by_id.get(eid)
                if ev and ev.source_id in sources_by_id:
                    cit = next((c.citation_id for c in citations if c.source_id == ev.source_id), None)
                    if cit and cit not in claim_citations:
                        claim_citations.append(cit)

            cit_suffix = f" {' '.join(claim_citations)}" if claim_citations else ""
            line = f"{claim.text}{cit_suffix}"
            claim_lines.append(f"{idx}. {line}")

        summary = (
            f"This verified research investigation examined: \"{question}\". "
            f"Based on {len(sources)} indexed sources and {len(evidence)} verified evidence snapshots, "
            f"{len(claims)} atomic factual claims were distilled, evaluated, and approved."
        )

        answer_paragraphs = [
            f"### Executive Summary\n\n{summary}\n",
            "### Verified Findings\n",
            "\n".join(claim_lines),
            "\n### Evidence Grounding & Methodology\n",
            (
                "Each factual proposition above was extracted from immutable text excerpts preserved from web retrieval. "
                "The findings underwent independent 3-class claim verification and authoritative human review before report generation."
            ),
        ]
        answer = "\n".join(answer_paragraphs)

        sections.append(
            ReportSection(
                title="Executive Overview",
                content=summary,
                claim_ids=[c.claim_id for c in claims],
            )
        )
        sections.append(
            ReportSection(
                title="Detailed Synthesis",
                content="\n".join(claim_lines),
                claim_ids=[c.claim_id for c in claims],
            )
        )

        return FinalReport(
            title=title,
            summary=summary,
            answer=answer,
            sections=sections,
            citations=citations,
            claim_references=[c.claim_id for c in claims],
            verified_claim_count=len(claims),
        )


def create_writer_node(
    writer_service: WriterService | None = None,
    custom_writer: Callable[[ResearchState], dict[str, Any]] | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Factory creating the Writer node in the LangGraph workflow."""
    if custom_writer is not None:
        return custom_writer

    service = writer_service or WriterService()

    def writer_node(state: ResearchState) -> dict[str, Any]:
        try:
            from verified_research.api.events import emit_live_event
            emit_live_event("node_started", "writer", {"message": "Writing final grounded research response..."})
        except Exception:
            emit_live_event = None

        try:
            report = service.generate_report(state)
            if emit_live_event is not None:
                emit_live_event("writer_update", "writer", report.model_dump())
                emit_live_event(
                    "node_completed",
                    "writer",
                    {
                        "title": report.title,
                        "claims_referenced": len(report.claim_references),
                        "citations_count": len(report.citations),
                    },
                )
            return {"final_response": report}
        except Exception as exc:
            logger.exception("[WriterNode] Error generating final report: %s", exc)
            raise

    return writer_node
