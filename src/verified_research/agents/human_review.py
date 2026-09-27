"""Human review node implementation for human-in-the-loop verification."""

import logging
from typing import Any, Callable
from langgraph.types import interrupt
from verified_research.graph.state import ResearchState
from verified_research.models.research import Claim, Evidence, HumanReview
from verified_research.models.traceability import DuplicateIdError, UnknownEvidenceError

logger = logging.getLogger(__name__)


def validate_edited_claims(
    edited_claims: list[Claim],
    available_evidence: list[Evidence],
) -> None:
    """Validate referential integrity of human-edited claims against existing evidence.

    Validation Rules:
        1. Non-empty: Must provide at least one edited claim.
        2. Identifier Uniqueness: Unique claim_ids across all edited claims.
        3. Evidence Reference: Every evidence_id in claim.evidence_ids must exist in available_evidence.

    Args:
        edited_claims: List of Claim objects supplied by the human editor.
        available_evidence: List of preserved Evidence excerpts present in state.

    Raises:
        ValueError: If edited_claims is empty.
        DuplicateIdError: If duplicate claim_ids are detected.
        UnknownEvidenceError: If any claim cites an evidence_id not present in available_evidence.
    """
    if not edited_claims:
        raise ValueError("Action 'edit' requires a non-empty list of 'edited_claims'.")

    # 1. Uniqueness check
    seen_ids = set()
    for claim in edited_claims:
        if claim.claim_id in seen_ids:
            raise DuplicateIdError(f"Duplicate claim_id detected in edited claims: '{claim.claim_id}'")
        seen_ids.add(claim.claim_id)

    # 2. Referential integrity check
    valid_evidence_ids = {e.evidence_id for e in available_evidence}
    for claim in edited_claims:
        for ev_id in claim.evidence_ids:
            if ev_id not in valid_evidence_ids:
                raise UnknownEvidenceError(
                    f"Edited claim '{claim.claim_id}' references unknown evidence_id '{ev_id}'. "
                    f"Available evidence IDs in state: {sorted(valid_evidence_ids)}"
                )


def build_review_payload(state: ResearchState) -> dict[str, Any]:
    """Construct a structured, serializable payload for human inspection prior to interrupt.

    Contains all claims, evidence excerpts, verification outcomes, and linked source context.
    Excludes any API keys, credentials, or sensitive state information.

    Args:
        state: Current ResearchState containing claims, evidence, and verification_results.

    Returns:
        Structured dictionary payload passed to LangGraph interrupt().
    """
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    verification_results = state.get("verification_results", [])
    sources = state.get("sources", [])

    evidence_map = {e.evidence_id: e for e in evidence}
    source_map = {s.source_id: s for s in sources}
    verification_map = {v.claim_id: v for v in verification_results}

    review_items = []
    for c in claims:
        item_evidence = []
        for ev_id in c.evidence_ids:
            ev = evidence_map.get(ev_id)
            if ev:
                src = source_map.get(ev.source_id)
                item_evidence.append(
                    {
                        "evidence_id": ev.evidence_id,
                        "source_id": ev.source_id,
                        "text": ev.text,
                        "source_title": src.title if src else None,
                        "source_url": src.url if src else None,
                    }
                )
        vr = verification_map.get(c.claim_id)
        review_items.append(
            {
                "claim_id": c.claim_id,
                "text": c.text,
                "evidence": item_evidence,
                "verification": (
                    {
                        "verdict": vr.verdict,
                        "confidence": vr.confidence,
                        "reasoning": vr.reasoning,
                        "evidence_ids": vr.evidence_ids,
                    }
                    if vr
                    else None
                ),
            }
        )

    return {
        "type": "human_review",
        "question": state.get("question", ""),
        "claims": [c.model_dump() for c in claims],
        "evidence": [e.model_dump() for e in evidence],
        "verification_results": [v.model_dump() for v in verification_results],
        "review_items": review_items,
        "human_research_cycles": state.get("human_research_cycles", 0),
    }


def create_human_review_node(
    interrupt_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Factory to create a human review node with optional injected interrupt callable.

    Contract:
        Input: state['claims'], state['evidence'], state['verification_results'], state.get('human_research_cycles', 0)
        Pauses via: interrupt(payload)
        Resume input: HumanReview instance or dict conforming to HumanReview schema
        Output:
            - approve: {'human_review': HumanReview}
            - edit: {'human_review': HumanReview, 'claims': list[Claim]}
            - research_more: {'human_review': HumanReview, 'human_feedback': str | None, 'research_iteration': 0}
            - reject: {'human_review': HumanReview}

    Args:
        interrupt_fn: Optional callable implementing interrupt behavior. Defaults to langgraph.types.interrupt.

    Returns:
        Callable node function conforming to LangGraph node specification.
    """
    active_interrupt = interrupt_fn if interrupt_fn is not None else interrupt

    def human_review_node(state: ResearchState) -> dict[str, Any]:
        """LangGraph node that pauses execution for human review and validates human decisions."""
        payload = build_review_payload(state)
        logger.info(
            "[HumanReview] Pausing graph for human review (%d claims, %d evidence)",
            len(state.get("claims", [])),
            len(state.get("evidence", [])),
        )

        raw_decision = active_interrupt(payload)

        # Validate human decision against structured schema
        if isinstance(raw_decision, HumanReview):
            validated_review = raw_decision
        elif isinstance(raw_decision, dict):
            validated_review = HumanReview.model_validate(raw_decision)
        else:
            raise ValueError(
                f"Human review decision must be a dict or HumanReview instance, got {type(raw_decision).__name__}"
            )

        logger.info("[HumanReview] Received human action: '%s'", validated_review.action)

        if validated_review.action == "edit":
            assert validated_review.edited_claims is not None
            validate_edited_claims(
                validated_review.edited_claims,
                state.get("evidence", []),
            )
            return {
                "human_review": validated_review,
                "claims": list(validated_review.edited_claims),
            }

        if validated_review.action == "research_more":
            return {
                "human_review": validated_review,
                "human_feedback": validated_review.feedback,
                "research_iteration": 0,  # Reset subgraph-internal counter for next research pass
            }

        # 'approve' or 'reject'
        return {
            "human_review": validated_review,
        }

    return human_review_node


# Default node instance using standard LangGraph interrupt
human_review_node = create_human_review_node()
