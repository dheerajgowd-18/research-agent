"""Evidence-grounded claim verifier node and verification service."""

import logging
from typing import Any, Callable, Protocol, runtime_checkable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.config.llm import get_chat_model
from verified_research.graph.state import ResearchState
from verified_research.models.research import Claim, Evidence, VerificationResult
from verified_research.models.traceability import UnknownEvidenceError

logger = logging.getLogger(__name__)


@runtime_checkable
class VerifierService(Protocol):
    """Protocol for claim verification implementations."""

    def verify_claim(self, claim: Claim, evidence_items: list[Evidence]) -> VerificationResult:
        """Evaluate whether the supplied evidence supports the given claim."""
        ...


class ClaimVerifierService:
    """LLM-based verifier service performing evidence-grounded entailment checks."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        retry_policy: Any | None = None,
    ) -> None:
        from verified_research.reliability.policy import RetryPolicy

        self.llm = llm
        self.retry_policy = retry_policy or RetryPolicy()

    def _get_model(self) -> BaseChatModel:
        if self.llm is not None:
            return self.llm
        return get_chat_model()

    def verify_claim(self, claim: Claim, evidence_items: list[Evidence]) -> VerificationResult:
        """Evaluate a claim against its specific supporting evidence snapshots.

        Strict Rules:
            1. Judge ONLY the supplied evidence. Do NOT use outside knowledge or web search.
            2. Verdict definitions:
               - SUPPORTED: The evidence directly supports the substantive factual proposition.
               - PARTIAL: The evidence supports only part of the claim, or a narrower scope.
               - UNSUPPORTED: The evidence contradicts the claim, is irrelevant, or provides no support.
            3. Check numbers, dates, entities, scope (all vs some), and temporal qualifiers.

        Args:
            claim: The atomic Claim to verify.
            evidence_items: The specific Evidence excerpts grounding the claim.

        Returns:
            A VerificationResult detailing verdict, confidence, reasoning, and evidence IDs.
        """
        from verified_research.reliability.policy import execute_with_retry

        model = self._get_model()

        system_instruction = (
            "You are a strict, objective evidence verifier for scientific and technical research.\n"
            "Your sole objective is to evaluate: 'Does the supplied evidence support the supplied claim?'\n\n"
            "VERDICT CRITERIA:\n"
            "- SUPPORTED: The supplied evidence directly, fully, and factually supports the claim.\n"
            "- PARTIAL: The evidence supports only part of a multi-part claim, or supports a strictly "
            "narrower scope (e.g. claim says 'all' but evidence says 'several').\n"
            "- UNSUPPORTED: The evidence contradicts the claim, is irrelevant, has mismatched entities, "
            "has mismatched numbers/dates, or fails to substantiate the assertion.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. Base your verdict EXCLUSIVELY on the supplied evidence excerpts.\n"
            "2. Do NOT use outside world knowledge or unstated assumptions.\n"
            "3. Do NOT search for external sources.\n"
            "4. Pay close attention to numbers, dates, entities, qualifiers, and negation.\n"
            "5. Provide concise, factual reasoning explaining your decision.\n"
            "6. Assign a model confidence score between 0.0 and 1.0 (representing model certainty, "
            "not empirical accuracy)."
        )

        formatted_evidence_blocks = []
        for idx, ev in enumerate(evidence_items, start=1):
            formatted_evidence_blocks.append(
                f"EVIDENCE {idx} [{ev.evidence_id}] (Source: {ev.source_id}):\n{ev.text}"
            )
        evidence_text = "\n\n".join(formatted_evidence_blocks)

        user_content = (
            f"CLAIM TO EVALUATE [{claim.claim_id}]:\n{claim.text}\n\n"
            f"SUPPLIED EVIDENCE ({len(evidence_items)} items):\n{evidence_text}\n\n"
            "Evaluate whether the supplied evidence supports the claim according to the strict criteria."
        )

        def _do_verification() -> Any:
            structured_model = model.with_structured_output(VerificationResult)
            return structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content),
                ]
            )

        try:
            result = execute_with_retry(
                operation=_do_verification,
                policy=self.retry_policy,
                component="verifier",
                operation_name="llm_verification",
                reraise_original=True,
            )
        except Exception as e:
            logger.error("[VERIFIER] LLM verification failed for claim '%s': %s", claim.claim_id, e)
            raise RuntimeError(f"Claim verification failed for '{claim.claim_id}': {e}") from e

        if not isinstance(result, VerificationResult):
            raise ValueError(
                f"Verifier expected VerificationResult model, got {type(result).__name__}"
            )

        # Enforce invariant: claim_id and evidence_ids must match input
        if result.claim_id != claim.claim_id or set(result.evidence_ids) != set(claim.evidence_ids):
            result = VerificationResult(
                claim_id=claim.claim_id,
                verdict=result.verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
                evidence_ids=list(claim.evidence_ids),
            )

        return result


def create_verifier_node(
    verifier_service: VerifierService | None = None,
    llm: BaseChatModel | None = None,
    custom_verifier: Callable[[Claim, list[Evidence]], VerificationResult] | None = None,
) -> Callable[[ResearchState], dict[str, list[VerificationResult]]]:
    """Factory to create a verifier node with optional injected service or custom mock function.

    Contract:
        Input: state['claims'] (list[Claim]), state['evidence'] (list[Evidence])
        Output: {'verification_results': list[VerificationResult]}

    Args:
        verifier_service: Optional service implementing VerifierService protocol.
        llm: Optional BaseChatModel instance used by default ClaimVerifierService.
        custom_verifier: Optional custom callable taking (claim, evidence_list) -> VerificationResult.

    Returns:
        A callable node function conforming to LangGraph node specification.
    """
    if custom_verifier is not None:
        verify_fn = custom_verifier
    elif verifier_service is not None:
        verify_fn = verifier_service.verify_claim
    else:
        default_service = ClaimVerifierService(llm=llm)
        verify_fn = default_service.verify_claim

    def verifier_node(state: ResearchState) -> dict[str, list[VerificationResult]]:
        """LangGraph node that verifies each claim in state against its cited evidence.

        Contract:
            INPUT:
                claims: list[Claim] (optional, default [])
                evidence: list[Evidence] (optional, default [])
            OUTPUT:
                verification_results: list[VerificationResult]
        """
        claims = state.get("claims", [])
        evidence = state.get("evidence", [])

        logger.info("[Verifier] claims=%d", len(claims))

        if not claims:
            return {"verification_results": []}

        # Build evidence lookup
        evidence_map = {ev.evidence_id: ev for ev in evidence}

        results: list[VerificationResult] = []

        from verified_research.observability.metadata import build_verifier_claim_metadata
        from verified_research.observability.tracer import record_metadata, trace_span

        for claim in claims:
            # Evidence resolution: resolve all evidence_ids cited by the claim
            resolved_evidence: list[Evidence] = []
            for ev_id in claim.evidence_ids:
                if ev_id not in evidence_map:
                    raise UnknownEvidenceError(
                        f"Claim '{claim.claim_id}' references unknown evidence_id '{ev_id}'. "
                        f"Available evidence IDs in state: {sorted(evidence_map.keys())}"
                    )
                resolved_evidence.append(evidence_map[ev_id])

            # Perform verification strictly on supplied evidence within structured trace span
            with trace_span(
                name=f"verify_claim:{claim.claim_id}",
                run_type="chain",
                metadata={"claim_id": claim.claim_id, "evidence_count": len(resolved_evidence)},
                tags=["verifier", "claim"],
            ):
                result = verify_fn(claim, resolved_evidence)
                try:
                    meta = build_verifier_claim_metadata(
                        claim_id=result.claim_id,
                        verdict=result.verdict,
                        confidence=result.confidence,
                        evidence_count=len(result.evidence_ids),
                        reasoning=result.reasoning,
                    )
                    record_metadata(**meta)
                except Exception:
                    pass

            logger.info(
                "[Verifier] claim_id=%s verdict=%s",
                result.claim_id,
                result.verdict,
            )
            results.append(result)

        # Enforce 1:1 invariant: exactly one VerificationResult per claim
        if len(results) != len(claims):
            raise RuntimeError(
                f"Verifier invariant violated: expected {len(claims)} results, produced {len(results)}"
            )

        try:
            record_metadata(claims_verified_count=len(results))
        except Exception:
            pass

        return {"verification_results": results}

    return verifier_node


# Default node instance using standard configuration
verifier_node = create_verifier_node()
