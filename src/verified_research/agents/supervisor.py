"""Supervisor orchestration agent and policies for the Verified Research Agent."""

import logging
from typing import Any, Callable, Protocol, runtime_checkable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from verified_research.agents.sufficiency import (
    HeuristicSufficiencyService,
    create_reuse_analyst_node,
)
from verified_research.config.llm import get_chat_model
from verified_research.config.settings import (
    DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
    DEFAULT_MAX_SUPERVISOR_STEPS,
)
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    ResearchReuseDecision,
    SupervisorDecision,
    SupervisorWorkerType,
)

logger = logging.getLogger(__name__)


class InvariantViolationError(ValueError):
    """Raised when a supervisor decision violates a quality-critical architectural invariant."""


@runtime_checkable
class SupervisorPolicy(Protocol):
    """Protocol for supervisor decision-making policies."""

    def evaluate(self, state: ResearchState) -> SupervisorDecision:
        """Evaluate current graph state and determine the next specialized worker."""
        ...


def validate_supervisor_decision(
    state: ResearchState,
    decision: SupervisorDecision,
    max_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS,
    max_human_cycles: int = DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
) -> SupervisorDecision:
    """Enforce architectural invariants on supervisor decisions.

    The supervisor is an orchestration layer, NOT an unconstrained source of truth.
    Quality-critical invariants cannot be bypassed by any policy (especially LLMs):

    Invariants:
        1. Step Limit: If supervisor_steps >= max_steps, must finish.
        2. Verification Guard: Claims cannot bypass verification to human review or finish.
           If claims exist in state and are not fully verified, next_worker MUST be 'verify'.
        3. Research Grounding Guard: Cannot verify if no claims exist in state.
           Must route to 'research'.
        4. Human Decision Finality: If human review action is approve/edit/reject,
           execution must finish.
        5. Human Research Cycle Bound: If human requested research_more, but cycles
           exceed max_human_cycles, execution must finish.

    Args:
        state: Current ResearchState.
        decision: The candidate SupervisorDecision produced by a policy.
        max_steps: Maximum supervisor steps allowed before mandatory finish.
        max_human_cycles: Maximum human-requested research passes allowed.

    Returns:
        The validated (or safely overridden) SupervisorDecision.
    """
    current_steps = state.get("supervisor_steps", 0)

    # Invariant 1: Step Limit
    if current_steps >= max_steps:
        logger.warning(
            "[Supervisor:Invariant] Step limit reached (%d >= %d). Overriding to 'finish'.",
            current_steps,
            max_steps,
        )
        return SupervisorDecision(
            next_worker="finish",
            reasoning=f"Supervisor step limit reached ({current_steps} >= {max_steps}); terminating safely.",
        )

    # Invariant 4 & 5: Human Review Action Finality & Bounds
    review = state.get("human_review")
    if review is not None:
        if review.action in ("approve", "edit", "reject"):
            if decision.next_worker != "finish":
                logger.warning(
                    "[Supervisor:Invariant] Human action '%s' concludes workflow. Overriding '%s' to 'finish'.",
                    review.action,
                    decision.next_worker,
                )
                return SupervisorDecision(
                    next_worker="finish",
                    reasoning=f"Human review finalized with action '{review.action}'.",
                )
        elif review.action == "research_more":
            cycles = state.get("human_research_cycles", 0)
            if cycles >= max_human_cycles and decision.next_worker != "finish":
                logger.warning(
                    "[Supervisor:Invariant] Human research cycle cap reached (%d >= %d). Overriding to 'finish'.",
                    cycles,
                    max_human_cycles,
                )
                return SupervisorDecision(
                    next_worker="finish",
                    reasoning=f"Human research cycle limit reached ({cycles} >= {max_human_cycles}).",
                )

    claims = state.get("claims", [])
    verifications = state.get("verification_results", [])

    # Invariant 3: Cannot verify without claims
    if not claims and decision.next_worker == "verify":
        logger.warning(
            "[Supervisor:Invariant] Cannot verify without claims. Overriding 'verify' to 'research'."
        )
        return SupervisorDecision(
            next_worker="research",
            reasoning="Cannot invoke verifier without claims in state; routing to research.",
        )

    # Invariant 2: Unverified claims cannot bypass verification
    if claims:
        verified_claim_ids = {v.claim_id for v in verifications}
        unverified_claims = [c for c in claims if c.claim_id not in verified_claim_ids]

        if unverified_claims and decision.next_worker in ("human_review", "finish"):
            logger.warning(
                "[Supervisor:Invariant] %d unverified claims exist. Overriding '%s' to 'verify'.",
                len(unverified_claims),
                decision.next_worker,
            )
            return SupervisorDecision(
                next_worker="verify",
                reasoning=(
                    f"Architectural invariant violation: {len(unverified_claims)} unverified claims "
                    "exist. Claims must be verified before proceeding to human review or finish."
                ),
            )

    return decision


class DeterministicSupervisorPolicy:
    """Rules-based supervisor policy enforcing deterministic workflow progression."""

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS,
        max_human_cycles: int = DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
    ) -> None:
        self.max_steps = max_steps
        self.max_human_cycles = max_human_cycles

    def evaluate(self, state: ResearchState) -> SupervisorDecision:
        current_steps = state.get("supervisor_steps", 0)
        if current_steps >= self.max_steps:
            return SupervisorDecision(
                next_worker="finish",
                reasoning=f"Supervisor step cap reached ({current_steps} >= {self.max_steps}).",
            )

        # 1. Evaluate Human Review outcome if present
        review = state.get("human_review")
        if review is not None:
            if review.action in ("approve", "edit", "reject"):
                return SupervisorDecision(
                    next_worker="finish",
                    reasoning=f"Human review completed with action '{review.action}'.",
                )
            if review.action == "research_more":
                cycles = state.get("human_research_cycles", 0)
                if cycles < self.max_human_cycles:
                    return SupervisorDecision(
                        next_worker="research",
                        reasoning=f"Human requested additional research (cycle {cycles + 1} of {self.max_human_cycles}).",
                    )
                return SupervisorDecision(
                    next_worker="finish",
                    reasoning=f"Maximum human research cycles reached ({cycles} >= {self.max_human_cycles}).",
                )

        # 2. Check for follow-up question continuity
        follow_up = state.get("follow_up_question")
        reuse_decision = state.get("reuse_decision")
        if follow_up and follow_up.strip() and reuse_decision is not None:
            if reuse_decision.decision == "REUSE":
                claims = state.get("claims", [])
                verifications = state.get("verification_results", [])
                verified_ids = {v.claim_id for v in verifications}
                if not claims or any(c.claim_id not in verified_ids for c in claims):
                    return SupervisorDecision(
                        next_worker="verify",
                        reasoning="Follow-up research reused existing evidence; synthesized claims require verification.",
                    )
                return SupervisorDecision(
                    next_worker="human_review",
                    reasoning="Reused research claims verified; awaiting human review.",
                )
            elif reuse_decision.decision == "RESEARCH_MORE":
                # Check if research has already run for this follow-up
                sources = state.get("sources", [])
                claims = state.get("claims", [])
                iteration = state.get("research_iteration", 0)
                if iteration == 0 and not claims:
                    return SupervisorDecision(
                        next_worker="research",
                        reasoning="Follow-up query requires additional research; routing to research subgraph.",
                    )

        # 3. Standard flow based on research, claims, and verification state
        sources = state.get("sources", [])
        findings = state.get("findings", [])
        claims = state.get("claims", [])
        verifications = state.get("verification_results", [])

        # No research conducted yet
        if not sources or not findings or not claims:
            return SupervisorDecision(
                next_worker="research",
                reasoning="Initial research pass required to retrieve sources and synthesize claims.",
            )

        # Claims exist: check verification completeness
        verified_ids = {v.claim_id for v in verifications}
        unverified = [c for c in claims if c.claim_id not in verified_ids]
        if unverified:
            return SupervisorDecision(
                next_worker="verify",
                reasoning=f"{len(unverified)} unverified claims detected; routing to claim-level verifier.",
            )

        # All claims verified and awaiting human review
        return SupervisorDecision(
            next_worker="human_review",
            reasoning="All factual claims verified against evidence; routing to human review.",
        )


class LLMSupervisorPolicy:
    """LLM-based supervisor policy utilizing structured output and invariant enforcement."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        max_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS,
        max_human_cycles: int = DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
        fallback_policy: SupervisorPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.max_steps = max_steps
        self.max_human_cycles = max_human_cycles
        self.fallback = fallback_policy or DeterministicSupervisorPolicy(
            max_steps=max_steps, max_human_cycles=max_human_cycles
        )

    def _get_model(self) -> BaseChatModel:
        if self.llm is not None:
            return self.llm
        return get_chat_model()

    def evaluate(self, state: ResearchState) -> SupervisorDecision:
        model = self._get_model()

        system_instruction = (
            "You are the central Supervisor orchestrator of a verified research system.\n"
            "Your objective is to coordinate specialized workers to produce verified, grounded research.\n\n"
            "AVAILABLE WORKERS:\n"
            "1. 'research': Invokes the Research Subgraph to search the web, analyze sources, and synthesize findings.\n"
            "2. 'verify': Invokes the Claim Verifier to evaluate atomic claims against preserved evidence excerpts.\n"
            "3. 'human_review': Invokes Human-in-the-Loop review to present verified claims to the user for approval.\n"
            "4. 'finish': Concludes orchestration when work is completed or approved.\n\n"
            "MANDATORY INVARIANTS:\n"
            "- If claims exist but are unverified, you MUST route to 'verify'. NEVER bypass verification.\n"
            "- If no research or claims exist, you MUST route to 'research'.\n"
            "- If human review approved, edited, or rejected the research, you MUST route to 'finish'.\n"
            "- If human requested 'research_more', route to 'research' unless cycle limit is reached.\n"
            "Select the next worker and provide clear rationale."
        )

        question = state.get("question", "")
        follow_up = state.get("follow_up_question")
        sources_count = len(state.get("sources", []))
        claims_count = len(state.get("claims", []))
        verifications_count = len(state.get("verification_results", []))
        review = state.get("human_review")
        review_str = f"action={review.action}, feedback={review.feedback}" if review else "None"
        steps = state.get("supervisor_steps", 0)

        user_content = (
            f"Active Question: {question}\n"
            f"Follow-Up Question: {follow_up}\n"
            f"Current State:\n"
            f"- Supervisor Steps: {steps}/{self.max_steps}\n"
            f"- Sources Retrieved: {sources_count}\n"
            f"- Claims Synthesized: {claims_count}\n"
            f"- Verifications Completed: {verifications_count}\n"
            f"- Human Review: {review_str}\n\n"
            "Decide which worker should execute next."
        )

        try:
            structured_model = model.with_structured_output(SupervisorDecision)
            result = structured_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content),
                ]
            )
            if not isinstance(result, SupervisorDecision):
                raise ValueError(
                    f"Expected SupervisorDecision, got {type(result).__name__}"
                )
            # Pass through invariant validation guard
            return validate_supervisor_decision(
                state=state,
                decision=result,
                max_steps=self.max_steps,
                max_human_cycles=self.max_human_cycles,
            )
        except Exception as e:
            logger.warning(
                "[Supervisor:LLM] LLM evaluation failed or violated contract (%s); falling back to deterministic policy",
                e,
            )
            return self.fallback.evaluate(state)


def create_supervisor_node(
    policy: SupervisorPolicy | None = None,
    max_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS,
    max_human_cycles: int = DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Factory creating the Supervisor node coordinating worker execution in the graph loop.

    Contract:
        Input: ResearchState
        Output:
            - supervisor_steps: int (incremented by 1)
            - supervisor_decision: SupervisorDecision
            - supervisor_termination_reason: str | None (set if terminating)
            - follow-up helper updates (if follow-up question requires initialization)
    """
    active_policy = policy or DeterministicSupervisorPolicy(
        max_steps=max_steps, max_human_cycles=max_human_cycles
    )

    def supervisor_node(state: ResearchState) -> dict[str, Any]:
        current_steps = state.get("supervisor_steps", 0)
        logger.info("[Supervisor] Step %d: evaluating state...", current_steps + 1)

        # Check hard step boundary
        if current_steps >= max_steps:
            logger.warning("[Supervisor] Max supervisor steps (%d) reached. Halting.", max_steps)
            decision = SupervisorDecision(
                next_worker="finish",
                reasoning=f"Maximum supervisor steps ({max_steps}) reached. Safe termination.",
            )
            return {
                "supervisor_steps": current_steps + 1,
                "supervisor_decision": decision,
                "supervisor_termination_reason": "MAX_SUPERVISOR_STEPS_REACHED",
            }

        updates: dict[str, Any] = {}

        # Handle follow-up question initialization if present and unassessed
        follow_up = state.get("follow_up_question")
        reuse_decision = state.get("reuse_decision")
        if follow_up and follow_up.strip() and reuse_decision is None:
            sources = state.get("sources", [])
            evidence = state.get("evidence", [])
            findings = state.get("findings", [])
            claims = state.get("claims", [])
            prev_questions = list(state.get("previous_questions", []))

            # Use heuristic sufficiency service to evaluate reuse
            sufficiency_svc = HeuristicSufficiencyService()
            reuse_decision = sufficiency_svc.evaluate_sufficiency(
                question=follow_up.strip(),
                sources=sources,
                findings=findings,
                claims=claims,
                evidence=evidence,
                previous_questions=prev_questions,
            )
            updates["reuse_decision"] = reuse_decision
            if follow_up.strip() not in prev_questions:
                prev_questions.append(follow_up.strip())
            updates["previous_questions"] = prev_questions
            updates["question"] = follow_up.strip()

            if reuse_decision.decision == "REUSE":
                # Synthesize follow-up claims from existing evidence
                reuse_analyst = create_reuse_analyst_node()
                synth = reuse_analyst({**state, **updates})
                updates["findings"] = synth["findings"]
                updates["claims"] = synth["claims"]
                updates["verification_results"] = []
                updates["human_review"] = None
            else:
                updates["research_iteration"] = 0
                updates["human_research_cycles"] = 0
                updates["claims"] = []
                updates["verification_results"] = []
                updates["human_review"] = None

        # Build effective state incorporating any immediate follow-up updates
        eval_state = {**state, **updates}

        # Evaluate policy
        raw_decision = active_policy.evaluate(eval_state)

        # Pass through invariant validation guard
        decision = validate_supervisor_decision(
            state=eval_state,
            decision=raw_decision,
            max_steps=max_steps,
            max_human_cycles=max_human_cycles,
        )

        logger.info(
            "[Supervisor] Step %d decision: next_worker='%s' (reasoning: %s)",
            current_steps + 1,
            decision.next_worker,
            decision.reasoning,
        )

        updates["supervisor_steps"] = current_steps + 1
        updates["supervisor_decision"] = decision

        if decision.next_worker == "finish":
            if current_steps + 1 >= max_steps:
                updates["supervisor_termination_reason"] = "MAX_SUPERVISOR_STEPS_REACHED"
            else:
                updates["supervisor_termination_reason"] = "COMPLETED"

        return updates

    return supervisor_node


# Default supervisor node instance using standard configuration
supervisor_node = create_supervisor_node()
