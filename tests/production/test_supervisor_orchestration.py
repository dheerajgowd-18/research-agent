"""Production Test Suite for Supervisor Orchestration and Graph Safety Guards.

Verifies:
1. Dynamic state-driven routing (research -> verify -> human_review -> finish).
2. Supervisor policy invariant enforcement against illegal worker bypass attempts.
3. Separation of independent counters (supervisor_steps vs research_iteration vs human_research_cycles).
4. Maximum supervisor steps limit enforcement in compiled graph.
5. Rejection termination reason recording.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from verified_research.agents.supervisor import (
    DeterministicSupervisorPolicy,
    LLMSupervisorPolicy,
    create_supervisor_node,
    validate_supervisor_decision,
)
from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    Source,
    SupervisorDecision,
    VerificationResult,
)
from verified_research.graph.state import ResearchState
from tests.production.conftest import (
    create_deterministic_mock_subgraph,
    create_deterministic_mock_verifier,
)


class TestSupervisorOrchestration:
    """Verifies supervisor state transitions and architectural safety boundaries."""

    def test_state_driven_progression_logic(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
        sample_verification_results: list[VerificationResult],
    ):
        """Supervisor deterministically advances through each worker based on state completeness."""
        policy = DeterministicSupervisorPolicy()

        # Phase 1: Unresearched state -> routes to research
        state_phase1 = {"question": "What is topological quantum computing?", "supervisor_steps": 0}
        d1 = policy.evaluate(state_phase1)
        assert d1.next_worker == "research"

        # Phase 2: Claims exist but not verified -> routes to verify
        state_phase2 = {
            "question": "What is topological quantum computing?",
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": [Finding(finding_id="f1", text="Findings", source_ids=["src_001"])],
            "claims": sample_claims,
            "verification_results": [],
            "supervisor_steps": 1,
        }
        d2 = policy.evaluate(state_phase2)
        assert d2.next_worker == "verify"

        # Phase 3: All claims verified, no human review -> routes to human_review
        state_phase3 = {**state_phase2, "verification_results": sample_verification_results, "supervisor_steps": 2}
        d3 = policy.evaluate(state_phase3)
        assert d3.next_worker == "human_review"

        # Phase 4: Human review approved -> routes to finish
        state_phase4 = {
            **state_phase3,
            "human_review": HumanReview(action="approve", feedback="Looks solid."),
            "supervisor_steps": 3,
        }
        d4 = policy.evaluate(state_phase4)
        assert d4.next_worker == "finish"

    def test_invariant_guard_intercepts_illegal_llm_bypass(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Supervisor invariant guard overrides an LLM policy trying to bypass verifier directly to human_review."""
        # Unverified claims in state
        state_with_unverified = {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": [Finding(finding_id="f1", text="Findings", source_ids=["src_001"])],
            "claims": sample_claims,
            "verification_results": [],  # NO verifications!
            "supervisor_steps": 1,
        }

        # Malicious or hallucinated decision attempting to bypass verification directly to human_review
        bypass_decision = SupervisorDecision(
            next_worker="human_review",
            reasoning="Skipping verification to save tokens and speed up review.",
        )

        guarded = validate_supervisor_decision(state_with_unverified, bypass_decision)
        # Invariant guard MUST override to 'verify'
        assert guarded.next_worker == "verify"
        assert "unverified claims exist" in guarded.reasoning.lower() or "claims must be verified" in guarded.reasoning.lower()

    def test_invariant_guard_intercepts_verify_without_claims(self):
        """Supervisor invariant guard overrides attempt to verify without claims to research."""
        state_empty = {"claims": [], "supervisor_steps": 0}
        invalid_verify_decision = SupervisorDecision(
            next_worker="verify",
            reasoning="Attempting to verify non-existent claims.",
        )

        guarded = validate_supervisor_decision(state_empty, invalid_verify_decision)
        assert guarded.next_worker == "research"

    def test_supervisor_steps_counter_isolated_from_subgraph_counters(self):
        """Supervisor steps increment independently from research_iteration and human_research_cycles."""
        node = create_supervisor_node(max_steps=8)

        state = {
            "question": "Counter isolation check",
            "supervisor_steps": 3,
            "research_iteration": 2,
            "human_research_cycles": 1,
            "sources": [],
            "claims": [],
        }

        updates = node(state)
        assert updates["supervisor_steps"] == 4
        # Subgraph and human counters must not be modified by supervisor node evaluation
        assert "research_iteration" not in updates or updates["research_iteration"] == 2
        assert "human_research_cycles" not in updates or updates["human_research_cycles"] == 1

    def test_supervisor_loop_hard_boundary_in_compiled_graph(self):
        """Compiled supervisor graph halts safely if supervisor steps reach max_steps."""
        # Create a looping mock subgraph that always returns empty claims, causing supervisor to keep routing to research
        def endlessly_looping_worker(state: ResearchState):
            return {
                "sources": [],
                "claims": [],
                "findings": [],
            }

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(
            custom_subgraph=endlessly_looping_worker,
            max_supervisor_steps=4,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "max-steps-test-001"}}
        final_state = graph.invoke({"question": "Infinite loop test"}, config)

        assert final_state["supervisor_steps"] >= 4
        assert final_state["supervisor_decision"].next_worker == "finish"
        assert final_state["supervisor_termination_reason"] == "MAX_SUPERVISOR_STEPS_REACHED"

    def test_supervisor_rejection_marks_terminal_rejected_status(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
        sample_verification_results: list[VerificationResult],
    ):
        """Supervisor node sets supervisor_termination_reason to REJECTED when human rejects."""
        node = create_supervisor_node()
        state = {
            "question": "Reject test",
            "supervisor_steps": 3,
            "sources": sample_sources,
            "evidence": sample_evidence,
            "claims": sample_claims,
            "verification_results": sample_verification_results,
            "human_review": HumanReview(action="reject", feedback="Sources unverified and biased."),
        }

        updates = node(state)
        assert updates["supervisor_decision"].next_worker == "finish"
        assert updates["supervisor_termination_reason"] == "REJECTED"
