"""Test suite verifying all 9 Core System Invariants across stateful graph execution.

Invariants Verified:
1. Claim-Evidence-Source Traceability (Claim -> Evidence -> Source)
2. Complete Verification Coverage (1:1 claim-to-verification mapping)
3. Human-in-the-Loop Gate (no finish without human review)
4. Bounded Research Loop (<= 3 critic loops)
5. Bounded Human Review Cycles (<= 2 research_more cycles)
6. Bounded Supervisor Loop (<= 8 supervisor steps)
7. Bounded Retries (<= 3 transient attempts)
8. Valid Routing Destinations (only allowed worker names)
9. Terminal Rejection Integrity (reject status is REJECTED, never COMPLETED)
"""

import pytest
from pydantic import ValidationError

from verified_research.agents.supervisor import (
    DeterministicSupervisorPolicy,
    create_supervisor_node,
    validate_supervisor_decision,
)
from verified_research.graph.router import (
    route_after_critic,
    route_after_human_review,
    route_after_supervisor,
)
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    SupervisorDecision,
    VerificationResult,
)
from verified_research.models.traceability import (
    UnknownEvidenceError,
    UnknownSourceError,
    validate_traceability,
)
from verified_research.reliability.models import MaxRetriesExceededError
from verified_research.reliability.policy import RetryPolicy, execute_with_retry


class TestCoreSystemInvariants:
    """Test suite for architectural invariants of the Verified Research Agent."""

    # -------------------------------------------------------------------------
    # Invariant 1: Claim -> Evidence -> Source Traceability
    # -------------------------------------------------------------------------
    def test_invariant_1_claim_evidence_source_traceability(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Every claim must point to valid evidence, which must point to valid source."""
        # 1. Valid traceability passes
        validate_traceability(
            claims=sample_claims,
            evidence=sample_evidence,
            sources=sample_sources,
        )

        # 2. Dangling evidence ID in claim raises UnknownEvidenceError
        corrupted_claim = Claim(
            claim_id="claim_dangling",
            text="Unfounded claim with nonexistent evidence.",
            evidence_ids=["ev_nonexistent_999"],
        )
        with pytest.raises(UnknownEvidenceError):
            validate_traceability(
                claims=[corrupted_claim],
                evidence=sample_evidence,
                sources=sample_sources,
            )

        # 3. Dangling source ID in evidence raises UnknownSourceError
        corrupted_evidence = Evidence(
            evidence_id="ev_corrupt",
            source_id="src_nonexistent_888",
            text="Evidence referencing phantom source.",
        )
        with pytest.raises(UnknownSourceError):
            validate_traceability(
                claims=[sample_claims[0]],
                evidence=[corrupted_evidence],
                sources=sample_sources,
            )

    # -------------------------------------------------------------------------
    # Invariant 2: Complete Verification Coverage
    # -------------------------------------------------------------------------
    def test_invariant_2_complete_verification_coverage(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
        sample_verification_results: list[VerificationResult],
    ):
        """Supervisor must never route to human_review or finish if ANY claim is unverified."""
        policy = DeterministicSupervisorPolicy()

        # Partial verification: only claim_001 is verified, claim_002 is not
        partial_results = [sample_verification_results[0]]
        state = {
            "question": "What is quantum coherence?",
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": [Finding(finding_id="f1", text="Finding", source_ids=["src_001"])],
            "claims": sample_claims,
            "verification_results": partial_results,
            "supervisor_steps": 2,
        }

        decision = policy.evaluate(state)
        # MUST route to verify, NOT human_review or finish
        assert decision.next_worker == "verify"
        assert "unverified" in decision.reasoning.lower() or "claim" in decision.reasoning.lower()

    # -------------------------------------------------------------------------
    # Invariant 3: Human-in-the-Loop Gate
    # -------------------------------------------------------------------------
    def test_invariant_3_human_in_the_loop_gate(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
        sample_verification_results: list[VerificationResult],
    ):
        """No execution can finish without explicit human review approval."""
        policy = DeterministicSupervisorPolicy()

        # State has claims and verification results, but human_review is None
        state = {
            "question": "What is quantum coherence?",
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": [Finding(finding_id="f1", text="Finding", source_ids=["src_001"])],
            "claims": sample_claims,
            "verification_results": sample_verification_results,
            "human_review": None,
            "supervisor_steps": 2,
        }

        decision = policy.evaluate(state)
        assert decision.next_worker == "human_review"
        assert decision.next_worker != "finish"

    # -------------------------------------------------------------------------
    # Invariant 4: Bounded Research Loop
    # -------------------------------------------------------------------------
    def test_invariant_4_bounded_research_loop(self):
        """Critic loop terminates at or before MAX_RESEARCH_LOOPS (3) iterations."""
        # When critic demands revision at iteration 0 -> continues to researcher
        state_iter_0 = {
            "research_iteration": 0,
            "critique": Critique(quality_score=0.4, should_research_again=True, weak_findings=["Depth needed"]),
        }
        assert route_after_critic(state_iter_0) == "researcher"

        # At iteration 2 -> continues to researcher (attempt 3)
        state_iter_2 = {
            "research_iteration": 2,
            "critique": Critique(quality_score=0.5, should_research_again=True),
        }
        assert route_after_critic(state_iter_2) == "researcher"

        # At iteration 3 (MAX reached) -> MUST route to "end", never loop indefinitely
        state_iter_3 = {
            "research_iteration": 3,
            "critique": Critique(quality_score=0.5, should_research_again=True),
        }
        assert route_after_critic(state_iter_3) == "end"

    # -------------------------------------------------------------------------
    # Invariant 5: Bounded Human Review Cycles
    # -------------------------------------------------------------------------
    def test_invariant_5_bounded_human_review_cycles(self):
        """Human research_more cycle terminates when exceeding MAX_HUMAN_REVIEW_CYCLES (2)."""
        # Cycle 0 requesting research_more -> allowed to research
        state_cycle_0 = {
            "human_research_cycles": 0,
            "human_review": HumanReview(action="research_more", feedback="Dig deeper."),
        }
        assert route_after_human_review(state_cycle_0) == "research"

        # Cycle 1 requesting research_more -> allowed to research
        state_cycle_1 = {
            "human_research_cycles": 1,
            "human_review": HumanReview(action="research_more", feedback="Need more data."),
        }
        assert route_after_human_review(state_cycle_1) == "research"

        # Cycle 2 requesting research_more -> cycle limit hit, forced to "end"
        state_cycle_2 = {
            "human_research_cycles": 2,
            "human_review": HumanReview(action="research_more", feedback="Need even more."),
        }
        assert route_after_human_review(state_cycle_2) == "end"

    # -------------------------------------------------------------------------
    # Invariant 6: Bounded Supervisor Loop
    # -------------------------------------------------------------------------
    def test_invariant_6_bounded_supervisor_loop(self):
        """Supervisor steps cannot exceed MAX_SUPERVISOR_STEPS (8)."""
        node = create_supervisor_node(max_steps=8)

        # At step 8 (step limit reached)
        state_step_8 = {
            "question": "Continuous test",
            "supervisor_steps": 8,
            "claims": [],
        }
        updates = node(state_step_8)
        assert updates["supervisor_steps"] == 9
        assert updates["supervisor_decision"].next_worker == "finish"
        assert updates["supervisor_termination_reason"] == "MAX_SUPERVISOR_STEPS_REACHED"

        # Router also respects max_steps
        assert route_after_supervisor(updates, max_steps=8) == "finish"

    # -------------------------------------------------------------------------
    # Invariant 7: Bounded Retries
    # -------------------------------------------------------------------------
    def test_invariant_7_bounded_retries(self):
        """Transient errors retry at most MAX_RETRIES (3) before failing."""
        attempts = 0

        def always_failing_transient():
            nonlocal attempts
            attempts += 1
            err = ConnectionError("Transient network blip")
            setattr(err, "status_code", 503)
            raise err

        policy = RetryPolicy(max_attempts=4, base_delay=0.001, backoff_factor=1.0)

        with pytest.raises(MaxRetriesExceededError):
            execute_with_retry(
                always_failing_transient,
                policy=policy,
                operation_name="failing_op",
            )

        # Initial attempt + 3 retries = 4 attempts total
        assert attempts == 4

    # -------------------------------------------------------------------------
    # Invariant 8: Valid Routing Destinations
    # -------------------------------------------------------------------------
    def test_invariant_8_valid_routing_destinations(self):
        """Supervisor decisions and router only accept allowed destinations."""
        # 1. Pydantic validation rejects invalid worker name
        with pytest.raises(ValidationError):
            SupervisorDecision(next_worker="unauthorized_worker", reasoning="Bypass attempt")

        # 2. validate_supervisor_decision overrides invalid actions or bypasses
        tampered_decision = SupervisorDecision(next_worker="finish", reasoning="Attempt bypass")
        state_with_unverified_claims = {
            "claims": [Claim(claim_id="c1", text="Unverified text", evidence_ids=["e1"])],
            "verification_results": [],
            "supervisor_steps": 1,
        }
        guarded = validate_supervisor_decision(state_with_unverified_claims, tampered_decision)
        assert guarded.next_worker == "verify"

        # 3. Router defaults safely to 'finish' if decision targets unknown worker
        invalid_state = {
            "supervisor_steps": 1,
            "supervisor_decision": SupervisorDecision(next_worker="finish", reasoning="OK"),
        }
        object.__setattr__(invalid_state["supervisor_decision"], "next_worker", "illegal_dest")
        assert route_after_supervisor(invalid_state) == "finish"

    # -------------------------------------------------------------------------
    # Invariant 9: Rejection Terminal Status Integrity
    # -------------------------------------------------------------------------
    def test_invariant_9_rejection_terminal_status(self):
        """Human review rejection yields REJECTED termination status, never COMPLETED."""
        node = create_supervisor_node(max_steps=8)
        state_rejected = {
            "question": "Controversial topic",
            "supervisor_steps": 4,
            "claims": [Claim(claim_id="c1", text="Sample", evidence_ids=["e1"])],
            "verification_results": [
                VerificationResult(claim_id="c1", verdict="SUPPORTED", confidence=0.9, reasoning="ok", evidence_ids=["e1"])
            ],
            "human_review": HumanReview(action="reject", feedback="Hallucinatory / low quality evidence."),
        }

        updates = node(state_rejected)
        assert updates["supervisor_decision"].next_worker == "finish"
        assert updates["supervisor_termination_reason"] == "REJECTED"
        assert updates["supervisor_termination_reason"] != "COMPLETED"
