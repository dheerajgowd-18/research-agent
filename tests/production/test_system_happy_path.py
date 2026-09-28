"""Production Happy-Path End-to-End Test.

Verifies the complete integrated system lifecycle:
START -> Supervisor -> Research Subgraph -> Supervisor -> Verifier
-> Supervisor -> Human Review (Interrupt) -> Resume (Approve)
-> Supervisor -> END (Terminal Finish).
"""

from pathlib import Path
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)
from verified_research.graph.state import ResearchState
from tests.production.conftest import (
    create_deterministic_mock_subgraph,
    create_deterministic_mock_verifier,
)


class TestSystemHappyPath:
    """Verifies that an end-to-end research request traverses all required phases cleanly."""

    def test_full_system_happy_path_with_approval(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Execute complete graph from question through interrupt, approval, and terminal finish."""
        checkpointer = MemorySaver()

        # Deterministic workers
        mock_subgraph = create_deterministic_mock_subgraph(
            sources=sample_sources,
            evidence=sample_evidence,
            claims=sample_claims,
            findings=[
                Finding(
                    finding_id="f_001",
                    text="Quantum coherence achieved millisecond scale in superconducting transmons at base temperatures sub-15mK.",
                    source_ids=["src_001", "src_002"],
                )
            ],
        )
        mock_verifier = create_deterministic_mock_verifier(verdict="SUPPORTED", confidence=0.98)

        graph = create_supervisor_graph(
            custom_subgraph=mock_subgraph,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "happy-path-thread-001"}}
        initial_input = {"question": "What are current breakthroughs in superconducting quantum systems?"}

        # 1. First execution: runs until Human Review interrupt
        result_pause = graph.invoke(initial_input, config)

        assert "__interrupt__" in result_pause
        interrupt_payload = result_pause["__interrupt__"][0].value
        assert "claims" in interrupt_payload or "verification_results" in interrupt_payload

        # Verify thread state at interrupt
        state_at_interrupt = graph.get_state(config)
        assert state_at_interrupt.next == ("human_review",)
        assert len(state_at_interrupt.values.get("claims", [])) == 2
        assert len(state_at_interrupt.values.get("verification_results", [])) == 2
        assert state_at_interrupt.values.get("supervisor_steps") >= 3

        # Verify all claims are verified before human review was invoked
        claim_ids = {c.claim_id for c in state_at_interrupt.values["claims"]}
        verified_claim_ids = {v.claim_id for v in state_at_interrupt.values["verification_results"]}
        assert claim_ids == verified_claim_ids

        # 2. Resume execution with human approval
        resume_cmd = Command(resume={"action": "approve", "feedback": "Excellent rigorous findings."})
        final_state = graph.invoke(resume_cmd, config)

        # 3. Verify terminal completion
        state_at_end = graph.get_state(config)
        assert state_at_end.next == ()

        # Assert final state integrity
        assert final_state["question"] == initial_input["question"]
        assert len(final_state["sources"]) == 2
        assert len(final_state["evidence"]) == 2
        assert len(final_state["claims"]) == 2
        assert len(final_state["verification_results"]) == 2
        assert isinstance(final_state["human_review"], HumanReview)
        assert final_state["human_review"].action == "approve"
        assert final_state["human_review"].feedback == "Excellent rigorous findings."
        assert final_state["supervisor_decision"].next_worker == "finish"
        assert final_state["supervisor_termination_reason"] == "COMPLETED"

        # Traceability assertion across entire artifact chain
        source_id_set = {s.source_id for s in final_state["sources"]}
        evidence_id_set = {e.evidence_id for e in final_state["evidence"]}

        for ev in final_state["evidence"]:
            assert ev.source_id in source_id_set, f"Evidence {ev.evidence_id} references unknown source {ev.source_id}"

        for clm in final_state["claims"]:
            assert clm.evidence_ids, f"Claim {clm.claim_id} has no evidence references"
            for ev_id in clm.evidence_ids:
                assert ev_id in evidence_id_set, f"Claim {clm.claim_id} references unknown evidence {ev_id}"

        for vr in final_state["verification_results"]:
            assert vr.claim_id in claim_ids
            assert vr.verdict == "SUPPORTED"
