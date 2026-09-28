"""Production Test Suite for Human-in-the-Loop (HITL) Workflows and Review Actions.

Verifies:
1. Resume with 'approve' -> transitions to supervisor finish (COMPLETED).
2. Resume with 'edit' (valid claims) -> updates claims in state, transitions to finish (COMPLETED).
3. Resume with 'edit' (invalid claims / unknown evidence) -> validation catches invalid inputs.
4. Resume with 'research_more' -> resets verification, passes feedback, bounds at MAX_HUMAN_REVIEW_CYCLES (2).
5. Resume with 'reject' -> terminates immediately with status REJECTED (never COMPLETED).
"""

from pathlib import Path
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)
from verified_research.models.traceability import (
    UnknownEvidenceError,
    validate_traceability,
)
from verified_research.graph.state import ResearchState
from tests.production.conftest import (
    create_deterministic_mock_subgraph,
    create_deterministic_mock_verifier,
)


class TestHumanInTheLoopWorkflows:
    """Verifies all human review actions and cycle limits."""

    def test_hitl_approve_workflow(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Approve action concludes research cleanly with COMPLETED status."""
        checkpointer = MemorySaver()
        mock_sub = create_deterministic_mock_subgraph(
            sources=sample_sources,
            evidence=sample_evidence,
            claims=sample_claims,
        )
        mock_ver = create_deterministic_mock_verifier(verdict="SUPPORTED")

        graph = create_supervisor_graph(
            custom_subgraph=mock_sub,
            custom_verifier=mock_ver,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "hitl-approve-test"}}

        # Pause at review
        graph.invoke({"question": "Test approve"}, config)
        state_at_interrupt = graph.get_state(config)
        assert state_at_interrupt.next == ("human_review",)

        # Resume with approve
        final = graph.invoke(Command(resume={"action": "approve", "feedback": "Approved as accurate."}), config)

        assert graph.get_state(config).next == ()
        assert final["human_review"].action == "approve"
        assert final["supervisor_decision"].next_worker == "finish"
        assert final["supervisor_termination_reason"] == "COMPLETED"

    def test_hitl_edit_valid_claims_workflow(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Edit action replaces claims with verified edited versions and finishes."""
        checkpointer = MemorySaver()
        mock_sub = create_deterministic_mock_subgraph(
            sources=sample_sources,
            evidence=sample_evidence,
            claims=sample_claims,
        )
        mock_ver = create_deterministic_mock_verifier(verdict="SUPPORTED")

        graph = create_supervisor_graph(
            custom_subgraph=mock_sub,
            custom_verifier=mock_ver,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "hitl-edit-test"}}

        graph.invoke({"question": "Test edit"}, config)

        edited_data = [
            {
                "claim_id": "claim_001",
                "text": "Refined factual statement with verified metrics.",
                "evidence_ids": ["ev_001"],
            }
        ]

        final = graph.invoke(Command(resume={"action": "edit", "edited_claims": edited_data}), config)

        assert graph.get_state(config).next == ()
        assert final["human_review"].action == "edit"
        assert len(final["claims"]) == 1
        assert final["claims"][0].text == "Refined factual statement with verified metrics."
        assert final["supervisor_decision"].next_worker == "finish"
        assert final["supervisor_termination_reason"] == "COMPLETED"

        # Validate edited traceability
        validate_traceability(
            claims=final["claims"],
            evidence=final["evidence"],
            sources=final["sources"],
        )

    def test_hitl_edit_invalid_evidence_id_rejected(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Edit with unknown evidence ID fails traceability validation."""
        checkpointer = MemorySaver()
        mock_sub = create_deterministic_mock_subgraph(
            sources=sample_sources,
            evidence=sample_evidence,
            claims=sample_claims,
        )
        mock_ver = create_deterministic_mock_verifier(verdict="SUPPORTED")

        graph = create_supervisor_graph(
            custom_subgraph=mock_sub,
            custom_verifier=mock_ver,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "hitl-edit-invalid"}}

        graph.invoke({"question": "Test edit invalid"}, config)

        corrupted_edit = [
            {
                "claim_id": "claim_001",
                "text": "Edited claim referencing phantom evidence.",
                "evidence_ids": ["ev_phantom_9999"],
            }
        ]

        # The human_review node immediately catches the phantom evidence reference
        with pytest.raises(UnknownEvidenceError, match="references unknown evidence_id"):
            graph.invoke(Command(resume={"action": "edit", "edited_claims": corrupted_edit}), config)

    def test_hitl_research_more_workflow_and_cycle_cap(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Research more triggers re-entry into research; caps strictly at max_human_research_cycles."""
        checkpointer = MemorySaver()
        research_count = 0

        def counting_researcher(state: ResearchState):
            nonlocal research_count
            research_count += 1
            return {
                "sources": sample_sources,
                "evidence": sample_evidence,
                "claims": sample_claims,
                "findings": [Finding(finding_id=f"f_{research_count}", text="Finding", source_ids=["src_001"])],
                "research_iteration": research_count,
            }

        mock_ver = create_deterministic_mock_verifier(verdict="SUPPORTED")

        graph = create_supervisor_graph(
            custom_subgraph=counting_researcher,
            custom_verifier=mock_ver,
            max_human_research_cycles=2,
            max_supervisor_steps=15,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "hitl-cycles-test"}}

        # Initial pass (pass 0)
        graph.invoke({"question": "Test cycles"}, config)
        assert research_count == 1
        assert graph.get_state(config).next == ("human_review",)

        # Human requests research_more (Cycle 1)
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Need more data 1"}), config)
        assert research_count == 2
        assert graph.get_state(config).next == ("human_review",)

        # Human requests research_more (Cycle 2, hits cap)
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Need more data 2"}), config)
        assert research_count == 3
        assert graph.get_state(config).next == ("human_review",)

        # Human requests research_more beyond cap -> Supervisor finishes workflow
        final = graph.invoke(Command(resume={"action": "research_more", "feedback": "Need more data 3"}), config)
        assert graph.get_state(config).next == ()
        assert research_count == 3  # Did not run a 4th time
        assert final["supervisor_decision"].next_worker == "finish"

    def test_hitl_reject_terminal_workflow(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """Reject action halts execution immediately with REJECTED status, without re-entering research."""
        checkpointer = MemorySaver()
        research_invocations = 0

        def single_run_researcher(state: ResearchState):
            nonlocal research_invocations
            research_invocations += 1
            return {
                "sources": sample_sources,
                "evidence": sample_evidence,
                "claims": sample_claims,
                "findings": [Finding(finding_id="f1", text="Findings", source_ids=["src_001"])],
            }

        mock_ver = create_deterministic_mock_verifier(verdict="SUPPORTED")

        graph = create_supervisor_graph(
            custom_subgraph=single_run_researcher,
            custom_verifier=mock_ver,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "hitl-reject-test"}}

        # Pause at review
        graph.invoke({"question": "Test rejection"}, config)
        assert research_invocations == 1
        assert graph.get_state(config).next == ("human_review",)

        # Resume with reject
        final = graph.invoke(Command(resume={"action": "reject", "feedback": "Poor quality evidence."}), config)

        assert graph.get_state(config).next == ()
        assert research_invocations == 1  # Research was NOT re-entered
        assert final["human_review"].action == "reject"
        assert final["supervisor_decision"].next_worker == "finish"
        assert final["supervisor_termination_reason"] == "REJECTED"
