"""Unit and integration tests for the Writer agent and final response generation."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from verified_research.agents.writer import (
    InvariantViolationError,
    WriterService,
    create_writer_node,
)
from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import (
    Citation,
    Claim,
    Evidence,
    FinalReport,
    Finding,
    HumanReview,
    ReportSection,
    Source,
    VerificationResult,
)


@pytest.fixture
def sample_research_state() -> dict:
    """Fixture providing a verified research state ready for final writing."""
    source_1 = Source(
        source_id="src_001",
        title="Reinforcement Learning: An Introduction",
        url="http://incompleteideas.net/book/the-book-2nd.html",
        content="Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal.",
    )
    source_2 = Source(
        source_id="src_002",
        title="Calibrated Language Models",
        url="https://arxiv.org/abs/2106.12345",
        content="Calibrated predictive distributions ensure that reported model confidence corresponds to empirical accuracy.",
    )
    ev_1 = Evidence(
        evidence_id="ev_001",
        source_id="src_001",
        text="Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal.",
    )
    ev_2 = Evidence(
        evidence_id="ev_002",
        source_id="src_002",
        text="Calibrated predictive distributions ensure that reported model confidence corresponds to empirical accuracy.",
    )
    claim_1 = Claim(
        claim_id="claim_001",
        text="Reinforcement learning algorithms map environmental states to reward-maximizing actions.",
        evidence_ids=["ev_001"],
    )
    claim_2 = Claim(
        claim_id="claim_002",
        text="Probability calibration ensures model confidence aligns with empirical correctness.",
        evidence_ids=["ev_002"],
    )
    verifications = [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="Direct textual grounding in Sutton & Barto reference.",
            evidence_ids=["ev_001"],
        ),
        VerificationResult(
            claim_id="claim_002",
            verdict="SUPPORTED",
            confidence=0.96,
            reasoning="Grounded in empirical calibration literature.",
            evidence_ids=["ev_002"],
        ),
    ]
    return {
        "question": "What is reinforcement learning and calibrated decisions?",
        "sources": [source_1, source_2],
        "evidence": [ev_1, ev_2],
        "claims": [claim_1, claim_2],
        "findings": [Finding(finding_id="f1", text="RL maps states to actions", source_ids=["src_001"])],
        "verification_results": verifications,
        "human_review": HumanReview(action="approve", feedback="Looks comprehensive."),
    }


class TestWriterServiceInvariants:
    """Test suite verifying WriterService invariant enforcement and grounded synthesis."""

    def test_writer_cannot_run_without_human_review(self, sample_research_state):
        """Invariant 1: Writer raises InvariantViolationError if human_review is missing."""
        state = dict(sample_research_state)
        state["human_review"] = None
        service = WriterService()
        with pytest.raises(InvariantViolationError, match="Writer cannot execute before Human Review"):
            service.generate_report(state)

    def test_writer_cannot_run_on_rejected_review(self, sample_research_state):
        """Invariant 1: Writer raises InvariantViolationError if human rejected the research."""
        state = dict(sample_research_state)
        state["human_review"] = HumanReview(action="reject", feedback="Unacceptable factual errors.")
        service = WriterService()
        with pytest.raises(InvariantViolationError, match="Writer cannot execute before Human Review"):
            service.generate_report(state)

    def test_writer_cannot_run_on_research_more_review(self, sample_research_state):
        """Invariant 1: Writer raises InvariantViolationError if review action is research_more."""
        state = dict(sample_research_state)
        state["human_review"] = HumanReview(action="research_more", feedback="Investigate further.")
        service = WriterService()
        with pytest.raises(InvariantViolationError, match="Writer cannot execute before Human Review"):
            service.generate_report(state)

    def test_writer_generates_grounded_report_on_approval(self, sample_research_state):
        """Writer produces structured FinalReport citing approved claims and sources."""
        service = WriterService()
        report = service.generate_report(sample_research_state)

        assert isinstance(report, FinalReport)
        assert len(report.title) > 5
        assert len(report.summary) > 20
        assert len(report.answer) > 50
        assert len(report.sections) >= 2
        assert len(report.citations) == 2
        assert set(report.claim_references) == {"claim_001", "claim_002"}
        assert report.verified_claim_count == 2

        # Verify citation mappings
        cit_sources = [c.source_id for c in report.citations]
        assert "src_001" in cit_sources
        assert "src_002" in cit_sources
        for cit in report.citations:
            assert cit.citation_id.startswith("[") and cit.citation_id.endswith("]")
            assert cit.source_title != ""
            assert cit.source_url.startswith("http")

    def test_writer_uses_edited_claims_when_action_is_edit(self, sample_research_state):
        """Invariant 2: When human review action is 'edit', Writer strictly uses edited claims."""
        edited_claim = Claim(
            claim_id="claim_001",
            text="User-edited claim: Reinforcement learning directly optimizes cumulative policy reward.",
            evidence_ids=["ev_001"],
        )
        state = dict(sample_research_state)
        state["human_review"] = HumanReview(action="edit", edited_claims=[edited_claim])

        service = WriterService()
        report = service.generate_report(state)

        assert report.claim_references == ["claim_001"]
        assert "User-edited claim" in report.answer
        assert "User-edited claim" in report.sections[1].content


class TestWriterGraphIntegration:
    """Test suite verifying Writer node execution and state transitions inside the Supervisor graph."""

    def test_writer_executes_after_human_approval_in_graph(self, sample_research_state):
        """In a graph with enable_writer=True, approving human review routes to writer and outputs final_response."""
        def mock_subgraph(state):
            return {
                "sources": sample_research_state["sources"],
                "evidence": sample_research_state["evidence"],
                "claims": sample_research_state["claims"],
                "findings": sample_research_state["findings"],
                "research_iteration": 1,
            }

        def mock_verifier(state):
            return {"verification_results": sample_research_state["verification_results"]}

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(
            custom_subgraph=mock_subgraph,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
            enable_writer=True,
        )
        config = {"configurable": {"thread_id": "test-writer-flow-001"}}

        # Run until human review interrupt
        graph.invoke({"question": sample_research_state["question"]}, config)
        state_at_interrupt = graph.get_state(config)
        assert state_at_interrupt.next == ("human_review",)

        # Resume with approve
        final_state = graph.invoke(
            Command(resume={"action": "approve", "feedback": "Approved for final publication."}),
            config,
        )

        assert graph.get_state(config).next == ()
        assert final_state["human_review"].action == "approve"
        assert "final_response" in final_state
        report = final_state["final_response"]
        assert isinstance(report, FinalReport)
        assert len(report.sections) >= 2
        assert report.verified_claim_count == 2
        assert final_state["supervisor_decision"].next_worker == "finish"
        assert final_state["supervisor_termination_reason"] == "COMPLETED"

    def test_writer_does_not_execute_on_rejection(self, sample_research_state):
        """When human review rejects, supervisor terminates immediately with REJECTED and no writer execution."""
        def mock_subgraph(state):
            return {
                "sources": sample_research_state["sources"],
                "evidence": sample_research_state["evidence"],
                "claims": sample_research_state["claims"],
                "findings": sample_research_state["findings"],
                "research_iteration": 1,
            }

        def mock_verifier(state):
            return {"verification_results": sample_research_state["verification_results"]}

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(
            custom_subgraph=mock_subgraph,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
            enable_writer=True,
        )
        config = {"configurable": {"thread_id": "test-writer-reject-001"}}

        # Run until human review interrupt
        graph.invoke({"question": sample_research_state["question"]}, config)

        # Resume with reject
        final_state = graph.invoke(
            Command(resume={"action": "reject", "feedback": "Flawed claims."}),
            config,
        )

        assert graph.get_state(config).next == ()
        assert final_state["human_review"].action == "reject"
        assert final_state.get("final_response") is None
        assert final_state["supervisor_decision"].next_worker == "finish"
        assert final_state["supervisor_termination_reason"] == "REJECTED"
