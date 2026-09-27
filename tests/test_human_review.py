"""Unit and integration tests for human-in-the-loop review workflow.

Covers:
1. HumanReview model validation and constraints.
2. Deterministic routing after human review.
3. LangGraph interrupt behavior and payload inspection.
4. Supported resume mechanisms (approve, reject, edit, research_more).
5. Edit validation against available evidence.
6. Cycle bounding (MAX_HUMAN_RESEARCH_CYCLES).
7. Human feedback propagation to subsequent research passes.
"""

from unittest.mock import MagicMock
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import ValidationError
from verified_research.agents.human_review import (
    build_review_payload,
    create_human_review_node,
    human_review_node,
    validate_edited_claims,
)
from verified_research.config.settings import (
    DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
)
from verified_research.graph.graph import (
    build_research_graph,
    create_research_graph,
)
from verified_research.graph.router import route_after_human_review
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)
from verified_research.models.traceability import (
    DuplicateIdError,
    UnknownEvidenceError,
)


# ==============================================================================
# 1. HumanReview Model Tests
# ==============================================================================


class TestHumanReviewModel:
    """Tests for structured HumanReview schema and validation constraints."""

    def test_valid_approve_action(self):
        review = HumanReview(action="approve")
        assert review.action == "approve"
        assert review.feedback is None
        assert review.edited_claims is None

    def test_valid_reject_action(self):
        review = HumanReview(action="reject", feedback="Irrelevant sources found.")
        assert review.action == "reject"
        assert review.feedback == "Irrelevant sources found."
        assert review.edited_claims is None

    def test_valid_research_more_action(self):
        review = HumanReview(
            action="research_more",
            feedback="Find primary peer-reviewed benchmark sources.",
        )
        assert review.action == "research_more"
        assert review.feedback == "Find primary peer-reviewed benchmark sources."
        assert review.edited_claims is None

    def test_valid_edit_action(self):
        edited = [
            Claim(
                claim_id="claim_001",
                text="Corrected factual proposition.",
                evidence_ids=["ev_001"],
            )
        ]
        review = HumanReview(action="edit", edited_claims=edited)
        assert review.action == "edit"
        assert review.edited_claims == edited

    def test_invalid_action_raises_validation_error(self):
        """TEST 9: Invalid action string must fail validation."""
        with pytest.raises(ValidationError):
            HumanReview.model_validate({"action": "something_else"})

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            HumanReview.model_validate(
                {"action": "approve", "unauthorized_extra_field": 123}
            )

    def test_edit_action_requires_non_empty_edited_claims(self):
        # Missing edited_claims
        with pytest.raises(ValueError, match="requires a non-empty list of 'edited_claims'"):
            HumanReview(action="edit", edited_claims=None)

        # Empty list of edited_claims
        with pytest.raises(ValueError, match="requires a non-empty list of 'edited_claims'"):
            HumanReview(action="edit", edited_claims=[])

    def test_non_edit_action_forbids_edited_claims(self):
        claim = Claim(
            claim_id="claim_001",
            text="Some claim",
            evidence_ids=["ev_001"],
        )
        with pytest.raises(ValueError, match="cannot include 'edited_claims'"):
            HumanReview(action="approve", edited_claims=[claim])

        with pytest.raises(ValueError, match="cannot include 'edited_claims'"):
            HumanReview(action="reject", edited_claims=[claim])

        with pytest.raises(ValueError, match="cannot include 'edited_claims'"):
            HumanReview(action="research_more", edited_claims=[claim])


# ==============================================================================
# 2. Routing Tests
# ==============================================================================


class TestRouteAfterHumanReview:
    """TEST 10: Tests covering all deterministic post-human-review routing paths."""

    def test_approve_routes_to_end(self):
        state = {
            "human_review": HumanReview(action="approve"),
            "human_research_cycles": 0,
        }
        assert route_after_human_review(state) == "end"

    def test_edit_routes_to_end(self):
        claims = [
            Claim(
                claim_id="claim_001",
                text="Edited claim",
                evidence_ids=["ev_001"],
            )
        ]
        state = {
            "human_review": HumanReview(action="edit", edited_claims=claims),
            "human_research_cycles": 0,
        }
        assert route_after_human_review(state) == "end"

    def test_reject_routes_to_end(self):
        state = {
            "human_review": HumanReview(action="reject"),
            "human_research_cycles": 0,
        }
        assert route_after_human_review(state) == "end"

    def test_research_more_under_limit_routes_to_research(self):
        # Under limit: 0 < 2, 1 < 2
        state_cycle_0 = {
            "human_review": HumanReview(action="research_more"),
            "human_research_cycles": 0,
        }
        assert route_after_human_review(state_cycle_0) == "research"

        state_cycle_1 = {
            "human_review": HumanReview(action="research_more"),
            "human_research_cycles": 1,
        }
        assert route_after_human_review(state_cycle_1) == "research"

    def test_research_more_at_limit_routes_to_end(self):
        # At limit: 2 >= 2
        state_at_limit = {
            "human_review": HumanReview(action="research_more"),
            "human_research_cycles": 2,
        }
        assert route_after_human_review(state_at_limit) == "end"

    def test_research_more_above_limit_routes_to_end(self):
        state_above_limit = {
            "human_review": HumanReview(action="research_more"),
            "human_research_cycles": 3,
        }
        assert route_after_human_review(state_above_limit) == "end"

    def test_missing_human_review_defaults_to_end(self):
        state = {"human_research_cycles": 0}
        assert route_after_human_review(state) == "end"


# ==============================================================================
# 3. Edit Validation Tests
# ==============================================================================


class TestEditValidation:
    """Tests for validating human-edited claims against existing evidence."""

    def test_valid_edit_referencing_known_evidence_passes(self):
        available_evidence = [
            Evidence(evidence_id="ev_001", source_id="src_001", text="Snippet 1"),
            Evidence(evidence_id="ev_002", source_id="src_002", text="Snippet 2"),
        ]
        edited = [
            Claim(
                claim_id="claim_001",
                text="Updated assertion",
                evidence_ids=["ev_001", "ev_002"],
            )
        ]
        # Should not raise
        validate_edited_claims(edited, available_evidence)

    def test_invalid_edit_referencing_unknown_evidence_fails(self):
        """TEST 5: Provide an edited claim referencing unknown evidence -> clear validation failure."""
        available_evidence = [
            Evidence(evidence_id="ev_001", source_id="src_001", text="Snippet 1"),
        ]
        edited = [
            Claim(
                claim_id="claim_001",
                text="Fabricated reference",
                evidence_ids=["ev_999"],  # Unknown evidence ID
            )
        ]
        with pytest.raises(UnknownEvidenceError, match="unknown evidence_id 'ev_999'"):
            validate_edited_claims(edited, available_evidence)

    def test_duplicate_claim_ids_in_edit_fails(self):
        available_evidence = [
            Evidence(evidence_id="ev_001", source_id="src_001", text="Snippet 1"),
        ]
        edited = [
            Claim(claim_id="claim_001", text="Assertion 1", evidence_ids=["ev_001"]),
            Claim(claim_id="claim_001", text="Assertion 2", evidence_ids=["ev_001"]),
        ]
        with pytest.raises(DuplicateIdError, match="Duplicate claim_id detected"):
            validate_edited_claims(edited, available_evidence)


# ==============================================================================
# 4. Mock Pipeline Setup for Interrupt & Resume Tests
# ==============================================================================


@pytest.fixture
def mock_pipeline_components():
    """Provides predictable mock nodes for parent graph integration testing."""
    sample_sources = [
        Source(
            source_id="src_001",
            title="Benchmark Report",
            url="https://example.com/bench",
            content="GNN latency is 12ms per batch.",
        )
    ]
    sample_evidence = [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="GNN latency is 12ms per batch.",
        )
    ]
    sample_claims = [
        Claim(
            claim_id="claim_001",
            text="GNN latency averages 12ms per batch.",
            evidence_ids=["ev_001"],
        )
    ]
    sample_verifications = [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="Evidence explicitly reports 12ms per batch latency.",
            evidence_ids=["ev_001"],
        )
    ]

    research_invocations = []

    def mock_subgraph(state):
        iteration = state.get("research_iteration", 0) + 1
        research_invocations.append(
            {
                "question": state.get("question"),
                "human_feedback": state.get("human_feedback"),
                "iteration": iteration,
            }
        )
        return {
            "question": state.get("question"),
            "sources": sample_sources,
            "findings": [
                Finding(
                    finding_id="finding_001",
                    text="GNN latency averages 12ms per batch.",
                    source_ids=["src_001"],
                )
            ],
            "critique": Critique(
                quality_score=0.95,
                should_research_again=False,
            ),
            "research_iteration": iteration,
            "evidence": sample_evidence,
            "claims": sample_claims,
        }

    def mock_verifier(state):
        return {"verification_results": sample_verifications}

    return {
        "subgraph": mock_subgraph,
        "verifier": mock_verifier,
        "sample_sources": sample_sources,
        "sample_evidence": sample_evidence,
        "sample_claims": sample_claims,
        "sample_verifications": sample_verifications,
        "research_invocations": research_invocations,
    }


# ==============================================================================
# 5. Interrupt & Resume Tests
# ==============================================================================


class TestHumanReviewGraphExecution:
    """Tests proving the graph pauses at human review and resumes via Command(resume=...)."""

    def test_human_review_interrupts_and_provides_payload(self, mock_pipeline_components):
        """TEST 1: Invoke graph. Execution pauses at human review. Verify interrupt payload."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-1"}}
        result = graph.invoke(
            {"question": "What is the inference latency of GNNs?"},
            config,
        )

        # 1. State must pause at human review
        state = graph.get_state(config)
        assert state.next == ("human_review",)
        assert len(state.tasks) == 1
        assert len(state.tasks[0].interrupts) == 1

        # 2. Verify interrupt payload structure
        payload = state.tasks[0].interrupts[0].value
        assert payload["type"] == "human_review"
        assert payload["question"] == "What is the inference latency of GNNs?"

        # Must include at minimum: claims, evidence, verification_results
        assert "claims" in payload
        assert len(payload["claims"]) == 1
        assert payload["claims"][0]["claim_id"] == "claim_001"

        assert "evidence" in payload
        assert len(payload["evidence"]) == 1
        assert payload["evidence"][0]["evidence_id"] == "ev_001"

        assert "verification_results" in payload
        assert len(payload["verification_results"]) == 1
        assert payload["verification_results"][0]["verdict"] == "SUPPORTED"

        # Check structured review_items linking claim, evidence, and verifier reasoning
        assert "review_items" in payload
        review_item = payload["review_items"][0]
        assert review_item["claim_id"] == "claim_001"
        assert review_item["text"] == "GNN latency averages 12ms per batch."
        assert review_item["evidence"][0]["text"] == "GNN latency is 12ms per batch."
        assert review_item["verification"]["verdict"] == "SUPPORTED"
        assert review_item["verification"]["confidence"] == 0.98

    def test_approve_resume_reaches_end(self, mock_pipeline_components):
        """TEST 2: Resume with action='approve' -> graph reaches END."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-approve"}}
        graph.invoke({"question": "Latency of GNNs?"}, config)

        # Resume with approve action
        final_state = graph.invoke(
            Command(resume={"action": "approve"}),
            config,
        )

        # Reaches END: no remaining tasks
        post_state = graph.get_state(config)
        assert post_state.next == ()

        # Assert final human_review state recorded
        assert "human_review" in final_state
        assert final_state["human_review"].action == "approve"

    def test_reject_resume_reaches_end_with_rejection_state(self, mock_pipeline_components):
        """TEST 3: Resume with action='reject' -> graph reaches END with rejection state."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-reject"}}
        graph.invoke({"question": "Latency of GNNs?"}, config)

        # Resume with reject action
        final_state = graph.invoke(
            Command(resume={"action": "reject", "feedback": "Methodology flawed."}),
            config,
        )

        # Reaches END
        post_state = graph.get_state(config)
        assert post_state.next == ()

        # Clearly identifiable rejection state
        assert final_state["human_review"].action == "reject"
        assert final_state["human_review"].feedback == "Methodology flawed."

    def test_edit_resume_validates_and_stores_edited_claims(self, mock_pipeline_components):
        """TEST 4: Resume with action='edit' -> edited claims are validated and stored in state."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-edit"}}
        graph.invoke({"question": "Latency of GNNs?"}, config)

        edited_claims_data = [
            {
                "claim_id": "claim_001",
                "text": "GNN latency measured at exactly 12ms per batch in benchmarking.",
                "evidence_ids": ["ev_001"],
            }
        ]

        # Resume with edit action
        final_state = graph.invoke(
            Command(resume={"action": "edit", "edited_claims": edited_claims_data}),
            config,
        )

        # Reaches END
        assert graph.get_state(config).next == ()

        # Edited claims stored in state['claims']
        assert "claims" in final_state
        stored_claims = final_state["claims"]
        assert len(stored_claims) == 1
        assert stored_claims[0].claim_id == "claim_001"
        assert stored_claims[0].text == "GNN latency measured at exactly 12ms per batch in benchmarking."
        assert stored_claims[0].evidence_ids == ["ev_001"]
        assert final_state["human_review"].action == "edit"

    def test_invalid_edit_fails_validation_on_resume(self, mock_pipeline_components):
        """TEST 5 on graph: Invalid edit referencing nonexistent evidence fails with UnknownEvidenceError."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-invalid-edit"}}
        graph.invoke({"question": "Latency of GNNs?"}, config)

        bad_edit = [
            {
                "claim_id": "claim_001",
                "text": "Unfounded statement",
                "evidence_ids": ["ev_nonexistent_999"],
            }
        ]

        with pytest.raises(UnknownEvidenceError, match="unknown evidence_id 'ev_nonexistent_999'"):
            graph.invoke(
                Command(resume={"action": "edit", "edited_claims": bad_edit}),
                config,
            )

    def test_research_more_routes_back_to_research_subgraph(self, mock_pipeline_components):
        """TEST 6: Resume with research_more -> graph routes back to research subgraph."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-research-more"}}
        graph.invoke({"question": "Initial question"}, config)

        assert len(mock_pipeline_components["research_invocations"]) == 1

        # Resume with research_more
        graph.invoke(
            Command(
                resume={
                    "action": "research_more",
                    "feedback": "Find primary sources about latency.",
                }
            ),
            config,
        )

        # Graph should have executed research again and paused at human review for cycle 1
        assert len(mock_pipeline_components["research_invocations"]) == 2
        state = graph.get_state(config)
        assert state.next == ("human_review",)
        assert state.values.get("human_research_cycles") == 1

    def test_human_research_limit_strictly_enforced(self, mock_pipeline_components):
        """TEST 7: Always choose research_more. Subgraph invoked at most MAX_HUMAN_RESEARCH_CYCLES times due to human loops."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-limit"}}

        # Pass 0 (initial execution)
        graph.invoke({"question": "Initial query"}, config)
        assert len(mock_pipeline_components["research_invocations"]) == 1
        assert graph.get_state(config).values.get("human_research_cycles", 0) == 0

        # Human Cycle 1
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Cycle 1 feedback"}), config)
        assert len(mock_pipeline_components["research_invocations"]) == 2
        assert graph.get_state(config).values.get("human_research_cycles") == 1

        # Human Cycle 2 (MAX_HUMAN_RESEARCH_CYCLES = 2 reached)
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Cycle 2 feedback"}), config)
        assert len(mock_pipeline_components["research_invocations"]) == 3
        assert graph.get_state(config).values.get("human_research_cycles") == 2

        # Human requests research_more again: limit is 2, so router routes to END
        final_state = graph.invoke(
            Command(resume={"action": "research_more", "feedback": "Cycle 3 feedback"}),
            config,
        )

        # Graph must terminate at END, without running research a 4th time
        assert graph.get_state(config).next == ()
        assert len(mock_pipeline_components["research_invocations"]) == 3
        assert final_state["human_research_cycles"] == 2

    def test_human_feedback_propagation_to_next_research_cycle(self, mock_pipeline_components):
        """TEST 8: Verify that human feedback reaches the next research cycle."""
        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=mock_pipeline_components["subgraph"],
            custom_verifier=mock_pipeline_components["verifier"],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-feedback-prop"}}
        graph.invoke({"question": "Initial query"}, config)

        feedback_instruction = "Focus strictly on FP16 hardware efficiency."
        graph.invoke(
            Command(resume={"action": "research_more", "feedback": feedback_instruction}),
            config,
        )

        # Inspect the 2nd research invocation
        invocations = mock_pipeline_components["research_invocations"]
        assert len(invocations) == 2
        second_invocation = invocations[1]
        assert second_invocation["human_feedback"] == feedback_instruction
