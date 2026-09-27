"""Dedicated unit and integration test suite for the Supervisor Architecture.

Verifies:
1. Supervisor state and decision schema validation (rejection of invalid worker names).
2. Deterministic supervisor policy and routing rules.
3. Supervisor loop (Supervisor -> Worker -> Supervisor transitions).
4. Maximum supervisor steps limit (prevents infinite supervisor loops).
5. Research worker execution (encapsulated research subgraph).
6. Verifier worker execution (claim-level verification).
7. Human review worker execution (interrupt and pause).
8. Human resume actions (approve, edit, research_more, reject).
9. Human research cycle limit enforcement.
10. Checkpoint persistence compatibility with SQLite checkpointer.
11. Follow-up research session compatibility (reuse and expansion).
12. LLM supervisor policy injection and invariant protection.
13. Separation of independent step counters (supervisor_steps vs research_iteration vs human_research_cycles).
"""

from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import MagicMock
from pydantic import ValidationError
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.critic import create_critic_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.agents.supervisor import (
    DeterministicSupervisorPolicy,
    LLMSupervisorPolicy,
    create_supervisor_node,
    validate_supervisor_decision,
)
from verified_research.graph.graph import (
    build_supervisor_graph,
    create_supervisor_graph,
)
from verified_research.graph.research_subgraph import create_research_subgraph
from verified_research.graph.router import route_after_supervisor
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
from verified_research.persistence.sqlite import create_sqlite_checkpointer


# ==============================================================================
# Deterministic Test Fixtures & Mocks
# ==============================================================================


class SpySearchService:
    """Mock search service tracking query history."""

    def __init__(self, predefined_sources: list[Source] | None = None) -> None:
        self.recorded_queries: list[str] = []
        self.predefined_sources = predefined_sources or []

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        self.recorded_queries.append(query)
        if self.predefined_sources:
            return self.predefined_sources
        idx = len(self.recorded_queries)
        return [
            Source(
                source_id=f"src_spy_{idx:03d}",
                title=f"Source on {query}",
                url=f"https://example.com/item/{idx}",
                content=f"Detailed evidence concerning {query}.",
            )
        ]


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            source_id="src_001",
            title="Graph Convolutional Networks Overview",
            url="https://arxiv.org/abs/gcn-overview",
            content="GCNs compute node representations via localized first-order spectral approximations.",
        ),
        Source(
            source_id="src_002",
            title="Citation Network Benchmarks",
            url="https://arxiv.org/abs/gnn-benchmarks",
            content="Standard GNN benchmarks include Cora, Citeseer, and Pubmed citation networks.",
        ),
    ]


@pytest.fixture
def sample_evidence(sample_sources) -> list[Evidence]:
    return [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="GCNs compute node representations via localized first-order spectral approximations.",
        ),
        Evidence(
            evidence_id="ev_002",
            source_id="src_002",
            text="Standard GNN benchmarks include Cora, Citeseer, and Pubmed citation networks.",
        ),
    ]


@pytest.fixture
def sample_findings() -> list[Finding]:
    return [
        Finding(
            finding_id="finding_001",
            text="GCNs compute node representations via spectral graph convolutions.",
            source_ids=["src_001"],
        ),
        Finding(
            finding_id="finding_002",
            text="Standard GNN benchmarks include Cora and Pubmed citation networks.",
            source_ids=["src_002"],
        ),
    ]


@pytest.fixture
def sample_claims() -> list[Claim]:
    return [
        Claim(
            claim_id="claim_001",
            text="GCNs compute node representations via spectral graph convolutions.",
            evidence_ids=["ev_001"],
        ),
        Claim(
            claim_id="claim_002",
            text="Standard GNN benchmarks include Cora and Pubmed citation networks.",
            evidence_ids=["ev_002"],
        ),
    ]


@pytest.fixture
def sample_verifications() -> list[VerificationResult]:
    return [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.99,
            reasoning="GCN spectral convolution directly matches excerpt.",
            evidence_ids=["ev_001"],
        ),
        VerificationResult(
            claim_id="claim_002",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="Benchmark datasets match evidence.",
            evidence_ids=["ev_002"],
        ),
    ]


# ==============================================================================
# 1. Supervisor State & Decision Schema Tests
# ==============================================================================


class TestSupervisorDecisionSchema:
    """Test suite verifying SupervisorDecision schema validation and rejection of invalid workers."""

    def test_valid_supervisor_decisions_accepted(self):
        """Valid worker names 'research', 'verify', 'human_review', and 'finish' are accepted."""
        for worker in ("research", "verify", "human_review", "finish"):
            decision = SupervisorDecision(
                next_worker=worker,
                reasoning=f"Valid decision routing to {worker}.",
            )
            assert decision.next_worker == worker
            assert "Valid decision" in decision.reasoning

    def test_invalid_worker_names_rejected(self):
        """Arbitrary worker names such as 'search_web', 'foo', or 'random_agent' are strictly rejected."""
        invalid_names = ["search_web", "random_agent", "foo", "writer", "summarize"]
        for bad_name in invalid_names:
            with pytest.raises(ValidationError):
                SupervisorDecision(
                    next_worker=bad_name,  # type: ignore[arg-type]
                    reasoning="Attempting invalid worker.",
                )

    def test_extra_fields_forbidden(self):
        """Extra unexpected fields are forbidden by pydantic model config."""
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_worker="research",
                reasoning="Reasoning",
                unrecognized_extra="not_allowed",  # type: ignore[call-arg]
            )

    def test_empty_reasoning_rejected(self):
        """Reasoning cannot be empty string."""
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_worker="research",
                reasoning="",
            )


# ==============================================================================
# 2. Deterministic Supervisor Routing Tests
# ==============================================================================


class TestDeterministicSupervisorRouting:
    """Test suite verifying deterministic supervisor policy routing across state variations."""

    def test_route_to_research_when_no_sources_or_claims(self):
        policy = DeterministicSupervisorPolicy()
        state = {"question": "What is Quantum Computing?"}
        decision = policy.evaluate(state)
        assert decision.next_worker == "research"

    def test_route_to_verify_when_unverified_claims_exist(self, sample_sources, sample_findings, sample_claims):
        policy = DeterministicSupervisorPolicy()
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": [],  # No verifications completed yet
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "verify"

    def test_route_to_human_review_when_all_claims_verified(
        self, sample_sources, sample_findings, sample_claims, sample_verifications
    ):
        policy = DeterministicSupervisorPolicy()
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": sample_verifications,
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "human_review"

    def test_route_to_finish_on_approved_human_review(
        self, sample_sources, sample_findings, sample_claims, sample_verifications
    ):
        policy = DeterministicSupervisorPolicy()
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": sample_verifications,
            "human_review": HumanReview(action="approve"),
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "finish"

    def test_route_to_finish_on_rejected_human_review(
        self, sample_sources, sample_findings, sample_claims, sample_verifications
    ):
        policy = DeterministicSupervisorPolicy()
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": sample_verifications,
            "human_review": HumanReview(action="reject", feedback="Insufficient evidence."),
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "finish"

    def test_route_to_research_on_research_more_action_within_limit(
        self, sample_sources, sample_findings, sample_claims, sample_verifications
    ):
        policy = DeterministicSupervisorPolicy(max_human_cycles=2)
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": sample_verifications,
            "human_review": HumanReview(action="research_more", feedback="Find primary benchmarks."),
            "human_research_cycles": 0,
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "research"

    def test_route_to_finish_on_research_more_when_cycle_limit_reached(
        self, sample_sources, sample_findings, sample_claims, sample_verifications
    ):
        policy = DeterministicSupervisorPolicy(max_human_cycles=2)
        state = {
            "question": "What is GCN?",
            "sources": sample_sources,
            "findings": sample_findings,
            "claims": sample_claims,
            "verification_results": sample_verifications,
            "human_review": HumanReview(action="research_more", feedback="Keep researching."),
            "human_research_cycles": 2,  # Limit reached
        }
        decision = policy.evaluate(state)
        assert decision.next_worker == "finish"


# ==============================================================================
# 3. Supervisor Loop Execution Tests
# ==============================================================================


class TestSupervisorLoopExecution:
    """Test suite verifying that Supervisor -> Worker -> Supervisor actually occurs."""

    def test_supervisor_loop_transitions_occur(self, sample_sources, sample_evidence, sample_findings, sample_claims):
        """Track transition history to verify that every worker returns control to the supervisor."""
        transition_history: list[str] = []

        def tracking_supervisor(state):
            step = state.get("supervisor_steps", 0)
            transition_history.append(f"supervisor_step_{step + 1}")
            # Delegate to standard deterministic node logic
            default_sup = create_supervisor_node()
            return default_sup(state)

        def tracking_researcher(state):
            transition_history.append("research_worker")
            return {
                "sources": sample_sources,
                "findings": sample_findings,
                "evidence": sample_evidence,
                "claims": sample_claims,
                "research_iteration": 1,
            }

        def tracking_verifier(state):
            transition_history.append("verifier_worker")
            return {
                "verification_results": [
                    VerificationResult(
                        claim_id=c.claim_id,
                        verdict="SUPPORTED",
                        confidence=0.95,
                        reasoning=f"Verified {c.claim_id}.",
                        evidence_ids=c.evidence_ids,
                    )
                    for c in state.get("claims", [])
                ]
            }

        def tracking_human_review(state):
            transition_history.append("human_review_worker")
            # In automated test without interactive pause, simulate approved review
            return {"human_review": HumanReview(action="approve")}

        builder = build_supervisor_graph(
            custom_subgraph=tracking_researcher,
            custom_verifier=tracking_verifier,
            custom_human_review=tracking_human_review,
            custom_supervisor=tracking_supervisor,
        )
        graph = builder.compile()

        final_state = graph.invoke({"question": "What is GCN?"})

        # Expected flow:
        # 1. supervisor_step_1 -> routes to research
        # 2. research_worker -> routes back to supervisor
        # 3. supervisor_step_2 -> routes to verifier
        # 4. verifier_worker -> routes back to supervisor
        # 5. supervisor_step_3 -> routes to human_review
        # 6. human_review_worker -> routes back to supervisor
        # 7. supervisor_step_4 -> finishes (routes to END)
        assert transition_history == [
            "supervisor_step_1",
            "research_worker",
            "supervisor_step_2",
            "verifier_worker",
            "supervisor_step_3",
            "human_review_worker",
            "supervisor_step_4",
        ]
        assert final_state["supervisor_steps"] == 4
        assert final_state["supervisor_decision"].next_worker == "finish"


# ==============================================================================
# 4. Maximum Supervisor Steps Tests
# ==============================================================================


class TestSupervisorStepLimits:
    """Test suite verifying that maximum supervisor steps strictly prevents infinite loops."""

    def test_max_supervisor_steps_halts_execution(self):
        """Graph configured with max_supervisor_steps=3 stops safely after 3 steps."""
        step_count = 0

        # Mock a loop between supervisor and research
        def perpetual_researcher(state):
            nonlocal step_count
            step_count += 1
            # Deliberately return empty claims to induce supervisor to keep requesting research
            return {"sources": [], "claims": []}

        graph = create_supervisor_graph(
            custom_subgraph=perpetual_researcher,
            max_supervisor_steps=3,
        )

        final_state = graph.invoke({"question": "Endless loop query"})

        # Must halt at 3 steps
        assert final_state["supervisor_steps"] >= 3
        assert final_state["supervisor_decision"].next_worker == "finish"
        assert final_state.get("supervisor_termination_reason") == "MAX_SUPERVISOR_STEPS_REACHED"


# ==============================================================================
# 5. Specialized Worker Execution Tests
# ==============================================================================


class TestSpecializedWorkersExecution:
    """Test suite verifying Research, Verifier, and Human Review workers execute properly."""

    def test_research_worker_runs_encapsulated_subgraph(self):
        """Research worker executes the full encapsulated research subgraph."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)

        def mock_analyst(state):
            sources = state.get("sources", [])
            src_id = sources[0].source_id if sources else "src_001"
            return {
                "evidence": [
                    Evidence(
                        evidence_id="ev_01",
                        source_id=src_id,
                        text="GNN excerpt.",
                    )
                ],
                "findings": [
                    Finding(
                        finding_id="f_01",
                        text="GNN finding.",
                        source_ids=[src_id],
                    )
                ],
                "claims": [
                    Claim(
                        claim_id="cl_01",
                        text="GNN claim.",
                        evidence_ids=["ev_01"],
                    )
                ],
            }

        critic = lambda state: {"critique": Critique(quality_score=0.95, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=mock_analyst,
            custom_critic=critic,
        )

        mock_verifier = lambda state: {
            "verification_results": [
                VerificationResult(
                    claim_id="cl_01",
                    verdict="SUPPORTED",
                    confidence=0.99,
                    reasoning="Verified via mock.",
                    evidence_ids=["ev_01"],
                )
            ]
        }
        mock_human = lambda state: {"human_review": HumanReview(action="approve")}

        graph = create_supervisor_graph(
            custom_subgraph=subgraph,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        final_state = graph.invoke({"question": "What are GNNs?"})
        assert len(search_spy.recorded_queries) > 0
        assert "sources" in final_state
        assert "findings" in final_state
        assert "claims" in final_state

    def test_verifier_worker_evaluates_claims_against_evidence(
        self, sample_sources, sample_evidence, sample_findings, sample_claims
    ):
        """Verifier worker independently evaluates claims without web search."""
        verified_claims = []

        def mock_verifier(claim, evidence):
            verified_claims.append(claim.claim_id)
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.98,
                reasoning="Verified.",
                evidence_ids=claim.evidence_ids,
            )

        mock_researcher = lambda state: {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": sample_findings,
            "claims": sample_claims,
        }
        mock_human = lambda state: {"human_review": HumanReview(action="approve")}

        graph = create_supervisor_graph(
            custom_subgraph=mock_researcher,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        final_state = graph.invoke({"question": "What is GCN?"})
        assert len(verified_claims) == len(sample_claims)
        assert len(final_state["verification_results"]) == len(sample_claims)


# ==============================================================================
# 6. Human Review Worker & Resume Actions Tests
# ==============================================================================


class TestHumanReviewInteraction:
    """Test suite verifying Human Review interrupt, resume actions, and cycle bounds."""

    def test_human_review_interrupts_and_resumes_with_approve(
        self, sample_sources, sample_evidence, sample_findings, sample_claims, sample_verifications
    ):
        checkpointer = MemorySaver()
        mock_researcher = lambda state: {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": sample_findings,
            "claims": sample_claims,
        }
        mock_verifier = lambda state: {"verification_results": sample_verifications}

        graph = create_supervisor_graph(
            custom_subgraph=mock_researcher,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-sup-approve"}}
        graph.invoke({"question": "What is GCN?"}, config)

        # 1. Pauses at human_review interrupt
        state = graph.get_state(config)
        assert state.next == ("human_review",)

        # 2. Resume with approve
        final_state = graph.invoke(Command(resume={"action": "approve"}), config)
        assert graph.get_state(config).next == ()
        assert final_state["human_review"].action == "approve"
        assert final_state["supervisor_decision"].next_worker == "finish"

    def test_human_review_resume_with_edit(
        self, sample_sources, sample_evidence, sample_findings, sample_claims, sample_verifications
    ):
        checkpointer = MemorySaver()
        mock_researcher = lambda state: {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": sample_findings,
            "claims": sample_claims,
        }
        mock_verifier = lambda state: {"verification_results": sample_verifications}

        graph = create_supervisor_graph(
            custom_subgraph=mock_researcher,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-sup-edit"}}
        graph.invoke({"question": "What is GCN?"}, config)

        edited_claims = [
            {
                "claim_id": "claim_001",
                "text": "Edited verified claim regarding GCNs.",
                "evidence_ids": ["ev_001"],
            }
        ]
        final_state = graph.invoke(
            Command(resume={"action": "edit", "edited_claims": edited_claims}),
            config,
        )
        assert graph.get_state(config).next == ()
        assert final_state["human_review"].action == "edit"
        assert final_state["claims"][0].text == "Edited verified claim regarding GCNs."
        assert final_state["supervisor_decision"].next_worker == "finish"

    def test_human_research_cycle_limit_enforced_by_supervisor(
        self, sample_sources, sample_evidence, sample_findings, sample_claims, sample_verifications
    ):
        """Repeated research_more requests hit max_human_research_cycles and terminate at finish."""
        checkpointer = MemorySaver()
        research_passes = 0

        def counting_researcher(state):
            nonlocal research_passes
            research_passes += 1
            return {
                "sources": sample_sources,
                "evidence": sample_evidence,
                "findings": sample_findings,
                "claims": sample_claims,
            }

        mock_verifier = lambda state: {"verification_results": sample_verifications}

        graph = create_supervisor_graph(
            custom_subgraph=counting_researcher,
            custom_verifier=mock_verifier,
            max_human_research_cycles=2,
            max_supervisor_steps=15,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-sup-cycles"}}

        # Initial pass
        graph.invoke({"question": "What is GCN?"}, config)
        assert graph.get_state(config).next == ("human_review",)
        assert research_passes == 1

        # Cycle 1: research_more
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Cycle 1"}), config)
        assert graph.get_state(config).next == ("human_review",)
        assert research_passes == 2

        # Cycle 2: research_more (reaches limit)
        graph.invoke(Command(resume={"action": "research_more", "feedback": "Cycle 2"}), config)
        assert graph.get_state(config).next == ("human_review",)
        assert research_passes == 3

        # Cycle 3: research_more beyond cap -> supervisor terminates with finish
        final_state = graph.invoke(Command(resume={"action": "research_more", "feedback": "Cycle 3"}), config)
        assert graph.get_state(config).next == ()
        assert research_passes == 3  # Did not run research a 4th time
        assert final_state["supervisor_decision"].next_worker == "finish"


# ==============================================================================
# 7. Checkpoint Persistence Compatibility Tests
# ==============================================================================


class TestPersistenceCompatibility:
    """Test suite verifying supervisor execution with SQLite checkpointer."""

    def test_supervisor_with_sqlite_checkpointer_and_resume(
        self, sample_sources, sample_evidence, sample_findings, sample_claims, sample_verifications
    ):
        mock_researcher = lambda state: {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": sample_findings,
            "claims": sample_claims,
        }
        mock_verifier = lambda state: {"verification_results": sample_verifications}

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "checkpoints.sqlite"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                checkpointer = create_sqlite_checkpointer(conn)
                graph = create_supervisor_graph(
                    custom_subgraph=mock_researcher,
                    custom_verifier=mock_verifier,
                    checkpointer=checkpointer,
                )

                thread_config = {"configurable": {"thread_id": "thread-sqlite-sup"}}

                # Run until interrupt
                graph.invoke({"question": "What is GCN architecture?"}, thread_config)
                paused = graph.get_state(thread_config)
                assert paused.next == ("human_review",)
                assert paused.values["supervisor_decision"].next_worker == "human_review"

                # Resume on same thread
                final_state = graph.invoke(Command(resume={"action": "approve"}), thread_config)
                assert graph.get_state(thread_config).next == ()
                assert final_state["supervisor_decision"].next_worker == "finish"
                assert final_state["human_review"].action == "approve"
            finally:
                conn.close()


# ==============================================================================
# 8. Follow-Up Research Sessions Compatibility Tests
# ==============================================================================


class TestFollowUpCompatibility:
    """Test suite verifying supervisor handles follow-up reuse and expansion."""

    def test_supervisor_handles_follow_up_reuse(
        self, sample_sources, sample_evidence, sample_findings, sample_claims
    ):
        """When evidence is sufficient, supervisor routes to verify (skipping research)."""
        search_spy = SpySearchService(predefined_sources=sample_sources)
        researcher = create_researcher_node(search_client=search_spy)
        analyst = lambda state: {"findings": sample_findings}
        critic = lambda state: {"critique": Critique(quality_score=0.9, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        mock_verifier = lambda claim, evidence: VerificationResult(
            claim_id=claim.claim_id,
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning=f"Verified {claim.claim_id}.",
            evidence_ids=claim.evidence_ids,
        )

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(
            custom_subgraph=subgraph,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-sup-reuse"}}

        # Preloaded state with existing GNN evidence
        prior_state = {
            "question": "What is GNN architecture?",
            "sources": sample_sources,
            "evidence": sample_evidence,
            "findings": sample_findings,
            "previous_questions": ["What is GNN architecture?"],
        }

        # Follow-up question covered by existing evidence
        graph.invoke(
            {
                **prior_state,
                "follow_up_question": "What are standard citation benchmarks like Cora and Pubmed?",
            },
            config,
        )

        # Pauses at human review
        paused = graph.get_state(config)
        assert paused.next == ("human_review",)

        # Research subgraph search was NOT called (reused evidence!)
        assert len(search_spy.recorded_queries) == 0
        assert paused.values["reuse_decision"].decision == "REUSE"


# ==============================================================================
# 9. LLM Supervisor Policy & Invariant Guard Tests
# ==============================================================================


class TestLLMSupervisorPolicy:
    """Test suite verifying LLM supervisor injection, structured output, and invariant enforcement."""

    def test_llm_supervisor_mock_injection(self):
        """Mock LLM returning structured SupervisorDecision is honored."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = SupervisorDecision(
            next_worker="verify",
            reasoning="Mock LLM decided claims require verification.",
        )
        mock_llm.with_structured_output.return_value = structured_mock

        policy = LLMSupervisorPolicy(llm=mock_llm)
        state = {
            "question": "Test query",
            "claims": [Claim(claim_id="c1", text="Text", evidence_ids=["e1"])],
            "verification_results": [],
        }

        decision = policy.evaluate(state)
        assert decision.next_worker == "verify"
        assert "Mock LLM" in decision.reasoning

    def test_llm_supervisor_invariant_violation_overridden(self):
        """If LLM attempts to route to human_review with unverified claims, invariant guard overrides to verify."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        # Malicious or confused LLM attempts to skip verification:
        structured_mock.invoke.return_value = SupervisorDecision(
            next_worker="human_review",
            reasoning="LLM mistakenly trying to bypass verification.",
        )
        mock_llm.with_structured_output.return_value = structured_mock

        policy = LLMSupervisorPolicy(llm=mock_llm)
        state = {
            "question": "Test query",
            "claims": [Claim(claim_id="c1", text="Unverified claim", evidence_ids=["e1"])],
            "verification_results": [],  # Unverified!
        }

        decision = policy.evaluate(state)
        # Must be overridden to verify
        assert decision.next_worker == "verify"
        assert "invariant violation" in decision.reasoning.lower()

    def test_llm_supervisor_fallback_on_error(self):
        """If LLM call fails, supervisor gracefully falls back to deterministic policy."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = RuntimeError("API unavailable")
        mock_llm.with_structured_output.return_value = structured_mock

        policy = LLMSupervisorPolicy(llm=mock_llm)
        state = {"question": "Initial question"}  # Empty state should route to research

        decision = policy.evaluate(state)
        assert decision.next_worker == "research"


# ==============================================================================
# 10. Separation of Step Counters Tests
# ==============================================================================


class TestStepCountersSeparation:
    """Test suite verifying supervisor_steps, research_iteration, and human_research_cycles are separate."""

    def test_step_counters_are_independent(self, sample_sources, sample_evidence, sample_findings, sample_claims, sample_verifications):
        checkpointer = MemorySaver()

        def mock_researcher(state):
            # Increments research_iteration
            it = state.get("research_iteration", 0) + 1
            return {
                "sources": sample_sources,
                "evidence": sample_evidence,
                "findings": sample_findings,
                "claims": sample_claims,
                "research_iteration": it,
            }

        mock_verifier = lambda state: {"verification_results": sample_verifications}

        graph = create_supervisor_graph(
            custom_subgraph=mock_researcher,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-counters-test"}}
        graph.invoke({"question": "What is GCN?"}, config)

        state = graph.get_state(config).values
        # supervisor_steps has advanced through supervisor transitions (>= 3)
        assert state.get("supervisor_steps", 0) >= 3
        # research_iteration counted passes within the subgraph (1)
        assert state.get("research_iteration") == 1
        # human_research_cycles has not occurred yet (0)
        assert state.get("human_research_cycles", 0) == 0
