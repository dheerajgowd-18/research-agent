"""Dedicated test suite for follow-up research sessions in Verified Research Agent.

Verifies:
1. First research run followed by related question.
2. Sufficient existing evidence -> reuse path taken (no web search).
3. Insufficient evidence -> research subgraph path taken.
4. Temporal mismatch detection (e.g. 2024 evidence vs 2026 question -> RESEARCH_MORE).
5. Entity/topic mismatch detection (unrelated research -> RESEARCH_MORE).
6. Claim-specific verification (Claim B independently verified against Evidence E1).
7. Evidence traceability (follow-up claims preserve Claim -> Evidence -> Source chain).
8. Research expansion (researcher receives follow-up objective, not original question).
9. Thread/session persistence with SQLite checkpointer across multi-question sessions.
10. Strict adherence to mock/injected services with zero live LLM or web calls.
"""

from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import MagicMock
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.critic import create_critic_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.agents.sufficiency import (
    HeuristicSufficiencyService,
    create_evaluate_sufficiency_node,
    create_reuse_analyst_node,
)
from verified_research.agents.verifier import create_verifier_node
from verified_research.graph.graph import (
    build_research_graph,
    create_research_graph,
)
from verified_research.graph.research_subgraph import create_research_subgraph
from verified_research.graph.router import route_after_sufficiency
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    ResearchReuseDecision,
    Source,
    VerificationResult,
)
from verified_research.models.traceability import validate_traceability
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
                title=f"Results for {query}",
                url=f"https://example.com/search/{idx}",
                content=f"Substantive research findings and evidence regarding {query}.",
            )
        ]


@pytest.fixture
def base_sources() -> list[Source]:
    return [
        Source(
            source_id="src_001",
            title="Graph Convolutional Networks Overview (2024)",
            url="https://arxiv.org/abs/gcn-2024",
            content="GCNs perform spectral graph convolutions using layer-wise propagation rules published in 2024.",
        ),
        Source(
            source_id="src_002",
            title="Graph Neural Network Benchmark Datasets",
            url="https://arxiv.org/abs/gnn-benchmarks",
            content="Standard GNN benchmarks include Cora, Citeseer, and Pubmed citation networks.",
        ),
    ]


@pytest.fixture
def base_evidence(base_sources) -> list[Evidence]:
    return [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="GCNs perform spectral graph convolutions using layer-wise propagation rules published in 2024.",
        ),
        Evidence(
            evidence_id="ev_002",
            source_id="src_002",
            text="Standard GNN benchmarks include Cora, Citeseer, and Pubmed citation networks.",
        ),
    ]


@pytest.fixture
def base_findings() -> list[Finding]:
    return [
        Finding(
            finding_id="finding_001",
            text="GCNs utilize spectral graph convolutions with localized first-order approximations.",
            source_ids=["src_001"],
        ),
        Finding(
            finding_id="finding_002",
            text="Common citation network benchmarks for GNNs are Cora, Citeseer, and Pubmed.",
            source_ids=["src_002"],
        ),
    ]


@pytest.fixture
def base_claims() -> list[Claim]:
    return [
        Claim(
            claim_id="claim_001",
            text="GCNs utilize spectral graph convolutions with localized first-order approximations.",
            evidence_ids=["ev_001"],
        ),
        Claim(
            claim_id="claim_002",
            text="Common citation network benchmarks for GNNs are Cora, Citeseer, and Pubmed.",
            evidence_ids=["ev_002"],
        ),
    ]


# ==============================================================================
# 1. Sufficiency Evaluator Unit Tests
# ==============================================================================


class TestSufficiencyEvaluator:
    """Test suite for research sufficiency evaluation logic."""

    def test_sufficiency_initial_run_requires_research(self):
        """Initial run with no existing sources or evidence always evaluates to RESEARCH_MORE."""
        service = HeuristicSufficiencyService()
        node = create_evaluate_sufficiency_node(sufficiency_service=service)

        state = {"question": "What is Quantum Computing?"}
        updates = node(state)

        assert updates["question"] == "What is Quantum Computing?"
        assert updates["reuse_decision"].decision == "RESEARCH_MORE"
        assert "Initial research" in updates["reuse_decision"].reasoning
        assert "previous_questions" in updates
        assert updates["previous_questions"] == ["What is Quantum Computing?"]

    def test_temporal_mismatch_detection(self, base_sources, base_evidence, base_findings, base_claims):
        """Temporal mismatch: Evidence from 2024 cannot answer a 2026 query -> RESEARCH_MORE."""
        service = HeuristicSufficiencyService()

        # Evidence only mentions 2024; question asks about 2026
        decision = service.evaluate_sufficiency(
            question="What are recent 2026 developments in Graph Neural Networks?",
            sources=base_sources,
            findings=base_findings,
            claims=base_claims,
            evidence=base_evidence,
        )

        assert decision.decision == "RESEARCH_MORE"
        assert "2026" in decision.reasoning
        assert any("2026" in topic for topic in decision.missing_topics)

    def test_entity_topic_mismatch_detection(self, base_sources, base_evidence, base_findings, base_claims):
        """Entity mismatch: Evidence on GNNs cannot answer a query on CRISPR -> RESEARCH_MORE."""
        service = HeuristicSufficiencyService()

        decision = service.evaluate_sufficiency(
            question="How does CRISPR Cas9 gene editing guide RNA binding work?",
            sources=base_sources,
            findings=base_findings,
            claims=base_claims,
            evidence=base_evidence,
        )

        assert decision.decision == "RESEARCH_MORE"
        assert "topics or entities not covered" in decision.reasoning.lower()
        assert len(decision.missing_topics) > 0

    def test_sufficient_existing_evidence_evaluates_to_reuse(
        self, base_sources, base_evidence, base_findings, base_claims
    ):
        """Scope coverage: Query asking about citation benchmarks already in evidence -> REUSE."""
        service = HeuristicSufficiencyService()

        decision = service.evaluate_sufficiency(
            question="What are the standard citation network benchmarks for GNNs like Cora and Citeseer?",
            sources=base_sources,
            findings=base_findings,
            claims=base_claims,
            evidence=base_evidence,
        )

        assert decision.decision == "REUSE"
        assert len(decision.missing_topics) == 0


# ==============================================================================
# 2. Sufficiency Router Unit Tests
# ==============================================================================


class TestSufficiencyRouter:
    """Test suite for deterministic routing after sufficiency evaluation."""

    def test_route_reuse_decision_to_reuse_synthesis(self):
        state = {
            "question": "Follow up question",
            "reuse_decision": ResearchReuseDecision(
                decision="REUSE",
                reasoning="Existing evidence is sufficient.",
                missing_topics=[],
            ),
        }
        dest = route_after_sufficiency(state)
        assert dest == "reuse_synthesis"

    def test_route_research_more_decision_to_research(self):
        state = {
            "question": "Follow up question",
            "reuse_decision": ResearchReuseDecision(
                decision="RESEARCH_MORE",
                reasoning="Missing 2026 data.",
                missing_topics=["2026"],
            ),
        }
        dest = route_after_sufficiency(state)
        assert dest == "research"

    def test_route_missing_decision_defaults_to_research(self):
        state = {"question": "Any question"}
        dest = route_after_sufficiency(state)
        assert dest == "research"


# ==============================================================================
# 3. Reuse Path Integration Tests (No Web Search)
# ==============================================================================


class TestReusePathExecution:
    """Test suite verifying reuse path synthesizes claims and routes to verification without search."""

    def test_sufficient_evidence_takes_reuse_path_without_search(
        self, base_sources, base_evidence, base_findings
    ):
        """When evidence is sufficient, graph takes reuse path: research subgraph is NOT called."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = lambda state: {"findings": base_findings}
        critic = lambda state: {"critique": Critique(quality_score=0.9, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        verified_claims = []

        def mock_verifier(claim, evidence):
            verified_claims.append(claim.claim_id)
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.98,
                reasoning=f"Verified {claim.claim_id} against {claim.evidence_ids}.",
                evidence_ids=claim.evidence_ids,
            )

        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=subgraph,
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-reuse"}}

        # Preload state as if Session 1 completed with GNN benchmarks
        initial_state = {
            "question": "What are GNN citation benchmarks?",
            "sources": base_sources,
            "evidence": base_evidence,
            "findings": base_findings,
            "previous_questions": ["What is GNN architecture?"],
        }

        # Invoke with a follow-up covered by existing evidence
        graph.invoke(
            {
                **initial_state,
                "follow_up_question": "What are the standard citation benchmarks for GNNs like Cora and Pubmed?",
            },
            config,
        )

        # 1. State must pause at human review
        paused_state = graph.get_state(config)
        assert paused_state.next == ("human_review",)

        # 2. Researcher search must NOT have been called (reused existing evidence!)
        assert len(search_spy.recorded_queries) == 0

        # 3. Reuse decision must be REUSE
        state_values = paused_state.values
        assert state_values["reuse_decision"].decision == "REUSE"

        # 4. Verifier was called on the newly synthesized follow-up claims
        assert len(verified_claims) > 0
        assert len(state_values["verification_results"]) == len(state_values["claims"])

        # 5. Resume human review with approve
        final_state = graph.invoke(Command(resume={"action": "approve"}), config)
        assert graph.get_state(config).next == ()
        assert final_state["human_review"].action == "approve"


# ==============================================================================
# 4. Research Expansion Path Integration Tests
# ==============================================================================


class TestResearchExpansionPath:
    """Test suite verifying research subgraph re-enters with the follow-up objective."""

    def test_insufficient_evidence_searches_follow_up_objective_not_original(
        self, base_sources, base_evidence, base_findings
    ):
        """When evidence is insufficient, researcher executes search for the follow-up objective."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)

        def mock_analyst(state):
            sources = state.get("sources", [])
            return {
                "findings": [
                    Finding(
                        finding_id="finding_expansion_001",
                        text="2026 GNN frameworks feature continuous dynamic graph convolution.",
                        source_ids=[sources[-1].source_id] if sources else ["src_001"],
                    )
                ]
            }

        critic = lambda state: {"critique": Critique(quality_score=0.92, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=mock_analyst,
            custom_critic=critic,
        )

        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=subgraph,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-test-expansion"}}

        initial_state = {
            "question": "What is GNN architecture?",
            "sources": base_sources,
            "evidence": base_evidence,
            "findings": base_findings,
            "previous_questions": ["What is GNN architecture?"],
        }

        # Follow-up question has a 2026 temporal mismatch
        follow_up_q = "What are recent 2026 developments in Graph Neural Networks?"
        graph.invoke(
            {
                **initial_state,
                "follow_up_question": follow_up_q,
            },
            config,
        )

        # 1. Reuse decision is RESEARCH_MORE
        paused_state = graph.get_state(config)
        assert paused_state.values["reuse_decision"].decision == "RESEARCH_MORE"

        # 2. Researcher was invoked and searched for follow-up objective, NOT original question
        assert len(search_spy.recorded_queries) > 0
        searched_text = " ".join(search_spy.recorded_queries)
        assert "2026" in searched_text
        assert "What is GNN architecture?" not in search_spy.recorded_queries

        # 3. New sources were appended to existing sources
        assert len(paused_state.values["sources"]) > len(base_sources)


# ==============================================================================
# 5. Claim-Specific Independent Verification Tests
# ==============================================================================


class TestClaimSpecificVerification:
    """Test suite verifying follow-up claims are independently verified even when reusing evidence."""

    def test_claim_b_independently_verified_against_evidence_e1(self, base_sources, base_evidence):
        """Claim B derived in follow-up must be independently evaluated by Verifier against Evidence E1."""
        verifier_evaluations: list[dict] = []

        def mock_verifier(claim, evidence):
            verifier_evaluations.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.text,
                "evidence_ids": claim.evidence_ids,
            })
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.96,
                reasoning=f"Independently verified {claim.claim_id} against {claim.evidence_ids}.",
                evidence_ids=claim.evidence_ids,
            )

        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_verifier=mock_verifier,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-claim-verification"}}

        # In Session 1, Claim A was verified against ev_001
        session_1_state = {
            "question": "What is spectral graph convolution?",
            "sources": base_sources,
            "evidence": base_evidence,
            "findings": [
                Finding(
                    finding_id="finding_001",
                    text="GCNs perform spectral graph convolutions using layer-wise propagation.",
                    source_ids=["src_001"],
                )
            ],
            "claims": [
                Claim(
                    claim_id="claim_A",
                    text="GCNs perform spectral graph convolutions using layer-wise propagation.",
                    evidence_ids=["ev_001"],
                )
            ],
            "verification_results": [
                VerificationResult(
                    claim_id="claim_A",
                    verdict="SUPPORTED",
                    confidence=0.99,
                    reasoning="Claim A verified in session 1.",
                    evidence_ids=["ev_001"],
                )
            ],
        }

        # Follow-up session asks related question reusing ev_001
        graph.invoke(
            {
                **session_1_state,
                "follow_up_question": "Explain the layer-wise propagation rule of spectral graph convolutions.",
            },
            config,
        )

        # Verify: Claim B was generated and passed to verifier
        assert len(verifier_evaluations) > 0
        evaluated_claim_ids = [e["claim_id"] for e in verifier_evaluations]
        # Must verify new claim B, NOT simply pass old verification result of Claim A
        assert "claim_A" not in evaluated_claim_ids
        for ev in verifier_evaluations:
            assert "ev_001" in ev["evidence_ids"]


# ==============================================================================
# 6. Evidence Traceability Chain Tests
# ==============================================================================


class TestEvidenceTraceability:
    """Test suite verifying follow-up claims preserve the Claim -> Evidence -> Source chain."""

    def test_reused_claims_preserve_complete_traceability(
        self, base_sources, base_evidence, base_findings
    ):
        """Claims generated in reuse path must pass structural traceability validation."""
        reuse_analyst = create_reuse_analyst_node()

        state = {
            "question": "How are citation benchmarks structured for GNNs?",
            "sources": base_sources,
            "evidence": base_evidence,
            "findings": base_findings,
        }

        output = reuse_analyst(state)
        claims = output["claims"]
        findings = output["findings"]

        assert len(claims) > 0
        # Enforce structural referential integrity: Claim -> Evidence -> Source
        validate_traceability(claims, base_evidence, base_sources)

        for claim in claims:
            assert len(claim.evidence_ids) > 0
            for ev_id in claim.evidence_ids:
                matching_ev = next(e for e in base_evidence if e.evidence_id == ev_id)
                assert matching_ev.source_id in {s.source_id for s in base_sources}


# ==============================================================================
# 7. Persistence Across Follow-Up Sessions (SQLite Checkpointer)
# ==============================================================================


class TestMultiSessionPersistence:
    """Test suite verifying SQLite checkpoint restoration across multi-question sessions on same thread."""

    def test_multi_question_session_on_same_thread_with_sqlite(
        self, base_sources, base_evidence, base_findings
    ):
        """Simulate two sequential research sessions on the same thread using SQLite persistence."""
        search_spy = SpySearchService(predefined_sources=base_sources)
        researcher = create_researcher_node(search_client=search_spy)
        analyst = lambda state: {
            "findings": base_findings,
            "evidence": base_evidence,
            "claims": [
                Claim(
                    claim_id="claim_001",
                    text="GCNs perform spectral graph convolutions using layer-wise propagation.",
                    evidence_ids=["ev_001"],
                )
            ],
        }
        critic = lambda state: {"critique": Critique(quality_score=0.95, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        def mock_verifier(claim, evidence):
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.95,
                reasoning=f"Verified {claim.claim_id}.",
                evidence_ids=claim.evidence_ids,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "checkpoints.sqlite"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                checkpointer = create_sqlite_checkpointer(conn)

                graph = create_research_graph(
                    custom_subgraph=subgraph,
                    custom_verifier=mock_verifier,
                    checkpointer=checkpointer,
                )

                thread_config = {"configurable": {"thread_id": "thread-followup-sqlite-1"}}

                # ------------------------------------------------------------------
                # SESSION 1: Initial research query
                # ------------------------------------------------------------------
                graph.invoke(
                    {"question": "What is the architecture of Graph Convolutional Networks?"},
                    thread_config,
                )

                # Interrupts at human review
                state_s1_pause = graph.get_state(thread_config)
                assert state_s1_pause.next == ("human_review",)
                assert state_s1_pause.values["question"] == "What is the architecture of Graph Convolutional Networks?"

                # Human approves Session 1
                graph.invoke(Command(resume={"action": "approve"}), thread_config)
                state_s1_end = graph.get_state(thread_config)
                assert state_s1_end.next == ()

                # ------------------------------------------------------------------
                # SESSION 2: Follow-up question on same thread (reuse path)
                # ------------------------------------------------------------------
                follow_up_q = "What are standard citation benchmarks for GCNs like Cora?"
                graph.invoke(
                    {"follow_up_question": follow_up_q},
                    thread_config,
                )

                # Pauses at human review in Session 2
                state_s2_pause = graph.get_state(thread_config)
                assert state_s2_pause.next == ("human_review",)

                # Assert state continuity & session boundary
                values_s2 = state_s2_pause.values
                assert values_s2["question"] == follow_up_q
                assert values_s2["reuse_decision"].decision == "REUSE"
                assert "What is the architecture of Graph Convolutional Networks?" in values_s2["previous_questions"]
                assert follow_up_q in values_s2["previous_questions"]

                # Verify claims are follow-up claims
                assert len(values_s2["claims"]) > 0
                assert len(values_s2["verification_results"]) == len(values_s2["claims"])

                # Human approves Session 2
                graph.invoke(Command(resume={"action": "approve"}), thread_config)
                state_s2_end = graph.get_state(thread_config)
                assert state_s2_end.next == ()
                assert state_s2_end.values["human_review"].action == "approve"

                # ------------------------------------------------------------------
                # Verify Checkpointer History
                # ------------------------------------------------------------------
                history = list(checkpointer.list(thread_config))
                assert len(history) >= 4  # multiple checkpoints recorded across both sessions
            finally:
                conn.close()


# ==============================================================================
# 8. End-to-End Session Continuity Lifecycle Tests
# ==============================================================================


class TestEndToEndSessionLifecycle:
    """End-to-end integration tests for multi-question session lifecycle."""

    def test_first_research_followed_by_expansion_question(self, base_sources, base_evidence):
        """Run session 1, complete with approval, then run session 2 with temporal expansion."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)

        def dynamic_analyst(state):
            question = state.get("question", "")
            sources = state.get("sources", [])
            return {
                "findings": [
                    Finding(
                        finding_id=f"finding_{len(sources)}",
                        text=f"Synthesized insight for '{question}'.",
                        source_ids=[sources[-1].source_id] if sources else ["src_001"],
                    )
                ]
            }

        critic = lambda state: {"critique": Critique(quality_score=0.95, should_research_again=False)}

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=dynamic_analyst,
            custom_critic=critic,
        )

        checkpointer = MemorySaver()
        graph = create_research_graph(
            custom_subgraph=subgraph,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "thread-e2e-expansion"}}

        # Session 1
        graph.invoke({"question": "What is Graph Neural Network architecture?"}, config)
        assert graph.get_state(config).next == ("human_review",)
        graph.invoke(Command(resume={"action": "approve"}), config)
        assert graph.get_state(config).next == ()

        # Session 2: Temporal mismatch triggers expansion
        follow_up = "What are 2026 advances in scalable GCN training?"
        graph.invoke({"follow_up_question": follow_up}, config)

        paused = graph.get_state(config)
        assert paused.next == ("human_review",)
        assert paused.values["reuse_decision"].decision == "RESEARCH_MORE"
        assert paused.values["question"] == follow_up

        # Human approves
        graph.invoke(Command(resume={"action": "approve"}), config)
        final = graph.get_state(config)
        assert final.next == ()
        assert final.values["human_review"].action == "approve"
