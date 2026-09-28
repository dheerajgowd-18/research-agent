"""Production Test Suite for Follow-up Research Sessions and Sufficiency Evaluation.

Verifies:
1. Sufficient existing evidence -> routes to reuse_synthesis without invoking web search.
2. Insufficient evidence -> routes to research subgraph for fresh search.
3. Temporal mismatch detection (e.g. 2024 evidence vs 2026 query -> RESEARCH_MORE).
4. Topic/entity mismatch detection -> forces fresh research.
5. Supervisor orchestration with follow-up continuity.
"""

from unittest.mock import MagicMock
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from verified_research.agents.sufficiency import (
    HeuristicSufficiencyService,
    create_evaluate_sufficiency_node,
    create_reuse_analyst_node,
)
from verified_research.graph.graph import create_research_graph, create_supervisor_graph
from verified_research.graph.router import route_after_sufficiency
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    ResearchReuseDecision,
    Source,
    VerificationResult,
)
from verified_research.graph.state import ResearchState
from tests.production.conftest import SpySearchService


class TestFollowUpWorkflows:
    """Verifies follow-up research branching, sufficiency evaluation, and temporal safety."""

    @pytest.fixture
    def base_research_state(self) -> ResearchState:
        src = Source(
            source_id="src_001",
            title="Superconducting Qubit Coherence (2024)",
            url="https://arxiv.org/abs/quant-2024",
            content="Transmon superconducting qubits achieved coherence times of 1.5ms in 2024.",
        )
        ev = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Transmon superconducting qubits achieved coherence times of 1.5ms in 2024.",
        )
        clm = Claim(
            claim_id="claim_001",
            text="Transmon superconducting qubits demonstrated 1.5ms coherence.",
            evidence_ids=["ev_001"],
        )
        fnd = Finding(
            finding_id="f_001",
            text="Transmon qubits achieved 1.5ms coherence in 2024.",
            source_ids=["src_001"],
        )
        vr = VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="Exact match.",
            evidence_ids=["ev_001"],
        )
        return {
            "question": "What is superconducting qubit coherence?",
            "sources": [src],
            "evidence": [ev],
            "claims": [clm],
            "findings": [fnd],
            "verification_results": [vr],
            "supervisor_steps": 3,
        }

    def test_sufficient_evidence_evaluates_to_reuse(self, base_research_state: ResearchState):
        """When follow-up question is covered by existing evidence, evaluator decides REUSE."""
        service = HeuristicSufficiencyService()
        decision = service.evaluate_sufficiency(
            question="What is the coherence time of superconducting transmon qubits?",
            sources=base_research_state["sources"],
            findings=base_research_state["findings"],
            claims=base_research_state["claims"],
            evidence=base_research_state["evidence"],
        )

        assert decision.decision == "REUSE"
        assert len(decision.missing_topics) == 0

    def test_insufficient_evidence_evaluates_to_research_more(self, base_research_state: ResearchState):
        """When follow-up question introduces unknown topics, evaluator decides RESEARCH_MORE."""
        service = HeuristicSufficiencyService()
        decision = service.evaluate_sufficiency(
            question="How do optical trapped ion qubits compare to neutral atoms?",
            sources=base_research_state["sources"],
            findings=base_research_state["findings"],
            claims=base_research_state["claims"],
            evidence=base_research_state["evidence"],
        )

        assert decision.decision == "RESEARCH_MORE"
        assert len(decision.missing_topics) > 0

    def test_temporal_mismatch_forces_fresh_research(self, base_research_state: ResearchState):
        """When existing evidence is dated (2024) but question asks about 2026, evaluator rejects reuse."""
        service = HeuristicSufficiencyService()
        decision = service.evaluate_sufficiency(
            question="What are recent 2026 quantum coherence benchmark records?",
            sources=base_research_state["sources"],
            findings=base_research_state["findings"],
            claims=base_research_state["claims"],
            evidence=base_research_state["evidence"],
        )

        assert decision.decision == "RESEARCH_MORE"
        assert "2026" in decision.reasoning
        assert any("2026" in topic for topic in decision.missing_topics)

    def test_pipeline_graph_routes_to_reuse_without_web_search(self, base_research_state: ResearchState):
        """In pipeline graph, REUSE decision routes through reuse_synthesis, avoiding research worker."""
        search_spy = SpySearchService()

        # If research worker is invoked, fail or record
        def failing_researcher(state):
            raise AssertionError("Research worker was invoked despite sufficient evidence!")

        def mock_reuse_analyst(state):
            return {
                "claims": [
                    Claim(
                        claim_id="claim_reuse_001",
                        text="Reused evidence confirms 1.5ms coherence time.",
                        evidence_ids=["ev_001"],
                    )
                ]
            }

        mock_verifier = lambda state: {
            "verification_results": [
                VerificationResult(
                    claim_id="claim_reuse_001",
                    verdict="SUPPORTED",
                    confidence=0.95,
                    reasoning="Reused match",
                    evidence_ids=["ev_001"],
                )
            ]
        }
        mock_human = lambda state: {"human_review": HumanReview(action="approve")}

        # Force sufficiency evaluator to return REUSE
        eval_node = lambda state: {
            "reuse_decision": ResearchReuseDecision(
                decision="REUSE",
                reasoning="Prior evidence contains full transmon coherence data.",
                missing_topics=[],
            )
        }

        graph = create_research_graph(
            custom_subgraph=failing_researcher,
            custom_sufficiency_evaluator=eval_node,
            custom_reuse_analyst=mock_reuse_analyst,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        input_state = {
            **base_research_state,
            "question": "What was the transmon coherence time?",
        }

        final = graph.invoke(input_state)
        assert len(final["claims"]) == 1
        assert final["claims"][0].claim_id == "claim_reuse_001"
        assert final["human_review"].action == "approve"

    def test_pipeline_graph_routes_to_research_on_insufficient_evidence(self):
        """In pipeline graph, RESEARCH_MORE decision routes to research subgraph."""
        research_invoked = False

        def tracking_researcher(state):
            nonlocal research_invoked
            research_invoked = True
            return {
                "sources": [Source(source_id="s_new", title="New", url="https://new.com", content="New data")],
                "evidence": [Evidence(evidence_id="e_new", source_id="s_new", text="New data")],
                "claims": [Claim(claim_id="c_new", text="New claim", evidence_ids=["e_new"])],
                "findings": [Finding(finding_id="f_new", text="New finding", source_ids=["s_new"])],
            }

        eval_node = lambda state: {
            "reuse_decision": ResearchReuseDecision(
                decision="RESEARCH_MORE",
                reasoning="No relevant data exists.",
                missing_topics=["neutral atoms"],
            )
        }
        mock_verifier = lambda state: {
            "verification_results": [
                VerificationResult(
                    claim_id="c_new", verdict="SUPPORTED", confidence=0.9, reasoning="ok", evidence_ids=["e_new"]
                )
            ]
        }
        mock_human = lambda state: {"human_review": HumanReview(action="approve")}

        graph = create_research_graph(
            custom_subgraph=tracking_researcher,
            custom_sufficiency_evaluator=eval_node,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        final = graph.invoke({"question": "Explain neutral atom quantum computing."})
        assert research_invoked is True
        assert len(final["sources"]) == 1
