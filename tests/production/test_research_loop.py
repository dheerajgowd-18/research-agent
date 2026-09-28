"""Production Test Suite for Research Subgraph Loops and Iteration Bounds.

Verifies:
1. Single-pass success (critic approves on iteration 1).
2. Multi-pass feedback-driven refinement (critic asks for revision, satisfied on iteration 2).
3. Hard iteration cap enforcement (critic continuously requests revision, bounded at 3).
4. State accumulation across iterations (sources and evidence accumulate cleanly).
"""

import pytest
from langgraph.graph import END
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.research_subgraph import create_research_subgraph
from verified_research.models.research import (
    AnalystOutput,
    Claim,
    Critique,
    Evidence,
    Finding,
    Source,
)
from verified_research.graph.state import ResearchState
from tests.production.conftest import SpySearchService


def make_mock_analyst():
    """Mock analyst generating findings, evidence, and claims from current sources."""

    def analyst(state: ResearchState):
        sources = state.get("sources", [])
        iteration = state.get("research_iteration", 0)
        findings = []
        evidence = []
        claims = []

        for idx, src in enumerate(sources):
            fid = f"finding_{src.source_id}"
            eid = f"ev_{src.source_id}"
            cid = f"claim_{src.source_id}"

            findings.append(Finding(finding_id=fid, text=f"Finding from {src.title}", source_ids=[src.source_id]))
            evidence.append(Evidence(evidence_id=eid, source_id=src.source_id, text=src.content))
            claims.append(Claim(claim_id=cid, text=f"Claim from {src.title}", evidence_ids=[eid]))

        return {
            "findings": findings,
            "evidence": evidence,
            "claims": claims,
        }

    return analyst


class TestResearchSubgraphExecution:
    """Production test cases for iterative research cycles."""

    def test_single_pass_success(self):
        """Critic accepts initial findings; graph terminates cleanly at iteration 1."""
        spy_search = SpySearchService()
        researcher = create_researcher_node(search_client=spy_search)
        analyst = make_mock_analyst()

        critic_invocations = 0

        def critic(state: ResearchState):
            nonlocal critic_invocations
            critic_invocations += 1
            return {
                "critique": Critique(
                    quality_score=0.95,
                    should_research_again=False,
                    missing_topics=[],
                    weak_findings=[],
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        initial_state = {"question": "What is quantum error correction?"}
        final_state = subgraph.invoke(initial_state)

        assert final_state["research_iteration"] == 1
        assert critic_invocations == 1
        assert len(spy_search.recorded_queries) == 1
        assert final_state["critique"].should_research_again is False
        assert len(final_state["sources"]) >= 1
        assert len(final_state["claims"]) >= 1

    def test_multi_pass_feedback_driven_refinement(self):
        """Critic rejects iteration 1 with targeted query; approves iteration 2."""
        spy_search = SpySearchService()
        researcher = create_researcher_node(search_client=spy_search)
        analyst = make_mock_analyst()

        critic_invocations = 0

        def critic(state: ResearchState):
            nonlocal critic_invocations
            critic_invocations += 1
            if critic_invocations == 1:
                return {
                    "critique": Critique(
                        quality_score=0.45,
                        should_research_again=True,
                        missing_topics=["Surface code thresholds"],
                        recommended_queries=["surface code fault tolerance threshold"],
                    )
                }
            return {
                "critique": Critique(
                    quality_score=0.92,
                    should_research_again=False,
                    missing_topics=[],
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        initial_state = {"question": "What is quantum error correction?"}
        final_state = subgraph.invoke(initial_state)

        assert final_state["research_iteration"] == 2
        assert critic_invocations == 2
        # Verify second query came from critic's recommendation
        assert len(spy_search.recorded_queries) == 2
        assert "surface code" in spy_search.recorded_queries[1].lower()
        assert final_state["critique"].should_research_again is False
        # Sources accumulated across both passes
        assert len(final_state["sources"]) >= 2

    def test_hard_cap_iteration_limit(self):
        """Critic perpetually demands revision; execution terminates strictly at MAX_ITERATIONS (3)."""
        spy_search = SpySearchService()
        researcher = create_researcher_node(search_client=spy_search)
        analyst = make_mock_analyst()

        critic_invocations = 0

        def relentless_critic(state: ResearchState):
            nonlocal critic_invocations
            critic_invocations += 1
            return {
                "critique": Critique(
                    quality_score=0.30,
                    should_research_again=True,
                    missing_topics=["Deeper mathematical proofs needed"],
                    recommended_queries=[f"quantum proof iteration {critic_invocations}"],
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=relentless_critic,
        )

        initial_state = {"question": "What are quantum algorithms?"}
        final_state = subgraph.invoke(initial_state)

        assert final_state["research_iteration"] == 3
        assert critic_invocations == 3
        assert len(spy_search.recorded_queries) == 3
        # Even though critic wanted more, router forced termination
        assert final_state["critique"].should_research_again is True

    def test_state_accumulation_across_cycles(self):
        """Ensure sources and evidence correctly accumulate rather than overwrite across passes."""
        spy_search = SpySearchService()
        researcher = create_researcher_node(search_client=spy_search)
        analyst = make_mock_analyst()

        cycle = 0

        def multi_cycle_critic(state: ResearchState):
            nonlocal cycle
            cycle += 1
            if cycle == 1:
                return {
                    "critique": Critique(
                        quality_score=0.5,
                        should_research_again=True,
                        recommended_queries=["topological braiding phase 2"],
                    )
                }
            return {
                "critique": Critique(
                    quality_score=0.88,
                    should_research_again=False,
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=multi_cycle_critic,
        )

        final_state = subgraph.invoke({"question": "Explain topological quantum computing."})

        # Sources from pass 1 and pass 2 are both present
        source_ids = [s.source_id for s in final_state["sources"]]
        assert len(source_ids) == len(set(source_ids)), "Duplicate source IDs detected"
        assert len(source_ids) >= 2
