"""Unit tests for deterministic router logic."""

import pytest
from verified_research.graph.router import route_after_critic
from verified_research.models.research import Critique


class TestRouteAfterCritic:
    """Tests for route_after_critic covering all deterministic branching conditions."""

    def test_good_critique_routes_to_end(self):
        """When critique indicates research is sufficient, route to 'end' regardless of iteration."""
        critique = Critique(
            quality_score=0.92,
            missing_topics=[],
            weak_findings=[],
            citation_gaps=[],
            recommended_queries=[],
            should_research_again=False,
        )

        # Iteration 1, 2, or 3
        for iter_count in (1, 2, 3):
            state = {
                "question": "What is quantum computing?",
                "critique": critique,
                "research_iteration": iter_count,
            }
            assert route_after_critic(state) == "end"

    def test_missing_critique_routes_to_end(self):
        """If critique is absent from state, route safely to 'end'."""
        state = {
            "question": "What is quantum computing?",
            "research_iteration": 1,
        }
        assert route_after_critic(state) == "end"

    def test_weak_critique_iteration_1_routes_to_researcher(self):
        """Weak critique on iteration 1 must loop back to 'researcher'."""
        critique = Critique(
            quality_score=0.45,
            missing_topics=["Topological qubits"],
            weak_findings=[],
            citation_gaps=[],
            recommended_queries=["topological qubits Majorana"],
            should_research_again=True,
        )
        state = {
            "question": "What is quantum computing?",
            "critique": critique,
            "research_iteration": 1,
        }
        assert route_after_critic(state) == "researcher"

    def test_weak_critique_iteration_2_routes_to_researcher(self):
        """Weak critique on iteration 2 must loop back to 'researcher'."""
        critique = Critique(
            quality_score=0.60,
            missing_topics=["Qubit connectivity"],
            weak_findings=[],
            citation_gaps=[],
            recommended_queries=["qubit connectivity architectures"],
            should_research_again=True,
        )
        state = {
            "question": "What is quantum computing?",
            "critique": critique,
            "research_iteration": 2,
        }
        assert route_after_critic(state) == "researcher"

    def test_weak_critique_iteration_3_routes_to_end(self):
        """Weak critique on iteration 3 (MAX_ITERATIONS) must terminate at 'end'."""
        critique = Critique(
            quality_score=0.65,
            missing_topics=["Still missing topic"],
            weak_findings=[],
            citation_gaps=[],
            recommended_queries=["more queries"],
            should_research_again=True,
        )
        state = {
            "question": "What is quantum computing?",
            "critique": critique,
            "research_iteration": 3,
        }
        assert route_after_critic(state) == "end"

    def test_weak_critique_iteration_above_max_routes_to_end(self):
        """If iteration somehow exceeds MAX_ITERATIONS, router must strictly terminate at 'end'."""
        critique = Critique(
            quality_score=0.30,
            should_research_again=True,
        )
        state = {
            "question": "Any question",
            "critique": critique,
            "research_iteration": 4,
        }
        assert route_after_critic(state) == "end"
