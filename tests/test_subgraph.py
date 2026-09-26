"""Tests for research/critique subgraph compilation, direct execution, and parent graph composition."""

from unittest.mock import MagicMock
import pytest
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.critic import create_critic_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.graph import build_research_graph, create_research_graph
from verified_research.graph.research_subgraph import (
    build_research_subgraph,
    create_research_subgraph,
)
from verified_research.models.research import Critique, Finding, Source


class SpySearchService:
    """Mock search service tracking query history."""

    def __init__(self) -> None:
        self.recorded_queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        self.recorded_queries.append(query)
        idx = len(self.recorded_queries)
        return [
            Source(
                source_id=f"src_{idx:03d}",
                title=f"Source on {query}",
                url=f"https://example.com/item/{idx}",
                content=f"Detailed evidence concerning {query}.",
            )
        ]


def create_mock_analyst():
    """Create mock analyst that grounds findings in all available sources."""

    def mock_analyst(state):
        sources = state.get("sources", [])
        findings = []
        for i, s in enumerate(sources, start=1):
            findings.append(
                Finding(
                    finding_id=f"finding_{i:03d}",
                    text=f"Finding grounded in {s.source_id}",
                    source_ids=[s.source_id],
                )
            )
        return {"findings": findings}

    return mock_analyst


class TestResearchSubgraphDirect:
    """Step 8: Direct unit & integration tests on the research subgraph."""

    def test_subgraph_compiles_successfully(self):
        builder = build_research_subgraph()
        subgraph = builder.compile()
        assert subgraph is not None

    def test_subgraph_node_structure(self):
        builder = build_research_subgraph()
        nodes = list(builder.nodes.keys())
        assert "researcher" in nodes
        assert "analyst" in nodes
        assert "critic" in nodes
        assert "research" not in nodes

    def test_subgraph_good_critique_terminates_single_pass(self):
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst()

        critic = lambda state: {
            "critique": Critique(
                quality_score=0.95,
                missing_topics=[],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=[],
                should_research_again=False,
            )
        }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        res = subgraph.invoke({"question": "What is quantum supremacy?"})

        assert res["question"] == "What is quantum supremacy?"
        assert res["research_iteration"] == 1
        assert len(res["sources"]) == 1
        assert len(res["findings"]) == 1
        assert res["critique"].should_research_again is False
        assert search_spy.recorded_queries == ["What is quantum supremacy?"]

    def test_subgraph_weak_then_good_loops_correctly(self):
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst()

        critic_calls = 0

        def mock_critic(state):
            nonlocal critic_calls
            critic_calls += 1
            if critic_calls == 1:
                return {
                    "critique": Critique(
                        quality_score=0.5,
                        missing_topics=["Noise mitigation"],
                        recommended_queries=["quantum noise mitigation surface codes"],
                        should_research_again=True,
                    )
                }
            return {
                "critique": Critique(
                    quality_score=0.9,
                    should_research_again=False,
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=mock_critic,
        )

        res = subgraph.invoke({"question": "What is quantum error correction?"})

        assert res["research_iteration"] == 2
        assert critic_calls == 2
        assert len(search_spy.recorded_queries) == 2
        assert search_spy.recorded_queries[0] == "What is quantum error correction?"
        assert search_spy.recorded_queries[1] == "quantum noise mitigation surface codes"
        assert res["critique"].should_research_again is False

    def test_subgraph_always_weak_terminates_at_3_iterations(self):
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst()

        critic_calls = 0

        def perpetual_weak_critic(state):
            nonlocal critic_calls
            critic_calls += 1
            return {
                "critique": Critique(
                    quality_score=0.3,
                    recommended_queries=[f"followup query {critic_calls}"],
                    should_research_again=True,
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=perpetual_weak_critic,
        )

        res = subgraph.invoke({"question": "Hard question with endless gaps"})

        assert res["research_iteration"] == 3
        assert critic_calls == 3
        assert len(search_spy.recorded_queries) == 3


class TestParentGraphComposition:
    """Step 9 & 10: Parent graph execution and encapsulation tests."""

    def test_parent_graph_encapsulation_structure(self):
        """Parent graph should contain 'research' as a single node, not individual worker nodes."""
        parent_builder = build_research_graph()
        parent_nodes = list(parent_builder.nodes.keys())

        # Parent must see research and verifier units
        assert "research" in parent_nodes
        assert "verifier" in parent_nodes
        assert len(parent_nodes) == 2

        # Parent must NOT register internal child nodes directly
        assert "researcher" not in parent_nodes
        assert "analyst" not in parent_nodes
        assert "critic" not in parent_nodes

    def test_parent_graph_executes_start_research_end(self):
        """Parent graph executes START -> research -> END and propagates full state."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst()

        critic = lambda state: {
            "critique": Critique(
                quality_score=0.92,
                should_research_again=False,
            )
        }

        # Build custom compiled subgraph
        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        # Inject into parent graph
        parent_graph = create_research_graph(custom_subgraph=subgraph)

        initial_state = {"question": "How do topological qubits function?"}
        final_state = parent_graph.invoke(initial_state)

        # Verify parent state reflects completed research
        assert final_state["question"] == initial_state["question"]
        assert final_state["research_iteration"] == 1
        assert "sources" in final_state
        assert "findings" in final_state
        assert "critique" in final_state
        assert len(final_state["sources"]) == 1
        assert len(final_state["findings"]) == 1
        assert final_state["findings"][0].source_ids == ["src_001"]

    def test_parent_graph_multi_cycle_propagation(self):
        """Parent graph transparently receives state after child completes multiple internal cycles."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst()

        critic_calls = 0

        def multi_critic(state):
            nonlocal critic_calls
            critic_calls += 1
            if critic_calls < 2:
                return {
                    "critique": Critique(
                        quality_score=0.5,
                        recommended_queries=["topological braiding algorithms"],
                        should_research_again=True,
                    )
                }
            return {
                "critique": Critique(
                    quality_score=0.94,
                    should_research_again=False,
                )
            }

        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=multi_critic,
        )

        parent_graph = create_research_graph(custom_subgraph=subgraph)
        final_state = parent_graph.invoke({"question": "Topological computing"})

        assert final_state["research_iteration"] == 2
        assert len(final_state["sources"]) == 2
        assert len(final_state["findings"]) == 2
        assert final_state["critique"].should_research_again is False
