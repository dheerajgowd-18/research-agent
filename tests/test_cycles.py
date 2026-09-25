"""Tests for cyclic graph execution, iteration bounding, and feedback propagation."""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.critic import create_critic_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.graph import create_research_graph
from verified_research.models.research import AnalystOutput, Critique, Finding, Source


class SpySearchService:
    """Mock search service that records all queries passed to it."""

    def __init__(self) -> None:
        self.recorded_queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        self.recorded_queries.append(query)
        idx = len(self.recorded_queries)
        return [
            Source(
                source_id=f"src_{idx:03d}",
                title=f"Source for {query}",
                url=f"https://example.com/item/{idx}",
                content=f"Substantive evidence regarding {query}.",
            )
        ]


def create_mock_analyst_node():
    """Create a mock analyst node that cites all available sources in state."""

    def mock_analyst(state):
        sources = state.get("sources", [])
        findings = []
        for i, s in enumerate(sources, start=1):
            findings.append(
                Finding(
                    finding_id=f"finding_{i:03d}",
                    text=f"Finding substantiated by {s.source_id}",
                    source_ids=[s.source_id],
                )
            )
        return {"findings": findings}

    return mock_analyst


class TestCyclicGraphExecution:
    """Comprehensive tests for Phase 2 cyclic routing, iteration bounds, and feedback."""

    def test_1_good_on_first_pass(self):
        """TEST 1: Critic says research is good on first pass.

        Execution: Researcher -> Analyst -> Critic -> END (1 iteration)
        """
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst_node()

        critic_call_count = 0

        def mock_critic(state):
            nonlocal critic_call_count
            critic_call_count += 1
            return {
                "critique": Critique(
                    quality_score=0.92,
                    missing_topics=[],
                    weak_findings=[],
                    citation_gaps=[],
                    recommended_queries=[],
                    should_research_again=False,
                )
            }

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=mock_critic,
        )

        final_state = graph.invoke({"question": "What is quantum computing?"})

        assert final_state["research_iteration"] == 1
        assert len(search_spy.recorded_queries) == 1
        assert search_spy.recorded_queries[0] == "What is quantum computing?"
        assert critic_call_count == 1
        assert final_state["critique"].should_research_again is False

    def test_2_weak_then_good(self):
        """TEST 2: First critic says weak, second says good.

        Execution: Researcher #1 -> Analyst #1 -> Critic #1
                   -> Researcher #2 -> Analyst #2 -> Critic #2 -> END
        Final iteration: 2
        """
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst_node()

        critic_call_count = 0

        def mock_critic(state):
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return {
                    "critique": Critique(
                        quality_score=0.45,
                        missing_topics=["Fault tolerance"],
                        weak_findings=[],
                        citation_gaps=[],
                        recommended_queries=["quantum fault tolerance thresholds"],
                        should_research_again=True,
                    )
                }
            else:
                return {
                    "critique": Critique(
                        quality_score=0.89,
                        missing_topics=[],
                        weak_findings=[],
                        citation_gaps=[],
                        recommended_queries=[],
                        should_research_again=False,
                    )
                }

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=mock_critic,
        )

        final_state = graph.invoke({"question": "What is quantum computing?"})

        assert final_state["research_iteration"] == 2
        assert critic_call_count == 2
        assert len(search_spy.recorded_queries) == 2
        # First query was initial question, second query was critic recommendation
        assert search_spy.recorded_queries[0] == "What is quantum computing?"
        assert search_spy.recorded_queries[1] == "quantum fault tolerance thresholds"
        assert final_state["critique"].should_research_again is False

    def test_3_always_weak_terminates_at_max_iterations(self):
        """TEST 3 (Mandatory): Critic always requests more research.

        Expected: Researcher executes exactly 3 times, NOT 4.
        Final iteration: 3.
        """
        search_spy = SpySearchService()
        researcher_call_count = 0
        analyst_call_count = 0
        critic_call_count = 0

        base_researcher = create_researcher_node(search_client=search_spy)

        def researcher_wrapper(state):
            nonlocal researcher_call_count
            researcher_call_count += 1
            return base_researcher(state)

        base_analyst = create_mock_analyst_node()

        def analyst_wrapper(state):
            nonlocal analyst_call_count
            analyst_call_count += 1
            return base_analyst(state)

        def mock_critic(state):
            nonlocal critic_call_count
            critic_call_count += 1
            return {
                "critique": Critique(
                    quality_score=0.35,
                    missing_topics=[f"Missing aspect {critic_call_count}"],
                    weak_findings=[],
                    citation_gaps=[],
                    recommended_queries=[f"query for round {critic_call_count}"],
                    should_research_again=True,
                )
            }

        graph = create_research_graph(
            custom_researcher=researcher_wrapper,
            custom_analyst=analyst_wrapper,
            custom_critic=mock_critic,
        )

        final_state = graph.invoke({"question": "Complex topic with endless gaps"})

        # Invariant checks:
        assert researcher_call_count == 3, f"Researcher called {researcher_call_count} times, expected exactly 3"
        assert analyst_call_count == 3, f"Analyst called {analyst_call_count} times, expected exactly 3"
        assert critic_call_count == 3, f"Critic called {critic_call_count} times, expected exactly 3"
        assert final_state["research_iteration"] == 3

    def test_5_feedback_propagation(self):
        """TEST 5: Verify that critic.recommended_queries are available to and used by next Researcher."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst_node()

        def mock_critic(state):
            iter_count = state.get("research_iteration", 0)
            if iter_count == 1:
                return {
                    "critique": Critique(
                        quality_score=0.5,
                        missing_topics=["Topic A"],
                        weak_findings=[],
                        citation_gaps=[],
                        recommended_queries=["targeted_followup_query_alpha"],
                        should_research_again=True,
                    )
                }
            return {
                "critique": Critique(
                    quality_score=0.9,
                    should_research_again=False,
                )
            }

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=mock_critic,
        )

        graph.invoke({"question": "Initial question"})

        assert len(search_spy.recorded_queries) == 2
        assert search_spy.recorded_queries[0] == "Initial question"
        assert search_spy.recorded_queries[1] == "targeted_followup_query_alpha"

    def test_6_no_infinite_loop(self):
        """TEST 6: Perpetual should_research_again=True cannot loop infinitely."""
        search_spy = SpySearchService()
        researcher = create_researcher_node(search_client=search_spy)
        analyst = create_mock_analyst_node()

        def eternal_critic(state):
            return {
                "critique": Critique(
                    quality_score=0.1,
                    recommended_queries=["query"],
                    should_research_again=True,
                )
            }

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=eternal_critic,
        )

        # invoke() must terminate without RecursionError or hanging
        final_state = graph.invoke({"question": "Unsolvable question"})
        assert final_state["research_iteration"] == 3

    def test_7_iteration_counter_progression(self):
        """TEST 7: Counter progression: 0 -> 1 -> 2 -> 3 (Never 4)."""
        observed_iterations: list[int] = []

        search_spy = SpySearchService()
        base_researcher = create_researcher_node(search_client=search_spy)

        def spy_researcher(state):
            res = base_researcher(state)
            observed_iterations.append(res["research_iteration"])
            return res

        analyst = create_mock_analyst_node()

        def mock_critic(state):
            return {
                "critique": Critique(
                    quality_score=0.4,
                    recommended_queries=["q"],
                    should_research_again=True,
                )
            }

        graph = create_research_graph(
            custom_researcher=spy_researcher,
            custom_analyst=analyst,
            custom_critic=mock_critic,
        )

        final_state = graph.invoke({"question": "Question", "research_iteration": 0})

        assert observed_iterations == [1, 2, 3]
        assert final_state["research_iteration"] == 3
        assert 4 not in observed_iterations
