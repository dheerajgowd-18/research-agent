"""LangGraph StateGraph construction with conditional routing and cyclic research."""

from typing import Callable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from verified_research.graph.router import route_after_critic
from verified_research.graph.state import ResearchState


def build_research_graph(
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> StateGraph:
    """Construct the StateGraph with cyclic feedback routing.

    Architecture:
        START
          ↓
        researcher
          ↓
        analyst
          ↓
        critic
          ↓
        route_after_critic
          ├── researcher (cycle when more research is needed and iteration < MAX_ITERATIONS)
          └── END (terminate when research is sufficient or iteration >= MAX_ITERATIONS)

    Args:
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.
        custom_critic: Optional critic node override.
        custom_router: Optional routing function override.

    Returns:
        Configured StateGraph instance ready for compilation.
    """
    if custom_researcher is None:
        from verified_research.agents.researcher import researcher_node

        active_researcher = researcher_node
    else:
        active_researcher = custom_researcher

    if custom_analyst is None:
        from verified_research.agents.analyst import analyst_node

        active_analyst = analyst_node
    else:
        active_analyst = custom_analyst

    if custom_critic is None:
        from verified_research.agents.critic import critic_node

        active_critic = critic_node
    else:
        active_critic = custom_critic

    active_router = custom_router or route_after_critic

    builder = StateGraph(ResearchState)

    builder.add_node("researcher", active_researcher)
    builder.add_node("analyst", active_analyst)
    builder.add_node("critic", active_critic)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "critic")

    builder.add_conditional_edges(
        "critic",
        active_router,
        {
            "researcher": "researcher",
            "end": END,
        },
    )

    return builder


def create_research_graph(
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> CompiledStateGraph:
    """Construct and compile the research pipeline graph with cyclic routing.

    Args:
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.
        custom_critic: Optional critic node override.
        custom_router: Optional routing function override.

    Returns:
        CompiledStateGraph executable via .invoke() or .stream().
    """
    builder = build_research_graph(
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
        custom_critic=custom_critic,
        custom_router=custom_router,
    )
    return builder.compile()
