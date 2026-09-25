"""LangGraph StateGraph construction for Phase 1 research pipeline."""

from typing import Callable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from verified_research.graph.state import ResearchState


def build_research_graph(
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
) -> StateGraph:
    """Construct the uncompiled StateGraph for Phase 1.

    Architecture:
        START -> researcher -> analyst -> END

    Args:
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.

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

    builder = StateGraph(ResearchState)

    builder.add_node("researcher", active_researcher)
    builder.add_node("analyst", active_analyst)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", END)

    return builder


def create_research_graph(
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
) -> CompiledStateGraph:
    """Construct and compile the Phase 1 research pipeline graph.

    Args:
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.

    Returns:
        CompiledStateGraph executable via .invoke() or .stream().
    """
    builder = build_research_graph(
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
    )
    return builder.compile()
