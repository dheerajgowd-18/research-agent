"""Parent StateGraph orchestrating the research pipeline via a research subgraph."""

from typing import Callable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from verified_research.graph.state import ResearchState


def build_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> StateGraph:
    """Construct the parent StateGraph orchestrating the research process.

    Parent Graph Architecture:
        START
          ↓
       research (encapsulated research/critique subgraph)
          ↓
         END

    The parent graph treats the research loop as a single logical unit ('research')
    without direct exposure to Researcher, Analyst, Critic, or internal routing.

    Args:
        custom_subgraph: Optional pre-compiled research subgraph. If None, builds
                         using create_research_subgraph().
        custom_researcher: Optional researcher node passed to default subgraph builder.
        custom_analyst: Optional analyst node passed to default subgraph builder.
        custom_critic: Optional critic node passed to default subgraph builder.
        custom_router: Optional router function passed to default subgraph builder.

    Returns:
        Configured parent StateGraph instance ready for compilation.
    """
    if custom_subgraph is not None:
        subgraph = custom_subgraph
    else:
        from verified_research.graph.research_subgraph import create_research_subgraph

        subgraph = create_research_subgraph(
            custom_researcher=custom_researcher,
            custom_analyst=custom_analyst,
            custom_critic=custom_critic,
            custom_router=custom_router,
        )

    builder = StateGraph(ResearchState)

    builder.add_node("research", subgraph)

    builder.add_edge(START, "research")
    builder.add_edge("research", END)

    return builder


def create_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> CompiledStateGraph:
    """Construct and compile the parent research pipeline graph.

    Args:
        custom_subgraph: Optional pre-compiled research subgraph.
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.
        custom_critic: Optional critic node override.
        custom_router: Optional router function override.

    Returns:
        CompiledStateGraph executable via .invoke() or .stream().
    """
    builder = build_research_graph(
        custom_subgraph=custom_subgraph,
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
        custom_critic=custom_critic,
        custom_router=custom_router,
    )
    return builder.compile()
