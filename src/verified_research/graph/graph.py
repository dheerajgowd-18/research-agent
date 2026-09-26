"""Parent StateGraph orchestrating research and evidence-grounded verification."""

from typing import Callable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from verified_research.graph.state import ResearchState


def build_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> StateGraph:
    """Construct the parent StateGraph orchestrating research and claim verification.

    Parent Graph Architecture:
        START
          ↓
       research (encapsulated research/critique subgraph)
          ↓
       verifier (evidence-grounded claim verifier node)
          ↓
         END

    The parent graph treats the iterative research loop as a single logical unit ('research')
    followed by the evidence-grounded claim verifier ('verifier').

    Args:
        custom_subgraph: Optional pre-compiled research subgraph. If None, builds
                         using create_research_subgraph().
        custom_verifier: Optional verifier node override.
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

    if custom_verifier is not None:
        import inspect
        from verified_research.agents.verifier import VerifierService, create_verifier_node

        if isinstance(custom_verifier, VerifierService):
            verifier = create_verifier_node(verifier_service=custom_verifier)
        elif callable(custom_verifier):
            sig = inspect.signature(custom_verifier)
            if len(sig.parameters) == 2:
                verifier = create_verifier_node(custom_verifier=custom_verifier)
            else:
                verifier = custom_verifier
        else:
            verifier = custom_verifier
    else:
        from verified_research.agents.verifier import verifier_node

        verifier = verifier_node

    builder = StateGraph(ResearchState)

    builder.add_node("research", subgraph)
    builder.add_node("verifier", verifier)

    builder.add_edge(START, "research")
    builder.add_edge("research", "verifier")
    builder.add_edge("verifier", END)

    return builder


def create_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
) -> CompiledStateGraph:
    """Construct and compile the parent research and verification pipeline graph.

    Args:
        custom_subgraph: Optional pre-compiled research subgraph.
        custom_verifier: Optional verifier node override.
        custom_researcher: Optional researcher node override.
        custom_analyst: Optional analyst node override.
        custom_critic: Optional critic node override.
        custom_router: Optional router function override.

    Returns:
        CompiledStateGraph executable via .invoke() or .stream().
    """
    builder = build_research_graph(
        custom_subgraph=custom_subgraph,
        custom_verifier=custom_verifier,
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
        custom_critic=custom_critic,
        custom_router=custom_router,
    )
    return builder.compile()
