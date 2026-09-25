"""Graph package."""

from typing import TYPE_CHECKING
from verified_research.graph.router import route_after_critic
from verified_research.graph.state import ResearchState

if TYPE_CHECKING:
    from verified_research.graph.graph import (
        build_research_graph,
        create_research_graph,
    )

__all__ = [
    "ResearchState",
    "route_after_critic",
    "build_research_graph",
    "create_research_graph",
]


def __getattr__(name: str):
    if name in ("build_research_graph", "create_research_graph"):
        from verified_research.graph import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
