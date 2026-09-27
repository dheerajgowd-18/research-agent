"""Graph package."""

from typing import TYPE_CHECKING
from verified_research.graph.router import (
    route_after_critic,
    route_after_human_review,
    route_after_sufficiency,
    route_after_supervisor,
)
from verified_research.graph.state import ResearchState

if TYPE_CHECKING:
    from verified_research.graph.graph import (
        build_research_graph,
        build_supervisor_graph,
        create_research_graph,
        create_supervisor_graph,
    )
    from verified_research.graph.research_subgraph import (
        build_research_subgraph,
        create_research_subgraph,
    )

__all__ = [
    "ResearchState",
    "route_after_critic",
    "route_after_human_review",
    "route_after_sufficiency",
    "route_after_supervisor",
    "build_research_graph",
    "create_research_graph",
    "build_supervisor_graph",
    "create_supervisor_graph",
    "build_research_subgraph",
    "create_research_subgraph",
]


def __getattr__(name: str):
    if name in (
        "build_research_graph",
        "create_research_graph",
        "build_supervisor_graph",
        "create_supervisor_graph",
    ):
        from verified_research.graph import graph

        return getattr(graph, name)
    if name in ("build_research_subgraph", "create_research_subgraph"):
        from verified_research.graph import research_subgraph

        return getattr(research_subgraph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
