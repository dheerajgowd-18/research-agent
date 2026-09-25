"""Researcher node implementation for Phase 1 research pipeline."""

import logging
from typing import Any, Callable
from verified_research.graph.state import ResearchState
from verified_research.models.research import Source
from verified_research.tools.search import SearchService, TavilySearchClient

logger = logging.getLogger(__name__)


def create_researcher_node(
    search_client: SearchService | None = None,
) -> Callable[[ResearchState], dict[str, list[Source]]]:
    """Factory to create a researcher node with an optional injected search client.

    Contract:
        Input: state['question'] (str)
        Output: {'sources': list[Source]}

    Args:
        search_client: Optional search service implementing SearchService protocol.
                       If None, instantiates TavilySearchClient.

    Returns:
        A callable node function conforming to LangGraph node specification.
    """
    client = search_client or TavilySearchClient()

    def researcher_node(state: ResearchState) -> dict[str, list[Source]]:
        """LangGraph node that performs web search and returns normalized sources.

        Contract:
            INPUT:
                question: str (required, non-empty)
            OUTPUT:
                sources: list[Source]
        """
        question = state.get("question")
        if not question or not question.strip():
            raise ValueError("Researcher node requires a non-empty 'question' in state.")

        clean_question = question.strip()
        logger.info("[RESEARCHER] received question: '%s'", clean_question)

        # Execute web search through abstraction
        sources = client.search(clean_question)

        logger.info("[RESEARCHER] retrieved %d sources", len(sources))
        return {"sources": sources}

    return researcher_node


# Default node instance using standard configuration
researcher_node = create_researcher_node()
