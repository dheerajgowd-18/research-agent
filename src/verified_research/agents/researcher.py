"""Researcher node implementation with iteration tracking and critic feedback integration."""

import logging
from typing import Callable
from verified_research.config.settings import DEFAULT_MAX_ITERATIONS, get_settings
from verified_research.graph.state import ResearchState
from verified_research.models.research import Source
from verified_research.tools.search import SearchService, TavilySearchClient

logger = logging.getLogger(__name__)


def create_researcher_node(
    search_client: SearchService | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Callable[[ResearchState], dict[str, list[Source] | int]]:
    """Factory to create a researcher node with optional injected search client and iteration limit.

    Contract:
        Input: state['question'] (str), state.get('critique'), state.get('research_iteration', 0)
        Output: {'sources': list[Source], 'research_iteration': int}

    Args:
        search_client: Optional search service implementing SearchService protocol.
                       If None, instantiates TavilySearchClient.
        max_iterations: Maximum allowed research iterations before hard cutoff.

    Returns:
        A callable node function conforming to LangGraph node specification.
    """
    client = search_client or TavilySearchClient()

    def researcher_node(state: ResearchState) -> dict[str, list[Source] | int]:
        """LangGraph node that performs web search, incorporates critic feedback, and increments iteration.

        Contract:
            INPUT:
                question: str (required, non-empty)
                sources: list[Source] (optional, default [])
                critique: Critique (optional)
                research_iteration: int (optional, default 0)
            OUTPUT:
                sources: list[Source]
                research_iteration: int
        """
        question = state.get("question")
        if not question or not question.strip():
            raise ValueError("Researcher node requires a non-empty 'question' in state.")

        clean_question = question.strip()
        current_iteration = state.get("research_iteration", 0) + 1

        if current_iteration > max_iterations:
            raise RuntimeError(
                f"Researcher invocation exceeded maximum iteration limit of {max_iterations}."
            )

        logger.info("[Researcher] iteration=%d", current_iteration)

        existing_sources: list[Source] = list(state.get("sources", []))
        existing_urls = {s.url for s in existing_sources}

        critique = state.get("critique")
        human_feedback = state.get("human_feedback")
        queries_to_search: list[str] = []

        if human_feedback and human_feedback.strip() and current_iteration == 1:
            # Targeted search based on human feedback from review
            logger.info("[RESEARCHER] executing query from human feedback: '%s'", human_feedback.strip())
            queries_to_search.append(human_feedback.strip())
        elif current_iteration == 1 or critique is None or not critique.recommended_queries:
            # First pass or no specific critic guidance: search original question
            queries_to_search.append(clean_question)
        else:
            # Subsequent pass: execute targeted searches guided by critic feedback
            # Bound search queries to top 2 to avoid unbounded calls
            queries_to_search = critique.recommended_queries[:2]
            logger.info(
                "[RESEARCHER] executing targeted queries from critic: %s",
                queries_to_search,
            )

        newly_retrieved: list[Source] = []
        for query in queries_to_search:
            batch = client.search(query)
            for s in batch:
                if s.url not in existing_urls:
                    new_id = f"src_{len(existing_sources) + len(newly_retrieved) + 1:03d}"
                    normalized_source = Source(
                        source_id=new_id,
                        title=s.title,
                        url=s.url,
                        content=s.content,
                    )
                    newly_retrieved.append(normalized_source)
                    existing_urls.add(s.url)

        combined_sources = existing_sources + newly_retrieved
        logger.info(
            "[RESEARCHER] retrieved %d new sources (total accumulated: %d)",
            len(newly_retrieved),
            len(combined_sources),
        )

        return {
            "sources": combined_sources,
            "research_iteration": current_iteration,
        }

    return researcher_node


# Default node instance using standard configuration
researcher_node = create_researcher_node()
