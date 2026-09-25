"""Deterministic routing logic following critic evaluation."""

import logging
from typing import Literal
from verified_research.config.settings import DEFAULT_MAX_ITERATIONS
from verified_research.graph.state import ResearchState

logger = logging.getLogger(__name__)


def route_after_critic(
    state: ResearchState,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Literal["researcher", "end"]:
    """Determine whether to loop back to the researcher or terminate the graph.

    Deterministic routing rules:
        1. If critique is missing or indicates research is sufficient:
           -> 'end'
        2. If research is insufficient AND iteration >= MAX_ITERATIONS:
           -> 'end' (hard bound preventing runaway cycles)
        3. If research is insufficient AND iteration < MAX_ITERATIONS:
           -> 'researcher' (re-enter research pass with critic queries)

    Args:
        state: Current graph state containing 'critique' and 'research_iteration'.
        max_iterations: Iteration limit cutoff (defaults to DEFAULT_MAX_ITERATIONS = 3).

    Returns:
        'researcher' to re-enter the research loop, or 'end' to conclude.
    """
    critique = state.get("critique")
    iteration = state.get("research_iteration", 0)

    if critique is None or not critique.should_research_again:
        logger.info("[Router] next=end")
        return "end"

    if iteration >= max_iterations:
        logger.info("[Router] max_iterations reached")
        logger.info("[Router] next=end")
        return "end"

    logger.info("[Router] next=researcher")
    return "researcher"
