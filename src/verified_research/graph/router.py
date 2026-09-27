"""Deterministic routing logic following critic evaluation."""

import logging
from typing import Literal
from verified_research.config.settings import (
    DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
    DEFAULT_MAX_ITERATIONS,
)
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


def route_after_human_review(
    state: ResearchState,
    max_human_research_cycles: int = DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
) -> Literal["research", "end"]:
    """Determine whether to re-enter the research subgraph or conclude execution.

    Deterministic routing rules:
        1. approve       -> 'end' (approved research concludes workflow)
        2. edit          -> 'end' (edited claims accepted, workflow concludes)
        3. reject        -> 'end' (rejected research terminates workflow)
        4. research_more ->
           - if human_research_cycles < max_human_research_cycles:
             -> 'research' (re-enter research subgraph for additional pass)
           - otherwise:
             -> 'end' (hard bound preventing runaway human research loops)

    Args:
        state: Current graph state containing 'human_review' and 'human_research_cycles'.
        max_human_research_cycles: Iteration limit cutoff (defaults to DEFAULT_MAX_HUMAN_RESEARCH_CYCLES = 2).

    Returns:
        'research' to re-enter research subgraph, or 'end' to conclude.
    """
    review = state.get("human_review")
    if review is None:
        logger.info("[Router:Human] No human_review found in state; next=end")
        return "end"

    action = review.action
    logger.info("[Router:Human] Evaluating human review action: '%s'", action)

    if action in ("approve", "edit", "reject"):
        logger.info("[Router:Human] Action '%s' terminates workflow. next=end", action)
        return "end"

    if action == "research_more":
        cycles = state.get("human_research_cycles", 0)
        if cycles < max_human_research_cycles:
            logger.info(
                "[Router:Human] Research cycle permitted (%d < %d). next=research",
                cycles,
                max_human_research_cycles,
            )
            return "research"
        else:
            logger.warning(
                "[Router:Human] Maximum human research cycles reached (%d >= %d). next=end",
                cycles,
                max_human_research_cycles,
            )
            return "end"

    # Fallback safe termination
    return "end"

