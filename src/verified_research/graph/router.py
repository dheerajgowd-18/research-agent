"""Deterministic routing logic following critic evaluation."""

import logging
from typing import Literal
from verified_research.config.settings import (
    DEFAULT_MAX_HUMAN_RESEARCH_CYCLES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_SUPERVISOR_STEPS,
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


def route_after_sufficiency(
    state: ResearchState,
) -> Literal["research", "reuse_synthesis"]:
    """Determine whether to re-enter research or synthesize from existing evidence.

    Deterministic routing rules:
        1. If reuse_decision exists and decision == 'REUSE':
           -> 'reuse_synthesis' (synthesize follow-up claims from existing evidence)
        2. Otherwise (decision == 'RESEARCH_MORE' or missing):
           -> 'research' (enter research subgraph for additional evidence)

    Args:
        state: Current graph state containing 'reuse_decision'.

    Returns:
        'reuse_synthesis' to reuse evidence, or 'research' to run research subgraph.
    """
    decision = state.get("reuse_decision")
    if decision is not None and decision.decision == "REUSE":
        logger.info(
            "[Router:Sufficiency] Decision is REUSE (reasoning: %s). next=reuse_synthesis",
            decision.reasoning,
        )
        return "reuse_synthesis"

    reasoning = decision.reasoning if decision else "No reuse decision present"
    logger.info(
        "[Router:Sufficiency] Decision is RESEARCH_MORE (reasoning: %s). next=research",
        reasoning,
    )
    return "research"


def route_after_supervisor(
    state: ResearchState,
    max_steps: int = DEFAULT_MAX_SUPERVISOR_STEPS,
) -> Literal["research", "verify", "human_review", "finish"]:
    """Determine which specialized worker node should execute next based on supervisor decision.

    Safety Rules:
        1. If supervisor_steps >= max_steps:
           -> 'finish' (hard limit preventing runaway supervisor cycles)
        2. If supervisor_decision is missing or next_worker is not in allowed workers:
           -> 'finish' (safe fallback preventing unvalidated routing)
        3. Otherwise:
           -> decision.next_worker ('research', 'verify', 'human_review', or 'finish')

    Args:
        state: Current graph state containing 'supervisor_steps' and 'supervisor_decision'.
        max_steps: Hard step boundary (defaults to DEFAULT_MAX_SUPERVISOR_STEPS = 8).

    Returns:
        One of 'research', 'verify', 'human_review', or 'finish'.
    """
    decision = state.get("supervisor_decision")
    if decision is None:
        logger.warning("[Router:Supervisor] No supervisor_decision in state. next=finish")
        return "finish"

    if decision.next_worker == "finish":
        logger.info("[Router:Supervisor] Supervisor decided to finish. next=finish")
        return "finish"

    steps = state.get("supervisor_steps", 0)
    if steps > max_steps:
        logger.warning(
            "[Router:Supervisor] Supervisor steps exceeded limit (%d > %d). next=finish",
            steps,
            max_steps,
        )
        return "finish"

    worker = decision.next_worker
    if worker not in ("research", "verify", "human_review", "finish"):
        logger.error(
            "[Router:Supervisor] Unrecognized worker '%s' in supervisor decision. next=finish",
            worker,
        )
        return "finish"

    logger.info("[Router:Supervisor] Routing to worker: '%s'", worker)
    return worker

