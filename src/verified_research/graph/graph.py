"""Parent StateGraph orchestrating research, claim verification, and human-in-the-loop review."""

from typing import Any, Callable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from verified_research.graph.state import ResearchState


def _wrap_research_subgraph(
    subgraph: CompiledStateGraph | Callable,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Wrap the research subgraph to manage human-initiated cycle counting at parent level."""

    def research_node(state: ResearchState) -> dict[str, Any]:
        result = (
            subgraph.invoke(state)
            if hasattr(subgraph, "invoke")
            else subgraph(state)
        )
        # If this pass was triggered by human selecting 'research_more', increment human_research_cycles
        review = state.get("human_review")
        if review and review.action == "research_more":
            current_cycles = state.get("human_research_cycles", 0)
            result["human_research_cycles"] = current_cycles + 1
        return result

    return research_node


def build_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
    custom_human_review: Callable | None = None,
    custom_human_router: Callable | None = None,
    custom_sufficiency_service: Any | None = None,
    custom_sufficiency_evaluator: Callable | None = None,
    custom_sufficiency_router: Callable | None = None,
    custom_reuse_analyst: Callable | None = None,
    use_supervisor: bool = False,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
) -> StateGraph:
    """Construct the parent StateGraph orchestrating research, verification, and human review.

    When use_supervisor=True, constructs the Supervisor orchestration graph:
        START -> supervisor -> [research | verifier | human_review | finish]
        workers -> supervisor
    When use_supervisor=False (default), constructs the sequential pipeline graph with sufficiency evaluation.
    """
    if use_supervisor:
        return build_supervisor_graph(
            custom_subgraph=custom_subgraph,
            custom_verifier=custom_verifier,
            custom_researcher=custom_researcher,
            custom_analyst=custom_analyst,
            custom_critic=custom_critic,
            custom_router=custom_router,
            custom_human_review=custom_human_review,
            custom_supervisor=custom_supervisor,
            custom_supervisor_policy=custom_supervisor_policy,
            max_supervisor_steps=max_supervisor_steps,
        )

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

    if custom_sufficiency_evaluator is not None:
        sufficiency_evaluator = custom_sufficiency_evaluator
    elif custom_sufficiency_service is not None:
        from verified_research.agents.sufficiency import create_evaluate_sufficiency_node

        sufficiency_evaluator = create_evaluate_sufficiency_node(
            sufficiency_service=custom_sufficiency_service
        )
    else:
        from verified_research.agents.sufficiency import evaluate_sufficiency_node

        sufficiency_evaluator = evaluate_sufficiency_node

    if custom_sufficiency_router is not None:
        sufficiency_router = custom_sufficiency_router
    else:
        from verified_research.graph.router import route_after_sufficiency

        sufficiency_router = route_after_sufficiency

    if custom_reuse_analyst is not None:
        reuse_analyst = custom_reuse_analyst
    else:
        from verified_research.agents.sufficiency import reuse_analyst_node

        reuse_analyst = reuse_analyst_node

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

    if custom_human_review is not None:
        human_review = custom_human_review
    else:
        from verified_research.agents.human_review import human_review_node

        human_review = human_review_node

    if custom_human_router is not None:
        human_router = custom_human_router
    else:
        from verified_research.graph.router import route_after_human_review

        human_router = route_after_human_review

    research_node = _wrap_research_subgraph(subgraph)

    builder = StateGraph(ResearchState)

    builder.add_node("evaluate_sufficiency", sufficiency_evaluator)
    builder.add_node("research", research_node)
    builder.add_node("reuse_synthesis", reuse_analyst)
    builder.add_node("verifier", verifier)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "evaluate_sufficiency")
    builder.add_conditional_edges(
        "evaluate_sufficiency",
        sufficiency_router,
        {
            "research": "research",
            "reuse_synthesis": "reuse_synthesis",
        },
    )
    builder.add_edge("research", "verifier")
    builder.add_edge("reuse_synthesis", "verifier")
    builder.add_edge("verifier", "human_review")
    builder.add_conditional_edges(
        "human_review",
        human_router,
        {
            "research": "research",
            "end": END,
        },
    )

    return builder


def build_supervisor_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
    custom_human_review: Callable | None = None,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
    max_human_research_cycles: int = 2,
) -> StateGraph:
    """Construct the Supervisor orchestration StateGraph.

    Architecture:
                        Supervisor
                       /    |     \\
                      ▼     ▼      ▼
                 Research Verify Human Review
                      │      │       │
                      └──────┴───────┘
                             │
                             ▼
                        Supervisor
                             │
                        finish/continue

    Workers:
        - research: Encapsulated Research Subgraph (researcher -> analyst -> critic)
        - verifier: Evidence-grounded claim-level verifier
        - human_review: Human-in-the-loop review with interrupt()

    Orchestrator:
        - supervisor: Evaluates state and decides next_worker ('research', 'verify', 'human_review', 'finish')
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

    if custom_human_review is not None:
        human_review = custom_human_review
    else:
        from verified_research.agents.human_review import human_review_node

        human_review = human_review_node

    if custom_supervisor is not None:
        supervisor = custom_supervisor
    else:
        from verified_research.agents.supervisor import create_supervisor_node

        supervisor = create_supervisor_node(
            policy=custom_supervisor_policy,
            max_steps=max_supervisor_steps,
            max_human_cycles=max_human_research_cycles,
        )

    from verified_research.graph.router import route_after_supervisor

    def supervisor_router(state: ResearchState) -> str:
        return route_after_supervisor(state, max_steps=max_supervisor_steps)

    research_worker = _wrap_research_subgraph(subgraph)

    builder = StateGraph(ResearchState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("research", research_worker)
    builder.add_node("verifier", verifier)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "research": "research",
            "verify": "verifier",
            "human_review": "human_review",
            "finish": END,
        },
    )

    # All workers transition back to the supervisor
    builder.add_edge("research", "supervisor")
    builder.add_edge("verifier", "supervisor")
    builder.add_edge("human_review", "supervisor")

    return builder


def create_supervisor_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
    custom_human_review: Callable | None = None,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
    max_human_research_cycles: int = 2,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Construct and compile the Supervisor orchestration graph."""
    builder = build_supervisor_graph(
        custom_subgraph=custom_subgraph,
        custom_verifier=custom_verifier,
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
        custom_critic=custom_critic,
        custom_router=custom_router,
        custom_human_review=custom_human_review,
        custom_supervisor=custom_supervisor,
        custom_supervisor_policy=custom_supervisor_policy,
        max_supervisor_steps=max_supervisor_steps,
        max_human_research_cycles=max_human_research_cycles,
    )
    return builder.compile(checkpointer=checkpointer)


def create_research_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
    custom_human_review: Callable | None = None,
    custom_human_router: Callable | None = None,
    custom_sufficiency_service: Any | None = None,
    custom_sufficiency_evaluator: Callable | None = None,
    custom_sufficiency_router: Callable | None = None,
    custom_reuse_analyst: Callable | None = None,
    use_supervisor: bool = False,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Construct and compile the parent research, verification, and HITL pipeline graph."""
    if use_supervisor:
        return create_supervisor_graph(
            custom_subgraph=custom_subgraph,
            custom_verifier=custom_verifier,
            custom_researcher=custom_researcher,
            custom_analyst=custom_analyst,
            custom_critic=custom_critic,
            custom_router=custom_router,
            custom_human_review=custom_human_review,
            custom_supervisor=custom_supervisor,
            custom_supervisor_policy=custom_supervisor_policy,
            max_supervisor_steps=max_supervisor_steps,
            checkpointer=checkpointer,
        )

    builder = build_research_graph(
        custom_subgraph=custom_subgraph,
        custom_verifier=custom_verifier,
        custom_researcher=custom_researcher,
        custom_analyst=custom_analyst,
        custom_critic=custom_critic,
        custom_router=custom_router,
        custom_human_review=custom_human_review,
        custom_human_router=custom_human_router,
        custom_sufficiency_service=custom_sufficiency_service,
        custom_sufficiency_evaluator=custom_sufficiency_evaluator,
        custom_sufficiency_router=custom_sufficiency_router,
        custom_reuse_analyst=custom_reuse_analyst,
    )
    return builder.compile(checkpointer=checkpointer)


