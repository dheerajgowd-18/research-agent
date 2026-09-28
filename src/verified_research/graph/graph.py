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
        try:
            from verified_research.api.events import get_execution_event_emitter
            emitter = get_execution_event_emitter()
        except Exception:
            emitter = None

        if hasattr(subgraph, "stream") and emitter is not None:
            emitter("node_started", "researcher", {"message": "Searching web and gathering sources..."})
            accumulated: dict[str, Any] = dict(state)
            for sub_chunk in subgraph.stream(state, stream_mode="updates"):
                if not isinstance(sub_chunk, dict):
                    continue
                if "researcher" in sub_chunk:
                    r_data = sub_chunk["researcher"]
                    accumulated.update(r_data)
                    sources = r_data.get("sources", [])
                    emitter("research_update", "researcher", {
                        "sources_count": len(accumulated.get("sources", [])),
                        "new_sources_count": len(sources),
                        "sources": [s.model_dump() if hasattr(s, "model_dump") else s for s in sources],
                    })
                    for s in sources:
                        emitter("source_found", "researcher", s.model_dump() if hasattr(s, "model_dump") else s)
                    emitter("node_completed", "researcher", {"sources_count": len(accumulated.get("sources", []))})
                    emitter("node_started", "analyst", {"message": "Synthesizing findings and distilling atomic claims..."})
                elif "analyst" in sub_chunk:
                    a_data = sub_chunk["analyst"]
                    accumulated.update(a_data)
                    claims = a_data.get("claims", [])
                    findings = a_data.get("findings", [])
                    emitter("analysis_update", "analyst", {
                        "claims_count": len(accumulated.get("claims", [])),
                        "findings_count": len(accumulated.get("findings", [])),
                        "claims": [c.model_dump() if hasattr(c, "model_dump") else c for c in claims],
                        "findings": [f.model_dump() if hasattr(f, "model_dump") else f for f in findings],
                    })
                    emitter("node_completed", "analyst", {"claims_count": len(claims)})
                    emitter("node_started", "critic", {"message": "Evaluating research quality and citation coverage..."})
                elif "critic" in sub_chunk:
                    c_data = sub_chunk["critic"]
                    accumulated.update(c_data)
                    critique = c_data.get("critique")
                    critique_dict = critique.model_dump() if hasattr(critique, "model_dump") else (critique or {})
                    emitter("critic_update", "critic", {"critique": critique_dict})
                    emitter("node_completed", "critic", {"quality_score": getattr(critique, "quality_score", None)})
                    if getattr(critique, "should_research_again", False):
                        emitter("node_started", "researcher", {"message": "Executing iterative research loop..."})
            result = accumulated
        else:
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
            result["human_review"] = None
            result["verification_results"] = []
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
    custom_writer: Callable | None = None,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
    max_human_research_cycles: int = 2,
    enable_writer: bool = False,
) -> StateGraph:
    """Construct the Supervisor orchestration StateGraph.

    Architecture:
                        Supervisor
                       /    |     \    \
                      ▼     ▼      ▼    ▼
                 Research Verify Human Writer
                      │      │       │    │
                      └──────┴───────┴────┘
                             │
                             ▼
                        Supervisor
                             │
                        finish/continue

    Workers:
        - research: Encapsulated Research Subgraph (researcher -> analyst -> critic)
        - verifier: Evidence-grounded claim-level verifier
        - human_review: Human-in-the-loop review with interrupt()
        - writer: Synthesizes evidence-grounded final research report
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

    writer_active = enable_writer or custom_writer is not None
    if custom_writer is not None:
        writer_worker = custom_writer
    elif enable_writer:
        from verified_research.agents.writer import create_writer_node

        writer_worker = create_writer_node()
    else:
        writer_worker = None

    if custom_supervisor is not None:
        supervisor = custom_supervisor
    else:
        from verified_research.agents.supervisor import create_supervisor_node

        supervisor = create_supervisor_node(
            policy=custom_supervisor_policy,
            max_steps=max_supervisor_steps,
            max_human_cycles=max_human_research_cycles,
            enable_writer=writer_active,
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
    if writer_worker is not None:
        builder.add_node("writer", writer_worker)

    conditional_targets = {
        "research": "research",
        "verify": "verifier",
        "human_review": "human_review",
        "finish": END,
    }
    if writer_worker is not None:
        conditional_targets["writer"] = "writer"

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        conditional_targets,
    )

    # All workers transition back to the supervisor
    builder.add_edge("research", "supervisor")
    builder.add_edge("verifier", "supervisor")
    builder.add_edge("human_review", "supervisor")
    if writer_worker is not None:
        builder.add_edge("writer", "supervisor")

    return builder


def create_supervisor_graph(
    custom_subgraph: CompiledStateGraph | Callable | None = None,
    custom_verifier: Callable | None = None,
    custom_researcher: Callable | None = None,
    custom_analyst: Callable | None = None,
    custom_critic: Callable | None = None,
    custom_router: Callable | None = None,
    custom_human_review: Callable | None = None,
    custom_writer: Callable | None = None,
    custom_supervisor: Callable | None = None,
    custom_supervisor_policy: Any | None = None,
    max_supervisor_steps: int = 8,
    max_human_research_cycles: int = 2,
    enable_writer: bool = False,
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
        custom_writer=custom_writer,
        custom_supervisor=custom_supervisor,
        custom_supervisor_policy=custom_supervisor_policy,
        max_supervisor_steps=max_supervisor_steps,
        max_human_research_cycles=max_human_research_cycles,
        enable_writer=enable_writer,
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


