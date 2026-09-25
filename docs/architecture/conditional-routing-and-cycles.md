# Architecture Design: Conditional Routing & Cycles

This document details the architectural rationale, routing mechanics, iteration bounding, and state propagation for the cyclic research loop in the **Verified Research Agent**.

---

## 1. Graph Architecture Diagram

```
       [START]
          │
          ▼
   ┌──►[Researcher]
   │      │
   │      ▼
   │   [Analyst]
   │      │
   │      ▼
   │   [Critic]
   │      │
   │      ▼
   │   [route_after_critic]
   │      │
   └──weak├── (iteration < MAX_ITERATIONS)
          │
      good└──► [END] (or iteration >= MAX_ITERATIONS)
```

---

## 2. Why the Initial Architecture Was Linear

The initial pipeline (`START -> researcher -> analyst -> END`) was intentionally designed as a linear directed acyclic graph (DAG) to establish:
- Clean typed state schemas (`ResearchState`).
- Strict node contracts (input/output boundaries).
- Fundamental provenance traceability (`Finding -> source_ids -> Source`).
- Basic service abstractions and dependency injection patterns.

A linear pipeline lacks any mechanism to evaluate whether the retrieved evidence was sufficient, relevant, or complete. It accepts whatever the initial search yields, regardless of quality or coverage gaps.

---

## 3. Why a Cycle is Introduced

Real-world research is inherently iterative:
1. An initial broad query reveals preliminary facts but uncovers unexpected nuances, missing perspectives, or citation gaps.
2. A critical evaluation step identifies what remains unanswered.
3. Follow-up research queries must be targeted directly at those specific gaps rather than repeating the broad original query.
4. Synthesis must be updated with the newly accumulated evidence.

Introducing a cyclic feedback edge transforms the agent from a static single-pass pipe into an adaptive, self-refining research system.

---

## 4. What Conditional Edges Are

In standard LangGraph execution, normal edges (`builder.add_edge("node_a", "node_b")`) represent fixed, unconditional transitions.

A **conditional edge** (`builder.add_conditional_edges(...)`) routes execution dynamically based on the output of a routing function. It inspects the current graph state after a node completes and selects the destination node from a pre-registered routing map:

```python
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "researcher": "researcher",
        "end": END,
    },
)
```

---

## 5. What Routing Functions Are

A routing function is a deterministic Python function that accepts the graph state as input and returns a string key corresponding to the next node in the graph:

```python
def route_after_critic(state: ResearchState) -> Literal["researcher", "end"]:
    ...
```

The routing function **does not execute business logic, search the web, or invoke LLMs**. It acts strictly as an objective traffic controller querying the state.

---

## 6. How State Controls Routing

Routing is governed entirely by state inspection:
- `state["critique"]`: Contains the structured `Critique` Pydantic model emitted by the `critic` node. Specifically, `critique.should_research_again` indicates whether research gaps persist.
- `state["research_iteration"]`: An integer tracking the number of completed research passes (e.g., 1, 2, 3).

The router makes decisions exclusively from these two state fields without external side effects.

---

## 7. Why Iteration Counters are Required

In cyclic graphs, an unbounded loop presents a catastrophic failure mode: **infinite recursion**.

If a critique continuously judges findings as insufficient, an unconstrained graph will loop endlessly, exhausting API quotas, running out of memory, or blocking execution indefinitely.

An explicit `research_iteration` counter stored in `ResearchState` guarantees that progress is tracked across cycles.

---

## 8. Why `MAX_ITERATIONS = 3`

`MAX_ITERATIONS = 3` serves as a practical, production-oriented balance:
- **Iteration 1**: Broad baseline search answering the initial user question.
- **Iteration 2**: Targeted gap-filling addressing the critic's initial recommendations.
- **Iteration 3**: Final refinement capturing long-tail nuances or secondary counterarguments.

Beyond 3 iterations, diminishing returns set in: LLMs typically recycle similar search queries, and additional web queries rarely alter synthesized consensus findings.

### The Bounded-Cycle Invariant
> *"The graph may revisit Researcher, but Researcher cannot execute more than MAX_ITERATIONS times."*

Even if the critic insists on `should_research_again = True`, the graph **must deterministically terminate at END** when `research_iteration >= MAX_ITERATIONS`.

---

## 9. Why Routing is Deterministic

Decision-making in this architecture is decoupled into two phases:
1. **Probabilistic Evaluation (Critic Node)**: An LLM evaluates semantic nuance, identifies missing topics, and produces a structured `Critique` object with `should_research_again: bool`.
2. **Deterministic Control Flow (Router Function)**: Pure Python logic routes based on boolean checks and integer comparisons:

```python
if critique is None or not critique.should_research_again:
    return "end"

if iteration >= MAX_ITERATIONS:
    return "end"

return "researcher"
```

This prevents non-deterministic hallucinations from corrupting graph flow and ensures 100% testable, predictable behavior.

---

## 10. How Critic Feedback Reaches the Next Researcher

Feedback propagation operates through the shared `ResearchState`:

```
[Critic]
   │ Emits: {"critique": Critique(recommended_queries=["..."])}
   ▼
[ResearchState] updated with critique
   │
   ▼
[Researcher] (Iteration 2+)
   │ Reads: state["critique"].recommended_queries
   │ Executes targeted searches on top recommendations
   ▼
Accumulates new sources with unique IDs (src_003, src_004)
```

1. On iteration 1, the researcher queries the user's raw `question`.
2. If the critic requests further research, it provides structured `recommended_queries: list[str]`.
3. On iteration 2+, the researcher reads `state["critique"].recommended_queries`, queries the search service, and merges newly discovered sources with existing ones while deduplicating URLs.

---

## 11. Why Subgraphs are Intentionally Deferred to Phase 3

In LangGraph, subgraphs allow encapsulating a multi-node cycle into an isolated node with its own private state schema and boundary contracts.

However, implementing cyclic control directly in the root graph first is crucial for understanding:
- How conditional edges interact with root state keys.
- How accumulation works without subgraph boundary mappings.
- How iteration counters propagate through unpartitioned state.

Phase 3 will cleanly extract `researcher -> analyst -> critic -> router` into a modular `ResearchSubgraph`.

---

## 12. How the Graph Terminates

The graph terminates at `END` under either of two conditions:

1. **Success Condition (Early Exit)**:
   - `critique.should_research_again == False` (research is deemed sufficient by the critic).
   - Execution concludes in 1 or 2 iterations.
2. **Hard Cutoff Condition (Safety Termination)**:
   - `research_iteration >= MAX_ITERATIONS` (3 iterations reached).
   - The router logs `[Router] max_iterations reached` and forces routing to `END`, returning the best findings accumulated so far.

---

## 13. Example Execution Traces

### Trace A: Good on First Pass (1 Iteration)
```
[Researcher] iteration=1
[RESEARCHER] retrieved 2 new sources (total accumulated: 2)
[Analyst] findings=2
[Critic] score=0.92 continue=False
[Router] next=end
```

### Trace B: Weak Then Good (2 Iterations)
```
[Researcher] iteration=1
[RESEARCHER] retrieved 2 new sources (total accumulated: 2)
[Analyst] findings=2
[Critic] score=0.62 continue=True
[Router] next=researcher
[Researcher] iteration=2
[RESEARCHER] executing targeted queries from critic: ['quantum fault tolerance thresholds surface codes']
[RESEARCHER] retrieved 2 new sources (total accumulated: 4)
[Analyst] findings=4
[Critic] score=0.89 continue=False
[Router] next=end
```

### Trace C: Always Weak (Hard Cutoff at 3 Iterations)
```
[Researcher] iteration=1
[Analyst] findings=2
[Critic] score=0.35 continue=True
[Router] next=researcher
[Researcher] iteration=2
[Analyst] findings=3
[Critic] score=0.45 continue=True
[Router] next=researcher
[Researcher] iteration=3
[Analyst] findings=4
[Critic] score=0.55 continue=True
[Router] max_iterations reached
[Router] next=end
```
