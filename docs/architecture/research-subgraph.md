# Architecture Design: Research & Critique Subgraph

This document details the architectural rationale, parent/child graph composition, state contracts, and encapsulation boundaries for the **Research Subgraph** in the **Verified Research Agent**.

---

## 1. Graph Architecture Diagrams

### Parent Graph
```
[START]
   │
   ▼
[research] (Compiled Research Subgraph)
   │
   ▼
 [END]
```

### Child Graph (Research Subgraph)
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

## 2. What a LangGraph Subgraph Is

In LangGraph, a **subgraph** is an independent `StateGraph` compiled into a `CompiledStateGraph` and registered as a standard node inside another `StateGraph`.

To the parent graph, the subgraph is simply a node (`"research"`). The parent invokes it with input state, waits for the subgraph's internal workflow to complete, and receives the resulting state update. Internally, the child graph can have its own complex topology—including multiple nodes, conditional edges, and bounded feedback cycles.

---

## 3. Why the Research Cycle is Extracted into a Subgraph

In earlier phases, the `researcher -> analyst -> critic -> router` cycle lived directly in the root graph. While functional for a simple prototype, embedding loops in the root graph creates architectural bottlenecks as systems grow:

1. **State Pollution & Complexity**: Downstream nodes (such as upcoming verifiers, human reviewers, or report writers) should not be aware of internal research iterations, intermediate query rewrites, or critic score adjustments.
2. **Reusability**: A self-contained research engine can be invoked multiple times across different workflows (e.g., initial research, gap-filling verification, follow-up Q&A) without duplicating graph definitions.
3. **Modularity**: Changes to research heuristics, search vendors, or analyst prompting remain strictly localized within the child graph without breaking parent pipeline contracts.

---

## 4. Parent vs. Child State

In the current architecture, the parent graph and child subgraph share the unified [`ResearchState`](file:///d:/research-agent/src/verified_research/graph/state.py) schema:

```python
class ResearchState(TypedDict):
    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
    critique: NotRequired[Critique]
    research_iteration: NotRequired[int]
```

### State Flow Between Parent and Child:
- **What enters the child**: The parent provides `{"question": "..."}` (and optionally `"research_iteration": 0`).
- **What the child modifies**: The child graph iterates internally, populating `sources`, `findings`, `critique`, and updating `research_iteration`.
- **What returns to the parent**: The completed state containing cumulative sources, grounded findings, the final critique evaluation, and total iterations executed.

---

## 5. Input and Output Contracts

### Subgraph Input Contract
- **Required**:
  - `question`: Non-empty `str`.
- **Optional**:
  - `research_iteration`: `int` (defaults to 0 if omitted).

### Subgraph Output Contract
- **Guaranteed Returns**:
  - `question`: The unmodified original research inquiry.
  - `sources`: `list[Source]` (all retrieved, deduplicated evidence).
  - `findings`: `list[Finding]` (synthesized insights citing valid `source_ids`).
  - `critique`: `Critique` (final evaluation assessing research completeness).
  - `research_iteration`: `int` (total completed research passes, bounded by `MAX_ITERATIONS = 3`).

---

## 6. Why Encapsulation is Useful

Encapsulation enforces the principle of least privilege in software architecture:
- **Structural Cleanliness**: The parent graph only registers one node: `parent_builder.nodes.keys() == ['research']`.
- **Testing in Isolation**: The research subgraph can be tested directly with mock search providers and mock LLMs without building the full outer system.
- **Independent Observability**: Tracing and latency measurements can be grouped by the parent's `"research"` milestone or drilled down into individual child node spans.

---

## 7. Why This is NOT Automatically a Multi-Agent System

It is critical to distinguish modular graph composition from a true multi-agent system:
- **State-Machine vs. Autonomous Agency**: The nodes (`researcher`, `analyst`, `critic`) are deterministic functions and structured prompt callers connected via a state machine.
- **No Autonomous Negotiation**: The critic does not negotiate or debate with the researcher. The researcher does not plan arbitrary tool invocations or autonomous subgoals.
- **Coordinated Pipeline**: The flow follows strict, programmatic routing based on schema inspection.

---

## 8. Subgraph vs. Normal Python Function vs. Agent

| Dimension | Normal Python Function | LangGraph Subgraph | Autonomous Agent |
| :--- | :--- | :--- | :--- |
| **Execution Model** | Imperative sequential code | Declarative StateGraph with nodes & edges | ReAct loop or dynamic tool-calling engine |
| **Flow Control** | Code loops (`while`, `for`) | Graph conditional edges and state transitions | LLM-directed tool selection |
| **State Inspection** | Local variables or return values | First-class, inspectable state dictionary | Agent scratchpad / memory |
| **Checkpoints & HITL** | Requires custom persistence code | Native LangGraph checkpointing & interrupts | Native agent checkpointing |
| **Composability** | Standard function composition | Composable as a node inside parent graphs | Spawned or orchestrated via supervisor |

---

## 9. Why Verifier is Not Yet Part of the Subgraph

The future `verifier` node is intentionally omitted from the research subgraph:
1. **Separation of Evidence Gathering vs. Truth Auditing**: The research subgraph's role is to gather information and synthesize initial findings. Verifying those findings at a sentence/claim level against raw source quotes is an auditing responsibility that belongs to a separate evaluation stage.
2. **Future Graph Topology**: In later phases, verification will sit between Research and Human Review:
   `START -> Research Subgraph -> Verifier -> Human Review -> Writer -> END`.

---

## 10. How This Prepares the Architecture for Later Phases

Extracting the research loop into a self-contained child graph lays the foundation for future capabilities:

1. **Phase 4 (Evidence & Claim Modeling)**: The child graph's `findings` will evolve into structured `Claim` entities backed by `Evidence` spans.
2. **Phase 5 (Verifier & Auditing)**: The parent graph will route from the `research` node directly into a `verifier` node.
3. **Phase 6 (Human-in-the-Loop & Checkpointing)**: The parent graph can pause execution after `research` completes, allowing human inspection before verification or report writing.
4. **Phase 7 (Writer & Report Synthesis)**: A dedicated `writer` node will take verified findings and produce a final publication-grade report.
