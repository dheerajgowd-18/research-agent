# Architecture Design: LangGraph State & Node Contracts (Phase 1)

This document details the architectural rationale, state schema boundaries, node contracts, and lifecycle transitions for Phase 1 of the **Verified Research Agent**.

---

## 1. Why Each State Field Exists

The Phase 1 state schema is defined as `ResearchState`:

```python
class ResearchState(TypedDict):
    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
```

### `question` (`str`)
- **Purpose**: Represents the core research objective initiated by the user.
- **Why in State**: It serves as the immutable input anchor for the entire pipeline. Both the `researcher` node (to form query context) and the `analyst` node (to direct semantic synthesis toward the user's specific intent) require access to the original question.

### `sources` (`list[Source]`)
- **Purpose**: Normalized, typed representations of external evidence retrieved from search.
- **Why in State**: The researcher must communicate its external discoveries to downstream nodes. Without `sources` in state, the downstream `analyst` cannot inspect, synthesize, or reference grounding documents. Each `Source` has a deterministic `source_id` (`src_001`, `src_002`, etc.) that serves as the root key for downstream citation and verification.

### `findings` (`list[Finding]`)
- **Purpose**: Structured atomic insights synthesized from the evidence.
- **Why in State**: Findings represent the primary synthesis output of the pipeline. In later phases, findings become the foundational units audited by critics and verifiers. Each finding retains explicit links to `source_ids`, ensuring verifiable provenance.

---

## 2. Why Certain Values are NOT in State

A fundamental principle of LangGraph state architecture is: **Only persist data that must cross node boundaries or be inspected by downstream/external observers.**

The following values are deliberately kept **local** to their respective nodes:

| Omitted Element | Where It Lives | Architectural Rationale |
| :--- | :--- | :--- |
| **Raw Tavily API JSON** | Local to `TavilySearchClient` | Raw payload includes provider-specific metadata (request IDs, search depth logs, raw ranking scores) that pollute graph state and tightly couple the graph schema to a single search provider. |
| **System & User Prompts** | Local to `analyst_node` | Prompts are implementation details of node execution, subject to prompt engineering iterations. Exposing prompts in state adds bloat without downstream utility. |
| **Intermediate Parsing / Tokens** | Local to nodes | Token counts, raw LLM completions, and JSON chunks are transient runtime artifacts. |
| **Local Iterator / Index Counters** | Local to normalization helpers | Sequence indices (`src_001`, `src_002`) are computed during normalization and sealed inside the immutable `Source` objects. |

---

## 3. Node Contracts

Each node has strict, explicit input/output boundaries with zero hidden dependencies.

### Researcher Node (`researcher_node`)
- **Input Contract**:
  - `question`: Non-empty `str`.
- **Output Contract**:
  - `{"sources": list[Source]}`
- **Responsibilities**:
  1. Receive research `question` from state.
  2. Validate query validity (reject empty/blank questions).
  3. Query search engine via the `SearchService` abstraction.
  4. Normalize raw search hits into immutable `Source` instances with uniform identifiers (`src_001`, etc.).
  5. Return state patch containing only `sources`.
- **Edge Conditions**:
  - When search returns 0 hits, logs a warning and returns `{"sources": []}`.

### Analyst Node (`analyst_node`)
- **Input Contract**:
  - `question`: Non-empty `str`.
  - `sources`: `list[Source]`.
- **Output Contract**:
  - `{"findings": list[Finding]}`
- **Responsibilities**:
  1. Read `question` and normalized `sources`.
  2. If `sources` is empty, cleanly return `{"findings": []}` without crashing.
  3. Invoke configured LLM with Pydantic structured output (`AnalystOutput`).
  4. **Strict Traceability Validation**: Validate that every `source_id` referenced in every `Finding` exists in `state["sources"]`.
  5. Reject invented or hallucinated source IDs by raising `InvalidSourceReferenceError`.
  6. Return state patch containing only `findings`.

---

## 4. State Transitions

The Phase 1 pipeline follows a deterministic linear execution without conditional branches or cyclic loops:

```
[START]
   │
   │  Initial State: {"question": "..."}
   ▼
[researcher]
   │
   │  State Patch: {"sources": [Source(...), ...]}
   ▼
[analyst]
   │
   │  State Patch: {"findings": [Finding(...), ...]}
   ▼
[END]
```

### Complete State Evolution Table

| Step | State Snapshot | Modifying Node |
| :--- | :--- | :--- |
| **0. Invocation** | `{question: "..."}` | Caller (User / API) |
| **1. Post-Researcher** | `{question: "...", sources: [src_001, src_002]}` | `researcher` |
| **2. Post-Analyst** | `{question: "...", sources: [...], findings: [finding_001, ...]}` | `analyst` |
| **3. Final Output** | Terminal state returned to caller | `END` |

---

## 5. Traceability Architecture

Traceability in Phase 1 establishes the bedrock for verified AI research:

```
Finding (e.g. 'finding_001')
   │
   ├── text: "..."
   └── source_ids: ["src_001", "src_002"]
          │
          ▼
       Source (e.g. 'src_001')
          ├── title: "..."
          ├── url: "https://..."
          └── content: "..."
```

No finding can exist detached from its supporting source. This prevents ungrounded hallucinations from silently contaminating subsequent stages.

---

## 6. Future Extensibility (Roadmap)

While Phase 1 deliberately stops at `analyst -> END`, the state architecture is explicitly designed to evolve seamlessly into subsequent phases:

1. **Claim & Evidence Decomposition** (Phase 2):
   - `Finding` will decompose into fine-grained `Claim` and `Evidence` entities:
     `Claim -> Evidence -> Source`.
2. **Critic & Verifier Nodes** (Phase 3):
   - A `critic` node will evaluate whether findings answer the question adequately, emitting a critique score or feedback.
   - A `verifier` node will cross-check each claim against raw source quotes, setting a `verification_status: Literal["verified", "unverified", "contradicted"]`.
3. **Cyclic Exploration & Search Expansion** (Phase 4):
   - Conditional edges routing from `critic` back to `researcher` when evidence gaps are identified.
4. **Human-in-the-Loop (HITL) & Checkpointing** (Phase 5):
   - Graph interruption before final report generation to allow user approval of claims and search directions.
5. **Report Writer** (Phase 6):
   - A `writer` node synthesizing an audited, citation-annotated final markdown report.
