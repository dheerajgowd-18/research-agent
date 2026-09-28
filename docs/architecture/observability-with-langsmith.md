# Phase 12 — Observability with LangSmith

## 1. Why Observability is Required

In traditional linear applications, textual log streams (`logger.info`, `logger.error`) are often adequate to reconstruct operational history. However, for a multi-agent, graph-based LLM system like the **Verified Research Agent**, flat logs are fundamentally insufficient:

1. **Non-Linear Graph Topology & Multi-Cycle Loops**: The application executes an outer supervisor orchestration loop, an internal research/critique subgraph loop, claim-level verification branches, and human-in-the-loop (HITL) interrupt-and-resume cycles. In flat text logs, log messages from multiple iterations interleave, obscuring the parent-child relationships and causal lineage of actions.
2. **Context & State Drift**: Node executions transform state incrementally. When a failure or unexpected decision occurs (e.g., an unwarranted critique cycle or supervisor routing to `research` instead of `verify`), logs cannot easily show the exact state delta and input context that caused the agent to take that transition.
3. **Structured vs. Unstructured Concurrency**: Operations occur across heterogeneous boundaries: HTTP web searches (Tavily), structured JSON LLM generations (Groq/OpenAI), SQLite checkpoint storage, and human reviews. Reconstructing latencies, attempt counts, and token costs from log files requires tedious manual parsing.
4. **Hierarchical Failure Attribution**: When an external API failure occurs (such as a 429 Rate Limit in Tavily or a schema validation failure in an LLM call), observability answers *which agent, which step, and which retry attempt* failed, and whether the system recovered gracefully.

LangSmith provides structured, distributed tracing tailored for agentic graphs, turning opaque execution runs into clear, inspectable run trees without altering runtime logic.

---

## 2. Trace Structure

The trace model in LangSmith maps directly to the application's actual graph and worker architecture.

### Concepts
- **Trace**: A complete end-to-end execution of a research session, starting at graph invocation and concluding at final termination or approval.
- **Run / Span**: An individual unit of work within the trace. Every LangGraph node invocation, external tool call, LLM prompt, or sub-operation is represented as a span with start time, end time, duration, inputs, outputs, tags, and metadata.
- **Nested Execution**: Spans form a directed parent-child tree reflecting the invocation hierarchy of the graph and subgraphs.

### Concrete Trace Hierarchy
```
Research Session (Graph Root Trace)
│
├── supervisor (step 1)
│
├── research (Research Worker / Encapsulated Subgraph)
│   ├── researcher
│   │   └── search (Tavily search tool call)
│   │       ├── tavily_search:attempt_1 (e.g. 429 Rate Limit)
│   │       └── tavily_search:attempt_2 (Success)
│   ├── analyst
│   │   └── llm_synthesis (ChatGroq / with_structured_output)
│   └── critic
│       └── llm_critique (ChatGroq / with_structured_output)
│
├── supervisor (step 2)
│
├── verifier (Verifier Worker)
│   ├── verify_claim:claim_001
│   │   └── llm_verification (ClaimVerifierService entailment check)
│   ├── verify_claim:claim_002
│   │   └── llm_verification (ClaimVerifierService entailment check)
│   └── verify_claim:claim_003
│       └── llm_verification (ClaimVerifierService entailment check)
│
├── supervisor (step 3)
│
├── human_review (Human Review Worker)
│   ├── interrupt_requested (Graph suspended via interrupt())
│   └── resumed (Graph resumed with human action: approve/edit/research_more/reject)
│
└── supervisor (step 4: next_worker='finish', termination_reason='COMPLETED')
```

---

## 3. What is Observed

The observability implementation captures operational data across all architectural layers:

### Graph Execution
- **Root trace run**: Captures full session duration, question, thread identifier, and terminal state.
- **Node transitions**: Start time, end time, and duration of every graph node (`supervisor`, `research`, `verifier`, `human_review`).

### Supervisor
- **Supervisor step**: Step counter bounded by `MAX_SUPERVISOR_STEPS` (default: 8).
- **Selected worker**: Decided worker node (`research`, `verify`, `human_review`, `finish`).
- **Structured decision**: Rationale and target worker.
- **Termination reason**: Explicit reason when halted (`COMPLETED`, `MAX_SUPERVISOR_STEPS_REACHED`).
- **Unverified claims count**: Number of claims awaiting verification.
- **Note**: Internal chain-of-thought is not logged; only structured operational reasoning is captured.

### Research Worker
- **Search operations**: Number of search operations executed.
- **Search queries**: Sanitized search query strings.
- **Source counts**: Newly retrieved sources vs. total accumulated sources in state.
- **Research iteration**: Subgraph loop index (0 to 3).
- **Critic outcome**: Quality score (0.0 to 1.0), `should_research_again` flag, missing topics count, and citation gaps count.
- **Expansion events**: Follow-up query recommendations.

### Search Tool
- **Tool span**: `search` run type with query and limit parameters.
- **Result count**: Number of normalized sources returned.
- **Network duration**: Latency of the Tavily API round-trip.

### LLM Calls
- **Provider & Model**: `groq`, `openai`, or `fake`, along with model identifier (e.g. `llama-3.1-8b-instant`).
- **Structured output status**: Success or failure of Pydantic schema validation.
- **Latency**: Invocation duration.
- **Token usage**: Prompt, completion, and total tokens when reported by provider.

### Verifier
- **Individual Claim Spans**: `verify_claim:{claim_id}` span for each evaluated claim.
- **Verdict**: Categorical classification (`SUPPORTED`, `PARTIAL`, `UNSUPPORTED`).
- **Confidence**: Model certainty score (0.0 to 1.0).
- **Evidence count**: Number of preserved evidence items cited by the claim.
- **Reasoning**: Concise grounding justification.

### Reliability & Retries
- **Operation details**: Component name and operation identifier (e.g. `search:tavily_search`).
- **Attempt number & Retry count**: Total attempts made and number of retries required.
- **Error classification**: Phase 11 `ErrorCategory` (`RATE_LIMIT`, `NETWORK_TIMEOUT`, `SERVER_ERROR`, `AUTH_ERROR`).
- **Delay tracking**: Calculated backoff delay in seconds.
- **Final status**: `success` or `failure`.

### Human-in-the-Loop (HITL)
- **Review requested**: Payload generation and claim counts before graph interrupt.
- **Interrupt occurred**: Suspension boundary timestamp.
- **Resume occurred**: Re-activation timestamp upon receiving human input.
- **Human action**: Selected action (`approve`, `edit`, `research_more`, `reject`).
- **Cycle count**: Human research pass index (`human_research_cycles`).
- **Edited claims count**: Number of modified claims if action was `edit`.
- **Feedback status**: Boolean flag indicating presence of human guidance (raw feedback text is redacted).

---

## 4. Metadata Strategy

Metadata attached to spans is operational, structured, and strictly bounded. Large graph states or full documents are never dumped into metadata.

### Metadata Schemas

| Component | Metadata Fields | Purpose |
| :--- | :--- | :--- |
| **Session** | `thread_id`, `session_id`, `phase`, `question_preview`, `question_id` | Correlating threads across multiple conversational turns |
| **Supervisor** | `supervisor_step`, `selected_worker`, `decision`, `termination_reason`, `unverified_claims_count`, `reasoning` | Understanding orchestration routing and stopping criteria |
| **Research** | `research_iteration`, `search_operation_count`, `search_queries`, `new_sources_count`, `total_sources_count`, `critic_quality_score` | Tracking retrieval depth and critique decisions |
| **Search** | `query`, `max_results`, `sources_count` | Auditing external information retrieval |
| **Verifier** | `claim_id`, `verdict`, `confidence`, `evidence_count`, `reasoning` | Inspecting claim-level evidence verification outcomes |
| **HITL** | `hitl_event`, `human_action`, `human_research_cycle`, `claims_count`, `edited_claims_count`, `feedback_present` | Tracking human intervention boundaries |
| **Reliability** | `component`, `operation`, `attempt_number`, `retry_count`, `error_category`, `final_status`, `duration_seconds` | Inspecting retry behavior, rate limits, and failure recovery |

---

## 5. Security and Privacy

Observability telemetry must never become a vector for credential leakage or private data exposure.

### Credential Handling
- **Environment Isolation**: Tracing configuration is loaded exclusively from environment variables or local `.env` files. Credentials are never hardcoded.
- **Git Protection**: `.gitignore` strictly ignores `.env`, `.env.*`, and `*.local` files, preserving only `.env.example`.
- **Display Masking**: When configuration is inspected via `get_tracing_config()`, API keys are masked (e.g. `lsv2...6789`) and never exposed in full.

### Automated Secret Scrubbing
All text and metadata pass through `sanitize_metadata()` and `sanitize_text()` before emission:
- **Key Name Blacklist**: Any dictionary key matching `api_key`, `token`, `secret`, `authorization`, `password`, etc., is replaced with `[REDACTED]`.
- **Regex Pattern Matching**: Known API key formats (`gsk_*`, `tvly-*`, `lsv2_*`, `sk-*`, `Bearer *`) in strings, URLs, or error messages are automatically replaced with `[REDACTED_SECRET]`.

### Sensitive Data Avoidance
- **Full Text Redaction**: Retrieved web page contents and long document dumps are omitted from metadata.
- **User Feedback Protection**: User-entered critique in HITL review is represented by a boolean `feedback_present=True` flag in telemetry metadata, rather than recording raw subjective commentary.

---

## 6. Latency Analysis

Latencies are captured natively by the tracing hierarchy:
- **Graph Total Latency**: Duration of root graph execution.
- **Node Execution Latencies**: Individual durations for `supervisor`, `research`, `analyst`, `critic`, `verifier`, and `human_review`.
- **Tool & API Latencies**: Round-trip time of Tavily web searches and LLM structured outputs.
- **Retry Delays**: Exponential backoff sleep durations are logged on retry events.
- **HITL Pause Duration**: Measured as the time interval between `interrupt_requested` and `resumed`.

Manual timing calculations are avoided where LangSmith's span timestamps provide high-precision measurements.

---

## 7. Token Usage and Provider Limits

Token usage is captured when reported by the underlying LLM provider:
- **Metrics**: `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- **Extraction**: `ObservabilityCallbackHandler` inspects `response.llm_output["token_usage"]` and `AIMessage.usage_metadata`.
- **Limitations**:
  - `FakeChatModel` does not emit token usage; counts remain 0.
  - Certain Groq model endpoints or streaming modes may omit token usage metadata. If token usage is absent from the provider's response, counts are not fabricated and limitations are documented.

---

## 8. Observability vs. Evaluation

A strict architectural distinction is maintained between Observability and Evaluation:

```
┌──────────────────────────────────────────────────────────┐
│                      EXECUTION                           │
│                         │                                │
│                         ▼                                │
│             OBSERVABILITY (Phase 12)                     │
│         "What happened during this run?"                 │
│  - Which path did the graph take?                        │
│  - Which workers executed and how often?                 │
│  - How many retries and what latency?                    │
│  - What claim verdicts were produced?                    │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│               EVALUATION (Phase 6 / Benchmark)           │
│           "How well did the system perform?"             │
│  - Precision, Recall, F1 score against ground truth      │
│  - Confusion matrix (Supported / Partial / Unsupported)  │
│  - Accuracy across standardized test suites              │
└──────────────────────────────────────────────────────────┘
```

Observability monitors telemetry for live executions; it does not grade or score the correctness of the research agent against ground-truth benchmarks.

---

## 9. Provider and Integration Limitations

1. **LRU Cache in LangSmith SDK**: `langsmith.utils.get_env_var` caches environment variables. In-process changes to environment variables require `get_env_var.cache_clear()`. The `setup_langsmith_environment()` helper clears this cache automatically.
2. **Network Isolation in Tests**: Hosted LangSmith calls require active internet access and a valid API key. All automated unit and regression tests run in hermetic mode with tracing disabled or using an in-memory client, avoiding external network dependencies.
3. **Interrupt Serialization**: LangGraph `interrupt()` pauses execution via Python exception handling in Pregel. Spans around the interrupt node capture the boundary before and after resumption.

---

## 10. Deferred Work

The following items are deliberately deferred to future roadmap phases:
- **Final System Evaluation Benchmarks**: End-to-end precision/recall evaluation against external ground truth datasets.
- **Production Dashboards & Alerting**: Real-time SLA monitors, cost threshold alerts, and cloud Grafana/LangSmith dashboards.
- **Web UI & Streaming Front-End**: Interactive chat front-end and real-time streaming interfaces.
- **Writer Agent**: Final narrative report synthesis worker.
- **Deployment & Cloud Infrastructure**: Containerization, serverless deployment, and production scaling.
