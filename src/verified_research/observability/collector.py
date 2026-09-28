"""In-memory operational metrics collector and LangChain callback handler."""

import logging
import threading
import time
from typing import Any
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Callback handler capturing operational counts, latency, and token usage across graph runs.

    Metrics collected:
        - llm_calls: Total count of LLM requests
        - search_calls: Total count of web search operations
        - retries: Total count of retry attempts across all components
        - verifier_operations: Total count of claims verified
        - supervisor_decisions: Total count of supervisor orchestration steps
        - token_usage: Prompt, completion, and total tokens reported by model provider
        - node_runs: Durations and execution records for individual graph nodes
        - events: Chronological stream of operational lifecycle events
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls: int = 0
        self.search_calls: int = 0
        self.retries: int = 0
        self.verifier_operations: int = 0
        self.supervisor_decisions: int = 0
        self.token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.active_runs: dict[UUID, dict[str, Any]] = {}
        self.node_runs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record start of an LLM call."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract token usage reported by model provider."""
        with self._lock:
            # 1. Check llm_output dict
            if response.llm_output and isinstance(response.llm_output, dict):
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
                if isinstance(usage, dict):
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                    self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

            # 2. Check individual chat generation usage_metadata
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None and hasattr(msg, "usage_metadata") and isinstance(msg.usage_metadata, dict):
                        meta = msg.usage_metadata
                        self.token_usage["prompt_tokens"] += meta.get("input_tokens", 0)
                        self.token_usage["completion_tokens"] += meta.get("output_tokens", 0)
                        self.token_usage["total_tokens"] += meta.get("total_tokens", 0)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record start of a graph chain or node."""
        node_name = name or (serialized.get("name") if serialized else None) or "node"
        with self._lock:
            self.active_runs[run_id] = {
                "name": node_name,
                "start_time": time.monotonic(),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "metadata": metadata or {},
            }

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record end of a graph chain or node and calculate latency."""
        with self._lock:
            info = self.active_runs.pop(run_id, None)
            if info:
                duration = time.monotonic() - info["start_time"]
                self.node_runs.append({
                    "name": info["name"],
                    "duration_seconds": round(duration, 4),
                    "parent_run_id": info["parent_run_id"],
                })

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool invocations."""
        tool_name = name or (serialized.get("name") if serialized else None) or ""
        with self._lock:
            if "search" in tool_name.lower():
                self.search_calls += 1

    def record_retry(
        self,
        component: str,
        operation: str,
        attempt: int,
        delay: float = 0.0,
        error_category: str | None = None,
    ) -> None:
        """Record a retry event and increment retry counter."""
        with self._lock:
            self.retries += 1
            self.events.append({
                "type": "retry",
                "component": component,
                "operation": operation,
                "attempt": attempt,
                "delay_seconds": delay,
                "category": str(error_category) if error_category else None,
                "timestamp": time.time(),
            })

    def record_search(self, query: str, results_count: int) -> None:
        """Record a search operation and increment search counter."""
        with self._lock:
            self.search_calls += 1
            self.events.append({
                "type": "search",
                "query": query[:100],
                "results_count": results_count,
                "timestamp": time.time(),
            })

    def record_verifier_operation(
        self,
        claim_id: str,
        verdict: str,
        confidence: float,
        evidence_count: int = 0,
    ) -> None:
        """Record a claim verification and increment verifier counter."""
        with self._lock:
            self.verifier_operations += 1
            self.events.append({
                "type": "verify_claim",
                "claim_id": claim_id,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_count": evidence_count,
                "timestamp": time.time(),
            })

    def record_supervisor_decision(
        self,
        step: int,
        worker: str,
        termination_reason: str | None = None,
    ) -> None:
        """Record a supervisor step and increment supervisor decisions counter."""
        with self._lock:
            self.supervisor_decisions += 1
            self.events.append({
                "type": "supervisor_decision",
                "step": step,
                "worker": worker,
                "termination_reason": termination_reason,
                "timestamp": time.time(),
            })

    def record_hitl_action(
        self,
        event: str,
        action: str | None = None,
        cycle: int = 0,
    ) -> None:
        """Record a Human-in-the-Loop review event."""
        with self._lock:
            self.events.append({
                "type": "human_review",
                "event": event,
                "action": action,
                "cycle": cycle,
                "timestamp": time.time(),
            })

    def get_summary(self) -> dict[str, Any]:
        """Produce a comprehensive summary of operational execution counts and performance."""
        with self._lock:
            return {
                "counts": {
                    "llm_calls": self.llm_calls,
                    "search_calls": self.search_calls,
                    "retries": self.retries,
                    "verifier_operations": self.verifier_operations,
                    "supervisor_decisions": self.supervisor_decisions,
                },
                "token_usage": dict(self.token_usage),
                "node_executions_count": len(self.node_runs),
                "node_runs": list(self.node_runs),
                "events_count": len(self.events),
                "events": list(self.events),
            }
