"""Backend service managing research graph execution, event streaming, and state recovery."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, AsyncIterator
import uuid

import anyio
from fastapi import HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from verified_research.agents.human_review import validate_edited_claims
from verified_research.api.models import (
    AgentEvent,
    AgentEventType,
    ResearchStateResponse,
    ResumeReviewRequest,
    UIStatusType,
)
from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import Claim, Evidence
from verified_research.reliability.classification import sanitize_error_message

logger = logging.getLogger(__name__)


def _to_dict(obj: Any) -> Any:
    """Recursively convert Pydantic models or containers into JSON-serializable dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


class ResearchService:
    """Service orchestrating research graphs, event streaming, and checkpoint reconnection."""

    def __init__(
        self,
        graph: CompiledStateGraph | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        """Initialize the ResearchService with a checkpointer and compiled graph."""
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = graph or create_supervisor_graph(checkpointer=self.checkpointer)

        # In-memory event buffers and subscriber queues per thread
        self._event_buffers: dict[str, list[AgentEvent]] = {}
        self._active_subscribers: dict[str, list[asyncio.Queue[AgentEvent]]] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._errors: dict[str, str] = {}
        self._questions: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def _emit_event(
        self,
        thread_id: str,
        event_type: AgentEventType,
        node: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Store event in thread buffer and broadcast to all active SSE subscribers."""
        event = AgentEvent(
            event_type=event_type,
            thread_id=thread_id,
            node=node,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data or {},
        )

        async with self._lock:
            if thread_id not in self._event_buffers:
                self._event_buffers[thread_id] = []
            self._event_buffers[thread_id].append(event)

            subscribers = self._active_subscribers.get(thread_id, [])
            for queue in subscribers:
                await queue.put(event)

        return event

    async def start_research(self, question: str, thread_id: str | None = None) -> str:
        """Initiate a research run and launch background graph execution."""
        tid = thread_id or f"research_{uuid.uuid4().hex[:12]}"

        async with self._lock:
            self._questions[tid] = question
            self._errors.pop(tid, None)
            if tid not in self._event_buffers:
                self._event_buffers[tid] = []

        await self._emit_event(
            thread_id=tid,
            event_type="run_started",
            data={"question": question},
        )

        task = asyncio.create_task(self._execute_graph(tid, {"question": question}))
        async with self._lock:
            self._running_tasks[tid] = task

        return tid

    async def resume_human_review(self, thread_id: str, request: ResumeReviewRequest) -> str:
        """Authoritatively validate human review action and resume graph execution."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)

        if not state.values and thread_id not in self._event_buffers:
            raise HTTPException(status_code=404, detail=f"Thread session '{thread_id}' not found.")

        if state.next != ("human_review",):
            raise HTTPException(
                status_code=400,
                detail=f"Thread '{thread_id}' is not awaiting human review (state.next={state.next}).",
            )

        # Authoritative server-side validation for 'edit' action
        if request.action == "edit":
            if not request.edited_claims:
                raise HTTPException(
                    status_code=422,
                    detail="Action 'edit' requires a non-empty list of 'edited_claims'.",
                )
            try:
                parsed_claims = [Claim.model_validate(c) for c in request.edited_claims]
                available_evidence = [
                    Evidence.model_validate(e) if not isinstance(e, Evidence) else e
                    for e in state.values.get("evidence", [])
                ]
                validate_edited_claims(parsed_claims, available_evidence)
            except Exception as err:
                raise HTTPException(status_code=422, detail=f"Invalid edited claims: {err}")

        # Emit resume event
        await self._emit_event(
            thread_id=thread_id,
            event_type="human_review_resumed",
            node="human_review",
            data={"action": request.action, "feedback": request.feedback},
        )

        resume_payload: dict[str, Any] = {"action": request.action}
        if request.feedback is not None:
            resume_payload["feedback"] = request.feedback
        if request.edited_claims is not None:
            resume_payload["edited_claims"] = request.edited_claims

        task = asyncio.create_task(self._execute_graph(thread_id, Command(resume=resume_payload)))
        async with self._lock:
            self._running_tasks[thread_id] = task

        return thread_id

    async def _execute_graph(self, thread_id: str, input_data: Any) -> None:
        """Run graph streaming in a thread pool and map chunks to typed AgentEvents in real time."""
        config = {"configurable": {"thread_id": thread_id}}
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        # Bridge callback for synchronous worker nodes executing in thread pool
        def bridge_emitter(event_type: str, node: str | None, data: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("live_event", (event_type, node, data)))

        def run_stream():
            from verified_research.api.events import set_execution_event_emitter
            set_execution_event_emitter(bridge_emitter)
            try:
                for chunk in self.graph.stream(input_data, config, stream_mode="updates"):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                set_execution_event_emitter(None)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        # Launch the stream in background worker thread
        stream_task = asyncio.create_task(anyio.to_thread.run_sync(run_stream))

        # Initial node_started event for supervisor
        await self._emit_event(
            thread_id=thread_id,
            event_type="node_started",
            node="supervisor",
            data={"message": "Supervisor evaluating state..."},
        )

        try:
            while True:
                msg_type, payload = await queue.get()
                if msg_type == "error":
                    raise payload
                elif msg_type == "done":
                    break
                elif msg_type == "live_event":
                    ev_type, ev_node, ev_data = payload
                    await self._emit_event(
                        thread_id=thread_id,
                        event_type=ev_type,
                        node=ev_node,
                        data=ev_data,
                    )
                elif msg_type == "chunk":
                    chunk = payload
                    if not isinstance(chunk, dict):
                        continue

                    # 1. Supervisor Worker Decisions
                    if "supervisor" in chunk:
                        sup_data = chunk["supervisor"]
                        decision = sup_data.get("supervisor_decision")
                        reasoning = decision.reasoning if decision else ""
                        next_worker = decision.next_worker if decision else "finish"
                        steps = sup_data.get("supervisor_steps", 0)

                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="supervisor_decision",
                            node="supervisor",
                            data={
                                "step": steps,
                                "next_worker": next_worker,
                                "reasoning": reasoning,
                                "termination_reason": sup_data.get("supervisor_termination_reason"),
                            },
                        )
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="node_completed",
                            node="supervisor",
                            data={"step": steps, "next_worker": next_worker},
                        )
                        if next_worker != "finish":
                            worker_label = "Research Subgraph" if next_worker == "research" else next_worker.capitalize()
                            await self._emit_event(
                                thread_id=thread_id,
                                event_type="node_started",
                                node=next_worker,
                                data={"message": f"Worker '{worker_label}' activated by supervisor"},
                            )

                    # 2. Research Subgraph Updates
                    if "research" in chunk:
                        res_data = chunk["research"]
                        sources = res_data.get("sources", [])
                        claims = res_data.get("claims", [])
                        findings = res_data.get("findings", [])

                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="research_update",
                            node="research",
                            data={
                                "sources_count": len(sources),
                                "claims_count": len(claims),
                                "findings_count": len(findings),
                            },
                        )
                        for s in sources:
                            await self._emit_event(
                                thread_id=thread_id,
                                event_type="source_found",
                                node="research",
                                data=_to_dict(s),
                            )
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="node_completed",
                            node="research",
                            data={"sources_count": len(sources), "claims_count": len(claims)},
                        )

                    # 3. Verifier Updates
                    if "verifier" in chunk:
                        ver_data = chunk["verifier"]
                        results = ver_data.get("verification_results", [])
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="verification_update",
                            node="verifier",
                            data={
                                "verification_results": _to_dict(results),
                                "count": len(results),
                            },
                        )
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="node_completed",
                            node="verifier",
                            data={"verified_count": len(results)},
                        )

                    # 4. Human Review Node Updates
                    if "human_review" in chunk:
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="node_completed",
                            node="human_review",
                            data=_to_dict(chunk["human_review"]),
                        )

                    # 5. Graph Interrupt (Human Review Required)
                    if "__interrupt__" in chunk:
                        interrupt_val = chunk["__interrupt__"][0].value
                        await self._emit_event(
                            thread_id=thread_id,
                            event_type="human_review_required",
                            node="human_review",
                            data=_to_dict(interrupt_val),
                        )

            # Ensure the worker thread task completes
            await stream_task

            # Check post-execution state for finality
            state = self.graph.get_state(config)
            if state.next == ():
                term_reason = state.values.get("supervisor_termination_reason")
                if term_reason == "REJECTED":
                    await self._emit_event(
                        thread_id=thread_id,
                        event_type="run_rejected",
                        data={
                            "reason": "Research rejected by human reviewer.",
                            "feedback": state.values.get("human_review", {}).feedback
                            if hasattr(state.values.get("human_review"), "feedback")
                            else None,
                        },
                    )
                else:
                    await self._emit_event(
                        thread_id=thread_id,
                        event_type="run_completed",
                        data={
                            "termination_reason": term_reason or "COMPLETED",
                            "claims_count": len(state.values.get("claims", [])),
                            "sources_count": len(state.values.get("sources", [])),
                            "evidence_count": len(state.values.get("evidence", [])),
                            "verified_count": len(state.values.get("verification_results", [])),
                        },
                    )

        except Exception as exc:
            logger.exception("Error executing graph for thread %s", thread_id)
            sanitized = sanitize_error_message(str(exc))
            async with self._lock:
                self._errors[thread_id] = sanitized
            await self._emit_event(
                thread_id=thread_id,
                event_type="run_failed",
                data={"error": sanitized},
            )
        finally:
            async with self._lock:
                self._running_tasks.pop(thread_id, None)

    async def get_state(self, thread_id: str) -> ResearchStateResponse:
        """Recover and return the full state snapshot for a thread session."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)

        async with self._lock:
            has_buffer = thread_id in self._event_buffers
            err = self._errors.get(thread_id)
            saved_q = self._questions.get(thread_id, "")

        if not state.values and not has_buffer:
            raise HTTPException(
                status_code=404,
                detail=f"Research session with thread_id '{thread_id}' not found.",
            )

        status: UIStatusType = "running"
        if err:
            status = "failed"
        elif state.next == ("human_review",):
            status = "waiting_for_human"
        elif state.next == ():
            term_reason = state.values.get("supervisor_termination_reason")
            if term_reason == "REJECTED":
                status = "rejected"
            else:
                status = "completed"
        elif state.next:
            status = "running"
        else:
            status = "idle"

        # Extract review context if waiting for human review
        review_context = None
        if state.tasks:
            for task in state.tasks:
                if task.interrupts:
                    review_context = _to_dict(task.interrupts[0].value)
                    break

        return ResearchStateResponse(
            thread_id=thread_id,
            status=status,
            question=state.values.get("question", saved_q),
            sources=_to_dict(state.values.get("sources", [])),
            evidence=_to_dict(state.values.get("evidence", [])),
            findings=_to_dict(state.values.get("findings", [])),
            claims=_to_dict(state.values.get("claims", [])),
            verification_results=_to_dict(state.values.get("verification_results", [])),
            human_review=_to_dict(state.values.get("human_review")),
            supervisor_decision=_to_dict(state.values.get("supervisor_decision")),
            supervisor_termination_reason=state.values.get("supervisor_termination_reason"),
            supervisor_steps=state.values.get("supervisor_steps", 0),
            research_iteration=state.values.get("research_iteration", 0),
            human_research_cycles=state.values.get("human_research_cycles", 0),
            review_context=review_context,
            error=err,
        )

    async def stream_events(self, thread_id: str) -> AsyncIterator[AgentEvent]:
        """Yield historical and live AgentEvents as an async generator for SSE."""
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        async with self._lock:
            # Replay all buffered events to catch up reconnecting clients
            buffered = list(self._event_buffers.get(thread_id, []))
            if thread_id not in self._active_subscribers:
                self._active_subscribers[thread_id] = []
            self._active_subscribers[thread_id].append(queue)

        # 1. Yield historical events
        for event in buffered:
            yield event

        # Check if run has already reached a stopping or interrupt point
        if buffered and buffered[-1].event_type in (
            "run_completed",
            "run_failed",
            "run_rejected",
            "human_review_required",
        ):
            async with self._lock:
                if thread_id in self._active_subscribers and queue in self._active_subscribers[thread_id]:
                    self._active_subscribers[thread_id].remove(queue)
            return

        # 2. Stream live events
        try:
            while True:
                event = await queue.get()
                yield event
                if event.event_type in ("run_completed", "run_failed", "run_rejected", "human_review_required"):
                    break
        finally:
            async with self._lock:
                if thread_id in self._active_subscribers and queue in self._active_subscribers[thread_id]:
                    self._active_subscribers[thread_id].remove(queue)
