"""Comprehensive test suite for the Verified Research Agent API, SSE Streaming, and UI endpoints."""

import json
from unittest.mock import MagicMock
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver

from verified_research.api import create_app
from verified_research.graph.graph import create_supervisor_graph
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)


@pytest.fixture
def mock_pipeline():
    """Create deterministic mock components for API tests."""
    sources = [
        Source(
            source_id="src_001",
            title="Enterprise RAG Security Guidelines",
            url="https://example.com/rag-security",
            content="Vector embeddings are vulnerable to prompt injection and indirect document poisoning.",
        )
    ]
    evidence = [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Vector embeddings are vulnerable to prompt injection and indirect document poisoning.",
        )
    ]
    claims = [
        Claim(
            claim_id="claim_001",
            text="Enterprise RAG systems face prompt injection and indirect poisoning risks.",
            evidence_ids=["ev_001"],
        )
    ]
    findings = [
        Finding(
            finding_id="f_001",
            text="Prompt injection and document poisoning affect enterprise RAG vectors.",
            source_ids=["src_001"],
        )
    ]
    verifications = [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.97,
            reasoning="Direct textual grounding in src_001.",
            evidence_ids=["ev_001"],
        )
    ]

    def mock_subgraph(state):
        return {
            "sources": sources,
            "evidence": evidence,
            "claims": claims,
            "findings": findings,
            "research_iteration": state.get("research_iteration", 0) + 1,
        }

    def mock_verifier(state):
        return {"verification_results": verifications}

    return {
        "sources": sources,
        "evidence": evidence,
        "claims": claims,
        "findings": findings,
        "verifications": verifications,
        "subgraph": mock_subgraph,
        "verifier": mock_verifier,
    }


@pytest.fixture
def test_app(mock_pipeline):
    """Instantiate test application with deterministic graph and in-memory checkpointer."""
    checkpointer = MemorySaver()
    graph = create_supervisor_graph(
        custom_subgraph=mock_pipeline["subgraph"],
        custom_verifier=mock_pipeline["verifier"],
        checkpointer=checkpointer,
    )
    return create_app(graph=graph, checkpointer=checkpointer)


# ==============================================================================
# 1. Health and UI Static File Serving Tests
# ==============================================================================


@pytest.mark.anyio
async def test_health_check(test_app):
    """Health check endpoint returns operational status."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"
        assert res.json()["service"] == "verified-research-agent"


@pytest.mark.anyio
async def test_ui_index_served(test_app):
    """Root UI path serves HTML single-page application."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Verified Research Agent" in res.text
        assert "id=\"hitlSection\"" in res.text


# ==============================================================================
# 2. Research Run Initiation & Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_start_research_valid_query(test_app):
    """Valid research query initiates a run and assigns a thread_id."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.post("/api/research", json={"question": "What are enterprise RAG risks?"})
        assert res.status_code == 202
        body = res.json()
        assert "thread_id" in body
        assert body["status"] == "running"
        assert body["thread_id"].startswith("research_")


@pytest.mark.anyio
async def test_start_research_invalid_query_rejected(test_app):
    """Short or empty research questions are rejected by Pydantic validation."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.post("/api/research", json={"question": "a"})
        assert res.status_code == 422


# ==============================================================================
# 3. Event Streaming and Ordering Tests
# ==============================================================================


@pytest.mark.anyio
async def test_event_stream_lifecycle_ordering(test_app):
    """SSE stream delivers events in the expected graph lifecycle order."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        # 1. Start research
        init_res = await client.post("/api/research", json={"question": "What are RAG risks?"})
        tid = init_res.json()["thread_id"]

        # 2. Consume SSE stream until human review interrupt
        events = []
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream_res:
            assert stream_res.status_code == 200
            assert "text/event-stream" in stream_res.headers["content-type"]
            async for line in stream_res.aiter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    events.append(evt)
                    if evt["event_type"] == "human_review_required":
                        break

        # 3. Verify event sequencing
        event_types = [e["event_type"] for e in events]
        assert "run_started" in event_types
        assert "supervisor_decision" in event_types
        assert "research_update" in event_types
        assert "source_found" in event_types
        assert "verification_update" in event_types
        assert "human_review_required" in event_types

        # First event is run_started, last before pause is human_review_required
        assert event_types[0] == "run_started"
        assert event_types[-1] == "human_review_required"


# ==============================================================================
# 4. Human Review Interrupt State & Traceability Tests
# ==============================================================================


@pytest.mark.anyio
async def test_human_review_interrupt_state_and_traceability(test_app):
    """Execution pauses at human_review, exposing review context and grounding links."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "What are RAG risks?"})
        tid = init_res.json()["thread_id"]

        # Stream until pause
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        # Query state snapshot
        state_res = await client.get(f"/api/research/{tid}")
        assert state_res.status_code == 200
        state = state_res.json()

        assert state["status"] == "waiting_for_human"
        assert len(state["claims"]) == 1
        assert len(state["evidence"]) == 1
        assert len(state["sources"]) == 1
        assert len(state["verification_results"]) == 1
        assert state["review_context"] is not None

        # Verify Claim -> Evidence -> Source linkage in API payload
        claim = state["claims"][0]
        evidence = state["evidence"][0]
        source = state["sources"][0]
        assert claim["evidence_ids"] == [evidence["evidence_id"]]
        assert evidence["source_id"] == source["source_id"]


# ==============================================================================
# 5. Human Review Actions: Approve, Edit, Research More, Reject
# ==============================================================================


@pytest.mark.anyio
async def test_hitl_approve_action(test_app):
    """Human approve action resumes execution to COMPLETED terminal state."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Approve test"})
        tid = init_res.json()["thread_id"]

        # Wait for interrupt
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        # Submit approve action
        resume_res = await client.post(
            f"/api/research/{tid}/resume",
            json={"action": "approve", "feedback": "Approved as verified."},
        )
        assert resume_res.status_code == 202
        assert resume_res.json()["status"] == "resuming"

        # Stream resumption until run_completed
        completed_seen = False
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    if evt["event_type"] == "run_completed":
                        completed_seen = True
                        break

        assert completed_seen is True

        # State check
        final_state = (await client.get(f"/api/research/{tid}")).json()
        assert final_state["status"] == "completed"
        assert final_state["human_review"]["action"] == "approve"
        assert final_state["supervisor_termination_reason"] == "COMPLETED"


@pytest.mark.anyio
async def test_hitl_edit_valid_claims(test_app):
    """Human edit action with valid claims and evidence updates claims and completes."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Edit test"})
        tid = init_res.json()["thread_id"]

        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        edited_data = [
            {
                "claim_id": "claim_001",
                "text": "Corrected assertion regarding RAG security vulnerabilities.",
                "evidence_ids": ["ev_001"],
            }
        ]

        resume_res = await client.post(
            f"/api/research/{tid}/resume",
            json={"action": "edit", "edited_claims": edited_data},
        )
        assert resume_res.status_code == 202

        # Stream until completion
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "run_completed" in line:
                    break

        final_state = (await client.get(f"/api/research/{tid}")).json()
        assert final_state["status"] == "completed"
        assert final_state["claims"][0]["text"] == "Corrected assertion regarding RAG security vulnerabilities."


@pytest.mark.anyio
async def test_hitl_edit_invalid_claims_rejected_by_backend(test_app):
    """Backend authoritatively rejects edits referencing unknown evidence IDs."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Invalid edit test"})
        tid = init_res.json()["thread_id"]

        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        # Attempt to edit with phantom evidence ID
        bad_edit = [
            {
                "claim_id": "claim_001",
                "text": "Hallucinated claim referencing phantom evidence.",
                "evidence_ids": ["ev_nonexistent_9999"],
            }
        ]

        resume_res = await client.post(
            f"/api/research/{tid}/resume",
            json={"action": "edit", "edited_claims": bad_edit},
        )
        assert resume_res.status_code == 422
        assert "references unknown evidence_id" in resume_res.json()["detail"]

        # Ensure state remains waiting_for_human and was NOT corrupted
        current_state = (await client.get(f"/api/research/{tid}")).json()
        assert current_state["status"] == "waiting_for_human"


@pytest.mark.anyio
async def test_hitl_research_more_action(test_app):
    """Human research_more action re-enters research loop and resets verifications."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Research more test"})
        tid = init_res.json()["thread_id"]

        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        # Submit research_more
        resume_res = await client.post(
            f"/api/research/{tid}/resume",
            json={"action": "research_more", "feedback": "Investigate vector injection in detail."},
        )
        assert resume_res.status_code == 202

        # Stream until next interrupt
        seen_research_update = False
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "research_update" in line:
                    seen_research_update = True
                if "human_review_required" in line:
                    break

        assert seen_research_update is True
        state = (await client.get(f"/api/research/{tid}")).json()
        assert state["human_research_cycles"] == 1


@pytest.mark.anyio
async def test_hitl_reject_action_produces_rejected_status(test_app):
    """Human reject action immediately concludes execution with terminal REJECTED status."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Reject test"})
        tid = init_res.json()["thread_id"]

        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        resume_res = await client.post(
            f"/api/research/{tid}/resume",
            json={"action": "reject", "feedback": "Evidence quality unsatisfactory."},
        )
        assert resume_res.status_code == 202

        # Stream until run_rejected
        rejected_seen = False
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    if evt["event_type"] == "run_rejected":
                        rejected_seen = True
                        break

        assert rejected_seen is True
        final_state = (await client.get(f"/api/research/{tid}")).json()
        assert final_state["status"] == "rejected"
        assert final_state["supervisor_termination_reason"] == "REJECTED"


# ==============================================================================
# 6. Reconnection and Failure Tests
# ==============================================================================


@pytest.mark.anyio
async def test_reconnection_recovers_thread_state_and_history(test_app):
    """Reconnecting to an existing thread recovers full state snapshot and event stream."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Reconnection test"})
        tid = init_res.json()["thread_id"]

        # Wait until paused
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if "human_review_required" in line:
                    break

        # Simulate browser disconnecting and reconnecting via GET /api/research/{tid}
        reconnect_res = await client.get(f"/api/research/{tid}")
        assert reconnect_res.status_code == 200
        rec_data = reconnect_res.json()
        assert rec_data["thread_id"] == tid
        assert rec_data["status"] == "waiting_for_human"
        assert len(rec_data["claims"]) == 1

        # Reconnecting to stream re-delivers buffered events
        buffered_events = []
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    buffered_events.append(json.loads(line[6:]))
                if len(buffered_events) >= 5:
                    break

        assert len(buffered_events) >= 5
        assert buffered_events[0]["event_type"] == "run_started"


@pytest.mark.anyio
async def test_backend_failure_sanitization_and_status(mock_pipeline):
    """Backend failures emit run_failed and set status to failed with sanitized error."""
    # Graph worker that raises an exception
    def failing_subgraph(state):
        raise ConnectionError("Upstream search provider timeout (token=sec_abc123secret)")

    checkpointer = MemorySaver()
    failing_graph = create_supervisor_graph(
        custom_subgraph=failing_subgraph,
        checkpointer=checkpointer,
    )
    app = create_app(graph=failing_graph, checkpointer=checkpointer)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Failure test"})
        tid = init_res.json()["thread_id"]

        failed_event_seen = False
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    if evt["event_type"] == "run_failed":
                        failed_event_seen = True
                        # Ensure secret tokens are not in the event payload
                        assert "sec_abc123secret" not in evt["data"]["error"]
                        break

        assert failed_event_seen is True
        state = (await client.get(f"/api/research/{tid}")).json()
        assert state["status"] == "failed"
        assert state["error"] is not None
        assert "sec_abc123secret" not in state["error"]


@pytest.mark.anyio
async def test_security_credentials_not_exposed(test_app):
    """Ensure no environment secrets or private variables are exposed in responses or static files."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        # Check UI file
        ui_res = await client.get("/")
        for sensitive_term in ["GROQ_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY", "sk-ant-"]:
            assert sensitive_term not in ui_res.text

        # Check health endpoint
        health_res = await client.get("/api/health")
        for sensitive_term in ["GROQ_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]:
            assert sensitive_term not in health_res.text


# ==============================================================================
# 7. Worker Node Granular Lifecycle & Live Event Tests
# ==============================================================================


@pytest.mark.anyio
async def test_node_started_and_completed_lifecycle(test_app):
    """Verify that node_started and node_completed events are emitted for supervisor and workers."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        init_res = await client.post("/api/research", json={"question": "Testing node lifecycle"})
        tid = init_res.json()["thread_id"]

        events = []
        async with client.stream("GET", f"/api/research/{tid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    evt = json.loads(line[6:])
                    events.append(evt)
                    if evt["event_type"] == "human_review_required":
                        break

        event_types = [e["event_type"] for e in events]
        nodes = [e.get("node") for e in events]

        # Verify node_started and node_completed are present
        assert "node_started" in event_types
        assert "node_completed" in event_types
        assert "supervisor" in nodes
        assert "verifier" in nodes


@pytest.mark.anyio
async def test_subgraph_live_event_emitter_updates():
    """Verify that emit_live_event dispatches analysis_update, critic_update, and retry events."""
    from verified_research.api.events import emit_live_event, set_execution_event_emitter

    emitted = []

    def mock_emitter(event_type, node, data):
        emitted.append((event_type, node, data))

    set_execution_event_emitter(mock_emitter)
    try:
        emit_live_event("analysis_update", "analyst", {"claims_count": 3})
        emit_live_event("critic_update", "critic", {"quality_score": 9})
        emit_live_event("retry", "researcher", {"attempt": 1, "delay": 2.0})

        assert len(emitted) == 3
        assert emitted[0][0] == "analysis_update"
        assert emitted[0][1] == "analyst"
        assert emitted[1][0] == "critic_update"
        assert emitted[1][1] == "critic"
        assert emitted[2][0] == "retry"
        assert emitted[2][1] == "researcher"
    finally:
        set_execution_event_emitter(None)


@pytest.mark.anyio
async def test_unknown_thread_id_returns_404(test_app):
    """Querying a non-existent thread returns 404 with a helpful message."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/api/research/research_nonexistent_99999")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

