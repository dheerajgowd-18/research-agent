"""FastAPI router endpoints for research lifecycle, state query, SSE streaming, and HITL resume."""

import json
import logging
from typing import AsyncIterator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from verified_research.api.models import (
    ResearchStateResponse,
    ResumeReviewRequest,
    StartResearchRequest,
    StartResearchResponse,
)
from verified_research.api.service import ResearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["Research"])


def get_research_service(request: Request) -> ResearchService:
    """Dependency injection helper retrieving the ResearchService from FastAPI app state."""
    return request.app.state.research_service


@router.post("", response_model=StartResearchResponse, status_code=202)
async def start_research(
    request_body: StartResearchRequest,
    service: ResearchService = Depends(get_research_service),
) -> StartResearchResponse:
    """Initiate a research investigation. Returns thread ID for streaming and reconnection."""
    thread_id = await service.start_research(
        question=request_body.question,
        thread_id=request_body.thread_id,
    )
    return StartResearchResponse(
        thread_id=thread_id,
        status="running",
        message="Research started successfully.",
    )


@router.get("/{thread_id}", response_model=ResearchStateResponse)
async def get_research_state(
    thread_id: str,
    service: ResearchService = Depends(get_research_service),
) -> ResearchStateResponse:
    """Retrieve full state snapshot for a given thread (supports session reconnection and polling)."""
    return await service.get_state(thread_id)


@router.get("/{thread_id}/stream")
async def stream_research_events(
    thread_id: str,
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    """Stream real-time AgentEvents via Server-Sent Events (SSE). Replays historical events first."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in service.stream_events(thread_id):
                # Standard SSE data framing with serialized AgentEvent JSON
                yield f"data: {event.model_dump_json()}\n\n"
        except Exception as exc:
            logger.error("SSE stream error for thread %s: %s", thread_id, exc)
            err_payload = json.dumps({"event_type": "run_failed", "thread_id": thread_id, "data": {"error": str(exc)}})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{thread_id}/resume", status_code=202)
async def resume_human_review(
    thread_id: str,
    resume_body: ResumeReviewRequest,
    service: ResearchService = Depends(get_research_service),
) -> dict[str, str]:
    """Resume execution of an interrupted thread with an authoritative human review action."""
    await service.resume_human_review(thread_id, resume_body)
    return {
        "thread_id": thread_id,
        "status": "resuming",
        "message": f"Resumed research with action '{resume_body.action}'.",
    }
