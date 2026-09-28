"""Data models and typed event contracts for the Verified Research API and UI boundary."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# UI and Execution State Model
UIStatusType = Literal[
    "idle",
    "running",
    "waiting_for_human",
    "resuming",
    "completed",
    "failed",
    "rejected",
]

# Structured Event Types for SSE Streaming
AgentEventType = Literal[
    "run_started",
    "node_started",
    "node_completed",
    "supervisor_decision",
    "research_update",
    "source_found",
    "verification_update",
    "human_review_required",
    "human_review_resumed",
    "run_completed",
    "run_failed",
    "run_rejected",
]


class AgentEvent(BaseModel):
    """Typed streaming event emitted by the backend to SSE listeners."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: AgentEventType = Field(
        ...,
        description="Categorical type of agent lifecycle event.",
    )
    thread_id: str = Field(
        ...,
        description="Persistent thread identifier associated with this execution.",
    )
    node: str | None = Field(
        default=None,
        description="Graph node responsible for this event (e.g. 'supervisor', 'research', 'verifier').",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of event generation.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload containing operational data, metrics, or review items.",
    )


class StartResearchRequest(BaseModel):
    """Request payload to initiate a new research run."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        description="The research question or topic to investigate.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID. If omitted, a unique thread ID is automatically generated.",
    )


class StartResearchResponse(BaseModel):
    """Immediate response after initiating a research run."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., description="Assigned thread ID for the research execution.")
    status: UIStatusType = Field(default="running", description="Initial lifecycle state.")
    message: str = Field(default="Research started successfully.", description="Status message.")


class ResumeReviewRequest(BaseModel):
    """Request payload submitted by a human reviewer to resolve an interrupt."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "research_more", "reject"] = Field(
        ...,
        description="Human review decision action.",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional textual feedback or guidance for additional research passes.",
    )
    edited_claims: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of edited Claim structures required when action is 'edit'.",
    )


class ResearchStateResponse(BaseModel):
    """Complete snapshot of research state returned for status polling and session reconnection."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., description="Thread identifier.")
    status: UIStatusType = Field(..., description="Current UI lifecycle state.")
    question: str = Field(default="", description="Research question.")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Retrieved sources.")
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="Preserved evidence excerpts.")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="Synthesized findings.")
    claims: list[dict[str, Any]] = Field(default_factory=list, description="Extracted factual claims.")
    verification_results: list[dict[str, Any]] = Field(default_factory=list, description="Verification verdicts.")
    human_review: dict[str, Any] | None = Field(default=None, description="Human review decision record.")
    supervisor_decision: dict[str, Any] | None = Field(default=None, description="Latest supervisor decision.")
    supervisor_termination_reason: str | None = Field(default=None, description="Terminal status reason.")
    supervisor_steps: int = Field(default=0, description="Cumulative supervisor steps.")
    research_iteration: int = Field(default=0, description="Research cycle iteration.")
    human_research_cycles: int = Field(default=0, description="Human research cycle count.")
    review_context: dict[str, Any] | None = Field(
        default=None,
        description="Active interrupt payload containing review items if waiting for human review.",
    )
    error: str | None = Field(default=None, description="Sanitized error description if failed.")
