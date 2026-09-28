"""Pydantic schemas and data contracts for end-to-end system evaluation."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExpectedProperties(BaseModel):
    """Declarative expectations for an evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_claims: int = Field(default=1, ge=0)
    min_sources: int = Field(default=1, ge=0)
    expected_verdicts: list[str] = Field(default_factory=list)
    hitl_action: Literal["approve", "edit", "research_more", "reject"] = Field(default="approve")
    is_follow_up: bool = Field(default=False)
    follow_up_to: str | None = Field(default=None)


class SystemEvalCase(BaseModel):
    """A single end-to-end system evaluation task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(description="Stable evaluation case identifier (e.g. sys_eval_001).")
    category: str = Field(description="Task category / behavioral characteristic.")
    question: str = Field(description="Input research query.")
    description: str = Field(description="Operational rationale for this evaluation case.")
    expected_properties: ExpectedProperties = Field(description="Evaluation expectations.")

    @field_validator("case_id", "category", "question")
    @classmethod
    def validate_non_empty(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"Field '{info.field_name}' must be a non-empty string.")
        return value.strip()


class CaseExecutionResult(BaseModel):
    """Observed empirical result for an individual evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(description="Case identifier.")
    question: str = Field(description="Research query.")
    category: str = Field(description="Task category.")
    status: str = Field(description="Terminal status: completed, rejected, or failed.")
    duration_seconds: float = Field(ge=0.0, description="End-to-end execution duration in seconds.")
    supervisor_steps: int = Field(ge=0, description="Total supervisor transitions.")
    research_iteration: int = Field(ge=0, description="Research cycle iterations.")
    human_research_cycles: int = Field(ge=0, description="Human review cycles.")
    claims_count: int = Field(ge=0, description="Total claims extracted.")
    evidence_count: int = Field(ge=0, description="Total evidence excerpts preserved.")
    sources_count: int = Field(ge=0, description="Total sources cited.")
    distinct_sources_count: int = Field(ge=0, description="Distinct sources cited.")
    claims_with_valid_evidence: int = Field(ge=0, description="Claims referencing valid evidence.")
    citation_coverage: float = Field(ge=0.0, le=1.0, description="Proportion of claims with valid evidence.")
    verification_coverage: float = Field(ge=0.0, le=1.0, description="Proportion of claims with verifications.")
    verdicts_summary: dict[str, int] = Field(description="Count of SUPPORTED, PARTIAL, and UNSUPPORTED verdicts.")
    supervisor_decisions: list[str] = Field(description="Sequence of worker routing choices.")
    supervisor_termination_reason: str | None = Field(default=None, description="Reason for finish.")
    hitl_action: str | None = Field(default=None, description="Human review action applied.")
    edited_claims_count: int = Field(default=0, ge=0, description="Number of edited claims if action was edit.")
    retries_count: int = Field(default=0, ge=0, description="Number of transient retries.")
    llm_calls_estimate: int = Field(default=0, ge=0, description="Estimated total LLM invocations.")
    search_calls_estimate: int = Field(default=0, ge=0, description="Estimated total search API queries.")
    error: str | None = Field(default=None, description="Sanitized error description if failed.")
    timestamp: str = Field(description="ISO 8601 execution timestamp.")


class ReliabilitySummary(BaseModel):
    """Reliability metrics across all evaluated runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_runs: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    rejected_runs: int = Field(ge=0)
    retry_triggering_failures: int = Field(ge=0)
    recovered_transient_failures: int = Field(ge=0)
    permanent_failures: int = Field(ge=0)
    total_retries: int = Field(ge=0)


class SystemEvaluationSummary(BaseModel):
    """Aggregated empirical benchmark metrics across the full evaluation dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_timestamp: str = Field(description="Execution start timestamp.")
    evaluation_environment: str = Field(description="Environment: live or deterministic_mock.")
    total_cases: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    rejected_runs: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    total_evidence: int = Field(ge=0)
    total_sources: int = Field(ge=0)
    mean_citation_coverage: float = Field(ge=0.0, le=1.0)
    mean_verification_coverage: float = Field(ge=0.0, le=1.0)
    mean_latency_seconds: float = Field(ge=0.0)
    median_latency_seconds: float = Field(ge=0.0)
    min_latency_seconds: float = Field(ge=0.0)
    max_latency_seconds: float = Field(ge=0.0)
    mean_supervisor_steps: float = Field(ge=0.0)
    mean_research_iterations: float = Field(ge=0.0)
    verdicts_breakdown: dict[str, int]
    hitl_actions_breakdown: dict[str, int]
    hitl_approval_rate: float = Field(ge=0.0, le=1.0)
    follow_up_stats: dict[str, Any]
    reliability_stats: ReliabilitySummary
    results: list[CaseExecutionResult]
