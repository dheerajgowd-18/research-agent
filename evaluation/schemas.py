"""Pydantic schemas and constants for verifier evaluation."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from verified_research.models.research import VerdictType

# Canonical label ordering used consistently across all evaluation components
CLASS_LABELS: tuple[VerdictType, ...] = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")


class EvalExample(BaseModel):
    """A single hand-labeled evaluation example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique evaluation example identifier (e.g. eval_001).")
    category: str = Field(description="Evaluation category (e.g. direct_support, negation).")
    claim: str = Field(description="Factual claim proposition to evaluate.")
    evidence: str = Field(description="Textual evidence excerpt supplied to verifier.")
    expected_verdict: VerdictType = Field(description="Ground truth verdict label.")

    @field_validator("id", "category", "claim", "evidence")
    @classmethod
    def validate_non_empty(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"Field '{info.field_name}' must be a non-empty string.")
        return value.strip()


class PredictionRecord(BaseModel):
    """Evaluation result for an individual example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Example ID.")
    category: str = Field(description="Category name.")
    claim: str = Field(description="Claim text.")
    evidence: str = Field(description="Evidence text.")
    expected: VerdictType = Field(description="Ground truth expected verdict.")
    predicted: VerdictType = Field(description="Model predicted verdict.")
    correct: bool = Field(description="Whether prediction matches ground truth.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model-reported confidence score.")
    reasoning: str = Field(description="Verifier reasoning.")
    evidence_ids: list[str] = Field(description="Evidence IDs associated with the claim.")


class ClassMetric(BaseModel):
    """Metrics for a single class label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_labels: list[str] = Field(
        default_factory=lambda: list(CLASS_LABELS),
        description="Fixed canonical class label ordering.",
    )
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_class: dict[str, ClassMetric] = Field(
        description="Per-class precision, recall, and F1 scores keyed by label."
    )
    confusion_matrix: list[list[int]] = Field(
        description="3x3 confusion matrix where rows=expected and columns=predicted."
    )
