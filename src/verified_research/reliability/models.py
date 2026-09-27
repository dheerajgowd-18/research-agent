"""Reliability data models and exceptions for the Verified Research Agent."""

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ErrorCategory(str, Enum):
    """Normalized classification categories for operational failures."""

    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    AUTH_ERROR = "auth_error"
    MALFORMED_OUTPUT = "malformed_output"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class ErrorInfo(BaseModel):
    """Structured, serializable error representation for reliability audit and checkpoint persistence.

    Guarantees:
        - Fully serializable across persistence checkpoints (no raw Exception objects).
        - Sanitized error message preventing credential leaks.
        - Immutable (frozen) with extra fields forbidden.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str = Field(
        ...,
        description="The system component where the failure occurred (e.g., 'search', 'analyst', 'verifier').",
    )
    operation: str = Field(
        ...,
        description="The specific operation attempted (e.g., 'tavily_search', 'llm_synthesis', 'llm_verification').",
    )
    error_type: str = Field(
        ...,
        description="The exception class name (e.g., 'TimeoutError', 'HTTPError', 'ValidationError').",
    )
    category: ErrorCategory = Field(
        ...,
        description="Normalized failure classification category.",
    )
    message: str = Field(
        ...,
        description="Sanitized error description with credentials and tokens redacted.",
    )
    retryable: bool = Field(
        ...,
        description="Whether this failure was classified as retryable under the active policy.",
    )
    attempts: int = Field(
        ...,
        ge=1,
        description="Total execution attempts made before success or termination.",
    )
    status_code: int | None = Field(
        default=None,
        description="Associated HTTP status code if applicable.",
    )
    details: dict[str, str] = Field(
        default_factory=dict,
        description="Additional serializable diagnostic metadata.",
    )


class OperationMetadata(BaseModel):
    """Execution telemetry recorded for an external operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: str = Field(..., description="Component name.")
    operation: str = Field(..., description="Operation name.")
    attempts: int = Field(..., ge=1, description="Total attempts executed.")
    status: Literal["success", "failure"] = Field(..., description="Terminal status of operation.")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Elapsed execution time.")
    error_info: ErrorInfo | None = Field(default=None, description="Failure details if status is failure.")


class ReliabilityError(RuntimeError):
    """Base exception for reliability layer failures."""

    def __init__(self, message: str, error_info: ErrorInfo | None = None) -> None:
        super().__init__(message)
        self.error_info = error_info


class NonRetryableError(ReliabilityError):
    """Raised when an operation fails with a permanent, non-retryable error."""


class MaxRetriesExceededError(ReliabilityError):
    """Raised when an operation fails after exhausting all allowed retry attempts."""
