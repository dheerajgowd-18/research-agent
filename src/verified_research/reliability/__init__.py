"""Reliability and fault tolerance layer for external operations."""

from verified_research.reliability.classification import (
    classify_error,
    extract_status_code,
    sanitize_error_message,
)
from verified_research.reliability.models import (
    ErrorCategory,
    ErrorInfo,
    MaxRetriesExceededError,
    NonRetryableError,
    OperationMetadata,
    ReliabilityError,
)
from verified_research.reliability.policy import (
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_JITTER,
    DEFAULT_RETRY_JITTER_FACTOR,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_MAX_DELAY,
    RetryPolicy,
    execute_with_retry,
)

__all__ = [
    "ErrorCategory",
    "ErrorInfo",
    "OperationMetadata",
    "ReliabilityError",
    "NonRetryableError",
    "MaxRetriesExceededError",
    "classify_error",
    "extract_status_code",
    "sanitize_error_message",
    "RetryPolicy",
    "execute_with_retry",
    "DEFAULT_RETRY_MAX_ATTEMPTS",
    "DEFAULT_RETRY_BASE_DELAY",
    "DEFAULT_RETRY_MAX_DELAY",
    "DEFAULT_RETRY_BACKOFF_FACTOR",
    "DEFAULT_RETRY_JITTER",
    "DEFAULT_RETRY_JITTER_FACTOR",
]
