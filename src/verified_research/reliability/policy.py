"""Retry policy and execution runner with bounded exponential backoff and jitter."""

import logging
import os
import random
import time
from typing import Callable, TypeVar
from verified_research.reliability.classification import classify_error, sanitize_error_message
from verified_research.reliability.models import (
    ErrorCategory,
    ErrorInfo,
    MaxRetriesExceededError,
    NonRetryableError,
    OperationMetadata,
    ReliabilityError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 60.0
DEFAULT_RETRY_BACKOFF_FACTOR = 2.0
DEFAULT_RETRY_JITTER = True
DEFAULT_RETRY_JITTER_FACTOR = 0.5


def _default_sleep(delay: float) -> None:
    """Default sleep function that avoids real waiting during automated pytest runs."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    time.sleep(delay)


class RetryPolicy:
    """Centralized retry policy supporting bounded exponential backoff, jitter, and injected sleep."""

    def __init__(
        self,
        max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
        base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        max_delay: float = DEFAULT_RETRY_MAX_DELAY,
        backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
        jitter: bool = DEFAULT_RETRY_JITTER,
        jitter_factor: float = DEFAULT_RETRY_JITTER_FACTOR,
        sleep_fn: Callable[[float], None] | None = None,
        random_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        """Initialize the RetryPolicy.

        Args:
            max_attempts: Maximum number of execution attempts (minimum 1).
            base_delay: Initial delay in seconds for the first retry.
            max_delay: Upper bound in seconds for any calculated delay.
            backoff_factor: Multiplier for exponential backoff (e.g., 2.0).
            jitter: Whether to add random jitter to delays.
            jitter_factor: Proportion of delay to vary with jitter (0.0 to 1.0).
            sleep_fn: Callable executing sleep (defaults to time.sleep; inject for deterministic tests).
            random_fn: Callable generating random floats (defaults to random.uniform; inject for tests).
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if max_delay < base_delay:
            raise ValueError("max_delay cannot be less than base_delay")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be at least 1.0")

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        self.sleep_fn = sleep_fn or _default_sleep
        self.random_fn = random_fn or random.uniform

    def calculate_delay(self, attempt: int) -> float:
        """Calculate backoff delay for the given attempt index (1-indexed).

        Formula:
            raw_delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))
            with jitter: bounded within [raw_delay * (1 - jitter_factor), min(max_delay, raw_delay * (1 + jitter_factor))]

        Args:
            attempt: Attempt count (1 for the first failure before retry 1, 2 for second, etc.).

        Returns:
            Delay in seconds.
        """
        raw_delay = min(
            self.max_delay,
            self.base_delay * (self.backoff_factor ** max(0, attempt - 1)),
        )

        if not self.jitter:
            return raw_delay

        low = max(0.0, raw_delay * (1.0 - self.jitter_factor))
        high = min(self.max_delay, raw_delay * (1.0 + self.jitter_factor))
        return self.random_fn(low, high)

    def sleep(self, delay: float) -> None:
        """Execute sleep using the configured sleep function."""
        if delay > 0:
            self.sleep_fn(delay)


def execute_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy | None = None,
    component: str = "unknown",
    operation_name: str = "operation",
    custom_classifier: Callable[[Exception], tuple[ErrorCategory, bool, int | None]] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    on_failure: Callable[[ErrorInfo], None] | None = None,
    reraise_original: bool = False,
) -> T:
    """Execute an external operation with bounded exponential backoff retries and error classification.

    Args:
        operation: Zero-argument callable performing the operation.
        policy: Configured RetryPolicy (defaults to default RetryPolicy).
        component: Architectural component identifier ('search', 'analyst', 'verifier', 'critic').
        operation_name: Diagnostic operation description ('tavily_search', 'llm_synthesis').
        custom_classifier: Optional custom classification function.
        on_retry: Callback invoked before sleeping for retry: on_retry(attempt, exception, delay).
        on_failure: Callback invoked when operation fails permanently or exhausts retries.
        reraise_original: If True, re-raises original exception with .error_info attached;
                          if False, raises typed ReliabilityError subclass (NonRetryableError / MaxRetriesExceededError).

    Returns:
        The return value of operation().

    Raises:
        NonRetryableError: When failure is permanent / non-retryable and reraise_original is False.
        MaxRetriesExceededError: When retryable failure exhausts max attempts and reraise_original is False.
        Exception: Original exception with .error_info attached when reraise_original is True.
    """
    active_policy = policy or RetryPolicy()
    classifier = custom_classifier or classify_error
    attempts = 0
    start_time = time.monotonic()

    while True:
        attempts += 1
        try:
            result = operation()
            duration = time.monotonic() - start_time
            logger.debug(
                "[%s:%s] Succeeded on attempt %d (duration=%.3fs)",
                component,
                operation_name,
                attempts,
                duration,
            )
            try:
                from verified_research.observability.tracer import record_metadata
                record_metadata(
                    component=component,
                    operation=operation_name,
                    attempt_number=attempts,
                    retry_count=attempts - 1,
                    final_status="success",
                    duration_seconds=round(duration, 3),
                )
            except Exception:
                pass
            return result
        except Exception as exc:
            category, is_retryable, status_code = classifier(exc)
            sanitized_msg = sanitize_error_message(str(exc))

            # Check if this failure is non-retryable or if we reached maximum attempts
            is_terminal = (not is_retryable) or (attempts >= active_policy.max_attempts)

            if is_terminal:
                duration = time.monotonic() - start_time
                error_info = ErrorInfo(
                    component=component,
                    operation=operation_name,
                    error_type=type(exc).__name__,
                    category=category,
                    message=sanitized_msg,
                    retryable=is_retryable,
                    attempts=attempts,
                    status_code=status_code,
                    details={
                        "duration_seconds": f"{duration:.3f}",
                        "terminal_reason": (
                            "non_retryable_error"
                            if not is_retryable
                            else "max_attempts_exceeded"
                        ),
                    },
                )

                # Attach error_info to the caught exception for inspection
                try:
                    setattr(exc, "error_info", error_info)
                except Exception:
                    pass

                if on_failure:
                    try:
                        on_failure(error_info)
                    except Exception as cb_err:
                        logger.warning("[%s:%s] on_failure callback failed: %s", component, operation_name, cb_err)

                try:
                    from verified_research.observability.tracer import record_metadata
                    record_metadata(
                        component=component,
                        operation=operation_name,
                        attempt_number=attempts,
                        retry_count=attempts - 1,
                        final_status="failure",
                        error_category=category.value,
                        error_type=type(exc).__name__,
                        duration_seconds=round(duration, 3),
                    )
                except Exception:
                    pass

                logger.error(
                    "[%s:%s] Permanent failure after %d attempt(s) [category=%s, retryable=%s]: %s",
                    component,
                    operation_name,
                    attempts,
                    category.value,
                    is_retryable,
                    sanitized_msg,
                )

                if reraise_original:
                    raise exc

                if not is_retryable:
                    raise NonRetryableError(
                        f"Non-retryable failure in {component}.{operation_name}: {sanitized_msg}",
                        error_info=error_info,
                    ) from exc

                raise MaxRetriesExceededError(
                    f"Exhausted {active_policy.max_attempts} attempts for {component}.{operation_name}: {sanitized_msg}",
                    error_info=error_info,
                ) from exc

            # Failure is retryable and attempts remain
            delay = active_policy.calculate_delay(attempts)
            try:
                from verified_research.observability.tracer import record_event
                record_event(
                    "retry_attempt",
                    component=component,
                    operation=operation_name,
                    attempt=attempts,
                    delay_seconds=round(delay, 3),
                    error_category=category.value,
                    error_type=type(exc).__name__,
                )
            except Exception:
                pass

            logger.warning(
                "[%s:%s] Attempt %d/%d failed with %s (%s). Retrying in %.2fs...",
                component,
                operation_name,
                attempts,
                active_policy.max_attempts,
                type(exc).__name__,
                sanitized_msg,
                delay,
            )

            if on_retry:
                try:
                    on_retry(attempts, exc, delay)
                except Exception as cb_err:
                    logger.warning("[%s:%s] on_retry callback failed: %s", component, operation_name, cb_err)

            active_policy.sleep(delay)
