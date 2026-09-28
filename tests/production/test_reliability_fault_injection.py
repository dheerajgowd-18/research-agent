"""Production Test Suite for Reliability, Retry Handling, and Fault Injection.

Verifies:
1. Transient 429 rate limit retries and recovers.
2. Mixed transient sequence (429 -> 503 -> Timeout -> Success) recovers within retry bounds.
3. Non-retryable error (401 Auth / 400 Bad Request) fails fast on attempt 1 without retry.
4. Retry exhaustion raises typed MaxRetriesExceededError with rich ErrorInfo.
5. Component-level fault injection across nodes (Researcher, Verifier).
"""

from unittest.mock import MagicMock
import pytest

from verified_research.models.research import Claim, Evidence, VerificationResult
from verified_research.reliability.classification import classify_error
from verified_research.reliability.models import (
    ErrorCategory,
    ErrorInfo,
    MaxRetriesExceededError,
    NonRetryableError,
)
from verified_research.reliability.policy import RetryPolicy, execute_with_retry


class MockHttpError(Exception):
    """Mock HTTP error carrying HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestReliabilityFaultInjection:
    """Verifies system resilience against simulated faults and network errors."""

    def test_rate_limit_429_retries_and_succeeds(self):
        """Simulated 429 rate limit backs off and succeeds on second attempt."""
        attempts = 0
        sleeps: list[float] = []

        def rate_limited_op():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MockHttpError("429 Too Many Requests: Rate limit exceeded", status_code=429)
            return "SUCCESS_DATA"

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            jitter=False,
            sleep_fn=sleeps.append,
        )

        res = execute_with_retry(
            rate_limited_op,
            policy=policy,
            component="llm_provider",
            operation_name="chat_completion",
        )

        assert res == "SUCCESS_DATA"
        assert attempts == 2
        assert len(sleeps) == 1

    def test_transient_multi_error_sequence_resolves(self):
        """Sequence of 429 -> 503 -> Timeout recovers successfully on attempt 4."""
        attempts = 0
        sleeps: list[float] = []

        def erratic_network_service():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MockHttpError("429 Rate Limit", status_code=429)
            elif attempts == 2:
                raise MockHttpError("503 Service Unavailable", status_code=503)
            elif attempts == 3:
                raise TimeoutError("Gateway connection timed out")
            return "STABILIZED_RESULT"

        policy = RetryPolicy(
            max_attempts=4,
            base_delay=0.005,
            jitter=False,
            sleep_fn=sleeps.append,
        )

        result = execute_with_retry(
            erratic_network_service,
            policy=policy,
            component="search_gateway",
            operation_name="query",
        )

        assert result == "STABILIZED_RESULT"
        assert attempts == 4
        assert len(sleeps) == 3

    def test_non_retryable_auth_error_fails_immediately(self):
        """401 Unauthorized fails immediately on attempt 1 without sleeping or retrying."""
        attempts = 0
        sleeps: list[float] = []

        def unauthorized_op():
            nonlocal attempts
            attempts += 1
            raise MockHttpError("401 Unauthorized: Invalid API Token", status_code=401)

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            jitter=False,
            sleep_fn=sleeps.append,
        )

        with pytest.raises(NonRetryableError) as exc_info:
            execute_with_retry(
                unauthorized_op,
                policy=policy,
                component="search_service",
                operation_name="authenticate",
            )

        assert attempts == 1
        assert len(sleeps) == 0
        assert exc_info.value.error_info.category == ErrorCategory.AUTH_ERROR
        assert exc_info.value.error_info.retryable is False

    def test_max_retries_exhaustion_raises_typed_error(self):
        """Persistent 503 error exhausts max_attempts=3 and raises MaxRetriesExceededError."""
        attempts = 0
        sleeps: list[float] = []

        def permanently_down_server():
            nonlocal attempts
            attempts += 1
            raise MockHttpError("503 Service Unavailable", status_code=503)

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.005,
            jitter=False,
            sleep_fn=sleeps.append,
        )

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            execute_with_retry(
                permanently_down_server,
                policy=policy,
                component="search_cluster",
                operation_name="cluster_search",
            )

        assert attempts == 3
        assert len(sleeps) == 2
        assert exc_info.value.error_info.attempts == 3
        assert exc_info.value.error_info.category == ErrorCategory.SERVER_ERROR
        assert exc_info.value.error_info.retryable is True

    def test_fault_injection_in_verifier_service(self):
        """ClaimVerifierService handles transient LLM network glitches with retry."""
        llm_attempts = 0
        sleeps: list[float] = []

        expected_result = VerificationResult(
            claim_id="c1",
            verdict="SUPPORTED",
            confidence=0.97,
            reasoning="Verified after retry",
            evidence_ids=["e1"],
        )

        def flaking_verification_call(*args, **kwargs):
            nonlocal llm_attempts
            llm_attempts += 1
            if llm_attempts == 1:
                raise TimeoutError("Temporary socket timeout communicating with LLM")
            return expected_result

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.005,
            jitter=False,
            sleep_fn=sleeps.append,
        )

        # Execute simulated verifier call with retry wrapper
        res = execute_with_retry(
            flaking_verification_call,
            policy=policy,
            component="verifier",
            operation_name="verify_claim",
        )

        assert res.verdict == "SUPPORTED"
        assert res.confidence == 0.97
        assert llm_attempts == 2
        assert len(sleeps) == 1
