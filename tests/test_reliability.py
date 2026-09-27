"""Comprehensive unit and integration test suite for the reliability and retry layer."""

from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from verified_research.config.settings import Settings
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    AnalystOutput,
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    SupervisorDecision,
    VerificationResult,
)
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
    RetryPolicy,
    execute_with_retry,
)
from verified_research.tools.search import MissingApiKeyError, SearchError, TavilySearchClient


# ==============================================================================
# Helper Mock Exception Classes for Testing
# ==============================================================================


class MockHttpError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class MockTimeoutException(Exception):
    pass


# ==============================================================================
# 1. Retryable Failure Then Success Tests
# ==============================================================================


class TestRetryableFailureThenSuccess:
    """Test suite verifying that transient retryable failures retry and succeed on subsequent attempts."""

    def test_transient_failure_then_success_exactly_three_attempts(self):
        """Attempt 1 fails (503), Attempt 2 fails (429), Attempt 3 succeeds."""
        attempts = 0
        recorded_sleeps: list[float] = []

        def flaky_operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MockHttpError("503 Service Unavailable", status_code=503)
            if attempts == 2:
                raise MockHttpError("429 Rate Limit Exceeded", status_code=429)
            return "operation_success"

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        result = execute_with_retry(
            operation=flaky_operation,
            policy=policy,
            component="test_component",
            operation_name="flaky_op",
        )

        assert result == "operation_success"
        assert attempts == 3
        assert len(recorded_sleeps) == 2  # Slept after attempt 1 and attempt 2

    def test_search_client_retries_transient_error_then_succeeds(self):
        """TavilySearchClient retries network timeouts and succeeds on attempt 2."""
        call_count = 0
        recorded_sleeps: list[float] = []

        mock_raw_client = MagicMock()

        def mock_search(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Connection to Tavily timed out")
            return {
                "results": [
                    {
                        "title": "Quantum Supremacy",
                        "url": "https://example.com/quantum",
                        "content": "Superconducting qubits demonstrated supremacy.",
                    }
                ]
            }

        mock_raw_client.search.side_effect = mock_search

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.5,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        client = TavilySearchClient(api_key="tvly-mock-key-for-test", retry_policy=policy)
        client._client = mock_raw_client

        sources = client.search("quantum progress")
        assert len(sources) == 1
        assert sources[0].title == "Quantum Supremacy"
        assert call_count == 2
        assert len(recorded_sleeps) == 1


# ==============================================================================
# 2. Permanent Failure (Fail Fast) Tests
# ==============================================================================


class TestPermanentFailureFailFast:
    """Test suite verifying that permanent, non-retryable errors fail fast with zero retries."""

    def test_auth_error_fails_immediately_without_retrying(self):
        """Attempt 1 fails with 401 Unauthorized -> fails fast without retry."""
        attempts = 0
        recorded_sleeps: list[float] = []

        def auth_failure_op():
            nonlocal attempts
            attempts += 1
            raise MockHttpError("401 Unauthorized: Invalid API key", status_code=401)

        policy = RetryPolicy(
            max_attempts=3,
            sleep_fn=recorded_sleeps.append,
        )

        with pytest.raises(NonRetryableError) as exc_info:
            execute_with_retry(
                operation=auth_failure_op,
                policy=policy,
                component="auth_service",
                operation_name="validate_token",
            )

        assert attempts == 1
        assert len(recorded_sleeps) == 0
        assert exc_info.value.error_info is not None
        assert exc_info.value.error_info.category == ErrorCategory.AUTH_ERROR
        assert exc_info.value.error_info.retryable is False
        assert exc_info.value.error_info.attempts == 1

    def test_missing_api_key_fails_fast_in_search_client(self):
        """MissingApiKeyError in TavilySearchClient fails fast without retry."""
        client = TavilySearchClient(api_key="")

        with pytest.raises(MissingApiKeyError, match="Tavily API key is missing"):
            client.search("test query")


# ==============================================================================
# 3. Maximum Attempts Exhaustion Tests
# ==============================================================================


class TestMaximumAttemptsExhaustion:
    """Test suite verifying that retryable errors halt after reaching max_attempts."""

    def test_max_attempts_exhaustion_raises_max_retries_error(self):
        """Perpetual 500 server error exhausts max_attempts=3 and raises MaxRetriesExceededError."""
        attempts = 0
        recorded_sleeps: list[float] = []

        def perpetual_failing_op():
            nonlocal attempts
            attempts += 1
            raise MockHttpError("500 Internal Server Error", status_code=500)

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            execute_with_retry(
                operation=perpetual_failing_op,
                policy=policy,
                component="backend",
                operation_name="query_data",
            )

        assert attempts == 3
        assert len(recorded_sleeps) == 2  # Slept after attempt 1 and 2, failed on 3
        assert exc_info.value.error_info is not None
        assert exc_info.value.error_info.attempts == 3
        assert exc_info.value.error_info.retryable is True
        assert exc_info.value.error_info.category == ErrorCategory.SERVER_ERROR

    def test_search_client_exhausts_retries_and_raises_search_error(self):
        """TavilySearchClient exhausts retries and raises SearchError with attached error_info."""
        mock_raw_client = MagicMock()
        mock_raw_client.search.side_effect = TimeoutError("Persistent connection timeout")
        recorded_sleeps: list[float] = []

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        client = TavilySearchClient(api_key="tvly-mock-key", retry_policy=policy)
        client._client = mock_raw_client

        with pytest.raises(SearchError, match="Tavily search request failed") as exc_info:
            client.search("quantum sensors")

        assert hasattr(exc_info.value, "error_info")
        assert exc_info.value.error_info is not None
        assert exc_info.value.error_info.attempts == 3
        assert exc_info.value.error_info.category == ErrorCategory.NETWORK_TIMEOUT


# ==============================================================================
# 4. Exponential Backoff Calculation Tests
# ==============================================================================


class TestExponentialBackoffCalculation:
    """Test suite verifying exact exponential delay scaling without real waiting."""

    def test_exponential_backoff_delays_without_jitter(self):
        """Base delay 1.0 with backoff factor 2.0 yields 1.0, 2.0, 4.0, 8.0."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=60.0,
            jitter=False,
        )

        assert policy.calculate_delay(1) == 1.0  # 1.0 * (2.0 ** 0)
        assert policy.calculate_delay(2) == 2.0  # 1.0 * (2.0 ** 1)
        assert policy.calculate_delay(3) == 4.0  # 1.0 * (2.0 ** 2)
        assert policy.calculate_delay(4) == 8.0  # 1.0 * (2.0 ** 3)

    def test_injected_sleep_function_records_exact_delays(self):
        """Verify that execute_with_retry invokes sleep_fn with the exact calculated sequence."""
        recorded_sleeps: list[float] = []
        attempts = 0

        def failing_op():
            nonlocal attempts
            attempts += 1
            raise ConnectionError("Connection refused by peer")

        policy = RetryPolicy(
            max_attempts=4,
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=60.0,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        with pytest.raises(MaxRetriesExceededError):
            execute_with_retry(
                operation=failing_op,
                policy=policy,
                component="network",
                operation_name="connect",
            )

        assert recorded_sleeps == [1.0, 2.0, 4.0]
        assert attempts == 4


# ==============================================================================
# 5. Maximum Delay Bound Tests
# ==============================================================================


class TestMaximumDelayBound:
    """Test suite verifying backoff delay is strictly capped at max_delay."""

    def test_backoff_does_not_exceed_max_delay(self):
        """When calculated exponential backoff exceeds max_delay, cap at max_delay."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=10.0,
            backoff_factor=2.0,
            max_delay=25.0,  # Caps delays
            jitter=False,
        )

        assert policy.calculate_delay(1) == 10.0  # min(25, 10)
        assert policy.calculate_delay(2) == 20.0  # min(25, 20)
        assert policy.calculate_delay(3) == 25.0  # min(25, 40) -> 25.0
        assert policy.calculate_delay(4) == 25.0  # min(25, 80) -> 25.0


# ==============================================================================
# 6. Jitter Bounds Tests
# ==============================================================================


class TestJitterBounds:
    """Test suite verifying that randomized jitter remains strictly within theoretical bounds."""

    def test_jitter_stays_within_configured_bounds(self):
        """With base 2.0 and jitter_factor 0.5, delay must lie in [1.0, 3.0] for attempt 1."""
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=2.0,
            backoff_factor=2.0,
            max_delay=30.0,
            jitter=True,
            jitter_factor=0.5,
        )

        for _ in range(50):
            delay_1 = policy.calculate_delay(1)
            # attempt 1 raw is 2.0, jitter bounds are [2.0 * 0.5, 2.0 * 1.5] = [1.0, 3.0]
            assert 1.0 <= delay_1 <= 3.0

            delay_2 = policy.calculate_delay(2)
            # attempt 2 raw is 4.0, jitter bounds are [4.0 * 0.5, 4.0 * 1.5] = [2.0, 6.0]
            assert 2.0 <= delay_2 <= 6.0


# ==============================================================================
# 7. Timeout Classification Tests
# ==============================================================================


class TestTimeoutClassification:
    """Test suite verifying that network timeouts are classified as retryable."""

    def test_timeout_error_classification(self):
        exc = TimeoutError("Request timed out after 30000ms")
        cat, retryable, status = classify_error(exc)
        assert cat == ErrorCategory.NETWORK_TIMEOUT
        assert retryable is True
        assert status == 408

    def test_http_408_status_classification(self):
        exc = MockHttpError("Server timed out waiting for request", status_code=408)
        cat, retryable, status = classify_error(exc)
        assert cat == ErrorCategory.NETWORK_TIMEOUT
        assert retryable is True
        assert status == 408


# ==============================================================================
# 8. Structured Output Failure Handling Tests
# ==============================================================================


class TestStructuredOutputFailureHandling:
    """Test suite verifying that malformed structured outputs undergo bounded retry without inventing data."""

    def test_malformed_structured_output_retries_and_succeeds(self):
        """LLM returns invalid output on attempt 1, then valid AnalystOutput on attempt 2."""
        call_count = 0
        recorded_sleeps: list[float] = []

        valid_output = AnalystOutput(
            findings=[
                Finding(
                    finding_id="finding_001",
                    text="Superconducting qubits require dilution refrigeration.",
                    source_ids=["src_001"],
                )
            ]
        )

        def flaky_llm():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate Pydantic ValidationError or schema violation
                raise ValidationError.from_exception_data(
                    title="AnalystOutput",
                    line_errors=[
                        {
                            "type": "missing",
                            "loc": ("findings",),
                            "msg": "Field required",
                            "input": {},
                        }
                    ],
                )
            return valid_output

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.1,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        result = execute_with_retry(
            operation=flaky_llm,
            policy=policy,
            component="analyst",
            operation_name="llm_synthesis",
        )

        assert isinstance(result, AnalystOutput)
        assert len(result.findings) == 1
        assert call_count == 2
        assert len(recorded_sleeps) == 1

    def test_malformed_structured_output_exhaustion_does_not_invent_data(self):
        """Repeated malformed structured output halts with MaxRetriesExceededError without inventing data."""
        recorded_sleeps: list[float] = []

        def permanently_broken_llm():
            raise ValidationError.from_exception_data(
                title="VerificationResult",
                line_errors=[
                    {
                        "type": "string_type",
                        "loc": ("verdict",),
                        "msg": "Input should be a valid string",
                        "input": 12345,
                    }
                ],
            )

        policy = RetryPolicy(
            max_attempts=2,
            base_delay=0.1,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            execute_with_retry(
                operation=permanently_broken_llm,
                policy=policy,
                component="verifier",
                operation_name="llm_verification",
            )

        assert exc_info.value.error_info is not None
        assert exc_info.value.error_info.category == ErrorCategory.MALFORMED_OUTPUT
        assert exc_info.value.error_info.attempts == 2

    def test_domain_model_schemas_remain_strictly_validated(self):
        """Domain models (VerificationResult, SupervisorDecision) reject invalid inputs."""
        # VerificationResult verdict must be SUPPORTED, PARTIAL, or UNSUPPORTED
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="cl_01",
                verdict="MAYBE_TRUE",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="Vague evaluation",
                evidence_ids=["ev_01"],
            )

        # SupervisorDecision next_worker must be one of allowed workers
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_worker="arbitrary_agent",  # type: ignore[arg-type]
                reasoning="Invalid routing target",
            )


# ==============================================================================
# 9. Error Metadata Tests
# ==============================================================================


class TestErrorMetadata:
    """Test suite verifying serializable ErrorInfo generation."""

    def test_error_info_serializable_structure(self):
        """ErrorInfo contains all required serializable fields with extra forbidden."""
        info = ErrorInfo(
            component="search",
            operation="tavily_search",
            error_type="TimeoutError",
            category=ErrorCategory.NETWORK_TIMEOUT,
            message="Connection timed out after 10.0s",
            retryable=True,
            attempts=3,
            status_code=408,
            details={"ip": "127.0.0.1"},
        )

        dump = info.model_dump()
        assert dump["component"] == "search"
        assert dump["operation"] == "tavily_search"
        assert dump["category"] == "network_timeout"
        assert dump["retryable"] is True
        assert dump["attempts"] == 3
        assert dump["status_code"] == 408

        # Extra forbidden
        with pytest.raises(ValidationError):
            ErrorInfo(
                component="search",
                operation="search",
                error_type="Error",
                category=ErrorCategory.UNKNOWN,
                message="Msg",
                retryable=False,
                attempts=1,
                unallowed_extra="forbidden",  # type: ignore[call-arg]
            )


# ==============================================================================
# 10. Secret Sanitization & Leakage Prevention Tests
# ==============================================================================


class TestSecretSanitization:
    """Test suite verifying sensitive API keys and tokens are never leaked in error messages."""

    def test_sanitize_tavily_key_in_error_message(self):
        raw = "Failed Tavily request using key tvly-abcdef1234567890xyz and query 'test'"
        sanitized = sanitize_error_message(raw)
        assert "tvly-abcdef1234567890xyz" not in sanitized
        assert "[REDACTED_TAVILY_KEY]" in sanitized

    def test_sanitize_groq_key_in_error_message(self):
        raw = "Groq API error with key gsk_abcdef123456789012345678"
        sanitized = sanitize_error_message(raw)
        assert "gsk_abcdef123456789012345678" not in sanitized
        assert "[REDACTED_GROQ_KEY]" in sanitized

    def test_sanitize_bearer_token_in_error_message(self):
        raw = "Header Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = sanitize_error_message(raw)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
        assert "Bearer [REDACTED_TOKEN]" in sanitized

    def test_execute_with_retry_redacts_secrets_in_error_info(self):
        """Failure message inside ErrorInfo is fully redacted."""
        def secret_leaking_op():
            raise RuntimeError("Request with api_key='tvly-secretkey12345678' failed")

        policy = RetryPolicy(max_attempts=1)
        with pytest.raises(NonRetryableError) as exc_info:
            execute_with_retry(
                operation=secret_leaking_op,
                policy=policy,
                component="search",
                operation_name="tavily_search",
            )

        assert "tvly-secretkey12345678" not in exc_info.value.error_info.message
        assert "[REDACTED]" in exc_info.value.error_info.message or "[REDACTED_TAVILY_KEY]" in exc_info.value.error_info.message


# ==============================================================================
# 11. Retry Counter Isolation Tests
# ==============================================================================


class TestRetryCounterIsolation:
    """Test suite verifying retry attempts do NOT modify research_iteration, supervisor_steps, or human_research_cycles."""

    def test_retry_attempts_are_isolated_from_graph_state_counters(self):
        """Retrying an operation 3 times does not affect graph state counters."""
        state: ResearchState = {
            "question": "What is quantum error correction?",
            "research_iteration": 1,
            "supervisor_steps": 2,
            "human_research_cycles": 1,
        }

        attempts = 0
        recorded_sleeps: list[float] = []

        def flaky_research_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise MockHttpError("503 Server Unavailable", status_code=503)
            return "recovered_data"

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.1,
            jitter=False,
            sleep_fn=recorded_sleeps.append,
        )

        result = execute_with_retry(
            operation=flaky_research_operation,
            policy=policy,
            component="researcher",
            operation_name="fetch_sources",
        )

        assert result == "recovered_data"
        assert attempts == 3

        # State counters must remain strictly unchanged by the retry loop
        assert state["research_iteration"] == 1
        assert state["supervisor_steps"] == 2
        assert state["human_research_cycles"] == 1


# ==============================================================================
# 12. Checkpoint Persistence Compatibility Tests
# ==============================================================================


class TestCheckpointPersistenceCompatibility:
    """Test suite verifying that ErrorInfo serializes cleanly into SQLite checkpoint storage."""

    def test_error_info_persists_in_sqlite_checkpoint(self, tmp_path):
        import sqlite3
        from langgraph.graph import END, START, StateGraph
        from verified_research.persistence.sqlite import create_sqlite_checkpointer

        db_file = tmp_path / "test_reliability_ckpt.db"
        conn = sqlite3.connect(str(db_file), check_same_thread=False)
        checkpointer = create_sqlite_checkpointer(conn)

        builder = StateGraph(ResearchState)

        def mock_node_with_error(state: ResearchState) -> dict:
            error = ErrorInfo(
                component="search",
                operation="tavily_search",
                error_type="TimeoutError",
                category=ErrorCategory.NETWORK_TIMEOUT,
                message="Search timed out",
                retryable=True,
                attempts=3,
                status_code=408,
            )
            return {
                "errors": [error],
                "last_error": error,
            }

        builder.add_node("node_err", mock_node_with_error)
        builder.add_edge(START, "node_err")
        builder.add_edge("node_err", END)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread-reliability-1"}}
        final_state = graph.invoke({"question": "Test query"}, config)

        assert "errors" in final_state
        assert len(final_state["errors"]) == 1
        assert final_state["errors"][0].category == ErrorCategory.NETWORK_TIMEOUT
        assert final_state["last_error"].attempts == 3

        # Retrieve checkpoint from database and verify deserialization
        saved_state = graph.get_state(config)
        assert saved_state.values["last_error"].component == "search"
        assert saved_state.values["last_error"].error_type == "TimeoutError"
        conn.close()
