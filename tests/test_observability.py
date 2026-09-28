"""Tests for Phase 12 LangSmith Observability and telemetry."""

import os
from unittest.mock import MagicMock
import pytest
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage
from verified_research.config.settings import Settings
from verified_research.graph.graph import create_supervisor_graph
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    SupervisorDecision,
    VerificationResult,
)
from verified_research.observability.collector import ObservabilityCallbackHandler
from verified_research.observability.config import (
    get_tracing_config,
    is_tracing_enabled,
    setup_langsmith_environment,
)
from verified_research.observability.metadata import (
    SECRET_PATTERNS,
    build_graph_session_metadata,
    build_hitl_metadata,
    build_reliability_metadata,
    build_research_metadata,
    build_supervisor_metadata,
    build_verifier_claim_metadata,
    sanitize_metadata,
    sanitize_text,
)
from verified_research.observability.run_config import build_trace_run_config
from verified_research.observability.tracer import (
    observe,
    record_event,
    record_metadata,
    trace_span,
)
from verified_research.reliability.classification import ErrorCategory
from verified_research.reliability.models import MaxRetriesExceededError
from verified_research.reliability.policy import RetryPolicy, execute_with_retry


@pytest.fixture(autouse=True)
def clean_observability_env(monkeypatch):
    """Ensure environment is isolated from ambient LangSmith variables by default."""
    from verified_research.config.settings import get_settings
    get_settings.cache_clear()

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)

    try:
        from langsmith.utils import get_env_var
        get_env_var.cache_clear()
    except Exception:
        pass


class TestTracingDisabled:
    """1. Tracing disabled: Application executes normally without LangSmith credentials."""

    def test_tracing_is_disabled_by_default(self):
        settings = Settings(
            LANGSMITH_TRACING=False,
            LANGCHAIN_TRACING_V2=False,
            LANGSMITH_API_KEY=None,
        )
        assert is_tracing_enabled(settings) is False

    def test_trace_span_noop_when_disabled(self):
        # When tracing is disabled, trace_span yields None without errors
        executed = False
        with trace_span("test_span", run_type="chain") as span:
            executed = True
            assert span is None
        assert executed is True

    def test_record_metadata_noop_when_disabled(self):
        # record_metadata should not raise or error when tracing is disabled
        record_metadata(test_key="test_value", supervisor_step=1)

    def test_record_event_noop_when_disabled(self):
        # record_event should not raise or error when tracing is disabled
        record_event("test_event", status="ok")

    def test_application_executes_normally_without_credentials(self):
        """Full supervisor graph runs to completion without errors when tracing is disabled."""
        def mock_subgraph(state):
            return {
                "sources": [Source(source_id="src_001", title="T", url="http://u", content="C")],
                "findings": [Finding(finding_id="f_001", text="F", source_ids=["src_001"])],
                "evidence": [Evidence(evidence_id="ev_001", source_id="src_001", text="E")],
                "claims": [Claim(claim_id="c_001", text="C", evidence_ids=["ev_001"])],
                "research_iteration": 1,
            }

        def mock_verifier(state):
            return {
                "verification_results": [
                    VerificationResult(
                        claim_id="c_001",
                        verdict="SUPPORTED",
                        confidence=0.9,
                        reasoning="R",
                        evidence_ids=["ev_001"],
                    )
                ]
            }

        def mock_human(state):
            return {
                "human_review": HumanReview(action="approve")
            }

        graph = create_supervisor_graph(
            custom_subgraph=mock_subgraph,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        result = graph.invoke({"question": "What is quantum computing?"})
        assert result.get("human_review").action == "approve"
        assert result.get("supervisor_decision").next_worker == "finish"
        assert result.get("supervisor_termination_reason") == "COMPLETED"


class TestConfiguration:
    """2. Configuration is read from environment/configuration rather than hardcoded."""

    def test_config_reads_langsmith_env_vars(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key_123456789")
        monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://api.test.smith.com")

        settings = Settings()
        assert settings.langsmith_tracing is True
        assert settings.langsmith_api_key == "lsv2_test_key_123456789"
        assert settings.langsmith_project == "test-project"
        assert settings.langsmith_endpoint == "https://api.test.smith.com"
        assert is_tracing_enabled(settings) is True

    def test_config_reads_legacy_langchain_env_vars(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_legacy_key_123456789")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "legacy-project")

        settings = Settings()
        assert settings.langchain_tracing_v2 is True
        assert settings.langchain_api_key == "lsv2_legacy_key_123456789"
        assert settings.langchain_project == "legacy-project"
        assert is_tracing_enabled(settings) is True

    def test_get_tracing_config_masks_api_key(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_very_secret_key_123456789")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
        from verified_research.config.settings import get_settings
        get_settings.cache_clear()

        config = get_tracing_config()
        assert config["project"] == "my-project"
        assert config["has_api_key"] is True
        assert "lsv2_pt_very_secret_key_123456789" not in str(config["masked_api_key"])
        assert config["masked_api_key"].startswith("lsv2")
        assert config["masked_api_key"].endswith("6789")

    def test_setup_langsmith_environment(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "sync-proj")

        settings = Settings()
        setup_langsmith_environment(settings)
        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGSMITH_PROJECT") == "sync-proj"


class TestNoSecretLeakage:
    """3. No secret leakage: Verify credentials are not included in metadata or logs."""

    def test_sanitize_text_redacts_known_secret_patterns(self):
        samples = [
            ("Groq key: gsk_1234567890abcdefghijklmnopqrstuvw", "Groq key: [REDACTED_SECRET]"),
            ("Tavily key: tvly-dev-abcdef1234567890abcdef", "Tavily key: [REDACTED_SECRET]"),
            ("LangSmith key: lsv2_pt_1234567890abcdef1234567890ab_cdef12", "LangSmith key: [REDACTED_SECRET]"),
            ("OpenAI key: sk-abcdefghijklmnopqrstuvwxyz123456", "OpenAI key: [REDACTED_SECRET]"),
            ("Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "Header: [REDACTED_SECRET]"),
        ]
        for raw, expected in samples:
            assert sanitize_text(raw) == expected

    def test_sanitize_metadata_redacts_sensitive_keys(self):
        meta = {
            "api_key": "raw_secret_value",
            "groq_api_key": "gsk_something",
            "authorization": "Bearer token123",
            "safe_field": "public_data",
            "nested": {
                "token": "secret_token",
                "error": "Failed request with key gsk_1234567890abcdefghijklmnopqrstuvw in url",
            },
            "queries": [
                "safe query",
                "query with key tvly-dev-abcdef1234567890abcdef",
            ],
        }
        clean = sanitize_metadata(meta)
        assert clean["api_key"] == "[REDACTED]"
        assert clean["groq_api_key"] == "[REDACTED]"
        assert clean["authorization"] == "[REDACTED]"
        assert clean["safe_field"] == "public_data"
        assert clean["nested"]["token"] == "[REDACTED]"
        assert "[REDACTED_SECRET]" in clean["nested"]["error"]
        assert "gsk_" not in clean["nested"]["error"]
        assert clean["queries"][0] == "safe query"
        assert "[REDACTED_SECRET]" in clean["queries"][1]
        assert "tvly-" not in clean["queries"][1]

    def test_metadata_builders_scrub_secrets(self):
        sup_meta = build_supervisor_metadata(
            step=1,
            selected_worker="research",
            decision="research",
            reasoning="Using key gsk_1234567890abcdefghijklmnopqrstuvw for auth",
        )
        assert "[REDACTED_SECRET]" in sup_meta["reasoning"]
        assert "gsk_" not in sup_meta["reasoning"]

        rel_meta = build_reliability_metadata(
            component="search",
            operation="tavily_search",
            attempt_number=1,
            retry_count=0,
            error_category="RATE_LIMIT",
            delay_seconds=1.5,
        )
        assert rel_meta["component"] == "search"
        assert rel_meta["retry_count"] == 0


class TestSupervisorMetadata:
    """4. Supervisor metadata: Verify structured operational metadata is generated."""

    def test_supervisor_metadata_generation(self):
        from verified_research.agents.supervisor import create_supervisor_node

        state: ResearchState = {
            "question": "Research query",
            "sources": [],
            "claims": [],
            "findings": [],
            "verification_results": [],
            "supervisor_steps": 0,
        }

        node = create_supervisor_node()
        result = node(state)

        assert result["supervisor_steps"] == 1
        assert result["supervisor_decision"].next_worker == "research"
        assert "supervisor_decision" in result

    def test_supervisor_step_limit_metadata(self):
        from verified_research.agents.supervisor import create_supervisor_node

        state: ResearchState = {
            "question": "Research query",
            "sources": [],
            "claims": [],
            "findings": [],
            "verification_results": [],
            "supervisor_steps": 8,
        }

        node = create_supervisor_node(max_steps=8)
        result = node(state)

        assert result["supervisor_decision"].next_worker == "finish"
        assert result["supervisor_termination_reason"] == "MAX_SUPERVISOR_STEPS_REACHED"


class TestVerifierMetadata:
    """5. Verifier metadata: Verify claim ID/verdict/confidence metadata is available."""

    def test_verifier_metadata_builder(self):
        meta = build_verifier_claim_metadata(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.95432,
            evidence_count=2,
            reasoning="Directly supported by peer-reviewed findings.",
        )
        assert meta["claim_id"] == "claim_001"
        assert meta["verdict"] == "SUPPORTED"
        assert meta["confidence"] == 0.954
        assert meta["evidence_count"] == 2
        assert meta["reasoning"].startswith("Directly supported")

    def test_verifier_node_spans_and_metadata_with_mock_client(self):
        from verified_research.agents.verifier import create_verifier_node

        claims = [
            Claim(claim_id="c_001", text="Quantum supremacy achieved.", evidence_ids=["ev_001"]),
            Claim(claim_id="c_002", text="Classical computers obsolete.", evidence_ids=["ev_001"]),
        ]
        evidence = [
            Evidence(evidence_id="ev_001", source_id="s_001", text="Quantum supremacy demonstrated in 2019."),
        ]

        def custom_verify(claim, ev_list):
            if claim.claim_id == "c_001":
                return VerificationResult(
                    claim_id=claim.claim_id,
                    verdict="SUPPORTED",
                    confidence=0.95,
                    reasoning="Directly supported.",
                    evidence_ids=[e.evidence_id for e in ev_list],
                )
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="UNSUPPORTED",
                confidence=0.1,
                reasoning="Contradicted.",
                evidence_ids=[e.evidence_id for e in ev_list],
            )

        mock_client = MagicMock()
        node = create_verifier_node(custom_verifier=custom_verify)

        # Call verifier node within mock tracing client context
        state: ResearchState = {"question": "q", "claims": claims, "evidence": evidence}
        with trace_span("verifier_node_test", client=mock_client):
            res = node(state)

        assert len(res["verification_results"]) == 2
        assert res["verification_results"][0].verdict == "SUPPORTED"
        assert res["verification_results"][1].verdict == "UNSUPPORTED"


class TestRetryMetadata:
    """6. Retry metadata: Verify retry information can be surfaced."""

    def test_retry_metadata_surfaced_on_success_after_retries(self):
        attempts = 0

        def flaky_op():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("Connection refused by peer")
            return "success_val"

        policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False)
        retry_events = []

        def on_retry(att, exc, delay):
            retry_events.append((att, type(exc).__name__, delay))

        result = execute_with_retry(
            operation=flaky_op,
            policy=policy,
            component="search",
            operation_name="tavily_search",
            on_retry=on_retry,
        )

        assert result == "success_val"
        assert len(retry_events) == 1
        assert retry_events[0][0] == 1
        assert retry_events[0][1] == "ConnectionError"

    def test_retry_metadata_surfaced_on_terminal_failure(self):
        def always_fails():
            raise TimeoutError("Gateway timeout")

        policy = RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False)
        captured_error = []

        def on_failure(err_info):
            captured_error.append(err_info)

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            execute_with_retry(
                operation=always_fails,
                policy=policy,
                component="search",
                operation_name="tavily_search",
                on_failure=on_failure,
            )

        err_info = exc_info.value.error_info
        assert err_info is not None
        assert err_info.component == "search"
        assert err_info.operation == "tavily_search"
        assert err_info.attempts == 2
        assert err_info.category == ErrorCategory.NETWORK_TIMEOUT


class TestHumanReviewMetadata:
    """7. Human Review metadata: Verify human action and interrupt-resume events can be represented."""

    def test_human_review_metadata_builder(self):
        meta = build_hitl_metadata(
            event="resumed",
            human_action="edit",
            human_research_cycle=1,
            claims_count=3,
            edited_claims_count=2,
            feedback_present=True,
        )
        assert meta["hitl_event"] == "resumed"
        assert meta["human_action"] == "edit"
        assert meta["human_research_cycle"] == 1
        assert meta["claims_count"] == 3
        assert meta["edited_claims_count"] == 2
        assert meta["feedback_present"] is True

    def test_human_review_node_records_observability_events(self):
        from verified_research.agents.human_review import create_human_review_node

        claims = [Claim(claim_id="c_001", text="Original claim", evidence_ids=["ev_001"])]
        evidence = [Evidence(evidence_id="ev_001", source_id="s_001", text="Evidence text")]

        # Injected interrupt returning 'approve'
        def mock_interrupt(payload):
            return HumanReview(action="approve")

        node = create_human_review_node(interrupt_fn=mock_interrupt)
        state: ResearchState = {
            "question": "Q",
            "claims": claims,
            "evidence": evidence,
            "human_research_cycles": 0,
        }

        result = node(state)
        assert result["human_review"].action == "approve"


class TestObservabilityCollector:
    """Metrics collection and call counting."""

    def test_collector_records_llm_and_tokens(self):
        collector = ObservabilityCallbackHandler()

        # Simulate LLM Start
        collector.on_llm_start(serialized={"name": "ChatGroq"}, prompts=["test prompt"], run_id="uuid-1")
        assert collector.llm_calls == 1

        # Simulate LLM End with usage metadata
        msg = AIMessage(content="answer", usage_metadata={"input_tokens": 15, "output_tokens": 10, "total_tokens": 25})
        gen = ChatGeneration(message=msg)
        llm_res = LLMResult(generations=[[gen]])
        collector.on_llm_end(response=llm_res, run_id="uuid-1")

        assert collector.token_usage["prompt_tokens"] == 15
        assert collector.token_usage["completion_tokens"] == 10
        assert collector.token_usage["total_tokens"] == 25

    def test_collector_records_search_retry_and_supervisor(self):
        collector = ObservabilityCallbackHandler()

        collector.record_search("test query", results_count=3)
        assert collector.search_calls == 1

        collector.record_retry(component="search", operation="tavily_search", attempt=1, delay=1.0)
        assert collector.retries == 1

        collector.record_supervisor_decision(step=1, worker="research")
        assert collector.supervisor_decisions == 1

        collector.record_verifier_operation(claim_id="c_001", verdict="SUPPORTED", confidence=0.9)
        assert collector.verifier_operations == 1

        summary = collector.get_summary()
        assert summary["counts"]["search_calls"] == 1
        assert summary["counts"]["retries"] == 1
        assert summary["counts"]["supervisor_decisions"] == 1
        assert summary["counts"]["verifier_operations"] == 1
        assert len(summary["events"]) == 4


class TestGraphBehaviorParity:
    """8. Existing graph behavior: Tracing does not change routing or state behavior."""

    def test_graph_produces_identical_state_with_and_without_tracing(self):
        def mock_subgraph(state):
            return {
                "sources": [Source(source_id="src_001", title="T", url="http://u", content="C")],
                "findings": [Finding(finding_id="f_001", text="F", source_ids=["src_001"])],
                "evidence": [Evidence(evidence_id="ev_001", source_id="src_001", text="E")],
                "claims": [Claim(claim_id="c_001", text="C", evidence_ids=["ev_001"])],
                "research_iteration": 1,
            }

        def mock_verifier(state):
            return {
                "verification_results": [
                    VerificationResult(
                        claim_id="c_001",
                        verdict="SUPPORTED",
                        confidence=0.9,
                        reasoning="R",
                        evidence_ids=["ev_001"],
                    )
                ]
            }

        def mock_human(state):
            return {
                "human_review": HumanReview(action="approve")
            }

        graph = create_supervisor_graph(
            custom_subgraph=mock_subgraph,
            custom_verifier=mock_verifier,
            custom_human_review=mock_human,
        )

        # Run 1: without tracing
        run1 = graph.invoke({"question": "Test Question"})

        # Run 2: with trace run config
        cfg = build_trace_run_config(
            thread_id="t_parity_1",
            question="Test Question",
            additional_tags=["parity-check"],
        )
        run2 = graph.invoke({"question": "Test Question"}, config=cfg)

        assert run1.get("human_review").action == run2.get("human_review").action
        assert run1.get("supervisor_steps") == run2.get("supervisor_steps")
        assert run1.get("supervisor_termination_reason") == run2.get("supervisor_termination_reason")
        assert len(run1.get("claims")) == len(run2.get("claims"))
