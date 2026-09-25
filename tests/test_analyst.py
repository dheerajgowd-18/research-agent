"""Unit tests for analyst node, structured output, and source traceability."""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from verified_research.agents.analyst import (
    InvalidSourceReferenceError,
    create_analyst_node,
    validate_finding_sources,
)
from verified_research.models.research import AnalystOutput, Finding, Source


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            source_id="src_001",
            title="Qubit Stability Study",
            url="https://example.com/stability",
            content="Coherence times increased to 500 microseconds.",
        ),
        Source(
            source_id="src_002",
            title="Cryogenic Control Hardware",
            url="https://example.com/cryo",
            content="New CMOS circuits operate reliably below 4 Kelvin.",
        ),
    ]


class TestAnalystNode:
    """Tests for analyst node contract, validation, and execution."""

    def test_analyst_success_with_valid_sources(self, sample_sources):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        mock_llm.with_structured_output.return_value = structured_mock

        mock_findings = [
            Finding(
                finding_id="finding_001",
                text="Coherence times have improved substantially.",
                source_ids=["src_001"],
            ),
            Finding(
                finding_id="finding_002",
                text="Integrated control circuitry reduces thermal load.",
                source_ids=["src_002"],
            ),
        ]
        structured_mock.invoke.return_value = AnalystOutput(findings=mock_findings)

        node = create_analyst_node(llm=mock_llm)
        state = {
            "question": "What are current breakthroughs in quantum computing?",
            "sources": sample_sources,
        }

        result = node(state)

        assert "findings" in result
        findings = result["findings"]
        assert len(findings) == 2
        assert findings[0].finding_id == "finding_001"
        assert findings[0].source_ids == ["src_001"]
        assert findings[1].finding_id == "finding_002"
        assert findings[1].source_ids == ["src_002"]

    def test_analyst_rejects_invented_source_ids(self, sample_sources):
        """The analyst must NOT invent source IDs not in state."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        mock_llm.with_structured_output.return_value = structured_mock

        # Finding references 'src_999' which is NOT in sample_sources
        hallucinated_findings = [
            Finding(
                finding_id="finding_001",
                text="Unsubstantiated claim with made-up source.",
                source_ids=["src_999"],
            )
        ]
        structured_mock.invoke.return_value = AnalystOutput(findings=hallucinated_findings)

        node = create_analyst_node(llm=mock_llm)
        state = {
            "question": "Any question",
            "sources": sample_sources,
        }

        with pytest.raises(InvalidSourceReferenceError, match="referenced invalid source_id 'src_999'"):
            node(state)

    def test_analyst_handles_empty_sources_cleanly(self):
        """When 0 sources are in state, analyst should return 0 findings without invoking LLM."""
        mock_llm = MagicMock(spec=BaseChatModel)
        node = create_analyst_node(llm=mock_llm)

        state = {
            "question": "Question with no sources found",
            "sources": [],
        }
        result = node(state)

        assert "findings" in result
        assert result["findings"] == []
        mock_llm.with_structured_output.assert_not_called()

    def test_analyst_missing_question_fails(self, sample_sources):
        node = create_analyst_node()

        with pytest.raises(ValueError, match="non-empty 'question'"):
            node({"question": "", "sources": sample_sources})

    def test_analyst_handles_llm_failure(self, sample_sources):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = RuntimeError("Rate limit exceeded")
        mock_llm.with_structured_output.return_value = structured_mock

        node = create_analyst_node(llm=mock_llm)
        state = {
            "question": "Valid question",
            "sources": sample_sources,
        }

        with pytest.raises(RuntimeError, match="Analyst LLM synthesis failed"):
            node(state)

    def test_analyst_handles_invalid_structured_output(self, sample_sources):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        # Returns wrong type instead of AnalystOutput
        structured_mock.invoke.return_value = {"error": "malformed"}
        mock_llm.with_structured_output.return_value = structured_mock

        node = create_analyst_node(llm=mock_llm)
        state = {
            "question": "Valid question",
            "sources": sample_sources,
        }

        with pytest.raises(ValueError, match="expected structured AnalystOutput"):
            node(state)


class TestValidateFindingSourcesHelper:
    """Direct unit tests for source ID validation utility."""

    def test_validation_passes_valid_ids(self, sample_sources):
        findings = [
            Finding(finding_id="f1", text="t1", source_ids=["src_001"]),
            Finding(finding_id="f2", text="t2", source_ids=["src_001", "src_002"]),
        ]
        # Should execute without raising
        validate_finding_sources(findings, sample_sources)

    def test_validation_fails_on_partial_invalid_id(self, sample_sources):
        findings = [
            Finding(finding_id="f1", text="t1", source_ids=["src_001", "src_unknown"]),
        ]
        with pytest.raises(InvalidSourceReferenceError, match="src_unknown"):
            validate_finding_sources(findings, sample_sources)
