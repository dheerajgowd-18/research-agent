"""Unit tests for critic node and research evaluation."""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from verified_research.agents.critic import create_critic_node
from verified_research.models.research import Critique, Finding, Source


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            source_id="src_001",
            title="Qubit Stability Study",
            url="https://example.com/stability",
            content="Coherence times increased to 500 microseconds.",
        )
    ]


@pytest.fixture
def sample_findings() -> list[Finding]:
    return [
        Finding(
            finding_id="finding_001",
            text="Coherence times have improved substantially.",
            source_ids=["src_001"],
        )
    ]


class TestCriticNode:
    """Tests for critic node contract, evaluation, and edge cases."""

    def test_critic_success_with_valid_llm_output(self, sample_sources, sample_findings):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        mock_critique = Critique(
            quality_score=0.88,
            missing_topics=["Fault-tolerance threshold comparison"],
            weak_findings=[],
            citation_gaps=[],
            recommended_queries=["fault tolerance threshold 2026"],
            should_research_again=False,
        )
        structured_mock.invoke.return_value = mock_critique
        mock_llm.with_structured_output.return_value = structured_mock

        node = create_critic_node(llm=mock_llm)
        state = {
            "question": "What are current breakthroughs in quantum computing?",
            "sources": sample_sources,
            "findings": sample_findings,
        }

        result = node(state)

        assert "critique" in result
        critique = result["critique"]
        assert critique.quality_score == 0.88
        assert critique.should_research_again is False
        assert critique.missing_topics == ["Fault-tolerance threshold comparison"]

    def test_critic_auto_flags_empty_findings(self, sample_sources):
        """When 0 findings are present, critic immediately sets should_research_again=True."""
        mock_llm = MagicMock(spec=BaseChatModel)
        node = create_critic_node(llm=mock_llm)

        state = {
            "question": "Research topic",
            "sources": sample_sources,
            "findings": [],
        }

        result = node(state)

        assert "critique" in result
        critique = result["critique"]
        assert critique.quality_score == 0.0
        assert critique.should_research_again is True
        assert len(critique.recommended_queries) >= 1
        # LLM should NOT have been invoked
        mock_llm.with_structured_output.assert_not_called()

    def test_critic_missing_question_fails(self, sample_findings):
        node = create_critic_node()

        with pytest.raises(ValueError, match="non-empty 'question'"):
            node({"question": "", "findings": sample_findings})

        with pytest.raises(ValueError, match="non-empty 'question'"):
            node({"findings": sample_findings})  # type: ignore

    def test_critic_llm_failure_wrapped(self, sample_sources, sample_findings):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = RuntimeError("Service unavailable")
        mock_llm.with_structured_output.return_value = structured_mock

        node = create_critic_node(llm=mock_llm)
        state = {
            "question": "Valid question",
            "sources": sample_sources,
            "findings": sample_findings,
        }

        with pytest.raises(RuntimeError, match="Critic evaluation failed"):
            node(state)

    def test_critic_invalid_structured_output(self, sample_sources, sample_findings):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = {"not": "a Critique model"}
        mock_llm.with_structured_output.return_value = structured_mock

        node = create_critic_node(llm=mock_llm)
        state = {
            "question": "Valid question",
            "sources": sample_sources,
            "findings": sample_findings,
        }

        with pytest.raises(ValueError, match="Critic expected Critique model"):
            node(state)
