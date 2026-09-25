"""Unit tests for Phase 1 data models and state schema."""

import pytest
from pydantic import ValidationError
from verified_research.graph.state import ResearchState
from verified_research.models.research import AnalystOutput, Critique, Finding, Source


class TestSourceModel:
    """Tests for Source Pydantic model."""

    def test_source_valid_instantiation(self):
        source = Source(
            source_id="src_001",
            title="Quantum Advantage Overview",
            url="https://example.com/quantum",
            content="Recent progress in quantum computing demonstrates key milestones.",
        )
        assert source.source_id == "src_001"
        assert source.title == "Quantum Advantage Overview"
        assert source.url == "https://example.com/quantum"
        assert "progress" in source.content

    def test_source_empty_fields_fail(self):
        with pytest.raises(ValidationError):
            Source(source_id="", title="Title", url="https://example.com", content="Content")

        with pytest.raises(ValidationError):
            Source(source_id="src_1", title="", url="https://example.com", content="Content")

        with pytest.raises(ValidationError):
            Source(source_id="src_1", title="Title", url="", content="Content")

        with pytest.raises(ValidationError):
            Source(source_id="src_1", title="Title", url="https://example.com", content="")

    def test_source_immutability(self):
        source = Source(
            source_id="src_001",
            title="Title",
            url="https://example.com",
            content="Content",
        )
        with pytest.raises(ValidationError):
            # Frozen model should not allow mutation
            source.title = "New Title"  # type: ignore


class TestFindingModel:
    """Tests for Finding Pydantic model."""

    def test_finding_valid_instantiation(self):
        finding = Finding(
            finding_id="finding_001",
            text="Superconducting qubits have reached lower error thresholds.",
            source_ids=["src_001", "src_002"],
        )
        assert finding.finding_id == "finding_001"
        assert len(finding.source_ids) == 2
        assert "src_001" in finding.source_ids

    def test_finding_empty_source_ids_fails(self):
        with pytest.raises(ValidationError):
            Finding(
                finding_id="finding_001",
                text="Finding text",
                source_ids=[],
            )

    def test_finding_empty_fields_fail(self):
        with pytest.raises(ValidationError):
            Finding(finding_id="", text="Text", source_ids=["src_001"])

        with pytest.raises(ValidationError):
            Finding(finding_id="finding_001", text="", source_ids=["src_001"])


class TestAnalystOutputModel:
    """Tests for AnalystOutput model."""

    def test_analyst_output_instantiation(self):
        finding = Finding(
            finding_id="finding_001",
            text="Finding text",
            source_ids=["src_001"],
        )
        output = AnalystOutput(findings=[finding])
        assert len(output.findings) == 1
        assert output.findings[0].finding_id == "finding_001"

    def test_analyst_output_default_empty(self):
        output = AnalystOutput()
        assert output.findings == []


class TestCritiqueModel:
    """Tests for Critique Pydantic model."""

    def test_critique_valid_instantiation(self):
        critique = Critique(
            quality_score=0.85,
            missing_topics=["Scalability benchmarks"],
            weak_findings=["finding_001"],
            citation_gaps=["Claim X needs URL"],
            recommended_queries=["quantum computing benchmark 2026"],
            should_research_again=False,
        )
        assert critique.quality_score == 0.85
        assert critique.should_research_again is False
        assert len(critique.missing_topics) == 1
        assert len(critique.recommended_queries) == 1

    def test_critique_score_bounds_validation(self):
        # Below 0.0 fails
        with pytest.raises(ValidationError):
            Critique(
                quality_score=-0.1,
                should_research_again=True,
            )

        # Above 1.0 fails
        with pytest.raises(ValidationError):
            Critique(
                quality_score=1.01,
                should_research_again=True,
            )

    def test_critique_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Critique(
                quality_score=0.5,
                should_research_again=True,
                extra_field="disallowed",  # type: ignore
            )

    def test_critique_defaults(self):
        critique = Critique(
            quality_score=0.5,
            should_research_again=True,
        )
        assert critique.missing_topics == []
        assert critique.weak_findings == []
        assert critique.citation_gaps == []
        assert critique.recommended_queries == []


class TestResearchStateSchema:

    """Tests for LangGraph ResearchState TypedDict schema."""

    def test_research_state_partial_instantiation(self):
        # Initial state only requires question
        state: ResearchState = {"question": "What is the state of quantum error correction?"}
        assert state["question"] == "What is the state of quantum error correction?"
        assert "sources" not in state
        assert "findings" not in state

    def test_research_state_full_instantiation(self):
        source = Source(
            source_id="src_001",
            title="Paper A",
            url="https://arxiv.org/abs/1234",
            content="Summary of error correction experiments.",
        )
        finding = Finding(
            finding_id="finding_001",
            text="Surface codes improve fault tolerance.",
            source_ids=["src_001"],
        )
        state: ResearchState = {
            "question": "What is quantum error correction?",
            "sources": [source],
            "findings": [finding],
        }
        assert len(state["sources"]) == 1
        assert len(state["findings"]) == 1
        assert state["findings"][0].source_ids == ["src_001"]
