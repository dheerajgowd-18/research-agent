"""Unit tests for Phase 1 data models and state schema."""

import pytest
from pydantic import ValidationError
from verified_research.graph.state import ResearchState
from verified_research.models.research import (
    AnalystOutput,
    Claim,
    Critique,
    Evidence,
    Finding,
    Source,
    VerificationResult,
)


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


class TestEvidenceModel:
    """Tests for Evidence Pydantic model."""

    def test_evidence_valid_instantiation(self):
        ev = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Coherence times reached 500 microseconds under cryogenic shielding.",
        )
        assert ev.evidence_id == "ev_001"
        assert ev.source_id == "src_001"
        assert "cryogenic" in ev.text

    def test_evidence_empty_fields_fail(self):
        with pytest.raises(ValidationError):
            Evidence(evidence_id="", source_id="src_001", text="some text")

        with pytest.raises(ValidationError):
            Evidence(evidence_id="ev_001", source_id="", text="some text")

        with pytest.raises(ValidationError):
            Evidence(evidence_id="ev_001", source_id="src_001", text="")

    def test_evidence_immutability(self):
        ev = Evidence(evidence_id="ev_001", source_id="src_001", text="text")
        with pytest.raises(ValidationError):
            ev.text = "modified text"  # type: ignore

    def test_evidence_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Evidence(
                evidence_id="ev_001",
                source_id="src_001",
                text="text",
                extra_field="disallowed",  # type: ignore
            )


class TestClaimModel:
    """Tests for Claim Pydantic model."""

    def test_claim_valid_instantiation(self):
        claim = Claim(
            claim_id="claim_001",
            text="Superconducting qubits exceeded fault-tolerant fidelity limits.",
            evidence_ids=["ev_001", "ev_002"],
        )
        assert claim.claim_id == "claim_001"
        assert len(claim.evidence_ids) == 2
        assert "ev_001" in claim.evidence_ids

    def test_claim_empty_evidence_ids_fails(self):
        with pytest.raises(ValidationError):
            Claim(
                claim_id="claim_001",
                text="Statement without evidence.",
                evidence_ids=[],
            )

    def test_claim_empty_fields_fail(self):
        with pytest.raises(ValidationError):
            Claim(claim_id="", text="Text", evidence_ids=["ev_001"])

        with pytest.raises(ValidationError):
            Claim(claim_id="claim_001", text="", evidence_ids=["ev_001"])

    def test_claim_immutability(self):
        claim = Claim(claim_id="claim_001", text="Text", evidence_ids=["ev_001"])
        with pytest.raises(ValidationError):
            claim.text = "New text"  # type: ignore

    def test_claim_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Claim(
                claim_id="claim_001",
                text="Text",
                evidence_ids=["ev_001"],
                confidence=0.99,  # type: ignore
            )


class TestResearchStateSchema:
    """Tests for LangGraph ResearchState TypedDict schema."""

    def test_research_state_partial_instantiation(self):
        # Initial state only requires question
        state: ResearchState = {"question": "What is the state of quantum error correction?"}
        assert state["question"] == "What is the state of quantum error correction?"
        assert "sources" not in state
        assert "findings" not in state
        assert "evidence" not in state
        assert "claims" not in state

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
        evidence = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Summary of error correction experiments.",
        )
        claim = Claim(
            claim_id="claim_001",
            text="Surface codes improve fault tolerance.",
            evidence_ids=["ev_001"],
        )
        state: ResearchState = {
            "question": "What is quantum error correction?",
            "sources": [source],
            "findings": [finding],
            "evidence": [evidence],
            "claims": [claim],
            "research_iteration": 1,
        }
        assert len(state["sources"]) == 1
        assert len(state["findings"]) == 1
        assert len(state["evidence"]) == 1
        assert len(state["claims"]) == 1
        assert state["claims"][0].evidence_ids == ["ev_001"]
        assert state["evidence"][0].source_id == "src_001"
        assert "verification_results" not in state

        # Also supports state with verification_results
        result = VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.95,
            reasoning="Evidence aligns directly with claim.",
            evidence_ids=["ev_001"],
        )
        state_with_verdicts: ResearchState = {
            "question": "What is quantum error correction?",
            "verification_results": [result],
        }
        assert len(state_with_verdicts["verification_results"]) == 1
        assert state_with_verdicts["verification_results"][0].verdict == "SUPPORTED"


class TestVerificationResultModel:
    """Tests for VerificationResult Pydantic model."""

    def test_verification_result_valid_instantiations(self):
        for verdict in ("SUPPORTED", "PARTIAL", "UNSUPPORTED"):
            res = VerificationResult(
                claim_id="claim_100",
                verdict=verdict,
                confidence=0.85,
                reasoning=f"Valid verdict: {verdict}",
                evidence_ids=["ev_100"],
            )
            assert res.claim_id == "claim_100"
            assert res.verdict == verdict
            assert res.confidence == 0.85
            assert res.reasoning == f"Valid verdict: {verdict}"
            assert res.evidence_ids == ["ev_100"]

    def test_verification_result_invalid_verdict_fails(self):
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="TRUE",  # Invalid verdict
                confidence=0.9,
                reasoning="Invalid verdict keyword",
                evidence_ids=["ev_100"],
            )

    def test_verification_result_confidence_bounds(self):
        # Negative confidence
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="SUPPORTED",
                confidence=-0.1,
                reasoning="Negative confidence",
                evidence_ids=["ev_100"],
            )
        # Exceeds 1.0
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="SUPPORTED",
                confidence=1.05,
                reasoning="Confidence above 1",
                evidence_ids=["ev_100"],
            )

    def test_verification_result_empty_fields_fail(self):
        # Empty claim_id
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="",
                verdict="SUPPORTED",
                confidence=0.9,
                reasoning="Reasoning",
                evidence_ids=["ev_100"],
            )
        # Empty reasoning
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="SUPPORTED",
                confidence=0.9,
                reasoning="",
                evidence_ids=["ev_100"],
            )
        # Empty evidence_ids
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="SUPPORTED",
                confidence=0.9,
                reasoning="Reasoning",
                evidence_ids=[],
            )

    def test_verification_result_immutability(self):
        res = VerificationResult(
            claim_id="claim_100",
            verdict="SUPPORTED",
            confidence=0.9,
            reasoning="Reasoning",
            evidence_ids=["ev_100"],
        )
        with pytest.raises(ValidationError):
            res.verdict = "UNSUPPORTED"  # type: ignore

    def test_verification_result_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            VerificationResult(
                claim_id="claim_100",
                verdict="SUPPORTED",
                confidence=0.9,
                reasoning="Reasoning",
                evidence_ids=["ev_100"],
                extra_field="disallowed",  # type: ignore
            )


