"""Deterministic unit tests for structural traceability validation, citation mapping, and coverage."""

import pytest
from pydantic import ValidationError
from verified_research.models.research import Claim, Evidence, Finding, Source
from verified_research.models.traceability import (
    DuplicateIdError,
    EmptyEvidenceError,
    UnknownEvidenceError,
    UnknownSourceError,
    calculate_citation_coverage,
    generate_claims_and_evidence_from_findings,
    get_claim_sources,
    validate_traceability,
)


@pytest.fixture
def sample_source() -> Source:
    return Source(
        source_id="src_001",
        title="Qubit Fidelity Benchmark 2026",
        url="https://nature.com/articles/qubit-benchmark",
        content="Surface code experiments demonstrated 2-qubit gate fidelities exceeding 99.9%.",
    )


@pytest.fixture
def sample_evidence(sample_source) -> Evidence:
    return Evidence(
        evidence_id="ev_001",
        source_id=sample_source.source_id,
        text="Surface code experiments demonstrated 2-qubit gate fidelities exceeding 99.9%.",
    )


class TestStructuralTraceabilityValidation:
    """Step 12 tests covering structural validation rules and invariants."""

    def test_1_valid_traceability(self, sample_source, sample_evidence):
        """TEST 1: Valid Claim -> Evidence -> Source chain succeeds."""
        claim = Claim(
            claim_id="claim_001",
            text="2-qubit gate fidelity exceeded 99.9% in surface code testing.",
            evidence_ids=[sample_evidence.evidence_id],
        )

        # Must execute cleanly without raising
        validate_traceability(
            claims=[claim],
            evidence=[sample_evidence],
            sources=[sample_source],
        )

    def test_2_unknown_evidence_fails(self, sample_source, sample_evidence):
        """TEST 2: Claim referencing unknown evidence ID must fail."""
        claim = Claim(
            claim_id="claim_001",
            text="Asserted proposition.",
            evidence_ids=["ev_unknown"],
        )

        with pytest.raises(UnknownEvidenceError, match="unknown evidence_id 'ev_unknown'"):
            validate_traceability(
                claims=[claim],
                evidence=[sample_evidence],
                sources=[sample_source],
            )

    def test_3_unknown_source_fails(self, sample_source):
        """TEST 3: Evidence referencing unknown source ID must fail."""
        orphan_evidence = Evidence(
            evidence_id="ev_001",
            source_id="src_unknown",
            text="Some evidence snippet.",
        )
        claim = Claim(
            claim_id="claim_001",
            text="Asserted proposition.",
            evidence_ids=[orphan_evidence.evidence_id],
        )

        with pytest.raises(UnknownSourceError, match="unknown source_id 'src_unknown'"):
            validate_traceability(
                claims=[claim],
                evidence=[orphan_evidence],
                sources=[sample_source],
            )

    def test_4_empty_evidence_fails_pydantic(self):
        """TEST 4: Claim with empty evidence_ids fails Pydantic validation."""
        with pytest.raises(ValidationError):
            Claim(
                claim_id="claim_001",
                text="Claim lacking grounding evidence.",
                evidence_ids=[],
            )

    def test_5_multiple_evidence_succeeds(self, sample_source):
        """TEST 5: Claim referencing multiple valid evidence items succeeds."""
        ev1 = Evidence(evidence_id="ev_001", source_id=sample_source.source_id, text="Excerpt 1")
        ev2 = Evidence(evidence_id="ev_002", source_id=sample_source.source_id, text="Excerpt 2")

        claim = Claim(
            claim_id="claim_001",
            text="Synthesized statement grounded across multiple excerpts.",
            evidence_ids=[ev1.evidence_id, ev2.evidence_id],
        )

        validate_traceability(
            claims=[claim],
            evidence=[ev1, ev2],
            sources=[sample_source],
        )

    def test_duplicate_ids_fail_validation(self, sample_source, sample_evidence):
        """Disallow duplicate IDs within sources, evidence, or claims."""
        # Duplicate source ID
        src_dup = Source(source_id="src_001", title="Dup", url="https://dup.com", content="Text")
        with pytest.raises(DuplicateIdError, match="Duplicate source_id"):
            validate_traceability(claims=[], evidence=[], sources=[sample_source, src_dup])

        # Duplicate evidence ID
        ev_dup = Evidence(evidence_id="ev_001", source_id="src_001", text="Text")
        with pytest.raises(DuplicateIdError, match="Duplicate evidence_id"):
            validate_traceability(
                claims=[],
                evidence=[sample_evidence, ev_dup],
                sources=[sample_source],
            )

        # Duplicate claim ID
        claim1 = Claim(claim_id="claim_001", text="T1", evidence_ids=["ev_001"])
        claim2 = Claim(claim_id="claim_001", text="T2", evidence_ids=["ev_001"])
        with pytest.raises(DuplicateIdError, match="Duplicate claim_id"):
            validate_traceability(
                claims=[claim1, claim2],
                evidence=[sample_evidence],
                sources=[sample_source],
            )


class TestCitationMapping:
    """Step 12 tests covering resolution from Claim to underlying Sources."""

    def test_6_claim_source_resolution(self, sample_source, sample_evidence):
        """TEST 6: get_claim_sources resolves claim -> evidence -> source."""
        claim = Claim(
            claim_id="claim_001",
            text="2-qubit gate fidelities exceeded 99.9%.",
            evidence_ids=[sample_evidence.evidence_id],
        )

        # Resolve by Claim object
        sources_by_obj = get_claim_sources(
            claim=claim,
            evidence=[sample_evidence],
            sources=[sample_source],
        )
        assert len(sources_by_obj) == 1
        assert sources_by_obj[0].source_id == "src_001"
        assert sources_by_obj[0].title == sample_source.title

        # Resolve by string ID
        sources_by_id = get_claim_sources(
            claim="claim_001",
            claims=[claim],
            evidence=[sample_evidence],
            sources=[sample_source],
        )
        assert len(sources_by_id) == 1
        assert sources_by_id[0].source_id == "src_001"


class TestCitationCoverage:
    """Step 12 tests covering citation coverage metric."""

    def test_7_citation_coverage_calculation(self):
        """TEST 7: 10 claims with 8 having valid evidence -> 0.8."""
        valid_evidence = [
            Evidence(evidence_id=f"ev_{i:03d}", source_id="src_001", text=f"Text {i}")
            for i in range(1, 9)
        ]

        # 8 claims with valid evidence IDs
        covered_claims = [
            Claim(claim_id=f"claim_{i:03d}", text=f"Fact {i}", evidence_ids=[f"ev_{i:03d}"])
            for i in range(1, 9)
        ]
        # 2 claims referencing nonexistent evidence IDs
        uncovered_claims = [
            Claim(claim_id=f"claim_{i:03d}", text=f"Fact {i}", evidence_ids=["ev_missing"])
            for i in range(9, 11)
        ]

        all_claims = covered_claims + uncovered_claims
        assert len(all_claims) == 10

        coverage = calculate_citation_coverage(all_claims, valid_evidence=valid_evidence)
        assert coverage == 0.8

    def test_citation_coverage_empty_claims(self):
        assert calculate_citation_coverage([]) == 0.0


class TestStructuralVsSemanticDistinction:
    """Step 12 tests demonstrating that structural validation is distinct from semantic truth."""

    def test_8_structural_vs_semantic(self):
        """TEST 8: Factual contradiction is structurally valid if IDs link correctly.

        Claim: "The method achieved 82% precision."
        Evidence: "The method achieved 62% precision."
        Structural validation MUST succeed without marking claim supported/unsupported.
        """
        source = Source(
            source_id="src_001",
            title="Benchmark Report",
            url="https://benchmark.org",
            content="The method achieved 62% precision on the standard evaluation set.",
        )
        evidence = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="The method achieved 62% precision on the standard evaluation set.",
        )
        contradictory_claim = Claim(
            claim_id="claim_001",
            text="The method achieved 82% precision.",
            evidence_ids=["ev_001"],
        )

        # Structural validation confirms referential integrity:
        # claim_001 -> ev_001 -> src_001
        validate_traceability(
            claims=[contradictory_claim],
            evidence=[evidence],
            sources=[source],
        )

        # Verification that structural check does NOT modify or annotate semantic verdict
        assert contradictory_claim.text == "The method achieved 82% precision."
        assert evidence.text == "The method achieved 62% precision on the standard evaluation set."
        # No 'verification_status' or 'is_supported' field exists in Phase 4 models


class TestFindingToClaimExtraction:
    """Tests for the deterministic bridge extracting Evidence and Claims from findings."""

    def test_extract_claims_and_evidence(self):
        sources = [
            Source(
                source_id="src_001",
                title="Paper A",
                url="https://paper.a",
                content="Quantum computing reduces circuit depth significantly.",
            ),
            Source(
                source_id="src_002",
                title="Paper B",
                url="https://paper.b",
                content="Cryogenic CMOS operates reliably at 4 Kelvin.",
            ),
        ]
        findings = [
            Finding(
                finding_id="finding_001",
                text="Circuit depth is reduced in modern quantum topologies.",
                source_ids=["src_001"],
            ),
            Finding(
                finding_id="finding_002",
                text="Cryogenic hardware maintains stability at low temperatures.",
                source_ids=["src_002"],
            ),
        ]

        evidence, claims = generate_claims_and_evidence_from_findings(findings, sources)

        assert len(evidence) == 2
        assert len(claims) == 2

        # Verify structural validity of generated artifacts
        validate_traceability(claims, evidence, sources)

        assert claims[0].claim_id == "claim_001"
        assert claims[0].evidence_ids == ["ev_001"]
        assert evidence[0].source_id == "src_001"
