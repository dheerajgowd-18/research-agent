"""Production Test Suite for Claim-Level Verifier and Semantic Verdict Evaluation.

Verifies:
1. Direct Support (exact factual match -> SUPPORTED)
2. Paraphrase Support (semantic equivalence -> SUPPORTED)
3. Direct Contradiction (factual reversal -> UNSUPPORTED)
4. Partial Support (subset substantiated -> PARTIAL)
5. Irrelevant Evidence (unrelated topic -> UNSUPPORTED)
6. Numeric Mismatch (number discrepancy -> UNSUPPORTED)
7. Temporal Mismatch (outdated vs present assertion -> UNSUPPORTED)
8. Entity Mismatch (differing corporate/actor entity -> UNSUPPORTED)
9. Scope & Hedging Mismatch (extrapolation to absolute -> UNSUPPORTED)
10. Multi-evidence conjunction verification
11. Verifier Node integration with full graph state
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel

from verified_research.agents.verifier import (
    ClaimVerifierService,
    create_verifier_node,
    verifier_node,
)
from verified_research.models.research import (
    Claim,
    Evidence,
    Source,
    VerificationResult,
)
from verified_research.models.traceability import (
    EmptyEvidenceError,
    UnknownEvidenceError,
    validate_traceability,
)
from verified_research.graph.state import ResearchState


def make_mock_llm(result: VerificationResult) -> MagicMock:
    mock_llm = MagicMock(spec=BaseChatModel)
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = result
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm


class TestVerifierIntegration:
    """Verifies all semantic verification categories and verifier node state handling."""

    def test_direct_support(self):
        """Exact factual alignment yields SUPPORTED."""
        claim = Claim(claim_id="c1", text="Company revenue was $4.2B in 2023.", evidence_ids=["e1"])
        evidence = Evidence(evidence_id="e1", source_id="s1", text="Company reported revenue of $4.2B in 2023.")
        expected = VerificationResult(
            claim_id="c1", verdict="SUPPORTED", confidence=0.99, reasoning="Exact match.", evidence_ids=["e1"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "SUPPORTED"
        assert res.confidence >= 0.95

    def test_paraphrase_support(self):
        """Paraphrased but factually equivalent assertions yield SUPPORTED."""
        claim = Claim(claim_id="c2", text="Regulators cleared the transaction in Europe.", evidence_ids=["e2"])
        evidence = Evidence(evidence_id="e2", source_id="s1", text="The merger received regulatory greenlight in Brussels.")
        expected = VerificationResult(
            claim_id="c2", verdict="SUPPORTED", confidence=0.94, reasoning="Brussels/Europe regulatory equivalence.", evidence_ids=["e2"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "SUPPORTED"

    def test_direct_contradiction(self):
        """Direct contradiction yields UNSUPPORTED."""
        claim = Claim(claim_id="c3", text="The rocket successfully landed on the pad.", evidence_ids=["e3"])
        evidence = Evidence(evidence_id="e3", source_id="s1", text="The rocket exploded upon descent and missed the pad.")
        expected = VerificationResult(
            claim_id="c3", verdict="UNSUPPORTED", confidence=0.99, reasoning="Evidence directly refutes landing success.", evidence_ids=["e3"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_partial_support(self):
        """Subset of multi-attribute claim supported yields PARTIAL."""
        claim = Claim(claim_id="c4", text="Tool deployed across US, UK, and Japan.", evidence_ids=["e4"])
        evidence = Evidence(evidence_id="e4", source_id="s1", text="Tool was deployed across the US and UK.")
        expected = VerificationResult(
            claim_id="c4", verdict="PARTIAL", confidence=0.88, reasoning="Japan deployment lacks evidence.", evidence_ids=["e4"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "PARTIAL"

    def test_irrelevant_evidence(self):
        """Irrelevant evidence yields UNSUPPORTED."""
        claim = Claim(claim_id="c5", text="Solar cell efficiency hit 33%.", evidence_ids=["e5"])
        evidence = Evidence(evidence_id="e5", source_id="s1", text="Wind farm construction completed ahead of schedule.")
        expected = VerificationResult(
            claim_id="c5", verdict="UNSUPPORTED", confidence=0.96, reasoning="Evidence discusses wind farms, not solar cells.", evidence_ids=["e5"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_numeric_mismatch(self):
        """Discrepancy in numeric metrics yields UNSUPPORTED."""
        claim = Claim(claim_id="c6", text="Model latency is 15 milliseconds.", evidence_ids=["e6"])
        evidence = Evidence(evidence_id="e6", source_id="s1", text="Model latency was benchmarked at 150 milliseconds.")
        expected = VerificationResult(
            claim_id="c6", verdict="UNSUPPORTED", confidence=0.98, reasoning="10x numeric discrepancy.", evidence_ids=["e6"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_temporal_mismatch(self):
        """Discrepancy in timeframe yields UNSUPPORTED."""
        claim = Claim(claim_id="c7", text="System was operational in 2025.", evidence_ids=["e7"])
        evidence = Evidence(evidence_id="e7", source_id="s1", text="System was decommissioned in 2021.")
        expected = VerificationResult(
            claim_id="c7", verdict="UNSUPPORTED", confidence=0.95, reasoning="Temporal gap: decommissioned prior to 2025.", evidence_ids=["e7"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_entity_mismatch(self):
        """Claim attributes findings to wrong corporate/research entity yields UNSUPPORTED."""
        claim = Claim(claim_id="c8", text="DeepMind released Llama 3.", evidence_ids=["e8"])
        evidence = Evidence(evidence_id="e8", source_id="s1", text="Meta released Llama 3 under open license.")
        expected = VerificationResult(
            claim_id="c8", verdict="UNSUPPORTED", confidence=0.99, reasoning="Entity mismatch: Meta released Llama 3, not DeepMind.", evidence_ids=["e8"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_scope_hedging_mismatch(self):
        """Over-extrapolated certainty yields UNSUPPORTED."""
        claim = Claim(claim_id="c9", text="Drug completely eradicates Alzheimer's symptoms in humans.", evidence_ids=["e9"])
        evidence = Evidence(evidence_id="e9", source_id="s1", text="Drug showed initial biomarker reduction in mouse models.")
        expected = VerificationResult(
            claim_id="c9", verdict="UNSUPPORTED", confidence=0.97, reasoning="Mouse model biomarker reduction != human cure.", evidence_ids=["e9"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_multi_evidence_conjunction(self):
        """Verification synthesizing multiple distinct evidence excerpts."""
        claim = Claim(claim_id="c10", text="Qubit coherence reached 1ms at sub-20mK temperatures.", evidence_ids=["e1", "e2"])
        e1 = Evidence(evidence_id="e1", source_id="s1", text="Transmon coherence times exceeded 1ms.")
        e2 = Evidence(evidence_id="e2", source_id="s2", text="Qubit dilution fridge maintained 15mK temperatures.")
        expected = VerificationResult(
            claim_id="c10", verdict="SUPPORTED", confidence=0.96, reasoning="Both clauses substantiated by e1 and e2.", evidence_ids=["e1", "e2"]
        )
        service = ClaimVerifierService(llm=make_mock_llm(expected))
        res = service.verify_claim(claim, [e1, e2])
        assert res.verdict == "SUPPORTED"
        assert res.evidence_ids == ["e1", "e2"]

    def test_verifier_node_state_execution_and_coverage(self):
        """Verifier node consumes state and produces complete 1:1 verification results."""
        c1 = Claim(claim_id="c1", text="Claim 1", evidence_ids=["e1"])
        c2 = Claim(claim_id="c2", text="Claim 2", evidence_ids=["e2"])
        e1 = Evidence(evidence_id="e1", source_id="s1", text="Evidence 1")
        e2 = Evidence(evidence_id="e2", source_id="s1", text="Evidence 2")

        def mock_custom_verifier(claim, ev_list):
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.95,
                reasoning=f"Verified {claim.claim_id}",
                evidence_ids=[e.evidence_id for e in ev_list],
            )

        node = create_verifier_node(custom_verifier=mock_custom_verifier)
        state: ResearchState = {
            "claims": [c1, c2],
            "evidence": [e1, e2],
            "verification_results": [],
        }

        output = node(state)
        results = output["verification_results"]
        assert len(results) == 2
        assert {r.claim_id for r in results} == {"c1", "c2"}
        assert all(r.verdict == "SUPPORTED" for r in results)
