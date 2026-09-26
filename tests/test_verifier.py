"""Unit and integration tests for the ClaimVerifierService, verifier_node, and parent graph verification."""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.agents.verifier import (
    ClaimVerifierService,
    create_verifier_node,
    verifier_node,
)
from verified_research.graph.graph import create_research_graph
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
from verified_research.models.traceability import UnknownEvidenceError
from verified_research.graph.research_subgraph import create_research_subgraph


def create_mock_llm_for_verifier(expected_result: VerificationResult) -> MagicMock:
    """Helper to create a mocked BaseChatModel returning a specific VerificationResult."""
    mock_llm = MagicMock(spec=BaseChatModel)
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = expected_result
    mock_llm.with_structured_output.return_value = structured_mock
    return mock_llm


class TestClaimVerifierCases:
    """Deterministic offline test cases corresponding to Step 12 TEST 1 through TEST 10."""

    def test_1_direct_support(self):
        """TEST 1 — Direct support: Exact factual match yields SUPPORTED."""
        claim = Claim(
            claim_id="claim_001",
            text="Company X generated $4.2B in revenue in 2023.",
            evidence_ids=["ev_001"],
        )
        evidence = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Company X reported revenue of $4.2B in 2023.",
        )
        expected = VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="The evidence directly and explicitly confirms the $4.2B revenue figure for 2023.",
            evidence_ids=["ev_001"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])

        assert res.verdict == "SUPPORTED"
        assert res.claim_id == "claim_001"
        assert res.evidence_ids == ["ev_001"]
        assert res.confidence >= 0.9

        # Verify prompt received both claim and evidence text
        mock_structured = mock_llm.with_structured_output.return_value
        call_messages = mock_structured.invoke.call_args[0][0]
        user_message_text = call_messages[1].content
        assert claim.text in user_message_text
        assert evidence.text in user_message_text

    def test_2_paraphrase_support(self):
        """TEST 2 — Paraphrase support: Semantically equivalent statements yield SUPPORTED."""
        claim = Claim(
            claim_id="claim_002",
            text="EU regulators approved the merger, allowing it to complete.",
            evidence_ids=["ev_002"],
        )
        evidence = Evidence(
            evidence_id="ev_002",
            source_id="src_001",
            text="The merger was finalized following regulatory approval in Brussels.",
        )
        expected = VerificationResult(
            claim_id="claim_002",
            verdict="SUPPORTED",
            confidence=0.95,
            reasoning="Brussels regulatory approval corresponds directly to EU regulator approval enabling finalization.",
            evidence_ids=["ev_002"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "SUPPORTED"
        assert res.confidence >= 0.9

    def test_3_contradiction(self):
        """TEST 3 — Contradiction: Direct factual contradiction yields UNSUPPORTED."""
        claim = Claim(
            claim_id="claim_003",
            text="The clinical trial successfully achieved its primary endpoint.",
            evidence_ids=["ev_003"],
        )
        evidence = Evidence(
            evidence_id="ev_003",
            source_id="src_002",
            text="The clinical trial failed to achieve its primary endpoint.",
        )
        expected = VerificationResult(
            claim_id="claim_003",
            verdict="UNSUPPORTED",
            confidence=0.99,
            reasoning="The evidence explicitly states the trial failed its primary endpoint, contradicting the claim.",
            evidence_ids=["ev_003"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"
        assert "failed" in res.reasoning

    def test_4_partial_support(self):
        """TEST 4 — Partial support: Claim over-claims geographical reach relative to evidence."""
        claim = Claim(
            claim_id="claim_004",
            text="Product A launched globally across North America, Europe, and Asia.",
            evidence_ids=["ev_004"],
        )
        evidence = Evidence(
            evidence_id="ev_004",
            source_id="src_003",
            text="Product A launched in North America and Europe.",
        )
        expected = VerificationResult(
            claim_id="claim_004",
            verdict="PARTIAL",
            confidence=0.90,
            reasoning="Evidence confirms launches in North America and Europe, but does not substantiate Asia or global launch.",
            evidence_ids=["ev_004"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "PARTIAL"

    def test_5_irrelevant_evidence(self):
        """TEST 5 — Irrelevant evidence: Evidence is completely unrelated to claim subject."""
        claim = Claim(
            claim_id="claim_005",
            text="The phone features an OLED display.",
            evidence_ids=["ev_005"],
        )
        evidence = Evidence(
            evidence_id="ev_005",
            source_id="src_004",
            text="The battery operates for 18 hours under typical usage.",
        )
        expected = VerificationResult(
            claim_id="claim_005",
            verdict="UNSUPPORTED",
            confidence=0.95,
            reasoning="The cited evidence discusses battery duration and makes no mention of the display technology.",
            evidence_ids=["ev_005"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_6_numeric_mismatch(self):
        """TEST 6 — Numeric mismatch: Accuracy number differs between evidence and claim."""
        claim = Claim(
            claim_id="claim_006",
            text="The model achieved 94.2% accuracy on the benchmark.",
            evidence_ids=["ev_006"],
        )
        evidence = Evidence(
            evidence_id="ev_006",
            source_id="src_005",
            text="The model achieved 84.2% accuracy on the benchmark.",
        )
        expected = VerificationResult(
            claim_id="claim_006",
            verdict="UNSUPPORTED",
            confidence=0.98,
            reasoning="Numeric mismatch: evidence reports 84.2% accuracy, whereas the claim asserts 94.2%.",
            evidence_ids=["ev_006"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_7_temporal_mismatch(self):
        """TEST 7 — Temporal mismatch: Past expiration vs present active claim."""
        claim = Claim(
            claim_id="claim_007",
            text="The policy is currently active.",
            evidence_ids=["ev_007"],
        )
        evidence = Evidence(
            evidence_id="ev_007",
            source_id="src_006",
            text="The policy was in effect from 2018 to 2020.",
        )
        expected = VerificationResult(
            claim_id="claim_007",
            verdict="UNSUPPORTED",
            confidence=0.92,
            reasoning="The evidence indicates the policy expired in 2020, offering no support for current activity.",
            evidence_ids=["ev_007"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_8_entity_mismatch(self):
        """TEST 8 — Entity mismatch: Different corporate actor named in claim."""
        claim = Claim(
            claim_id="claim_008",
            text="Company Gamma acquired Company Beta.",
            evidence_ids=["ev_008"],
        )
        evidence = Evidence(
            evidence_id="ev_008",
            source_id="src_007",
            text="Company Alpha acquired Company Beta.",
        )
        expected = VerificationResult(
            claim_id="claim_008",
            verdict="UNSUPPORTED",
            confidence=0.99,
            reasoning="Entity mismatch: the acquirer in the evidence is Company Alpha, not Company Gamma.",
            evidence_ids=["ev_008"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_9_scope_hedging_mismatch(self):
        """TEST 9 — Scope / Hedging mismatch: Tentative finding escalated to absolute cure."""
        claim = Claim(
            claim_id="claim_009",
            text="The compound definitely cures cancer.",
            evidence_ids=["ev_009"],
        )
        evidence = Evidence(
            evidence_id="ev_009",
            source_id="src_008",
            text="Early studies suggest the compound may inhibit tumor growth.",
        )
        expected = VerificationResult(
            claim_id="claim_009",
            verdict="UNSUPPORTED",
            confidence=0.97,
            reasoning="Preliminary tumor growth inhibition does not substantiate an absolute curative claim.",
            evidence_ids=["ev_009"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        assert res.verdict == "UNSUPPORTED"

    def test_10_multiple_evidence_items(self):
        """TEST 10 — Multiple evidence items: Claim requires conjunction of two evidence pieces."""
        claim = Claim(
            claim_id="claim_010",
            text="The model was trained on 15T tokens across 1024 GPUs.",
            evidence_ids=["ev_010_a", "ev_010_b"],
        )
        e1 = Evidence(
            evidence_id="ev_010_a",
            source_id="src_009",
            text="Model was trained on 15T tokens.",
        )
        e2 = Evidence(
            evidence_id="ev_010_b",
            source_id="src_009",
            text="Training took 24 days on 1024 GPUs.",
        )
        expected = VerificationResult(
            claim_id="claim_010",
            verdict="SUPPORTED",
            confidence=0.96,
            reasoning="Combining ev_010_a (15T tokens) and ev_010_b (1024 GPUs) provides complete support for the claim.",
            evidence_ids=["ev_010_a", "ev_010_b"],
        )
        mock_llm = create_mock_llm_for_verifier(expected)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [e1, e2])

        assert res.verdict == "SUPPORTED"
        assert set(res.evidence_ids) == {"ev_010_a", "ev_010_b"}

        # Verify prompt formatted both evidence blocks
        mock_structured = mock_llm.with_structured_output.return_value
        call_messages = mock_structured.invoke.call_args[0][0]
        user_message_text = call_messages[1].content
        assert "ev_010_a" in user_message_text
        assert "ev_010_b" in user_message_text
        assert e1.text in user_message_text
        assert e2.text in user_message_text


class TestVerifierNodeIntegrity:
    """Step 12 TEST 11 and TEST 12: Evidence resolution, 1:1 invariant, and error handling."""

    def test_11_invalid_evidence_id_fails_before_llm_call(self):
        """TEST 11 — Invalid evidence ID resolution raises UnknownEvidenceError before calling LLM."""
        claim = Claim(
            claim_id="claim_valid",
            text="Claim citing missing evidence",
            evidence_ids=["ev_non_existent"],
        )
        state: ResearchState = {
            "question": "Sample topic",
            "claims": [claim],
            "evidence": [
                Evidence(
                    evidence_id="ev_different",
                    source_id="src_1",
                    text="Some available evidence",
                )
            ],
        }

        mock_llm = MagicMock(spec=BaseChatModel)
        node = create_verifier_node(llm=mock_llm)

        with pytest.raises(UnknownEvidenceError, match="unknown evidence_id 'ev_non_existent'"):
            node(state)

        # Confirm LLM was never called
        mock_llm.with_structured_output.assert_not_called()

    def test_12_one_result_per_claim_multi_claim_input(self):
        """TEST 12 — Input with 5 claims produces exactly 5 VerificationResult objects with matching claim_ids."""
        claims = [
            Claim(claim_id=f"claim_{i}", text=f"Factual statement number {i}", evidence_ids=[f"ev_{i}"])
            for i in range(1, 6)
        ]
        evidence_items = [
            Evidence(evidence_id=f"ev_{i}", source_id=f"src_{i}", text=f"Supporting text for statement {i}")
            for i in range(1, 6)
        ]

        def fake_verifier(claim: Claim, ev_list: list[Evidence]) -> VerificationResult:
            return VerificationResult(
                claim_id=claim.claim_id,
                verdict="SUPPORTED",
                confidence=0.9,
                reasoning=f"Verified statement {claim.claim_id}",
                evidence_ids=claim.evidence_ids,
            )

        node = create_verifier_node(custom_verifier=fake_verifier)

        state: ResearchState = {
            "question": "Multi-claim research topic",
            "claims": claims,
            "evidence": evidence_items,
        }

        result = node(state)

        assert "verification_results" in result
        results = result["verification_results"]
        assert len(results) == 5

        # Check 1:1 correspondence
        result_claim_ids = [r.claim_id for r in results]
        expected_claim_ids = [c.claim_id for c in claims]
        assert result_claim_ids == expected_claim_ids

    def test_verifier_node_empty_claims(self):
        """Verifier node handles empty or missing claims gracefully."""
        node = create_verifier_node()
        state: ResearchState = {"question": "No claims available"}
        result = node(state)
        assert result == {"verification_results": []}

    def test_verifier_service_llm_error_wrapped(self):
        """LLM invocation failure is wrapped in a RuntimeError."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.side_effect = RuntimeError("API connection timeout")
        mock_llm.with_structured_output.return_value = structured_mock

        service = ClaimVerifierService(llm=mock_llm)
        claim = Claim(claim_id="c1", text="Text", evidence_ids=["e1"])
        evidence = Evidence(evidence_id="e1", source_id="s1", text="Text")

        with pytest.raises(RuntimeError, match="Claim verification failed for 'c1'"):
            service.verify_claim(claim, [evidence])

    def test_verifier_service_invalid_return_type_rejected(self):
        """If structured output does not return a VerificationResult, ValueError is raised."""
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = {"not_a_model": True}
        mock_llm.with_structured_output.return_value = structured_mock

        service = ClaimVerifierService(llm=mock_llm)
        claim = Claim(claim_id="c1", text="Text", evidence_ids=["e1"])
        evidence = Evidence(evidence_id="e1", source_id="s1", text="Text")

        with pytest.raises(ValueError, match="Verifier expected VerificationResult model"):
            service.verify_claim(claim, [evidence])

    def test_verifier_enforces_id_invariants_if_llm_hallucinates_ids(self):
        """If LLM returns incorrect claim_id or evidence_ids, the service corrects them to input IDs."""
        claim = Claim(claim_id="claim_true_id", text="Statement", evidence_ids=["ev_correct"])
        evidence = Evidence(evidence_id="ev_correct", source_id="s1", text="Statement")

        hallucinated_result = VerificationResult(
            claim_id="hallucinated_claim_id",
            verdict="SUPPORTED",
            confidence=0.88,
            reasoning="Valid reason",
            evidence_ids=["hallucinated_ev_id"],
        )
        mock_llm = create_mock_llm_for_verifier(hallucinated_result)
        service = ClaimVerifierService(llm=mock_llm)

        res = service.verify_claim(claim, [evidence])
        # Invariant enforced
        assert res.claim_id == "claim_true_id"
        assert res.evidence_ids == ["ev_correct"]
        assert res.verdict == "SUPPORTED"


class TestParentGraphWithVerifier:
    """Integration test verifying START -> research -> verifier -> END pipeline."""

    def test_parent_graph_runs_research_and_verifier(self):
        """Full parent graph executes child research subgraph, then verifier node, populating results."""
        # Setup mock researcher and analyst
        source = Source(
            source_id="src_001",
            title="Quantum Benchmark",
            url="https://example.com/qb",
            content="Surface codes achieved 99.5% fidelity in superconducting qubits.",
        )
        evidence = Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Surface codes achieved 99.5% fidelity in superconducting qubits.",
        )
        claim = Claim(
            claim_id="claim_001",
            text="Surface codes achieved 99.5% fidelity in superconducting qubits.",
            evidence_ids=["ev_001"],
        )
        finding = Finding(
            finding_id="f_001",
            text="Surface codes achieved 99.5% fidelity in superconducting qubits.",
            source_ids=["src_001"],
        )

        mock_search = MagicMock()
        mock_search.search.return_value = [source]
        researcher = create_researcher_node(search_client=mock_search)

        # Analyst returning findings and claims + evidence in state
        def custom_analyst(state: ResearchState):
            return {
                "findings": [finding],
                "evidence": [evidence],
                "claims": [claim],
            }

        critic = lambda state: {
            "critique": Critique(
                quality_score=0.95,
                should_research_again=False,
            )
        }

        # Child research subgraph
        subgraph = create_research_subgraph(
            custom_researcher=researcher,
            custom_analyst=custom_analyst,
            custom_critic=critic,
        )

        # Verifier
        def custom_verifier(c: Claim, evs: list[Evidence]) -> VerificationResult:
            return VerificationResult(
                claim_id=c.claim_id,
                verdict="SUPPORTED",
                confidence=0.98,
                reasoning="Exact fidelity match from superconducting benchmark.",
                evidence_ids=c.evidence_ids,
            )

        # Build parent graph with injected subgraph and verifier
        parent_graph = create_research_graph(
            custom_subgraph=subgraph,
            custom_verifier=custom_verifier,
        )

        initial_state: ResearchState = {
            "question": "What fidelity did superconducting surface codes achieve?",
        }

        final_state = parent_graph.invoke(initial_state)

        # Verify full state
        assert final_state["question"] == initial_state["question"]
        assert len(final_state["sources"]) == 1
        assert len(final_state["findings"]) == 1
        assert len(final_state["claims"]) == 1
        assert len(final_state["evidence"]) == 1
        assert "verification_results" in final_state

        results = final_state["verification_results"]
        assert len(results) == 1
        assert results[0].claim_id == "claim_001"
        assert results[0].verdict == "SUPPORTED"
        assert results[0].confidence == 0.98
        assert results[0].evidence_ids == ["ev_001"]
