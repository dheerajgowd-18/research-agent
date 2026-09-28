"""Production Test Suite for State Corruption, Schema Tampering, and Architectural Bypass Prevention.

Verifies:
1. Malformed or missing input states are rejected or handled defensively.
2. Referential integrity corruption (duplicate IDs, empty evidence IDs, dangling links) is blocked.
3. Architectural bypass attempts (bypassing verifier or HITL) are actively intercepted and corrected.
4. Routing guard prevents execution of unmapped or unauthorized graph workers.
"""

from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from verified_research.agents.supervisor import (
    create_supervisor_node,
    validate_supervisor_decision,
)
from verified_research.graph.router import route_after_supervisor
from verified_research.models.research import (
    Claim,
    Evidence,
    Finding,
    HumanReview,
    Source,
    SupervisorDecision,
    VerificationResult,
)
from verified_research.models.traceability import (
    DuplicateIdError,
    EmptyEvidenceError,
    UnknownEvidenceError,
    UnknownSourceError,
    validate_traceability,
)
from verified_research.graph.state import ResearchState


class TestStateCorruptionAndBypass:
    """Verifies that state corruption and graph routing bypasses are strictly prevented."""

    # -------------------------------------------------------------------------
    # 1. Referential Integrity Corruption Tests
    # -------------------------------------------------------------------------
    def test_duplicate_source_id_detected_and_rejected(self):
        """Duplicate source IDs in state trigger DuplicateIdError."""
        sources = [
            Source(source_id="src_001", title="Source A", url="https://a.com", content="A"),
            Source(source_id="src_001", title="Source B", url="https://b.com", content="B"),
        ]
        evidence = [Evidence(evidence_id="ev_001", source_id="src_001", text="A")]
        claims = [Claim(claim_id="clm_001", text="Claim A", evidence_ids=["ev_001"])]

        with pytest.raises(DuplicateIdError, match="Duplicate source_id detected"):
            validate_traceability(claims=claims, evidence=evidence, sources=sources)

    def test_duplicate_evidence_id_detected_and_rejected(self):
        """Duplicate evidence IDs in state trigger DuplicateIdError."""
        sources = [Source(source_id="src_001", title="Source A", url="https://a.com", content="A")]
        evidence = [
            Evidence(evidence_id="ev_001", source_id="src_001", text="Text 1"),
            Evidence(evidence_id="ev_001", source_id="src_001", text="Text 2"),
        ]
        claims = [Claim(claim_id="clm_001", text="Claim A", evidence_ids=["ev_001"])]

        with pytest.raises(DuplicateIdError, match="Duplicate evidence_id detected"):
            validate_traceability(claims=claims, evidence=evidence, sources=sources)

    def test_claim_with_empty_evidence_ids_rejected(self):
        """Claim with empty evidence_ids triggers ValidationError or EmptyEvidenceError."""
        with pytest.raises(ValidationError):
            Claim(claim_id="clm_empty", text="Ungrounded statement", evidence_ids=[])

    def test_dangling_evidence_in_claim_rejected(self):
        """Claim referencing non-existent evidence_id triggers UnknownEvidenceError."""
        sources = [Source(source_id="src_001", title="Source A", url="https://a.com", content="A")]
        evidence = [Evidence(evidence_id="ev_001", source_id="src_001", text="A")]
        claims = [Claim(claim_id="clm_001", text="Claim A", evidence_ids=["ev_phantom_999"])]

        with pytest.raises(UnknownEvidenceError):
            validate_traceability(claims=claims, evidence=evidence, sources=sources)

    def test_dangling_source_in_evidence_rejected(self):
        """Evidence referencing non-existent source_id triggers UnknownSourceError."""
        sources = [Source(source_id="src_001", title="Source A", url="https://a.com", content="A")]
        evidence = [Evidence(evidence_id="ev_001", source_id="src_phantom_999", text="A")]
        claims = [Claim(claim_id="clm_001", text="Claim A", evidence_ids=["ev_001"])]

        with pytest.raises(UnknownSourceError):
            validate_traceability(claims=claims, evidence=evidence, sources=sources)

    # -------------------------------------------------------------------------
    # 2. Architectural Bypass Interception Tests
    # -------------------------------------------------------------------------
    def test_bypass_verifier_to_human_review_intercepted(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """If a policy attempts to bypass verifier directly to human_review, guard forces routing to verify."""
        state = {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "claims": sample_claims,
            "verification_results": [],  # No verifications completed
            "supervisor_steps": 1,
        }
        malicious_decision = SupervisorDecision(
            next_worker="human_review",
            reasoning="Attempting to skip verifier node.",
        )

        guarded = validate_supervisor_decision(state, malicious_decision)
        assert guarded.next_worker == "verify"
        assert "unverified claims exist" in guarded.reasoning.lower()

    def test_bypass_verifier_to_finish_intercepted(
        self,
        sample_sources: list[Source],
        sample_evidence: list[Evidence],
        sample_claims: list[Claim],
    ):
        """If a policy attempts to bypass verifier directly to finish, guard forces routing to verify."""
        state = {
            "sources": sample_sources,
            "evidence": sample_evidence,
            "claims": sample_claims,
            "verification_results": [],
            "supervisor_steps": 1,
        }
        malicious_decision = SupervisorDecision(
            next_worker="finish",
            reasoning="Attempting to finish early without verification.",
        )

        guarded = validate_supervisor_decision(state, malicious_decision)
        assert guarded.next_worker == "verify"

    def test_verify_worker_without_claims_redirected_to_research(self):
        """If a policy attempts to call verifier when no claims exist in state, guard redirects to research."""
        state = {"claims": [], "verification_results": [], "supervisor_steps": 0}
        invalid_decision = SupervisorDecision(
            next_worker="verify",
            reasoning="Calling verifier on empty state.",
        )

        guarded = validate_supervisor_decision(state, invalid_decision)
        assert guarded.next_worker == "research"

    def test_unauthorized_worker_name_tampering_routed_safely_to_finish(self):
        """If state contains a tampered decision targeting an unauthorized worker, router safely defaults to finish."""
        tampered_state = {
            "supervisor_steps": 2,
            "supervisor_decision": SupervisorDecision(next_worker="finish", reasoning="OK"),
        }
        # Tamper directly with the object attribute
        object.__setattr__(tampered_state["supervisor_decision"], "next_worker", "unauthorized_worker")

        # Router logs error and safely falls back to 'finish'
        routed = route_after_supervisor(tampered_state)
        assert routed == "finish"
