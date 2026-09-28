"""Shared fixtures and mocks for Phase 13 Production-Style Testing Suite."""

from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel

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
from verified_research.graph.state import ResearchState


class SpySearchService:
    """Mock search service tracking query history deterministically."""

    def __init__(self, predefined_sources: list[Source] | None = None) -> None:
        self.recorded_queries: list[str] = []
        self.predefined_sources = predefined_sources or []

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        self.recorded_queries.append(query)
        if self.predefined_sources:
            return self.predefined_sources
        idx = len(self.recorded_queries)
        return [
            Source(
                source_id=f"src_{idx:03d}",
                title=f"Source on {query}",
                url=f"https://example.com/item/{idx}",
                content=f"Detailed evidence concerning {query}.",
            )
        ]


@pytest.fixture
def sample_sources() -> list[Source]:
    return [
        Source(
            source_id="src_001",
            title="Quantum Coherence Dynamics",
            url="https://arxiv.org/abs/quant-coherence",
            content="Superconducting transmon qubits achieved coherence times exceeding 1.5 milliseconds.",
        ),
        Source(
            source_id="src_002",
            title="Cryogenic Control Systems",
            url="https://nature.com/articles/cryo-controls",
            content="Dilution refrigerators maintain base temperatures below 15 millikelvin with automated gas handling.",
        ),
    ]


@pytest.fixture
def sample_evidence(sample_sources: list[Source]) -> list[Evidence]:
    return [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Superconducting transmon qubits achieved coherence times exceeding 1.5 milliseconds.",
        ),
        Evidence(
            evidence_id="ev_002",
            source_id="src_002",
            text="Dilution refrigerators maintain base temperatures below 15 millikelvin with automated gas handling.",
        ),
    ]


@pytest.fixture
def sample_claims(sample_evidence: list[Evidence]) -> list[Claim]:
    return [
        Claim(
            claim_id="claim_001",
            text="Transmon qubits demonstrated coherence times exceeding 1.5 milliseconds.",
            evidence_ids=["ev_001"],
        ),
        Claim(
            claim_id="claim_002",
            text="Dilution refrigerators sustain operational temperatures below 15 millikelvin.",
            evidence_ids=["ev_002"],
        ),
    ]


@pytest.fixture
def sample_verification_results(sample_claims: list[Claim]) -> list[VerificationResult]:
    return [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.98,
            reasoning="Direct match to experimental measurements in src_001.",
            evidence_ids=["ev_001"],
        ),
        VerificationResult(
            claim_id="claim_002",
            verdict="SUPPORTED",
            confidence=0.95,
            reasoning="Explicit temperature metrics confirmed in src_002.",
            evidence_ids=["ev_002"],
        ),
    ]


@pytest.fixture
def standard_research_state(
    sample_sources: list[Source],
    sample_evidence: list[Evidence],
    sample_claims: list[Claim],
    sample_verification_results: list[VerificationResult],
) -> ResearchState:
    return {
        "question": "What are current breakthroughs in superconducting quantum systems?",
        "sources": sample_sources,
        "evidence": sample_evidence,
        "findings": [
            Finding(
                finding_id="finding_001",
                text="Superconducting qubits have reached millisecond-scale coherence at sub-15mK temperatures.",
                source_ids=["src_001", "src_002"],
            )
        ],
        "claims": sample_claims,
        "verification_results": sample_verification_results,
        "supervisor_steps": 2,
        "research_iteration": 1,
        "human_research_cycles": 0,
    }


def create_deterministic_mock_subgraph(
    sources: list[Source] | None = None,
    evidence: list[Evidence] | None = None,
    claims: list[Claim] | None = None,
    findings: list[Finding] | None = None,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a mock research subgraph returning valid structured artifacts."""

    def mock_subgraph(state: ResearchState) -> dict[str, Any]:
        s = sources or [
            Source(
                source_id="src_sub_001",
                title="Mock Subgraph Source",
                url="https://example.com/sub",
                content="Mocked evidence text from deterministic subgraph.",
            )
        ]
        e = evidence or [
            Evidence(
                evidence_id="ev_sub_001",
                source_id="src_sub_001",
                text="Mocked evidence text from deterministic subgraph.",
            )
        ]
        c = claims or [
            Claim(
                claim_id="claim_sub_001",
                text="Mocked claim asserting verified findings.",
                evidence_ids=["ev_sub_001"],
            )
        ]
        f = findings or [
            Finding(
                finding_id="finding_sub_001",
                text="Deterministic mock research finding.",
                source_ids=["src_sub_001"],
            )
        ]
        return {
            "sources": s,
            "evidence": e,
            "claims": c,
            "findings": f,
            "research_iteration": state.get("research_iteration", 0) + 1,
            "is_sufficient": True,
        }

    return mock_subgraph


def create_deterministic_mock_verifier(
    verdict: str = "SUPPORTED",
    confidence: float = 0.95,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Create a mock verifier node that verifies all claims in state."""

    def mock_verifier(state: ResearchState) -> dict[str, Any]:
        claims = state.get("claims", [])
        results = [
            VerificationResult(
                claim_id=claim.claim_id,
                verdict=verdict,
                confidence=confidence,
                reasoning=f"Deterministic verification of claim {claim.claim_id}.",
                evidence_ids=list(claim.evidence_ids),
            )
            for claim in claims
        ]
        return {"verification_results": results}

    return mock_verifier
