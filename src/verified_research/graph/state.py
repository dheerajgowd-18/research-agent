from typing import NotRequired, TypedDict
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    Source,
    VerificationResult,
)


class ResearchState(TypedDict):
    """LangGraph state schema for verified research pipeline.

    Contract:
        START: Pipeline is initialized with 'question' (and optionally 'research_iteration': 0).
        researcher: Consumes 'question', optional 'critique', increments 'research_iteration',
                    and populates/updates 'sources'.
        analyst: Consumes 'question' and 'sources', populates/updates 'findings',
                 'evidence', and 'claims'.
        critic: Consumes 'question', 'sources', and 'findings', populates 'critique'.
        router: Inspects 'critique' and 'research_iteration' to conditionally route
                back to 'researcher' or terminate child subgraph.
        verifier: Consumes 'claims' and 'evidence', produces 'verification_results'.
        END: Pipeline concludes with claims, evidence, and verification_results.

    Attributes:
        question: The substantive research question provided at graph initiation.
        sources: List of normalized Source objects retrieved during research.
        findings: List of synthesized Finding objects with source attributions.
        critique: Structured evaluation output assessing research sufficiency.
        research_iteration: Counter of completed research passes (0, 1, 2, 3).
        evidence: List of atomic, preserved Evidence excerpts extracted from sources.
        claims: List of specific, testable factual Claims referencing evidence IDs.
        verification_results: Evidence-grounded verification results for each claim.
    """

    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
    critique: NotRequired[Critique]
    research_iteration: NotRequired[int]
    evidence: NotRequired[list[Evidence]]
    claims: NotRequired[list[Claim]]
    verification_results: NotRequired[list[VerificationResult]]
