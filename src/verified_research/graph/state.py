from typing import NotRequired, TypedDict
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)


class ResearchState(TypedDict):
    """LangGraph state schema for verified research pipeline.

    Contract & State Ownership Boundary:
        1. Research Subgraph Internal State:
           - question: str
           - sources: list[Source]
           - findings: list[Finding]
           - critique: Critique
           - research_iteration: int (subgraph-internal loop counter: researcher -> analyst -> critic)
        2. Evidence & Verification State:
           - evidence: list[Evidence]
           - claims: list[Claim]
           - verification_results: list[VerificationResult]
        3. Parent / HITL Orchestration State:
           - human_review: HumanReview (structured review decision from human)
           - human_research_cycles: int (counter of human-requested research passes; separate from research_iteration)
           - human_feedback: str | None (targeted guidance for follow-up research cycles)
           - max_human_cycles_reached: bool (indicates human research cycle limit has been reached)

    Attributes:
        question: The substantive research question provided at graph initiation.
        sources: List of normalized Source objects retrieved during research.
        findings: List of synthesized Finding objects with source attributions.
        critique: Structured evaluation output assessing research sufficiency.
        research_iteration: Counter of completed internal research passes (0, 1, 2, 3).
        evidence: List of atomic, preserved Evidence excerpts extracted from sources.
        claims: List of specific, testable factual Claims referencing evidence IDs.
        verification_results: Evidence-grounded verification results for each claim.
        human_review: Structured review decision and feedback from human reviewer.
        human_research_cycles: Counter of human-requested research passes (bounded by MAX_HUMAN_RESEARCH_CYCLES).
        human_feedback: Human guidance or instructions for next research pass.
        max_human_cycles_reached: Flag indicating whether human research iteration cap was reached.
    """

    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
    critique: NotRequired[Critique]
    research_iteration: NotRequired[int]
    evidence: NotRequired[list[Evidence]]
    claims: NotRequired[list[Claim]]
    verification_results: NotRequired[list[VerificationResult]]
    human_review: NotRequired[HumanReview]
    human_research_cycles: NotRequired[int]
    human_feedback: NotRequired[str | None]
    max_human_cycles_reached: NotRequired[bool]

