from typing import NotRequired, TypedDict
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    ResearchReuseDecision,
    Source,
    SupervisorDecision,
    VerificationResult,
)


class ResearchState(TypedDict):
    """LangGraph state schema for verified research pipeline.

    Contract & State Ownership Boundary:
        1. Conversation / Session Context:
           - question: str (current substantive research question)
           - follow_up_question: str | None (optional follow-up query submitted to ongoing session)
           - previous_questions: list[str] (history of previously researched questions in this thread)
           - reuse_decision: ResearchReuseDecision (structured sufficiency evaluation: REUSE vs RESEARCH_MORE)
        2. Research Subgraph Internal State:
           - sources: list[Source]
           - findings: list[Finding]
           - critique: Critique
           - research_iteration: int (subgraph-internal loop counter: researcher -> analyst -> critic)
        3. Evidence & Verification State:
           - evidence: list[Evidence]
           - claims: list[Claim]
           - verification_results: list[VerificationResult]
        4. Parent / HITL Orchestration State:
           - human_review: HumanReview (structured review decision from human)
           - human_research_cycles: int (counter of human-requested research passes; separate from research_iteration)
           - human_feedback: str | None (targeted guidance for follow-up research cycles)
           - max_human_cycles_reached: bool (indicates human research cycle limit has been reached)
        5. Supervisor Orchestration State:
           - supervisor_steps: int (counter of supervisor loop transitions; separate from iterations & cycles)
           - supervisor_decision: SupervisorDecision (structured decision directing worker execution)
           - supervisor_termination_reason: str | None (explicit explanation if stopped safely or capped)

    Attributes:
        question: The substantive research question currently active in the graph.
        follow_up_question: Explicit follow-up query submitted to an existing session.
        previous_questions: Chronological list of previously researched questions in the thread.
        reuse_decision: Structured decision whether existing research is sufficient or more is needed.
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
        supervisor_steps: Counter of completed supervisor transitions (bounded by MAX_SUPERVISOR_STEPS).
        supervisor_decision: Most recent structured supervisor decision.
        supervisor_termination_reason: Explicit termination rationale when supervisor halts execution.
    """

    question: str
    follow_up_question: NotRequired[str | None]
    previous_questions: NotRequired[list[str]]
    reuse_decision: NotRequired[ResearchReuseDecision]
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
    critique: NotRequired[Critique]
    research_iteration: NotRequired[int]
    evidence: NotRequired[list[Evidence]]
    claims: NotRequired[list[Claim]]
    verification_results: NotRequired[list[VerificationResult]]
    human_review: NotRequired[HumanReview | None]
    human_research_cycles: NotRequired[int]
    human_feedback: NotRequired[str | None]
    max_human_cycles_reached: NotRequired[bool]
    supervisor_steps: NotRequired[int]
    supervisor_decision: NotRequired[SupervisorDecision]
    supervisor_termination_reason: NotRequired[str | None]


