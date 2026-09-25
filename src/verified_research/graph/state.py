from typing import NotRequired, TypedDict
from verified_research.models.research import Critique, Finding, Source


class ResearchState(TypedDict):
    """LangGraph state schema for research pipeline with conditional routing.

    Contract:
        START: Pipeline is initialized with 'question' (and optionally 'research_iteration': 0).
        researcher: Consumes 'question', optional 'critique', increments 'research_iteration',
                    and populates/updates 'sources'.
        analyst: Consumes 'question' and 'sources', populates/updates 'findings'.
        critic: Consumes 'question', 'sources', and 'findings', populates 'critique'.
        router: Inspects 'critique' and 'research_iteration' to conditionally route
                back to 'researcher' or terminate at END.

    Attributes:
        question: The substantive research question provided at graph initiation.
        sources: List of normalized Source objects retrieved during research.
        findings: List of synthesized Finding objects with source attributions.
        critique: Structured evaluation output assessing research sufficiency.
        research_iteration: Counter of completed research passes (0, 1, 2, 3).
    """

    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
    critique: NotRequired[Critique]
    research_iteration: NotRequired[int]

