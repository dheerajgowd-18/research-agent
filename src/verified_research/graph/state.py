"""State schema definition for the Verified Research Agent graph."""

from typing import NotRequired, TypedDict
from verified_research.models.research import Finding, Source


class ResearchState(TypedDict):
    """LangGraph state schema for Phase 1 research pipeline.

    Contract:
        START: Pipeline is initialized with 'question'.
        researcher: Consumes 'question', appends/populates 'sources'.
        analyst: Consumes 'question' and 'sources', appends/populates 'findings'.
        END: Graph concludes with all three fields populated.

    Attributes:
        question: The substantive research question provided at graph initiation.
        sources: List of normalized Source objects retrieved during research.
        findings: List of synthesized Finding objects with source attributions.
    """

    question: str
    sources: NotRequired[list[Source]]
    findings: NotRequired[list[Finding]]
