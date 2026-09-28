from typing import Literal
from typing_extensions import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator



class Source(BaseModel):
    """Normalized web or document source retrieved during research.

    Attributes:
        source_id: Unique identifier for the source (e.g., 'src_001').
        title: Title of the web page or document.
        url: Direct URL of the source.
        content: Extracted textual content or snippet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the source, e.g. 'src_001'.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Title of the source document or article.",
    )
    url: str = Field(
        ...,
        min_length=1,
        description="URL where the source was retrieved.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Extracted textual content or summary snippet.",
    )


class Finding(BaseModel):
    """A synthesized research finding grounded in specific sources.

    Attributes:
        finding_id: Unique identifier for the finding (e.g., 'finding_001').
        text: Synthesized insight or factual summary.
        source_ids: List of source IDs supporting this finding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the finding, e.g. 'finding_001'.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The substantive text of the synthesized finding.",
    )
    source_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of source_ids that directly support this finding.",
    )


class Evidence(BaseModel):
    """A specific textual excerpt/snapshot extracted from a Source.

    Evidence preserves the exact textual context from a source at the time of research,
    ensuring that downstream claim verification does not rely on mutable external URLs.

    Attributes:
        evidence_id: Unique identifier for the evidence snapshot (e.g., 'ev_001').
        source_id: Identifier of the parent Source from which this excerpt was taken.
        text: The preserved textual excerpt/snapshot used for factual grounding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the evidence excerpt, e.g. 'ev_001'.",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the Source from which this evidence is drawn, e.g. 'src_001'.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Preserved textual snapshot/excerpt from the source.",
    )


class Claim(BaseModel):
    """An atomic, testable factual statement that can be verified against evidence.

    Unlike a high-level analytical Finding (which synthesizes multiple themes),
    a Claim is an atomic factual proposition tied directly to one or more Evidence excerpts.

    Attributes:
        claim_id: Unique identifier for the claim (e.g., 'claim_001').
        text: The specific factual assertion to be verified.
        evidence_ids: Non-empty list of evidence IDs directly grounding this claim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the claim, e.g. 'claim_001'.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="The substantive factual assertion of the claim.",
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of evidence_ids directly supporting this claim (at least 1 required).",
    )



class AnalystOutput(BaseModel):
    """Structured container model for LLM responses during analysis."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(
        default_factory=list,
        description="Collection of synthesized findings with source attributions.",
    )


class Critique(BaseModel):
    """Structured evaluation output produced by the critic node.

    Attributes:
        quality_score: Quantitative assessment of research quality (0.0 to 1.0).
        missing_topics: Substantive topics or angles missing from current findings.
        weak_findings: Specific findings that need stronger evidence or clarification.
        citation_gaps: Identified claims lacking direct or credible source support.
        recommended_queries: Targeted queries recommended for follow-up research passes.
        should_research_again: Boolean flag indicating if another research cycle is warranted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall research quality score between 0.0 and 1.0.",
    )
    missing_topics: list[str] = Field(
        default_factory=list,
        description="Substantive topics or angles missing from current findings.",
    )
    weak_findings: list[str] = Field(
        default_factory=list,
        description="Specific findings that need stronger evidence or clarification.",
    )
    citation_gaps: list[str] = Field(
        default_factory=list,
        description="Identified claims lacking direct or credible source support.",
    )
    recommended_queries: list[str] = Field(
        default_factory=list,
        description="Targeted queries recommended for follow-up research passes.",
    )
    should_research_again: bool = Field(
        ...,
        description="True if further research iteration is warranted; False if research is sufficient.",
    )


VerdictType = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]


class VerificationResult(BaseModel):
    """Result of evaluating a specific claim against its cited evidence.

    Attributes:
        claim_id: Identifier of the Claim being evaluated.
        verdict: Categorical evaluation: 'SUPPORTED', 'PARTIAL', or 'UNSUPPORTED'.
        confidence: Numerical score between 0.0 and 1.0 representing model certainty.
        reasoning: Explicit rationale explaining why the evidence supports, partially supports,
                   or fails to support the claim.
        evidence_ids: List of evidence IDs that were examined for this verdict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the Claim evaluated.",
    )
    verdict: VerdictType = Field(
        ...,
        description="Verification outcome: SUPPORTED, PARTIAL, or UNSUPPORTED.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model-reported confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Justification for the verdict based strictly on supplied evidence.",
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of evidence IDs supplied for this evaluation.",
    )


HumanActionType = Literal["approve", "edit", "research_more", "reject"]


class HumanReview(BaseModel):
    """Structured human review input and decision record.

    Attributes:
        action: Structured decision action ('approve', 'edit', 'research_more', 'reject').
        feedback: Optional textual guidance or instructions (e.g. for follow-up research passes).
        edited_claims: Optional list of validated Claim instances (required if action is 'edit').
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: HumanActionType = Field(
        ...,
        description="Review action: approve, edit, research_more, or reject.",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional human guidance or feedback for research cycles.",
    )
    edited_claims: list[Claim] | None = Field(
        default=None,
        description="List of edited Claim objects required when action is 'edit'.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        if self.action == "edit":
            if not self.edited_claims:
                raise ValueError("Action 'edit' requires a non-empty list of 'edited_claims'.")
        elif self.edited_claims is not None:
            raise ValueError(f"Action '{self.action}' cannot include 'edited_claims'.")
        return self


ReuseDecisionType = Literal["REUSE", "RESEARCH_MORE"]


class ResearchReuseDecision(BaseModel):
    """Structured evaluation of whether existing research is sufficient for a follow-up question.

    Attributes:
        decision: 'REUSE' if existing evidence/findings are sufficient, else 'RESEARCH_MORE'.
        reasoning: Explicit justification detailing why existing research is sufficient or deficient.
        missing_topics: Specific topics, temporal ranges, or entities missing from current research.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: ReuseDecisionType = Field(
        ...,
        description="Whether existing research is sufficient (REUSE) or new research is needed (RESEARCH_MORE).",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Explicit reasoning detailing why existing evidence is sufficient or insufficient.",
    )
    missing_topics: list[str] = Field(
        default_factory=list,
        description="List of specific gaps, temporal requirements, or topics not covered in existing research.",
    )


SupervisorWorkerType = Literal["research", "verify", "human_review", "writer", "finish"]


class SupervisorDecision(BaseModel):
    """Structured decision produced by the Supervisor orchestrator.

    Attributes:
        next_worker: The target worker node to execute next ('research', 'verify', 'human_review', 'writer', 'finish').
        reasoning: Explicit rationale justifying the orchestration decision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    next_worker: SupervisorWorkerType = Field(
        ...,
        description="The specialized worker to invoke next, or 'finish' to conclude orchestration.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Explicit justification detailing why this worker was chosen based on current state.",
    )


class ReportSection(BaseModel):
    """A thematic section in the final user-facing research report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
        description="Heading of the section.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Substantive synthesized text of the section with inline citations.",
    )
    claim_ids: list[str] = Field(
        default_factory=list,
        description="List of verified claim IDs grounded in this section.",
    )


class Citation(BaseModel):
    """An explicit source attribution linking text to preserved evidence and source documents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_id: str = Field(
        ...,
        min_length=1,
        description="Citation identifier or marker (e.g. '[1]').",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description="ID of the referenced source document.",
    )
    claim_ids: list[str] = Field(
        default_factory=list,
        description="Claim IDs supported by this citation.",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Preserved evidence IDs referenced by this citation.",
    )
    source_title: str = Field(
        ...,
        min_length=1,
        description="Title of the referenced source document.",
    )
    source_url: str = Field(
        ...,
        min_length=1,
        description="URL of the referenced source document.",
    )


class FinalReport(BaseModel):
    """A complete, structured, and verified research response produced by the Writer node.

    Attributes:
        title: Concise editorial title of the research investigation.
        summary: Executive summary of key verified conclusions.
        answer: Full synthesized research response incorporating inline citations.
        sections: Thematic sub-sections breaking down findings.
        citations: Mapped source citations linking text to evidence.
        claim_references: All verified claim IDs incorporated in the report.
        verified_claim_count: Number of verified claims incorporated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(
        ...,
        min_length=1,
        description="Concise editorial title of the research report.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="Executive summary of the verified findings.",
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="Comprehensive synthesized answer with inline citations.",
    )
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Structured sections detailing the research findings.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Mapped citations linking statements to evidence and sources.",
    )
    claim_references: list[str] = Field(
        default_factory=list,
        description="List of verified claim IDs referenced in the report.",
    )
    verified_claim_count: int = Field(
        default=0,
        ge=0,
        description="Count of verified claims incorporated into the report.",
    )





