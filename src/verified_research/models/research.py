"""Data models for research sources, findings, and analyst outputs."""

from pydantic import BaseModel, ConfigDict, Field


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

