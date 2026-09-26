"""Structural traceability validation, citation mapping, and coverage calculations."""

import logging
from verified_research.models.research import Claim, Evidence, Finding, Source

logger = logging.getLogger(__name__)


class TraceabilityError(ValueError):
    """Base error for structural traceability validation failures."""


class UnknownEvidenceError(TraceabilityError):
    """Raised when a Claim references an evidence_id not present in evidence state."""


class UnknownSourceError(TraceabilityError):
    """Raised when an Evidence item references a source_id not present in source state."""


class EmptyEvidenceError(TraceabilityError):
    """Raised when a Claim references an empty list of evidence IDs."""


class DuplicateIdError(TraceabilityError):
    """Raised when duplicate IDs are found within sources, evidence, or claims."""


def validate_traceability(
    claims: list[Claim],
    evidence: list[Evidence],
    sources: list[Source],
) -> None:
    """Perform deterministic structural validation of the Claim -> Evidence -> Source chain.

    Validation Rules:
        1. Identifier Uniqueness: Unique IDs within sources, evidence, and claims collections.
        2. Evidence -> Source: Every Evidence.source_id must exist in sources.
        3. Claim -> Evidence: Every Claim.evidence_ids must be non-empty and exist in evidence.

    IMPORTANT ARCHITECTURAL DISTINCTION:
        This validation verifies purely STRUCTURAL referential integrity.
        It confirms that identifier links exist and are well-formed.
        It DOES NOT perform semantic verification of whether the evidence text
        substantively confirms, refutes, or supports the claim statement.
        (Semantic verification is explicitly deferred to Phase 5).

    Args:
        claims: Collection of factual claims.
        evidence: Collection of preserved evidence excerpts.
        sources: Collection of retrieved grounding sources.

    Raises:
        DuplicateIdError: If duplicate source_id, evidence_id, or claim_id are detected.
        UnknownSourceError: If an evidence excerpt cites a nonexistent source_id.
        UnknownEvidenceError: If a claim cites a nonexistent evidence_id.
        EmptyEvidenceError: If a claim contains an empty evidence_ids list.
    """
    # 1. Validate Source ID uniqueness
    source_ids = set()
    for s in sources:
        if s.source_id in source_ids:
            raise DuplicateIdError(f"Duplicate source_id detected: '{s.source_id}'")
        source_ids.add(s.source_id)

    # 2. Validate Evidence ID uniqueness and link to Source
    evidence_ids = set()
    for ev in evidence:
        if ev.evidence_id in evidence_ids:
            raise DuplicateIdError(f"Duplicate evidence_id detected: '{ev.evidence_id}'")
        evidence_ids.add(ev.evidence_id)

        if ev.source_id not in source_ids:
            raise UnknownSourceError(
                f"Evidence '{ev.evidence_id}' references unknown source_id '{ev.source_id}'. "
                f"Valid source IDs: {sorted(source_ids)}"
            )

    # 3. Validate Claim ID uniqueness and link to Evidence
    claim_ids = set()
    for claim in claims:
        if claim.claim_id in claim_ids:
            raise DuplicateIdError(f"Duplicate claim_id detected: '{claim.claim_id}'")
        claim_ids.add(claim.claim_id)

        if not claim.evidence_ids:
            raise EmptyEvidenceError(
                f"Claim '{claim.claim_id}' must reference at least one evidence_id."
            )

        for ev_id in claim.evidence_ids:
            if ev_id not in evidence_ids:
                raise UnknownEvidenceError(
                    f"Claim '{claim.claim_id}' references unknown evidence_id '{ev_id}'. "
                    f"Valid evidence IDs: {sorted(evidence_ids)}"
                )

    logger.debug(
        "[TRACEABILITY] Validated %d claims against %d evidence items and %d sources.",
        len(claims),
        len(evidence),
        len(sources),
    )


def get_claim_sources(
    claim: Claim | str,
    evidence: list[Evidence],
    sources: list[Source],
    claims: list[Claim] | None = None,
) -> list[Source]:
    """Resolve the complete chain from a Claim through its Evidence to underlying Sources.

    Claim -> Evidence -> Source

    Args:
        claim: Either a Claim model instance or a string claim_id.
        evidence: Available evidence snapshots.
        sources: Available source models.
        claims: Optional list of claims needed if claim is passed as a string ID.

    Returns:
        Deduplicated list of Source instances grounding the claim's evidence.

    Raises:
        ValueError: If claim is a string ID and claims list is not provided or ID not found.
        UnknownEvidenceError: If claim cites an evidence_id not present in evidence.
        UnknownSourceError: If evidence cites a source_id not present in sources.
    """
    if isinstance(claim, str):
        if not claims:
            raise ValueError("Must provide 'claims' list when looking up claim by string ID.")
        claim_obj = next((c for c in claims if c.claim_id == claim), None)
        if claim_obj is None:
            raise ValueError(f"Claim ID '{claim}' not found in provided claims.")
    else:
        claim_obj = claim

    evidence_map = {e.evidence_id: e for e in evidence}
    source_map = {s.source_id: s for s in sources}

    matched_sources: list[Source] = []
    seen_source_ids = set()

    for ev_id in claim_obj.evidence_ids:
        if ev_id not in evidence_map:
            raise UnknownEvidenceError(
                f"Claim '{claim_obj.claim_id}' references unknown evidence_id '{ev_id}'."
            )
        ev = evidence_map[ev_id]

        if ev.source_id not in source_map:
            raise UnknownSourceError(
                f"Evidence '{ev.evidence_id}' references unknown source_id '{ev.source_id}'."
            )

        src = source_map[ev.source_id]
        if src.source_id not in seen_source_ids:
            seen_source_ids.add(src.source_id)
            matched_sources.append(src)

    return matched_sources


def calculate_citation_coverage(
    claims: list[Claim],
    valid_evidence: list[Evidence] | None = None,
) -> float:
    """Calculate the citation coverage ratio for a collection of claims.

    Definition:
        citation_coverage = (number of claims with valid evidence) / (total number of claims)

    NOTE:
        Citation coverage measures structural evidence presence.
        It DOES NOT imply claims are factually true or semantically verified.

    Args:
        claims: List of claims to evaluate.
        valid_evidence: Optional list of available evidence items. If provided,
                        claims are only counted as covered if all their evidence_ids
                        exist within valid_evidence.

    Returns:
        Float ratio between 0.0 and 1.0 (returns 0.0 if claims list is empty).
    """
    if not claims:
        return 0.0

    valid_ev_ids = {e.evidence_id for e in valid_evidence} if valid_evidence is not None else None

    covered_count = 0
    for claim in claims:
        if not claim.evidence_ids:
            continue
        if valid_ev_ids is not None:
            if all(ev_id in valid_ev_ids for ev_id in claim.evidence_ids):
                covered_count += 1
        else:
            covered_count += 1

    return covered_count / len(claims)


def generate_claims_and_evidence_from_findings(
    findings: list[Finding],
    sources: list[Source],
) -> tuple[list[Evidence], list[Claim]]:
    """Deterministic extractor bridging Phase 1 findings into Phase 4 Evidence and Claims.

    Creates:
        1. An Evidence excerpt snapshot for each source supporting a finding.
        2. A Claim instance corresponding to the atomic proposition asserted by the finding.

    Args:
        findings: Synthesized findings from the analyst.
        sources: Normalized research sources.

    Returns:
        Tuple of (list[Evidence], list[Claim]).
    """
    source_map = {s.source_id: s for s in sources}
    evidence_list: list[Evidence] = []
    claims_list: list[Claim] = []

    # Map (source_id, excerpt) to stable evidence_id
    evidence_lookup: dict[tuple[str, str], str] = {}

    for finding_idx, finding in enumerate(findings, start=1):
        claim_ev_ids: list[str] = []

        for sid in finding.source_ids:
            if sid in source_map:
                src = source_map[sid]
                # Extract excerpt snapshot (using full content or first substantive sentence)
                excerpt = src.content.strip()
                key = (sid, excerpt)

                if key not in evidence_lookup:
                    ev_id = f"ev_{len(evidence_list) + 1:03d}"
                    evidence_obj = Evidence(
                        evidence_id=ev_id,
                        source_id=sid,
                        text=excerpt,
                    )
                    evidence_list.append(evidence_obj)
                    evidence_lookup[key] = ev_id

                claim_ev_ids.append(evidence_lookup[key])

        if claim_ev_ids:
            claim_id = f"claim_{finding_idx:03d}"
            claims_list.append(
                Claim(
                    claim_id=claim_id,
                    text=finding.text,
                    evidence_ids=claim_ev_ids,
                )
            )

    return evidence_list, claims_list
