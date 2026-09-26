"""Local demo script executing graph with mocked search, analyst, and critic."""

import logging
from typing import Any
from verified_research.agents.critic import create_critic_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.graph import create_research_graph
from verified_research.models.research import Critique, Finding, Source
from verified_research.models.traceability import (
    calculate_citation_coverage,
    generate_claims_and_evidence_from_findings,
    get_claim_sources,
    validate_traceability,
)
from verified_research.tools.search import SearchService

# Configure logging format to observe state transitions cleanly
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DemoSearchService(SearchService):
    """Demo search provider returning query-specific mock results."""

    def __init__(self) -> None:
        self.call_count = 0

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        self.call_count += 1
        if "fault tolerance" in query.lower():
            return [
                Source(
                    source_id=f"src_{self.call_count:03d}",
                    title="Quantum Fault-Tolerance Thresholds in Surface Codes",
                    url="https://phys-review.org/qec-surface-codes",
                    content=(
                        "Surface code scaling past the fault-tolerant threshold proves that "
                        "logical error rates decrease exponentially with code distance."
                    ),
                )
            ]
        return [
            Source(
                source_id=f"src_{self.call_count:03d}",
                title="Superconducting Qubit Advances in 2026",
                url="https://quantum-research.org/advances-2026",
                content=(
                    "Recent experiments demonstrate 2-qubit gate fidelities exceeding 99.9% "
                    "using fluxonium and transmon architecture hybrids."
                ),
            )
        ]


def main() -> None:
    print("=" * 70)
    print("VERIFIED RESEARCH AGENT - EVIDENCE & CLAIM MODEL DEMO")
    print("Traceability: Claim -> Evidence -> Source")
    print("=" * 70)

    # 1. Setup search service
    search_service = DemoSearchService()
    researcher = create_researcher_node(search_client=search_service)

    # 2. Setup mock analyst producing findings, evidence snapshots, and claims
    def mock_analyst(state: dict[str, Any]) -> dict[str, Any]:
        sources = state.get("sources", [])
        findings = []
        for i, s in enumerate(sources, start=1):
            findings.append(
                Finding(
                    finding_id=f"finding_{i:03d}",
                    text=f"Insight synthesized from: {s.title}",
                    source_ids=[s.source_id],
                )
            )

        evidence, claims = generate_claims_and_evidence_from_findings(findings, sources)
        validate_traceability(claims, evidence, sources)
        logger.info(
            "[Analyst] findings=%d evidence=%d claims=%d",
            len(findings),
            len(evidence),
            len(claims),
        )
        return {"findings": findings, "evidence": evidence, "claims": claims}

    # 3. Setup mock critic with 2-pass progression (weak on pass 1, good on pass 2)
    critic_invocations = 0

    def mock_critic(state: dict[str, Any]) -> dict[str, Critique]:
        nonlocal critic_invocations
        critic_invocations += 1
        if critic_invocations == 1:
            critique = Critique(
                quality_score=0.62,
                missing_topics=["Fault-tolerance threshold details"],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=["quantum fault tolerance thresholds surface codes"],
                should_research_again=True,
            )
        else:
            critique = Critique(
                quality_score=0.91,
                missing_topics=[],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=[],
                should_research_again=False,
            )
        logger.info(
            "[Critic] score=%.2f continue=%s",
            critique.quality_score,
            critique.should_research_again,
        )
        return {"critique": critique}

    # 4. Build and compile graph
    graph = create_research_graph(
        custom_researcher=researcher,
        custom_analyst=mock_analyst,
        custom_critic=mock_critic,
    )

    # 5. Invoke graph
    user_question = "What are the recent milestones in quantum computing hardware and error correction?"
    print(f"\n[USER QUESTION]: {user_question}\n")

    initial_state = {"question": user_question}
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FINAL GRAPH STATE")
    print("=" * 70)
    print(f"Question: {final_state['question']}")
    print(f"Completed Research Iterations: {final_state['research_iteration']}\n")

    critique = final_state.get("critique")
    if critique:
        print(f"Final Critique Score: {critique.quality_score:.2f}")
        print(f"Should Research Again: {critique.should_research_again}\n")

    sources = final_state.get("sources", [])
    evidence = final_state.get("evidence", [])
    claims = final_state.get("claims", [])

    print(f"Sources Accumulated ({len(sources)}):")
    for s in sources:
        print(f"  [{s.source_id}] {s.title} ({s.url})")

    print(f"\nEvidence Snapshots ({len(evidence)}):")
    for ev in evidence:
        print(f"  [{ev.evidence_id}] (Source: {ev.source_id}) Excerpt: '{ev.text[:65]}...'")

    print(f"\nFactual Claims ({len(claims)}):")
    for c in claims:
        print(f"  [{c.claim_id}] {c.text}")
        print(f"      Grounding Evidence IDs: {c.evidence_ids}")

    coverage = calculate_citation_coverage(claims, evidence)
    print(f"\nCitation Coverage: {coverage:.1%}")

    print("\n" + "=" * 70)
    print("CLAIM -> EVIDENCE -> SOURCE RESOLUTION TRACE")
    print("=" * 70)
    for c in claims:
        grounding_sources = get_claim_sources(c, evidence, sources)
        source_titles = [f"[{s.source_id}] {s.title}" for s in grounding_sources]
        print(f"Claim '{c.claim_id}' -> Evidence {c.evidence_ids} -> Sources: {source_titles}")

    print("\nExecution complete. Structural traceability and citation coverage verified.")


if __name__ == "__main__":
    main()
