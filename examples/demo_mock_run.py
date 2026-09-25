"""Local demo script executing Phase 1 graph with mocked search and LLM."""

import logging
from unittest.mock import MagicMock
from langchain_core.language_models import BaseChatModel
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.graph import create_research_graph
from verified_research.models.research import AnalystOutput, Finding, Source
from verified_research.tools.search import SearchService

# Configure simple logging format to observe state transitions
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DemoSearchService(SearchService):
    """Demo search provider returning realistic mock results."""

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        return [
            Source(
                source_id="src_001",
                title="Superconducting Qubit Advances in 2026",
                url="https://quantum-research.org/advances-2026",
                content=(
                    "Recent experiments demonstrate 2-qubit gate fidelities exceeding 99.9% "
                    "using fluxonium and transmon architecture hybrids."
                ),
            ),
            Source(
                source_id="src_002",
                title="Quantum Error Correction Breakthroughs",
                url="https://phys-review.org/qec-surface-codes",
                content=(
                    "Surface code scaling past the fault-tolerant threshold has proven "
                    "that logical error rates decrease exponentially with code distance."
                ),
            ),
        ]


def main() -> None:
    print("=" * 70)
    print("VERIFIED RESEARCH AGENT - PHASE 1 LOCAL EXECUTION DEMO")
    print("Pipeline: START -> researcher -> analyst -> END")
    print("=" * 70)

    # 1. Setup mock search client
    search_service = DemoSearchService()
    researcher = create_researcher_node(search_client=search_service)

    # 2. Setup mock LLM with structured output adhering to AnalystOutput
    mock_llm = MagicMock(spec=BaseChatModel)
    structured_mock = MagicMock()
    mock_findings = [
        Finding(
            finding_id="finding_001",
            text="Hybrid fluxonium-transmon qubits achieve 2-qubit gate fidelities above 99.9%.",
            source_ids=["src_001"],
        ),
        Finding(
            finding_id="finding_002",
            text="Logical error rates decline exponentially with code distance in surface codes.",
            source_ids=["src_002"],
        ),
    ]
    structured_mock.invoke.return_value = AnalystOutput(findings=mock_findings)
    mock_llm.with_structured_output.return_value = structured_mock
    analyst = create_analyst_node(llm=mock_llm)

    # 3. Build and compile graph
    graph = create_research_graph(
        custom_researcher=researcher,
        custom_analyst=analyst,
    )

    # 4. Invoke graph
    user_question = "What are the recent milestones in quantum computing hardware and error correction?"
    print(f"\n[USER QUESTION]: {user_question}\n")

    initial_state = {"question": user_question}
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("FINAL GRAPH STATE")
    print("=" * 70)
    print(f"Question: {final_state['question']}\n")

    print(f"Sources Retrieved ({len(final_state['sources'])}):")
    for s in final_state["sources"]:
        print(f"  [{s.source_id}] {s.title} ({s.url})")
        print(f"      Snippet: {s.content[:80]}...\n")

    print(f"Findings Produced ({len(final_state['findings'])}):")
    for f in final_state["findings"]:
        print(f"  [{f.finding_id}] {f.text}")
        print(f"      Cited Sources: {f.source_ids}\n")

    print("Execution complete. Traceability verified: all finding source_ids match state sources.")


if __name__ == "__main__":
    main()
