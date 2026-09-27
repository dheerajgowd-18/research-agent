"""Integration tests for the LangGraph research pipeline."""

from unittest.mock import MagicMock
import pytest
from langchain_core.language_models import BaseChatModel
from verified_research.agents.analyst import create_analyst_node
from verified_research.agents.researcher import create_researcher_node
from verified_research.graph.graph import build_research_graph, create_research_graph
from verified_research.models.research import (
    AnalystOutput,
    Critique,
    Finding,
    Source,
    VerificationResult,
)


class MockSearchService:
    """Mock search service returning predictable Source objects."""

    def __init__(self, sources: list[Source] | None = None) -> None:
        self.sources = sources if sources is not None else [
            Source(
                source_id="src_001",
                title="Graph Neural Networks for Chemistry",
                url="https://arxiv.org/abs/gnn-chem",
                content="GNNs accurately predict molecular properties.",
            ),
            Source(
                source_id="src_002",
                title="Transformer Applications in Drug Discovery",
                url="https://nature.com/articles/transformers-drug",
                content="Attention mechanisms model binding affinity effectively.",
            ),
        ]

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        return self.sources


@pytest.fixture
def mock_llm_factory():
    def _create(findings: list[Finding]):
        mock_llm = MagicMock(spec=BaseChatModel)
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = AnalystOutput(findings=findings)
        mock_llm.with_structured_output.return_value = structured_mock
        return mock_llm

    return _create


class TestResearchGraphExecution:
    """Tests for end-to-end execution of the LangGraph state architecture."""

    def test_full_graph_execution_success(self, mock_llm_factory):
        expected_findings = [
            Finding(
                finding_id="finding_001",
                text="GNNs are established tools for molecular property prediction.",
                source_ids=["src_001"],
            ),
            Finding(
                finding_id="finding_002",
                text="Transformers model chemical binding affinity with high precision.",
                source_ids=["src_002"],
            ),
        ]
        mock_llm = mock_llm_factory(expected_findings)

        researcher = create_researcher_node(search_client=MockSearchService())
        analyst = create_analyst_node(llm=mock_llm)
        critic = lambda state: {
            "critique": Critique(
                quality_score=0.95,
                missing_topics=[],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=[],
                should_research_again=False,
            )
        }

        mock_verifier = lambda claim, evidence: VerificationResult(
            claim_id=claim.claim_id,
            verdict="SUPPORTED",
            confidence=0.95,
            reasoning="Verified in graph integration test.",
            evidence_ids=claim.evidence_ids,
        )

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
            custom_verifier=mock_verifier,
        )

        initial_state = {"question": "How are neural architectures used in drug discovery?"}
        final_state = graph.invoke(initial_state)

        # Assert full state populated
        assert final_state["question"] == initial_state["question"]
        assert "sources" in final_state
        assert "findings" in final_state

        sources = final_state["sources"]
        findings = final_state["findings"]

        assert len(sources) == 2
        assert len(findings) == 2

        # Assert traceability: each finding's source_ids must map to actual sources
        source_id_map = {s.source_id: s for s in sources}
        for finding in findings:
            for sid in finding.source_ids:
                assert sid in source_id_map
                assert isinstance(source_id_map[sid], Source)

        assert "verification_results" in final_state
        assert len(final_state["verification_results"]) == len(final_state.get("claims", []))

    def test_graph_handles_empty_search_results(self):
        # Empty search results -> researcher returns [], analyst returns []
        researcher = create_researcher_node(search_client=MockSearchService(sources=[]))
        analyst = create_analyst_node(llm=MagicMock(spec=BaseChatModel))

        critic = lambda state: {
            "critique": Critique(
                quality_score=0.0,
                missing_topics=["No sources"],
                weak_findings=[],
                citation_gaps=[],
                recommended_queries=[],
                should_research_again=False,
            )
        }

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
            custom_critic=critic,
        )

        initial_state = {"question": "Completely unknown topic with 0 hits"}
        final_state = graph.invoke(initial_state)

        assert final_state["question"] == initial_state["question"]
        assert final_state["sources"] == []
        assert final_state["findings"] == []

    def test_graph_fails_on_empty_question(self):
        researcher = create_researcher_node(search_client=MockSearchService())
        analyst = create_analyst_node()

        graph = create_research_graph(
            custom_researcher=researcher,
            custom_analyst=analyst,
        )

        with pytest.raises(ValueError, match="non-empty 'question'"):
            graph.invoke({"question": ""})

    def test_graph_node_structure(self):
        builder = build_research_graph()
        nodes = list(builder.nodes.keys())
        assert "research" in nodes
        assert "verifier" in nodes
        assert "human_review" in nodes
        assert len(nodes) == 3



