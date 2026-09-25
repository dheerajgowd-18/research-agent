"""Unit tests for researcher node and search abstraction."""

from unittest.mock import MagicMock
import pytest
from verified_research.agents.researcher import create_researcher_node
from verified_research.models.research import Source
from verified_research.tools.search import (
    MissingApiKeyError,
    SearchError,
    TavilySearchClient,
)


class DummySearchClient:
    """Mock search client for testing researcher node."""

    def __init__(self, sources: list[Source] | None = None) -> None:
        self._sources = sources if sources is not None else [
            Source(
                source_id="src_001",
                title="Mock Source 1",
                url="https://example.com/1",
                content="Mock content 1",
            ),
            Source(
                source_id="src_002",
                title="Mock Source 2",
                url="https://example.com/2",
                content="Mock content 2",
            ),
        ]

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        return self._sources


class TestResearcherNode:
    """Tests for researcher node contract and behavior."""

    def test_researcher_returns_normalized_sources(self):
        client = DummySearchClient()
        node = create_researcher_node(search_client=client)

        state = {"question": "What are topological quantum computers?"}
        result = node(state)

        assert "sources" in result
        sources = result["sources"]
        assert len(sources) == 2
        assert isinstance(sources[0], Source)
        assert sources[0].source_id == "src_001"
        assert sources[1].source_id == "src_002"

    def test_researcher_empty_search_results(self):
        client = DummySearchClient(sources=[])
        node = create_researcher_node(search_client=client)

        state = {"question": "Obscure unknown question"}
        result = node(state)

        assert "sources" in result
        assert result["sources"] == []

    def test_researcher_missing_question_fails(self):
        node = create_researcher_node(search_client=DummySearchClient())

        with pytest.raises(ValueError, match="non-empty 'question'"):
            node({"question": ""})

        with pytest.raises(ValueError, match="non-empty 'question'"):
            node({})  # type: ignore


class TestTavilySearchClient:
    """Tests for Tavily search client and result normalization."""

    def test_tavily_missing_api_key_raises_error(self, monkeypatch):
        # Ensure no API key is present in settings
        monkeypatch.setenv("TAVILY_API_KEY", "")
        client = TavilySearchClient(api_key="")

        with pytest.raises(MissingApiKeyError, match="Tavily API key is missing"):
            client.search("query")

    def test_tavily_normalization_success(self, monkeypatch):
        client = TavilySearchClient(api_key="mock_key")
        mock_raw_client = MagicMock()
        mock_raw_client.search.return_value = {
            "results": [
                {
                    "title": "Quantum Supremacy Demo",
                    "url": "https://nature.com/articles/s41586",
                    "content": "A milestone in computational capability was reached.",
                },
                {
                    "title": "Scaling Quantum Systems",
                    "url": "https://science.org/doi/10.1126",
                    "content": "Addressing decoherence through material improvements.",
                },
            ]
        }
        monkeypatch.setattr(client, "_get_client", lambda: mock_raw_client)

        sources = client.search("quantum progress")
        assert len(sources) == 2
        assert sources[0].source_id == "src_001"
        assert sources[0].title == "Quantum Supremacy Demo"
        assert sources[1].source_id == "src_002"
        assert sources[1].url == "https://science.org/doi/10.1126"

    def test_tavily_handles_empty_query_cleanly(self):
        client = TavilySearchClient(api_key="mock_key")
        sources = client.search("   ")
        assert sources == []

    def test_tavily_api_error_wrapped(self, monkeypatch):
        client = TavilySearchClient(api_key="mock_key")
        mock_raw_client = MagicMock()
        mock_raw_client.search.side_effect = RuntimeError("Network timeout")
        monkeypatch.setattr(client, "_get_client", lambda: mock_raw_client)

        with pytest.raises(SearchError, match="Tavily search request failed"):
            client.search("any query")
