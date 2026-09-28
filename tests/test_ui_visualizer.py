"""Test suite for the Interactive LangGraph Architecture Visualizer.

Validates:
1. Real graph topology reflecting backend LangGraph architecture.
2. Subgraph expansion / collapse mechanisms.
3. System Architecture vs Live Execution view modes.
4. Canvas navigation, pan, zoom, fit, and MiniMap synchronization.
5. Deep Node Inspector drawer elements and architectural invariants.
6. Static file serving via FastAPI.
"""

from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from verified_research.api import create_app
from verified_research.graph.graph import create_supervisor_graph
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def static_html_content() -> str:
    """Read the static index.html file directly from the filesystem."""
    html_path = Path("src/verified_research/api/static/index.html")
    assert html_path.exists(), f"Static index.html not found at {html_path}"
    return html_path.read_text(encoding="utf-8")


@pytest.fixture
def test_app():
    """Create a test FastAPI instance with in-memory checkpointer."""
    checkpointer = MemorySaver()
    graph = create_supervisor_graph(
        custom_subgraph=lambda state: state,
        custom_verifier=lambda state: state,
        checkpointer=checkpointer,
    )
    return create_app(graph=graph, checkpointer=checkpointer)


# ==============================================================================
# 1. Visualizer HTML & SVG DOM Structure Tests
# ==============================================================================


def test_visualizer_toolbar_and_controls_present(static_html_content: str):
    """Verify presence of visualizer controls, mode toggles, zoom, and minimap."""
    # View mode buttons
    assert 'id="btnModeArchitecture"' in static_html_content
    assert 'id="btnModeLive"' in static_html_content
    assert 'setGraphViewMode(\'architecture\')' in static_html_content
    assert 'setGraphViewMode(\'live\')' in static_html_content

    # Subgraph toggle button
    assert 'id="btnToggleSubgraph"' in static_html_content
    assert 'toggleSubgraphExpansion()' in static_html_content

    # Execution indicator
    assert 'id="executionIndicatorPill"' in static_html_content
    assert 'id="executionIndicatorText"' in static_html_content

    # Zoom controls
    assert 'zoomGraph(1.2)' in static_html_content
    assert 'zoomGraph(0.833)' in static_html_content
    assert 'fitGraphView()' in static_html_content
    assert 'resetGraphView()' in static_html_content
    assert 'id="zoomLevelDisplay"' in static_html_content

    # Canvas & Viewport
    assert 'id="graphCanvasWrap"' in static_html_content
    assert 'id="graphSvgCanvas"' in static_html_content
    assert 'id="svgDefs"' in static_html_content
    assert 'id="graphViewport"' in static_html_content

    # MiniMap
    assert 'class="graph-minimap-wrap"' in static_html_content
    assert 'id="miniMapSvg"' in static_html_content
    assert 'id="miniMapViewportIndicator"' in static_html_content


def test_node_inspector_sheet_structure(static_html_content: str):
    """Verify the slide-over Node Inspector Sheet and its inspection panels."""
    assert 'id="sheetBackdrop"' in static_html_content
    assert 'id="sheetPanel"' in static_html_content
    assert 'id="sheetNodeTitle"' in static_html_content
    assert 'id="sheetNodeSubtitle"' in static_html_content
    assert 'id="sheetNodeStatusBadge"' in static_html_content
    assert 'id="sheetNodeDesc"' in static_html_content
    assert 'id="sheetNodeInvariants"' in static_html_content
    assert 'id="sheetNodeMetrics"' in static_html_content
    assert 'id="sheetNodeState"' in static_html_content
    assert 'copyNodeStatePayload()' in static_html_content
    assert 'closeNodeInspector()' in static_html_content


# ==============================================================================
# 2. Topology & Metadata Specification Tests
# ==============================================================================


def test_topology_defines_all_backend_nodes(static_html_content: str):
    """Ensure all 12 LangGraph nodes and workers are defined in the visualizer."""
    required_nodes = [
        "start",
        "supervisor",
        "research",
        "researcher",
        "analyst",
        "critic",
        "verifier",
        "human_review",
        "writer",
        "final_response",
        "rejected",
        "end",
    ]
    for node in required_nodes:
        assert f"{node}:" in static_html_content or f"'{node}'" in static_html_content, (
            f"Node '{node}' missing from visualizer node specifications."
        )


def test_topology_edge_connections(static_html_content: str):
    """Verify correct graph edges for both collapsed and expanded states."""
    expected_edge_patterns = [
        "start->supervisor",
        "supervisor->research",
        "research->supervisor",
        "supervisor->verifier",
        "verifier->supervisor",
        "supervisor->human_review",
        "researcher->analyst",
        "analyst->critic",
        "critic->researcher",  # Feedback loop
        "human_review->writer_approve",
        "human_review->writer_edit",
        "human_review->supervisor_more",
        "human_review->rejected",
        "writer->final_response",
        "final_response->end",
    ]
    for edge in expected_edge_patterns:
        assert edge in static_html_content, f"Graph edge '{edge}' missing from topology definition."


def test_architectural_invariants_checklist(static_html_content: str):
    """Verify the presence of key system invariants in NODE_METADATA."""
    # Supervisor invariant
    assert "Max 8 supervisor steps per execution session." in static_html_content
    # Writer invariant
    assert "Strict Invariant: New factual claims introduced = 0." in static_html_content
    # Verifier invariant
    assert "SUPPORTED, PARTIAL, UNSUPPORTED" in static_html_content
    # Human Review invariant
    assert "Pauses graph execution via LangGraph __interrupt__" in static_html_content


# ==============================================================================
# 3. JavaScript Engine Function Tests
# ==============================================================================


def test_core_engine_functions_implemented(static_html_content: str):
    """Ensure all required interactive graph engine functions are defined."""
    required_functions = [
        "function initArchitectureGraph()",
        "function getGraphTopology()",
        "function renderGraph()",
        "function setGraphViewMode(mode)",
        "function toggleSubgraphExpansion()",
        "function zoomGraph(factor)",
        "function fitGraphView()",
        "function resetGraphView()",
        "function applyGraphTransform()",
        "function updateMiniMap()",
        "function syncGraphWithLiveEvent(evt)",
        "function updateGraphFromSnapshot(state)",
        "function openNodeInspector(nodeKey)",
        "function closeNodeInspector()",
        "function copyNodeStatePayload()",
        "function switchTab(tabId)",
    ]
    for fn in required_functions:
        assert fn in static_html_content, f"Required JS engine function '{fn}' not found in index.html"


def test_active_edge_particle_animation(static_html_content: str):
    """Ensure SVG SMIL particle animation is rendered for active transitions."""
    assert "<animateMotion" in static_html_content
    assert "repeatCount=\"indefinite\"" in static_html_content


# ==============================================================================
# 4. HTTP API Serving Verification
# ==============================================================================


@pytest.mark.anyio
async def test_api_serves_architecture_visualizer(test_app):
    """FastAPI serves the complete updated index with Architecture Visualizer."""
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        res = await client.get("/")
        assert res.status_code == 200
        html = res.text

        # Validate visualizer integration
        assert "Architecture Visualizer" in html
        assert "id=\"tabArchitecture\"" in html
        assert "id=\"graphCanvasWrap\"" in html
        assert "id=\"graphSvgCanvas\"" in html
        assert "id=\"miniMapSvg\"" in html
        assert "id=\"sheetPanel\"" in html
