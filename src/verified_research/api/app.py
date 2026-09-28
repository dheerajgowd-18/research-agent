"""FastAPI application factory for the Verified Research Agent presentation and streaming layer."""

from pathlib import Path
from typing import Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph.state import CompiledStateGraph

from verified_research.api.routes import router as research_router
from verified_research.api.service import ResearchService


def create_app(
    service: ResearchService | None = None,
    graph: CompiledStateGraph | None = None,
    checkpointer: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        service: Optional injected ResearchService instance (for tests).
        graph: Optional compiled LangGraph supervisor graph.
        checkpointer: Optional persistence checkpointer instance.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Verified Research Agent API",
        description="Streaming API and presentation layer for ground-truth research and verification.",
        version="0.1.0",
    )

    # Enable CORS for frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store research service on app state for dependency injection
    app.state.research_service = service or ResearchService(
        graph=graph,
        checkpointer=checkpointer,
    )

    # Register API endpoints
    app.include_router(research_router)

    # Health check endpoint
    @app.get("/api/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": "verified-research-agent", "version": "0.1.0"}

    # Serve static frontend UI
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/ui", include_in_schema=False)
        async def serve_ui():
            index_path = static_dir / "index.html"
            return FileResponse(str(index_path))

    return app


# Default application instance for uvicorn
app = create_app()
