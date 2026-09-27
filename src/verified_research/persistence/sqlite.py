"""SQLite checkpoint persistence configuration for LangGraph state."""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Iterator
from langgraph.checkpoint.sqlite import SqliteSaver
from verified_research.config.settings import get_settings

logger = logging.getLogger(__name__)

# Registered domain types allowed during checkpoint deserialization
CHECKPOINT_ALLOWED_TYPES: tuple[tuple[str, str], ...] = (
    ("verified_research.models.research", "Source"),
    ("verified_research.models.research", "Finding"),
    ("verified_research.models.research", "Evidence"),
    ("verified_research.models.research", "Claim"),
    ("verified_research.models.research", "AnalystOutput"),
    ("verified_research.models.research", "Critique"),
    ("verified_research.models.research", "VerificationResult"),
    ("verified_research.models.research", "HumanReview"),
)


def create_sqlite_checkpointer(
    conn: sqlite3.Connection,
    setup: bool = True,
) -> SqliteSaver:
    """Create a configured SqliteSaver instance with domain type deserialization allowlist.

    Args:
        conn: Open SQLite connection.
        setup: If True, executes table creation DDL if tables do not already exist.

    Returns:
        Configured SqliteSaver checkpointer instance.
    """
    saver = SqliteSaver(conn).with_allowlist(CHECKPOINT_ALLOWED_TYPES)
    if setup:
        saver.setup()
    return saver


@contextmanager
def get_sqlite_checkpointer(
    db_path: str | Path | None = None,
    setup: bool = True,
) -> Iterator[SqliteSaver]:
    """Context manager creating and managing the lifecycle of an SQLite checkpointer.

    Ensures connection setup, safe type deserialization, and clean closing on context exit.

    Args:
        db_path: Path to the SQLite database file, or ':memory:'. Defaults to configured setting.
        setup: If True, executes schema setup on first connection.

    Yields:
        Configured SqliteSaver ready for graph compilation.
    """
    target_path = (
        str(db_path)
        if db_path is not None
        else get_settings().checkpoint_db_path
    )

    # Ensure parent directory exists for file-based paths
    if target_path != ":memory:":
        path_obj = Path(target_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[Persistence] Opening SQLite checkpointer database at '%s'", target_path)
    conn = sqlite3.connect(target_path, check_same_thread=False)
    try:
        saver = create_sqlite_checkpointer(conn, setup=setup)
        yield saver
    finally:
        logger.info("[Persistence] Closing SQLite connection for '%s'", target_path)
        conn.close()
