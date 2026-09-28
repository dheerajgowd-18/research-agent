"""Production Test Suite for Stateful Checkpoint Recovery Across Real Process Boundaries.

Verifies:
1. Process A executes supervisor graph up to Human Review interrupt and persists state to SQLite checkpointer.
2. Process A exits completely.
3. Process B launches in a fresh OS subprocess, recovers exact state from SQLite.
4. Process B resumes execution without re-executing completed research/verification nodes.
5. Process B finishes cleanly at END with terminal COMPLETED status.
"""

import os
from pathlib import Path
import subprocess
import sys
import pytest


class TestCheckpointProcessBoundary:
    """Verifies state serialization, storage, and cross-process restoration with SQLite."""

    def test_supervisor_checkpoint_recovery_across_subprocess_boundary(self, tmp_path: Path):
        """Execute Process A to interrupt -> Exit -> Execute Process B to resume -> Complete."""
        db_file = str(tmp_path / "supervisor_process_recovery.db")
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        src_path = str(Path(repo_root) / "src")

        # Process A Script: Runs initial research until interrupt
        process_a_code = f"""
import sys
sys.path.insert(0, r"{src_path}")
from langgraph.types import Command
from verified_research.models.research import Claim, Evidence, Finding, VerificationResult, Source
from verified_research.persistence import get_sqlite_checkpointer
from verified_research.graph.graph import create_supervisor_graph

with get_sqlite_checkpointer(r"{db_file}") as checkpointer:
    def mock_subgraph(state):
        return {{
            "sources": [
                Source(source_id="src_001", title="Quantum Paper", url="https://example.com/p", content="Coherence 2ms")
            ],
            "findings": [
                Finding(finding_id="f_001", text="Coherence is 2ms", source_ids=["src_001"])
            ],
            "evidence": [
                Evidence(evidence_id="ev_001", source_id="src_001", text="Coherence 2ms")
            ],
            "claims": [
                Claim(claim_id="claim_001", text="Coherence is 2ms", evidence_ids=["ev_001"])
            ],
        }}

    def mock_verifier(state):
        return {{
            "verification_results": [
                VerificationResult(
                    claim_id="claim_001",
                    verdict="SUPPORTED",
                    confidence=0.99,
                    reasoning="Direct match.",
                    evidence_ids=["ev_001"],
                )
            ]
        }}

    graph = create_supervisor_graph(
        custom_subgraph=mock_subgraph,
        custom_verifier=mock_verifier,
        checkpointer=checkpointer,
    )
    config = {{"configurable": {{"thread_id": "prod-process-boundary-test"}}}}
    res = graph.invoke({{"question": "What is superconducting coherence time?"}}, config)
    assert "__interrupt__" in res, "Expected execution to pause at human_review interrupt"
    print("PROCESS_A_SUCCESS: Execution paused at human_review interrupt.")
"""

        p1 = subprocess.run(
            [sys.executable, "-c", process_a_code],
            capture_output=True,
            text=True,
        )
        assert p1.returncode == 0, f"Process A failed:\nSTDOUT: {p1.stdout}\nSTDERR: {p1.stderr}"
        assert "PROCESS_A_SUCCESS" in p1.stdout

        # Verify DB file exists and is populated
        assert os.path.exists(db_file)
        assert os.path.getsize(db_file) > 0

        # Process B Script: Launches in a completely new Python process and resumes
        process_b_code = f"""
import sys
sys.path.insert(0, r"{src_path}")
from langgraph.types import Command
from verified_research.persistence import get_sqlite_checkpointer
from verified_research.graph.graph import create_supervisor_graph

with get_sqlite_checkpointer(r"{db_file}") as checkpointer:
    # Fail-safes: If research or verifier are re-executed on resume, raise error
    def fail_subgraph(state):
        raise RuntimeError("Subprocess B error: Research subgraph re-executed unexpectedly!")

    def fail_verifier(state):
        raise RuntimeError("Subprocess B error: Verifier re-executed unexpectedly!")

    graph = create_supervisor_graph(
        custom_subgraph=fail_subgraph,
        custom_verifier=fail_verifier,
        checkpointer=checkpointer,
    )
    config = {{"configurable": {{"thread_id": "prod-process-boundary-test"}}}}

    # Verify recovered thread state
    recovered_state = graph.get_state(config)
    assert recovered_state.next == ("human_review",), f"Expected next='human_review', got {{recovered_state.next}}"
    assert len(recovered_state.values["claims"]) == 1
    assert recovered_state.values["claims"][0].claim_id == "claim_001"
    assert len(recovered_state.values["verification_results"]) == 1
    assert recovered_state.values["verification_results"][0].verdict == "SUPPORTED"

    # Resume with approval
    final = graph.invoke(Command(resume={{"action": "approve", "feedback": "Verified in Process B."}}), config)
    assert final["human_review"].action == "approve"
    assert final["supervisor_decision"].next_worker == "finish"
    assert final["supervisor_termination_reason"] == "COMPLETED"
    assert graph.get_state(config).next == ()
    print("PROCESS_B_SUCCESS: Checkpoint recovered across process boundary and completed.")
"""

        p2 = subprocess.run(
            [sys.executable, "-c", process_b_code],
            capture_output=True,
            text=True,
        )
        assert p2.returncode == 0, f"Process B failed:\nSTDOUT: {p2.stdout}\nSTDERR: {p2.stderr}"
        assert "PROCESS_B_SUCCESS" in p2.stdout
