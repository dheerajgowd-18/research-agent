"""Unit and integration tests for SQLite checkpoint persistence in LangGraph."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from verified_research.agents.human_review import create_human_review_node
from verified_research.graph.graph import create_research_graph
from verified_research.models.research import (
    Claim,
    Critique,
    Evidence,
    Finding,
    HumanReview,
    Source,
    VerificationResult,
)
from verified_research.persistence import (
    CHECKPOINT_ALLOWED_TYPES,
    create_sqlite_checkpointer,
    get_sqlite_checkpointer,
)


# ==============================================================================
# Helper Mock Pipeline
# ==============================================================================


def create_mock_pipeline_components():
    """Build deterministic mock nodes for testing persistence without LLMs or network."""
    sample_sources = [
        Source(
            source_id="src_001",
            title="Quantum Memory Benchmark",
            url="https://example.com/quantum",
            content="Coherence time reached 1.5 seconds at 15mK.",
        )
    ]
    sample_evidence = [
        Evidence(
            evidence_id="ev_001",
            source_id="src_001",
            text="Coherence time reached 1.5 seconds at 15mK.",
        )
    ]
    sample_claims = [
        Claim(
            claim_id="claim_001",
            text="Quantum memory coherence time reaches 1.5 seconds.",
            evidence_ids=["ev_001"],
        )
    ]
    sample_verifications = [
        VerificationResult(
            claim_id="claim_001",
            verdict="SUPPORTED",
            confidence=0.99,
            reasoning="Exact numerical match with excerpt.",
            evidence_ids=["ev_001"],
        )
    ]

    invocations = {"research": 0, "verifier": 0}

    def mock_subgraph(state):
        invocations["research"] += 1
        iteration = state.get("research_iteration", 0) + 1
        return {
            "question": state.get("question"),
            "sources": sample_sources,
            "findings": [
                Finding(
                    finding_id="finding_001",
                    text="Coherence time reached 1.5 seconds at 15mK.",
                    source_ids=["src_001"],
                )
            ],
            "critique": Critique(
                quality_score=0.95,
                should_research_again=False,
            ),
            "research_iteration": iteration,
            "evidence": sample_evidence,
            "claims": sample_claims,
        }

    def mock_verifier(state):
        invocations["verifier"] += 1
        return {"verification_results": sample_verifications}

    return {
        "subgraph": mock_subgraph,
        "verifier": mock_verifier,
        "invocations": invocations,
        "sample_sources": sample_sources,
        "sample_evidence": sample_evidence,
        "sample_claims": sample_claims,
        "sample_verifications": sample_verifications,
    }


# ==============================================================================
# 1. Checkpointer Integration Tests
# ==============================================================================


class TestSqliteCheckpointerIntegration:
    """Tests verifying SQLite checkpointer compilation and table initialization."""

    def test_sqlite_checkpointer_schema_setup(self, tmp_path):
        db_file = tmp_path / "test_schema.db"
        with get_sqlite_checkpointer(db_file) as checkpointer:
            assert isinstance(checkpointer, SqliteSaver)

        # Reopen connection directly to verify SQLite tables were created
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "checkpoints" in tables
        assert "writes" in tables

    def test_graph_compiles_and_runs_with_sqlite_checkpointer(self, tmp_path):
        db_file = tmp_path / "test_graph_run.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )

            config = {"configurable": {"thread_id": "thread-compile-test"}}
            res = graph.invoke({"question": "Quantum coherence benchmark"}, config)

            # Execution pauses at human review
            assert "__interrupt__" in res
            state = graph.get_state(config)
            assert state.next == ("human_review",)
            assert pipeline["invocations"]["research"] == 1
            assert pipeline["invocations"]["verifier"] == 1


# ==============================================================================
# 2. Thread Isolation Tests
# ==============================================================================


class TestThreadIsolation:
    """Tests proving that distinct thread IDs maintain completely isolated checkpoint state."""

    def test_independent_thread_execution_and_state(self, tmp_path):
        db_file = tmp_path / "isolation.db"

        with get_sqlite_checkpointer(db_file) as checkpointer:
            pipeline = create_mock_pipeline_components()
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )

            config_a = {"configurable": {"thread_id": "thread-001"}}
            config_b = {"configurable": {"thread_id": "thread-002"}}

            # Run thread-001
            graph.invoke({"question": "Question for Thread A"}, config_a)
            # Run thread-002
            graph.invoke({"question": "Question for Thread B"}, config_b)

            # Inspect state of thread A
            state_a = graph.get_state(config_a)
            assert state_a.values["question"] == "Question for Thread A"
            assert state_a.next == ("human_review",)

            # Inspect state of thread B
            state_b = graph.get_state(config_b)
            assert state_b.values["question"] == "Question for Thread B"
            assert state_b.next == ("human_review",)

            # Resume thread A with approve
            final_a = graph.invoke(Command(resume={"action": "approve"}), config_a)
            assert final_a["human_review"].action == "approve"
            assert graph.get_state(config_a).next == ()

            # Thread B must remain interrupted and unmutated
            state_b_after = graph.get_state(config_b)
            assert state_b_after.next == ("human_review",)
            assert "human_review" not in state_b_after.values

            # Resume thread B with reject
            final_b = graph.invoke(Command(resume={"action": "reject"}), config_b)
            assert final_b["human_review"].action == "reject"
            assert graph.get_state(config_b).next == ()


# ==============================================================================
# 3. Interrupt Persistence & State Integrity
# ==============================================================================


class TestInterruptPersistenceAndStateIntegrity:
    """Tests confirming all state models survive checkpoint serialization and deserialization."""

    def test_checkpoint_stores_and_restores_complete_domain_state(self, tmp_path):
        db_file = tmp_path / "domain_state.db"
        pipeline = create_mock_pipeline_components()

        # Step 1: Execute graph until interrupt
        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-domain-test"}}
            graph.invoke({"question": "State integrity test question"}, config)

        # Step 2: Open a fresh checkpointer connection to the same SQLite database
        with get_sqlite_checkpointer(db_file) as checkpointer2:
            graph2 = create_research_graph(checkpointer=checkpointer2)
            recovered_state = graph2.get_state(config)

            # Verify next node
            assert recovered_state.next == ("human_review",)

            # Verify all domain models survived deserialization with exact types and fields
            values = recovered_state.values
            assert values["question"] == "State integrity test question"

            claims = values["claims"]
            assert len(claims) == 1
            assert isinstance(claims[0], Claim)
            assert claims[0].claim_id == "claim_001"
            assert claims[0].text == "Quantum memory coherence time reaches 1.5 seconds."
            assert claims[0].evidence_ids == ["ev_001"]

            evidence = values["evidence"]
            assert len(evidence) == 1
            assert isinstance(evidence[0], Evidence)
            assert evidence[0].evidence_id == "ev_001"
            assert evidence[0].source_id == "src_001"

            results = values["verification_results"]
            assert len(results) == 1
            assert isinstance(results[0], VerificationResult)
            assert results[0].verdict == "SUPPORTED"
            assert results[0].confidence == 0.99


# ==============================================================================
# 4. Critical Process Restart Boundary Recovery
# ==============================================================================


class TestProcessRestartRecovery:
    """Tests demonstrating checkpoint persistence across genuine process boundaries."""

    def test_restart_recovery_across_subprocesses(self, tmp_path):
        """CRITICAL RESTART TEST:
        Stage 1: Process A runs until interrupt on thread 'phase8-restart-test'.
        Stage 2: Process A exits completely.
        Stage 3: Process B launches, recovers state via same thread_id and SQLite DB.
        Stage 4: Process B resumes without restarting research.
        """
        db_file = str(tmp_path / "restart_experiment.db")
        repo_root = str(Path(__file__).resolve().parent.parent)
        src_path = str(Path(repo_root) / "src")

        # Stage 1 Subprocess Script
        stage1_code = f"""
import sys
sys.path.insert(0, r"{src_path}")
from verified_research.models.research import Claim, Evidence, VerificationResult
from verified_research.persistence import get_sqlite_checkpointer
from verified_research.graph.graph import create_research_graph

with get_sqlite_checkpointer(r"{db_file}") as checkpointer:
    def mock_subgraph(state):
        return {{
            "claims": [Claim(claim_id="claim_001", text="Coherence is 1.5s", evidence_ids=["ev_001"])],
            "evidence": [Evidence(evidence_id="ev_001", source_id="src_001", text="Coherence is 1.5s")],
        }}
    def mock_verifier(state):
        return {{
            "verification_results": [
                VerificationResult(claim_id="claim_001", verdict="SUPPORTED", confidence=0.98, reasoning="Exact match", evidence_ids=["ev_001"])
            ]
        }}
    graph = create_research_graph(
        custom_subgraph=mock_subgraph,
        custom_verifier=mock_verifier,
        checkpointer=checkpointer,
    )
    config = {{"configurable": {{"thread_id": "phase8-restart-test"}}}}
    res = graph.invoke({{"question": "What is coherence time?"}}, config)
    assert "__interrupt__" in res
    print("STAGE 1 COMPLETED: Execution paused at human review and persisted to SQLite.")
"""

        p1 = subprocess.run(
            [sys.executable, "-c", stage1_code],
            capture_output=True,
            text=True,
        )
        assert p1.returncode == 0, f"Process 1 failed with error:\n{p1.stderr}"
        assert "STAGE 1 COMPLETED" in p1.stdout

        # Verify DB file exists and has non-zero size
        assert os.path.exists(db_file)
        assert os.path.getsize(db_file) > 0

        # Stage 2 Subprocess Script (Resuming in a fresh process)
        stage2_code = f"""
import sys
sys.path.insert(0, r"{src_path}")
from langgraph.types import Command
from verified_research.models.research import Claim
from verified_research.persistence import get_sqlite_checkpointer
from verified_research.graph.graph import create_research_graph

with get_sqlite_checkpointer(r"{db_file}") as checkpointer:
    # Ensure research and verifier nodes raise if called during resume
    def fail_subgraph(state):
        raise RuntimeError("Research subgraph should NOT be called on resume!")
    def fail_verifier(state):
        raise RuntimeError("Verifier should NOT be called on resume!")

    graph = create_research_graph(
        custom_subgraph=fail_subgraph,
        custom_verifier=fail_verifier,
        checkpointer=checkpointer,
    )
    config = {{"configurable": {{"thread_id": "phase8-restart-test"}}}}

    # Verify recovered state
    state = graph.get_state(config)
    assert state.next == ("human_review",), f"Expected next='human_review', got {{state.next}}"
    assert len(state.values["claims"]) == 1
    assert state.values["claims"][0].claim_id == "claim_001"

    # Resume with approve action
    res = graph.invoke(Command(resume={{"action": "approve"}}), config)
    assert res["human_review"].action == "approve"
    assert graph.get_state(config).next == ()
    print("STAGE 2 COMPLETED: Resumed across process boundary and reached END.")
"""

        p2 = subprocess.run(
            [sys.executable, "-c", stage2_code],
            capture_output=True,
            text=True,
        )
        assert p2.returncode == 0, f"Process 2 failed with error:\n{p2.stderr}"
        assert "STAGE 2 COMPLETED" in p2.stdout


# ==============================================================================
# 5. Resume Continuation & Human Actions with SQLite Checkpointer
# ==============================================================================


class TestResumeActionsWithSqliteCheckpointer:
    """Tests confirming all Phase 7 actions function identically with persistent SQLite storage."""

    def test_approve_resume_finishes_graph(self, tmp_path):
        db_file = tmp_path / "approve.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-approve"}}
            graph.invoke({"question": "Test approve"}, config)

            final_state = graph.invoke(Command(resume={"action": "approve"}), config)
            assert final_state["human_review"].action == "approve"
            assert graph.get_state(config).next == ()

    def test_reject_resume_records_rejection_state(self, tmp_path):
        db_file = tmp_path / "reject.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-reject"}}
            graph.invoke({"question": "Test reject"}, config)

            final_state = graph.invoke(
                Command(resume={"action": "reject", "feedback": "Evidence insufficient."}),
                config,
            )
            assert final_state["human_review"].action == "reject"
            assert final_state["human_review"].feedback == "Evidence insufficient."
            assert graph.get_state(config).next == ()

    def test_edit_resume_stores_edited_claims(self, tmp_path):
        db_file = tmp_path / "edit.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-edit"}}
            graph.invoke({"question": "Test edit"}, config)

            edited_data = [
                {
                    "claim_id": "claim_001",
                    "text": "Corrected assertion text.",
                    "evidence_ids": ["ev_001"],
                }
            ]
            final_state = graph.invoke(
                Command(resume={"action": "edit", "edited_claims": edited_data}),
                config,
            )
            assert final_state["human_review"].action == "edit"
            assert len(final_state["claims"]) == 1
            assert final_state["claims"][0].text == "Corrected assertion text."
            assert graph.get_state(config).next == ()

    def test_research_more_resumes_into_research_subgraph(self, tmp_path):
        db_file = tmp_path / "research_more.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-research-more"}}
            graph.invoke({"question": "Initial question"}, config)
            assert pipeline["invocations"]["research"] == 1

            # Resume with research_more
            graph.invoke(
                Command(
                    resume={
                        "action": "research_more",
                        "feedback": "Look into superconducting circuits.",
                    }
                ),
                config,
            )

            # Subgraph was re-entered and graph paused again at human review
            assert pipeline["invocations"]["research"] == 2
            state = graph.get_state(config)
            assert state.next == ("human_review",)
            assert state.values.get("human_research_cycles") == 1
            assert state.values.get("human_feedback") == "Look into superconducting circuits."

    def test_human_research_cycle_limit_enforced_with_persistence(self, tmp_path):
        db_file = tmp_path / "cycle_limit.db"
        pipeline = create_mock_pipeline_components()

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(
                custom_subgraph=pipeline["subgraph"],
                custom_verifier=pipeline["verifier"],
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "thread-cycle-limit"}}

            # Pass 0
            graph.invoke({"question": "Initial question"}, config)
            # Cycle 1
            graph.invoke(Command(resume={"action": "research_more"}), config)
            # Cycle 2 (MAX = 2 reached)
            graph.invoke(Command(resume={"action": "research_more"}), config)
            # Cycle 3 requested -> router routes to END
            final_state = graph.invoke(Command(resume={"action": "research_more"}), config)

            assert graph.get_state(config).next == ()
            assert pipeline["invocations"]["research"] == 3  # 1 initial + 2 human loops
            assert final_state["human_research_cycles"] == 2


# ==============================================================================
# 6. Invalid Thread Behavior Tests
# ==============================================================================


class TestInvalidThreadBehavior:
    """Tests confirming error handling when querying or resuming invalid thread IDs."""

    def test_querying_uninitialized_thread_returns_empty_state(self, tmp_path):
        db_file = tmp_path / "invalid_thread.db"

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(checkpointer=checkpointer)
            bad_config = {"configurable": {"thread_id": "nonexistent-thread-404"}}

            state = graph.get_state(bad_config)
            assert state.values == {}
            assert state.next == ()

    def test_resuming_uninitialized_thread_fails_cleanly(self, tmp_path):
        db_file = tmp_path / "invalid_resume.db"

        with get_sqlite_checkpointer(db_file) as checkpointer:
            graph = create_research_graph(checkpointer=checkpointer)
            bad_config = {"configurable": {"thread_id": "nonexistent-thread-404"}}

            # Attempting to resume a non-interrupted thread must raise ValueError
            with pytest.raises(ValueError):
                graph.invoke(Command(resume={"action": "approve"}), bad_config)
