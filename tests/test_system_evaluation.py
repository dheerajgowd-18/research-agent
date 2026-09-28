"""Tests for Phase 15 System Evaluation Framework.

Covers:
- Dataset loading and validation (Pydantic models)
- Invalid case rejection and duplicate case detection
- Citation coverage and verification coverage calculations
- Per-class verifier metrics, macro-F1, and confusion matrix
- Aggregate system metrics calculation
- Empty results and missing data handling
- Serialization to JSON
- Markdown report generation structure
- Mock evaluation execution
- Latency, retry, supervisor, HITL, and follow-up stats aggregation
- Regression testing against known states
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from evaluation.system_eval_schemas import (
    SystemEvalCase,
    ExpectedProperties,
    CaseExecutionResult,
    SystemEvaluationSummary,
)
from evaluation.system_eval import (
    load_system_eval_dataset,
    calculate_case_metrics,
    aggregate_system_metrics,
    compute_verifier_performance,
    generate_system_evaluation_report,
    run_system_evaluation,
)
from verified_research.models.research import (
    Claim,
    Evidence,
    Source,
    VerificationResult,
    SupervisorDecision,
    HumanReview,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_case() -> SystemEvalCase:
    return SystemEvalCase(
        case_id="test_case_001",
        category="factual",
        question="What is the architecture of LangGraph?",
        description="Verify factual extraction of LangGraph architecture",
        expected_properties=ExpectedProperties(
            min_claims=1,
            min_sources=1,
            hitl_action="approve",
        ),
    )


@pytest.fixture
def sample_state():
    source = Source(
        source_id="s1",
        url="https://example.com/source1",
        title="Example Source 1",
        content="LangGraph uses Pregel-inspired cyclic execution graphs.",
    )
    evidence = Evidence(
        evidence_id="e1",
        source_id="s1",
        text="LangGraph uses Pregel-inspired cyclic execution graphs.",
    )
    claim1 = Claim(
        claim_id="c1",
        text="LangGraph is based on Pregel cyclic graph execution.",
        evidence_ids=["e1"],
    )
    claim2 = Claim(
        claim_id="c2",
        text="LangGraph requires Kubernetes for all workflows.",
        evidence_ids=["e1"],
    )
    decision = SupervisorDecision(
        next_worker="human_review",
        reasoning="Initial research and verification complete",
    )
    human_review = HumanReview(
        action="approve",
        feedback=None,
    )
    return {
        "question": "What is LangGraph?",
        "research_iteration": 1,
        "human_research_cycles": 0,
        "sources": [source],
        "evidence": [evidence],
        "claims": [claim1, claim2],
        "verification_results": [
            VerificationResult(
                claim_id="c1",
                verdict="SUPPORTED",
                confidence=0.95,
                reasoning="Direct match",
                evidence_ids=["e1"],
            ),
            VerificationResult(
                claim_id="c2",
                verdict="UNSUPPORTED",
                confidence=0.90,
                reasoning="Refuted",
                evidence_ids=["e1"],
            ),
        ],
        "supervisor_decisions": [decision],
        "supervisor_termination_reason": "completed_approved",
        "human_review": human_review,
        "edited_claims": [],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Dataset Loading & Schema Tests
# ---------------------------------------------------------------------------

def test_system_eval_dataset_loads_correctly():
    dataset_path = Path("data/system_eval.json")
    assert dataset_path.exists()
    cases = load_system_eval_dataset(str(dataset_path))
    assert len(cases) >= 15
    assert all(isinstance(c, SystemEvalCase) for c in cases)
    assert cases[0].case_id == "sys_eval_001"


def test_system_eval_dataset_schema_valid():
    case = SystemEvalCase(
        case_id="case_valid_01",
        category="factual",
        question="Is Python dynamic?",
        description="Verify dynamic nature of Python",
        expected_properties=ExpectedProperties(min_claims=1),
    )
    assert case.case_id == "case_valid_01"
    assert case.expected_properties.min_claims == 1
    assert case.expected_properties.hitl_action == "approve"


def test_invalid_case_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        SystemEvalCase(
            case_id="",  # empty id violates validation
            category="factual",
            question="Question?",
            description="desc",
            expected_properties=ExpectedProperties(),
        )


def test_duplicate_case_ids_rejected(tmp_path):
    dup_file = tmp_path / "dup.json"
    dup_file.write_text(
        json.dumps([
            {
                "case_id": "dup_01",
                "category": "factual",
                "question": "Q1",
                "description": "D1",
                "expected_properties": {},
            },
            {
                "case_id": "dup_01",
                "category": "factual",
                "question": "Q2",
                "description": "D2",
                "expected_properties": {},
            },
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate case ID 'dup_01' found"):
        load_system_eval_dataset(str(dup_file))


def test_invalid_json_format_rejected(tmp_path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text(json.dumps({"not": "a_list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset root must be a list of cases"):
        load_system_eval_dataset(str(invalid_file))


# ---------------------------------------------------------------------------
# Coverage & Metrics Calculation Tests
# ---------------------------------------------------------------------------

def test_citation_coverage_calculation(sample_case, sample_state):
    result = calculate_case_metrics(
        case=sample_case,
        final_state=sample_state,
        duration_seconds=1.23,
    )
    assert result.claims_count == 2
    assert result.claims_with_valid_evidence == 2
    assert result.citation_coverage == 1.0


def test_citation_coverage_with_missing_evidence(sample_case, sample_state):
    sample_state["claims"][1] = Claim(
        claim_id="c2",
        text="LangGraph requires Kubernetes for all workflows.",
        evidence_ids=["non_existent_e"],
    )
    result = calculate_case_metrics(
        case=sample_case,
        final_state=sample_state,
        duration_seconds=1.0,
    )
    assert result.claims_count == 2
    assert result.claims_with_valid_evidence == 1
    assert result.citation_coverage == 0.5


def test_citation_coverage_zero_claims(sample_case, sample_state):
    sample_state["claims"] = []
    result = calculate_case_metrics(
        case=sample_case,
        final_state=sample_state,
        duration_seconds=0.5,
    )
    assert result.claims_count == 0
    assert result.citation_coverage == 1.0


def test_verification_coverage_calculation(sample_case, sample_state):
    result = calculate_case_metrics(
        case=sample_case,
        final_state=sample_state,
        duration_seconds=1.0,
    )
    assert result.verification_coverage == 1.0
    assert result.verdicts_summary == {"SUPPORTED": 1, "PARTIAL": 0, "UNSUPPORTED": 1}


# ---------------------------------------------------------------------------
# Verifier Metrics (Precision, Recall, F1, Confusion Matrix) Tests
# ---------------------------------------------------------------------------

def test_compute_verifier_performance_perfect():
    gold = [
        {"claim": "A", "evidence": "E", "gold_verdict": "SUPPORTED"},
        {"claim": "B", "evidence": "E", "gold_verdict": "PARTIAL"},
        {"claim": "C", "evidence": "E", "gold_verdict": "UNSUPPORTED"},
    ]
    preds = [
        {"claim": "A", "predicted_verdict": "SUPPORTED"},
        {"claim": "B", "predicted_verdict": "PARTIAL"},
        {"claim": "C", "predicted_verdict": "UNSUPPORTED"},
    ]
    perf = compute_verifier_performance(gold, preds)
    assert perf["accuracy"] == 1.0
    assert perf["macro_f1"] == 1.0
    assert perf["per_class"]["SUPPORTED"]["f1"] == 1.0
    assert perf["per_class"]["PARTIAL"]["f1"] == 1.0
    assert perf["per_class"]["UNSUPPORTED"]["f1"] == 1.0
    assert perf["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_compute_verifier_performance_imperfect():
    gold = [
        {"claim": "A", "evidence": "E", "gold_verdict": "SUPPORTED"},
        {"claim": "B", "evidence": "E", "gold_verdict": "SUPPORTED"},
        {"claim": "C", "evidence": "E", "gold_verdict": "UNSUPPORTED"},
    ]
    preds = [
        {"claim": "A", "predicted_verdict": "SUPPORTED"},
        {"claim": "B", "predicted_verdict": "UNSUPPORTED"},
        {"claim": "C", "predicted_verdict": "UNSUPPORTED"},
    ]
    perf = compute_verifier_performance(gold, preds)
    assert perf["accuracy"] == pytest.approx(2 / 3, rel=1e-2)
    assert perf["per_class"]["SUPPORTED"]["precision"] == 1.0
    assert perf["per_class"]["SUPPORTED"]["recall"] == 0.5
    assert perf["per_class"]["UNSUPPORTED"]["precision"] == 0.5
    assert perf["per_class"]["UNSUPPORTED"]["recall"] == 1.0


def test_compute_verifier_performance_empty():
    perf = compute_verifier_performance([], [])
    assert perf["accuracy"] == 0.0
    assert perf["macro_f1"] == 0.0


# ---------------------------------------------------------------------------
# Aggregation & Edge Cases Tests
# ---------------------------------------------------------------------------

def test_aggregate_system_metrics_empty_rejected():
    with pytest.raises(ValueError, match="Cannot aggregate empty evaluation results."):
        aggregate_system_metrics([])


def test_aggregate_system_metrics_calculation():
    r1 = CaseExecutionResult(
        case_id="c1",
        question="Q1",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=3,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=2,
        evidence_count=2,
        sources_count=2,
        distinct_sources_count=2,
        claims_with_valid_evidence=2,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={"SUPPORTED": 2},
        supervisor_decisions=["research", "verifier", "human_review"],
        supervisor_termination_reason="completed_approved",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=1,
        llm_calls_estimate=4,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    r2 = CaseExecutionResult(
        case_id="c2",
        question="Q2",
        category="multi_turn_follow_up",
        status="completed",
        duration_seconds=3.0,
        supervisor_steps=5,
        research_iteration=2,
        human_research_cycles=1,
        claims_count=4,
        evidence_count=3,
        sources_count=3,
        distinct_sources_count=2,
        claims_with_valid_evidence=3,
        citation_coverage=0.75,
        verification_coverage=1.0,
        verdicts_summary={"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 1},
        supervisor_decisions=["research", "critic", "verifier", "human_review"],
        supervisor_termination_reason="completed_approved",
        hitl_action="edit",
        edited_claims_count=1,
        retries_count=0,
        llm_calls_estimate=6,
        search_calls_estimate=2,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    r3 = CaseExecutionResult(
        case_id="c3",
        question="Q3",
        category="human_rejection",
        status="rejected",
        duration_seconds=0.5,
        supervisor_steps=2,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={"UNSUPPORTED": 1},
        supervisor_decisions=["research", "human_review"],
        supervisor_termination_reason="human_rejected",
        hitl_action="reject",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=2,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )

    summary = aggregate_system_metrics([r1, r2, r3])

    assert summary.total_cases == 3
    assert summary.successful_runs == 2
    assert summary.rejected_runs == 1
    assert summary.failed_runs == 0
    assert summary.total_claims == 7
    assert summary.total_evidence == 6
    assert summary.total_sources == 6
    assert summary.mean_citation_coverage == pytest.approx((1.0 + 0.75 + 1.0) / 3, rel=1e-3)
    assert summary.mean_verification_coverage == 1.0
    assert summary.min_latency_seconds == 0.5
    assert summary.max_latency_seconds == 3.0
    assert summary.median_latency_seconds == 1.0
    assert summary.mean_supervisor_steps == pytest.approx((3 + 5 + 2) / 3, abs=0.01)
    assert summary.verdicts_breakdown == {"SUPPORTED": 4, "PARTIAL": 1, "UNSUPPORTED": 2}
    assert summary.hitl_actions_breakdown == {"approve": 1, "edit": 1, "reject": 1, "research_more": 0}
    assert summary.hitl_approval_rate == pytest.approx(1 / 3, rel=1e-3)
    assert summary.follow_up_stats["total_follow_ups"] == 1
    assert summary.follow_up_stats["mean_follow_up_steps"] == 5.0
    assert summary.reliability_stats.total_retries == 1


def test_duplicate_result_detected():
    r1 = CaseExecutionResult(
        case_id="dup_result_01",
        question="Q1",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=3,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=1,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    with pytest.raises(ValueError, match="Duplicate case_id in results: dup_result_01"):
        aggregate_system_metrics([r1, r1])


def test_serialization_of_summary_to_json(tmp_path):
    r = CaseExecutionResult(
        case_id="c_ser",
        question="Q",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=3,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={"SUPPORTED": 1},
        supervisor_decisions=["research"],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=1,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    summary = aggregate_system_metrics([r])
    out_file = tmp_path / "summary.json"
    out_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["total_cases"] == 1
    assert loaded["successful_runs"] == 1
    assert loaded["results"][0]["case_id"] == "c_ser"


def test_markdown_report_generation_structure():
    r = CaseExecutionResult(
        case_id="c_rep",
        question="Report Question?",
        category="factual",
        status="completed",
        duration_seconds=0.25,
        supervisor_steps=4,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=2,
        evidence_count=2,
        sources_count=2,
        distinct_sources_count=2,
        claims_with_valid_evidence=2,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={"SUPPORTED": 2},
        supervisor_decisions=["research", "verifier", "human_review"],
        supervisor_termination_reason="completed_approved",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=3,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    summary = aggregate_system_metrics([r])
    verifier_perf = {
        "accuracy": 0.95,
        "macro_f1": 0.92,
        "per_class": {
            "SUPPORTED": {"precision": 1.0, "recall": 0.9, "f1": 0.95, "support": 10},
            "PARTIAL": {"precision": 0.9, "recall": 0.9, "f1": 0.90, "support": 5},
            "UNSUPPORTED": {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 10},
        },
        "confusion_matrix": [[9, 1, 0], [0, 5, 0], [0, 0, 10]],
    }
    report = generate_system_evaluation_report(summary, verifier_performance=verifier_perf)

    # Validate mandatory report sections
    assert "# Final System Evaluation Report" in report
    assert "## 1. Executive Summary" in report
    assert "## 2. System Under Evaluation" in report
    assert "## 3. Evaluation Dataset" in report
    assert "## 4. Research & Evidence Metrics" in report
    assert "## 5. Claim-Level Verifier Performance" in report
    assert "## 6. End-to-End System Performance & Latency" in report
    assert "## 7. Reliability & Error Recovery" in report
    assert "## 8. Human-in-the-Loop (HITL) Results" in report
    assert "## 9. Follow-Up Research Continuity" in report
    assert "## 10. Failure Analysis & Boundary Cases" in report
    assert "## 11. Limitations" in report
    assert "## 12. Conclusions" in report
    assert "Macro-F1 Score" in report


def test_mock_evaluation_mode_completes_without_errors(tmp_path):
    cases = [
        SystemEvalCase(
            case_id="quick_mock_01",
            category="factual",
            question="What is the speed of light?",
            description="Quick test 1",
            expected_properties=ExpectedProperties(min_claims=1, hitl_action="approve"),
        ),
        SystemEvalCase(
            case_id="quick_mock_02",
            category="human_rejection",
            question="Is perpetual motion possible?",
            description="Quick test 2",
            expected_properties=ExpectedProperties(min_claims=1, hitl_action="reject"),
        ),
    ]
    out_json = tmp_path / "results.json"
    out_rep = tmp_path / "report.md"

    summary = run_system_evaluation(
        cases=cases,
        use_mock=True,
        output_path=str(out_json),
        report_path=str(out_rep),
    )

    assert summary.total_cases == 2
    assert summary.successful_runs == 1
    assert summary.rejected_runs == 1
    assert summary.failed_runs == 0
    assert out_json.exists()
    assert out_rep.exists()


def test_regression_evaluation_on_known_state(sample_case, sample_state):
    result = calculate_case_metrics(
        case=sample_case,
        final_state=sample_state,
        duration_seconds=0.042,
    )
    assert result.case_id == "test_case_001"
    assert result.status == "completed"
    assert result.claims_count == 2
    assert result.evidence_count == 1
    assert result.sources_count == 1
    assert result.claims_with_valid_evidence == 2
    assert result.citation_coverage == 1.0
    assert result.verification_coverage == 1.0
    assert result.verdicts_summary == {"SUPPORTED": 1, "PARTIAL": 0, "UNSUPPORTED": 1}
    assert result.hitl_action == "approve"
    assert result.duration_seconds == 0.042


def test_latency_aggregation():
    latencies = [1.0, 2.0, 3.0, 4.0, 10.0]
    results = [
        CaseExecutionResult(
            case_id=f"c_{i}",
            question=f"Q{i}",
            category="factual",
            status="completed",
            duration_seconds=lat,
            supervisor_steps=3,
            research_iteration=1,
            human_research_cycles=0,
            claims_count=1,
            evidence_count=1,
            sources_count=1,
            distinct_sources_count=1,
            claims_with_valid_evidence=1,
            citation_coverage=1.0,
            verification_coverage=1.0,
            verdicts_summary={},
            supervisor_decisions=[],
            supervisor_termination_reason="completed",
            hitl_action="approve",
            edited_claims_count=0,
            retries_count=0,
            llm_calls_estimate=1,
            search_calls_estimate=1,
            error=None,
            timestamp="2026-09-28T00:00:00Z",
        )
        for i, lat in enumerate(latencies)
    ]
    summary = aggregate_system_metrics(results)
    assert summary.min_latency_seconds == 1.0
    assert summary.max_latency_seconds == 10.0
    assert summary.mean_latency_seconds == 4.0
    assert summary.median_latency_seconds == 3.0


def test_retry_stats_aggregation():
    r1 = CaseExecutionResult(
        case_id="r_01",
        question="Q1",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=3,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=2,
        llm_calls_estimate=1,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    r2 = CaseExecutionResult(
        case_id="r_02",
        question="Q2",
        category="factual",
        status="failed",
        duration_seconds=0.5,
        supervisor_steps=1,
        research_iteration=0,
        human_research_cycles=0,
        claims_count=0,
        evidence_count=0,
        sources_count=0,
        distinct_sources_count=0,
        claims_with_valid_evidence=0,
        citation_coverage=0.0,
        verification_coverage=0.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="error",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=3,
        llm_calls_estimate=0,
        search_calls_estimate=0,
        error="Search timeout",
        timestamp="2026-09-28T00:00:00Z",
    )
    summary = aggregate_system_metrics([r1, r2])
    assert summary.reliability_stats.total_runs == 2
    assert summary.reliability_stats.successful_runs == 1
    assert summary.reliability_stats.permanent_failures == 1
    assert summary.reliability_stats.total_retries == 5


def test_supervisor_steps_and_iteration_aggregation():
    r1 = CaseExecutionResult(
        case_id="s_01",
        question="Q1",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=4,
        research_iteration=2,
        human_research_cycles=1,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=5,
        search_calls_estimate=2,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    r2 = CaseExecutionResult(
        case_id="s_02",
        question="Q2",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=6,
        research_iteration=3,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=7,
        search_calls_estimate=3,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    summary = aggregate_system_metrics([r1, r2])
    assert summary.mean_supervisor_steps == 5.0
    assert summary.mean_research_iterations == 2.5


def test_hitl_action_breakdown_aggregation():
    results = [
        CaseExecutionResult(
            case_id=f"h_{i}",
            question=f"Q{i}",
            category="factual",
            status="completed" if action != "reject" else "rejected",
            duration_seconds=1.0,
            supervisor_steps=3,
            research_iteration=1,
            human_research_cycles=0,
            claims_count=1,
            evidence_count=1,
            sources_count=1,
            distinct_sources_count=1,
            claims_with_valid_evidence=1,
            citation_coverage=1.0,
            verification_coverage=1.0,
            verdicts_summary={},
            supervisor_decisions=[],
            supervisor_termination_reason="completed",
            hitl_action=action,
            edited_claims_count=1 if action == "edit" else 0,
            retries_count=0,
            llm_calls_estimate=1,
            search_calls_estimate=1,
            error=None,
            timestamp="2026-09-28T00:00:00Z",
        )
        for i, action in enumerate(["approve", "approve", "edit", "reject"])
    ]
    summary = aggregate_system_metrics(results)
    assert summary.hitl_actions_breakdown["approve"] == 2
    assert summary.hitl_actions_breakdown["edit"] == 1
    assert summary.hitl_actions_breakdown["reject"] == 1
    assert summary.hitl_actions_breakdown["research_more"] == 0
    assert summary.hitl_approval_rate == 0.5


def test_follow_up_stats_aggregation():
    r1 = CaseExecutionResult(
        case_id="fu_01",
        question="Initial question",
        category="factual",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=3,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=1,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    r2 = CaseExecutionResult(
        case_id="fu_02",
        question="Follow up question",
        category="multi_turn_follow_up",
        status="completed",
        duration_seconds=1.0,
        supervisor_steps=5,
        research_iteration=1,
        human_research_cycles=0,
        claims_count=1,
        evidence_count=1,
        sources_count=1,
        distinct_sources_count=1,
        claims_with_valid_evidence=1,
        citation_coverage=1.0,
        verification_coverage=1.0,
        verdicts_summary={},
        supervisor_decisions=[],
        supervisor_termination_reason="completed",
        hitl_action="approve",
        edited_claims_count=0,
        retries_count=0,
        llm_calls_estimate=1,
        search_calls_estimate=1,
        error=None,
        timestamp="2026-09-28T00:00:00Z",
    )
    summary = aggregate_system_metrics([r1, r2])
    assert summary.follow_up_stats["total_follow_ups"] == 1
    assert summary.follow_up_stats["mean_follow_up_steps"] == 5.0

