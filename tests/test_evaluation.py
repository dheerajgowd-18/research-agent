"""Unit and integration tests for the verifier evaluation framework."""

from pathlib import Path
import pytest
from pydantic import ValidationError
from evaluation.evaluate_verifier import (
    generate_evaluation_report,
    load_dataset,
    run_evaluation,
    save_predictions,
)
from evaluation.metrics import calculate_evaluation_metrics, format_confusion_matrix_ascii
from evaluation.schemas import CLASS_LABELS, EvalExample, PredictionRecord
from verified_research.models.research import Claim, Evidence, VerificationResult


class MockVerifierService:
    """Mock verifier returning canned responses based on claim_id or a fixed verdict."""

    def __init__(self, responses: dict[str, str] | None = None, default_verdict: str = "SUPPORTED"):
        self.responses = responses or {}
        self.default_verdict = default_verdict
        self.call_count = 0

    def verify_claim(self, claim: Claim, evidence_items: list[Evidence]) -> VerificationResult:
        self.call_count += 1
        verdict = self.responses.get(claim.claim_id, self.default_verdict)
        return VerificationResult(
            claim_id=claim.claim_id,
            verdict=verdict,  # type: ignore[arg-type]
            confidence=0.95,
            reasoning=f"Mock evaluated {claim.claim_id} as {verdict}",
            evidence_ids=[ev.evidence_id for ev in evidence_items],
        )


class TestEvaluationSystem:
    """Tests covering Step 15 requirements TEST 1 through TEST 10."""

    def test_1_correct_prediction_produces_correct_true(self):
        """TEST 1: Correct prediction produces correct=True."""
        example = EvalExample(
            id="eval_001",
            category="direct_support",
            claim="Sky is blue.",
            evidence="The sky is blue.",
            expected_verdict="SUPPORTED",
        )
        mock_verifier = MockVerifierService(default_verdict="SUPPORTED")

        records, metrics = run_evaluation([example], verifier_service=mock_verifier)

        assert len(records) == 1
        assert records[0].expected == "SUPPORTED"
        assert records[0].predicted == "SUPPORTED"
        assert records[0].correct is True
        assert metrics.accuracy == 1.0

    def test_2_incorrect_prediction_produces_correct_false(self):
        """TEST 2: Incorrect prediction produces correct=False."""
        example = EvalExample(
            id="eval_002",
            category="contradiction",
            claim="Trial succeeded.",
            evidence="Trial failed.",
            expected_verdict="UNSUPPORTED",
        )
        mock_verifier = MockVerifierService(default_verdict="SUPPORTED")

        records, metrics = run_evaluation([example], verifier_service=mock_verifier)

        assert len(records) == 1
        assert records[0].expected == "UNSUPPORTED"
        assert records[0].predicted == "SUPPORTED"
        assert records[0].correct is False
        assert metrics.accuracy == 0.0

    def test_3_metrics_use_correct_three_class_label_set(self):
        """TEST 3: Metrics use the canonical three-class label set in fixed order."""
        expected = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
        predicted = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]

        metrics = calculate_evaluation_metrics(expected, predicted)

        assert metrics.class_labels == ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
        assert list(metrics.per_class.keys()) == ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
        assert len(metrics.confusion_matrix) == 3
        assert len(metrics.confusion_matrix[0]) == 3

    def test_4_confusion_matrix_orientation_rows_expected_cols_predicted(self):
        """TEST 4: Confusion matrix uses: rows = expected, columns = predicted."""
        # 1 expected SUPPORTED predicted as UNSUPPORTED
        # Row 0 (SUPPORTED), Col 2 (UNSUPPORTED) must be 1
        expected = ["SUPPORTED"]
        predicted = ["UNSUPPORTED"]

        metrics = calculate_evaluation_metrics(expected, predicted)
        cm = metrics.confusion_matrix

        # Indices: 0: SUPPORTED, 1: PARTIAL, 2: UNSUPPORTED
        # Row 0 (Expected SUPPORTED), Column 2 (Predicted UNSUPPORTED)
        assert cm[0][2] == 1
        assert cm[0][0] == 0  # Row 0, Col 0 is 0
        assert cm[2][0] == 0  # Row 2 (Expected UNSUPPORTED), Col 0 (Predicted SUPPORTED) is 0

        # Another case: 1 expected PARTIAL predicted as SUPPORTED
        # Row 1 (PARTIAL), Col 0 (SUPPORTED)
        expected2 = ["PARTIAL"]
        predicted2 = ["SUPPORTED"]
        metrics2 = calculate_evaluation_metrics(expected2, predicted2)
        assert metrics2.confusion_matrix[1][0] == 1
        assert metrics2.confusion_matrix[0][1] == 0

    def test_5_macro_f1_calculated_correctly(self):
        """TEST 5: Macro-F1 is calculated correctly as unweighted mean of per-class F1."""
        # Setup:
        # SUPPORTED: 1 expected, 1 predicted correctly -> F1 = 1.0
        # PARTIAL: 1 expected, 1 predicted correctly -> F1 = 1.0
        # UNSUPPORTED: 1 expected, predicted as PARTIAL -> F1 = 0.0
        expected = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
        predicted = ["SUPPORTED", "PARTIAL", "PARTIAL"]

        metrics = calculate_evaluation_metrics(expected, predicted)

        f1_s = metrics.per_class["SUPPORTED"].f1
        f1_p = metrics.per_class["PARTIAL"].f1
        f1_u = metrics.per_class["UNSUPPORTED"].f1

        expected_macro = round((f1_s + f1_p + f1_u) / 3.0, 4)
        assert metrics.macro_f1 == expected_macro

    def test_6_missing_invalid_expected_verdict_rejected(self):
        """TEST 6: Missing or invalid expected verdict is rejected."""
        with pytest.raises(ValidationError):
            EvalExample(
                id="eval_test",
                category="test",
                claim="Claim",
                evidence="Evidence",
                expected_verdict="INVALID_VERDICT",  # type: ignore[arg-type]
            )

        with pytest.raises(ValidationError):
            EvalExample(
                id="eval_test",
                category="test",
                claim="Claim",
                evidence="Evidence",
                expected_verdict="",  # type: ignore[arg-type]
            )

        # Invalid label in calculate_evaluation_metrics
        with pytest.raises(ValueError, match="Invalid expected label"):
            calculate_evaluation_metrics(["UNKNOWN_LABEL"], ["SUPPORTED"])

    def test_7_evaluation_dataset_schema_validation(self):
        """TEST 7: Evaluation dataset schema is validated against EvalExample."""
        dataset_path = Path("data/verifier_eval.json")
        assert dataset_path.exists(), "Dataset data/verifier_eval.json must exist."

        examples = load_dataset(dataset_path)
        assert len(examples) >= 20

        for ex in examples:
            assert isinstance(ex, EvalExample)
            assert ex.expected_verdict in CLASS_LABELS

    def test_8_all_20_examples_have_valid_ids(self):
        """TEST 8: All examples have valid, unique IDs."""
        examples = load_dataset("data/verifier_eval.json")
        ids = [ex.id for ex in examples]

        assert len(ids) == 20
        assert len(set(ids)) == 20, "All example IDs must be strictly unique."

        for example_id in ids:
            assert example_id.startswith("eval_")
            assert len(example_id) > 5

    def test_9_all_examples_contain_claim_evidence_expected_verdict(self):
        """TEST 9: All examples contain non-empty claim, evidence, and expected_verdict."""
        examples = load_dataset("data/verifier_eval.json")

        for ex in examples:
            assert ex.claim.strip(), f"Example {ex.id} has empty claim"
            assert ex.evidence.strip(), f"Example {ex.id} has empty evidence"
            assert ex.expected_verdict in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}
            assert ex.category.strip(), f"Example {ex.id} has empty category"

    def test_10_evaluation_produces_exactly_one_prediction_per_input_example(self):
        """TEST 10: Evaluation produces exactly one prediction per input example."""
        examples = load_dataset("data/verifier_eval.json")
        mock_verifier = MockVerifierService(default_verdict="SUPPORTED")

        records, metrics = run_evaluation(examples, verifier_service=mock_verifier)

        assert len(records) == len(examples)
        assert mock_verifier.call_count == len(examples)

        record_ids = [r.id for r in records]
        example_ids = [e.id for e in examples]
        assert record_ids == example_ids


class TestEvaluationReportingAndSerialization:
    """Tests for prediction serialization and report generation."""

    def test_save_predictions_creates_valid_json(self, tmp_path):
        """Prediction serialization preserves all fields."""
        record = PredictionRecord(
            id="eval_001",
            category="direct_support",
            claim="Claim text",
            evidence="Evidence text",
            expected="SUPPORTED",
            predicted="SUPPORTED",
            correct=True,
            confidence=0.98,
            reasoning="Valid support",
            evidence_ids=["ev_eval_001"],
        )
        out_file = tmp_path / "test_predictions.json"
        saved_path = save_predictions([record], out_file)

        assert saved_path.exists()
        import json
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "eval_001"
        assert data[0]["correct"] is True

    def test_generate_evaluation_report_content(self, tmp_path):
        """Report generation writes all required markdown sections."""
        record = PredictionRecord(
            id="eval_001",
            category="direct_support",
            claim="Claim text",
            evidence="Evidence text",
            expected="SUPPORTED",
            predicted="SUPPORTED",
            correct=True,
            confidence=0.98,
            reasoning="Direct match",
            evidence_ids=["ev_eval_001"],
        )
        metrics = calculate_evaluation_metrics(["SUPPORTED"], ["SUPPORTED"])
        metadata = {
            "timestamp": "2026-09-26T12:00:00Z",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "temperature": 0.0,
        }
        out_file = tmp_path / "VERIFIER_EVALUATION.md"
        report_path = generate_evaluation_report(metrics, [record], metadata, out_file)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# Verifier Evaluation Report" in content
        assert "## 1. Objective" in content
        assert "## 2. Evaluation Dataset" in content
        assert "## 3. Evaluation Method" in content
        assert "## 4. Results" in content
        assert "## 5. Confusion Matrix" in content
        assert "## 6. Failure Analysis" in content
        assert "## 7. Limitations" in content
        assert "## 8. Reproducibility Metadata" in content
        assert "## 9. Conclusion" in content

    def test_ascii_confusion_matrix_formatting(self):
        """ASCII matrix formatting correctly renders headers, divider, and labels."""
        cm = [[4, 0, 0], [0, 2, 1], [0, 0, 13]]
        ascii_out = format_confusion_matrix_ascii(cm)
        assert "Predicted" in ascii_out
        assert "Expected S" in ascii_out
        assert "Expected P" in ascii_out
        assert "Expected U" in ascii_out
        assert "Orientation: Rows = Expected, Columns = Predicted" in ascii_out
