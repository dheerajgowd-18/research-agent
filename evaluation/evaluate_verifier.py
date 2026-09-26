"""Evaluation runner for the claim-level verifier.

Loads the evaluation dataset, evaluates each example against the verifier service,
computes classification metrics, saves prediction records, and generates a markdown report.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Sequence
from evaluation.metrics import calculate_evaluation_metrics, format_confusion_matrix_ascii
from evaluation.schemas import (
    CLASS_LABELS,
    EvalExample,
    EvaluationMetrics,
    PredictionRecord,
)
from verified_research.agents.verifier import ClaimVerifierService, VerifierService
from verified_research.config.settings import get_settings
from verified_research.models.research import Claim, Evidence

logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str | Path = "data/verifier_eval.json") -> list[EvalExample]:
    """Load and validate the evaluation dataset.

    Args:
        dataset_path: Path to verifier_eval.json.

    Returns:
        List of validated EvalExample instances.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If JSON is malformed, duplicates exist, or schema validation fails.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON dataset at {path}: {e}") from e

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset root must be a list of examples, got {type(raw_data).__name__}")

    examples: list[EvalExample] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item at index {idx} must be an object/dict")
        example = EvalExample.model_validate(item)
        if example.id in seen_ids:
            raise ValueError(f"Duplicate example ID '{example.id}' found at index {idx}")
        seen_ids.add(example.id)
        examples.append(example)

    return examples


def run_evaluation(
    examples: Sequence[EvalExample],
    verifier_service: VerifierService | None = None,
) -> tuple[list[PredictionRecord], EvaluationMetrics]:
    """Evaluate a sequence of examples using the provided verifier service.

    Args:
        examples: Validated evaluation examples.
        verifier_service: VerifierService instance. If None, instantiates ClaimVerifierService().

    Returns:
        Tuple of (list of PredictionRecords, EvaluationMetrics).
    """
    if not examples:
        raise ValueError("Cannot run evaluation on empty example list.")

    verifier = verifier_service if verifier_service is not None else ClaimVerifierService()

    records: list[PredictionRecord] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []

    for example in examples:
        # Construct isolated Claim and Evidence pair
        ev_id = f"ev_{example.id}"
        src_id = f"src_{example.id}"
        claim = Claim(
            claim_id=example.id,
            text=example.claim,
            evidence_ids=[ev_id],
        )
        evidence = Evidence(
            evidence_id=ev_id,
            source_id=src_id,
            text=example.evidence,
        )

        result = verifier.verify_claim(claim, [evidence])

        is_correct = bool(result.verdict == example.expected_verdict)

        record = PredictionRecord(
            id=example.id,
            category=example.category,
            claim=example.claim,
            evidence=example.evidence,
            expected=example.expected_verdict,
            predicted=result.verdict,
            correct=is_correct,
            confidence=result.confidence,
            reasoning=result.reasoning,
            evidence_ids=result.evidence_ids,
        )
        records.append(record)
        expected_labels.append(example.expected_verdict)
        predicted_labels.append(result.verdict)

    # Compute classification metrics
    metrics = calculate_evaluation_metrics(
        expected=expected_labels,
        predicted=predicted_labels,
        labels=CLASS_LABELS,
    )

    return records, metrics


def save_predictions(
    records: Sequence[PredictionRecord],
    output_path: str | Path = "reports/verifier_predictions.json",
) -> Path:
    """Serialize and save prediction records to a JSON file.

    Args:
        records: List of PredictionRecord instances.
        output_path: Destination path.

    Returns:
        Path to the saved predictions file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [record.model_dump() for record in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def generate_evaluation_report(
    metrics: EvaluationMetrics,
    records: Sequence[PredictionRecord],
    metadata: dict[str, str | int | float],
    output_path: str | Path = "reports/VERIFIER_EVALUATION.md",
) -> Path:
    """Generate the comprehensive evaluation markdown report.

    Args:
        metrics: Computed EvaluationMetrics.
        records: Full list of PredictionRecords.
        metadata: Execution and environment metadata (date, provider, model).
        output_path: Destination path for VERIFIER_EVALUATION.md.

    Returns:
        Path to the written report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    categories = sorted({r.category for r in records})
    failures = [r for r in records if not r.correct]
    cm_ascii = format_confusion_matrix_ascii(metrics.confusion_matrix)

    # Build per-class table
    per_class_rows = []
    for label in CLASS_LABELS:
        m = metrics.per_class[label]
        per_class_rows.append(
            f"| `{label}` | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {m.support} |"
        )
    per_class_table = "\n".join(per_class_rows)

    # Build failure analysis entries
    if failures:
        failure_entries = []
        for f in failures:
            failure_entries.append(
                f"### Example `{f.id}` — Category: `{f.category}`\n"
                f"- **Claim**: \"{f.claim}\"\n"
                f"- **Evidence**: \"{f.evidence}\"\n"
                f"- **Expected Verdict**: `{f.expected}`\n"
                f"- **Predicted Verdict**: `{f.predicted}` (Confidence: {f.confidence:.2f})\n"
                f"- **Model Reasoning**: {f.reasoning}\n"
                f"- **Observed Discrepancy**: Expected `{f.expected}`, but model predicted `{f.predicted}`.\n"
                f"- **Hypothesized Failure Mode**: "
                f"{_hypothesize_failure_mode(f)}\n"
            )
        failure_section = "\n".join(failure_entries)
    else:
        failure_section = (
            "No classification errors observed on this evaluation set. "
            "All predicted verdicts matched ground truth labels."
        )

    report_content = f"""# Verifier Evaluation Report

## 1. Objective

The objective of this evaluation is to empirically assess the evidentiary grounding accuracy of the claim-level verifier using an explicit, hand-labeled evaluation dataset.

The evaluation tests the verifier's ability to discriminate between:
- **`SUPPORTED`**: The supplied evidence directly and fully entails the substantive proposition.
- **`PARTIAL`**: The evidence supports only a subset of the claim, or the claim overstates the scope/hedging of the evidence.
- **`UNSUPPORTED`**: The evidence contradicts the claim, is irrelevant, or contains entity, numeric, temporal, or polarity mismatches.

> **Important Distinction**: This evaluation assesses grounding against provided text, not universal real-world truth. Results reflect performance on this specific {len(records)}-example benchmark and must not be overgeneralized to unconstrained real-world accuracy.

---

## 2. Evaluation Dataset

- **Dataset Size**: {len(records)} hand-labeled examples
- **Location**: `data/verifier_eval.json`
- **Number of Categories**: {len(categories)}
- **Categories Covered**: {", ".join(f"`{c}`" for c in categories)}
- **Class Label Set**: `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`

### Labeling Methodology
Ground truth labels were constructed completely independently of the verifier model's outputs. Each example was written by a human engineer with an explicit target entailment relationship (direct support, paraphrase, partial support, contradiction, irrelevance, numeric mismatch, temporal mismatch, entity mismatch, scope mismatch, or negation). Expected labels were assigned based strictly on classical natural language inference (NLI) semantics.

---

## 3. Evaluation Method

The evaluation pipeline executes as follows:

```
Claim + Evidence Snapshot
          │
          ▼
   ClaimVerifierService
          │
          ▼
  Structured Output (VerificationResult)
  [verdict, confidence, reasoning]
          │
          ▼
  Compare Predicted vs Ground Truth
          │
          ▼
  Calculate Multi-Class Classification Metrics
  (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

Every example is passed to the verifier with its corresponding isolated evidence snapshot. Model predictions, confidence scores, and reasoning are captured deterministically without test-time mutation.

---

## 4. Results

### Global Metrics
- **Total Examples**: {len(records)}
- **Correct Predictions**: {len(records) - len(failures)}
- **Incorrect Predictions**: {len(failures)}
- **Overall Accuracy**: **{metrics.accuracy:.4f}** ({metrics.accuracy * 100:.1f}%)
- **Macro-Averaged F1**: **{metrics.macro_f1:.4f}**

### Per-Class Performance
| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
{per_class_table}

---

## 5. Confusion Matrix

The confusion matrix uses the canonical orientation:
- **Rows**: Expected Ground Truth
- **Columns**: Predicted Verdict

```
{cm_ascii}
```

### Interpretation
- Diagonal entries indicate correct classifications where expected equals predicted.
- Off-diagonal entries indicate systematic misclassification patterns (e.g. false positives where `UNSUPPORTED` is predicted as `SUPPORTED`, or boundary confusion between `PARTIAL` and `SUPPORTED`).

---

## 6. Failure Analysis

{failure_section}

---

## 7. Limitations

1. **Small Dataset Size**: The evaluation dataset comprises {len(records)} examples. While designed to probe critical failure modes, statistical power is limited.
2. **Hand-Constructed Examples**: Synthetic examples may not fully capture the stylistic diversity, noise, or length of raw web scrapes.
3. **Model & Provider Dependence**: Results depend on the underlying model (`{metadata.get("model", "unknown")}`) and provider (`{metadata.get("provider", "unknown")}`). Model updates or different architectures will yield different performance profiles.
4. **Subjectivity at Decision Boundaries**: The boundary between `PARTIAL` and `UNSUPPORTED` for complex scope qualifiers involves human judgment.
5. **Prompt Sensitivity**: Slight modifications to system instructions or prompt formatting can shift classification thresholds.
6. **No Universal Accuracy Claim**: This evaluation does NOT demonstrate that the verifier is "universally reliable." It demonstrates measurable performance on this specific benchmark under controlled conditions.

---

## 8. Reproducibility Metadata

- **Evaluation Date**: {metadata.get("timestamp", "unknown")}
- **LLM Provider**: `{metadata.get("provider", "unknown")}`
- **LLM Model**: `{metadata.get("model", "unknown")}`
- **LLM Temperature**: `{metadata.get("temperature", "unknown")}`
- **Dataset Path**: `{metadata.get("dataset_path", "data/verifier_eval.json")}`
- **Predictions Output**: `reports/verifier_predictions.json`
- **Evaluation Script**: `evaluation/evaluate_verifier.py`

---

## 9. Conclusion

On the {len(records)}-example evaluation set, the verifier achieved an accuracy of **{metrics.accuracy:.4f}** and a Macro-F1 of **{metrics.macro_f1:.4f}**. The confusion matrix and error analysis reveal how the model handles evidentiary nuance, highlighting areas where fine-grained qualifiers or partial containment require continued verification rigor.
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return path


def _hypothesize_failure_mode(record: PredictionRecord) -> str:
    """Generate an evidence-based hypothesis for a misclassification."""
    exp = record.expected
    pred = record.predicted
    cat = record.category

    if exp == "PARTIAL" and pred == "SUPPORTED":
        return "Model exhibited lenience bias: accepted partial or regionally restricted evidence as full support for a broader claim."
    elif exp == "PARTIAL" and pred == "UNSUPPORTED":
        return "Model treated the missing portion of a multi-part claim as grounds for total rejection rather than recognizing partial support."
    elif exp == "UNSUPPORTED" and pred == "SUPPORTED":
        if cat == "numeric_mismatch":
            return "Model missed numerical discrepancy between evidence and claim."
        elif cat == "temporal_mismatch":
            return "Model failed to account for temporal qualifiers (e.g. past tenure vs current activity)."
        elif cat == "entity_mismatch":
            return "Model confused distinct named entities or failed entity attribution check."
        elif cat == "contradiction":
            return "Model suffered from negation blindness or failed to recognize direct antonymous refutation."
        elif cat == "scope_mismatch":
            return "Model inflated preliminary or exploratory findings into definitive universal claims."
        else:
            return "Model exhibited prior knowledge leakage or false positive hallucination."
    elif exp == "SUPPORTED" and pred == "UNSUPPORTED":
        return "Model was overly strict or failed to resolve acceptable paraphrastic equivalence."
    elif exp == "SUPPORTED" and pred == "PARTIAL":
        return "Model was overly cautious: classified complete semantic support as merely partial."
    else:
        return f"Misclassified {exp} as {pred} under category '{cat}'."


def main() -> None:
    """CLI entry point to execute evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()

    dataset_path = Path("data/verifier_eval.json")
    print(f"Loading evaluation dataset from {dataset_path}...")
    examples = load_dataset(dataset_path)
    print(f"Loaded {len(examples)} examples.")

    metadata: dict[str, str | int | float] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "dataset_path": str(dataset_path),
        "num_examples": len(examples),
    }

    print(f"Running evaluation using provider={settings.llm_provider}, model={settings.llm_model}...")
    records, metrics = run_evaluation(examples)

    # Save predictions
    pred_path = save_predictions(records)
    print(f"Saved {len(records)} predictions to {pred_path}")

    # Generate markdown report
    report_path = generate_evaluation_report(metrics, records, metadata)
    print(f"Generated evaluation report at {report_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Macro-F1:  {metrics.macro_f1:.4f}")
    print("\nPer-Class Metrics:")
    for label, cm in metrics.per_class.items():
        print(f"  {label:<12} Precision: {cm.precision:.4f}  Recall: {cm.recall:.4f}  F1: {cm.f1:.4f}  (n={cm.support})")
    print("\nConfusion Matrix (Rows=Expected, Cols=Predicted):")
    print(format_confusion_matrix_ascii(metrics.confusion_matrix))


if __name__ == "__main__":
    main()
