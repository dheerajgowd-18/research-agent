"""Metric calculations and confusion matrix generation using scikit-learn."""

from typing import Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from evaluation.schemas import CLASS_LABELS, ClassMetric, EvaluationMetrics
from verified_research.models.research import VerdictType


def calculate_evaluation_metrics(
    expected: Sequence[VerdictType | str],
    predicted: Sequence[VerdictType | str],
    labels: Sequence[VerdictType | str] = CLASS_LABELS,
) -> EvaluationMetrics:
    """Calculate multi-class classification metrics with canonical class ordering.

    Canonical label ordering:
        ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]

    Confusion Matrix Orientation:
        rows = expected (ground truth)
        columns = predicted

    Args:
        expected: Sequence of expected ground truth labels.
        predicted: Sequence of predicted labels.
        labels: Ordered list of labels (defaults to CLASS_LABELS).

    Returns:
        EvaluationMetrics instance with accuracy, macro-F1, per-class metrics,
        and confusion matrix.
    """
    labels_list = list(labels)

    # Validate that inputs are non-empty and matched in length
    if len(expected) != len(predicted):
        raise ValueError(
            f"Mismatched lengths: expected {len(expected)} vs predicted {len(predicted)}"
        )
    if not expected:
        raise ValueError("Cannot calculate metrics on empty input sequences.")

    # Validate all labels belong to the allowed label set
    allowed_set = set(labels_list)
    for idx, e in enumerate(expected):
        if e not in allowed_set:
            raise ValueError(f"Invalid expected label '{e}' at index {idx}. Allowed: {labels_list}")
    for idx, p in enumerate(predicted):
        if p not in allowed_set:
            raise ValueError(f"Invalid predicted label '{p}' at index {idx}. Allowed: {labels_list}")

    # Accuracy
    acc = float(accuracy_score(expected, predicted))

    # Macro F1
    macro = float(
        f1_score(expected, predicted, labels=labels_list, average="macro", zero_division=0.0)
    )

    # Per-class precision, recall, F1, support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        expected,
        predicted,
        labels=labels_list,
        zero_division=0.0,
    )

    per_class_metrics: dict[str, ClassMetric] = {}
    for idx, label in enumerate(labels_list):
        per_class_metrics[str(label)] = ClassMetric(
            precision=float(precisions[idx]),
            recall=float(recalls[idx]),
            f1=float(f1s[idx]),
            support=int(supports[idx]),
        )

    # Confusion matrix: sklearn produces rows=y_true (expected), columns=y_pred (predicted)
    cm_array = confusion_matrix(expected, predicted, labels=labels_list)
    cm_matrix: list[list[int]] = cm_array.astype(int).tolist()

    return EvaluationMetrics(
        class_labels=[str(l) for l in labels_list],
        accuracy=round(acc, 4),
        macro_f1=round(macro, 4),
        per_class={k: ClassMetric(
            precision=round(v.precision, 4),
            recall=round(v.recall, 4),
            f1=round(v.f1, 4),
            support=v.support,
        ) for k, v in per_class_metrics.items()},
        confusion_matrix=cm_matrix,
    )


def format_confusion_matrix_ascii(
    cm: list[list[int]],
    labels: Sequence[str] = ("SUPPORTED", "PARTIAL", "UNSUPPORTED"),
) -> str:
    """Format the 3x3 confusion matrix as an aligned ASCII table.

    Orientation:
        rows = expected
        columns = predicted
    """
    short_labels = [l[0] for l in labels]  # S, P, U
    header = "                 Predicted\n"
    col_headers = "              " + "  ".join(f"{sl:>4}" for sl in short_labels) + "\n"
    divider = "            +" + "-" * (len(short_labels) * 6 + 1) + "\n"

    rows = []
    for idx, row in enumerate(cm):
        label_prefix = f"Expected {short_labels[idx]}  |"
        cells = "  ".join(f"{val:>4}" for val in row)
        rows.append(f"{label_prefix}  {cells}")

    legend = (
        "\n\nLegend:\n"
        + "\n".join(f"  {sl} = {full}" for sl, full in zip(short_labels, labels))
        + "\n  Orientation: Rows = Expected, Columns = Predicted"
    )

    return header + col_headers + divider + "\n".join(rows) + legend
