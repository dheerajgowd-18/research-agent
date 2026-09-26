# Verifier Evaluation Report

## 1. Objective

The objective of this evaluation is to empirically assess the evidentiary grounding accuracy of the claim-level verifier using an explicit, hand-labeled evaluation dataset.

The evaluation tests the verifier's ability to discriminate between:
- **`SUPPORTED`**: The supplied evidence directly and fully entails the substantive proposition.
- **`PARTIAL`**: The evidence supports only a subset of the claim, or the claim overstates the scope/hedging of the evidence.
- **`UNSUPPORTED`**: The evidence contradicts the claim, is irrelevant, or contains entity, numeric, temporal, or polarity mismatches.

> **Important Distinction**: This evaluation assesses grounding against provided text, not universal real-world truth. Results reflect performance on this specific 20-example benchmark and must not be overgeneralized to unconstrained real-world accuracy.

---

## 2. Evaluation Dataset

- **Dataset Size**: 20 hand-labeled examples
- **Location**: `data/verifier_eval.json`
- **Number of Categories**: 10
- **Categories Covered**: `contradiction`, `direct_support`, `entity_mismatch`, `irrelevant_evidence`, `negation`, `numeric_mismatch`, `paraphrase`, `partial_support`, `scope_mismatch`, `temporal_mismatch`
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
- **Total Examples**: 20
- **Correct Predictions**: 20
- **Incorrect Predictions**: 0
- **Overall Accuracy**: **1.0000** (100.0%)
- **Macro-Averaged F1**: **1.0000**

### Per-Class Performance
| Class Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `SUPPORTED` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `PARTIAL` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `UNSUPPORTED` | 1.0000 | 1.0000 | 1.0000 | 13 |

---

## 5. Confusion Matrix

The confusion matrix uses the canonical orientation:
- **Rows**: Expected Ground Truth
- **Columns**: Predicted Verdict

```
                 Predicted
                 S     P     U
            +-------------------
Expected S  |     4     0     0
Expected P  |     0     3     0
Expected U  |     0     0    13

Legend:
  S = SUPPORTED
  P = PARTIAL
  U = UNSUPPORTED
  Orientation: Rows = Expected, Columns = Predicted
```

### Interpretation
- Diagonal entries indicate correct classifications where expected equals predicted.
- Off-diagonal entries indicate systematic misclassification patterns (e.g. false positives where `UNSUPPORTED` is predicted as `SUPPORTED`, or boundary confusion between `PARTIAL` and `SUPPORTED`).

---

## 6. Failure Analysis

No classification errors observed on this evaluation set. All predicted verdicts matched ground truth labels.

---

## 7. Limitations

1. **Small Dataset Size**: The evaluation dataset comprises 20 examples. While designed to probe critical failure modes, statistical power is limited.
2. **Hand-Constructed Examples**: Synthetic examples may not fully capture the stylistic diversity, noise, or length of raw web scrapes.
3. **Model & Provider Dependence**: Results depend on the underlying model (`openai/gpt-oss-20b`) and provider (`groq`). Model updates or different architectures will yield different performance profiles.
4. **Subjectivity at Decision Boundaries**: The boundary between `PARTIAL` and `UNSUPPORTED` for complex scope qualifiers involves human judgment.
5. **Prompt Sensitivity**: Slight modifications to system instructions or prompt formatting can shift classification thresholds.
6. **No Universal Accuracy Claim**: This evaluation does NOT demonstrate that the verifier is "universally reliable." It demonstrates measurable performance on this specific benchmark under controlled conditions.

---

## 8. Reproducibility Metadata

- **Evaluation Date**: 2026-09-26T15:41:35.099545+00:00
- **LLM Provider**: `groq`
- **LLM Model**: `openai/gpt-oss-20b`
- **LLM Temperature**: `0.0`
- **Dataset Path**: `data\verifier_eval.json`
- **Predictions Output**: `reports/verifier_predictions.json`
- **Evaluation Script**: `evaluation/evaluate_verifier.py`

---

## 9. Conclusion

On the 20-example evaluation set, the verifier achieved an accuracy of **1.0000** and a Macro-F1 of **1.0000**. The confusion matrix and error analysis reveal how the model handles evidentiary nuance, highlighting areas where fine-grained qualifiers or partial containment require continued verification rigor.
