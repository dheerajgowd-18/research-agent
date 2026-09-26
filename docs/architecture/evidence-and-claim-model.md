# Architecture Design: Evidence & Claim Model

This document details the architectural rationale, data models, traceability mechanisms, and structural validation rules for **Phase 4 (Evidence & Claim Modeling)** in the **Verified Research Agent**.

---

## 1. Traceability Hierarchy

```
       [Claim]          (Atomic, testable factual proposition)
          │
          ▼
      [Evidence]        (Immutable textual snapshot excerpt)
          │
          ▼
       [Source]         (Retrieved document/URL container)
```

### Key Architectural Invariant:
$$\text{Structural Validation} \neq \text{Semantic Verification}$$

---

## 2. Why Source Alone is Insufficient

In Phase 1, findings linked directly to entire documents (`Finding -> source_ids -> Source`).
While this established basic provenance, linking directly to entire web pages or documents is insufficient for true verification:
1. **Scope Mismatch**: A web source may contain thousands of words covering dozens of topics. Pointing to `src_001` does not identify *which* specific sentence or paragraph supports a given assertion.
2. **Web Page Mutability**: External web content changes, moves, or gets deleted. Without preserving exact textual excerpts, retrospective auditing is fragile.
3. **Automated Verification Difficulty**: A future verification agent or Natural Language Inference (NLI) model cannot reliably evaluate a 20-word claim against a 5,000-word document without hallucinating or running out of context window.

---

## 3. What Evidence Means

An **`Evidence`** instance represents a specific, preserved textual excerpt extracted from a `Source`.

```python
class Evidence(BaseModel):
    evidence_id: str  # e.g., 'ev_001'
    source_id: str    # e.g., 'src_001'
    text: str         # Preserved excerpt/snapshot
```

### Core Characteristics:
- **Exact Text Snapshot**: Contains the actual passage used by the analyst, captured at the moment of research.
- **Single Source Parent**: Every evidence item links to exactly one `source_id`.
- **Decoupled from Live URLs**: Downstream verifiers inspect `Evidence.text` without re-fetching live URLs.

---

## 4. What Claim Means

A **`Claim`** represents an atomic, unambiguous factual proposition that can later be independently checked for truth against evidence excerpts.

```python
class Claim(BaseModel):
    claim_id: str            # e.g., 'claim_001'
    text: str                # Specific factual proposition
    evidence_ids: list[str]  # Must contain at least 1 evidence ID
```

### Core Characteristics:
- **Atomic Proposition**: Focused on a single testable fact (e.g., *"Hybrid fluxonium-transmon qubits achieve 2-qubit gate fidelities above 99.9%."*).
- **Mandatory Grounding**: A claim cannot exist without at least one supporting `evidence_id`.
- **Many-to-Many Linking**: A claim may cite multiple evidence excerpts, and an evidence excerpt may support multiple claims.

---

## 5. Finding vs. Claim

It is vital to distinguish analytical **Findings** from factual **Claims**:

| Dimension | Finding (`Finding`) | Claim (`Claim`) |
| :--- | :--- | :--- |
| **Purpose** | Analytical synthesis answering the research inquiry | Atomic factual statement prepared for verification |
| **Granularity** | Macro-level synthesis (often combines themes) | Micro-level proposition (single falsifiable assertion) |
| **Target Citation** | Directly references `source_ids` | Directly references `evidence_ids` |
| **Downstream Consumer** | Human reader / Report writer | Automated Verifier / NLI model |

---

## 6. Claim $\rightarrow$ Evidence $\rightarrow$ Source Traceability

Every factual assertion retains end-to-end provenance:

```
Claim: claim_001
  text: "Surface code experiments demonstrated 2-qubit gate fidelities exceeding 99.9%."
  evidence_ids: ["ev_001"]
        │
        ▼
Evidence: ev_001
  source_id: "src_001"
  text: "Recent experiments demonstrate 2-qubit gate fidelities exceeding 99.9% using fluxonium hybrids."
        │
        ▼
Source: src_001
  title: "Superconducting Qubit Advances in 2026"
  url: "https://quantum-research.org/advances-2026"
```

The helper [`get_claim_sources(claim_id, evidence, sources)`](file:///d:/research-agent/src/verified_research/models/traceability.py) resolves this chain deterministically, returning the exact `Source` objects underlying the claim's evidence.

---

## 7. Why Stable Identifiers are Used

Using deterministic string identifiers (`src_001`, `ev_001`, `claim_001`) ensures:
- **Zero Ambiguity**: Clear distinction between sources, excerpts, and assertions.
- **Inspectability**: Human operators and debugging logs can read `['ev_001', 'ev_002']` and follow the exact trail.
- **Database Readiness**: These string IDs map directly into relational primary/foreign keys or graph database edges in production deployments.

---

## 8. Evidence Snapshots

The `Evidence.text` field holds a local immutable snapshot.
- If the original article at `https://example.com/item` is updated, paywalled, or taken offline tomorrow, the `Evidence.text` stored in the graph state remains intact.
- The verifier judges whether `Claim.text` follows from `Evidence.text`, guaranteeing deterministic reproducibility regardless of external web volatility.

---

## 9. Structural Validation vs. Semantic Verification

```
+--------------------------------------------------------------------------+
|                       STRUCTURAL VALIDATION (Phase 4)                    |
|                                                                          |
|   Checks referential integrity:                                          |
|     - Do all referenced evidence_ids exist in evidence state?            |
|     - Do all referenced source_ids exist in sources state?               |
|     - Are IDs unique and non-empty?                                      |
|                                                                          |
|   Rule: DOES NOT check whether evidence actually proves the claim!       |
+--------------------------------------------------------------------------+
                                    ≠
+--------------------------------------------------------------------------+
|                       SEMANTIC VERIFICATION (Phase 5)                    |
|                                                                          |
|   Checks semantic truth and entailment:                                  |
|     - Does Evidence.text actually entail Claim.text?                     |
|     - Is the claim supported, contradicted, or ungrounded?               |
|     - Requires NLI model or specialized verification judge.              |
+--------------------------------------------------------------------------+
```

### Illustrative Example:
- **Claim**: *"The method achieved 82% precision."*
- **Evidence**: *"The method achieved 62% precision."*
- **Structural Validation Outcome**: **PASS** (Both IDs exist and link correctly).
- **Semantic Verification Outcome**: **CONTRADICTED** (Deferred to Phase 5).

---

## 10. Citation Coverage

**Citation Coverage** measures the structural proportion of claims grounded by valid evidence:

$$\text{Citation Coverage} = \frac{\text{Number of Claims with Valid Evidence}}{\text{Total Number of Claims}}$$

### Important Warning:
- **Citation Coverage $\neq$ Verifier Accuracy**:
  A citation coverage of `1.0` (100%) simply means that every claim points to an existing evidence excerpt. It **does not guarantee** that any of the claims are factually true or that the evidence text entails them.

---

## 11. How Phase 4 Prepares for Phase 5 (The Verifier)

Phase 4 establishes the precise data structures needed for the verifier node:
1. The verifier will receive pairs of `(Claim.text, Evidence.text)`.
2. Because each `Evidence` is a compact excerpt, the verifier can perform focused Natural Language Inference without prompt dilution.
3. The verifier will append a `VerificationVerdict` (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`) to each claim without modifying the underlying evidence provenance.
