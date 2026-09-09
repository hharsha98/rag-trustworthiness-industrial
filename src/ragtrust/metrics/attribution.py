"""Attribution -- METRICS.md Part II.2.

A claim is "supported" iff its best-supporting passage's entailment
probability >= tau. Vacuous-truth conventions (see METRICS.md):
  precision = 1.0 when no citation is emitted
  recall    = 1.0 when no claim is supported
  f1        = 0.0 when precision + recall == 0
"""
from dataclasses import dataclass


@dataclass
class AttributionResult:
    precision: float
    recall: float
    f1: float


def attribution(claims: list, citations: dict, passages: list, nli, tau: float = 0.5) -> AttributionResult:
    n_claims = len(claims)
    n_passages = len(passages)

    supported = [False] * n_claims
    entailment_matrix = [[0.0] * n_passages for _ in range(n_claims)]

    if n_claims and n_passages:
        pairs = [(d, c) for c in claims for d in passages]
        probs = nli.batch_probs(pairs)
        for i in range(n_claims):
            row = probs[i * n_passages:(i + 1) * n_passages]
            for j, r in enumerate(row):
                entailment_matrix[i][j] = r["entailment"]
            supported[i] = max(entailment_matrix[i]) >= tau

    if not citations:
        precision = 1.0
    else:
        correct = 0
        for claim_idx, passage_idx in citations.items():
            if 0 <= claim_idx < n_claims and 0 <= passage_idx < n_passages:
                if entailment_matrix[claim_idx][passage_idx] >= tau:
                    correct += 1
        precision = correct / len(citations)

    supported_indices = [i for i in range(n_claims) if supported[i]]
    if not supported_indices:
        recall = 1.0
    else:
        cited_supported = sum(1 for i in supported_indices if i in citations)
        recall = cited_supported / len(supported_indices)

    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return AttributionResult(precision=precision, recall=recall, f1=f1)
