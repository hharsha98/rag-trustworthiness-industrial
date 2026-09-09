"""Faithfulness (grounding) -- METRICS.md Part II.1.

s_i = max_j p_ent(premise=d_j, hypothesis=c_i)
F   = mean_i s_i
kappa = mean_i max_j p_con(d_j, c_i)   (contradiction rate, reported separately)

Note the premise/hypothesis order: (passage, claim) -- this must not be
swapped, since NLI is not symmetric.
"""
from dataclasses import dataclass


@dataclass
class FaithfulnessResult:
    score: float
    per_claim: list
    contradiction_rate: float
    support_index: list


def faithfulness(claims: list, passages: list, nli) -> FaithfulnessResult:
    if not claims or not passages:
        return FaithfulnessResult(score=0.0, per_claim=[], contradiction_rate=0.0, support_index=[])

    n_claims = len(claims)
    n_passages = len(passages)

    pairs = [(d, c) for c in claims for d in passages]
    probs = nli.batch_probs(pairs)

    per_claim = []
    support_index = []
    contradiction_per_claim = []
    for i in range(n_claims):
        row = probs[i * n_passages:(i + 1) * n_passages]
        entailments = [r["entailment"] for r in row]
        contradictions = [r["contradiction"] for r in row]
        best_j = max(range(n_passages), key=lambda j: entailments[j])
        per_claim.append(entailments[best_j])
        support_index.append(best_j)
        contradiction_per_claim.append(max(contradictions))

    score = sum(per_claim) / n_claims
    contradiction_rate = sum(contradiction_per_claim) / n_claims
    return FaithfulnessResult(
        score=score,
        per_claim=per_claim,
        contradiction_rate=contradiction_rate,
        support_index=support_index,
    )
