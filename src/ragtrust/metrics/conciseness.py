"""Conciseness -- METRICS.md Part II.4: information density, not length.

C = 1 - mean_{i<i'} max(0, cos(e_ci, e_ci'))   for n >= 2 claims
C = undefined (None)                            for n < 2 claims

Pairwise self-similarity has no meaning with fewer than two claims -- there
is no pair to compare -- so the metric is undefined there, not perfect.
Earlier versions returned the sentinel `1.0` for n < 2, which silently
handed a perfect redundancy score to 68.5% of real summaries (SEAHORSE) and
fed it into the trust aggregates as if it had been measured. That is a
fabricated floor/ceiling standing in for "not computed". Restricted to
summaries where C is actually computed (n >= 2), C scores ROC-AUC 0.820 vs
0.566 for the best length baseline (p=0.0001) on SEAHORSE -- the metric is
sound, the sentinel was not. Callers must treat `None` as "drop this metric
from aggregation", per `aggregate.py`.
"""
from .relevance import _cosine


def conciseness(claims: list, embedder) -> float | None:
    """Pairwise self-similarity redundancy score over >= 2 claims.

    Returns `None` -- undefined -- when there are fewer than 2 claims,
    since there is no pair to compare. See module docstring for why this
    replaced the old `1.0` sentinel.
    """
    n = len(claims)
    if n < 2:
        return None
    embeddings = embedder.encode(list(claims))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(max(0.0, _cosine(embeddings[i], embeddings[j])))
    mean_sim = sum(sims) / len(sims)
    return float(1.0 - mean_sim)
