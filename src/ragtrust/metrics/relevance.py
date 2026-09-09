"""Relevance metrics -- METRICS.md Part II.3.

Context relevance and answer relevance are kept as separate metrics. `scaled=False`
returns the raw cosine mean, unclamped -- used for calibration analysis (see
experiments/13_relevance_validation.py). `scaled=True` (the default, used for real
scoring) maps into [0,1] by clamping at zero.

On the choice of mapping. An affine map `t -> (1+t)/2` also sends [-1,1] to [0,1]
and has the appeal of preserving order over negative similarities; it was considered
and rejected. It is wrong for this purpose: sentence encoders essentially never
produce negative cosine on real text, so in practice it puts a **floor of ~0.5**
under the metric. A floor like that is exactly the kind of defect a trustworthiness
metric must not have: a bounded-below metric cannot signal failure, and inside the
non-compensatory geometric aggregate it can never pull the overall score down.
Clamping gives up ordering among negative cosines -- which do not occur in
practice -- and buys the property that actually matters: content orthogonal to the
query scores 0, not 0.5.

Embedder convention used throughout this package: `embedder.encode(list_of_str)
-> array-like of shape (n, d)`.
"""
import numpy as np


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def context_relevance(query: str, passages: list, embedder, scaled: bool = True) -> float:
    if not passages:
        return 0.0
    q_emb = embedder.encode([query])[0]
    p_embs = embedder.encode(list(passages))
    sims = []
    for p_emb in p_embs:
        c = _cosine(q_emb, p_emb)
        sims.append(max(0.0, c) if scaled else c)
    return float(np.mean(sims))


def max_context_similarity(query: str, passages: list, embedder) -> float:
    """Highest cosine similarity between the query and any retrieved passage, in [0,1].

    Exists so the pipeline's pre-generation abstention gate has a scale-stable quantity
    to threshold. `Passage.score` cannot serve that purpose: dense retrieval returns
    cosine, BM25 returns unbounded term-weight sums, RRF returns reciprocal-rank sums
    capped near 2/(rrf_k+1) ~ 0.03, and a cross-encoder returns unbounded logits. A
    single threshold compared against `score` would mean something different in every
    retrieval mode -- and against RRF it would reject everything.
    """
    if not passages:
        return 0.0
    q_emb = embedder.encode([query])[0]
    p_embs = embedder.encode(list(passages))
    return max(max(0.0, _cosine(q_emb, p)) for p in p_embs)


def answer_relevance(query: str, generated_questions: list, embedder) -> float:
    if not generated_questions:
        return 0.0
    q_emb = embedder.encode([query])[0]
    gq_embs = embedder.encode(list(generated_questions))
    sims = [_cosine(q_emb, g) for g in gq_embs]
    return float(np.mean(sims))


def ndcg_at_k(ranked_ids: list, relevant_ids: list, k: int) -> float:
    relevant = set(relevant_ids)
    ranked = list(ranked_ids)[:k]
    dcg = sum((1.0 if rid in relevant else 0.0) / np.log2(i + 2) for i, rid in enumerate(ranked))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list, relevant_ids: list, k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    ranked = set(list(ranked_ids)[:k])
    return float(len(ranked & relevant) / len(relevant))


def mrr(ranked_ids: list, relevant_ids: list) -> float:
    relevant = set(relevant_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in relevant:
            return float(1.0 / (i + 1))
    return 0.0
