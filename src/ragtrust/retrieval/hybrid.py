"""Hybrid dense+sparse retrieval, fused with Reciprocal Rank Fusion (RRF).

Why RRF and not a weighted sum of scores. `Retriever` scores are cosine similarities
in [-1, 1] (roughly [0, 1] on real text); `BM25Retriever` scores are unbounded
Okapi weights that grow with query length and corpus statistics. These two scales
are not comparable, so combining them with a weighted sum would require calibrating
a weight/normalisation per corpus -- exactly the kind of fitted constant this project
avoids (see `Config.retrieval_gate`'s docstring for the same principle applied
elsewhere). RRF sidesteps the problem entirely: it only looks at each retriever's
*rank order*, not its score magnitude, so no calibration is needed.

    score(d) = sum_r 1 / (rrf_k + rank_r(d))

summed over every retriever `r` in which document `d` appears, where `rank_r(d)` is
1-based. `rrf_k` (default 60, the value from the original RRF paper) discounts the
importance of rank differences deep in the list, so one retriever burying a document
at rank 200 does not veto a document another retriever ranks 1st.
"""
from .index import Passage


class HybridRetriever:
    def __init__(self, dense, sparse, rrf_k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def build(self, passages: list) -> "HybridRetriever":
        self.dense.build(passages)
        self.sparse.build(passages)
        return self

    def search(self, query: str, k: int, candidates: int = None) -> list:
        if candidates is None:
            candidates = max(4 * k, 20)

        dense_results = self.dense.search(query, candidates)
        sparse_results = self.sparse.search(query, candidates)

        rrf_scores: dict = {}
        text_by_id: dict = {}
        for ranking in (dense_results, sparse_results):
            for rank, passage in enumerate(ranking, start=1):
                rrf_scores[passage.id] = rrf_scores.get(passage.id, 0.0) + 1.0 / (self.rrf_k + rank)
                text_by_id.setdefault(passage.id, passage.text)

        fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
        top = fused[:k]
        return [Passage(id=doc_id, text=text_by_id[doc_id], score=float(score))
                for doc_id, score in top]
