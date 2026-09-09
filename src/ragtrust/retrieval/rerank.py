"""Cross-encoder reranking on top of a base retriever.

A cross-encoder scores a (query, passage) pair jointly (unlike a bi-encoder, which
embeds each side independently and compares vectors), so it can pick up on
interactions a dense or sparse retriever's independent representations miss. The
cost is that it must run once per candidate pair, so it is applied only to a shallow
candidate list from a cheap first-stage retriever, not the whole corpus.

*** SCORE-SCALE WARNING ***
`sentence_transformers.CrossEncoder` (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2")
returns raw, unbounded logits -- not a cosine similarity and not a probability. A
"good" match might score 6.0 and a "bad" one -8.0; these numbers carry no fixed
meaning outside relative ranking. `Config.retrieval_gate` (default 0.30) is a
cosine threshold, calibrated on BEIR data by `experiments/09_gate_calibration.py`,
so thresholding a reranked `Passage.score` against it would compare two different
unit systems.

The pipeline itself is no longer exposed to this: its abstention gate does not read
`Passage.score` at all, but recomputes cosine directly via
`metrics.relevance.max_context_similarity`, which is scale-stable across every
retrieval mode. The warning still applies to any caller that thresholds a reranked
score of its own. Callers needing a [0,1]-ish confidence should use
`.normalized_scores` (sigmoid of the raw logits, computed for the most recent
`.search()` call) instead of `Passage.score`, and should NOT feed cross-encoder
`Passage.score` values into `retrieval_gate`. This repository wires `rerank` as an
independent, default-OFF config flag (not composed with `retrieval_gate` logic) for
exactly this reason -- see the wiring note in `pipeline.py`/`config.py`.
"""
import math

from .index import Passage


class CrossEncoderReranker:
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 candidates: int = 20):
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidates = candidates
        self._model = None
        # Sigmoid-mapped scores for the passages returned by the most recent
        # `.search()` call, in the same order as that call's return value. Populated
        # so callers can get a bounded, comparable confidence without touching
        # `Passage.score`, which stays a raw logit (see module docstring).
        self.normalized_scores: list = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def build(self, passages: list) -> "CrossEncoderReranker":
        self.base_retriever.build(passages)
        return self

    def search(self, query: str, k: int) -> list:
        candidates = self.base_retriever.search(query, self.candidates)
        if not candidates:
            self.normalized_scores = []
            return []

        pairs = [(query, p.text) for p in candidates]
        logits = self.model.predict(pairs)

        rescored = [
            Passage(id=p.id, text=p.text, score=float(logit))
            for p, logit in zip(candidates, logits)
        ]
        order = sorted(range(len(rescored)), key=lambda i: rescored[i].score, reverse=True)
        top_idx = order[:k]

        self.normalized_scores = [1.0 / (1.0 + math.exp(-rescored[i].score)) for i in top_idx]
        return [rescored[i] for i in top_idx]
