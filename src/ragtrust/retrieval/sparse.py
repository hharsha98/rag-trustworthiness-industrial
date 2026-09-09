"""Dependency-free BM25 (Okapi) sparse retrieval.

`BM25Retriever` mirrors `Retriever` (`.build(passages)` / `.search(query, k)`) from
`retrieval/index.py`, so it can be swapped in wherever a dense retriever is used --
this is what makes `HybridRetriever` (retrieval/hybrid.py) able to treat both
retrievers uniformly.

Tokenisation is deliberately minimal: lowercase, split on runs of non-alphanumeric
characters. No stemming, no stopword removal, no lemmatisation -- BM25's term
weighting (IDF + saturation) already does most of the useful work on a small corpus,
and adding a stemmer would be one more thing to justify and test for a component
that exists to give the dense retriever a lexical-overlap alternative to fuse with.
"""
import math
import re
from collections import Counter

from .index import Passage

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> list:
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


class BM25Retriever:
    """Okapi BM25 over an in-memory corpus.

    score(q, d) = sum_{t in q} IDF(t) * ( f(t,d) * (k1+1) ) / ( f(t,d) + k1*(1 - b + b*|d|/avgdl) )

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )  -- the "+1" (BM25+ style) keeps
    IDF non-negative for terms that appear in every document, which matters here because
    a single-document corpus would otherwise divide-by/near-zero or go negative.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passages: list = []
        self._doc_tokens: list = []
        self._doc_len: list = []
        self._avgdl: float = 0.0
        self._df: Counter = Counter()
        self._idf: dict = {}
        self._n_docs: int = 0

    def build(self, passages: list) -> "BM25Retriever":
        self.passages = list(passages)
        self._doc_tokens = [_tokenize(p) for p in self.passages]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._n_docs = len(self.passages)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

        self._df = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] += 1

        self._idf = {
            term: math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self._df.items()
        }
        return self

    def _score(self, query_tokens: list, doc_index: int) -> float:
        doc_tf = Counter(self._doc_tokens[doc_index])
        doc_len = self._doc_len[doc_index]
        # avgdl > 0 whenever there is at least one non-empty document; guarded anyway.
        denom_len_term = self.b * (doc_len / self._avgdl) if self._avgdl > 0 else 0.0
        score = 0.0
        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue  # out-of-vocabulary query term contributes nothing
            f = doc_tf.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + denom_len_term)
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, k: int) -> list:
        if not self.passages:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens or not any(t in self._idf for t in query_tokens):
            return []

        scored = [(i, self._score(query_tokens, i)) for i in range(self._n_docs)]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:k]
        return [Passage(id=i, text=self.passages[i], score=float(s)) for i, s in top]
