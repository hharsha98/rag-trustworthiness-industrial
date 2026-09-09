import numpy as np
import pytest

from ragtrust.retrieval.index import Retriever


class DictEmbedder:
    """Fake embedder returning hand-picked vectors, for exact control over
    cosine vs. L2 geometry in tests. No model download."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def encode(self, texts, **kwargs) -> np.ndarray:
        return np.array([self.mapping[t] for t in texts], dtype="float32")


def test_normalized_retriever_ranking_matches_cosine_ordering():
    passages = ["a", "b", "c"]
    mapping = {
        "query": [1.0, 0.0],
        "a": [1.0, 0.0],   # cos = 1.0
        "b": [0.0, 1.0],   # cos = 0.0
        "c": [0.7, 0.7],   # cos ~ 0.707
    }
    embedder = DictEmbedder(mapping)
    retriever = Retriever(embedder, normalize=True).build(passages)

    results = retriever.search("query", k=3)
    assert [p.text for p in results] == ["a", "c", "b"]

    for p in results:
        v = np.array(mapping[p.text])
        q = np.array(mapping["query"])
        expected_cos = float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q)))
        assert p.score == pytest.approx(expected_cos, abs=1e-5)


