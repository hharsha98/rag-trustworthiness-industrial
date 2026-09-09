"""Tests for BM25 sparse retrieval, RRF hybrid fusion, and the Config/pipeline
wiring that selects between them. Fast, no model downloads, no network.

Anything needing the real cross-encoder is marked `@pytest.mark.slow` (skipped by
default -- see pyproject.toml's `addopts = "-m \"not slow\""`).
"""
import pytest

from conftest import FakeEmbedder

from ragtrust.config import Config
from ragtrust.pipeline import RAGTrustPipeline
from ragtrust.retrieval.hybrid import HybridRetriever
from ragtrust.retrieval.index import Passage, Retriever
from ragtrust.retrieval.sparse import BM25Retriever


# --------------------------------------------------------------------------- BM25


def test_bm25_ranks_exact_term_match_above_unrelated_document():
    passages = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum entanglement describes correlated particle states.",
    ]
    retriever = BM25Retriever().build(passages)
    results = retriever.search("fox jumps", k=2)
    assert results
    assert results[0].text == passages[0]


def test_bm25_out_of_vocabulary_query_returns_empty_without_raising():
    retriever = BM25Retriever().build(["alpha beta gamma", "delta epsilon zeta"])
    results = retriever.search("nonexistent term zzzqqq", k=5)
    assert results == []


def test_bm25_empty_corpus_returns_empty_without_raising():
    retriever = BM25Retriever().build([])
    results = retriever.search("anything", k=5)
    assert results == []


def test_bm25_single_document_corpus_does_not_raise():
    retriever = BM25Retriever().build(["only one document here"])
    results = retriever.search("document", k=1)
    assert len(results) == 1
    assert results[0].id == 0


# --------------------------------------------------------------------------- RRF


class _StubRetriever:
    """Fixed rankings, for exact arithmetic control over RRF fusion."""

    def __init__(self, ranking: list):
        self._ranking = ranking

    def build(self, passages):
        return self

    def search(self, query, k):
        return self._ranking[:k]


def test_rrf_fusion_produces_arithmetically_expected_order():
    # dense ranks: A(1), B(2), C(3)
    dense = _StubRetriever([
        Passage(id=0, text="A", score=0.9),
        Passage(id=1, text="B", score=0.8),
        Passage(id=2, text="C", score=0.7),
    ])
    # sparse ranks: C(1), A(2), B(3)
    sparse = _StubRetriever([
        Passage(id=2, text="C", score=5.0),
        Passage(id=0, text="A", score=4.0),
        Passage(id=1, text="B", score=3.0),
    ])
    rrf_k = 60
    hybrid = HybridRetriever(dense, sparse, rrf_k=rrf_k)

    results = hybrid.search("query", k=3, candidates=3)

    expected = {
        0: 1 / (rrf_k + 1) + 1 / (rrf_k + 2),  # A: dense rank1 + sparse rank2
        1: 1 / (rrf_k + 2) + 1 / (rrf_k + 3),  # B: dense rank2 + sparse rank3
        2: 1 / (rrf_k + 3) + 1 / (rrf_k + 1),  # C: dense rank3 + sparse rank1
    }
    expected_order = sorted(expected, key=expected.get, reverse=True)

    assert [p.id for p in results] == expected_order
    for p in results:
        assert p.score == pytest.approx(expected[p.id], abs=1e-9)


def test_hybrid_returns_exactly_k_results_when_corpus_larger_than_k():
    passages = [f"document number {i} about topic {i % 3}" for i in range(30)]
    hybrid = HybridRetriever(
        Retriever(FakeEmbedder(), normalize=True), BM25Retriever(),
    ).build(passages)
    results = hybrid.search("topic document", k=5)
    assert len(results) == 5


# --------------------------------------------------------------------------- Config


def test_config_rejects_invalid_retrieval_mode():
    with pytest.raises(ValueError):
        Config(retrieval_mode="not_a_real_mode")


def test_config_default_yields_hybrid_retriever_without_reranking():
    """The default is hybrid, not dense.

    That default was set by measurement, not preference: on BEIR/SciFact every
    alternative beat plain dense retrieval at every k, each improvement surviving a 95%
    paired bootstrap CI (experiments/08_beir_ablation.py). Reranking wins by more still,
    but stays opt-in because it roughly doubles query latency and its unbounded scores
    are not comparable with `retrieval_gate`.
    """
    cfg = Config()
    assert cfg.retrieval_mode == "hybrid"
    assert cfg.rerank is False

    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    retriever = pipeline.retriever
    assert isinstance(retriever, HybridRetriever)
    assert type(retriever.dense) is Retriever
    assert retriever.dense.normalize is True
    assert isinstance(retriever.sparse, BM25Retriever)


def test_explicit_dense_mode_still_yields_a_plain_retriever():
    cfg = Config(retrieval_mode="dense")
    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    assert type(pipeline.retriever) is Retriever
    assert pipeline.retriever.normalize is True


def test_config_sparse_mode_yields_bm25_retriever():
    cfg = Config(retrieval_mode="sparse")
    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    assert isinstance(pipeline.retriever, BM25Retriever)


def test_config_hybrid_mode_yields_hybrid_retriever():
    cfg = Config(retrieval_mode="hybrid")
    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    assert isinstance(pipeline.retriever, HybridRetriever)


# ------------------------------------------------------- abstention gate scale-stability


def test_retrieval_gate_is_scale_stable_across_retrieval_modes():
    """The pre-generation gate must not depend on the retriever's score scale.

    Regression test for a real bug. The gate originally thresholded `Passage.score`,
    which is cosine for dense retrieval but a reciprocal-rank sum for RRF -- bounded by
    about 2/(rrf_k+1), roughly 0.03. Switching the default to hybrid therefore put every
    score below the 0.25 gate and the pipeline abstained on *every* question, including
    ones it had just retrieved good passages for. The gate now measures cosine
    similarity directly, so the same threshold means the same thing in every mode.
    """
    from ragtrust.metrics.relevance import max_context_similarity

    embedder = FakeEmbedder()
    passages = [
        "Max-pooling downsamples a feature map by keeping the maximum activation.",
        "Sensor fusion combines measurements from several sensors.",
        "A Markov Decision Process defines states, actions and rewards.",
    ]
    on_topic = max_context_similarity(
        "What does max-pooling do to a feature map?", passages, embedder)
    off_topic = max_context_similarity(
        "Who painted the Mona Lisa?", passages, embedder)

    # Bounded, and ordered the way the gate depends on.
    for value in (on_topic, off_topic):
        assert 0.0 <= value <= 1.0
    assert on_topic > off_topic

    # Independent of how ranking scored anything: no retriever is involved at all.
    assert max_context_similarity("anything", [], embedder) == 0.0


def test_gate_threshold_is_not_comparable_to_rrf_scores():
    """Documents *why* the gate cannot read RRF scores: they are structurally tiny.

    An RRF score is a sum of 1/(rrf_k + rank) over the rankings a document appears in,
    so with the default rrf_k=60 no document can exceed ~0.033 -- an order of magnitude
    below the 0.25 default gate.
    """
    rrf_k = 60
    best_possible_rrf = 2 * (1.0 / (rrf_k + 1))  # top rank in both dense and sparse
    assert best_possible_rrf < Config().retrieval_gate


# --------------------------------------------------------------------------- rerank (slow)


@pytest.mark.slow
def test_rerank_wraps_base_retriever_with_real_cross_encoder():
    from ragtrust.retrieval.rerank import CrossEncoderReranker

    cfg = Config(rerank=True)
    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    assert isinstance(pipeline.retriever, CrossEncoderReranker)
