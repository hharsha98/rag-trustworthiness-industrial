"""Tests for routing.py::route -- the trust-gated agentic retrieval router
that classifies a query as ROUTE_ABSTAIN/ROUTE_SINGLE/ROUTE_ITERATIVE using
`max_context_similarity`, BEFORE any generation call. No model downloads, no
network -- uses the fake embedder/NLI from conftest.py.
"""
from ragtrust.config import Config
from ragtrust.generation.base import GeneratedAnswer
from ragtrust.pipeline import RAGTrustPipeline
from ragtrust.routing import ROUTE_ABSTAIN, ROUTE_ITERATIVE, ROUTE_SINGLE, route

PASSAGE_A = "Photosynthesis converts sunlight into chemical energy inside plant leaves."


class CountingGenerator:
    """Records every `generate()` call -- used to assert `route()` never
    spends a generation call in any branch, including ROUTE_ABSTAIN."""

    def __init__(self):
        self.calls = 0

    def generate(self, query, passages):
        self.calls += 1
        return GeneratedAnswer(text="should never be called", citations={})


def _pipeline(**config_kwargs) -> RAGTrustPipeline:
    from ragtrust.metrics.nli import FakeNLI
    from conftest import FakeEmbedder

    cfg = Config(**config_kwargs)
    return RAGTrustPipeline(cfg, nli=FakeNLI(), embedder=FakeEmbedder())


def _pipeline_with_passage() -> RAGTrustPipeline:
    # retrieval_gate=0.3 and route_iterate_margin=0.15 are Config's own
    # defaults, spelled out here so the three query strings below (chosen to
    # land in each band, verified empirically against FakeEmbedder's hashing)
    # remain meaningful even if a default changes elsewhere in this file.
    pipeline = _pipeline(retrieval_gate=0.3, route_iterate_margin=0.15, k=1)
    pipeline.index_texts([PASSAGE_A])
    return pipeline


def test_route_below_gate_is_abstain_and_spends_zero_generation_calls():
    pipeline = _pipeline_with_passage()
    generator = CountingGenerator()
    pipeline._generator = generator

    result, top = route(pipeline, "Something about stock markets and finance entirely unrelated topic.", 0.15)

    assert result == ROUTE_ABSTAIN
    assert top < pipeline.config.retrieval_gate
    assert generator.calls == 0, "route() must never call generate() in any branch"


def test_route_mid_band_is_iterative():
    pipeline = _pipeline_with_passage()
    generator = CountingGenerator()
    pipeline._generator = generator

    # A single word overlapping the passage -- verified empirically to land
    # at similarity ~0.333, between retrieval_gate (0.3) and
    # retrieval_gate + iterate_margin (0.45).
    result, top = route(pipeline, "energy", 0.15)

    assert result == ROUTE_ITERATIVE
    assert pipeline.config.retrieval_gate <= top < pipeline.config.retrieval_gate + 0.15
    assert generator.calls == 0


def test_route_high_similarity_is_single():
    pipeline = _pipeline_with_passage()
    generator = CountingGenerator()
    pipeline._generator = generator

    result, top = route(pipeline, PASSAGE_A, 0.15)

    assert result == ROUTE_SINGLE
    assert top >= pipeline.config.retrieval_gate + 0.15
    assert generator.calls == 0


def test_route_returns_the_similarity_it_scored():
    pipeline = _pipeline_with_passage()
    pipeline._generator = CountingGenerator()

    result, top = route(pipeline, PASSAGE_A, 0.15)

    assert isinstance(top, float)
    assert 0.0 <= top <= 1.0
