"""Tests for the answer_relevance (R_ans) metric wiring -- METRICS.md Part II.3.

R_ans is computed by back-generation: ask the generator for N questions the answer
would answer (Generator.generate_questions, an OPTIONAL protocol method -- see
generation/base.py), embed them, and take the mean cosine against the original
query's embedding (metrics/relevance.py::answer_relevance, already tested
elsewhere -- this file tests the wiring that produces `generated_questions` and
plugs the score into the pipeline).

Fast/offline: no model downloads, no network -- uses the fake embedder/NLI from
conftest.py (no package __init__.py in tests/, so `from conftest import
FakeEmbedder`, same pattern as test_pipeline.py and test_metric_properties.py) and
stub generators defined here.
"""
import json

import pytest
import requests
from hypothesis import given
from hypothesis import strategies as st

from conftest import FakeEmbedder

from ragtrust.config import Config
from ragtrust.generation.base import GeneratedAnswer, GenerationError
from ragtrust.generation.ollama import parse_questions
from ragtrust.metrics.aggregate import aggregate_arithmetic, aggregate_geometric
from ragtrust.metrics.nli import FakeNLI
from ragtrust.pipeline import RAGTrustPipeline

QUERY = "How does photosynthesis work?"
CORPUS_TEXT = "Photosynthesis converts sunlight into chemical energy in plants."
ANSWER_TEXT = "Photosynthesis converts sunlight into chemical energy."


class GeneratorWithQuestions:
    """Stub generator that supports back-generation."""

    def __init__(self, text=ANSWER_TEXT, questions=None):
        self.text = text
        self.questions = (
            ["How does photosynthesis work?", "What does photosynthesis produce?"]
            if questions is None else questions
        )
        self.calls = 0
        self.question_calls = []

    def generate(self, query, passages):
        self.calls += 1
        return GeneratedAnswer(text=self.text, citations={})

    def generate_questions(self, answer, n):
        self.question_calls.append((answer, n))
        return list(self.questions)[:n]


class GeneratorWithoutQuestions:
    """Stub generator with no `generate_questions` -- the common case: most
    backends have no need for one, per Generator's OPTIONAL-method contract."""

    def __init__(self, text=ANSWER_TEXT):
        self.text = text
        self.calls = 0

    def generate(self, query, passages):
        self.calls += 1
        return GeneratedAnswer(text=self.text, citations={})


class GeneratorQuestionsFail:
    """Stub generator whose back-generation call always raises."""

    def __init__(self, text=ANSWER_TEXT, exc=None):
        self.text = text
        self.exc = exc or GenerationError("Ollama backend unreachable at http://fake")
        self.calls = 0

    def generate(self, query, passages):
        self.calls += 1
        return GeneratedAnswer(text=self.text, citations={})

    def generate_questions(self, answer, n):
        raise self.exc


def _pipeline(**config_kwargs) -> RAGTrustPipeline:
    # retrieval_gate/abstain_threshold set wide open, matching test_pipeline.py's
    # "grounded answer" tests, so the answer path (not either abstention gate) is
    # what's under test here.
    cfg = Config(retrieval_gate=-2.0, abstain_threshold=0.1, **config_kwargs)
    return RAGTrustPipeline(cfg, nli=FakeNLI(), embedder=FakeEmbedder())


def _indexed(pipeline: RAGTrustPipeline) -> RAGTrustPipeline:
    pipeline.index_texts([CORPUS_TEXT])
    return pipeline


# --------------------------------------------------------------------- pipeline


def test_answer_relevance_is_a_float_in_0_1_when_enabled_and_supported():
    pipeline = _pipeline(answer_relevance=True)
    generator = GeneratorWithQuestions()
    pipeline._generator = generator
    _indexed(pipeline)

    result = pipeline.answer(QUERY)

    assert result.abstained is False
    score = result.metrics["answer_relevance"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Default answer_relevance_n_questions is 3; the generator's own answer text
    # is what gets back-generated from.
    assert generator.question_calls == [(ANSWER_TEXT, 3)]


def test_answer_relevance_none_when_backend_lacks_generate_questions():
    pipeline = _pipeline(answer_relevance=True)
    generator = GeneratorWithoutQuestions()
    pipeline._generator = generator
    _indexed(pipeline)

    result = pipeline.answer(QUERY)

    assert result.abstained is False
    assert result.metrics["answer_relevance"] is None


def test_answer_relevance_none_when_flag_off_even_if_backend_supports_it():
    pipeline = _pipeline(answer_relevance=False)
    generator = GeneratorWithQuestions()
    pipeline._generator = generator
    _indexed(pipeline)

    result = pipeline.answer(QUERY)

    assert result.abstained is False
    assert result.metrics["answer_relevance"] is None
    assert generator.question_calls == [], (
        "generate_questions must not even be called when Config.answer_relevance is off"
    )


def test_answer_relevance_degrades_to_none_on_generation_error_and_warns():
    pipeline = _pipeline(answer_relevance=True)
    generator = GeneratorQuestionsFail()
    pipeline._generator = generator
    _indexed(pipeline)

    with pytest.warns(UserWarning, match="answer_relevance"):
        result = pipeline.answer(QUERY)

    assert result.abstained is False, "a failed back-generation call must never fail the answer"
    assert result.metrics["answer_relevance"] is None


def test_answer_relevance_none_when_back_generation_returns_no_questions():
    pipeline = _pipeline(answer_relevance=True)
    generator = GeneratorWithQuestions(questions=[])
    pipeline._generator = generator
    _indexed(pipeline)

    result = pipeline.answer(QUERY)

    assert result.abstained is False
    assert result.metrics["answer_relevance"] is None


# ---------------------------------------------------------------- config

def test_default_weights_are_unchanged_by_the_answer_relevance_addition():
    """Guards against silently changing everyone's trust scores: the default
    `weights` dict must keep exactly its original four keys/values."""
    cfg = Config()
    assert cfg.weights == {
        "faithfulness": 0.4,
        "attribution": 0.2,
        "relevance": 0.2,
        "conciseness": 0.2,
    }
    assert "answer_relevance" not in cfg.weights


def test_answer_relevance_config_defaults():
    cfg = Config()
    assert cfg.answer_relevance is False
    assert cfg.answer_relevance_n_questions == 3


def test_config_rejects_non_positive_answer_relevance_n_questions():
    with pytest.raises(ValueError, match="answer_relevance_n_questions"):
        Config(answer_relevance_n_questions=0)
    with pytest.raises(ValueError, match="answer_relevance_n_questions"):
        Config(answer_relevance_n_questions=-1)


def test_flag_off_produces_byte_identical_trust_to_pre_change_baseline():
    """Regression pin: these exact numbers were captured from the pipeline
    before answer_relevance existed (StubGenerator, same corpus/query/config as
    test_pipeline.py::test_answer_succeeds_when_grounded). The flag defaults to
    off, so default behaviour -- and therefore every existing trust score --
    must not move by even a rounding bit."""
    from test_pipeline import StubGenerator

    generator = StubGenerator(text="Photosynthesis converts sunlight into chemical energy.")
    pipeline = _pipeline()
    pipeline._generator = generator
    pipeline.index_texts(["Photosynthesis converts sunlight into chemical energy in plants."])

    result = pipeline.answer("How does photosynthesis work?")

    assert result.abstained is False
    assert result.metrics["answer_relevance"] is None
    assert result.metrics["faithfulness"] == pytest.approx(0.75, abs=1e-12)
    assert result.metrics["attribution"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["relevance"] == pytest.approx(0.17677669529663687, abs=1e-12)
    assert result.metrics["conciseness"] is None
    assert result.trust["arithmetic"] == pytest.approx(0.41919417382415924, abs=1e-12)
    assert result.trust["geometric"] == pytest.approx(0.0, abs=1e-12)
    assert result.trust["weights"] == {
        "faithfulness": 0.4, "attribution": 0.2, "relevance": 0.2, "conciseness": 0.2,
    }


# ---------------------------------------------------------- weights participation


def test_answer_relevance_participates_in_trust_weights_when_added_by_the_user():
    """Because aggregate_arithmetic/aggregate_geometric iterate over the weight
    keys present in `metrics`, a user who wants answer_relevance weighted just
    adds the key to `weights` -- no pipeline code change needed."""
    custom_weights = {
        "faithfulness": 0.25, "attribution": 0.25, "relevance": 0.25, "answer_relevance": 0.25,
    }
    pipeline = _pipeline(answer_relevance=True, weights=custom_weights)
    generator = GeneratorWithQuestions()
    pipeline._generator = generator
    _indexed(pipeline)

    result = pipeline.answer(QUERY)

    assert result.abstained is False
    assert result.trust["weights"] == custom_weights
    assert result.metrics["answer_relevance"] is not None
    assert "answer_relevance" in result.trust["weights"]
    # answer_relevance actually took part: dropping it from the weight basis
    # used by the aggregate must change the result the pipeline reported.
    scored = {k: result.metrics[k] for k in custom_weights if result.metrics.get(k) is not None}
    without_ans_rel = {k: v for k, v in scored.items() if k != "answer_relevance"}
    weights_without = {k: v for k, v in custom_weights.items() if k != "answer_relevance"}
    arithmetic_without = aggregate_arithmetic(without_ans_rel, weights_without)
    assert result.trust["arithmetic"] != pytest.approx(arithmetic_without)


@given(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)
def test_prop3_still_holds_with_five_weighted_dimensions(a, b, c, d, e):
    """Proposition 3 (T_geom <= T_arith, weighted AM-GM) is proved for any
    weights summing to 1 -- adding a fifth dimension (answer_relevance) must not
    break it."""
    weights = {
        "faithfulness": 0.3, "attribution": 0.2, "relevance": 0.2,
        "conciseness": 0.15, "answer_relevance": 0.15,
    }
    metrics = {
        "faithfulness": a, "attribution": b, "relevance": c,
        "conciseness": d, "answer_relevance": e,
    }
    geometric = aggregate_geometric(metrics, weights)
    arithmetic = aggregate_arithmetic(metrics, weights)
    assert 0.0 <= geometric <= 1.0 + 1e-9
    assert geometric <= arithmetic + 1e-9


# --------------------------------------------------------- question parsing


def test_parse_questions_strips_numbered_lines():
    text = "1. What is X?\n2. What is Y?\n3) What is Z?"
    assert parse_questions(text, 10) == ["What is X?", "What is Y?", "What is Z?"]


def test_parse_questions_strips_bulleted_lines():
    text = "- What is X?\n* What is Y?\n• What is Z?"
    assert parse_questions(text, 10) == ["What is X?", "What is Y?", "What is Z?"]


def test_parse_questions_drops_blank_lines():
    text = "What is X?\n\n   \nWhat is Y?\n"
    assert parse_questions(text, 10) == ["What is X?", "What is Y?"]


def test_parse_questions_caps_at_n_even_when_more_are_returned():
    text = "\n".join(f"Question {i}?" for i in range(10))
    assert parse_questions(text, 3) == ["Question 0?", "Question 1?", "Question 2?"]


def test_parse_questions_empty_text_yields_empty_list():
    assert parse_questions("", 5) == []
    assert parse_questions("   \n\n  ", 5) == []


# ------------------------------------------------------ backend implementations


class _FakeResponse:
    """Minimal stand-in for requests.Response, monkeypatched in for
    requests.post so these backend tests stay offline."""

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


def test_ollama_generator_generate_questions_parses_response(monkeypatch):
    from ragtrust.generation.ollama import OllamaGenerator

    monkeypatch.setattr(
        "ragtrust.generation.ollama.requests.post",
        lambda *a, **kw: _FakeResponse({"response": "1. Q1?\n2. Q2?\n"}),
    )
    result = OllamaGenerator().generate_questions("some answer", 5)
    assert result == ["Q1?", "Q2?"]


def test_ollama_generator_generate_questions_wraps_request_exception():
    from ragtrust.generation.ollama import OllamaGenerator

    def _raise(*a, **kw):
        raise requests.ConnectionError("boom")

    import ragtrust.generation.ollama as ollama_module

    original_post = ollama_module.requests.post
    ollama_module.requests.post = _raise
    try:
        with pytest.raises(GenerationError):
            OllamaGenerator().generate_questions("some answer", 3)
    finally:
        ollama_module.requests.post = original_post


def test_hf_api_generator_generate_questions_parses_response(monkeypatch):
    from ragtrust.generation.hf_api import HFAPIGenerator

    monkeypatch.setenv("HF_TOKEN", "fake-token-for-tests")
    monkeypatch.setattr(
        "ragtrust.generation.hf_api.requests.post",
        lambda *a, **kw: _FakeResponse([{"generated_text": "- Q1?\n- Q2?\n"}]),
    )
    result = HFAPIGenerator().generate_questions("some answer", 5)
    assert result == ["Q1?", "Q2?"]


def test_hf_api_generator_generate_questions_requires_token(monkeypatch):
    from ragtrust.generation.hf_api import HFAPIGenerator

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(GenerationError, match="HF_TOKEN"):
        HFAPIGenerator().generate_questions("some answer", 3)


def test_cached_generator_serves_recorded_generated_questions(tmp_path):
    from ragtrust.generation.cached import CachedGenerator

    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({
        "Some question?": {
            "answer": "A cached answer.",
            "generated_questions": ["Q1?", "Q2?", "Q3?", "Q4?"],
        },
    }))
    gen = CachedGenerator(cache_file)

    assert gen.generate_questions("A cached answer.", 2) == ["Q1?", "Q2?"]


def test_cached_generator_raises_when_no_generated_questions_recorded(tmp_path):
    """The common case today: no entry carries "generated_questions", so
    CachedGenerator raises rather than fabricating -- see its module docstring.
    The pipeline turns that into answer_relevance = None."""
    from ragtrust.generation.cached import CachedGenerator

    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({"Some question?": "A cached answer."}))
    gen = CachedGenerator(cache_file)

    with pytest.raises(GenerationError):
        gen.generate_questions("A cached answer.", 2)
