"""Tests for Contextual Retrieval chunk preprocessing (ingest/contextualize.py).
No network, no model downloads -- generators here are simple in-process fakes.
"""
import pytest

from ragtrust.ingest.contextualize import (
    CONTEXT_MAX_WORDS,
    _clean_blurb,
    build_context_prompt,
    contextualize_chunks,
)


class FakeCompleteGenerator:
    """Returns a deterministic, distinct blurb per call; records call count."""

    def __init__(self, prefix: str = "BLURB"):
        self.prefix = prefix
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"{self.prefix} {self.calls}."


class NoCompleteGenerator:
    """A generator backend that never implemented complete() -- the common
    case, since most backends only need `generate`."""

    def generate(self, query, passages):
        raise NotImplementedError


class RaisingOnSecondCallGenerator:
    """complete() raises on its 2nd call, succeeds otherwise -- simulates a
    flaky backend failing on one chunk among several."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated backend failure")
        return f"Blurb for call {self.calls}."


def _chunks(n: int) -> list:
    return [{"page": i + 1, "text": f"Chunk number {i} content."} for i in range(n)]


# --------------------------------------------------------------- prompt shape


def test_build_context_prompt_truncates_document_and_instructs_single_sentence():
    doc = "X" * 20000
    prompt = build_context_prompt(doc, "some chunk", max_document_chars=100)

    # Only the first 100 chars of the (huge) document text appear in the prompt.
    assert "X" * 100 in prompt
    assert "X" * 101 not in prompt
    assert "some chunk" in prompt
    assert "ONLY" in prompt  # "Answer with ONLY that sentence" instruction


# ------------------------------------------------------------- ordering/shape


def test_contextualize_chunks_preserves_order_and_length():
    chunks = _chunks(5)
    generator = FakeCompleteGenerator()
    result = contextualize_chunks(chunks, "a document", generator, workers=4)

    assert len(result) == len(chunks)
    for chunk, r in zip(chunks, result):
        assert r["page"] == chunk["page"]
        assert r["source_text"] == chunk["text"]
        assert chunk["text"] in r["text"]
        assert r["text"] != r["source_text"]  # contextualisation did add a blurb


# ------------------------------------------------------------------ degrading


def test_generator_without_complete_passes_through_unchanged():
    chunks = _chunks(3)
    generator = NoCompleteGenerator()

    with pytest.warns(UserWarning):
        result = contextualize_chunks(chunks, "a document", generator)

    assert len(result) == 3
    for chunk, r in zip(chunks, result):
        assert r["text"] == r["source_text"] == chunk["text"]


def test_generator_raising_on_one_chunk_falls_back_only_that_chunk():
    chunks = _chunks(3)
    generator = RaisingOnSecondCallGenerator()

    # workers=1 makes call order deterministic and equal to chunk order, so
    # "raises on the 2nd chunk" (the spec's phrasing) maps onto "raises on the
    # 2nd call" here.
    with pytest.warns(UserWarning):
        result = contextualize_chunks(chunks, "a document", generator, workers=1)

    assert result[0]["text"] != result[0]["source_text"]
    assert result[1]["text"] == result[1]["source_text"], "the failing chunk must fall back, not crash"
    assert result[2]["text"] != result[2]["source_text"], "chunks after the failure must still be contextualised"


# ---------------------------------------------------------------------- cache


def test_cache_round_trip_issues_llm_calls_only_on_first_run(tmp_path):
    chunks = _chunks(4)
    cache_dir = str(tmp_path)

    gen1 = FakeCompleteGenerator()
    result1 = contextualize_chunks(
        chunks, "a document", gen1, cache_dir=cache_dir, model_tag="model-a"
    )
    assert gen1.calls == 4

    gen2 = FakeCompleteGenerator(prefix="SHOULD-NOT-BE-CALLED")
    result2 = contextualize_chunks(
        chunks, "a document", gen2, cache_dir=cache_dir, model_tag="model-a"
    )
    assert gen2.calls == 0, "a cache hit must not call complete() again"
    assert [r["text"] for r in result1] == [r["text"] for r in result2]


def test_different_model_tag_is_a_cache_miss(tmp_path):
    # The cache key is sha256(model_tag + "\x00" + chunk_text), so changing
    # the model tag must not silently reuse another model's blurbs.
    chunks = _chunks(2)
    cache_dir = str(tmp_path)

    gen1 = FakeCompleteGenerator()
    contextualize_chunks(chunks, "a document", gen1, cache_dir=cache_dir, model_tag="model-a")

    gen2 = FakeCompleteGenerator()
    contextualize_chunks(chunks, "a document", gen2, cache_dir=cache_dir, model_tag="model-b")
    assert gen2.calls == 2


# --------------------------------------------------------------- blurb cleanup


def test_blurb_trimmed_to_max_words():
    long_blurb = " ".join(f"word{i}" for i in range(100))
    cleaned = _clean_blurb(long_blurb)
    assert len(cleaned.split()) == CONTEXT_MAX_WORDS


def test_blurb_strips_leading_context_preamble():
    preambled = "Context: This chunk discusses the topic in depth."
    cleaned = _clean_blurb(preambled)
    assert not cleaned.lower().startswith("context:")


def test_empty_blurb_after_cleanup_falls_back_to_source_text():
    class EmptyGenerator:
        def complete(self, prompt: str) -> str:
            return "   "  # blank after strip()

    chunks = _chunks(1)
    result = contextualize_chunks(chunks, "a document", EmptyGenerator())
    assert result[0]["text"] == result[0]["source_text"]
