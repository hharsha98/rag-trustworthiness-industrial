"""Tests for the pipeline surface used by the CLI and service: indexing,
persistence, and the two abstention gates. No model downloads, no network --
uses the fake embedder/NLI from conftest.py and a stub generator defined here.
"""
import json
from pathlib import Path

import pytest

from ragtrust.config import Config
from ragtrust.generation.base import GeneratedAnswer
from ragtrust.pipeline import RAGTrustPipeline

ROOT = Path(__file__).resolve().parents[1]
DEMO_CORPUS = ROOT / "data" / "demo_corpus.md"


class StubGenerator:
    """Records how many times it was called, so tests can assert a gate
    fired before generation was ever attempted."""

    def __init__(self, text: str = "A stub answer.", citations: dict = None):
        self.text = text
        self.citations = citations or {}
        self.calls = 0
        # The passages `answer()` actually handed to `generate()` on the most
        # recent call -- used by the contextual-retrieval invariant test to
        # confirm the generator never sees an LLM-written blurb.
        self.last_passages = None

    def generate(self, query, passages):
        self.calls += 1
        self.last_passages = passages
        return GeneratedAnswer(text=self.text, citations=self.citations)


def _pipeline(**config_kwargs) -> RAGTrustPipeline:
    from ragtrust.metrics.nli import FakeNLI
    from conftest import FakeEmbedder

    cfg = Config(**config_kwargs)
    return RAGTrustPipeline(cfg, nli=FakeNLI(), embedder=FakeEmbedder())


# --------------------------------------------------------------------- indexing


def test_index_corpus_on_demo_corpus_yields_substantive_passages():
    pipeline = _pipeline()
    pipeline.index_corpus(str(DEMO_CORPUS))

    assert len(pipeline.passages_text) > 0
    mean_words = sum(len(t.split()) for t in pipeline.passages_text) / len(pipeline.passages_text)
    assert mean_words > 30, f"mean words per passage was only {mean_words}"


def test_index_dir_picks_up_multiple_files_with_distinct_sources(tmp_path):
    (tmp_path / "a.md").write_text(
        "## Topic A\n\n" + ("Sentence about topic A with enough content. " * 6) + "\n"
    )
    (tmp_path / "b.md").write_text(
        "## Topic B\n\n" + ("Sentence about topic B with enough content. " * 6) + "\n"
    )

    pipeline = _pipeline()
    pipeline.index_dir(str(tmp_path))

    sources = {meta["source"] for meta in pipeline.passage_meta.values()}
    assert len(sources) >= 2
    assert sources == {"a.md", "b.md"}


def test_install_raises_on_empty_input():
    pipeline = _pipeline()
    with pytest.raises(ValueError):
        pipeline.index_texts([])


# ------------------------------------------------------------------ persistence


def test_save_and_load_round_trips_passages_and_metadata(tmp_path):
    pipeline = _pipeline(embed_model="fake-embed-v1")
    texts = ["Passage one about robotics.", "Passage two about neural networks."]
    meta = [{"source": "doc.md", "page": 1}, {"source": "doc.md", "page": 2}]
    pipeline.index_texts(texts, meta)
    pipeline.save(str(tmp_path))

    assert (tmp_path / "passages.json").exists()
    assert (tmp_path / "index.faiss").exists()
    stored = json.loads((tmp_path / "passages.json").read_text())
    assert stored["embed_model"] == "fake-embed-v1"

    reloaded = _pipeline(embed_model="fake-embed-v1")
    reloaded.load(str(tmp_path))

    assert reloaded.passages_text == texts
    assert reloaded.passage_meta == {0: meta[0], 1: meta[1]}


def test_load_raises_valueerror_on_embed_model_mismatch(tmp_path):
    writer = _pipeline(embed_model="fake-embed-a")
    writer.index_texts(["some passage text here"])
    writer.save(str(tmp_path))

    reader = _pipeline(embed_model="fake-embed-b")
    with pytest.raises(ValueError, match="embed_model"):
        reader.load(str(tmp_path))


# ------------------------------------------------------------------------ gates


def test_retrieval_gate_fires_before_generation():
    # retrieval_gate above the maximum possible cosine similarity (1.0) means
    # the gate always fires, regardless of corpus/query content.
    generator = StubGenerator()
    pipeline = _pipeline(retrieval_gate=1.5)
    pipeline._generator = generator
    pipeline.index_texts(["Some passage about photosynthesis in plants."])

    result = pipeline.answer("What is photosynthesis?")

    assert result.abstained is True
    assert "relevance gate" in result.abstain_reason
    assert generator.calls == 0, "generator must not be called once the retrieval gate fires"


def test_grounding_gate_fires_when_retrieval_passes_but_nothing_supports_answer():
    # retrieval_gate below the minimum possible cosine similarity (-1.0) means
    # the pre-generation gate never fires, so generation always runs.
    generator = StubGenerator(text="The quantum flux capacitor stabilizes warp fields.")
    pipeline = _pipeline(retrieval_gate=-2.0, abstain_threshold=0.5)
    pipeline._generator = generator
    pipeline.index_texts(["Photosynthesis converts sunlight into chemical energy in plants."])

    result = pipeline.answer("How does photosynthesis work?")

    assert generator.calls == 1, "generation must run once the retrieval gate is cleared"
    assert result.abstained is True
    assert "support" in result.abstain_reason.lower()


def test_answer_succeeds_when_grounded():
    generator = StubGenerator(text="Photosynthesis converts sunlight into chemical energy.")
    pipeline = _pipeline(retrieval_gate=-2.0, abstain_threshold=0.1)
    pipeline._generator = generator
    pipeline.index_texts(["Photosynthesis converts sunlight into chemical energy in plants."])

    result = pipeline.answer("How does photosynthesis work?")

    assert result.abstained is False
    assert result.abstain_reason is None


# ------------------------------------------------------------------ result shape


def test_to_dict_is_json_serialisable_for_answered_and_abstained_results():
    grounded_generator = StubGenerator(text="Photosynthesis converts sunlight into chemical energy.")
    grounded = _pipeline(retrieval_gate=-2.0, abstain_threshold=0.1)
    grounded._generator = grounded_generator
    grounded.index_texts(["Photosynthesis converts sunlight into chemical energy in plants."])
    answered_result = grounded.answer("How does photosynthesis work?")
    json.dumps(answered_result.to_dict())  # must not raise

    declined = _pipeline(retrieval_gate=1.5)
    declined._generator = StubGenerator()
    declined.index_texts(["Some unrelated passage."])
    abstained_result = declined.answer("Anything?")
    json.dumps(abstained_result.to_dict())  # must not raise


def test_is_trustworthy_is_false_whenever_abstained():
    declined = _pipeline(retrieval_gate=1.5)
    declined._generator = StubGenerator()
    declined.index_texts(["Some unrelated passage."])
    result = declined.answer("Anything?")

    assert result.abstained is True
    assert result.is_trustworthy is False


# ------------------------------------------------------ contextual retrieval


class RecordingFakeNLI:
    """Wraps FakeNLI's scoring but records every premise it is ever asked to
    score, so a test can assert a forbidden string never reaches NLI."""

    def __init__(self):
        from ragtrust.metrics.nli import FakeNLI

        self._inner = FakeNLI()
        self.premises_seen: list = []

    def probs(self, premise, hypothesis):
        self.premises_seen.append(premise)
        return self._inner.probs(premise, hypothesis)

    def batch_probs(self, pairs):
        self.premises_seen.extend(p for p, _ in pairs)
        return self._inner.batch_probs(pairs)


def test_contextual_blurb_never_reaches_nli_generator_or_output():
    """THE invariant this feature exists to protect (see pipeline.py::answer()'s
    `replace(p, text=self.source_text(p.id))` line and ingest/contextualize.py's
    module docstring): an LLM-generated blurb used to improve retrieval must
    never reach the NLI premise, the generator, citations, or anything in the
    JSON result. If it did, a claim could be scored "faithful" because it is
    entailed by text the model invented rather than by the corpus.
    """
    sentinel = "ZZSENTINEL topic overview."
    source = "Photosynthesis converts sunlight into chemical energy in plants."
    contextualized_text = f"{sentinel} {source}"

    nli = RecordingFakeNLI()
    generator = StubGenerator(text="Photosynthesis converts sunlight into chemical energy.")
    pipeline = _pipeline(retrieval_gate=-2.0, abstain_threshold=0.1)
    pipeline._nli = nli
    pipeline._generator = generator
    # Installed directly (bypassing contextualize_chunks) so the test controls
    # the exact contextualised/source pair rather than depending on an LLM.
    pipeline._install(
        [contextualized_text], [{"source": "doc.md", "page": 1}], [source]
    )

    assert sentinel in pipeline.passages_text[0]
    assert sentinel not in pipeline.passage_source_text[0]

    result = pipeline.answer("How does photosynthesis work?")

    assert result.abstained is False

    for premise in nli.premises_seen:
        assert sentinel not in premise, "sentinel blurb reached an NLI premise"

    assert generator.last_passages is not None
    for p in generator.last_passages:
        assert sentinel not in getattr(p, "text", p), "sentinel blurb reached the generator"

    for p in result.passages:
        assert sentinel not in getattr(p, "text", p), "sentinel blurb reached result.passages"

    dumped = json.dumps(result.to_dict())
    assert sentinel not in dumped, "sentinel blurb reached to_dict() output"


def test_source_text_falls_back_to_passages_text_when_out_of_range():
    pipeline = _pipeline()
    pipeline.index_texts(["Some passage about robotics."])
    # passage_source_text has length 1; index 5 is out of range for it, so
    # source_text() must fall back to passages_text rather than raising.
    assert pipeline.source_text(0) == "Some passage about robotics."


def test_save_and_load_round_trips_passage_source_text(tmp_path):
    pipeline = _pipeline(embed_model="fake-embed-v1")
    texts = ["ZZBLURB one. Passage one about robotics.", "ZZBLURB two. Passage two about nets."]
    source_texts = ["Passage one about robotics.", "Passage two about nets."]
    meta = [{"source": "doc.md", "page": 1}, {"source": "doc.md", "page": 2}]
    pipeline._install(texts, meta, source_texts)
    pipeline.save(str(tmp_path))

    stored = json.loads((tmp_path / "passages.json").read_text())
    assert stored["source_passages"] == source_texts

    reloaded = _pipeline(embed_model="fake-embed-v1")
    reloaded.load(str(tmp_path))

    assert reloaded.passages_text == texts
    assert reloaded.passage_source_text == source_texts


def test_load_without_source_passages_key_is_backward_compatible(tmp_path):
    # Simulates an index persisted before this feature existed: passages.json
    # has no "source_passages" key at all.
    pipeline = _pipeline(embed_model="fake-embed-v1")
    texts = ["Passage one about robotics.", "Passage two about neural networks."]
    pipeline.index_texts(texts)
    pipeline.save(str(tmp_path))

    stored = json.loads((tmp_path / "passages.json").read_text())
    del stored["source_passages"]
    (tmp_path / "passages.json").write_text(json.dumps(stored))

    reloaded = _pipeline(embed_model="fake-embed-v1")
    reloaded.load(str(tmp_path))

    assert reloaded.passage_source_text == reloaded.passages_text == texts


def test_config_contextual_defaults_off_and_index_corpus_is_unchanged():
    # Config().contextual is False, so index_corpus must produce byte-for-byte
    # the same passages as before this feature existed -- no generator call,
    # no blurb, nothing.
    plain = _pipeline()
    plain.index_corpus(str(DEMO_CORPUS))

    assert plain.config.contextual is False
    assert plain.passage_source_text == plain.passages_text
