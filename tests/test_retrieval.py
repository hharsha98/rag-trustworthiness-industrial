import numpy as np
import pytest

import ragtrust.retrieval.index as index_mod
from ragtrust.retrieval.index import Retriever, import_faiss


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


# --- FAISS thread cap -------------------------------------------------------
#
# torch, sklearn and faiss each vendor their own libomp, and with more than one
# in a process FAISS's OpenMP regions segfault once an index is large enough for
# it to spawn threads -- observed at 22,878 vectors, absent at 5,183. The crash
# is a SIGSEGV in a native thread, so there is no exception to assert on and no
# traceback to read; these tests pin the guard itself instead.


@pytest.fixture
def uncapped_faiss(monkeypatch):
    """Reset the once-only cap flag and record omp_set_num_threads calls."""
    faiss = pytest.importorskip("faiss")
    calls: list = []
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append, raising=False)
    monkeypatch.setattr(index_mod, "_FAISS_THREADS_SET", False)
    monkeypatch.delenv("RAGTRUST_FAISS_THREADS", raising=False)
    return faiss, calls


def test_import_faiss_caps_threads_to_one_by_default(uncapped_faiss):
    faiss, calls = uncapped_faiss
    assert import_faiss() is faiss
    assert calls == [1]


def test_import_faiss_honours_thread_override(uncapped_faiss, monkeypatch):
    _, calls = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "4")
    import_faiss()
    assert calls == [4]


def test_import_faiss_caps_once_not_per_call(uncapped_faiss):
    _, calls = uncapped_faiss
    import_faiss()
    import_faiss()
    import_faiss()
    assert calls == [1], "the cap is global to the process; re-applying it per call is waste"


def test_import_faiss_survives_a_build_without_openmp(uncapped_faiss, monkeypatch):
    """A faiss compiled without OpenMP has no omp_set_num_threads at all.

    Nothing to cap means nothing to collide, so the correct behaviour is to
    carry on -- an AttributeError here would stop the service from starting.
    """
    faiss, _ = uncapped_faiss
    monkeypatch.delattr(faiss, "omp_set_num_threads", raising=False)
    assert import_faiss() is faiss


def test_import_faiss_survives_an_unparseable_override(uncapped_faiss, monkeypatch):
    """A typo in the env var must not take the service down with it.

    The cap is a safety guard; failing closed on a malformed value would turn a
    harmless misconfiguration into an outage.
    """
    faiss, calls = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "two")
    assert import_faiss() is faiss
    assert calls == [], "a bad value leaves faiss at its own default rather than guessing"


