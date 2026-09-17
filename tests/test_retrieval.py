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


# --- FAISS import choke point -----------------------------------------------
#
# torch, sklearn and faiss each vendor their own libomp, and with more than one
# in a process FAISS's OpenMP regions segfault once an index is large enough for
# it to spawn threads -- observed at 22,878 vectors, absent at 5,183. That crash
# is a SIGSEGV in a native thread: no exception to assert on, no traceback.
#
# The deployed fix is RAGTRUST_FAISS_THREADS, set in deploy/.env.example, which
# caps faiss alone. OMP_NUM_THREADS was tried first and withdrawn: it caps every
# OpenMP consumer including PyTorch, which runs the NLI that dominates this
# pipeline, and throttled production to one core on an 8-core host.
# What IS testable is that import_faiss does NOT call omp_set_num_threads on its
# own -- because that call forces faiss's libomp to initialise, and whichever
# library brings up its copy second then dies with `OMP: Error #15` and SIGABRT.
# Measured: `create_app()` aborted at startup in both import orders, surviving
# only when scikit-learn happened to be imported first. These tests pin the
# absence of that call, which is the actual safety property.


@pytest.fixture
def uncapped_faiss(monkeypatch):
    """Reset the once-only flag and record any omp_set_num_threads calls."""
    faiss = pytest.importorskip("faiss")
    calls: list = []
    monkeypatch.setattr(faiss, "omp_set_num_threads", calls.append, raising=False)
    monkeypatch.setattr(index_mod, "_FAISS_THREADS_SET", False)
    monkeypatch.delenv("RAGTRUST_FAISS_THREADS", raising=False)
    return faiss, calls


def test_import_faiss_does_not_touch_openmp_by_default(uncapped_faiss):
    """The regression guard. Calling omp_set_num_threads here aborted startup.

    An earlier revision capped threads unconditionally from this function and
    broke `create_app()` on macOS with SIGABRT -- a 100% failure, strictly worse
    than the size-dependent segfault it was meant to prevent.
    """
    faiss, calls = uncapped_faiss
    assert import_faiss() is faiss
    assert calls == [], "forcing faiss's libomp up early is what causes OMP Error #15"


def test_import_faiss_forces_the_cap_only_when_explicitly_asked(uncapped_faiss, monkeypatch):
    """Opt-in escape hatch, for profiling on a platform where it is known safe."""
    _, calls = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "4")
    import_faiss()
    assert calls == [4]


def test_import_faiss_applies_an_explicit_cap_once_not_per_call(uncapped_faiss, monkeypatch):
    _, calls = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "1")
    import_faiss()
    import_faiss()
    import_faiss()
    assert calls == [1], "the setting is global to the process; re-applying it per call is waste"


def test_import_faiss_survives_a_build_without_openmp(uncapped_faiss, monkeypatch):
    """A faiss compiled without OpenMP has no omp_set_num_threads at all.

    Nothing to cap means nothing to collide, so the correct behaviour is to
    carry on -- an AttributeError here would stop the service from starting.
    """
    faiss, _ = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "1")
    monkeypatch.delattr(faiss, "omp_set_num_threads", raising=False)
    assert import_faiss() is faiss


def test_import_faiss_survives_an_unparseable_override(uncapped_faiss, monkeypatch):
    """A typo in the env var must not take the service down with it.

    Failing closed on a malformed value would turn a harmless misconfiguration
    into an outage.
    """
    faiss, calls = uncapped_faiss
    monkeypatch.setenv("RAGTRUST_FAISS_THREADS", "two")
    assert import_faiss() is faiss
    assert calls == [], "a bad value leaves faiss at its own default rather than guessing"


