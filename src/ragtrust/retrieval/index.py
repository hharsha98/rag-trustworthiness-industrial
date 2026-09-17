"""Dense retrieval over an in-memory FAISS index.

`Retriever` uses L2-normalised embeddings with `faiss.IndexFlatIP`, so the
index's ranking geometry matches cosine scoring used everywhere else in this
package.
"""
import os
from dataclasses import dataclass

import numpy as np

_FAISS_THREADS_SET = False


def import_faiss():
    """Import faiss and return the module -- the one place this package does so.

    torch, sklearn and faiss each vendor their own copy of libomp. With more
    than one loaded into a process, FAISS's OpenMP parallel regions can
    segfault -- SIGSEGV with no Python traceback, because the crash happens in
    a native thread that has no Python frames to print. It is size-dependent,
    since FAISS only spawns those threads once an index is large enough:
    reproducible at 22,878 vectors while absent at 5,183. `POST /corpora`
    indexes an arbitrary uploaded document, so a large enough upload can take
    the whole service down.

    **Cap it with RAGTRUST_FAISS_THREADS**, which deploy/.env.example sets to 1.

    An earlier revision of this docstring said to use OMP_NUM_THREADS instead.
    That was wrong in production, and the correction is worth keeping:
    OMP_NUM_THREADS caps *every* OpenMP consumer in the process, PyTorch
    included, and PyTorch runs the NLI entailment that dominates this pipeline
    -- entail plus score were 79s of a 99s answer on the deployed host. Setting
    it throttled the whole service to one core, on a machine with 8 of them, to
    protect an index operation that was never the bottleneck. Startup alone went
    85s -> 40s on removing it. It does prevent the segfault; it just costs far
    more than it saves.

    Calling `faiss.omp_set_num_threads()` is deliberately NOT the default,
    because it is worse than the problem it solves. That call is what forces
    faiss's libomp to *initialise*; importing faiss alone does not. Once it
    has, the next library to bring up its own copy hits the runtime's
    duplicate check and the process dies with `OMP: Error #15` and SIGABRT.
    Measured on macOS, `create_app()` aborting at startup in both orders:

        import torch; faiss.omp_set_num_threads(1)   -> exit 134
        faiss.omp_set_num_threads(1); import torch   -> exit 134
        import torch, sklearn; import_faiss()        -> exit 0

    The third line survives only because scikit-learn smooths over duplicate
    OpenMP runtimes as a side effect of being imported -- so whether the call
    aborts depends on which unrelated library happened to load first. Trading a
    load-dependent segfault for an import-order-dependent startup abort is not
    a fix. Linux does not trip the check, which is exactly how this reached the
    deployed service before being caught locally.

    So the call stays opt-in: RAGTRUST_FAISS_THREADS forces it, on a platform
    where it is known safe. The deployed host is Linux and sets it; macOS
    development leaves it unset and keeps faiss's own default, where the
    segfault needs an index above ~22k vectors to appear and the benchmarks that
    build one pass OMP_NUM_THREADS=1 on the command line for that single run.
    """
    global _FAISS_THREADS_SET
    import faiss

    requested = os.environ.get("RAGTRUST_FAISS_THREADS", "")
    if requested and not _FAISS_THREADS_SET:
        try:
            faiss.omp_set_num_threads(int(requested))
        except (AttributeError, ValueError):
            # A faiss built without OpenMP exposes no omp_set_num_threads, and a
            # malformed env value must not stop the service from starting. In
            # both cases the library's default thread count stands.
            pass
        _FAISS_THREADS_SET = True
    return faiss


@dataclass
class Passage:
    id: int
    text: str
    score: float


class Retriever:
    def __init__(self, model_name, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self.passages: list = []
        self.index = None

    @property
    def model(self):
        # Lazy-load. An already-constructed embedder may be injected directly --
        # this is how tests supply a fake, dependency-free encoder without
        # downloading real model weights.
        #
        # The string check must come FIRST. A duck-type test like
        # `hasattr(model_name, "encode")` is always true for a `str`, because
        # `str.encode` is the bytes encoder -- so a plain model name would be
        # mistaken for an embedder, and `"msmarco-...".encode(list_of_texts)`
        # would raise a TypeError instead of loading the model.
        if self._model is None:
            if isinstance(self.model_name, str):
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            elif hasattr(self.model_name, "encode"):
                self._model = self.model_name
            else:
                raise TypeError(
                    "model_name must be a model-name string or an object exposing "
                    f".encode(); got {type(self.model_name).__name__}"
                )
        return self._model

    def _encode(self, texts) -> np.ndarray:
        embeddings = np.asarray(self.model.encode(list(texts)), dtype="float32")
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings

    def build(self, passages: list):
        faiss = import_faiss()

        self.passages = list(passages)
        embeddings = self._encode(self.passages)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) if self.normalize else faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self

    def search(self, query: str, k: int) -> list:
        q = self._encode([query])
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(Passage(id=int(idx), text=self.passages[int(idx)], score=float(score)))
        return results
