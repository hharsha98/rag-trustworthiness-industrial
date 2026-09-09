"""Dense retrieval over an in-memory FAISS index.

`Retriever` uses L2-normalised embeddings with `faiss.IndexFlatIP`, so the
index's ranking geometry matches cosine scoring used everywhere else in this
package.
"""
from dataclasses import dataclass

import numpy as np


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
        import faiss

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
