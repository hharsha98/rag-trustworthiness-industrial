"""Shared, dependency-free test fixtures: no model downloads, no network."""
import os

# macOS-specific fix: PyTorch and FAISS each bundle their own OpenMP runtime.
# Importing both in the same process aborts the interpreter ("Fatal Python
# error: Aborted") unless this is set before either is imported. This must
# stay at the very top of the first module pytest loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib

import numpy as np
import pytest

from ragtrust.metrics.nli import FakeNLI


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download; same
    text always maps to the same vector, across processes and runs."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in text.lower().split():
            idx = _stable_hash(token) % self.dim
            vec[idx] += 1.0
        if not np.any(vec):
            rng = np.random.default_rng(_stable_hash(text) % (2**32))
            vec = rng.normal(size=self.dim)
        return vec

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._vector(t) for t in texts])


@pytest.fixture
def fake_nli():
    return FakeNLI()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
