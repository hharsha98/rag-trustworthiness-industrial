"""Tests for the ragtrust HTTP service, using fastapi.testclient.TestClient.

`RAGTrustPipeline.embedder`/`.nli` are monkeypatched to the fake,
dependency-free implementations from conftest.py (same technique as
test_cli.py) so building the app and answering never downloads a model or
touches the network.
"""
import pytest
from fastapi.testclient import TestClient

from ragtrust.config import Config
from ragtrust.generation.base import GeneratedAnswer, GenerationError
from ragtrust.metrics.nli import FakeNLI
from ragtrust.pipeline import RAGTrustPipeline
from ragtrust.service import create_app
# See the note in test_cli.py: `tests/` is not a package, so conftest is imported by
# its bare module name rather than as `tests.conftest`.
from conftest import FakeEmbedder


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    monkeypatch.setattr(RAGTrustPipeline, "embedder", property(lambda self: FakeEmbedder()))
    monkeypatch.setattr(RAGTrustPipeline, "nli", property(lambda self: FakeNLI()))


class StubGenerator:
    def __init__(self, text="Photosynthesis converts sunlight into chemical energy.",
                 citations=None, error: str = None):
        self.text = text
        self.citations = citations or {}
        self.error = error
        self.calls = 0

    def generate(self, query, passages):
        self.calls += 1
        if self.error:
            raise GenerationError(self.error)
        return GeneratedAnswer(text=self.text, citations=self.citations)


def _build_index(tmp_path, embed_model="fake-embed-v1") -> str:
    cfg = Config(embed_model=embed_model)
    pipeline = RAGTrustPipeline(cfg, embedder=FakeEmbedder())
    pipeline.index_texts(
        ["Photosynthesis converts sunlight into chemical energy in plants."]
    )
    index_dir = str(tmp_path / "idx")
    pipeline.save(index_dir)
    return index_dir


def _client(tmp_path, generator=None, **config_kwargs) -> TestClient:
    config_kwargs.setdefault("embed_model", "fake-embed-v1")
    index_dir = _build_index(tmp_path, embed_model=config_kwargs["embed_model"])
    cfg = Config(**config_kwargs)
    app = create_app(index_dir, config=cfg, generator=generator or StubGenerator())
    return TestClient(app)


def test_health_reports_index_and_model_info(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["passages"] == 1
    assert body["embed_model"] == "fake-embed-v1"
    assert "nli_model" in body


def test_config_returns_effective_config(tmp_path):
    client = _client(tmp_path, k=3)
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["embed_model"] == "fake-embed-v1"
    assert body["k"] == 3


def test_answer_happy_path(tmp_path):
    client = _client(
        tmp_path,
        generator=StubGenerator(text="Photosynthesis converts sunlight into chemical energy."),
        embed_model="fake-embed-v1", retrieval_gate=-2.0, abstain_threshold=0.1,
    )
    resp = client.post("/answer", json={"question": "How does photosynthesis work?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert "trust" in body
    assert "metrics" in body
    assert "passages" in body
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0


def test_answer_empty_question_returns_422(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/answer", json={"question": ""})
    assert resp.status_code == 422


def test_answer_missing_question_returns_422(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/answer", json={})
    assert resp.status_code == 422


def test_answer_abstention_is_a_successful_response(tmp_path):
    # retrieval_gate above the maximum possible cosine similarity (1.0)
    # guarantees the pre-generation gate fires, regardless of corpus/query.
    client = _client(tmp_path, retrieval_gate=1.5)
    resp = client.post("/answer", json={"question": "Anything?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is True
    assert body["abstain_reason"] is not None
    assert body["is_trustworthy"] is False


def test_answer_unreachable_generator_returns_503(tmp_path):
    client = _client(
        tmp_path,
        generator=StubGenerator(error="Ollama backend unreachable at http://localhost:11434"),
        retrieval_gate=-2.0,
    )
    resp = client.post("/answer", json={"question": "How does photosynthesis work?"})
    assert resp.status_code == 503
    assert "unreachable" in resp.json()["detail"].lower()
