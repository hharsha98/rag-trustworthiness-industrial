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


# ------------------------------------------------------ dashboard mount / /api/examples


def test_create_app_still_exposes_health_and_answer_without_dashboard_dir(tmp_path):
    """`dashboard/` does not exist in this repo (a frontend built separately,
    per the task boundaries), so every test in this file already exercises the
    "missing frontend" path -- create_app must still start (no crash, just a
    logged warning) and /health, /answer must still be routed normally rather
    than swallowed by a static-file mount or 404."""
    client = _client(
        tmp_path,
        generator=StubGenerator(text="Photosynthesis converts sunlight into chemical energy."),
        retrieval_gate=-2.0, abstain_threshold=0.1,
    )
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.post("/answer", json={"question": "How does photosynthesis work?"})
    assert resp.status_code == 200
    assert "trust" in resp.json()


def test_api_examples_returns_a_list_without_erroring(tmp_path):
    # data/cached_answers.json exists in this repo, so this also covers the
    # "file present" branch; test_api_examples_empty_when_cached_answers_file_absent
    # below covers the "file absent" branch explicitly with a monkeypatched root.
    client = _client(tmp_path)
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        assert set(body[0].keys()) == {"question", "category"}


def test_api_examples_empty_when_cached_answers_file_absent(tmp_path, monkeypatch):
    import ragtrust.service as service_module

    monkeypatch.setattr(service_module, "_REPO_ROOT", tmp_path)
    client = _client(tmp_path)
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    assert resp.json() == []


# --------------------------------------------------------- corpus upload / /corpora

# Distinct from the bundled index's "Photosynthesis..." passage (see
# _build_index above), so a test can tell whether an answer's passages came
# from the upload or from the base corpus just by which words show up.
_BADGER_MD = (
    b"## Habitat\n\n"
    b"Badgers dig extensive burrow systems called setts, which can house "
    b"multiple generations across many decades of continuous use.\n"
)


def _upload_client(tmp_path, monkeypatch, **config_kwargs) -> TestClient:
    monkeypatch.setenv("RAGTRUST_UPLOAD_DIR", str(tmp_path / "uploads"))
    config_kwargs.setdefault("retrieval_gate", -2.0)
    config_kwargs.setdefault("abstain_threshold", 0.1)
    return _client(tmp_path, **config_kwargs)


def test_upload_corpus_returns_201_and_is_then_listed(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    resp = client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
    assert resp.status_code == 201
    body = resp.json()
    assert "corpus_id" in body
    assert body["passages"] >= 1
    assert body["filename"] == "badgers.md"

    listing = client.get("/corpora").json()
    assert any(r["corpus_id"] == body["corpus_id"] for r in listing)


def test_answer_with_corpus_id_uses_uploaded_passages(tmp_path, monkeypatch):
    client = _upload_client(
        tmp_path, monkeypatch,
        generator=StubGenerator(text="Badgers dig burrows called setts."),
    )
    corpus_id = client.post(
        "/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")}
    ).json()["corpus_id"]

    resp = client.post("/answer", json={"question": "What do badgers dig?", "corpus_id": corpus_id})
    assert resp.status_code == 200
    body = resp.json()
    passage_text = " ".join(p["text"] for p in body["passages"])
    assert "Badgers" in passage_text
    assert "Photosynthesis" not in passage_text


def test_answer_with_no_corpus_id_still_uses_the_base_corpus(tmp_path, monkeypatch):
    # Existing behaviour (the base "Photosynthesis..." corpus) must be
    # unaffected by the presence of the upload machinery.
    client = _upload_client(tmp_path, monkeypatch)
    resp = client.post("/answer", json={"question": "How does photosynthesis work?"})
    assert resp.status_code == 200
    passage_text = " ".join(p["text"] for p in resp.json()["passages"])
    assert "Photosynthesis" in passage_text


def test_answer_with_unknown_wellformed_corpus_id_returns_404(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    unknown_id = "deadbeef" * 4  # 32 hex chars, matches CORPUS_ID_RE, but was never created
    resp = client.post("/answer", json={"question": "Anything?", "corpus_id": unknown_id})
    assert resp.status_code == 404


def test_answer_with_malformed_corpus_id_returns_404_not_500(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    resp = client.post("/answer", json={"question": "Anything?", "corpus_id": "../x"})
    assert resp.status_code == 404


def test_upload_oversized_returns_413(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGTRUST_MAX_UPLOAD_MB", "1")
    client = _upload_client(tmp_path, monkeypatch)
    oversized = b"x" * (2 * 1024 * 1024)
    resp = client.post("/corpora", files={"file": ("big.txt", oversized, "text/plain")})
    assert resp.status_code == 413


def test_upload_unsupported_extension_returns_400(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    resp = client.post("/corpora", files={"file": ("report.docx", b"whatever bytes", "application/octet-stream")})
    assert resp.status_code == 400


def test_delete_corpus_then_answering_with_it_returns_404(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    corpus_id = client.post(
        "/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")}
    ).json()["corpus_id"]

    resp = client.delete(f"/corpora/{corpus_id}")
    assert resp.status_code == 204

    resp = client.post("/answer", json={"question": "Anything?", "corpus_id": corpus_id})
    assert resp.status_code == 404


def test_health_reports_corpora_count(tmp_path, monkeypatch):
    client = _upload_client(tmp_path, monkeypatch)
    assert client.get("/health").json()["corpora"] == 0
    client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
    assert client.get("/health").json()["corpora"] == 1


# ---------------------------------------------------- upload abuse protection
#
# create_app reads RAGTRUST_UPLOAD_RATE_LIMIT / _GLOBAL_LIMIT / _RATE_WINDOW /
# _TOKEN / RAGTRUST_TRUST_PROXY once, at app-creation time (see ratelimit.py
# and service.py::create_app), so every test below sets the env var(s) it
# needs BEFORE calling _upload_client -- which builds the app -- rather than
# after.


def test_sixth_upload_in_window_returns_429_with_retry_after(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGTRUST_UPLOAD_RATE_LIMIT", "5")
    client = _upload_client(tmp_path, monkeypatch)
    for _ in range(5):
        resp = client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
        assert resp.status_code == 201

    resp = client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    retry_after = int(resp.headers["Retry-After"])
    assert 0 < retry_after <= 3600
    assert "5" in resp.json()["detail"]


def test_global_limit_trips_even_with_a_different_forwarded_ip_per_request(tmp_path, monkeypatch):
    # The scenario the global limiter exists for: under IPv6 a single host can
    # control a /64 (billions of addresses), so a per-client limit alone
    # bounds nothing if every request looks like a new client. Each request
    # here presents a distinct X-Forwarded-For (trust_proxy=True, so
    # client_key actually uses it) -- the per-client limiter never sees the
    # same key twice, yet the shared global budget still runs out.
    monkeypatch.setenv("RAGTRUST_TRUST_PROXY", "true")
    monkeypatch.setenv("RAGTRUST_UPLOAD_RATE_LIMIT", "100")  # generous: must not be what trips
    monkeypatch.setenv("RAGTRUST_UPLOAD_GLOBAL_LIMIT", "3")
    client = _upload_client(tmp_path, monkeypatch)

    for i in range(3):
        resp = client.post(
            "/corpora",
            files={"file": ("badgers.md", _BADGER_MD, "text/markdown")},
            headers={"X-Forwarded-For": f"2001:db8::{i:x}"},
        )
        assert resp.status_code == 201

    resp = client.post(
        "/corpora",
        files={"file": ("badgers.md", _BADGER_MD, "text/markdown")},
        headers={"X-Forwarded-For": "2001:db8::ffff"},
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_upload_token_unset_keeps_uploads_public(tmp_path, monkeypatch):
    monkeypatch.delenv("RAGTRUST_UPLOAD_TOKEN", raising=False)
    client = _upload_client(tmp_path, monkeypatch)
    resp = client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
    assert resp.status_code == 201


def test_upload_token_set_rejects_missing_or_wrong_and_accepts_correct(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGTRUST_UPLOAD_TOKEN", "s3cr3t-token")
    client = _upload_client(tmp_path, monkeypatch)

    resp = client.post("/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing or invalid upload token."

    resp = client.post(
        "/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")},
        headers={"X-Upload-Token": "wrong-guess"},
    )
    assert resp.status_code == 401

    resp = client.post(
        "/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")},
        headers={"X-Upload-Token": "s3cr3t-token"},
    )
    assert resp.status_code == 201


def test_correct_token_bypasses_both_rate_limiters(tmp_path, monkeypatch):
    # A valid token marks the caller as an authenticated operator, not
    # anonymous traffic -- both limits stay at 1 for this test, and every
    # request still succeeds because the token check short-circuits before
    # either limiter is ever consulted.
    monkeypatch.setenv("RAGTRUST_UPLOAD_TOKEN", "s3cr3t-token")
    monkeypatch.setenv("RAGTRUST_UPLOAD_RATE_LIMIT", "1")
    monkeypatch.setenv("RAGTRUST_UPLOAD_GLOBAL_LIMIT", "1")
    client = _upload_client(tmp_path, monkeypatch)

    for _ in range(3):
        resp = client.post(
            "/corpora", files={"file": ("badgers.md", _BADGER_MD, "text/markdown")},
            headers={"X-Upload-Token": "s3cr3t-token"},
        )
        assert resp.status_code == 201


def test_corpora_limits_returns_policy_and_never_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGTRUST_UPLOAD_TOKEN", "s3cr3t-token")
    monkeypatch.setenv("RAGTRUST_UPLOAD_RATE_LIMIT", "5")
    monkeypatch.setenv("RAGTRUST_UPLOAD_GLOBAL_LIMIT", "30")
    monkeypatch.setenv("RAGTRUST_UPLOAD_RATE_WINDOW", "3600")
    client = _upload_client(tmp_path, monkeypatch)

    resp = client.get("/corpora/limits")
    assert resp.status_code == 200
    assert resp.json() == {
        "per_client": 5, "global": 30, "window_seconds": 3600, "token_required": True,
    }
    assert "s3cr3t-token" not in resp.text


def test_corpora_limits_reports_token_not_required_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RAGTRUST_UPLOAD_TOKEN", raising=False)
    client = _upload_client(tmp_path, monkeypatch)
    assert client.get("/corpora/limits").json()["token_required"] is False
