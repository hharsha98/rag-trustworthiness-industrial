"""Tests for the OpenAI-compatible generator (OmniRoute etc.), fast and
offline: `requests.post` is monkeypatched, so no network call is ever made.

`tests/` is not a package (see the note in test_cli.py/test_service.py), so
fixtures from conftest.py are imported by bare module name.
"""
import requests

from ragtrust.generation.base import GenerationError
from ragtrust.generation.openai_compat import OpenAICompatGenerator


class _FakePassage:
    def __init__(self, text):
        self.text = text


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json


# --------------------------------------------------------------------- generate


def test_generate_yields_generated_answer_with_citations_like_ollama(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        text = "The process runs in two stages [2]. It finishes after that [1]."
        return _FakeResponse(_chat_response(text))

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key="secret-key")
    passages = [_FakePassage("passage one"), _FakePassage("passage two")]
    result = gen.generate("What happens?", passages)

    assert result.text == "The process runs in two stages [2]. It finishes after that [1]."
    # Same parse_citations contract as OllamaGenerator: marker [n] -> passage index n-1,
    # keyed by sentence position.
    assert result.citations == {0: 1, 1: 0}
    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"


def test_request_body_carries_temperature_zero_and_seed(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(_chat_response("An answer [1]."))

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(
        base_url="http://127.0.0.1:20128", api_key="k", seed=42,
    )
    gen.generate("Q?", [_FakePassage("p")])

    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["seed"] == 42


def test_non_2xx_status_raises_generation_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"error": "boom"}, status_code=500)

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key="k")
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError:
        pass


def test_connection_error_raises_generation_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key="k")
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError:
        pass


def test_unparseable_response_raises_generation_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key="k")
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError:
        pass


# ---------------------------------------------------------------- generate_questions


def test_generate_questions_parses_numbered_and_bulleted_lines_and_respects_n(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        text = (
            "1. What is the process?\n"
            "2) How many stages are there?\n"
            "- What happens after the first stage?\n"
            "* An extra bullet that should be dropped by n=3\n"
        )
        return _FakeResponse(_chat_response(text))

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key="k")
    questions = gen.generate_questions("The process runs in two stages.", n=3)

    assert questions == [
        "What is the process?",
        "How many stages are there?",
        "What happens after the first stage?",
    ]


# --------------------------------------------------------------------------- config


def test_env_var_configuration_used_when_constructor_args_omitted(monkeypatch):
    monkeypatch.setenv("RAGTRUST_LLM_BASE_URL", "http://127.0.0.1:20128")
    monkeypatch.setenv("RAGTRUST_LLM_API_KEY", "env-key")
    monkeypatch.setenv("RAGTRUST_LLM_MODEL", "auto/best-reasoning")

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(_chat_response("An answer [1]."))

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator()
    gen.generate("Q?", [_FakePassage("p")])

    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"
    assert captured["json"]["model"] == "auto/best-reasoning"


def test_missing_base_url_raises_generation_error(monkeypatch):
    monkeypatch.delenv("RAGTRUST_LLM_BASE_URL", raising=False)
    gen = OpenAICompatGenerator(api_key="k")
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError:
        pass


# ------------------------------------------------------------------- key never leaks


def test_api_key_never_appears_in_exception_message(monkeypatch):
    secret = "sk-super-secret-value-12345"

    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key=secret)
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError as exc:
        assert secret not in str(exc)


def test_api_key_never_appears_in_non_2xx_exception_message(monkeypatch):
    secret = "sk-super-secret-value-12345"

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"error": "boom"}, status_code=401)

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key=secret)
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError as exc:
        assert secret not in str(exc)


def test_api_key_never_appears_in_unparseable_response_exception_message(monkeypatch):
    secret = "sk-super-secret-value-12345"

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({"unexpected": "shape"})

    monkeypatch.setattr(requests, "post", fake_post)

    gen = OpenAICompatGenerator(base_url="http://127.0.0.1:20128", api_key=secret)
    try:
        gen.generate("Q?", [_FakePassage("p")])
        assert False, "expected GenerationError"
    except GenerationError as exc:
        assert secret not in str(exc)
