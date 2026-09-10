"""OpenAI-compatible generator: HTTP call to any OpenAI chat-completions-style
router (e.g. OmniRoute on a deployment VPS, or any other `/v1/chat/completions`
provider).

Reuses `build_prompt`/`parse_citations` and `build_question_prompt`/
`parse_questions` from `ollama.py` so the citation contract -- how the prompt
asks for `[n]` markers and how they are parsed back into a claim->passage map --
stays identical across every generation backend. Only the transport (HTTP
payload shape, auth header) differs.

Degrades gracefully: any connection failure, timeout, non-2xx status, or
unparseable response is raised as a `GenerationError` rather than propagating
a raw exception or hanging -- the request has a bounded timeout.
"""
import os

import requests

from .base import GeneratedAnswer, GenerationError
from .ollama import build_prompt, build_question_prompt, parse_citations, parse_questions

DEFAULT_MODEL = "auto/best-fast"
TIMEOUT_S = 60.0


class OpenAICompatGenerator:
    """OpenAI-compatible chat-completions generator, deterministic by default.

    *** On temperature=0 being the default. ***
    Same reasoning as `OllamaGenerator` (see its docstring): this package exists
    to *measure* answers, and a sampled generator makes the trust score
    unrepeatable for an unchanged question. `temperature=0.0` plus a fixed
    `seed` are sent on every request so repeated runs are comparable, matching
    what `Config.seed` already promises elsewhere in the pipeline. Callers who
    deliberately want sampling can pass a higher `temperature` explicitly.

    Configuration is environment-driven so a deployment only needs to set env
    vars, not code: `base_url`/`api_key`/`model` fall back to
    `RAGTRUST_LLM_BASE_URL`/`RAGTRUST_LLM_API_KEY`/`RAGTRUST_LLM_MODEL` when the
    corresponding constructor argument is left as `None`.
    """

    def __init__(self, base_url: str = None, api_key: str = None,
                 model: str = "auto/best-fast", timeout: float = TIMEOUT_S,
                 temperature: float = 0.0, seed: int = 0):
        self.base_url = (base_url if base_url is not None
                          else os.environ.get("RAGTRUST_LLM_BASE_URL"))
        self.api_key = (api_key if api_key is not None
                         else os.environ.get("RAGTRUST_LLM_API_KEY"))
        if model == "auto/best-fast" and os.environ.get("RAGTRUST_LLM_MODEL"):
            # Only defer to the env var when the caller left `model` at its
            # default -- an explicit argument always wins.
            model = os.environ["RAGTRUST_LLM_MODEL"]
        self.model = model
        # Read timeout is env-overridable because the right value depends on what
        # sits behind the endpoint, not on this code. A router that queues behind
        # a busy upstream, or cold-starts a provider, can take well over a minute
        # for a single completion -- observed against a live deployment, where the
        # default 60s produced a "backend unreachable" 503 for an endpoint that
        # was in fact healthy, merely slow. Raising it trades a longer worst-case
        # wait for not mislabelling slowness as failure.
        if timeout == TIMEOUT_S and os.environ.get("RAGTRUST_LLM_TIMEOUT"):
            try:
                timeout = float(os.environ["RAGTRUST_LLM_TIMEOUT"])
            except ValueError:
                pass  # keep the default rather than crash on a malformed value
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self) -> str:
        if not self.base_url:
            raise GenerationError(
                "OpenAI-compatible backend is not configured: set RAGTRUST_LLM_BASE_URL "
                "or pass base_url."
            )
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    def _post(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.seed,
        }
        try:
            response = requests.post(
                self._url(), headers=self._headers(), json=payload, timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            # `exc` is a requests exception rendered from the request/response
            # objects, never from `self.api_key` directly, so the key itself is
            # never interpolated into this message.
            raise GenerationError(
                f"OpenAI-compatible backend unreachable at {self.base_url}: {exc}"
            ) from exc

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GenerationError(
                f"Unexpected OpenAI-compatible response shape from {self.base_url}: {exc}"
            ) from exc

    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        prompt = build_prompt(query, passages)
        text = self._post(prompt)
        return GeneratedAnswer(text=text, citations=parse_citations(text))

    def generate_questions(self, answer: str, n: int) -> list:
        """Back-generation for `answer_relevance` (METRICS.md Part II.3): a
        second chat-completions call asking the model for `n` questions this
        answer would answer. Same `GenerationError` contract as `generate` --
        the pipeline catches it and degrades `answer_relevance` to `None`
        rather than let this diagnostic call break answering.
        """
        prompt = build_question_prompt(answer, n)
        text = self._post(prompt)
        return parse_questions(text, n)

    def complete(self, prompt: str) -> str:
        """Raw completion (generation/base.py::Generator.complete) -- used by
        ingest/contextualize.py for indexing-time blurb generation. No citation
        prompting/parsing: the caller gets the model's text verbatim."""
        return self._post(prompt)
