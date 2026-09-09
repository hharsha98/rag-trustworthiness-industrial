"""Hugging Face Inference API-backed generator.

Reads the API token from the HF_TOKEN environment variable. Degrades
gracefully: a missing token, connection failure, timeout, or unexpected
response shape is raised as a `GenerationError` rather than propagating a
raw exception or hanging -- the request has a short (30s) timeout.
"""
import os

import requests

from .base import GeneratedAnswer, GenerationError
from .ollama import build_prompt, build_question_prompt, parse_citations, parse_questions

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TIMEOUT_S = 30


class HFAPIGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = TIMEOUT_S):
        self.model = model
        self.timeout = timeout

    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise GenerationError("HF_TOKEN environment variable is not set.")

        prompt = build_prompt(query, passages)
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.post(
                url, headers=headers, json={"inputs": prompt}, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GenerationError(f"HF Inference API unreachable for {self.model}: {exc}") from exc

        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        else:
            raise GenerationError(f"Unexpected HF Inference API response shape: {data!r}")

        return GeneratedAnswer(text=text, citations=parse_citations(text))

    def generate_questions(self, answer: str, n: int) -> list:
        """Back-generation for `answer_relevance` (METRICS.md Part II.3): a second
        HF Inference API call asking the model for `n` questions this answer would
        answer. Same degrade-gracefully contract as `generate`: any failure is
        raised as `GenerationError`, which the pipeline catches to fall back to
        `answer_relevance = None` rather than let this diagnostic call break
        answering.
        """
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise GenerationError("HF_TOKEN environment variable is not set.")

        prompt = build_question_prompt(answer, n)
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.post(
                url, headers=headers, json={"inputs": prompt}, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GenerationError(f"HF Inference API unreachable for {self.model}: {exc}") from exc

        data = response.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        else:
            raise GenerationError(f"Unexpected HF Inference API response shape: {data!r}")

        return parse_questions(text, n)
