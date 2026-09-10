"""Ollama-backed generator: HTTP call to a local Ollama server.

Degrades gracefully: any connection failure, timeout, or HTTP error is
raised as a `GenerationError` rather than propagating a raw exception or
hanging -- the request has a short (30s) timeout.
"""
import re

import requests

from .base import GeneratedAnswer, GenerationError

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_URL = "http://localhost:11434/api/generate"
TIMEOUT_S = 30

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Strips a leading list marker -- "1.", "2)", "-", "*", "•" -- so a numbered or
# bulleted line of generated questions parses down to just the question text.
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.\)]|[-*•])\s*")


def build_prompt(query: str, passages: list) -> str:
    # Passages are numbered from 1, the ordinary convention for citations, so the
    # marker number is one greater than the passage's list index; `parse_citations`
    # converts back. Keep the two in step -- an off-by-one here silently attributes
    # every claim to the wrong passage, which surfaces as an attribution score of
    # zero rather than as an error.
    lines = []
    for i, p in enumerate(passages, start=1):
        text = getattr(p, "text", p)
        lines.append(f"[{i}] {text}")
    context_block = "\n".join(lines)
    # The citation instruction is stated as hard rules WITH a worked example,
    # because the bare instruction is not reliably followed by small local models.
    # Observed failure (llama3.2:3b on the demo corpus): the model emitted three
    # uncited numbered claims followed by one trailing sentence, "These claims are
    # supported by [1]." `parse_citations` correctly attached the only marker to
    # that trailing sentence, which no passage entails, so attribution scored
    # 0.000 -- and because T_geom is zero if any dimension is zero, a correct,
    # well-grounded answer was reported as NOT trustworthy. The metric was right
    # and the generation was non-compliant, so the fix belongs here rather than
    # in `attribution()`.
    return (
        "Answer the question using ONLY the numbered passages below.\n\n"
        "CITATION RULES (mandatory):\n"
        "1. End EVERY sentence that makes a claim with a citation marker [n], "
        "where n is the number of the passage supporting that sentence.\n"
        "2. Do NOT group citations into a summary sentence. A trailing line such "
        "as \"These claims are supported by [1].\" is wrong -- every claim "
        "sentence carries its own marker.\n"
        "3. If several passages support a sentence, cite the single best one.\n"
        "4. Write plain prose sentences. Do not number the claims as a list.\n\n"
        # The example MUST stay domain-neutral. An earlier version illustrated the
        # format with a sentence about max-pooling; asked a max-pooling question,
        # llama3.2:3b copied the example verbatim into its answer as fact. A
        # deliberately generic example cannot be mistaken for corpus content.
        "Correct format (illustrating only the citation style):\n"
        "The process runs in two stages [2]. The second stage begins only after "
        "the first one succeeds [1].\n\n"
        f"Passages:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )


def build_question_prompt(answer: str, n: int) -> str:
    """Prompt for RAGAS-style answer-relevance back-generation (METRICS.md
    Part II.3): ask the model for exactly `n` questions this answer would answer."""
    return (
        f"Write exactly {n} questions that the following answer would be a good "
        "response to. Output ONLY the questions, one per line, with no numbering, "
        "bullets, or extra commentary.\n\n"
        f"Answer: {answer}\n\nQuestions:"
    )


def parse_questions(text: str, n: int) -> list:
    """Split `text` into non-empty lines, strip leading numbering/bullets, and
    return at most `n` of them. Used to parse back-generated questions for
    `answer_relevance` -- see `build_question_prompt`."""
    questions = []
    for line in text.splitlines():
        line = _LIST_MARKER_RE.sub("", line.strip()).strip()
        if not line:
            continue
        questions.append(line)
        if len(questions) >= n:
            break
    return questions


def parse_citations(text: str) -> dict:
    """Map claim index (sentence position in `text`) to the 0-based passage index
    of the first `[n]` marker in that sentence.

    Markers are 1-based (see `build_prompt`), so the marker number is decremented.
    A `[0]` marker would be out of range and is ignored rather than wrapping round
    to the last passage.
    """
    citations = {}
    for i, sentence in enumerate(_SENTENCE_SPLIT_RE.split(text)):
        match = _CITATION_RE.search(sentence)
        if match:
            passage_index = int(match.group(1)) - 1
            if passage_index >= 0:
                citations[i] = passage_index
    return citations


class OllamaGenerator:
    """Ollama generator, deterministic by default.

    *** On temperature=0 being the default. ***
    Ollama samples at temperature 0.8 unless told otherwise. That is a reasonable
    default for a chat assistant and the wrong one here: this package exists to
    *measure* answers, and a sampled generator makes the measurement unrepeatable.
    Observed on the demo corpus with llama3.2:3b, asking one identical question
    three times, the geometric trust score came back 0.000, then 0.525, then
    0.000 -- not because grounding changed, but because the model formatted its
    citations differently each time. A trust score that moves when nothing about
    the question or corpus moved is not a measurement.

    `temperature=0.0` and a fixed `seed` make repeated runs comparable, which is
    what `Config.seed` already promises everywhere else in the pipeline. Callers
    who deliberately want sampling (e.g. generating varied drafts) can pass a
    higher temperature explicitly.
    """

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 timeout: float = TIMEOUT_S, temperature: float = 0.0, seed: int = 0):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed

    def _options(self) -> dict:
        return {"temperature": self.temperature, "seed": self.seed}

    def _post(self, prompt: str) -> str:
        # Single transport path for every prompt this backend sends (generate,
        # generate_questions, complete) so the timeout/GenerationError handling
        # can't drift between them -- see the module docstring.
        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": self._options()},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GenerationError(f"Ollama backend unreachable at {self.url}: {exc}") from exc

        data = response.json()
        return data.get("response", "")

    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        prompt = build_prompt(query, passages)
        text = self._post(prompt)
        return GeneratedAnswer(text=text, citations=parse_citations(text))

    def generate_questions(self, answer: str, n: int) -> list:
        """Back-generation for `answer_relevance` (METRICS.md Part II.3): a second
        HTTP call asking the model for `n` questions this answer would answer.

        Reuses the same timeout/`GenerationError` handling as `generate` -- callers
        (the pipeline) are expected to catch `GenerationError` and degrade
        `answer_relevance` to `None` rather than let this diagnostic call break
        answering.
        """
        prompt = build_question_prompt(answer, n)
        text = self._post(prompt)
        return parse_questions(text, n)

    def complete(self, prompt: str) -> str:
        """Raw completion (generation/base.py::Generator.complete) -- used by
        ingest/contextualize.py for indexing-time blurb generation. No citation
        prompting/parsing: the caller gets the model's text verbatim."""
        return self._post(prompt)
