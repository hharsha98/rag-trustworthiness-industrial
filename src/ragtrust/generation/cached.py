"""Cached generator: serves pre-recorded answers from a JSON file keyed by
question text, so a hosted demo never errors when no LLM backend is
reachable.

Accepted JSON shapes, per entry:
{
  "Question A?": "Plain text answer.",
  "Question B?": {"text": "Answer with [1] a citation.", "citations": {"1": 0}},
  "Question C?": {"answer": "Answer with [1] a marker.", "passages": [...], ...}
}

The third form is what `data/cached_answers.json` uses. Both "text" and "answer" are
read; when no explicit "citations" map is present, inline [n] markers in the text are
parsed instead.

`generate_questions` (optional Generator method, see generation/base.py) serves
recorded back-generation questions, never fabricated ones: an entry may additionally
carry a "generated_questions" list, e.g.
{"Question C?": {"answer": "...", "generated_questions": ["...", "..."]}}. When the
entry that produced `answer` has no such recorded list -- true of every entry today --
this raises `GenerationError`, exactly like a real backend that could not
back-generate, so the pipeline degrades `answer_relevance` to `None` instead of
inventing questions a replayed answer never actually had.
"""
import json
from pathlib import Path

from .base import GeneratedAnswer, GenerationError
from .ollama import parse_citations


class CachedGenerator:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self._cache = None

    def _load(self) -> dict:
        if self._cache is None:
            if not self.cache_path.exists():
                raise GenerationError(f"Cache file not found: {self.cache_path}")
            with open(self.cache_path) as f:
                self._cache = json.load(f)
        return self._cache

    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        cache = self._load()
        entry = cache.get(query)
        if entry is None:
            raise GenerationError(f"No cached answer for question: {query!r}")

        if isinstance(entry, str):
            text = entry
            citations = parse_citations(text)
        else:
            # Accept "answer" as well as "text" -- data/cached_answers.json uses
            # "answer"; reading only "text" silently yields an empty string, which
            # makes the pipeline abstain on every question regardless of what was
            # actually cached.
            text = entry.get("text") or entry.get("answer") or ""
            if not text:
                raise GenerationError(
                    f"Cached entry for {query!r} has neither 'text' nor 'answer'; "
                    f"keys present: {sorted(entry)}"
                )
            raw = entry.get("citations")
            # Fall back to parsing inline [n] markers when no explicit citation map
            # was recorded, which is the normal case for generated answers.
            citations = ({int(k): v for k, v in raw.items()} if raw
                         else parse_citations(text))

        return GeneratedAnswer(text=text, citations=citations)

    def generate_questions(self, answer: str, n: int) -> list:
        """Serve a recorded "generated_questions" list for the entry whose answer
        text matches `answer`, if one was recorded. Otherwise raise
        `GenerationError` -- see module docstring: a replayed answer has no model
        behind it to back-generate with, so this never fabricates questions.
        """
        cache = self._load()
        for entry in cache.values():
            if not isinstance(entry, dict):
                continue
            text = entry.get("text") or entry.get("answer")
            if text == answer:
                cached_questions = entry.get("generated_questions")
                if cached_questions:
                    return list(cached_questions)[:n]
        raise GenerationError(
            "No cached generated_questions recorded for this answer; "
            "CachedGenerator does not fabricate back-generation questions."
        )
