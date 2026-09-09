"""Generator interface shared by all backends (ollama, hf_api, cached)."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GeneratedAnswer:
    text: str
    citations: dict = field(default_factory=dict)


class Generator(Protocol):
    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        ...

    def generate_questions(self, answer: str, n: int) -> list[str]:
        """OPTIONAL. Back-generate up to `n` questions that `answer` would answer,
        for the RAGAS-style answer-relevance metric (metrics/relevance.py
        ::answer_relevance) -- see METRICS.md Part II.3.

        This method is NOT part of the required Generator surface: a backend that
        cannot support it (e.g. one with no second model call available, or one
        that only replays fixed answers) simply does not define it. The pipeline
        checks for its presence with `getattr(generator, "generate_questions",
        None)` and degrades `answer_relevance` to `None` when it is absent, when
        `Config.answer_relevance` is off, or when calling it fails -- it is a
        diagnostic add-on and must never be able to break the primary answer path.
        """
        ...


class GenerationError(Exception):
    """Raised when a generation backend is unreachable, mis-configured, or
    returns an unusable response. Callers should catch this rather than
    let a hung request or raw HTTP exception propagate."""
