"""Regressions for two defects that running the documented Quickstart exposed
and that 301 unit tests did not.

Both are the same species of gap: every existing test exercised the *metric*
layer, and nothing exercised the shipped `ragtrust ask` path a reader of the
README actually runs. Validation depth does not substitute for running the
program.

1. `_print_human_result` formatted every metric with `:.3f`. Since
   `answer_relevance` is None by default (its flag is opt-in) and `conciseness`
   is None below two claims, the primary command raised
   `TypeError: unsupported format string passed to NoneType.__format__`
   on the default configuration -- `ragtrust ask` crashed every time.

2. `OllamaGenerator` sent no sampling options, so Ollama's default temperature
   (0.8) applied. Asking one identical question three times produced geometric
   trust scores of 0.000, 0.525 and 0.000 -- the score moved when neither the
   question nor the corpus had. A package whose purpose is measurement must
   generate deterministically by default.
"""
import io
from contextlib import redirect_stdout

from ragtrust.cli import _print_human_result
from ragtrust.generation.ollama import DEFAULT_MODEL, OllamaGenerator, build_prompt


def _result(metrics: dict) -> dict:
    return {
        "answer": "Some answer [1].",
        "abstained": False,
        "abstain_reason": None,
        "metrics": metrics,
        "trust": {"arithmetic": 0.5, "geometric": 0.4},
        "is_trustworthy": True,
    }


# --------------------------------------------------------------- rendering


def test_print_human_result_renders_none_metric_without_raising():
    """The exact crash: a None metric must not blow up the ask path."""
    out = io.StringIO()
    with redirect_stdout(out):
        _print_human_result(_result({"faithfulness": 0.8, "answer_relevance": None}))
    text = out.getvalue()
    assert "n/a" in text
    assert "0.800" in text


def test_print_human_result_marks_undefined_rather_than_printing_zero():
    """`None` means undefined, not zero. Rendering it as 0.000 would claim the
    metric was measured and found absent -- a different, and false, statement.
    It must also never print the bare word "None"."""
    out = io.StringIO()
    with redirect_stdout(out):
        _print_human_result(_result({"conciseness": None}))
    text = out.getvalue()
    assert "n/a" in text
    assert "0.000" not in text
    assert "None" not in text


def test_print_human_result_handles_every_metric_being_none():
    out = io.StringIO()
    with redirect_stdout(out):
        _print_human_result(_result({"conciseness": None, "answer_relevance": None}))
    assert out.getvalue().count("n/a") == 2


def test_print_human_result_still_formats_ordinary_floats():
    out = io.StringIO()
    with redirect_stdout(out):
        _print_human_result(_result({"faithfulness": 0.8161, "attribution": 0.4}))
    text = out.getvalue()
    assert "0.816" in text
    assert "0.400" in text


# ------------------------------------------------------------ determinism


class _CapturingPost:
    """Stand-in for `requests.post` that records the JSON body it was sent."""

    def __init__(self):
        self.payload = None

    def __call__(self, url, json=None, timeout=None):
        self.payload = json

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"response": "A sentence [1]."}

        return _Resp()


def test_generate_requests_deterministic_sampling_by_default(monkeypatch):
    post = _CapturingPost()
    monkeypatch.setattr("ragtrust.generation.ollama.requests.post", post)

    OllamaGenerator().generate("q", ["passage one"])

    assert post.payload["options"]["temperature"] == 0.0, (
        "default must be greedy decoding -- a sampled generator makes the trust "
        "score unrepeatable for an unchanged question and corpus"
    )
    assert post.payload["options"]["seed"] == 0
    assert post.payload["model"] == DEFAULT_MODEL


def test_generate_questions_uses_the_same_deterministic_options(monkeypatch):
    """Back-generation feeds `answer_relevance`; if it samples, that metric moves
    between runs for an unchanged answer."""
    post = _CapturingPost()
    monkeypatch.setattr("ragtrust.generation.ollama.requests.post", post)

    OllamaGenerator().generate_questions("some answer", 3)

    assert post.payload["options"]["temperature"] == 0.0
    assert post.payload["options"]["seed"] == 0


def test_sampling_remains_available_when_explicitly_requested():
    """Determinism is a default, not a prohibition."""
    gen = OllamaGenerator(temperature=0.7, seed=42)
    assert gen._options() == {"temperature": 0.7, "seed": 42}


# -------------------------------------------------- prompt example neutrality


def test_citation_example_in_prompt_is_domain_neutral():
    """An earlier version of this fix illustrated the citation format with a
    sentence about max-pooling. Asked a max-pooling question, llama3.2:3b copied
    that example verbatim into its answer as though it were retrieved fact. The
    worked example must not resemble plausible corpus content.
    """
    prompt = build_prompt("What is max-pooling?", ["Max-pooling downsamples."])
    example_region = prompt.split("Correct format")[1].split("Passages:")[0].lower()
    for leaky in ("max-pooling", "receptive field", "feature map", "convolution"):
        assert leaky not in example_region, (
            f"citation example mentions {leaky!r}; a model can echo it as fact"
        )
