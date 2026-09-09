"""Gradio demo for the RAG trustworthiness project.

This is the recruiter-facing artefact: it must load fast, never crash, and make the
system's trustworthiness metrics -- faithfulness, attribution, relevance, and
conciseness -- visible for every answer it produces.

Models are loaded lazily -- nothing heavy happens at import time. Every user-facing
handler is wrapped so an exception becomes a friendly `gr.Warning` instead of a
traceback in the UI.

Run locally:
    python app/app.py

Backend selection (env vars):
    RAGTRUST_GENERATOR   one of {ollama, hf_api, cached}
                         default: "hf_api" if HF_TOKEN is set, else "cached"
    RAGTRUST_NLI_MODEL   default "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    HF_TOKEN             HuggingFace Inference API token, used by the hf_api backend
    OLLAMA_HOST          e.g. "http://localhost:11434", used by the ollama backend
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import gradio as gr

# --------------------------------------------------------------------------------------
# Module-level constants -- fill these in before publishing.
# --------------------------------------------------------------------------------------
REPO_URL = "https://github.com/REPLACE_WITH_USERNAME/rag-trustworthiness-industrial"

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHED_ANSWERS_PATH = DATA_DIR / "cached_answers.json"
# The default corpus is original text written for this repository. Override with
# RAGTRUST_CORPUS to point at your own corpus document instead.
CORPUS_PATH = Path(os.environ.get("RAGTRUST_CORPUS", str(DATA_DIR / "demo_corpus.md")))


def _load_passages() -> list:
    """Corpus text as retrievable passages, via overlapping windows.

    Windowing keeps a heading together with the prose beneath it. Splitting on lines
    instead yields short fragments that are mostly headings, which an NLI model
    cannot entail and a generator cannot answer from.
    """
    from ragtrust.ingest.loader import chunk_passages, load_corpus

    return [c["text"] for c in chunk_passages(load_corpus(str(CORPUS_PATH)))]


# --------------------------------------------------------------------------------------
# Lazy, process-wide singletons. Nothing here runs at import time.
# --------------------------------------------------------------------------------------
_STATE = {
    "pipeline": None,
    "nli_scorer": None,
    "cached_answers": None,
}


def _nli_model_name() -> str:
    return os.environ.get("RAGTRUST_NLI_MODEL", DEFAULT_NLI_MODEL)


def _generator_backend_name() -> str:
    backend = os.environ.get("RAGTRUST_GENERATOR")
    if backend:
        return backend
    return "hf_api" if os.environ.get("HF_TOKEN") else "cached"


# --------------------------------------------------------------------------------------
# Cached-answers helpers (pure file IO, safe to call eagerly-ish, but we still only
# do it on first use so the module import itself stays instant). These back both the
# preset-question dropdown and the fallback shown when the live pipeline is
# unavailable, so the demo remains useful without an inference token.
# --------------------------------------------------------------------------------------
def _load_cached_answers() -> dict:
    if _STATE["cached_answers"] is not None:
        return _STATE["cached_answers"]
    try:
        cached = json.loads(CACHED_ANSWERS_PATH.read_text())
    except Exception:
        cached = {}
    _STATE["cached_answers"] = cached
    return cached


def _question_choices() -> list:
    return [f"[{v.get('category', '?')}] {q}" for q, v in _load_cached_answers().items()]


def _strip_choice_prefix(choice: str) -> str:
    if choice and choice.startswith("[") and "]" in choice:
        return choice.split("]", 1)[1].strip()
    return choice or ""


def _find_cached_answer(question: str) -> dict | None:
    q = (question or "").strip().lower()
    for key, val in _load_cached_answers().items():
        if key.strip().lower() == q:
            return val
    return None


# --------------------------------------------------------------------------------------
# Lazy model / pipeline construction
# --------------------------------------------------------------------------------------
def _build_generator():
    backend = _generator_backend_name()
    if backend == "ollama":
        from ragtrust.generation.ollama import OllamaGenerator

        return OllamaGenerator()
    if backend == "hf_api":
        import ragtrust.generation.hf_api as hf_api_mod

        # The API contract this app was written against names this class
        # `HFInferenceGenerator`; the implementation currently ships it as
        # `HFAPIGenerator`. Support both without editing src/.
        cls = getattr(hf_api_mod, "HFInferenceGenerator", None) or getattr(
            hf_api_mod, "HFAPIGenerator"
        )
        return cls()
    from ragtrust.generation.cached import CachedGenerator

    return CachedGenerator(str(CACHED_ANSWERS_PATH))


def _get_pipeline():
    """Build (once) and return the RAGTrustPipeline, indexed on the demo corpus."""
    if _STATE["pipeline"] is not None:
        return _STATE["pipeline"]

    from ragtrust.config import Config
    from ragtrust.pipeline import RAGTrustPipeline

    cfg = Config(nli_model=_nli_model_name())
    generator = _build_generator()
    try:
        pipe = RAGTrustPipeline(cfg, generator=generator)
    except TypeError:
        # Constructor signature may not take a generator kwarg yet.
        pipe = RAGTrustPipeline(cfg)
    pipe.index_texts(_load_passages())
    _STATE["pipeline"] = pipe
    return pipe


def _get_nli_scorer():
    if _STATE["nli_scorer"] is not None:
        return _STATE["nli_scorer"]
    from ragtrust.metrics.nli import NLIScorer

    scorer = NLIScorer(model_name=_nli_model_name())
    _STATE["nli_scorer"] = scorer
    return scorer


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def _truncate(text: str, n: int = 200) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def _empty_ask_outputs(note: str = ""):
    answer_md = f"*{note}*" if note else ""
    return (
        answer_md,
        [],
        [],
        "",
        gr.update(visible=False),
    )


# --------------------------------------------------------------------------------------
# Tab 1: Ask
# --------------------------------------------------------------------------------------
def _render_live_result(res, pipe=None, question: str = "") -> tuple:
    answer = getattr(res, "answer", "") or ""
    passages = getattr(res, "passages", []) or []
    claims = getattr(res, "claims", []) or []
    citations = getattr(res, "citations", {}) or {}
    metrics = getattr(res, "metrics", {}) or {}
    abstained = bool(getattr(res, "abstained", False))

    # Invert citations: claim_idx -> passage_rank  =>  passage_rank -> [claim_idx, ...]
    #
    # Citation markers refer to a passage's RANK in the retrieved list (the [n] the
    # generator was shown, 1-based, converted to a 0-based rank by parse_citations),
    # not to Passage.id, which is the passage's index in the whole corpus. Keying
    # this lookup by Passage.id makes every row read "-" even when attribution is
    # non-zero.
    citing_claims = {}
    for claim_idx, passage_rank in citations.items():
        citing_claims.setdefault(passage_rank, []).append(claim_idx)

    passage_rows = []
    for rank, p in enumerate(passages):
        cited_by = sorted(citing_claims.get(rank, []))
        cited_str = ", ".join(f"c{i}" for i in cited_by) if cited_by else "-"
        passage_rows.append(
            [
                f"{rank + 1} (corpus #{getattr(p, 'id', '?')})",
                round(float(getattr(p, "score", 0.0)), 4),
                _truncate(getattr(p, "text", "")),
                cited_str,
            ]
        )

    # Metrics (best-effort key lookup -- exact metric dict keys are an
    # implementation detail of RAGTrustPipeline that may still be in flux).
    aliases = {
        "Faithfulness": ["faithfulness", "faithfulness_score"],
        "Explainability / Attribution": [
            "attribution_f1",
            "attribution",
            "explainability_f1",
            "explainability",
        ],
        "Relevance": ["relevance", "context_relevance"],
        # A separate row, not folded into "Relevance" above: context relevance
        # ("relevance") is always present once an answer is produced, so a lookup
        # that tried "answer_relevance" only as a fallback in that same list would
        # never actually reach it. answer_relevance is None whenever
        # Config.answer_relevance is off (the default) or back-generation was
        # unavailable/failed -- the isinstance check below already renders that
        # as "n/a", same as every other undefined metric here.
        "Answer Relevance": ["answer_relevance"],
        "Conciseness": ["conciseness"],
    }
    faithfulness_result = getattr(res, "faithfulness", None)

    scorecard_rows = []
    for label, keys in aliases.items():
        value = None
        for k in keys:
            if k in metrics:
                value = metrics[k]
                break
        if value is None and label == "Faithfulness" and faithfulness_result is not None:
            value = getattr(faithfulness_result, "score", None)
        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        scorecard_rows.append([label, value_str])

    t_arith = "n/a"
    t_geom = "n/a"
    try:
        from ragtrust.metrics.aggregate import aggregate_arithmetic, aggregate_geometric

        # Equal weights over the four headline dimensions; both aggregate_* functions
        # silently ignore any weight key absent from `metrics`, so this is safe even if
        # the pipeline's metric dict keys drift.
        weights = {"faithfulness": 0.25, "attribution_f1": 0.25, "relevance": 0.25, "conciseness": 0.25}
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric_metrics:
            t_arith = f"{aggregate_arithmetic(numeric_metrics, weights):.4f}"
            t_geom = f"{aggregate_geometric(numeric_metrics, weights):.4f}"
    except Exception as e:
        t_arith = f"n/a ({e.__class__.__name__})"
        t_geom = t_arith

    aggregate_md = f"**T_arith** (compensatory): `{t_arith}`&nbsp;&nbsp;&nbsp; **T_geom** (non-compensatory): `{t_geom}`"

    answer_md = f"### Answer\n\n{answer}\n\n---\n\n{aggregate_md}"

    return (
        answer_md,
        passage_rows,
        scorecard_rows,
        f"{len(claims)} claim(s) decomposed from the answer.",
        gr.update(visible=abstained),
    )


def _render_fallback_result(item: dict) -> tuple:
    answer = item.get("answer", "")
    answer_md = f"### Answer (cached, not live)\n\n{answer}"
    scorecard_rows = [
        ["Faithfulness", "n/a (live pipeline unavailable)"],
        ["Explainability / Attribution", "n/a (live pipeline unavailable)"],
        ["Relevance", "n/a (live pipeline unavailable)"],
        ["Conciseness", "n/a (live pipeline unavailable)"],
    ]
    return (answer_md, [], scorecard_rows, "Cached reference data has no passage/citation detail.", gr.update(visible=False))


def ask_handler(question: str, progress=gr.Progress()):
    question = (question or "").strip()
    if not question:
        gr.Warning("Please enter or select a question first.")
        return _empty_ask_outputs()

    try:
        progress(0.05, desc="Loading models (first call can take a minute)...")
        pipe = _get_pipeline()
        progress(0.6, desc="Retrieving passages and generating an answer...")
        res = pipe.answer(question)
        progress(1.0, desc="Done")
        gr.Info("Answer generated by the live pipeline.")
        return _render_live_result(res, pipe=pipe, question=question)
    except Exception as e:
        traceback.print_exc()
        gr.Warning(
            f"Live pipeline unavailable right now ({e.__class__.__name__}: {e}). "
            "Falling back to a cached answer for this question, if any."
        )
        item = _find_cached_answer(question)
        if item is None:
            gr.Warning("No cached answer exists for this exact question either.")
            return _empty_ask_outputs(
                note="No live pipeline and no cached answer for this question. "
                "Try one of the preset questions from the dropdown."
            )
        return _render_fallback_result(item)


def on_dropdown_select(choice: str):
    return _strip_choice_prefix(choice)


# --------------------------------------------------------------------------------------
# Tab 2: About
# --------------------------------------------------------------------------------------
ABOUT_MD_TEMPLATE = f"""
## RAG Trustworthiness -- Industrial

A retrieval-augmented generation system that ships trustworthiness metrics alongside
every answer: faithfulness, attribution, relevance, and conciseness, computed from a
shared claim x passage NLI matrix.

- **Repository:** [{REPO_URL}]({REPO_URL})

### Backends in this Space
- Generator: `{_generator_backend_name()}` (set via `RAGTRUST_GENERATOR` env var)
- NLI model: `{_nli_model_name()}` (set via `RAGTRUST_NLI_MODEL` env var)

### Layers
1. **Retrieval-augmented generation** -- FAISS + sentence-transformers retrieval, with a
   pluggable generator (Ollama / HF Inference API / cached answers for this Space).
2. **Trustworthiness evaluation** -- a shared claim x passage NLI matrix drives
   faithfulness, contradiction, attribution, relevance, and conciseness.
3. **Abstention gate** -- refuses to answer when nothing in the corpus supports a query.

See `ARCHITECTURE.md` and `METRICS.md` in the repository root for the full technical
writeup.
"""


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Trustworthiness") as demo:
        gr.Markdown("# RAG Trustworthiness -- Industrial\nA RAG system that ships trustworthiness metrics.")

        with gr.Tabs():
            # ---------------- Ask ----------------
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column(scale=1):
                        question_dropdown = gr.Dropdown(
                            choices=_question_choices(),
                            label="Preset questions (cached, from data/cached_answers.json)",
                        )
                        question_box = gr.Textbox(
                            label="Question", placeholder="Ask about the robotics corpus...", lines=2
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                    with gr.Column(scale=2):
                        abstain_banner = gr.Markdown(
                            "## The pipeline abstained\nNothing in the corpus supports this question "
                            "confidently enough to answer.",
                            visible=False,
                        )
                        answer_out = gr.Markdown(label="Answer")

                gr.Markdown("### Retrieved passages")
                passages_out = gr.Dataframe(
                    headers=["id", "score", "text (truncated)", "cited by"],
                    datatype=["number", "number", "str", "str"],
                    row_count=0,
                    column_count=4,
                    interactive=False,
                )

                gr.Markdown("### Trust metrics")
                scorecard_out = gr.Dataframe(
                    headers=["Metric", "Score"],
                    datatype=["str", "str"],
                    row_count=4,
                    column_count=2,
                    interactive=False,
                )
                claims_note = gr.Markdown("")

                question_dropdown.change(on_dropdown_select, inputs=question_dropdown, outputs=question_box)
                ask_btn.click(
                    ask_handler,
                    inputs=question_box,
                    outputs=[answer_out, passages_out, scorecard_out, claims_note, abstain_banner],
                )

            # ---------------- About ----------------
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD_TEMPLATE)

    return demo


if __name__ == "__main__":
    build_demo().launch()
