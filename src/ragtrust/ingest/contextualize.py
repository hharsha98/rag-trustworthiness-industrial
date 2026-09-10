"""Contextual Retrieval (Anthropic's method): prepend an LLM-written situating
blurb to each chunk before it is embedded/BM25-indexed.

Why this exists. A chunk taken out of its document routinely loses the entity,
section heading, or date that makes it findable -- e.g. a chunk that says "the
model improved throughput by 20%" says nothing about *which* model unless the
reader has the surrounding document. Prepending a one-sentence blurb that
names what the chunk is part of measurably improves both dense and BM25
retrieval, because the missing context becomes indexable text instead of only
being present two paragraphs away.

*** THE INVARIANT THIS MODULE MUST NOT VIOLATE ***
The blurb is LLM-GENERATED text -- the model can invent detail, get an entity
wrong, or hallucinate a date. It is a retrieval optimisation, never a fact.
This module returns `text` (contextualised, for embedding/BM25) and
`source_text` (the untouched original) as two separate fields specifically so
that nothing downstream of retrieval -- the NLI premise, the generator prompt,
citations -- can be handed the blurb by accident. Enforcing that separation is
`pipeline.py::answer()`'s job (see the comment at its `replace(p, ...)` line);
this module's job is only to make sure the two fields are correct and never
silently collapsed into one.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONTEXT_MAX_WORDS = 60

# Stripped if the model prepends it despite instructions -- small local models
# routinely ignore "output only the sentence" and add a preamble anyway.
_PREAMBLE_PREFIXES = (
    "context:",
    "this chunk",
    "this passage",
    "this section",
    "here is",
    "here's",
    "the following",
)


def build_context_prompt(document_text: str, chunk_text: str, max_document_chars: int = 12000) -> str:
    """Prompt asking for a single situating sentence for `chunk_text` within
    `document_text`.

    `document_text` is truncated to `max_document_chars` rather than sent in
    full: a long document would blow the context window of a small local
    model (this targets llama3.2:3b-class backends), and the situating
    information a chunk needs -- title, section headings, the entities the
    document is about -- is overwhelmingly concentrated near the top of the
    document, which truncation preserves.
    """
    truncated = (document_text or "")[:max_document_chars]
    return (
        "Here is a document:\n"
        f"{truncated}\n\n"
        "Here is a chunk from that document:\n"
        f"{chunk_text}\n\n"
        "Write a single short sentence (max 60 words) that situates this chunk "
        "within the overall document, so the chunk can be correctly identified "
        "when searched for on its own. State the section or topic the chunk "
        "belongs to, and name any entity or date the chunk refers to only "
        "implicitly (e.g. via a pronoun or an unstated subject).\n\n"
        "Answer with ONLY that sentence. No preamble, no quotation marks, no "
        "\"Context:\" prefix, no restating of these instructions."
    )


def _clean_blurb(raw: str) -> str:
    """Trim to `CONTEXT_MAX_WORDS` and strip a leading preamble the model adds
    despite `build_context_prompt` telling it not to."""
    text = (raw or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix in _PREAMBLE_PREFIXES:
        if lowered.startswith(prefix):
            # Drop everything up to and including the first ':' if present,
            # otherwise leave it -- most preambles observed in practice are
            # "Context: <sentence>" or "This chunk discusses <sentence>".
            if ":" in text[:40]:
                text = text.split(":", 1)[1].strip()
            break
    words = text.split()
    if len(words) > CONTEXT_MAX_WORDS:
        text = " ".join(words[:CONTEXT_MAX_WORDS])
    return text


def _cache_key(model_tag: str, chunk_text: str) -> str:
    return hashlib.sha256(f"{model_tag}\x00{chunk_text}".encode("utf-8")).hexdigest()


def _load_cache(cache_dir: str) -> dict:
    path = Path(cache_dir) / "contextualize_cache.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # A corrupt/partial cache file must not take down indexing -- treat it
        # as empty and let it be rewritten below.
        return {}


def _save_cache(cache_dir: str, cache: dict) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "contextualize_cache.json").write_text(json.dumps(cache))


def contextualize_chunks(chunks: list, document_text: str, generator, *,
                          workers: int = 4, cache_dir: str = None,
                          model_tag: str = "") -> list:
    """Prepend an LLM-written situating blurb to each chunk.

    Returns `[{"page": int, "text": "<blurb> <original>", "source_text": "<original>"}, ...]`,
    same order and length as `chunks`. `text` is for embedding/BM25 only;
    `source_text` is the original chunk and is what everything downstream of
    retrieval must use (see the module docstring).

    Degrades, never crashes: if `generator` has no `complete`, or a call
    raises, or returns empty text, that chunk falls back to
    `text == source_text` (uncontextualised) and processing continues.
    Contextualisation is a retrieval optimisation on top of a corpus that must
    remain indexable even when the LLM backend is unavailable or flaky for a
    subset of chunks -- one bad chunk must not fail the whole index build.
    """
    complete = getattr(generator, "complete", None)
    n = len(chunks)
    results: list = [None] * n
    failure_count = 0

    if not callable(complete):
        # No LLM call available at all -- every chunk passes through unchanged.
        # Logged once below (a single warning naming the whole corpus) rather
        # than per-chunk, since this is the expected, common case for a
        # generator backend that never implemented `complete`.
        for i, c in enumerate(chunks):
            results[i] = {"page": c["page"], "text": c["text"], "source_text": c["text"]}
        warnings.warn(
            f"Generator has no complete() method; {n} chunk(s) indexed without "
            "contextualisation."
        )
        return results

    cache = _load_cache(cache_dir) if cache_dir else {}
    cache_dirty = False

    def _contextualize_one(i: int) -> dict:
        nonlocal cache_dirty
        chunk = chunks[i]
        chunk_text = chunk["text"]
        key = _cache_key(model_tag, chunk_text) if cache_dir else None
        if key is not None and key in cache:
            blurb = cache[key]
        else:
            try:
                prompt = build_context_prompt(document_text, chunk_text)
                raw = complete(prompt)
                blurb = _clean_blurb(raw)
            except Exception:
                blurb = ""
            if key is not None:
                cache[key] = blurb
                cache_dirty = True

        if not blurb:
            return {"page": chunk["page"], "text": chunk_text, "source_text": chunk_text}
        return {
            "page": chunk["page"],
            "text": f"{blurb} {chunk_text}",
            "source_text": chunk_text,
        }

    # `executor.map` preserves input order in its output order regardless of
    # completion order, unlike `as_completed` -- required here because passage
    # ids downstream (citations, metadata, FAISS rows) are positional indices
    # into `passages_text`, so contextualized chunk i MUST land at result[i].
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, result in zip(range(n), executor.map(_contextualize_one, range(n))):
            results[i] = result
            if result["text"] == result["source_text"]:
                failure_count += 1

    if cache_dir and cache_dirty:
        _save_cache(cache_dir, cache)

    if failure_count:
        # One aggregate warning, not one per chunk -- a flaky backend failing
        # on many chunks would otherwise flood the log for a non-fatal,
        # already-handled degradation.
        warnings.warn(
            f"{failure_count}/{n} chunk(s) fell back to uncontextualised text "
            "(no blurb generated or generation failed)."
        )

    return results
