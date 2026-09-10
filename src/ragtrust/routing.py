"""Query router for trust-gated agentic retrieval.

Decides, BEFORE spending a single generation call, whether a query is worth
answering at all, answerable confidently in one retrieval pass, or borderline
enough that the extra generation calls of `agentic.answer_iterative`'s
iterative loop are worth paying for.
"""
from __future__ import annotations

from .metrics.relevance import max_context_similarity

ROUTE_ABSTAIN = "abstain"
ROUTE_SINGLE = "single"
ROUTE_ITERATIVE = "iterative"


def route(pipeline, query: str, iterate_margin: float = 0.15) -> tuple:
    """Classify `query` using `max_context_similarity` over the top-`config.k`
    retrieved passages.

    This reuses `Config.retrieval_gate` rather than introducing a new,
    uncalibrated heuristic threshold. `retrieval_gate` is already calibrated
    on third-party data -- experiments/09_gate_calibration.py, ROC-AUC 0.962
    pooled across easy/medium/hard negative tiers (see Config.retrieval_gate's
    docstring) -- to answer exactly the question this router needs answered
    first: "is anything in the corpus plausibly about this query at all?" The
    router is reusing an existing calibrated measurement, not guessing a
    second one.

    - similarity below `retrieval_gate` -> ROUTE_ABSTAIN. This is the SAME
      condition `answer_with`'s Gate 1 would fire on, so there is no point
      spending a generation call (or a reformulation call) to find that out
      the slow way -- the caller can skip straight to declining.
    - similarity at or above `retrieval_gate + iterate_margin` -> ROUTE_SINGLE.
      Comfortably past the gate: a single `answer()`/`answer_with()` pass is
      expected to be enough, so the extra cost of iterating is not worth it.
    - otherwise (the band between the two) -> ROUTE_ITERATIVE. Close enough to
      the gate that a single pass is a coin flip; this is exactly the regime
      `answer_iterative` exists for, where a reformulated retrieval query has
      a real chance of pulling the measured trust up past threshold.

    Returns `(route, top_similarity)` so a caller can log or display the score
    that drove the decision without recomputing it.
    """
    retrieved = pipeline.retriever.search(query, pipeline.config.k)
    passage_texts = [pipeline.source_text(p.id) for p in retrieved]
    top = max_context_similarity(query, passage_texts, pipeline.embedder)

    if top < pipeline.config.retrieval_gate:
        return ROUTE_ABSTAIN, top
    if top >= pipeline.config.retrieval_gate + iterate_margin:
        return ROUTE_SINGLE, top
    return ROUTE_ITERATIVE, top
