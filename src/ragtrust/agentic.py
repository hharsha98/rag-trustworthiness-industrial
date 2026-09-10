"""Trust-gated agentic retrieval: an iterative retrieval loop whose stopping
criterion is this repository's own calibrated trust measurement, not an LLM
self-assessment.

Agentic RAG loops retrieval and decides when to stop. In almost every
implementation the stopping criterion is an LLM asked whether it is
satisfied -- a model grading its own work. Nothing about that self-grading is
calibrated: it is a second opinion from the same kind of model that produced
the answer, on the same evidence, with no measured relationship to whether
the answer is actually grounded. This repository already has a calibrated
measurement of answer quality -- `AnswerResult.trust["geometric"]`, the
non-compensatory aggregate `answer_with()` computes from faithfulness,
attribution, relevance and conciseness (see METRICS.md and
`Config.abstain_threshold`'s calibration comment: ROC-AUC 0.962 pooled on
BEIR/SciFact). `answer_iterative` below uses THAT as the loop controller
instead: keep retrieving only while the measured trust of the best answer so
far is below threshold, stop the moment it clears, and never let the loop
report an answer that never cleared it. Substituting a calibrated measurement
for a self-assessment is the entire point of this module -- keep that
substitution visible in the code below, not just in this docstring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .metrics.relevance import max_context_similarity
from .pipeline import AnswerResult, RAGTrustPipeline

# Explicitly asks for a REFORMULATION, not an answer and not a broader
# question -- see `_reformulate`'s docstring for why both of those failure
# modes matter here.
_REFORMULATE_PROMPT = (
    "Rewrite the following search query using different wording, synonyms, or "
    "phrasing that could surface passages the original wording might have "
    "missed. Preserve the exact same information need -- do not broaden it "
    "into a different question, and do not answer it.\n\n"
    "Output ONLY the rewritten query, nothing else.\n\n"
    "Query: {query}\nRewritten query:"
)


@dataclass(frozen=True)
class Round:
    """One retrieval-and-score pass of `answer_iterative`'s loop."""

    index: int
    # The retrieval query actually used this round -- round 1 is always the
    # original question verbatim; later rounds are `_reformulate`'s output.
    query: str
    top_similarity: float
    n_passages: int  # size of the accumulated passage pool after this round
    trust: float  # geometric aggregate, or 0.0 when this round's answer abstained
    abstained: bool
    stop_reason: str = ""  # "" while the loop is still continuing past this round


@dataclass(frozen=True)
class IterativeResult:
    result: AnswerResult  # the FINAL answer -- see `answer_iterative`'s honesty comment
    rounds: list
    llm_calls: int

    def to_dict(self) -> dict:
        return {
            "result": self.result.to_dict(),
            "rounds": [asdict(r) for r in self.rounds],
            "llm_calls": self.llm_calls,
        }


def _reformulate(generator, query: str) -> tuple:
    """Ask `generator` to rewrite `query` for a second retrieval pass.

    Returns `(new_query_or_None, calls_spent)`. Degrades to `(None, ...)` on
    ANY failure -- no `complete` method, an exception, or output that is
    empty or byte-identical to `query` -- rather than raising. This mirrors
    `ingest/contextualize.py`'s graceful-degradation pattern: contextualisation
    at index time falls back to the uncontextualised chunk rather than failing
    the index build, and reformulation at query time falls back to stopping
    the loop rather than failing the answer. `calls_spent` is 1 whenever
    `complete` existed and was actually invoked (even if it raised or came
    back unusable -- the call was attempted and its cost incurred), and 0 when
    there was no `complete` method to call at all.
    """
    complete = getattr(generator, "complete", None)
    if not callable(complete):
        return None, 0
    try:
        rewritten = complete(_REFORMULATE_PROMPT.format(query=query))
    except Exception:
        return None, 1
    rewritten = (rewritten or "").strip()
    if not rewritten or rewritten == query:
        return None, 1
    return rewritten, 1


def answer_iterative(pipeline: RAGTrustPipeline, question: str, max_rounds: int = 3) -> IterativeResult:
    """Retrieve, score, and -- if the measured trust of the best answer so far
    is below `Config.abstain_threshold` -- reformulate the retrieval query and
    try again, up to `max_rounds` times.

    The ORIGINAL `question` is always what gets answered and scored every
    round (`pipeline.answer_with(question, pooled)`); only the *retrieval*
    query is ever reformulated. Scoring a reformulated question would measure
    whether the system answered a question the user never asked -- the trust
    score exists to certify an answer to THIS question, not to some rewording
    of it that happened to retrieve better.
    """
    if pipeline._generator is None:
        raise ValueError("A generator must be provided to produce an answer.")
    if not pipeline.passages_text:
        raise ValueError("Nothing indexed. Call index_corpus/index_dir/load first.")

    rounds: list = []
    llm_calls = 0
    # Passage id -> Passage, in the order each id was FIRST seen. A later
    # round's hit for an id already in the pool is dropped rather than
    # overwriting it, so this doubles as "preserve best rank": round 1 runs
    # against the original question, which is the retrieval query most likely
    # to rank a passage well, so its ranking for a given id is kept over a
    # later reformulation's ranking for the same id.
    pooled: dict = {}
    best_result = None
    best_trust = -1.0
    current_query = question

    for i in range(1, max_rounds + 1):
        round_hits = pipeline.retriever.search(current_query, pipeline.config.k)
        new_ids = [p.id for p in round_hits if p.id not in pooled]

        if i > 1 and not new_ids:
            # This round's reformulation retrieved nothing the pool didn't
            # already have. Stop here WITHOUT calling `answer_with` -- scoring
            # an identical pool again would just burn a generation call to
            # reproduce the previous round's number. Replay that round's
            # top_similarity/trust/abstained since nothing was recomputed.
            prev = rounds[-1]
            rounds.append(Round(
                index=i, query=current_query, top_similarity=prev.top_similarity,
                n_passages=len(pooled), trust=prev.trust, abstained=prev.abstained,
                stop_reason="no_new_passages",
            ))
            break

        # Union, not replace -- see the module docstring on `pooled` above.
        # A later reformulation can retrieve WORSE passages than the original
        # question did (a rewording can drift off-topic just as easily as it
        # can surface a synonym match); discarding round 1's passages in favour
        # of round 2's would let the loop lose ground it had already gained,
        # rather than only ever accumulating more evidence to score against.
        for p in round_hits:
            if p.id not in pooled:
                pooled[p.id] = p
        pooled_list = list(pooled.values())

        result = pipeline.answer_with(question, pooled_list)

        # Did this round actually spend a generation call? `answer_with` can
        # abstain at Gate 1 (relevance) BEFORE ever calling `generate()` -- see
        # pipeline.py. `"faithfulness"` only enters `result.metrics` once claims
        # have been split and NLI-scored, i.e. strictly after `generate()` ran,
        # so its presence is a reliable signal of a spent generation call
        # without coupling to `abstain_reason`'s wording or wrapping the
        # generator to count its own calls.
        if "faithfulness" in result.metrics:
            llm_calls += 1

        trust = 0.0 if result.abstained else result.trust.get("geometric", 0.0)
        # Recomputed independently of `answer_with`'s internal gate value (which
        # is not returned on `AnswerResult`) using the exact same calibrated
        # signal and the same source-text swap the gate itself uses, so this
        # number reports what the gate actually saw, not an approximation of it.
        source_texts = [pipeline.source_text(p.id) for p in pooled_list]
        top_similarity = max_context_similarity(question, source_texts, pipeline.embedder)

        if best_result is None or trust > best_trust:
            best_result, best_trust = result, trust

        if trust >= pipeline.config.abstain_threshold:
            rounds.append(Round(
                index=i, query=current_query, top_similarity=top_similarity,
                n_passages=len(pooled_list), trust=trust, abstained=result.abstained,
                stop_reason="trust_threshold_met",
            ))
            break

        if i == max_rounds:
            rounds.append(Round(
                index=i, query=current_query, top_similarity=top_similarity,
                n_passages=len(pooled_list), trust=trust, abstained=result.abstained,
                stop_reason="max_rounds",
            ))
            break

        new_query, calls_spent = _reformulate(pipeline._generator, current_query)
        llm_calls += calls_spent
        if new_query is None:
            rounds.append(Round(
                index=i, query=current_query, top_similarity=top_similarity,
                n_passages=len(pooled_list), trust=trust, abstained=result.abstained,
                stop_reason="reformulation_unavailable",
            ))
            break

        rounds.append(Round(
            index=i, query=current_query, top_similarity=top_similarity,
            n_passages=len(pooled_list), trust=trust, abstained=result.abstained,
            stop_reason="",
        ))
        current_query = new_query

    # *** THE HONESTY PROPERTY THIS LOOP MUST NOT VIOLATE ***
    # The returned answer is the HIGHEST-trust round, not simply the last one --
    # a worse reformulation running last must never bump a better earlier round
    # out of the result. And if MULTIPLE rounds ran and NONE of their trust
    # ever cleared `config.abstain_threshold`, the returned result MUST be an
    # abstention, even if `answer_with` itself judged some round's per-claim
    # support good enough to answer (its Gate 2 checks only the single best
    # claim; the geometric aggregate here is the stricter, non-compensatory,
    # whole-answer measurement -- see METRICS.md). Iterating must never be able
    # to let a later, no-better round of comparison dress up a bad answer as
    # the "winner" and present it as trustworthy: that is the entire safety
    # property that lets this loop substitute a calibrated measurement for an
    # LLM's self-assessment and still be trusted. If this branch is ever
    # changed to return `best_result` unconditionally whenever len(rounds) > 1,
    # that property is gone.
    #
    # The rule applies uniformly, including when only ONE round ran, and that is
    # deliberate: it makes `answer_iterative` stricter than `pipeline.answer()`
    # rather than equivalent to it at max_rounds=1. The two have different
    # contracts. `answer()` is single-shot and gates on Gate 2's per-claim check,
    # returning low-aggregate answers for callers to judge via `is_trustworthy`.
    # `answer_iterative` promises something narrower -- "I retrieve until the
    # measured trust clears the threshold, and say so when it never did" -- and a
    # controller that only engages once a second round happens would make that
    # promise conditional on round count. It would also mean an identical answer
    # reports `abstained=False` after one round and `abstained=True` after two,
    # so the outcome would depend on how many rounds ran alongside it rather than
    # on the answer itself. Callers wanting the looser single-shot semantics
    # should call `pipeline.answer()`, which is unchanged.
    if best_trust >= pipeline.config.abstain_threshold or best_result.abstained:
        final_result = best_result
    else:
        final_result = replace(
            best_result,
            abstained=True,
            answer="Not answerable from this corpus.",
            abstain_reason=(
                f"No round's answer reached the trust threshold across "
                f"{len(rounds)} round(s) (best geometric trust {best_trust:.3f} "
                f"< {pipeline.config.abstain_threshold})."
            ),
        )

    return IterativeResult(result=final_result, rounds=rounds, llm_calls=llm_calls)
