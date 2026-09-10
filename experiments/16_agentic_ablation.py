#!/usr/bin/env python3
"""Does trust-gated agentic retrieval help -- and what does it cost?

`agentic.py::answer_iterative` replaces the usual "ask an LLM if it's satisfied"
agentic stopping rule with this repository's own calibrated trust measurement
(`AnswerResult.trust["geometric"]`): keep retrieving/reformulating while the
best round's measured trust is below `Config.abstain_threshold`, stop the
moment it clears, and never report an answer that never cleared it.
`routing.py::route` decides, before spending a generation call, whether a
query is worth abstaining on outright, answering once, or paying for that
iterative loop. This script measures whether either of those actually helps,
against a THIRD-PARTY labelled query set -- not a self-authored one.

*** Dataset: reused, not invented (experiment 09's construction) ***
Positives (answerable): BEIR/SciFact's 300 judged test queries, scored against
the SciFact corpus (5,183 biomedical-claim-verification abstracts) they were
written to be answered from. Negatives (unanswerable): queries sampled from
three other BEIR sets scored against the SAME SciFact corpus -- BeIR/quora
(general questions), BeIR/fiqa (financial questions), BeIR/nfcorpus
(biomedical questions -- a NOISY negative tier, since some NFCorpus queries
may genuinely be answerable from SciFact; see experiments/09_gate_calibration.py's
module docstring for the same caveat). `--n-answerable`/`--n-unanswerable`
(default 100/100) sample deterministically with the repo's SEED (0), split
evenly across the three negative sources.

*** Three arms ***
    1. single     -- `pipeline.answer(q)`.
    2. routed     -- `routing.route(...)` decides per query: abstain (no
                      generation call at all), single (`pipeline.answer(q)`),
                      or iterative (`answer_iterative(q, max_rounds=
                      config.agentic_max_rounds)`).
    3. iterative  -- always `answer_iterative(q, max_rounds=
                      config.agentic_max_rounds)`.

*** Cost is a headline column, not a footnote ***
Every arm reports mean LLM calls/query, total LLM calls, and mean wall-clock
latency/query ALONGSIDE the quality metrics -- an arm that only changes cost
at equal quality is a real, reportable finding here (see the verdict logic in
`arm_verdict` below), not a null result to bury.

*** Metrics ***
Per arm, over the SAME fixed query set (so pairing against the `single`
baseline is valid throughout): answered-rate on answerable queries;
abstention-rate on unanswerable queries; abstention precision (of all
abstentions, fraction truly unanswerable); Youden's J = TPR - FPR for
abstention as an unanswerability detector, where TPR = abstention-rate on
unanswerable and FPR = 1 - answered-rate on answerable. J is used instead of
F1 because J does not depend on the answerable:unanswerable ratio in the
query set (only the two error RATES), whereas F1 shifts with prevalence even
when nothing about the underlying detector changed -- this repository already
made exactly that argument when choosing calibration objectives in
experiments/14_hagrid_calibration.py, and it applies identically here to
scoring abstention as a detector. Also: mean geometric trust on answered
queries; mean LLM calls/query (+ total); mean wall-clock latency/query.

Bootstrap 95% CIs (10,000 resamples) on every rate metric, PLUS paired
bootstrap CIs of each arm's difference from `single` for every metric that is
well-defined per-query over a FIXED subset (answered-rate over the answerable
subset, abstention-rate/abstention-precision over the union, LLM calls and
latency over every query) -- same queries scored by every arm, so pairing is
valid, exactly as experiments/08 and 15 already argue for their own paired
diffs. Youden's J is a linear combination of two rates computed over two
DISJOINT subsets (answerable vs unanswerable), so neither the repo's existing
`bootstrap_ci_mean` (one array) nor `paired_bootstrap_ci_diff` (one shared
index) can express its CI directly -- `bootstrap_ci_sum_of_two_means` below is
the minimal extension that reuses their exact resampling pattern (seeded
`numpy.random.Generator`, integers-with-replacement, quantile at alpha/1-
alpha) for a statistic built from two independent subset means instead of one.
Bootstrap machinery -- `bootstrap_ci_mean`, `paired_bootstrap_ci_diff`,
`N_BOOT`, `CI`, `SEED`, and the BEIR/SciFact loader -- is imported from
experiments/15_contextual_ablation.py (which itself re-exports experiment
08's versions unchanged), not reimplemented; this script only adds the two
ratio/two-subset variants above that the existing helpers structurally cannot
express.

*** Honesty requirements (the point of this script) ***
`answer_iterative` is deliberately STRICTER than `pipeline.answer()`: it
forces abstention whenever no round's geometric trust clears
`abstain_threshold`, even at max_rounds=1 (see agentic.py's own comment on
this). So the `iterative` arm is EXPECTED to abstain more, including on some
answerable queries the `single` arm would have answered. This script prints
answered-rate(answerable) and abstention-rate(unanswerable) side by side in
one table, every time, so that trade-off is visible together -- never only
the favourable half. Verdicts state plainly whether an arm HELPS, HURTS, or
is NOT AN IMPROVEMENT (a paired CI including zero is reported as exactly
that phrase), and separate the quality verdict from the cost verdict: if an
arm's only real effect is fewer LLM calls at statistically indistinguishable
quality, the verdict says exactly that rather than dressing up a cost win as
a quality win. No thresholds here are tuned to make a particular arm look
good, and no query is dropped from the set any arm is scored against.

*** Generators: never Ollama in this run ***
`--generator {stub,ollama}`, default `stub`. The Ollama server this repo
talks to is (per this run's operator) saturated by another long benchmark
right now, so this script defaults to, and this run always uses, a
zero-network `StubGenerator` (see its class docstring) that still exercises
every real code path in `agentic.py`/`routing.py`/`pipeline.py` -- reformulation,
passage pooling, every stop reason, and llm_calls counting -- deterministically.
The `--generator ollama` option exists for a later run once the server is
free; it is never invoked by the run this script's own docstring reports.
Embeddings (`sentence-transformers/msmarco-distilbert-base-v4`) and NLI
(`roberta-large-mnli`) ARE real models -- both already sit in the local
Hugging Face cache alongside the BEIR datasets this script needs, so building
the corpus index and scoring faithfulness/attribution needs no network access
either; see `main()`'s use of `local_files_only=True` throughout.

Usage:
    python experiments/16_agentic_ablation.py                               # full run: 100/100
    python experiments/16_agentic_ablation.py --n-answerable 20 --n-unanswerable 20   # smoke run
    python experiments/16_agentic_ablation.py --generator ollama            # NOT used by this run

Exit code: always 0 (a measurement, not a pass/fail gate), except 1 if the
BEIR data this script needs is not already in the local Hugging Face cache
(no network access is attempted to fill the gap -- see module docstring).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch/faiss imports

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtrust.agentic import answer_iterative  # noqa: E402
from ragtrust.config import Config  # noqa: E402
from ragtrust.generation.base import GeneratedAnswer  # noqa: E402
from ragtrust.ingest.loader import segment_sentences  # noqa: E402
from ragtrust.metrics.faithfulness import FaithfulnessResult  # noqa: E402
from ragtrust.pipeline import AnswerResult, RAGTrustPipeline  # noqa: E402
from ragtrust.routing import ROUTE_ABSTAIN, ROUTE_ITERATIVE, ROUTE_SINGLE, route  # noqa: E402

CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"

NEGATIVE_SOURCES = {
    "quora": "BeIR/quora",
    "fiqa": "BeIR/fiqa",
    "nfcorpus": "BeIR/nfcorpus",
}
TIER_ORDER = ["quora", "fiqa", "nfcorpus"]


def _load_experiment_module(name: str, filename: str):
    """Load `experiments/<filename>` by file path -- its module name starts
    with a digit, so it is not importable normally (mirrors experiments/09
    and /15's identical helper of the same purpose). Loading experiment 15
    this way triggers no network/model access; see that module's own
    docstring ("Reuse, not reimplementation")."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_contextual15 = _load_experiment_module("_contextual_ablation_16dep", "15_contextual_ablation.py")

load_scifact = _contextual15.load_scifact
qrels_to_lookup = _contextual15.qrels_to_lookup
build_retriever = _contextual15.build_retriever
bootstrap_ci_mean = _contextual15.bootstrap_ci_mean
paired_bootstrap_ci_diff = _contextual15.paired_bootstrap_ci_diff
N_BOOT = _contextual15.N_BOOT
CI = _contextual15.CI
SEED = _contextual15.SEED
DEFAULT_EMBED_MODEL = _contextual15.DEFAULT_EMBED_MODEL


# ============================================================================
# Bootstrap helpers this script adds -- see module docstring's "Bootstrap
# machinery" paragraph for why the repo's existing `bootstrap_ci_mean` /
# `paired_bootstrap_ci_diff` cannot express these two statistics directly.
# ============================================================================


def bootstrap_ci_sum_of_two_means(arr_a: np.ndarray, arr_b: np.ndarray,
                                   n_boot: int = N_BOOT, ci: float = CI, seed: int = SEED) -> tuple:
    """CI on mean(arr_a) + mean(arr_b), resampling `arr_a` and `arr_b`
    INDEPENDENTLY each iteration. `arr_a`/`arr_b` are two DISJOINT query
    subsets (e.g. answerable vs unanswerable) with no shared index to pair
    on -- used for Youden's J (= answered_rate[answerable] +
    abstention_rate[unanswerable] - 1) and its paired diff-from-single
    (the "-1" cancels in a difference of two J's, so callers computing a
    diff pass this function's output straight through unadjusted; callers
    computing one arm's own J subtract 1 from both the point estimate and
    both CI bounds themselves)."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(arr_a), len(arr_b)
    sums = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)
        sums[i] = arr_a[ia].mean() + arr_b[ib].mean()
    alpha = (1 - ci) / 2
    return float(np.quantile(sums, alpha)), float(np.quantile(sums, 1 - alpha))


def _ratio_point(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    d = int(denominator_mask.sum())
    if d == 0:
        return float("nan")
    return float((numerator_mask & denominator_mask).sum()) / float(d)


def bootstrap_ci_ratio(numerator_mask: np.ndarray, denominator_mask: np.ndarray,
                        n_boot: int = N_BOOT, ci: float = CI, seed: int = SEED) -> tuple:
    """CI on sum(numerator_mask & denominator_mask) / sum(denominator_mask) --
    e.g. abstention precision = P(truly unanswerable | abstained). Resamples
    QUERY INDICES jointly (both masks move together per query) because the
    denominator itself is random under resampling; `bootstrap_ci_mean`
    assumes a fixed-size array of independent per-query values and cannot
    express a ratio whose denominator changes with the resample. Iterations
    where no resampled query satisfies `denominator_mask` are skipped
    (precision is undefined there, not 0)."""
    rng = np.random.default_rng(seed)
    n = len(denominator_mask)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = denominator_mask[idx]
        d = int(denom.sum())
        if d == 0:
            continue
        vals.append(float((numerator_mask[idx] & denom).sum()) / float(d))
    if not vals:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))


def paired_bootstrap_ci_diff_ratio(denom_arm: np.ndarray, denom_single: np.ndarray,
                                    numerator_mask: np.ndarray, n_boot: int = N_BOOT,
                                    ci: float = CI, seed: int = SEED) -> tuple:
    """Paired CI on (precision_arm - precision_single), resampling the SAME
    query indices for both arms each iteration -- both arms scored the same
    queries, so pairing is valid (mirrors `paired_bootstrap_ci_diff`'s
    contract). Iterations where EITHER arm's resampled denominator is 0 are
    skipped."""
    rng = np.random.default_rng(seed)
    n = len(numerator_mask)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        num = numerator_mask[idx]
        da, ds = denom_arm[idx], denom_single[idx]
        sa, ss = int(da.sum()), int(ds.sum())
        if sa == 0 or ss == 0:
            continue
        diffs.append(float((num & da).sum()) / float(sa) - float((num & ds).sum()) / float(ss))
    if not diffs:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))


# ============================================================================
# Dataset -- reused from experiment 09's construction, not reinvented. See
# module docstring's "Dataset" section.
# ============================================================================


def _split_evenly(n: int, k: int) -> list:
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def load_negative_queries(repo_id: str, n: int, seed: int) -> list:
    """Sample `n` query texts (fixed seed) from a BEIR dataset's queries
    split, entirely from the local Hugging Face cache (`local_files_only=
    True`) -- this script must never attempt a download (module docstring's
    HARD CONSTRAINT). experiment 09's `load_negative_queries` has no such
    guard, which is why this is a small local variant rather than an import
    of that one."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = hf_hub_download(repo_id, "queries/queries-00000-of-00001.parquet",
                                repo_type="dataset", local_files_only=True)
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"{repo_id} queries are not in the local Hugging Face cache and "
            f"downloads are disabled for this run (module docstring: never "
            f"download). Pre-populate the cache once with network access, or "
            f"reduce --n-unanswerable to fit what IS cached."
        ) from e
    df = pd.read_parquet(path)
    texts = df["text"].dropna().astype(str).tolist()
    if n >= len(texts):
        return texts
    rng = random.Random(seed)
    return rng.sample(texts, n)


def build_query_set(n_answerable: int, n_unanswerable: int, seed: int) -> dict:
    """Returns {"queries": [...], "is_answerable": np.ndarray[bool],
    "source": [...], "corpus_texts": [...], "corpus_meta": [...]}.

    Positives: BEIR/SciFact's judged test queries (answerable against the
    SciFact corpus). Negatives: `n_unanswerable` queries split as evenly as
    possible across quora/fiqa/nfcorpus, all scored against the SAME SciFact
    corpus -- see module docstring."""
    corpus_df, queries_df, qrels_df = load_scifact(local_files_only=True)
    corpus_ids = corpus_df["_id"].astype(str).tolist()
    corpus_texts = [f"{title} {text}".strip()
                    for title, text in zip(corpus_df["title"], corpus_df["text"])]
    corpus_meta = [{"source": "BeIR/scifact", "doc_id": corpus_ids[i]} for i in range(len(corpus_texts))]

    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    qrels_lookup = qrels_to_lookup(qrels_df)
    positive_qids = sorted(qrels_lookup.keys())
    positive_queries = [queries_by_id[qid] for qid in positive_qids if qid in queries_by_id]

    used_answerable = min(n_answerable, len(positive_queries))
    if used_answerable < n_answerable:
        print(f"NOTE: requested --n-answerable {n_answerable}, but BEIR/SciFact "
              f"has only {len(positive_queries)} judged test queries. Using all "
              f"{used_answerable}.")
    answerable = random.Random(seed).sample(positive_queries, used_answerable)

    tier_counts = _split_evenly(n_unanswerable, len(TIER_ORDER))
    unanswerable: list = []
    unanswerable_source: list = []
    actual_tier_counts = {}
    for tier, want in zip(TIER_ORDER, tier_counts):
        repo_id = NEGATIVE_SOURCES[tier]
        got = load_negative_queries(repo_id, want, seed)
        if len(got) < want:
            print(f"NOTE: requested {want} negatives from {repo_id}, but only "
                  f"{len(got)} queries are available there. Using all {len(got)}.")
        unanswerable += got
        unanswerable_source += [tier] * len(got)
        actual_tier_counts[tier] = len(got)

    queries = answerable + unanswerable
    is_answerable = np.array([True] * len(answerable) + [False] * len(unanswerable), dtype=bool)
    source = ["scifact"] * len(answerable) + unanswerable_source

    return {
        "queries": queries, "is_answerable": is_answerable, "source": source,
        "corpus_texts": corpus_texts, "corpus_meta": corpus_meta,
        "n_answerable_available": len(positive_queries), "n_answerable_used": len(answerable),
        "n_unanswerable_used": len(unanswerable), "unanswerable_tier_counts": actual_tier_counts,
    }


# ============================================================================
# Generators
# ============================================================================


_REFORMULATE_QUERY_RE = re.compile(r"Query: (.*)\nRewritten query:", re.S)


def _first_sentence(passage) -> str:
    """First sentence of a `Passage` (or plain string), via `segment_sentences`
    (ingest/loader.py) -- no second sentence splitter is written here."""
    text = getattr(passage, "text", str(passage))
    sentences = segment_sentences(text)
    return sentences[0] if sentences else text.strip()


class StubGenerator:
    """Deterministic, zero-network generator (module docstring: "never Ollama
    in this run"). Exercises every real code path `agentic.py` cares about --
    reformulation, passage pooling, every stop reason -- without a model.

    `generate()`: quotes the TOP retrieved passage's first sentence with a
    correct `[1]` citation -- for MOST queries. This is well-grounded by
    construction (the claim is lifted verbatim from the passage the citation
    points at), so it clears `abstain_threshold` on round 1 for those queries,
    exercising the single-round path and "trust_threshold_met" at round 1.

    For queries in hash-bucket 1 of 3 (deterministic on the ORIGINAL question
    text, which `answer_with`/`answer_iterative` always pass to `generate()`
    regardless of round -- see agentic.py's module docstring), round 1
    instead quotes the LAST pooled passage's first sentence but still cites
    `[1]` (the top passage) -- an unsupported claim, which drives attribution
    (and therefore the geometric aggregate) toward 0 and forces the loop past
    round 1. From the SECOND time this generator is asked about that same
    question (i.e. round 2+), it switches back to the well-grounded top-
    passage citation, so those queries' loops can recover once reformulation
    has had a chance to pool a better passage -- exercising reformulation,
    pooling, and (when reformulated retrieval turns up nothing new) the
    "no_new_passages" stop reason too, all deterministically. See `main()`'s
    printed multi-round diagnostic for the empirical count this produces.

    `complete()` (reformulation, agentic.py's `_reformulate`): returns the
    prompt's embedded query with a fixed suffix appended -- guaranteed
    non-empty and never byte-identical to the input, satisfying
    `_reformulate`'s degrade-on-no-change contract, with zero network calls.
    """

    def __init__(self):
        self.generate_calls = 0
        self.complete_calls = 0
        self._seen: dict = {}

    def generate(self, query: str, passages: list) -> GeneratedAnswer:
        self.generate_calls += 1
        n_seen = self._seen.get(query, 0) + 1
        self._seen[query] = n_seen
        if not passages:
            return GeneratedAnswer(text="Not answerable from this corpus.", citations={})
        bucket = int(hashlib.sha1(query.encode()).hexdigest(), 16) % 3
        if bucket != 1 or n_seen >= 2 or len(passages) < 2:
            sentence = _first_sentence(passages[0])
        else:
            sentence = _first_sentence(passages[-1])  # weak: cites [1] but quotes a different passage
        return GeneratedAnswer(text=f"{sentence} [1]", citations={0: 0})

    def complete(self, prompt: str) -> str:
        self.complete_calls += 1
        m = _REFORMULATE_QUERY_RE.search(prompt)
        query = m.group(1).strip() if m else prompt.strip()
        return f"{query} -- alternate phrasing"


# ============================================================================
# Pipeline construction
# ============================================================================


def build_pipeline(config: Config, embed_model: str, corpus_texts: list, corpus_meta: list,
                    generator) -> RAGTrustPipeline:
    """Build a real `RAGTrustPipeline` over the SciFact corpus, reusing
    experiment 08/15's `CachedRetriever`-backed `build_retriever` (via the
    module-level `build_retriever` imported above) so the corpus embedding
    matrix already cached under data/benchmarks/ is loaded, not recomputed.
    `nli` is left unset -- `pipeline.nli` lazy-loads the real, already-locally
    -cached `roberta-large-mnli` on first use. This is the first experiment
    in this repo to run the full pipeline (retrieval + real NLI scoring)
    against BEIR/SciFact rather than only retrieval metrics."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embed_model)
    pipeline = RAGTrustPipeline(config, generator=generator, embedder=embedder)
    pipeline._retriever = build_retriever(
        config.retrieval_mode, config.rerank, embedder, CACHE_DIR,
        embed_model, config.rerank_model,
    )
    pipeline.index_texts(corpus_texts, meta=corpus_meta)
    return pipeline


def _route_abstain_result(config: Config, top_similarity: float) -> AnswerResult:
    """The abstention `routing.route` decides on, constructed WITHOUT a
    generation call -- `route()` itself only returns `(route, top_similarity)`,
    not a full `AnswerResult`, and there is deliberately no reason to spend a
    generation call just to build a display object for a query the router
    already decided not to answer. Shape mirrors `pipeline.py::_declined`."""
    return AnswerResult(
        answer="Not answerable from this corpus.",
        passages=[], claims=[], citations={}, metrics={},
        abstained=True,
        faithfulness=FaithfulnessResult(score=0.0, per_claim=[], contradiction_rate=0.0, support_index=[]),
        abstain_reason=(
            f"Routed to abstain before any generation call: best retrieved "
            f"similarity {top_similarity:.3f} < retrieval_gate {config.retrieval_gate}."
        ),
        trust={"arithmetic": 0.0, "geometric": 0.0, "weights": dict(config.weights)},
        sources={},
    )


# ============================================================================
# Arms -- each returns one dict per query with a uniform shape, so the
# aggregation code below never needs to know which arm produced a record.
# ============================================================================


def run_single(pipeline: RAGTrustPipeline, query: str) -> dict:
    t0 = time.perf_counter()
    result = pipeline.answer(query)
    latency = time.perf_counter() - t0
    # "faithfulness" only enters `result.metrics` once `generate()` actually
    # ran (Gate 1 can abstain before that) -- the same call-counting signal
    # `agentic.py::answer_iterative` itself uses, reused here rather than
    # invented a second way to ask the same question.
    llm_calls = 1 if "faithfulness" in result.metrics else 0
    return {"abstained": result.abstained, "trust": result.trust.get("geometric", 0.0),
            "llm_calls": llm_calls, "latency": latency, "route": None,
            "n_rounds": 1, "stop_reasons": []}


def run_iterative(pipeline: RAGTrustPipeline, query: str, max_rounds: int) -> dict:
    t0 = time.perf_counter()
    it = answer_iterative(pipeline, query, max_rounds=max_rounds)
    latency = time.perf_counter() - t0
    return {"abstained": it.result.abstained, "trust": it.result.trust.get("geometric", 0.0),
            "llm_calls": it.llm_calls, "latency": latency, "route": None,
            "n_rounds": len(it.rounds), "stop_reasons": [r.stop_reason for r in it.rounds]}


def run_routed(pipeline: RAGTrustPipeline, query: str, max_rounds: int, iterate_margin: float) -> dict:
    t0 = time.perf_counter()
    decision, top_sim = route(pipeline, query, iterate_margin=iterate_margin)
    if decision == ROUTE_ABSTAIN:
        result = _route_abstain_result(pipeline.config, top_sim)
        llm_calls, n_rounds, stop_reasons = 0, 0, []
    elif decision == ROUTE_SINGLE:
        result = pipeline.answer(query)
        llm_calls = 1 if "faithfulness" in result.metrics else 0
        n_rounds, stop_reasons = 1, []
    else:
        assert decision == ROUTE_ITERATIVE
        it = answer_iterative(pipeline, query, max_rounds=max_rounds)
        result = it.result
        llm_calls, n_rounds, stop_reasons = it.llm_calls, len(it.rounds), [r.stop_reason for r in it.rounds]
    latency = time.perf_counter() - t0
    return {"abstained": result.abstained, "trust": result.trust.get("geometric", 0.0),
            "llm_calls": llm_calls, "latency": latency, "route": decision,
            "n_rounds": n_rounds, "stop_reasons": stop_reasons}


# ============================================================================
# Aggregation
# ============================================================================


def _fmt_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "[undefined]"
    return f"[{lo:+.3f}, {hi:+.3f}]"


def compute_arm_metrics(records: list, is_answerable: np.ndarray) -> dict:
    abstained = np.array([r["abstained"] for r in records], dtype=bool)
    trust = np.array([r["trust"] for r in records], dtype=float)
    llm_calls = np.array([r["llm_calls"] for r in records], dtype=float)
    latency = np.array([r["latency"] for r in records], dtype=float)
    answered = ~abstained
    answerable_mask = is_answerable
    unanswerable_mask = ~is_answerable

    answered_indicator = answered[answerable_mask].astype(float)
    abstained_indicator = abstained[unanswerable_mask].astype(float)

    answered_rate = float(answered_indicator.mean()) if len(answered_indicator) else float("nan")
    answered_rate_ci = bootstrap_ci_mean(answered_indicator) if len(answered_indicator) else (float("nan"),) * 2
    abstention_rate = float(abstained_indicator.mean()) if len(abstained_indicator) else float("nan")
    abstention_rate_ci = bootstrap_ci_mean(abstained_indicator) if len(abstained_indicator) else (float("nan"),) * 2

    precision_point = _ratio_point(unanswerable_mask, abstained)
    precision_ci = bootstrap_ci_ratio(unanswerable_mask, abstained)

    j_point = answered_rate + abstention_rate - 1.0
    j_lo, j_hi = bootstrap_ci_sum_of_two_means(answered_indicator, abstained_indicator)
    j_ci = (j_lo - 1.0, j_hi - 1.0)

    trust_answered = trust[answered]
    mean_trust = float(trust_answered.mean()) if trust_answered.size else float("nan")
    mean_trust_ci = bootstrap_ci_mean(trust_answered) if trust_answered.size else (float("nan"),) * 2

    return {
        "n": len(records),
        "answered_rate_on_answerable": {"point": answered_rate, "ci95": list(answered_rate_ci)},
        "abstention_rate_on_unanswerable": {"point": abstention_rate, "ci95": list(abstention_rate_ci)},
        "abstention_precision": {"point": precision_point, "ci95": list(precision_ci)},
        "youdens_j": {"point": j_point, "ci95": list(j_ci)},
        "mean_geometric_trust_on_answered": {"point": mean_trust, "ci95": list(mean_trust_ci),
                                              "n_answered": int(answered.sum())},
        "mean_llm_calls_per_query": {"point": float(llm_calls.mean()),
                                      "ci95": list(bootstrap_ci_mean(llm_calls))},
        "total_llm_calls": int(llm_calls.sum()),
        "mean_latency_s_per_query": {"point": float(latency.mean()),
                                      "ci95": list(bootstrap_ci_mean(latency))},
    }


def compute_paired_diffs(records_arm: list, records_single: list, is_answerable: np.ndarray) -> dict:
    """Paired bootstrap 95% CIs of `records_arm` - `records_single`, index-
    for-index over the IDENTICAL query set both arms were scored against."""
    abst_arm = np.array([r["abstained"] for r in records_arm], dtype=bool)
    abst_single = np.array([r["abstained"] for r in records_single], dtype=bool)
    calls_arm = np.array([r["llm_calls"] for r in records_arm], dtype=float)
    calls_single = np.array([r["llm_calls"] for r in records_single], dtype=float)
    lat_arm = np.array([r["latency"] for r in records_arm], dtype=float)
    lat_single = np.array([r["latency"] for r in records_single], dtype=float)

    answerable_mask = is_answerable
    unanswerable_mask = ~is_answerable

    ans_ind_arm = (~abst_arm)[answerable_mask].astype(float)
    ans_ind_single = (~abst_single)[answerable_mask].astype(float)
    answered_rate_diff_point = float(ans_ind_arm.mean() - ans_ind_single.mean())
    answered_rate_diff_ci = paired_bootstrap_ci_diff(ans_ind_arm, ans_ind_single)

    abst_ind_arm = abst_arm[unanswerable_mask].astype(float)
    abst_ind_single = abst_single[unanswerable_mask].astype(float)
    abstention_rate_diff_point = float(abst_ind_arm.mean() - abst_ind_single.mean())
    abstention_rate_diff_ci = paired_bootstrap_ci_diff(abst_ind_arm, abst_ind_single)

    # J_arm - J_single = (TPR_arm-TPR_single) - (FPR_arm-FPR_single). FPR = 1 -
    # answered_rate, so its diff is exactly -(answered_rate diff); substituting
    # gives J's diff as the SUM of the two diffs computed above -- see
    # bootstrap_ci_sum_of_two_means's docstring.
    j_diff_point = answered_rate_diff_point + abstention_rate_diff_point
    j_diff_ci = bootstrap_ci_sum_of_two_means(
        ans_ind_arm - ans_ind_single, abst_ind_arm - abst_ind_single)

    precision_diff_point = _ratio_point(unanswerable_mask, abst_arm) - _ratio_point(unanswerable_mask, abst_single)
    precision_diff_ci = paired_bootstrap_ci_diff_ratio(abst_arm, abst_single, unanswerable_mask)

    calls_diff_point = float(calls_arm.mean() - calls_single.mean())
    calls_diff_ci = paired_bootstrap_ci_diff(calls_arm, calls_single)

    lat_diff_point = float(lat_arm.mean() - lat_single.mean())
    lat_diff_ci = paired_bootstrap_ci_diff(lat_arm, lat_single)

    return {
        "answered_rate_on_answerable_diff": {"point": answered_rate_diff_point, "ci95": list(answered_rate_diff_ci)},
        "abstention_rate_on_unanswerable_diff": {"point": abstention_rate_diff_point,
                                                   "ci95": list(abstention_rate_diff_ci)},
        "abstention_precision_diff": {"point": precision_diff_point, "ci95": list(precision_diff_ci)},
        "youdens_j_diff": {"point": j_diff_point, "ci95": list(j_diff_ci)},
        "mean_llm_calls_per_query_diff": {"point": calls_diff_point, "ci95": list(calls_diff_ci)},
        "mean_latency_s_per_query_diff": {"point": lat_diff_point, "ci95": list(lat_diff_ci)},
    }


def arm_verdict(arm: str, diffs: dict, arm_metrics: dict, single_metrics: dict) -> str:
    """HELPS / HURTS / "not an improvement" (verbatim, per module docstring),
    with the quality verdict (Youden's J) and the cost verdict (mean LLM
    calls/query) stated separately and ALWAYS both printed -- an arm whose
    only real effect is cost must say exactly that, never be dressed up as a
    quality win, and a quality win/loss must never be reported without its
    cost alongside it."""
    j = diffs["youdens_j_diff"]
    j_lo, j_hi = j["ci95"]
    quality_excludes_zero = not (np.isnan(j_lo) or np.isnan(j_hi)) and not (j_lo <= 0 <= j_hi)

    if quality_excludes_zero and j["point"] > 0:
        quality = (f"{arm} HELPS relative to single: Youden's J changes by {j['point']:+.3f} "
                   f"(paired 95% CI {_fmt_ci(j_lo, j_hi)}), excluding zero.")
    elif quality_excludes_zero and j["point"] < 0:
        quality = (f"{arm} HURTS relative to single: Youden's J changes by {j['point']:+.3f} "
                   f"(paired 95% CI {_fmt_ci(j_lo, j_hi)}), excluding zero -- worse abstention "
                   f"quality as an unanswerability detector.")
    else:
        quality = (f"{arm} is NOT AN IMPROVEMENT over single in abstention quality: Youden's J "
                   f"changes by {j['point']:+.3f} (paired 95% CI {_fmt_ci(j_lo, j_hi)}), which "
                   f"includes zero.")

    c = diffs["mean_llm_calls_per_query_diff"]
    c_lo, c_hi = c["ci95"]
    cost_excludes_zero = not (np.isnan(c_lo) or np.isnan(c_hi)) and not (c_lo <= 0 <= c_hi)
    single_calls = single_metrics["mean_llm_calls_per_query"]["point"]
    arm_calls = arm_metrics["mean_llm_calls_per_query"]["point"]

    if cost_excludes_zero and c["point"] < 0:
        cost = (f" Its measurable benefit here is COST: mean LLM calls/query drops from "
                f"{single_calls:.2f} to {arm_calls:.2f} (paired 95% CI on the difference "
                f"{_fmt_ci(c_lo, c_hi)}), excluding zero.")
    elif cost_excludes_zero and c["point"] > 0:
        cost = (f" It also COSTS MORE: mean LLM calls/query rises from {single_calls:.2f} to "
                f"{arm_calls:.2f} (paired 95% CI {_fmt_ci(c_lo, c_hi)}), excluding zero.")
    else:
        cost = (f" Cost is not measurably different either (mean LLM calls/query "
                f"{single_calls:.2f} -> {arm_calls:.2f}, paired 95% CI {_fmt_ci(c_lo, c_hi)}, "
                f"includes zero).")

    return quality + cost


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-answerable", type=int, default=100,
                    help="Answerable (BEIR/SciFact) queries to sample (default: 100).")
    p.add_argument("--n-unanswerable", type=int, default=100,
                    help="Unanswerable queries to sample, split evenly across "
                         "quora/fiqa/nfcorpus (default: 100).")
    p.add_argument("--seed", type=int, default=SEED, help=f"Sampling seed (default: {SEED}, the repo's).")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--generator", choices=["stub", "ollama"], default="stub",
                    help="'stub' (default): deterministic, zero-network -- see StubGenerator's "
                         "docstring. 'ollama': real generation via the local Ollama server -- "
                         "NOT used by the run this script's own docstring reports (module "
                         "docstring: 'never Ollama in this run').")
    p.add_argument("--ollama-model", default="llama3.2:3b")
    return p.parse_args()


def main() -> int:
    t_start = time.time()
    args = parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("Missing dependency 'pyarrow' (needed to read BEIR's parquet files). "
              "Install it with: uv pip install pyarrow")
        return 1

    print(f"Building labelled query set: --n-answerable {args.n_answerable} "
          f"--n-unanswerable {args.n_unanswerable} (seed {args.seed}), entirely from the "
          f"local Hugging Face cache (module docstring: never download) ...")
    try:
        data = build_query_set(args.n_answerable, args.n_unanswerable, args.seed)
    except RuntimeError as e:
        print(f"Could not build the query set without downloading: {e}")
        return 1

    n_total = len(data["queries"])
    print(f"  {data['n_answerable_used']} answerable (of {data['n_answerable_available']} "
          f"BEIR/SciFact judged queries available), {data['n_unanswerable_used']} unanswerable "
          f"{data['unanswerable_tier_counts']}. {n_total} queries total.")
    print(f"Corpus: {len(data['corpus_texts'])} SciFact documents.")

    if args.generator == "stub":
        generator = StubGenerator()
        print("Generator: StubGenerator (deterministic, zero-network -- see module docstring).")
    else:
        from ragtrust.generation.ollama import OllamaGenerator

        generator = OllamaGenerator(model=args.ollama_model)
        print(f"Generator: OllamaGenerator({args.ollama_model!r}) -- REAL network calls to Ollama.")

    config = Config(embed_model=args.embed_model)
    print(f"Building pipeline (embed_model={config.embed_model}, nli_model={config.nli_model}, "
          f"retrieval_mode={config.retrieval_mode}, k={config.k}, "
          f"abstain_threshold={config.abstain_threshold}, retrieval_gate={config.retrieval_gate}, "
          f"agentic_max_rounds={config.agentic_max_rounds}, "
          f"route_iterate_margin={config.route_iterate_margin}) ...")
    pipeline = build_pipeline(config, args.embed_model, data["corpus_texts"], data["corpus_meta"], generator)
    print(f"Pipeline built in {time.time() - t_start:.1f}s.")

    is_answerable = data["is_answerable"]
    queries = data["queries"]

    records: dict = {"single": [], "routed": [], "iterative": []}
    t_arms = time.time()
    for i, q in enumerate(queries):
        records["single"].append(run_single(pipeline, q))
        records["routed"].append(run_routed(pipeline, q, config.agentic_max_rounds, config.route_iterate_margin))
        records["iterative"].append(run_iterative(pipeline, q, config.agentic_max_rounds))
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(f"  ... scored {i + 1}/{n_total} queries across all 3 arms "
                  f"({time.time() - t_arms:.1f}s elapsed)")

    # ------------------------------------------------------------ multi-round check
    # The stub is REQUIRED (module docstring's smoke-test section) to actually
    # force >1 round on some queries -- otherwise the loop's reformulation/
    # pooling/stop-reason machinery is untested. Verified empirically here,
    # not just asserted in a comment.
    iter_rounds = [r["n_rounds"] for r in records["iterative"]]
    n_multi_round = sum(1 for n in iter_rounds if n > 1)
    stop_reason_counts = Counter(sr for r in records["iterative"] for sr in r["stop_reasons"])
    route_counts = Counter(r["route"] for r in records["routed"])

    print(f"\nMulti-round check (iterative arm): {n_multi_round}/{n_total} queries ran more than "
          f"one round. Stop-reason tally across all rounds: {dict(stop_reason_counts)}.")
    if n_multi_round == 0:
        print("WARNING: no query exercised more than one round -- the reformulation/pooling loop "
              "was NOT tested this run. See StubGenerator's docstring; its hash-bucket weakening "
              "may need adjustment for this particular query set.")

    # The stub answers mechanically from the top passage, so its answers score badly
    # on grounding no matter which arm produced them. That is fine for exercising the
    # loop -- reformulation, pooling, stop reasons, call counting -- and useless as a
    # measurement of whether iterating helps, because every arm is handicapped by the
    # same bad generator and the answered-rate column is measuring the stub rather
    # than the retrieval strategy. Printed as a banner rather than left as a
    # footnote: this file writes a results JSON, and a reader who finds
    # agentic_ablation.json without this warning would have no way to tell stub
    # output from a real run.
    if args.generator == "stub":
        banner = ("*** STUB GENERATOR: harness check only, NOT a measurement. "
                  "Re-run with --generator ollama for real numbers. ***")
        print("\n" + "*" * len(banner))
        print(banner)
        print("*" * len(banner))

    # ------------------------------------------------------------------- metrics
    metrics = {arm: compute_arm_metrics(records[arm], is_answerable) for arm in ("single", "routed", "iterative")}
    diffs = {arm: compute_paired_diffs(records[arm], records["single"], is_answerable)
             for arm in ("routed", "iterative")}
    verdicts = {arm: arm_verdict(arm, diffs[arm], metrics[arm], metrics["single"]) for arm in ("routed", "iterative")}

    # ------------------------------------------------------------------- printed table
    j_label = "Youden's J"
    print("\n" + "=" * 100)
    print(f"{'arm':<10} | {'answered-rate':<22} | {'abstention-rate':<22} | "
          f"{'abst.precision':<22} | {j_label:<22}")
    print(f"{'':<10} | {'(answerable)':<22} | {'(unanswerable)':<22} | {'':<22} | {'':<22}")
    print("-" * 100)
    for arm in ("single", "routed", "iterative"):
        m = metrics[arm]
        ar = m["answered_rate_on_answerable"]
        ab = m["abstention_rate_on_unanswerable"]
        pr = m["abstention_precision"]
        j = m["youdens_j"]
        print(f"{arm:<10} | {ar['point']:.3f} {_fmt_ci(*ar['ci95']):<14} | "
              f"{ab['point']:.3f} {_fmt_ci(*ab['ci95']):<14} | "
              f"{pr['point']:.3f} {_fmt_ci(*pr['ci95']):<14} | "
              f"{j['point']:+.3f} {_fmt_ci(*j['ci95']):<14}")
    print("=" * 100)

    print(f"\n{'arm':<10} | {'mean trust':<20} | {'mean LLM calls/q':<22} | {'total calls':<12} | "
          f"{'mean latency/q (s)':<22}")
    print("-" * 100)
    for arm in ("single", "routed", "iterative"):
        m = metrics[arm]
        mt = m["mean_geometric_trust_on_answered"]
        mc = m["mean_llm_calls_per_query"]
        ml = m["mean_latency_s_per_query"]
        print(f"{arm:<10} | {mt['point']:.3f} {_fmt_ci(*mt['ci95']):<12} | "
              f"{mc['point']:.3f} {_fmt_ci(*mc['ci95']):<14} | {m['total_llm_calls']:<12} | "
              f"{ml['point']:.4f} {_fmt_ci(*ml['ci95']):<14}")
    print("=" * 100)

    print(f"\nRouted arm's own decisions: {dict(route_counts)}")

    print("\n--- Paired diffs from `single` (95% CI; a CI spanning zero reads 'not an improvement') ---")
    for arm in ("routed", "iterative"):
        d = diffs[arm]
        print(f"\n{arm}:")
        for key, label in [
            ("answered_rate_on_answerable_diff", "answered-rate(answerable) diff"),
            ("abstention_rate_on_unanswerable_diff", "abstention-rate(unanswerable) diff"),
            ("abstention_precision_diff", "abstention-precision diff"),
            ("youdens_j_diff", "Youden's J diff"),
            ("mean_llm_calls_per_query_diff", "mean LLM calls/query diff"),
            ("mean_latency_s_per_query_diff", "mean latency/query diff (s)"),
        ]:
            v = d[key]
            print(f"  {label:<38} {v['point']:+.4f}  {_fmt_ci(*v['ci95'])}")

    print("\n--- Verdicts ---")
    for arm in ("routed", "iterative"):
        print(f"\n{verdicts[arm]}")

    # ------------------------------------------------------------------------- JSON
    json_out = {
        "_method": (
            "Positives: BEIR/scifact-qrels/test.tsv judged queries scored against the "
            "BeIR/scifact corpus. Negatives: fixed-seed samples from BeIR/quora, BeIR/fiqa, "
            "BeIR/nfcorpus (the 'hard', same-domain, possibly-noisy tier), split evenly, "
            "scored against the SAME SciFact corpus -- reusing experiments/09's construction. "
            "Youden's J = TPR - FPR for abstention as an unanswerability detector; used instead "
            "of F1 because it does not depend on the answerable:unanswerable ratio in the query "
            "set (see experiments/14_hagrid_calibration.py for the same argument applied to a "
            "different metric). All rate/diff CIs are 10,000-resample bootstraps; diffs from "
            "`single` are PAIRED (same query set, same order, across every arm)."
        ),
        "run_params": {
            "n_answerable_requested": args.n_answerable, "n_unanswerable_requested": args.n_unanswerable,
            "n_answerable_used": data["n_answerable_used"], "n_unanswerable_used": data["n_unanswerable_used"],
            "n_answerable_available": data["n_answerable_available"],
            "unanswerable_tier_counts": data["unanswerable_tier_counts"],
            # Explicit rather than left for a reader to infer from "generator":
            # a consumer of this JSON must be able to tell a harness check from a
            # real measurement with one unambiguous field.
            "is_real_measurement": args.generator != "stub",
            "seed": args.seed, "generator": args.generator, "embed_model": config.embed_model,
            "nli_model": config.nli_model, "retrieval_mode": config.retrieval_mode, "k": config.k,
            "retrieval_gate": config.retrieval_gate, "abstain_threshold": config.abstain_threshold,
            "agentic_max_rounds": config.agentic_max_rounds, "route_iterate_margin": config.route_iterate_margin,
            "n_boot": N_BOOT, "ci": CI,
        },
        "multi_round_check": {
            "n_multi_round_queries": n_multi_round, "n_total_queries": n_total,
            "stop_reason_counts": dict(stop_reason_counts),
        },
        "routed_arm_route_counts": dict(route_counts),
        "arms": metrics,
        "paired_diffs_from_single": diffs,
        "verdicts": verdicts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "agentic_ablation.json"
    out_path.write_text(json.dumps(json_out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
