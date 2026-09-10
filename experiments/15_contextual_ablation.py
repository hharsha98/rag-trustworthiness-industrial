#!/usr/bin/env python3
"""Does Contextual Retrieval (Anthropic's method) help on third-party data?

`src/ragtrust/ingest/contextualize.py` prepends an LLM-written situating blurb
to each chunk before it is embedded/BM25-indexed. This script measures whether
that actually improves retrieval on BEIR/SciFact -- the same third-party
benchmark `experiments/08_beir_ablation.py` uses (5,183 biomedical-claim-
verification abstracts, 300 judged queries, `BeIR/scifact-qrels`) -- rather
than assuming a technique with a plausible mechanism must help.

Four arms, so the contextual effect is isolated from the hybrid effect instead
of confounded with it:

    1. dense                          -- baseline
    2. hybrid                         -- BM25 + dense via RRF (repo default;
                                          experiment 08 found nDCG@10 ~= 0.644
                                          here vs ~= 0.529 for dense)
    3. hybrid + contextual
    4. hybrid + contextual + rerank

Arm 3 vs arm 2 is the headline comparison: retrieval mode (hybrid) is held
constant, so any difference isolates contextualisation specifically.

*** THE HONEST WEAK CASE, STATED UP FRONT ***
Contextual Retrieval is designed for chunks carved out of a larger document,
where the chunk alone has lost context (an entity, a section heading, a date)
that the parent document still has. Every BEIR/SciFact corpus row is already
a single, self-contained abstract -- there is no larger parent document to
draw context from. Contextualising it means asking the LLM to situate the
abstract within itself, which can at best paraphrase or lightly rephrase it,
not surface information from "elsewhere in the document" because there is no
elsewhere. This is not a fair test of the technique's best case (e.g. a long
filing or manual chunked into many pieces); it is closer to a worst case for
it. Whatever this script measures is a lower bound on the technique's
headroom on documents shaped like this, not a verdict on the technique in
general. This caveat is printed prominently in the script's own output, not
left in a comment for a reader to miss.

Statistics mirror experiment 08 exactly: per-arm mean with a bootstrap 95% CI,
plus -- since every arm scores the identical query set -- a PAIRED bootstrap
95% CI on the difference from the `hybrid` arm (not `dense`: the headline
question here is the contextual effect, not the hybrid-vs-dense effect
experiment 08 already answered). A CI that includes zero is reported as
exactly that: not an improvement. This script is written to report a null or
negative result as plainly as a positive one -- there is no code path in
which only "contextual helps" prints cleanly.

*** Reuse, not reimplementation ***
`experiments/` is a directory of standalone scripts, not a Python package
(there is no `experiments/__init__.py`), and `08_beir_ablation.py` is not a
valid module identifier (a bare module name can't start with a digit) --
`from 08_beir_ablation import x` is a SyntaxError. So this script loads it by
file path with `importlib.util` and pulls its BEIR loading, metric, caching,
and bootstrap helpers from the loaded module instead of copying them:
`load_scifact`, `qrels_to_lookup`, `mrr_at_k`, `embedding_cache_path`,
`bootstrap_ci_mean`, `paired_bootstrap_ci_diff`, `build_retriever`, and the
`N_BOOT` / `CI` / `SEED` / `RERANK_CANDIDATES` constants. Importing the file
this way has no side effects: everything in experiment 08 that touches the
network or a model happens inside `main()`, which this script never calls.

*** Contextualising 5,183 independent documents, not one chunked document ***
`contextualize_chunks(chunks, document_text, generator, ...)` assumes every
chunk in one call shares a single parent `document_text` -- true for a real
chunked document, not true here: SciFact's rows are independent documents,
each its own single passage. So `contextualize_corpus` below calls
`contextualize_chunks` once per document (chunks=[the document],
document_text=the same document), reusing its prompt-building, blurb-
cleaning, and graceful degradation on a per-document basis exactly as
intended, while handling the *cross-document* parallelism and caching itself:
`contextualize_chunks` loads and saves its whole cache file inside every call
(see `ingest/contextualize.py::_load_cache`/`_save_cache`), which is fine for
one call covering many chunks of one document but would race -- multiple
threads each doing their own read-modify-write of the same file -- if called
5,183 times concurrently with its own file cache turned on. `contextualize_corpus`
therefore calls it with `cache_dir=None` (disabling its internal file cache)
and keeps one in-memory cache, loaded once and saved periodically plus once at
the end, keyed with the exact same `_cache_key(model_tag, chunk_text)` hash
`contextualize_chunks` uses internally -- so the cache file this script writes
is byte-for-byte the same format `contextualize_chunks` would have produced.

Cache (mandatory): `data/benchmarks/contextualize_cache.json` (gitignored). A
second run with the same `--contextual-model` issues zero LLM calls -- every
document is a cache hit. Progress prints every 250 documents (count,
elapsed). Parallelised with `workers=8` LLM calls in flight.

Corpus-embedding caching (separate from the above) reuses experiment 08's
`CachedRetriever` via `build_retriever`, keyed by embed-model name. Arms 1-2
index the plain corpus text and, on a full (non `--max-docs`) run, reuse
experiment 08's *existing* embedding cache file directly (same model, same
corpus, same text construction -- same embeddings). Arms 3-4 index the
contextualised text, which is different text and therefore needs its own
cache key (`<embed-model>__contextual`) -- reusing arm 1-2's cache file for
it would silently score the contextual arms against the wrong (uncontextualised)
embeddings. A `--max-docs` smoke run gets its own key too
(`<embed-model>__max<N>`), so a truncated smoke run can never overwrite (or be
served) the full-corpus cache experiment 08 depends on.

Usage:
    python experiments/15_contextual_ablation.py                  # full run, all 5,183 docs (headline)
    python experiments/15_contextual_ablation.py --max-docs 300   # fast smoke run -- NOT the headline number

Exit code: always 0 (a measurement, not a pass/fail gate), except 1 if
`pyarrow` is missing or the BEIR download fails offline -- see `main()`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch/faiss imports

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_module(name: str, filename: str):
    """Load `experiments/<filename>` by path (see module docstring: 'Reuse, not
    reimplementation'). Importing this way triggers no network/model access --
    experiment 08's own file only touches those inside `main()`."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_beir = _load_module("_beir_ablation_08", "08_beir_ablation.py")

load_scifact = _beir.load_scifact
qrels_to_lookup = _beir.qrels_to_lookup
mrr_at_k = _beir.mrr_at_k
embedding_cache_path = _beir.embedding_cache_path
bootstrap_ci_mean = _beir.bootstrap_ci_mean
paired_bootstrap_ci_diff = _beir.paired_bootstrap_ci_diff
build_retriever = _beir.build_retriever
N_BOOT = _beir.N_BOOT
CI = _beir.CI
SEED = _beir.SEED
RERANK_CANDIDATES = _beir.RERANK_CANDIDATES
DEFAULT_EMBED_MODEL = _beir.DEFAULT_EMBED_MODEL
DEFAULT_RERANK_MODEL = _beir.DEFAULT_RERANK_MODEL

from ragtrust.metrics.relevance import ndcg_at_k, recall_at_k  # noqa: E402
from ragtrust.ingest.contextualize import (  # noqa: E402
    contextualize_chunks,
    _cache_key,
    _load_cache,
    _save_cache,
)
from ragtrust.generation.ollama import OllamaGenerator  # noqa: E402

CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"

CONTEXTUAL_MODEL_DEFAULT = "llama3.2:3b"
CONTEXT_WORKERS_DEFAULT = 8
PROGRESS_EVERY = 250
K = 10  # headline k throughout -- the one number this experiment exists to produce
BASELINE = "hybrid"  # every non-baseline arm's paired diff is against THIS arm,
                      # not "dense" -- the headline question is the contextual
                      # effect, with retrieval mode held constant.


# ============================================================================
# Contextualising an independent-document corpus (see module docstring).
# ============================================================================


def contextualize_corpus(corpus_texts: list, generator, cache_dir: Path, model_tag: str,
                          workers: int = CONTEXT_WORKERS_DEFAULT,
                          progress_every: int = PROGRESS_EVERY) -> tuple:
    """Return (contextualized_texts, n_llm_calls_this_run).

    Each document is its own single chunk with itself as `document_text` (see
    module docstring). Caching is a single in-memory dict, loaded once, saved
    every `progress_every` documents and once more at the end -- avoiding the
    multi-writer race `contextualize_chunks`'s own per-call file cache would
    hit if invoked concurrently 5,183 times (see module docstring)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(str(cache_dir))
    cache_lock = threading.Lock()
    n = len(corpus_texts)

    def _one(i: int):
        doc = corpus_texts[i]
        key = _cache_key(model_tag, doc)
        with cache_lock:
            cached = cache.get(key)
        if cached is not None:
            return i, cached, False
        # cache_dir=None: this call's own file cache is disabled on purpose --
        # this function owns the single on-disk cache file instead (see above).
        out = contextualize_chunks(
            [{"page": 0, "text": doc}], doc, generator,
            workers=1, cache_dir=None, model_tag=model_tag,
        )[0]
        with cache_lock:
            cache[key] = out["text"]
        return i, out["text"], True

    results: list = [None] * n
    t0 = time.time()
    completed = 0
    n_llm_calls = 0
    # executor.map is consumed only on this (the main) thread, so `completed`
    # and `n_llm_calls` need no lock even though the work itself runs on
    # `workers` threads -- only `cache` (mutated inside `_one`) does.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, text, was_llm_call in executor.map(_one, range(n)):
            results[i] = text
            if was_llm_call:
                n_llm_calls += 1
            completed += 1
            if completed % progress_every == 0 or completed == n:
                with cache_lock:
                    _save_cache(str(cache_dir), dict(cache))
                print(f"    ... contextualised {completed}/{n} documents "
                      f"({time.time() - t0:.1f}s elapsed, {n_llm_calls} LLM call(s) so far)")

    return results, n_llm_calls


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-docs", type=int, default=None,
                    help="Cap the corpus to its first N documents for a fast smoke run. "
                         "Default: the full ~5,183-document corpus -- the headline run. "
                         "A capped run is prominently marked as NOT the headline number.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    p.add_argument("--contextual-model", default=CONTEXTUAL_MODEL_DEFAULT)
    p.add_argument("--workers", type=int, default=CONTEXT_WORKERS_DEFAULT,
                    help="Parallel LLM calls in flight while contextualising (default 8).")
    return p.parse_args()


def main() -> int:
    t_start = time.time()
    args = parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("Missing dependency 'pyarrow' (needed to read BEIR's parquet files). "
              "Install it with: uv pip install pyarrow   (or: uv pip install -e '.[benchmarks]')")
        return 1

    print("Loading BEIR/SciFact (corpus, queries, qrels) ...")
    try:
        corpus_df, queries_df, qrels_df = load_scifact()
    except Exception as e:
        print(f"Could not load the BEIR/SciFact benchmark: {e}")
        print("This most likely means no network access to huggingface.co. Connect "
              "to the network and retry -- downloads are cached locally afterwards, "
              "so subsequent runs do not need it again.")
        return 1

    corpus_ids_full = corpus_df["_id"].astype(str).tolist()
    corpus_texts_full = [f"{title} {text}".strip()
                          for title, text in zip(corpus_df["title"], corpus_df["text"])]
    n_corpus_full = len(corpus_texts_full)

    truncated = args.max_docs is not None and args.max_docs < n_corpus_full
    if truncated:
        corpus_ids = corpus_ids_full[:args.max_docs]
        corpus_texts = corpus_texts_full[:args.max_docs]
    else:
        corpus_ids = corpus_ids_full
        corpus_texts = corpus_texts_full
    n_corpus = len(corpus_texts)

    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    qrels_lookup = qrels_to_lookup(qrels_df)
    # ALL judged queries, always -- no subsampling here (unlike experiment 08's
    # --queries flag). "No dropping queries" is one of this script's own
    # honesty requirements; --max-docs caps the corpus for speed, never the
    # query set.
    eval_qids = sorted(qrels_lookup.keys())

    print()
    if truncated:
        banner = (f"*** SMOKE RUN: corpus capped to --max-docs {args.max_docs} "
                   f"(of {n_corpus_full} total). THIS IS NOT THE HEADLINE NUMBER. "
                   "Re-run without --max-docs for that. ***")
        print("*" * len(banner))
        print(banner)
        print("*" * len(banner))
        print()
    print(f"Corpus: {n_corpus} documents"
          + (f" (truncated from {n_corpus_full})" if truncated else "") + ".")
    print(f"Evaluating all {len(eval_qids)} judged queries from BeIR/scifact-qrels/test.tsv "
          "against every arm (same query set for every arm, so paired comparisons are valid).")
    print()
    print("CAVEAT -- the honest weak case for Contextual Retrieval on this benchmark:")
    print("each SciFact document is a single, self-contained abstract with no larger")
    print("parent document it was chunked from. The situating blurb below therefore")
    print("places each abstract only within itself, not within surrounding pages or")
    print("sections the way the technique is designed for -- this is close to a worst")
    print("case for it, not a fair test of its best case. Any effect measured here is")
    print("a lower bound on the technique's headroom on documents shaped like this.")
    print()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Loading embedder '{args.embed_model}' ...")
    embedder = SentenceTransformer(args.embed_model)
    print(f"Loading reranker '{args.rerank_model}' ...")
    shared_cross_encoder = CrossEncoder(args.rerank_model)
    print()

    print(f"Contextualising {n_corpus} documents with Ollama '{args.contextual_model}' "
          f"({args.workers} parallel workers; cache: {CACHE_DIR / 'contextualize_cache.json'}) ...")
    generator = OllamaGenerator(model=args.contextual_model)
    t_ctx = time.time()
    contextualized_texts, n_llm_calls = contextualize_corpus(
        corpus_texts, generator, CACHE_DIR, args.contextual_model,
        workers=args.workers, progress_every=PROGRESS_EVERY,
    )
    ctx_time = time.time() - t_ctx
    n_cache_hits = n_corpus - n_llm_calls
    print(f"Contextualisation done in {ctx_time:.1f}s: {n_llm_calls} LLM call(s) issued, "
          f"{n_cache_hits} document(s) served from cache.")
    if n_llm_calls == 0:
        print("Zero LLM calls this run -- every document was a cache hit, as required "
              "for a repeat run.")
    print()

    # Embedding-cache keys: plain vs contextualised text is different text and must
    # never share a cache slot (see module docstring); a truncated corpus must never
    # share one with the full corpus either.
    plain_key = args.embed_model if not truncated else f"{args.embed_model}__max{args.max_docs}"
    ctx_key = f"{plain_key}__contextual"

    # 'hybrid+rerank' is a CONTROL, and the experiment is not interpretable without
    # it. Without that arm the only reranked configuration is the contextual one, so
    # its diff against plain 'hybrid' moves two variables at once (contextualisation
    # AND reranking) and the combined gain reads as evidence for contextualisation
    # when reranking alone may account for all of it -- or more than all of it. With
    # the control present the technique is isolated at both levels: hybrid ->
    # hybrid+contextual measures it without reranking, and hybrid+rerank ->
    # hybrid+contextual+rerank measures it with.
    arm_defs = [
        ("dense", "dense", False, corpus_texts, plain_key),
        ("hybrid", "hybrid", False, corpus_texts, plain_key),
        ("hybrid+rerank", "hybrid", True, corpus_texts, plain_key),
        ("hybrid+contextual", "hybrid", False, contextualized_texts, ctx_key),
        ("hybrid+contextual+rerank", "hybrid", True, contextualized_texts, ctx_key),
    ]

    results: dict = {}
    for i, (name, mode, rerank, texts, embed_key) in enumerate(arm_defs, start=1):
        print(f"[{i}/{len(arm_defs)}] {name} -- building ...")
        t0 = time.time()
        retriever = build_retriever(mode, rerank, embedder, CACHE_DIR, embed_key,
                                     args.rerank_model, shared_cross_encoder)
        retriever.build(texts)
        build_time = time.time() - t0

        ndcg_vals, recall_vals, mrr_vals = [], [], []
        t1 = time.time()
        for qid in eval_qids:
            query_text = queries_by_id[qid]
            relevant_ids = qrels_lookup[qid]
            retrieved = retriever.search(query_text, K)
            ranked_ids = [corpus_ids[p.id] for p in retrieved]
            ndcg_vals.append(ndcg_at_k(ranked_ids, relevant_ids, K))
            recall_vals.append(recall_at_k(ranked_ids, relevant_ids, K))
            mrr_vals.append(mrr_at_k(ranked_ids, relevant_ids, K))
        score_time = time.time() - t1

        results[name] = {
            "ndcg": np.array(ndcg_vals),
            "recall": np.array(recall_vals),
            "mrr": np.array(mrr_vals),
        }
        print(f"  {name}: built in {build_time:.1f}s, scored {len(eval_qids)} queries "
              f"in {score_time:.1f}s")
    print()

    # ------------------------------------------------------------------- CIs & JSON

    arms: dict = {}
    for name, vals in results.items():
        row = {}
        for metric, arr in vals.items():
            mean = float(arr.mean())
            lo, hi = bootstrap_ci_mean(arr)
            entry = {"mean": mean, "ci": [lo, hi]}
            if name != BASELINE:
                base_arr = results[BASELINE][metric]
                d_lo, d_hi = paired_bootstrap_ci_diff(arr, base_arr)
                entry["vs_hybrid_diff_mean"] = float((arr - base_arr).mean())
                entry["vs_hybrid_ci"] = [d_lo, d_hi]
                entry["ci_excludes_zero"] = bool(d_lo > 0.0 or d_hi < 0.0)
                entry["is_improvement"] = bool(d_lo > 0.0)
                entry["is_degradation"] = bool(d_hi < 0.0)
            row[metric] = entry
        arms[name] = row

    # ------------------------------------------------------------------- table

    print(f"Contextual Retrieval ablation -- BeIR/SciFact, n={len(eval_qids)} queries, "
          f"n_corpus={n_corpus}{' (TRUNCATED, smoke run)' if truncated else ''}, k={K}")
    print(f"Bootstrap: {N_BOOT} resamples, {int(CI * 100)}% CI, seed={SEED}. "
          f"Paired diffs are vs '{BASELINE}' (not 'dense').")
    print()
    header = f"| arm | nDCG@{K} | Recall@{K} | MRR@{K} | nDCG@{K} vs {BASELINE} (95% CI) |"
    print(header)
    print("|---|---|---|---|---|")
    for name in results:
        r = arms[name]
        ndcg, recall, mrr_r = r["ndcg"], r["recall"], r["mrr"]
        if name == BASELINE:
            diff_str = "-- (baseline for diff)"
        else:
            lo, hi = ndcg["vs_hybrid_ci"]
            diff_str = f"{ndcg['vs_hybrid_diff_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if ndcg["is_improvement"]:
                diff_str += " (improvement)"
            elif ndcg["is_degradation"]:
                diff_str += " (degradation)"
            else:
                diff_str += " (CI includes zero -- NOT an improvement)"
        print(f"| {name} | {ndcg['mean']:.3f} [{ndcg['ci'][0]:.3f}, {ndcg['ci'][1]:.3f}] "
              f"| {recall['mean']:.3f} [{recall['ci'][0]:.3f}, {recall['ci'][1]:.3f}] "
              f"| {mrr_r['mean']:.3f} [{mrr_r['ci'][0]:.3f}, {mrr_r['ci'][1]:.3f}] "
              f"| {diff_str} |")
    print()

    # ------------------------------------------------------------------- verdict

    headline_name = "hybrid+contextual"
    headline = arms[headline_name]["ndcg"]
    d_lo, d_hi = headline["vs_hybrid_ci"]
    d_mean = headline["vs_hybrid_diff_mean"]

    if headline["is_improvement"]:
        verdict = "IMPROVES"
        verdict_line = (
            f"Contextual retrieval **improves** nDCG@{K} on this benchmark: "
            f"{headline_name} beats {BASELINE} by {d_mean:+.3f} nDCG@{K} "
            f"(paired 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]), which excludes zero."
        )
    elif headline["is_degradation"]:
        verdict = "DEGRADES"
        verdict_line = (
            f"Contextual retrieval **degrades** nDCG@{K} on this benchmark: "
            f"{headline_name} is worse than {BASELINE} by {d_mean:+.3f} nDCG@{K} "
            f"(paired 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]), which excludes zero on the "
            "negative side."
        )
    else:
        verdict = "DOES NOT MEASURABLY CHANGE"
        verdict_line = (
            f"Contextual retrieval **does not measurably change** nDCG@{K} on this "
            f"benchmark: the paired 95% CI on {headline_name} vs {BASELINE} is "
            f"[{d_lo:+.3f}, {d_hi:+.3f}], which includes zero. A CI that includes "
            "zero is not an improvement -- this difference is not distinguishable "
            "from query-sampling noise at this sample size."
        )

    if truncated:
        verdict_line = (
            f"[SMOKE RUN on {n_corpus}/{n_corpus_full} documents -- NOT the headline "
            f"number] {verdict_line}"
        )

    print("VERDICT:", verdict_line)
    print()
    print("Reminder: SciFact documents are single, self-contained abstracts with no "
          "larger parent document -- the honest weak case stated above. This measures "
          "a lower bound on the technique's headroom on documents shaped like this, "
          "not its performance on the multi-chunk documents it is designed for.")
    print()

    # ------------------------------------------------------------------- write JSON

    json_out = {
        "_method": (
            f"Binary nDCG@{K}, Recall@{K}, MRR@{K} (SciFact relevance is binary) over "
            f"{len(eval_qids)} queries from BeIR/scifact-qrels/test.tsv, against a "
            f"{n_corpus}-document BeIR/scifact corpus. Bootstrap ({int(CI * 100)}% CI, "
            f"{N_BOOT} resamples, seed={SEED}): 'ci' is a per-arm CI on the raw mean "
            f"(resample queries); 'vs_hybrid_ci' is a PAIRED bootstrap CI on the "
            f"difference from the '{BASELINE}' arm (resample query pairs), since the "
            "headline question is the contextual effect with retrieval mode held "
            "constant, not the hybrid-vs-dense effect experiment 08 already answered. "
            "A 'vs_hybrid_ci' that contains 0 is reported as NOT an improvement."
        ),
        "caveat": (
            "SciFact documents are single, self-contained abstracts with no larger "
            "parent document to draw context from -- the honest weak case for "
            "Contextual Retrieval. The situating blurb places each abstract only "
            "within itself. This is a lower bound on the technique's headroom on "
            "documents shaped like this, not a fair test of its intended best case "
            "(a chunk drawn from a much larger document)."
        ),
        "benchmark": "BeIR/scifact (third-party corpus, queries, and judgments)",
        "n_queries": len(eval_qids),
        "n_corpus": n_corpus,
        "n_corpus_full": n_corpus_full,
        "truncated_smoke_run": truncated,
        "max_docs_arg": args.max_docs,
        "k": K,
        "rerank_candidates": RERANK_CANDIDATES,
        "embed_model": args.embed_model,
        "rerank_model": args.rerank_model,
        "contextual_model": args.contextual_model,
        "n_llm_calls_this_run": n_llm_calls,
        "n_cache_hits_this_run": n_cache_hits,
        "baseline_for_diff": BASELINE,
        "headline_comparison": f"{headline_name} vs {BASELINE}",
        "arms": arms,
        "verdict": verdict,
        "verdict_line": verdict_line,
    }
    if truncated:
        json_out["_smoke_run_warning"] = (
            f"This run used --max-docs {args.max_docs} out of {n_corpus_full} total "
            "documents. It is NOT the headline number. Re-run without --max-docs for "
            "the full-corpus result."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "contextual_ablation.json"
    out_path.write_text(json.dumps(json_out, indent=2))
    print(f"Wrote {out_path}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
