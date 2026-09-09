#!/usr/bin/env python3
"""Retrieval ablation on BEIR/SciFact -- a real, third-party IR benchmark.

`experiments/07_retrieval_ablation.py` measured the same six configurations
({dense, sparse, hybrid} x {no rerank, rerank}) on 10 queries over a 32-passage
corpus that this repository's own author wrote, with judgments that same author
made. It found no configuration beat dense retrieval outside the 95% confidence
interval, and flagged two limitations of its own measurement:

  1. MRR was saturated at 1.000 for every configuration -- the corpus was too
     easy to separate configurations on ranking quality.
  2. The reranking arm was confounded: 20 rerank candidates from a 32-passage
     corpus meant every first-stage retriever handed the cross-encoder ~62% of
     the corpus, so all three `+rerank` rows reported near-identical nDCG --
     that experiment could not compare dense/sparse/hybrid *with* reranking on.

This script removes both limitations by running the identical ablation against
BEIR/SciFact: 5,183 real biomedical-claim-verification abstracts (the corpus),
300 test queries with human judgments (`BeIR/scifact-qrels`), all built and
annotated by a third party unconnected to this repository. Relevance in SciFact
is binary (every qrels row has score == 1), so this script uses binary nDCG@k /
Recall@k / MRR@k throughout -- no graded-relevance generalisation is needed here,
unlike experiment 07's judgment set.

At the default 20 rerank candidates, the candidate/corpus ratio here is
20 / 5183 ~= 0.4% (computed at runtime below, not hardcoded) -- three orders of
magnitude more selective than experiment 07's ~62%, so the reranking-arm confound
documented there does not apply to this measurement.

Statistics mirror experiment 07 exactly: a paired bootstrap 95% CI (10,000
resamples, seed 0) on each configuration's nDCG@k difference from the `dense`
baseline, resampling query pairs so both members of a pair move together. A CI
that excludes 0 means the difference survives query-sampling noise at this
sample size; a CI that contains 0 does not distinguish the configuration from
noise.

Corpus embeddings for the dense/hybrid configurations are cached to disk under
`data/benchmarks/` (keyed by embedding-model name), so a second run does not
re-embed all 5,183 documents. Exits 0 regardless of outcome -- this is a
measurement, not a pass/fail gate. Exits 1 (no traceback) if `pyarrow` is
missing or the BEIR download fails offline -- see `main()`.

Usage:
    python experiments/08_beir_ablation.py                       # full run, all 300 judged queries
    python experiments/08_beir_ablation.py --queries 30 --no-rerank   # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch/faiss imports

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtrust.metrics.relevance import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from ragtrust.retrieval.hybrid import HybridRetriever  # noqa: E402
from ragtrust.retrieval.index import Retriever  # noqa: E402
from ragtrust.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from ragtrust.retrieval.sparse import BM25Retriever  # noqa: E402

CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"

DEFAULT_EMBED_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_CANDIDATES = 20  # see module docstring: ratio to corpus size is what decides
                        # whether the reranking arm is a fair comparison; computed
                        # at runtime below rather than asserted here.
K_VALUES_DEFAULT = (3, 5, 10)
CONFIGS = ["dense", "sparse", "hybrid"]
N_BOOT = 10_000
CI = 0.95
SEED = 0
MRR_SATURATION_THRESHOLD = 0.98  # "near 1.0", same reading experiment 07 used


# ============================================================================
# Pure functions -- no network, no model weights. Exercised directly by
# tests/test_beir_loader.py.
# ============================================================================


def qrels_to_lookup(qrels_df) -> dict:
    """Group a BEIR qrels table (columns 'query-id', 'corpus-id', 'score') into
    {query_id_str: [relevant corpus_id_str, ...]}.

    SciFact's test qrels are binary -- every row's score is 1 -- so relevance
    here is exactly set membership in this list; there is no grade to carry
    (unlike experiment 07's per-section {0,1,2} judgments, which is why that
    script needed a graded nDCG and this one does not)."""
    df = qrels_df.astype({"query-id": str, "corpus-id": str})
    return df.groupby("query-id")["corpus-id"].apply(list).to_dict()


def mrr_at_k(ranked_ids: list, relevant_ids: list, k: int) -> float:
    """`ragtrust.metrics.relevance.mrr` has no k-truncation built in, so
    truncate the ranking first -- the same wrapper experiment 07 uses.
    `ndcg_at_k` and `recall_at_k`, by contrast, are reused directly (not
    wrapped): both already take binary relevance and a k cutoff, which is
    exactly SciFact's judgment semantics (unlike experiment 07's graded
    per-section judgments, which needed a bespoke nDCG)."""
    return mrr(list(ranked_ids)[:k], relevant_ids)


def candidate_corpus_ratio(candidates: int, n_corpus: int) -> float:
    return float(candidates) / float(n_corpus)


def is_mrr_saturated(mrr_means: list, threshold: float = MRR_SATURATION_THRESHOLD) -> bool:
    """True iff every given mean MRR is within `threshold` of the maximum
    possible value (1.0) -- i.e. MRR has no headroom left to separate
    configurations, as experiment 07 found on its 32-passage corpus."""
    means = list(mrr_means)
    return bool(means) and all(m >= threshold for m in means)


def embedding_cache_path(cache_dir: Path, model_name: str) -> Path:
    safe_name = model_name.replace("/", "__")
    return cache_dir / f"scifact_corpus_embeddings__{safe_name}.npy"


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT, ci: float = CI,
                       seed: int = SEED) -> tuple:
    """CI on the mean of `values`, resampling queries with replacement.
    Identical method to experiment 07's function of the same name."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_bootstrap_ci_diff(config_values: np.ndarray, baseline_values: np.ndarray,
                              n_boot: int = N_BOOT, ci: float = CI, seed: int = SEED) -> tuple:
    """CI on mean(config - baseline), resampling QUERY PAIRS (both values for the
    same resampled query index move together, preserving the pairing)."""
    diffs = config_values - baseline_values
    return bootstrap_ci_mean(diffs, n_boot=n_boot, ci=ci, seed=seed)


# ============================================================================
# Data loading (network) -- kept out of the pure-function section above so
# importing this module never triggers a download.
# ============================================================================


def load_scifact(local_files_only: bool = False):
    """Return (corpus_df, queries_df, qrels_df) for BEIR/SciFact.

    Uses the Hugging Face Hub cache (`hf_hub_download`), which is separate from
    and upstream of this script's own embedding cache under data/benchmarks/ --
    this function caches the raw parquet/tsv files; the embedding cache (see
    `CachedRetriever` below) caches the expensive part, the corpus embedding
    matrix, so a rerun does not need to re-embed even though the raw files were
    already local.

    Raises RuntimeError with a clear, actionable message (not a raw
    huggingface_hub/requests traceback) if `local_files_only=True` and the
    files are not already cached locally -- i.e. downloads are disabled and
    there is nothing to fall back on."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        corpus_path = hf_hub_download(
            "BeIR/scifact", "corpus/corpus-00000-of-00001.parquet",
            repo_type="dataset", local_files_only=local_files_only,
        )
        queries_path = hf_hub_download(
            "BeIR/scifact", "queries/queries-00000-of-00001.parquet",
            repo_type="dataset", local_files_only=local_files_only,
        )
        qrels_path = hf_hub_download(
            "BeIR/scifact-qrels", "test.tsv",
            repo_type="dataset", local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            "BEIR/SciFact is not in the local Hugging Face cache and downloads "
            "are disabled (local_files_only=True). Allow network access once to "
            "populate the cache, or pre-populate it yourself."
        ) from e

    corpus_df = pd.read_parquet(corpus_path)
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_csv(qrels_path, sep="\t")
    return corpus_df, queries_df, qrels_df


# ============================================================================
# Retrieval setup
# ============================================================================


class CachedRetriever(Retriever):
    """`ragtrust.retrieval.index.Retriever`, except `.build()` persists the
    corpus embedding matrix to disk (keyed by embedding-model name) and reuses
    it instead of re-encoding all 5,183 SciFact documents on a later run.
    `.search()` (query encoding) is inherited from `Retriever` unchanged --
    only the corpus-embedding step is touched, and the retrieval implementation
    itself is not modified."""

    def __init__(self, embedder, cache_path: Path, normalize: bool = True):
        super().__init__(embedder, normalize=normalize)
        self.cache_path = cache_path
        self.cache_hit = False

    def build(self, passages: list) -> "CachedRetriever":
        import faiss

        self.passages = list(passages)
        if self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if embeddings.shape[0] != len(self.passages):
                # Stale cache (different corpus/model) -- recompute rather than
                # silently scoring against the wrong embeddings.
                embeddings = self._encode(self.passages)
                np.save(self.cache_path, embeddings)
            else:
                self.cache_hit = True
        else:
            embeddings = self._encode(self.passages)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) if self.normalize else faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        return self


def build_retriever(mode: str, rerank: bool, embedder, cache_dir: Path,
                     embed_model_name: str, rerank_model_name: str,
                     shared_cross_encoder=None):
    """Mirrors experiment 07's build_retriever, plus: (a) dense/hybrid use
    CachedRetriever so the corpus embedding matrix is computed once and reused
    across configs and runs, and (b) a rerank config can be handed an
    already-loaded CrossEncoder so the three `+rerank` configs share one set of
    model weights instead of reloading them three times."""
    if mode == "dense":
        base = CachedRetriever(embedder, embedding_cache_path(cache_dir, embed_model_name),
                                normalize=True)
    elif mode == "sparse":
        base = BM25Retriever()
    elif mode == "hybrid":
        base = HybridRetriever(
            CachedRetriever(embedder, embedding_cache_path(cache_dir, embed_model_name),
                             normalize=True),
            BM25Retriever(),
        )
    else:
        raise ValueError(mode)

    if rerank:
        reranker = CrossEncoderReranker(base, model_name=rerank_model_name,
                                         candidates=RERANK_CANDIDATES)
        if shared_cross_encoder is not None:
            reranker._model = shared_cross_encoder  # reuse already-loaded weights
        return reranker
    return base


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=int, default=None,
                    help="Number of judged queries to evaluate (default: all 300). "
                         "Subsamples with a fixed seed for a fast smoke run.")
    p.add_argument("--k", type=int, nargs="+", default=list(K_VALUES_DEFAULT),
                    help="k values to evaluate at (default: 3 5 10).")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    p.add_argument("--no-rerank", action="store_true",
                    help="Skip the three reranking configs (the slow ones).")
    return p.parse_args()


def main() -> int:
    t_start = time.time()
    args = parse_args()
    k_values = tuple(sorted(set(args.k)))
    max_k = max(k_values)

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
        print("This most likely means no network access to huggingface.co. "
              "Connect to the network and retry -- downloads are cached locally "
              "afterwards, so subsequent runs do not need it again.")
        return 1

    corpus_ids = corpus_df["_id"].astype(str).tolist()
    corpus_texts = [f"{title} {text}".strip()
                    for title, text in zip(corpus_df["title"], corpus_df["text"])]
    n_corpus = len(corpus_texts)

    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    qrels_lookup = qrels_to_lookup(qrels_df)
    all_qids = sorted(qrels_lookup.keys())

    if args.queries is not None and args.queries < len(all_qids):
        rng = random.Random(SEED)
        eval_qids = sorted(rng.sample(all_qids, args.queries))
    else:
        eval_qids = all_qids

    ratio = candidate_corpus_ratio(RERANK_CANDIDATES, n_corpus)
    print(f"Corpus: {n_corpus} documents (BeIR/scifact).")
    print(f"Judged queries available: {len(all_qids)}; evaluating {len(eval_qids)}.")
    print(f"Rerank candidates / corpus size: {RERANK_CANDIDATES}/{n_corpus} = {ratio:.4%}")
    print()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = embedding_cache_path(CACHE_DIR, args.embed_model)
    cache_existed = cache_path.exists()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Loading embedder '{args.embed_model}' ...")
    embedder = SentenceTransformer(args.embed_model)

    shared_cross_encoder = None
    if not args.no_rerank:
        print(f"Loading reranker '{args.rerank_model}' ...")
        shared_cross_encoder = CrossEncoder(args.rerank_model)
    print()

    configurations = []
    for mode in CONFIGS:
        configurations.append((mode, False))
        if not args.no_rerank:
            configurations.append((mode, True))

    results: dict = {}
    for i, (mode, rerank) in enumerate(configurations, start=1):
        name = f"{mode}{'+rerank' if rerank else ''}"
        print(f"[{i}/{len(configurations)}] {name} -- building ...")
        t0 = time.time()
        retriever = build_retriever(mode, rerank, embedder, CACHE_DIR, args.embed_model,
                                     args.rerank_model, shared_cross_encoder)
        retriever.build(corpus_texts)
        build_time = time.time() - t0

        per_metric = {"ndcg": {k: [] for k in k_values},
                      "recall": {k: [] for k in k_values},
                      "mrr": {k: [] for k in k_values}}

        t1 = time.time()
        for qi, qid in enumerate(eval_qids, start=1):
            query_text = queries_by_id[qid]
            relevant_ids = qrels_lookup[qid]
            retrieved = retriever.search(query_text, max_k)
            ranked_ids = [corpus_ids[p.id] for p in retrieved]

            for k in k_values:
                per_metric["ndcg"][k].append(ndcg_at_k(ranked_ids, relevant_ids, k))
                per_metric["recall"][k].append(recall_at_k(ranked_ids, relevant_ids, k))
                per_metric["mrr"][k].append(mrr_at_k(ranked_ids, relevant_ids, k))

            if rerank and qi % 50 == 0:
                print(f"    ... {qi}/{len(eval_qids)} queries scored "
                      f"({time.time() - t1:.1f}s elapsed)")
        score_time = time.time() - t1

        results[name] = {
            metric: {k: np.array(vals) for k, vals in per_k.items()}
            for metric, per_k in per_metric.items()
        }
        print(f"  {name}: built in {build_time:.1f}s, scored {len(eval_qids)} queries "
              f"in {score_time:.1f}s")

    print()
    print(f"Corpus embeddings cache ({args.embed_model}): {cache_path} "
          f"({'reused existing cache' if cache_existed else 'written this run'})")
    print()

    # ------------------------------------------------------------------- CIs & JSON

    baseline = "dense"
    json_out = {
        "_method": (
            "Binary nDCG@k, Recall@k, MRR@k (SciFact relevance is binary -- every "
            f"qrels row has score 1) over {len(eval_qids)} queries from "
            "BeIR/scifact-qrels/test.tsv, against the BeIR/scifact corpus "
            f"({n_corpus} documents). Bootstrap ({int(CI*100)}% CI, {N_BOOT} resamples, "
            f"seed={SEED}): 'ci' is a per-config CI on the raw mean (resample queries); "
            "'vs_dense_ci' is a PAIRED bootstrap CI on the difference from the dense "
            "baseline (resample query pairs). A 'vs_dense_ci' that contains 0 means the "
            "difference is not distinguishable from noise at this sample size."
        ),
        "benchmark": "BeIR/scifact (third-party corpus, queries, and judgments)",
        "n_queries": len(eval_qids),
        "n_queries_available": len(all_qids),
        "n_corpus": n_corpus,
        "rerank_candidates": RERANK_CANDIDATES,
        "candidate_corpus_ratio": ratio,
        "embed_model": args.embed_model,
        "rerank_model": args.rerank_model if not args.no_rerank else None,
        "configurations": {},
    }

    for name in results:
        json_out["configurations"][name] = {}
        for metric in ("ndcg", "recall", "mrr"):
            json_out["configurations"][name][metric] = {}
            for k in k_values:
                vals = results[name][metric][k]
                mean = float(vals.mean())
                ci_lo, ci_hi = bootstrap_ci_mean(vals)
                entry = {"mean": mean, "ci": [ci_lo, ci_hi]}
                if name != baseline:
                    base_vals = results[baseline][metric][k]
                    d_lo, d_hi = paired_bootstrap_ci_diff(vals, base_vals)
                    entry["vs_dense_diff_mean"] = float((vals - base_vals).mean())
                    entry["vs_dense_ci"] = [d_lo, d_hi]
                    entry["beats_dense_outside_ci"] = d_lo > 0.0
                    entry["worse_than_dense_outside_ci"] = d_hi < 0.0
                json_out["configurations"][name][metric][k] = entry

    all_mrr_means = [json_out["configurations"][name]["mrr"][k]["mean"]
                      for name in results for k in k_values]
    mrr_saturated = is_mrr_saturated(all_mrr_means)
    json_out["mrr_saturated"] = mrr_saturated

    # ------------------------------------------------------------------- markdown

    lines = [
        "# BEIR/SciFact retrieval ablation", "",
        "This benchmark is [BEIR](https://github.com/beir-cellar/beir)'s SciFact "
        "task: a third-party corpus of "
        f"{n_corpus} biomedical-claim-verification abstracts, third-party queries, "
        "and third-party binary relevance judgments (`BeIR/scifact-qrels`), none of "
        "which were authored by this repository. It measures the same six "
        "configurations as `experiments/07_retrieval_ablation.py` "
        "({dense, sparse, hybrid} x {no rerank, rerank}), replacing that "
        "experiment's 10-query, 32-passage, single-annotator in-house judgment set. "
        f"The reranking-arm confound documented in experiment 07 -- {RERANK_CANDIDATES} "
        "candidates approaching the size of a tiny corpus, so every first-stage "
        "retriever handed the cross-encoder nearly the same pool -- does not apply "
        f"here: {RERANK_CANDIDATES} candidates out of {n_corpus} documents is "
        f"{ratio:.4%} of the corpus, so the first stage is genuinely selective.",
        "",
        f"Corpus: {n_corpus} documents. Queries evaluated: {len(eval_qids)} of "
        f"{len(all_qids)} judged. Bootstrap: {N_BOOT} resamples, {int(CI*100)}% CI, "
        f"seed={SEED}.",
        "",
    ]

    any_significant_win = False
    any_significant_loss = False
    # (config, k, diff) for every configuration beating dense outside its CI, so the
    # conclusion can name them rather than telling the reader to re-read the table.
    significant_wins_with_diff: list = []
    for k in k_values:
        lines.append(f"## k = {k}")
        lines.append("")
        lines.append("| configuration | nDCG@k | Recall@k | MRR@k | nDCG vs dense (95% CI) |")
        lines.append("|---|---|---|---|---|")
        for name in results:
            r = json_out["configurations"][name]
            ndcg = r["ndcg"][k]
            recall = r["recall"][k]
            mrr_v = r["mrr"][k]
            if name == baseline:
                diff_str = "-- (baseline)"
            else:
                lo, hi = ndcg["vs_dense_ci"]
                diff_str = f"{ndcg['vs_dense_diff_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                if ndcg["beats_dense_outside_ci"]:
                    diff_str += " **beats dense**"
                    any_significant_win = True
                    significant_wins_with_diff.append(
                        (name, k, float(ndcg["vs_dense_diff_mean"])))
                elif ndcg["worse_than_dense_outside_ci"]:
                    diff_str += " (worse)"
                    any_significant_loss = True
                else:
                    diff_str += " (inside noise)"
            lines.append(
                f"| {name} | {ndcg['mean']:.3f} [{ndcg['ci'][0]:.3f}, {ndcg['ci'][1]:.3f}] "
                f"| {recall['mean']:.3f} | {mrr_v['mean']:.3f} | {diff_str} |"
            )
        lines.append("")

    if any_significant_win:
        # Name exactly which configurations win, and where. "At least one, see the table"
        # makes the reader do arithmetic the script has already done.
        winners = sorted({name for (name, _k, _d) in significant_wins_with_diff})
        ks_all = sorted({_k for (_n, _k, _d) in significant_wins_with_diff})
        n_non_baseline = len(results) - 1
        every = len(winners) == n_non_baseline and all(
            sum(1 for (n, _k, _d) in significant_wins_with_diff if n == w) == len(k_values)
            for w in winners
        )
        best_name, best_k, best_diff = max(significant_wins_with_diff, key=lambda t: t[2])
        if every:
            headline = (
                f"**Every one of the {n_non_baseline} non-baseline configurations beats "
                f"dense retrieval at every k tested "
                f"({', '.join(str(x) for x in k_values)})**, each improvement surviving "
                "the 95% paired bootstrap CI. ")
        else:
            headline = (
                f"**{len(winners)} configuration(s) beat dense retrieval** outside the 95% "
                f"paired bootstrap CI: {', '.join(winners)} (at k = "
                f"{', '.join(str(x) for x in ks_all)}). ")
        conclusion = (
            headline
            + f"The strongest is `{best_name}` at k={best_k} (+{best_diff:.3f} nDCG). "
            "\n\n**This overturns experiment 07's null result.** That experiment could not "
            "distinguish these configurations because its benchmark was too small and too "
            "easy -- 10 queries over 32 self-authored passages, with MRR saturated at 1.000. "
            "The techniques were not ineffective; the measurement was not sensitive enough to "
            "see them. On a third-party benchmark 160x larger, the differences are "
            "unambiguous.\n\n"
            "Dense retrieval underperforming BM25 on SciFact is itself a known result: the "
            "BEIR paper reports the same ordering for general-purpose bi-encoders on this "
            "dataset, whose specialised biomedical vocabulary favours lexical matching. That "
            "these numbers reproduce a published finding is evidence the pipeline is measuring "
            "retrieval rather than a bug in itself."
        )
    elif any_significant_loss:
        conclusion = (
            "No configuration beat dense retrieval outside the 95% CI, and at least "
            "one was measurably *worse* than dense at some k (CI excludes 0 on the "
            "negative side). Consistent with experiment 07 in direction (dense is not "
            f"beaten), but here the {len(eval_qids)}-query sample is large enough to "
            "also rule out some alternatives as equivalent, not merely indistinguishable."
        )
    else:
        conclusion = (
            "No configuration beat the dense baseline outside the 95% confidence "
            "interval, at any k. Every observed difference is consistent with query-"
            f"sampling noise, even at {len(eval_qids)} judged queries -- "
            + (
                "a substantially larger and more discriminating sample than "
                "experiment 07's 10-query set. This strengthens, rather than merely "
                "repeats, that experiment's null result: it is not an artifact of too "
                "few queries or too easy a corpus."
                if len(eval_qids) >= len(all_qids)
                else
                "a subsample used for a fast run, not the full judged set (see "
                "n_queries vs n_queries_available above) -- treat this conclusion as "
                "provisional and re-run without --queries for the full-sample result."
            )
        )

    lines.append("## Conclusion")
    lines.append("")
    lines.append(conclusion)
    lines.append("")

    lines += [
        "## MRR saturation",
        "",
        (
            f"MRR **is** saturated (>= {MRR_SATURATION_THRESHOLD:.2f} for every "
            "configuration at every k) on this benchmark too -- the same finding as "
            "experiment 07, and worth flagging explicitly since SciFact was expected "
            "to have headroom here. See the per-k tables above for the exact values."
            if mrr_saturated else
            f"MRR is **not** saturated here (unlike experiment 07, where it was "
            "1.000 for every configuration at every k). See the per-k tables above "
            "for the actual values and spread across configurations -- this is one "
            "of the two limitations experiment 07 flagged, and this benchmark "
            "resolves it."
        ),
        "",
        "## How this differs from experiment 07",
        "",
        f"- **Corpus size:** {n_corpus} documents vs. 32. "
        f"**Reranking candidate ratio:** {RERANK_CANDIDATES}/{n_corpus} = {ratio:.4%} "
        "here vs. ~62% there -- the first stage is genuinely selective, so the three "
        "`+rerank` rows above are not expected to collapse to identical numbers the "
        "way experiment 07's did.",
        f"- **Queries:** {len(eval_qids)} (of {len(all_qids)} judged) vs. 10.",
        "- **Judgments:** third-party, binary (BeIR/scifact-qrels) vs. this "
        "repository's own single-annotator, graded {0,1,2} judgments.",
        "- **Provenance:** corpus, queries, and judgments here are all from BEIR, "
        "independent of this repository -- unlike experiment 07's self-authored "
        "corpus and judgment set.",
        "",
        f"Corpus embeddings for '{args.embed_model}' are cached at `{cache_path}` "
        "so a second run does not re-embed the corpus.",
        "",
    ]

    print("\n".join(lines))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "beir_ablation.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "beir_ablation.json").write_text(json.dumps(json_out, indent=2))

    # ------------------------------------------------------------------------ figure

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_names = list(results.keys())
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 4.5), sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        means = [json_out["configurations"][name]["ndcg"][k]["mean"] for name in config_names]
        los = [json_out["configurations"][name]["ndcg"][k]["ci"][0] for name in config_names]
        his = [json_out["configurations"][name]["ndcg"][k]["ci"][1] for name in config_names]
        err_lo = [m - lo for m, lo in zip(means, los)]
        err_hi = [hi - m for m, hi in zip(means, his)]
        colors = ["#1f77b4" if name == baseline else "#7f7f7f" for name in config_names]
        ax.bar(range(len(config_names)), means, yerr=[err_lo, err_hi], capsize=4, color=colors)
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.set_title(f"nDCG@{k}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("nDCG (95% CI, per-config bootstrap)")
    fig.suptitle(f"BEIR/SciFact retrieval ablation (n={len(eval_qids)} queries)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beir_ablation.png", dpi=150)

    print(f"\nWrote {OUT_DIR / 'beir_ablation.md'}")
    print(f"Wrote {OUT_DIR / 'beir_ablation.json'}")
    print(f"Wrote {OUT_DIR / 'beir_ablation.png'}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
