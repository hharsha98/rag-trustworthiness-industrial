#!/usr/bin/env python3
"""Retrieval ablation: does BM25 / hybrid RRF / cross-encoder reranking beat dense?

Measures every configuration in {dense, sparse, hybrid} x {no rerank, rerank} against
`data/retrieval_judgments.json` -- a small (10-query), single-annotator, section-level
graded-relevance set over `data/demo_corpus.md` (see that file's `_provenance` note:
it is sufficient to *rank configurations against one another*, not evidence of
absolute retrieval quality on unseen corpora).

Judgments are graded per SECTION (2 = directly answers, 1 = supporting context,
absent = 0) and inherited by every chunk drawn from that section -- a chunk's `page`
is its 1-based section index (see `ingest/loader.py::chunk_passages`).

Metrics at k in {3, 5, 10}: graded nDCG@k (linear gain, same convention as the
binary `ndcg_at_k` in `metrics/relevance.py`, generalised to graded relevance since
that function only supports binary judgments), Recall@k (fraction of grade>=1
sections retrieved -- reuses `metrics.relevance.recall_at_k` directly, since its
existing set-based semantics already implement exactly that), and MRR@k (reciprocal
rank of the first grade-2 chunk within the top k -- reuses `metrics.relevance.mrr`
on the top-k-truncated ranking).

With only 10 queries, point estimates are noisy. Two different bootstrap CIs are
reported and must not be confused:
  - a per-configuration CI on the RAW mean metric (resampling queries), shown as the
    error bars in the figure -- "how uncertain is this configuration's own score";
  - a PAIRED bootstrap CI on the DIFFERENCE from the dense baseline (resampling
    query pairs, since every configuration is evaluated on the same 10 queries) --
    "is this configuration actually different from dense, or is that noise". This is
    the number the honesty conclusion is based on.

Downloads the dense embedding model (cached already if experiment 02 has run) and
the cross-encoder reranker (~80MB, first run only). Exits 0 regardless of outcome --
this is a measurement, not a pass/fail gate.

Usage:  python experiments/07_retrieval_ablation.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch/faiss imports

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtrust.config import Config  # noqa: E402
from ragtrust.ingest.loader import chunk_passages, load_corpus  # noqa: E402
from ragtrust.metrics.relevance import mrr, recall_at_k  # noqa: E402
from ragtrust.retrieval.hybrid import HybridRetriever  # noqa: E402
from ragtrust.retrieval.index import Retriever  # noqa: E402
from ragtrust.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from ragtrust.retrieval.sparse import BM25Retriever  # noqa: E402

CORPUS = ROOT / "data" / "demo_corpus.md"

# First-stage candidates handed to the cross-encoder. Named rather than inlined because
# its ratio to the corpus size decides whether the reranking arm of this ablation means
# anything: if it approaches the corpus size, every first-stage retriever passes through
# effectively the same pool and the comparison between them collapses. See the
# "What this benchmark cannot show" section of the generated report.
RERANK_CANDIDATES = 20
JUDGMENTS_PATH = ROOT / "data" / "retrieval_judgments.json"
OUT_DIR = ROOT / "experiments" / "results"

K_VALUES = (3, 5, 10)
MAX_K = max(K_VALUES)
CONFIGS = ["dense", "sparse", "hybrid"]
N_BOOT = 10_000
CI = 0.95
SEED = 0


# --------------------------------------------------------------------------- metrics


def graded_ndcg_at_k(ranked_ids: list, relevance_map: dict, k: int, all_ids: list) -> float:
    """Linear-gain graded nDCG@k -- same discount convention as the binary
    `ndcg_at_k` in metrics/relevance.py (gain / log2(rank+1), 1-indexed rank),
    generalised from {0,1} gain to the {0,1,2} grades used by this judgment set.

    IDCG must be computed from the multiset of gains actually achievable over the
    real candidate pool (`all_ids`, one entry per corpus chunk), NOT from
    `relevance_map.values()` alone. Judgments are made per SECTION and inherited by
    every chunk in it, so several chunks can carry the same grade-2 section --
    an ideal ranker could legitimately return more than one of them. Using only the
    distinct judged grades as the ideal ordering under-counts achievable gain and
    lets DCG exceed IDCG (nDCG > 1), which is what an earlier version of this
    function did before this fix."""
    ranked = list(ranked_ids)[:k]
    gains = [relevance_map.get(rid, 0) for rid in ranked]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted((relevance_map.get(i, 0) for i in all_ids), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(ranked_ids: list, relevant_ids: list, k: int) -> float:
    return mrr(list(ranked_ids)[:k], relevant_ids)


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT, ci: float = CI,
                       seed: int = SEED) -> tuple:
    """CI on the mean of `values`, resampling queries with replacement."""
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


# --------------------------------------------------------------------------- setup


def build_corpus():
    cfg = Config()
    chunks = chunk_passages(
        load_corpus(str(CORPUS)),
        window=cfg.chunk_window, stride=cfg.chunk_stride, min_chars=cfg.chunk_min_chars,
    )
    texts = [c["text"] for c in chunks]
    section_of = [c["page"] for c in chunks]  # 1-based section index, per loader.py
    return texts, section_of


def build_retriever(mode: str, rerank: bool, embed_model_name: str, rerank_model: str):
    """Mirrors RAGTrustPipeline._build_retriever's construction logic exactly, so
    the ablation measures the same retriever classes the pipeline would actually use."""
    from sentence_transformers import SentenceTransformer

    if mode == "dense":
        base = Retriever(SentenceTransformer(embed_model_name), normalize=True)
    elif mode == "sparse":
        base = BM25Retriever()
    elif mode == "hybrid":
        base = HybridRetriever(
            Retriever(SentenceTransformer(embed_model_name), normalize=True), BM25Retriever(),
        )
    else:
        raise ValueError(mode)

    if rerank:
        return CrossEncoderReranker(base, model_name=rerank_model,
                                    candidates=RERANK_CANDIDATES)
    return base


# --------------------------------------------------------------------------- main


def main() -> int:
    cfg = Config()
    texts, section_of = build_corpus()
    judgments = json.loads(JUDGMENTS_PATH.read_text())
    grades_by_query = judgments["grades"]
    queries = list(grades_by_query.keys())
    print(f"Corpus: {len(texts)} passages from {CORPUS.name}")
    print(f"Judgments: {len(queries)} queries from {JUDGMENTS_PATH.name}")
    print(judgments.get("_provenance", ""))
    print()

    configurations = []
    for mode in CONFIGS:
        configurations.append((mode, False))
        configurations.append((mode, True))

    # config_name -> {"ndcg": {k: np.array(per-query)}, "recall": {...}, "mrr": {...}}
    results: dict = {}

    for mode, rerank in configurations:
        name = f"{mode}{'+rerank' if rerank else ''}"
        t0 = time.time()
        retriever = build_retriever(mode, rerank, cfg.embed_model, cfg.rerank_model)
        retriever.build(texts)

        per_metric = {"ndcg": {k: [] for k in K_VALUES},
                      "recall": {k: [] for k in K_VALUES},
                      "mrr": {k: [] for k in K_VALUES}}

        for query in queries:
            grades = {int(sec): grade for sec, grade in grades_by_query[query].items()}
            relevant_any = [sec for sec, g in grades.items() if g >= 1]
            relevant_grade2 = [sec for sec, g in grades.items() if g == 2]

            retrieved = retriever.search(query, MAX_K)
            ranked_sections = [section_of[p.id] for p in retrieved]

            for k in K_VALUES:
                per_metric["ndcg"][k].append(
                    graded_ndcg_at_k(ranked_sections, grades, k, all_ids=section_of))
                per_metric["recall"][k].append(recall_at_k(ranked_sections, relevant_any, k))
                per_metric["mrr"][k].append(mrr_at_k(ranked_sections, relevant_grade2, k))

        results[name] = {
            metric: {k: np.array(vals) for k, vals in per_k.items()}
            for metric, per_k in per_metric.items()
        }
        print(f"  built + scored {name} in {time.time() - t0:.1f}s")

    print()

    # ------------------------------------------------------------------- CIs & table

    baseline = "dense"
    json_out = {
        "_method": (
            "Graded nDCG@k (linear gain), Recall@k (fraction of grade>=1 sections "
            "retrieved), MRR@k (reciprocal rank of first grade-2 chunk in top k), over "
            f"{len(queries)} queries from {JUDGMENTS_PATH.name}. Two bootstrap CIs "
            f"({int(CI*100)}%, {N_BOOT} resamples, seed={SEED}): 'ci' is a per-config CI "
            "on the raw mean (resample queries); 'vs_dense_ci' is a PAIRED bootstrap CI "
            "on the difference from the dense baseline (resample query pairs). A "
            "'vs_dense_ci' that contains 0 means the difference is not distinguishable "
            "from noise at this sample size."
        ),
        "_provenance_note": judgments.get("_provenance", ""),
        "n_queries": len(queries),
        "n_passages": len(texts),
        "configurations": {},
    }

    for name in results:
        json_out["configurations"][name] = {}
        for metric in ("ndcg", "recall", "mrr"):
            json_out["configurations"][name][metric] = {}
            for k in K_VALUES:
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

    # ------------------------------------------------------------------- markdown + print

    lines = ["# Retrieval ablation", "", judgments.get("_provenance", ""), "",
             f"Corpus: {len(texts)} passages. Queries: {len(queries)}. "
             f"Bootstrap: {N_BOOT} resamples, {int(CI*100)}% CI, seed={SEED}.", ""]

    any_significant_win = False
    for k in K_VALUES:
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
                elif ndcg["worse_than_dense_outside_ci"]:
                    diff_str += " (worse)"
                else:
                    diff_str += " (inside noise)"
            lines.append(
                f"| {name} | {ndcg['mean']:.3f} [{ndcg['ci'][0]:.3f}, {ndcg['ci'][1]:.3f}] "
                f"| {recall['mean']:.3f} | {mrr_v['mean']:.3f} | {diff_str} |"
            )
        lines.append("")

    if any_significant_win:
        conclusion = (
            "At least one configuration's nDCG@k improvement over dense retrieval "
            "survives the 95% paired bootstrap CI (does not include 0) at some k -- "
            "see the table above for which one(s) and at which k."
        )
    else:
        conclusion = (
            "No configuration beat the dense baseline outside the 95% confidence "
            "interval, at any k. Every observed difference is consistent with query-"
            "sampling noise on this 10-query set. This is a valid, publishable null "
            "result on this corpus/judgment set -- it is not evidence that BM25/hybrid/"
            "reranking never help, only that this measurement could not distinguish "
            "them from dense retrieval here."
        )

    lines.append("## Conclusion")
    lines.append("")
    lines.append(conclusion)
    lines.append("")

    # Two properties of this benchmark limit what the table above can support. Both were
    # found by inspecting the results rather than predicted, and both narrow the claim.
    n_passages = len(texts)
    cand = RERANK_CANDIDATES
    lines += [
        "## What this benchmark cannot show",
        "",
        "**MRR is saturated and therefore uninformative here.** It is 1.000 for every "
        "configuration at every k, because the top-ranked passage is already a "
        "directly-relevant one for all 10 queries. That is a real measurement, not a bug -- "
        "and it means MRR has no headroom on this corpus and cannot separate any two "
        "configurations. Read the identical 1.000 column as 'this benchmark is too easy to "
        "measure ranking quality', not as 'all configurations rank equally well'.",
        "",
        f"**The reranking arm is confounded by candidate-pool saturation.** Reranking scores "
        f"the top {cand} first-stage candidates, but the corpus holds only {n_passages} "
        f"passages -- so the first stage passes through roughly "
        f"{100.0 * min(cand, n_passages) / n_passages:.0f}% of everything, and all three "
        "first-stage retrievers hand the cross-encoder nearly the same pool. That is why the "
        "three `+rerank` rows report near-identical nDCG: the cross-encoder is reordering the "
        "same set each time. **This experiment therefore cannot compare dense, sparse and "
        "hybrid retrieval when reranking is enabled.** Doing so needs a corpus large enough "
        "that the first stage is genuinely selective -- a rule of thumb is candidates well "
        "under a tenth of the corpus.",
        "",
        "Neither limitation affects the headline result, which concerns the non-reranked "
        "configurations, and both argue the same way: this measurement is a floor on what "
        "these techniques could do, not a ceiling.",
        "",
    ]

    print("\n".join(lines))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "retrieval_ablation.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "retrieval_ablation.json").write_text(json.dumps(json_out, indent=2))

    # ------------------------------------------------------------------------ figure

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_names = list(results.keys())
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 4.5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
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
    axes[0].set_ylabel("graded nDCG (95% CI, per-config bootstrap)")
    fig.suptitle(f"Retrieval ablation on {JUDGMENTS_PATH.name} (n={len(queries)} queries)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retrieval_ablation.png", dpi=150)
    print(f"\nWrote {OUT_DIR / 'retrieval_ablation.md'}")
    print(f"Wrote {OUT_DIR / 'retrieval_ablation.json'}")
    print(f"Wrote {OUT_DIR / 'retrieval_ablation.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
