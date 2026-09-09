#!/usr/bin/env python3
"""Calibrate the pre-generation retrieval gate (`Config.retrieval_gate`)
on real data.

`RAGTrustPipeline.answer()` runs two abstention gates in increasing cost order
(pipeline.py). This script measures only the first one:

    top = max_context_similarity(query, passage_texts, self.embedder)   # in [0, 1]
    if top < self.config.retrieval_gate:
        return self._declined(...)

The gate's original value, 0.25, was fitted on a 32-passage self-authored toy corpus:
separation there was ~0.72 in-corpus against ~0.08 out-of-corpus, and 0.25 was picked to
sit in the open space between two anecdotal numbers rather than measured against a
distribution. This script replaces that with an operating point measured
against BEIR/SciFact.

Positives (corpus *can* answer): BEIR/SciFact's 300 judged test queries, scored
against the SciFact corpus (5,183 documents) it was written to be answered from.

Negatives (corpus *cannot* answer): queries from three other BEIR sets, scored
against the SAME SciFact corpus -- a query about hotel refunds or Pythagoras' theorem
has no business retrieving a biomedical abstract:
    - BeIR/quora    (general questions)         -- EASY negatives
    - BeIR/fiqa     (financial questions)        -- MEDIUM negatives (a coherent
      domain, but unrelated to biomedicine)
    - BeIR/nfcorpus (biomedical questions)       -- HARD negatives, and *noisy*: some
      NFCorpus queries may be genuinely answerable from SciFact, since both are
      biomedical. The hard-tier false-acceptance rate is reported as a pessimistic
      UPPER BOUND on the true error rate, not a clean measurement -- see `main()`.

The similarity computed here is the same quantity the gate thresholds: the highest
cosine similarity, clamped to [0, 1], between the query embedding and any SciFact
corpus document's embedding (see `max_similarity_to_corpus` below). This is computed
directly against the full corpus embedding matrix rather than through a live
`Retriever.search()` call, which is equivalent for `retrieval_mode="dense"` (the top-k
by cosine necessarily contains the single highest-cosine document once k >= 1) and a
mild, conservative over-estimate for the deployed "hybrid" default (RRF could in
principle drop the highest-cosine document from its top-k in favour of a lexical
match) -- conservative in the direction of admitting more, not fewer, so it does not
understate false-abstention risk.

**The core design decision -- an asymmetric objective, not accuracy/F1:**
A false abstention (refusing an answerable question) is a hard, unrecoverable failure.
A false acceptance (letting an unanswerable question through) costs one generation
call and is then caught by the SECOND gate (grounding/faithfulness), which sits
downstream. So this gate should be tuned for high sensitivity -- near-zero false
abstention -- accepting a higher false-acceptance rate in exchange, because false
acceptances have a safety net and false abstentions do not. A balanced metric such as
F1 or "the threshold that maximizes accuracy" would optimize the wrong thing here and
is deliberately NOT reported as a recommendation basis.

Corpus embeddings are cached under data/benchmarks/ by experiment 08 and reused
here unchanged (loaded via `08_beir_ablation.py`'s own `load_scifact` /
`embedding_cache_path` / `CachedRetriever`, imported by path since experiment
filenames start with a digit and are not importable packages).

Usage:
    python experiments/09_gate_calibration.py                     # full run (300/tier)
    python experiments/09_gate_calibration.py --negatives 30      # fast smoke run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch/faiss imports

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtrust.config import Config  # noqa: E402
from ragtrust.validation.stats import bootstrap_ci, roc_auc  # noqa: E402

OUT_DIR = ROOT / "experiments" / "results"
CACHE_DIR = ROOT / "data" / "benchmarks"

DEFAULT_EMBED_MODEL = Config().embed_model
CURRENT_GATE = Config().retrieval_gate  # whatever is configured -- what we are evaluating

NEGATIVE_SOURCES = {
    "easy": ("BeIR/quora", "general questions, unrelated to biomedicine"),
    "medium": ("BeIR/fiqa", "financial questions -- a coherent domain, still unrelated"),
    "hard": ("BeIR/nfcorpus", "biomedical questions -- same broad domain as SciFact; "
                              "NOISY negatives, see module docstring"),
}
TIER_ORDER = ["easy", "medium", "hard"]

THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP = 0.00, 0.60, 0.01
RETENTION_TARGETS = [0.99, 0.975, 0.95]
N_BOOT = 10_000
BOOT_SEED = 0

_BEIR_ABLATION_PATH = ROOT / "experiments" / "08_beir_ablation.py"


def _load_beir_ablation_module():
    """Import experiment 08 by file path (its module name starts with a digit,
    so it cannot be imported normally) to reuse its SciFact loader and
    corpus-embedding cache instead of duplicating either."""
    spec = importlib.util.spec_from_file_location("beir_ablation_09dep", _BEIR_ABLATION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("beir_ablation_09dep", module)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Pure functions -- no network, no model weights. Exercised directly by
# tests/test_gate_calibration.py.
# ============================================================================


def build_threshold_grid(lo: float = THRESHOLD_MIN, hi: float = THRESHOLD_MAX,
                          step: float = THRESHOLD_STEP) -> np.ndarray:
    n_steps = round((hi - lo) / step)
    return np.round(np.linspace(lo, hi, n_steps + 1), 10)


def retention_rate(pos_scores, threshold: float) -> float:
    """Fraction of answerable (positive) queries the gate would NOT abstain
    on at this threshold -- i.e. 1 - false-abstention rate."""
    pos_scores = np.asarray(pos_scores, dtype=float)
    if pos_scores.size == 0:
        return float("nan")
    return float(np.mean(pos_scores >= threshold))


def false_acceptance_rate(neg_scores, threshold: float) -> float:
    """Fraction of unanswerable (negative) queries that would incorrectly
    pass the gate at this threshold."""
    neg_scores = np.asarray(neg_scores, dtype=float)
    if neg_scores.size == 0:
        return float("nan")
    return float(np.mean(neg_scores >= threshold))


def sweep_thresholds(pos_scores, neg_scores_by_tier: dict, thresholds=None) -> list:
    """Threshold table: for each candidate threshold, retention on positives
    and the false-acceptance rate for each negative tier.

    Retention is monotonically non-increasing in `threshold` (raising the bar
    can only exclude more, never fewer, positives) -- see
    tests/test_gate_calibration.py."""
    if thresholds is None:
        thresholds = build_threshold_grid()
    rows = []
    for t in thresholds:
        row = {"threshold": float(t), "retention": retention_rate(pos_scores, t)}
        for tier, neg_scores in neg_scores_by_tier.items():
            row[f"far_{tier}"] = false_acceptance_rate(neg_scores, t)
        rows.append(row)
    return rows


def recommended_threshold(pos_scores, thresholds, retention_floor: float) -> float:
    """The largest candidate threshold whose retention still meets
    `retention_floor` (e.g. 0.99 for <=1% false abstention).

    If no threshold in the grid meets the floor (positives too spread out /
    floor too strict for the grid's resolution), falls back to the smallest
    threshold in the grid -- the most permissive option available -- rather
    than raising, since abstaining from a recommendation is worse than
    returning the most conservative-toward-retention answer on hand."""
    thresholds = np.asarray(sorted(thresholds), dtype=float)
    meeting = [t for t in thresholds if retention_rate(pos_scores, t) >= retention_floor]
    if not meeting:
        return float(thresholds.min())
    return float(max(meeting))


# ============================================================================
# Data loading (network) -- kept out of the pure-function section above so
# importing this module never triggers a download.
# ============================================================================


def load_negative_queries(repo_id: str, n: int, seed: int) -> list:
    """Sample `n` query texts (fixed seed) from a BEIR dataset's queries split."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "queries/queries-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    texts = df["text"].dropna().astype(str).tolist()
    if n >= len(texts):
        return texts
    rng = random.Random(seed)
    return rng.sample(texts, n)


def max_similarity_to_corpus(query_embs: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Highest cosine similarity, clamped to [0, 1], between each query and any
    corpus document. Both inputs are assumed L2-normalised (as `CachedRetriever`
    stores them), so cosine similarity is the plain dot product -- see
    `metrics/relevance.max_context_similarity`, which this mirrors exactly except
    for scoring against the whole corpus instead of a retrieved top-k (see module
    docstring for why that is equivalent-or-conservative here)."""
    sims = query_embs @ corpus_embs.T
    return np.clip(sims.max(axis=1), 0.0, 1.0)


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--negatives", type=int, default=300,
                    help="Negative queries sampled per tier (default: 300).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0).")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    return p.parse_args()


def _percentiles(values: np.ndarray) -> dict:
    pct = np.percentile(values, [5, 25, 50, 75, 95])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
            "p75": float(pct[3]), "p95": float(pct[4])}


def main() -> int:
    t_start = time.time()
    args = parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("Missing dependency 'pyarrow' (needed to read BEIR's parquet files). "
              "Install it with: uv pip install pyarrow")
        return 0

    print("Loading experiment 08 (BEIR/SciFact loader + embedding cache) ...")
    beir08 = _load_beir_ablation_module()

    print("Loading BEIR/SciFact (corpus, queries, qrels) ...")
    try:
        corpus_df, queries_df, qrels_df = beir08.load_scifact()
    except Exception as e:
        print(f"Could not load BEIR/SciFact: {e}")
        print("This most likely means no network access to huggingface.co. This is a "
              "measurement script, not a pass/fail gate -- exiting 0 with no results.")
        return 0

    corpus_texts = [f"{title} {text}".strip()
                    for title, text in zip(corpus_df["title"], corpus_df["text"])]
    n_corpus = len(corpus_texts)
    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
    qrels_lookup = beir08.qrels_to_lookup(qrels_df)
    positive_qids = sorted(qrels_lookup.keys())
    positive_queries = [queries_by_id[qid] for qid in positive_qids if qid in queries_by_id]
    print(f"Corpus: {n_corpus} documents. Positive (answerable) queries: "
          f"{len(positive_queries)} (BeIR/scifact-qrels/test.tsv).")

    print(f"Loading embedder '{args.embed_model}' ...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embed_model)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = beir08.embedding_cache_path(CACHE_DIR, args.embed_model)
    cache_existed = cache_path.exists()
    retriever = beir08.CachedRetriever(embedder, cache_path, normalize=True)
    retriever.build(corpus_texts)  # loads from cache if present; never re-embeds otherwise
    corpus_embs = np.load(cache_path).astype("float32")
    print(f"Corpus embeddings: {cache_path} "
          f"({'reused existing cache' if cache_existed else 'written this run'})")

    def encode_normalized(texts: list) -> np.ndarray:
        embs = np.asarray(embedder.encode(list(texts)), dtype="float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    print("\nScoring positives against the SciFact corpus ...")
    pos_embs = encode_normalized(positive_queries)
    pos_scores = max_similarity_to_corpus(pos_embs, corpus_embs)

    neg_scores_by_tier = {}
    neg_queries_by_tier = {}
    for tier in TIER_ORDER:
        repo_id, desc = NEGATIVE_SOURCES[tier]
        print(f"Loading negatives ({tier}: {repo_id}) -- {desc} ...")
        neg_queries = load_negative_queries(repo_id, args.negatives, args.seed)
        neg_queries_by_tier[tier] = neg_queries
        print(f"  scoring {len(neg_queries)} {tier} negatives against SciFact corpus ...")
        neg_embs = encode_normalized(neg_queries)
        neg_scores_by_tier[tier] = max_similarity_to_corpus(neg_embs, corpus_embs)

    all_neg_scores = np.concatenate([neg_scores_by_tier[t] for t in TIER_ORDER])

    # ------------------------------------------------------------------ distributions

    distributions = {"positive": _percentiles(pos_scores)}
    for tier in TIER_ORDER:
        distributions[f"negative_{tier}"] = _percentiles(neg_scores_by_tier[tier])
    distributions["negative_pooled"] = _percentiles(all_neg_scores)

    # ------------------------------------------------------------------ ROC-AUC + CI

    def _auc_section(neg_scores):
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        point, lo, hi = bootstrap_ci(scores, labels, roc_auc, n=N_BOOT, seed=BOOT_SEED)
        return {"auc": point, "ci95": [lo, hi], "n_pos": int(len(pos_scores)),
                "n_neg": int(len(neg_scores))}

    auc_results = {tier: _auc_section(neg_scores_by_tier[tier]) for tier in TIER_ORDER}
    auc_results["pooled"] = _auc_section(all_neg_scores)

    # ------------------------------------------------------------------ threshold table

    thresholds = build_threshold_grid()
    table = sweep_thresholds(pos_scores, neg_scores_by_tier, thresholds)

    recommended = {
        f"{int(r*1000)/10:g}%": recommended_threshold(pos_scores, thresholds, r)
        for r in RETENTION_TARGETS
    }
    headline_threshold = recommended["99%"]

    def _row_at(threshold: float) -> dict:
        # thresholds are on a fixed 0.01 grid; snap to nearest grid point for lookup.
        idx = int(round((threshold - THRESHOLD_MIN) / THRESHOLD_STEP))
        idx = max(0, min(idx, len(table) - 1))
        return table[idx]

    current_row = _row_at(CURRENT_GATE)
    recommended_row = _row_at(headline_threshold)

    # current-gate verdict: is the configured gate too high, too low, or fine, relative to the
    # >=99%-retention objective this script argues for?
    if current_row["retention"] < 0.99:
        far_at_reco = {t: recommended_row[f"far_{t}"] for t in TIER_ORDER}
        far_at_current = {t: current_row[f"far_{t}"] for t in TIER_ORDER}
        verdict = (
            f"{CURRENT_GATE} is TOO HIGH for a >=99%-retention objective: it retains only "
            f"{current_row['retention']:.1%} of answerable queries (false-abstention "
            f"rate {1 - current_row['retention']:.1%}), which is a hard, unrecoverable "
            f"failure under this system's own asymmetry argument. The recommended "
            f"threshold {headline_threshold:.2f} retains >=99% while raising the "
            f"false-acceptance rate only modestly (easy {far_at_current['easy']:.1%} -> "
            f"{far_at_reco['easy']:.1%}, medium {far_at_current['medium']:.1%} -> "
            f"{far_at_reco['medium']:.1%}, hard(noisy) {far_at_current['hard']:.1%} -> "
            f"{far_at_reco['hard']:.1%})."
        )
    elif abs(headline_threshold - CURRENT_GATE) <= THRESHOLD_STEP:
        verdict = (
            f"{CURRENT_GATE} is already close to the measured >=99%-retention operating point "
            f"({headline_threshold:.2f}). It retains {current_row['retention']:.1%} of "
            f"answerable queries here."
        )
    else:
        verdict = (
            f"{CURRENT_GATE} is LOWER than necessary for a >=99%-retention objective: it already "
            f"retains {current_row['retention']:.1%} of answerable queries, and the "
            f"threshold could be raised to {headline_threshold:.2f} (still meeting "
            f">=99% retention) to reduce false acceptance without giving up sensitivity."
        )

    # ------------------------------------------------------------------ JSON output

    json_out = {
        "_method": (
            "Positives: BEIR/scifact-qrels/test.tsv judged queries scored against the "
            "BeIR/scifact corpus. Negatives: fixed-seed samples from BeIR/quora (easy), "
            "BeIR/fiqa (medium), BeIR/nfcorpus (hard, noisy -- see caveat) scored "
            "against the SAME SciFact corpus. Similarity = max cosine (clamped to "
            "[0,1]) between the query embedding and any corpus document embedding, "
            "matching metrics/relevance.max_context_similarity. Objective is "
            ">=99% retention of answerable queries (near-zero false abstention), NOT "
            "balanced accuracy/F1 -- see module docstring for the asymmetry argument."
        ),
        "embed_model": args.embed_model,
        "n_corpus": n_corpus,
        "n_positive": int(len(pos_scores)),
        "n_negative_per_tier": {t: int(len(neg_scores_by_tier[t])) for t in TIER_ORDER},
        "current_gate": CURRENT_GATE,
        "current_gate_retention": current_row["retention"],
        "current_gate_far": {t: current_row[f"far_{t}"] for t in TIER_ORDER},
        "recommended_threshold": recommended,
        "recommended_threshold_far": {t: recommended_row[f"far_{t}"] for t in TIER_ORDER},
        "verdict": verdict,
        "distributions": distributions,
        "roc_auc": auc_results,
        "threshold_table": table,
        "nfcorpus_caveat": (
            "NFCorpus is biomedical, the same broad domain as SciFact. Some of its "
            "queries may be genuinely answerable from the SciFact corpus, making them "
            "noisy (mislabeled) negatives. The 'hard' tier's false-acceptance rate is "
            "therefore a PESSIMISTIC UPPER BOUND on the true error rate against "
            "genuinely out-of-corpus biomedical queries, not a clean measurement."
        ),
    }

    # ------------------------------------------------------------------ markdown

    lines = [
        "# Retrieval gate calibration", "",
        "Calibrating `Config.retrieval_gate` (the pre-generation abstention gate in "
        "`RAGTrustPipeline.answer()`) against real data, replacing the constant fitted "
        "on a 32-passage toy corpus.", "",
        "**Objective.** The two errors this gate can make are not equally bad. A false "
        "abstention (refusing an answerable question) is a hard, unrecoverable failure "
        "-- the user gets nothing. A false acceptance (letting an unanswerable question "
        "through) costs one generation call and is then caught by the second, "
        "post-generation grounding gate. So the right operating point is "
        "**high-sensitivity: minimize false abstention, tolerate more false "
        "acceptance** -- not the threshold that maximizes accuracy or F1, which would "
        "balance the two errors as if they cost the same.", "",
        f"Positives: {len(pos_scores)} BEIR/SciFact judged test queries, scored against "
        f"the {n_corpus}-document SciFact corpus. Negatives: {args.negatives} sampled "
        "queries per tier from BeIR/quora (easy), BeIR/fiqa (medium), BeIR/nfcorpus "
        "(hard), all scored against the same SciFact corpus.", "",
        "**Caveat on the hard tier.** NFCorpus is biomedical, like SciFact. Some of its "
        "queries may genuinely be answerable from the SciFact corpus -- they are noisy "
        "negatives, not confirmed-unanswerable ones. The hard-tier false-acceptance "
        "rate below is a **pessimistic upper bound**, not a clean error rate.", "",
        "## Similarity distributions", "",
        "| group | mean | median | p5 | p25 | p50 | p75 | p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, key in [("positive (answerable)", "positive"),
                       ("negative -- easy (quora)", "negative_easy"),
                       ("negative -- medium (fiqa)", "negative_medium"),
                       ("negative -- hard (nfcorpus, noisy)", "negative_hard"),
                       ("negative -- pooled", "negative_pooled")]:
        d = distributions[key]
        lines.append(f"| {label} | {d['mean']:.3f} | {d['median']:.3f} | {d['p5']:.3f} | "
                     f"{d['p25']:.3f} | {d['p50']:.3f} | {d['p75']:.3f} | {d['p95']:.3f} |")
    lines.append("")

    lines += ["## ROC-AUC (separating positives from negatives)", "",
              "| tier | AUC | 95% CI | n_pos | n_neg |", "|---|---|---|---|---|"]
    for tier in TIER_ORDER + ["pooled"]:
        r = auc_results[tier]
        lines.append(f"| {tier} | {r['auc']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
                     f"| {r['n_pos']} | {r['n_neg']} |")
    lines.append("")

    lines += ["## Recommended thresholds", "",
              "Largest threshold that still retains at least the given fraction of "
              "answerable queries (<= the complementary false-abstention rate):", "",
              "| retention floor | threshold | retention achieved | FAR easy | FAR medium | FAR hard(noisy) |",
              "|---|---|---|---|---|---|"]
    for label, r in [("99%", RETENTION_TARGETS[0]), ("97.5%", RETENTION_TARGETS[1]),
                     ("95%", RETENTION_TARGETS[2])]:
        t = recommended_threshold(pos_scores, thresholds, r)
        row = _row_at(t)
        lines.append(f"| {label} | {t:.2f} | {row['retention']:.1%} | "
                     f"{row['far_easy']:.1%} | {row['far_medium']:.1%} | {row['far_hard']:.1%} |")
    lines.append("")

    lines += [f"## Where the current gate ({CURRENT_GATE}) sits", "",
              f"At threshold {CURRENT_GATE}: retains **{current_row['retention']:.1%}** of "
              f"answerable queries; admits {current_row['far_easy']:.1%} of easy, "
              f"{current_row['far_medium']:.1%} of medium, and {current_row['far_hard']:.1%} "
              "of hard(noisy) negatives.", "", verdict, "",
              f"**Recommendation: set `Config.retrieval_gate` to {headline_threshold:.2f}** "
              "(largest threshold meeting >=99% retention on this data). This is a "
              "recommendation only -- `src/ragtrust/config.py` is out of scope for this "
              "script and was not modified.", "",
              "## Figure", "", "![gate calibration](gate_calibration.png)", ""]

    print("\n".join(lines))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate_calibration.md").write_text("\n".join(lines) + "\n")
    (OUT_DIR / "gate_calibration.json").write_text(json.dumps(json_out, indent=2))

    # ------------------------------------------------------------------------ figure

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(pos_scores, bins=bins, alpha=0.55, label=f"positive (n={len(pos_scores)})",
            color="#2ca02c", density=True)
    colors = {"easy": "#1f77b4", "medium": "#ff7f0e", "hard": "#d62728"}
    for tier in TIER_ORDER:
        ax.hist(neg_scores_by_tier[tier], bins=bins, alpha=0.45,
                label=f"negative -- {tier} (n={len(neg_scores_by_tier[tier])})",
                color=colors[tier], density=True)
    ax.axvline(CURRENT_GATE, color="black", linestyle="--", linewidth=2,
               label=f"current gate = {CURRENT_GATE:.2f}")
    ax.axvline(headline_threshold, color="black", linestyle=":", linewidth=2,
               label=f"recommended (>=99% retention) = {headline_threshold:.2f}")
    ax.set_xlabel("max cosine similarity to SciFact corpus")
    ax.set_ylabel("density")
    ax.set_title("Retrieval gate calibration: positive vs. negative tiers")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_calibration.png", dpi=150)

    print(f"\nWrote {OUT_DIR / 'gate_calibration.md'}")
    print(f"Wrote {OUT_DIR / 'gate_calibration.json'}")
    print(f"Wrote {OUT_DIR / 'gate_calibration.png'}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
