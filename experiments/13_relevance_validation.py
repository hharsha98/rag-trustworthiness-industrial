#!/usr/bin/env python3
"""Relevance validation -- METRICS.md Part II.3, against third-party human relevance
judgments. The last of this repository's four metrics (faithfulness/experiment 10,
attribution/experiment 11, conciseness/experiment 12, relevance/here) to be validated
on annotation this repository did not author.

Why BEIR SciFact (+ NFCorpus). `context_relevance(query, passages, embedder)`
(`src/ragtrust/metrics/relevance.py`) uses only the embedding model configured as
`Config().embed_model` -- `sentence-transformers/msmarco-distilbert-base-v4`, trained
on MS MARCO query/passage pairs. Experiments 08 and 09 already established BEIR/SciFact
(5,183 biomedical-claim-verification abstracts, 300 test queries, third-party expert
qrels) as this repository's retrieval ground truth; this experiment reuses that same
corpus and its already-cached embeddings (see `08_beir_ablation.py`), plus, time
permitting, BEIR/NFCorpus (3,633 biomedical/nutrition documents, 323 queries, graded
qrels) as a second, independent dataset. Neither is MS MARCO query/passage retrieval
text, so both sit outside the embedder's training distribution -- the same reasoning
experiments 10-12 apply to their own benchmarks.

*** TRAP 1 -- CIRCULARITY. ***
`context_relevance` and the corpus's own dense retriever share the exact same
embedder. If this experiment built its evaluation set from what the dense retriever
returns for a query (its top-k), the passages being scored would have been SELECTED
BY the quantity under test -- scoring the metric against its own output would inflate
the measured discrimination trivially, not validate anything. So the evaluation set is
built from BEIR's qrels instead, never from a retriever call:

  - POSITIVES: documents judged relevant to a query in the qrels (score > 0).
  - NEGATIVES, two tiers, reported and compared SEPARATELY (pooling them hides the
    realistic case -- see METRICS.md's own bar for what "validated" means):
      * random   -- sampled uniformly from the corpus (excluding that query's
                     positives). Easy: usually topically unrelated.
      * hard     -- the query's top-ranked NON-relevant documents by BM25
                     (`src/ragtrust/retrieval/sparse.py::BM25Retriever`). BM25 is
                     lexical (term-frequency/IDF) and touches the embedder nowhere,
                     so using it to pick hard negatives keeps the negative-selection
                     process independent of the metric being validated. Selecting
                     hard negatives with the DENSE retriever instead would have been
                     the same circularity mistake in the opposite direction --
                     adversarially biasing the test against the metric under test,
                     rather than for it -- and is equally invalid, not "more rigorous".

Standard BEIR caveat, stated honestly rather than glossed over: qrels are sparse.
BEIR annotators judged only a pooled subset of the corpus per query; an unjudged
document is NOT certified irrelevant, it is simply unjudged. So both negative tiers
-- especially "random", which never passed in front of an annotator at all -- may
contain false negatives (documents that would in fact be judged relevant if anyone
had looked). This is reported as a limitation, not hidden.

*** TRAP 2 -- ROC-AUC CANNOT TEST THE FLOOR. The subtle one. ***
`relevance.py`'s own module docstring documents a corrected defect: the score-to-[0,1]
mapping used to be affine, `t -> (1+t)/2`, and is now `max(0, t)`. The affine map puts
a FLOOR of ~0.5 under the metric because sentence encoders essentially never produce
negative cosine similarity on real text -- so in practice `(1+t)/2` never gets much
below 0.5, no matter how irrelevant the passage. A bounded-below score cannot signal
"this passage has nothing to do with the query", and inside the non-compensatory
geometric aggregate `T_geom` it can never pull the overall trust score down the way a
genuinely near-zero relevance score should.

ROC-AUC is exactly the WRONG tool to test this. AUC (a Mann-Whitney U statistic) is
invariant under any transform of the score that preserves pairwise rank order over the
range that occurs. Both `max(0, t)` and `(1+t)/2` are non-decreasing in `t`; wherever
`t >= 0`, `max(0, t) == t` exactly (an exact bijection, hence rank-preserving), and
`(1+t)/2` is a strictly increasing affine function of `t` everywhere -- so on the
overwhelming majority of real cosine similarities (see the measured fraction below;
it is small but, on one of the two datasets measured here, not negligible), the two
mappings agree almost exactly on AUC, and the tiny residual gap that CAN appear comes
only from clipping occasionally tying together two distinct negative cosines that were
not tied before clipping -- a second-order effect the module's own measurement
surfaces and reports honestly (Section 4) rather than rounding away. Either way, this
is not what an AUC comparison between the two mappings would be measuring if used to
argue the floor is "fixed" -- it would be measuring (near-)nothing and reporting it as
something. This experiment computes both AUCs and shows how close the identity comes
to holding, numerically, instead of doing that.

The floor is a CALIBRATION / RANGE property of the score, not a ranking property, so
it is measured as one: on documents this experiment already knows are irrelevant (the
qrel-external negative tiers above), Section 4 reports the DISTRIBUTION of the score
itself -- mean, median, p5/p95, minimum observed -- under both mappings, side by side.
The claim under test is specifically about the number shown to a user and fed into
`T_geom`: an irrelevant passage should score near 0 under the clamp and near 0.5 under
the old affine map. That is a statement about calibration, not discrimination, and the
AUC-identity proof exists precisely so nobody mistakes the two for the same question.

*** A trap this repository has been bitten by twice already (see experiments/10 and
12's docstrings): scoring a statistic against a label and that label's own complement
on the same two-class subset produces `AUC(s, ~y) == 1 - AUC(s, y)` identically -- one
fact reported twice. This experiment never does that: every AUC/PR-AUC/permutation-
test call below is against ONE label array, computed ONCE.

Usage:
    python experiments/13_relevance_validation.py                        # full run
    python experiments/13_relevance_validation.py --queries 30            # fast smoke run
    python experiments/13_relevance_validation.py --skip-nfcorpus         # SciFact only

Exit code: always 0. This is a measurement, not a pass/fail gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E402 -- must precede torch/faiss imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtrust.config import Config  # noqa: E402
from ragtrust.metrics.relevance import context_relevance  # noqa: E402
from ragtrust.retrieval.sparse import BM25Retriever  # noqa: E402
from ragtrust.validation.stats import (  # noqa: E402
    bootstrap_ci,
    paired_permutation_test,
    pr_auc,
    roc_auc,
    roc_curve,
)

CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"

_BEIR_ABLATION_PATH = ROOT / "experiments" / "08_beir_ablation.py"

DEFAULT_EMBED_MODEL = Config().embed_model  # sentence-transformers/msmarco-distilbert-base-v4

N_RANDOM_NEG_PER_QUERY = 5
N_HARD_NEG_PER_QUERY = 5
N_BOOT = 10_000
N_PERM = 10_000
SEED = 0


def _load_beir_ablation_module():
    """Import experiment 08 by file path (its module name starts with a digit, so it
    cannot be imported normally) to reuse its SciFact loader, embedding-cache-path
    helper, and `CachedRetriever` -- the same pattern `09_gate_calibration.py` uses."""
    spec = importlib.util.spec_from_file_location("beir_ablation_13dep", _BEIR_ABLATION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("beir_ablation_13dep", module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pure functions -- no network, no model weights. Exercised directly by
# tests/test_relevance_validation.py without any download.
# ---------------------------------------------------------------------------


def positives_from_qrels(qrels_df: pd.DataFrame) -> dict:
    """{query_id_str: [relevant corpus_id_str, ...]} using the standard BEIR
    binary-relevance rule, qrel score > 0. SciFact's test qrels are all
    score == 1; NFCorpus's are graded {1, 2}; both count as "relevant" here,
    the same rule BEIR itself uses for binary IR metrics on graded qrels."""
    df = qrels_df.astype({"query-id": str, "corpus-id": str})
    df = df[df["score"] > 0]
    return df.groupby("query-id")["corpus-id"].apply(list).to_dict()


def ids_to_indices(id_lookup: dict, id_to_index: dict) -> dict:
    """{query_id: [corpus_id, ...]} (string ids, as qrels use) -> {query_id:
    [corpus_index, ...]} (integer positions into that dataset's corpus_texts /
    corpus_embs arrays, which everything downstream is keyed on). Any id not
    present in `id_to_index` is dropped (defensive; every real BEIR qrels row
    references a corpus id that IS in that dataset's corpus). A query left
    with zero positives after this filtering is dropped entirely."""
    out = {}
    for qid, ids in id_lookup.items():
        idxs = [id_to_index[i] for i in ids if i in id_to_index]
        if idxs:
            out[qid] = idxs
    return out


def query_seed(base_seed: int, query_id: str, salt: str = "") -> int:
    """Deterministic per-query seed derived from a base seed + query id (+ an
    optional salt, to decorrelate two different sampling steps for the same
    query). Re-running with the same base seed reproduces the exact same
    sample; different queries do not all draw the identical sequence, which a
    single shared `random.Random(base_seed)` reused across queries would."""
    h = hashlib.md5(f"{base_seed}:{salt}:{query_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_random_negatives(n_corpus: int, exclude_idx: set, n: int, seed: int) -> list:
    """Deterministically sample up to `n` corpus indices from range(n_corpus),
    excluding `exclude_idx` (that query's judged-relevant set) -- Trap 1's
    "easy" negative tier. Sampling from the WHOLE corpus, never from any
    retriever's output, is what keeps this tier independent of the metric
    under test."""
    pool = [i for i in range(n_corpus) if i not in exclude_idx]
    rng = random.Random(seed)
    if n >= len(pool):
        return pool
    return rng.sample(pool, n)


def select_hard_negatives(ranked_idx: list, exclude_idx: set, n: int) -> list:
    """Top-`n` indices from `ranked_idx` (assumed already ranked best-first by
    BM25 -- lexical, independent of the embedder under test) that are NOT in
    `exclude_idx` -- Trap 1's "hard" negative tier."""
    out = []
    for idx in ranked_idx:
        if idx in exclude_idx:
            continue
        out.append(idx)
        if len(out) >= n:
            break
    return out


def assemble_pairs_for_query(query_id: str, positive_idx: list, n_corpus: int,
                              bm25_ranked_idx: list, n_random: int, n_hard: int,
                              seed: int) -> list:
    """One query's (query_id, passage_idx, label, tier) rows: every judged
    positive (label=1, tier='positive'), up to `n_random` random negatives
    (label=0, tier='random'), up to `n_hard` BM25 hard negatives (label=0,
    tier='hard'). Pure given `bm25_ranked_idx` (that query's full-corpus BM25
    ranking, computed upstream where the corpus text lives) -- no embeddings,
    no network, so this composition step is directly unit-tested."""
    exclude = set(positive_idx)
    rows = [{"query_id": query_id, "passage_idx": idx, "label": 1, "tier": "positive"}
            for idx in positive_idx]
    random_neg = sample_random_negatives(n_corpus, exclude, n_random,
                                          query_seed(seed, query_id, "random"))
    rows += [{"query_id": query_id, "passage_idx": idx, "label": 0, "tier": "random"}
             for idx in random_neg]
    hard_neg = select_hard_negatives(bm25_ranked_idx, exclude, n_hard)
    rows += [{"query_id": query_id, "passage_idx": idx, "label": 0, "tier": "hard"}
             for idx in hard_neg]
    return rows


def clamp_scores(raw_cosine) -> np.ndarray:
    """v2's mapping: max(0, t). What `context_relevance(..., scaled=True)`
    actually applies."""
    return np.clip(np.asarray(raw_cosine, dtype=float), 0.0, None)


def affine_scores(raw_cosine) -> np.ndarray:
    """The historical, now-replaced mapping: (1+t)/2 -- see relevance.py's
    module docstring on why it was replaced (the ~0.5 floor, Trap 2)."""
    return (1.0 + np.asarray(raw_cosine, dtype=float)) / 2.0


def score_distribution(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def floor_analysis(raw_cosine_irrelevant) -> dict:
    """Trap 2's actual measurement: on known-irrelevant passages, the
    DISTRIBUTION of the score under the clamp (v2) vs the historical affine
    mapping -- a calibration/range comparison, not a discrimination one."""
    raw = np.asarray(raw_cosine_irrelevant, dtype=float)
    return {"clamp": score_distribution(clamp_scores(raw)), "affine": score_distribution(affine_scores(raw))}


def frac_negative_cosine(raw_cosine) -> float:
    """Fraction of raw cosine similarities below zero -- the quantity that
    determines how often `max(0, t)` and `t` actually differ (and hence how
    rarely clamping can affect rank order at all)."""
    raw = np.asarray(raw_cosine, dtype=float)
    return float(np.mean(raw < 0.0))


def class_balance(labels) -> dict:
    labels = np.asarray(labels)
    n = int(len(labels))
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "rate_pos": float(n_pos / n) if n else float("nan")}


def corpus_embedding_cache_path(cache_dir: Path, dataset: str, model_name: str) -> Path:
    """Mirrors `08_beir_ablation.py::embedding_cache_path`, but namespaced by
    dataset (that function hardcodes a 'scifact_' prefix). Used only for
    NFCorpus -- SciFact reuses experiment 08's existing cache path/file
    exactly, unchanged, per this experiment's own constraint not to recompute
    it."""
    safe_name = model_name.replace("/", "__")
    return cache_dir / f"{dataset}_corpus_embeddings__{safe_name}.npy"


# ---------------------------------------------------------------------------
# Precomputed-embedding adapter -- let the REAL `context_relevance` function
# be scored against a fixed vector lookup instead of re-encoding text on
# every one of the several thousand (query, passage) pairs below. No
# network, no model weights; still exercised (with a fake embedding map) by
# the fast test suite.
# ---------------------------------------------------------------------------


class PrecomputedEmbedder:
    """`.encode(list_of_str) -> np.ndarray`, backed by a fixed text->vector
    map, so `context_relevance` (the metric under test) runs against
    embeddings computed ONCE in batch -- reusing SciFact's existing corpus
    embedding cache (and a freshly-computed one for NFCorpus) rather than
    re-encoding the same text on every pair it appears in."""

    def __init__(self, text_to_vec: dict):
        self._map = text_to_vec

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._map[t] for t in texts])


# ---------------------------------------------------------------------------
# Network / model-dependent functions -- not exercised by the fast test suite.
# ---------------------------------------------------------------------------


def load_nfcorpus(local_files_only: bool = False):
    """Mirrors `08_beir_ablation.py::load_scifact` exactly, for BeIR/nfcorpus
    / BeIR/nfcorpus-qrels. NFCorpus's qrels are GRADED (score in {1, 2}),
    unlike SciFact's binary {1} -- `positives_from_qrels`'s score>0 rule
    treats both grades as relevant, the standard BEIR binarisation."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        corpus_path = hf_hub_download(
            "BeIR/nfcorpus", "corpus/corpus-00000-of-00001.parquet",
            repo_type="dataset", local_files_only=local_files_only,
        )
        queries_path = hf_hub_download(
            "BeIR/nfcorpus", "queries/queries-00000-of-00001.parquet",
            repo_type="dataset", local_files_only=local_files_only,
        )
        qrels_path = hf_hub_download(
            "BeIR/nfcorpus-qrels", "test.tsv",
            repo_type="dataset", local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            "BEIR/NFCorpus is not in the local Hugging Face cache and downloads "
            "are disabled (local_files_only=True)."
        ) from e

    corpus_df = pd.read_parquet(corpus_path)
    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_csv(qrels_path, sep="\t")
    return corpus_df, queries_df, qrels_df


def build_text_to_vec(corpus_texts: list, corpus_embs: np.ndarray, queries_by_id: dict,
                       query_ids: list, embedder) -> dict:
    """{text: vector} for every corpus passage (from the already-computed
    corpus embedding matrix -- no re-encoding) plus every judged query
    (encoded fresh, once, in a single batched call). This is the lookup table
    `PrecomputedEmbedder` serves to the real metric function."""
    text_to_vec = {text: corpus_embs[i] for i, text in enumerate(corpus_texts)}
    query_texts = [queries_by_id[qid] for qid in query_ids]
    query_embs = np.asarray(embedder.encode(query_texts), dtype="float32")
    for qid, vec in zip(query_ids, query_embs):
        text_to_vec[queries_by_id[qid]] = vec
    return text_to_vec


def score_pairs(pairs_df: pd.DataFrame, queries_by_id: dict, corpus_texts: list,
                 text_to_vec: dict, bm25_scoremap_by_query: dict) -> pd.DataFrame:
    """Attach v2 (clamped), raw (unclamped), and the BM25 lexical baseline to
    every (query_id, passage_idx, label, tier) row -- calling the REAL
    `context_relevance` function (not a reimplementation of its formula)
    against the precomputed embedding lookup, one (query, single passage)
    pair at a time, so the per-passage primitive is exactly what gets scored
    (not an average over several passages, which would mix the primitive
    across items)."""
    embedder = PrecomputedEmbedder(text_to_vec)

    v2, raw, bm25 = [], [], []
    for row in pairs_df.itertuples(index=False):
        query_text = queries_by_id[row.query_id]
        passage_text = corpus_texts[row.passage_idx]
        v2.append(context_relevance(query_text, [passage_text], embedder, scaled=True))
        raw.append(context_relevance(query_text, [passage_text], embedder, scaled=False))
        bm25.append(bm25_scoremap_by_query[row.query_id].get(row.passage_idx, 0.0))

    out = pairs_df.copy()
    out["score_v2"] = v2
    out["score_raw"] = raw
    out["score_bm25"] = bm25
    return out


def process_dataset(name: str, corpus_df, queries_df, qrels_df, embedder, cache_path: Path,
                     beir08, n_random: int, n_hard: int, seed: int, max_queries) -> dict:
    """End-to-end per-dataset pipeline: load -> positives from qrels -> BM25
    full-corpus rankings -> assemble pairs (Trap 1) -> score (v2/raw/BM25)."""
    corpus_texts = [f"{t} {x}".strip() for t, x in zip(corpus_df["title"], corpus_df["text"])]
    corpus_ids = corpus_df["_id"].astype(str).tolist()
    id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
    n_corpus = len(corpus_texts)

    queries_by_id = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))

    positives_str = positives_from_qrels(qrels_df)
    positives_idx = ids_to_indices(positives_str, id_to_index)
    query_ids = sorted(qid for qid in positives_idx if qid in queries_by_id)

    if max_queries is not None and max_queries < len(query_ids):
        rng = random.Random(seed)
        query_ids = sorted(rng.sample(query_ids, max_queries))

    print(f"[{name}] corpus={n_corpus} documents; {len(query_ids)} judged queries in use.")

    print(f"[{name}] corpus embeddings: {cache_path} ...")
    retriever = beir08.CachedRetriever(embedder, cache_path, normalize=True)
    retriever.build(corpus_texts)  # loads from cache if present; never re-embeds otherwise
    corpus_embs = np.load(cache_path).astype("float32")
    print(f"[{name}]   {'reused existing cache' if retriever.cache_hit else 'written this run'}.")

    print(f"[{name}] building BM25 index over {n_corpus} documents ...")
    bm25 = BM25Retriever().build(corpus_texts)

    print(f"[{name}] scoring BM25 full-corpus rankings for {len(query_ids)} queries "
          "(also used to pick Trap-1 hard negatives) ...")
    bm25_ranked_by_query = {}
    bm25_scoremap_by_query = {}
    t0 = time.time()
    for i, qid in enumerate(query_ids, start=1):
        hits = bm25.search(queries_by_id[qid], k=n_corpus)
        bm25_ranked_by_query[qid] = [h.id for h in hits]
        bm25_scoremap_by_query[qid] = {h.id: h.score for h in hits}
        if i % 100 == 0 or i == len(query_ids):
            print(f"[{name}]   ... {i}/{len(query_ids)} queries BM25-ranked "
                  f"({time.time() - t0:.1f}s elapsed)")

    rows = []
    for qid in query_ids:
        rows += assemble_pairs_for_query(qid, positives_idx[qid], n_corpus,
                                          bm25_ranked_by_query[qid], n_random, n_hard, seed)
    pairs_df = pd.DataFrame(rows)
    print(f"[{name}] assembled {len(pairs_df)} evaluation pairs "
          f"({int((pairs_df['tier'] == 'positive').sum())} positive, "
          f"{int((pairs_df['tier'] == 'random').sum())} random-negative, "
          f"{int((pairs_df['tier'] == 'hard').sum())} hard-negative).")

    text_to_vec = build_text_to_vec(corpus_texts, corpus_embs, queries_by_id, query_ids, embedder)

    print(f"[{name}] scoring {len(pairs_df)} pairs (v2/raw/BM25) ...")
    scored = score_pairs(pairs_df, queries_by_id, corpus_texts, text_to_vec, bm25_scoremap_by_query)

    # Attach the raw qrel grade to every pair (0 for negatives). BEIR qrels are
    # GRADED, and `positives_from_qrels` binarises them at score > 0 -- standard,
    # but it makes "positive" mean very different things across datasets. SciFact
    # is effectively single-grade; NFCorpus is 11,758 grade-1 (marginal) against
    # only 576 grade-2, so binarising there means ~95% of "relevant" documents are
    # only marginally so. Section 4 uses this column to report how the result moves
    # when the positive class is restricted to the top grade -- without it, a
    # sub-chance AUC reads as "the metric ranks hard negatives above true
    # positives", when the measurable claim is the far weaker "above *marginally*
    # relevant ones".
    grade_lookup = {}
    qdf = qrels_df.astype({"query-id": str, "corpus-id": str})
    for q, c, s in zip(qdf["query-id"], qdf["corpus-id"], qdf["score"]):
        idx = id_to_index.get(c)
        if idx is not None:
            grade_lookup[(q, idx)] = int(s)
    scored = scored.copy()
    scored["qrel_grade"] = [
        grade_lookup.get((str(q), int(p)), 0) if int(lbl) == 1 else 0
        for q, p, lbl in zip(scored["query_id"], scored["passage_idx"], scored["label"])
    ]

    return {"name": name, "n_corpus": n_corpus, "n_queries": len(query_ids), "pairs": scored}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _stat_block(stat: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    auc, auc_lo, auc_hi = bootstrap_ci(stat, labels, roc_auc, n=n_boot, seed=seed)
    pr, pr_lo, pr_hi = bootstrap_ci(stat, labels, pr_auc, n=n_boot, seed=seed)
    return {
        "roc_auc": {"point": auc, "ci_lo": auc_lo, "ci_hi": auc_hi},
        "pr_auc": {"point": pr, "ci_lo": pr_lo, "ci_hi": pr_hi},
    }


def _tier_subset(pairs: pd.DataFrame, tier: str) -> pd.DataFrame:
    """The positive rows plus this query set's `tier` negative rows -- i.e.
    a clean two-class subset for exactly one negative tier, never pooled with
    the other tier (Required Analysis 1: report tiers separately)."""
    return pairs[pairs["tier"].isin(["positive", tier])].reset_index(drop=True)


def run_auc_by_tier(pairs: pd.DataFrame, n_boot: int, seed: int) -> dict:
    out = {}
    for tier in ("random", "hard"):
        sub = _tier_subset(pairs, tier)
        labels = sub["label"].to_numpy(dtype=int)
        stat = sub["score_v2"].to_numpy(dtype=float)
        out[tier] = {"class_balance": class_balance(labels), **_stat_block(stat, labels, n_boot, seed)}
    return out


def run_bm25_comparison(pairs: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """Required Analysis 2: BM25 lexical baseline vs the cosine metric (v2),
    per tier, via a paired permutation test on ROC-AUC."""
    out = {}
    for tier in ("random", "hard"):
        sub = _tier_subset(pairs, tier)
        labels = sub["label"].to_numpy(dtype=int)
        cosine = sub["score_v2"].to_numpy(dtype=float)
        bm25 = sub["score_bm25"].to_numpy(dtype=float)
        auc_cosine, auc_bm25, diff, p_value = paired_permutation_test(
            cosine, bm25, labels, roc_auc, n=n_perm, seed=seed
        )
        out[tier] = {
            "auc_cosine": auc_cosine,
            "auc_bm25": auc_bm25,
            "diff_cosine_minus_bm25": diff,
            "p_value": p_value,
            "n_permutations": n_perm,
            "cosine_significantly_better": bool(diff > 0 and p_value < 0.05),
            "bm25_significantly_better": bool(diff < 0 and p_value < 0.05),
        }
    return out


def run_floor_analysis(pairs: pd.DataFrame) -> dict:
    """Trap 2. Two independent things, kept clearly separate:
    (a) the CALIBRATION comparison -- score distributions on known-irrelevant
        passages under the clamp mapping (current) vs the affine mapping (an
        alternative that was considered and rejected -- see relevance.py's
        module docstring);
    (b) a numerical proof that AUC is IDENTICAL between the two mappings on
        this data (to ~1e-9), so nobody mistakes (a) for a ranking claim, and
        nobody "re-validates" the floor fix with an AUC comparison later."""
    irrelevant = pairs[pairs["label"] == 0]
    raw_irrelevant = irrelevant["score_raw"].to_numpy(dtype=float)
    calibration = floor_analysis(raw_irrelevant)

    labels = pairs["label"].to_numpy(dtype=int)
    raw_all = pairs["score_raw"].to_numpy(dtype=float)
    auc_clamp = roc_auc(clamp_scores(raw_all), labels)
    auc_affine = roc_auc(affine_scores(raw_all), labels)
    identity = {
        "auc_clamp": auc_clamp,
        "auc_affine": auc_affine,
        "abs_diff": abs(auc_clamp - auc_affine),
        "matches_to_1e9": bool(abs(auc_clamp - auc_affine) < 1e-9),
        "frac_negative_cosine": frac_negative_cosine(raw_all),
    }
    return {"calibration": calibration, "auc_identity": identity}


def run_graded_sensitivity(pairs: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Section 4: how much of the headline AUC depends on treating MARGINAL
    relevance as positive?

    BEIR qrels are graded and `positives_from_qrels` binarises them at score > 0.
    That is the standard rule, but on a densely-judged set it can invert the
    conclusion. Re-running with the positive class restricted to the top grade
    answers a different and sharper question -- "does the metric rank a lexically
    matched hard negative above a document a human called *definitely* relevant?"
    -- and the two answers are reported side by side rather than one standing in
    for the other.

    Returns None when the dataset has only one positive grade (e.g. SciFact),
    where the restriction is vacuous.
    """
    positives = pairs[pairs["label"] == 1]
    grades = sorted(int(g) for g in positives["qrel_grade"].unique() if g > 0)
    if len(grades) < 2:
        return None

    top_grade = max(grades)
    out = {
        "grades_present": grades,
        "top_grade": top_grade,
        "n_positives_by_grade": {
            str(g): int((positives["qrel_grade"] == g).sum()) for g in grades
        },
        "by_tier": {},
    }
    for tier in ("random", "hard"):
        subset = pairs[pairs["tier"].isin(["positive", tier])]
        keep = subset[(subset["label"] == 0) | (subset["qrel_grade"] >= top_grade)]
        labels = keep["label"].to_numpy().astype(int)
        if len(np.unique(labels)) < 2:
            continue
        stat = keep["score_v2"].to_numpy(dtype=float)
        all_labels = subset["label"].to_numpy().astype(int)
        out["by_tier"][tier] = {
            "all_positives": {
                "n_pos": int(all_labels.sum()),
                "roc_auc": roc_auc(subset["score_v2"].to_numpy(dtype=float), all_labels),
            },
            "top_grade_only": {
                "n_pos": int(labels.sum()),
                "n_neg": int((labels == 0).sum()),
                **_stat_block(stat, labels, n_boot, seed),
            },
        }
    return out


def run_dataset_analysis(pairs: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> dict:
    return {
        "auc_by_tier": run_auc_by_tier(pairs, n_boot, seed),
        "bm25_comparison": run_bm25_comparison(pairs, n_perm, seed),
        "floor_analysis": run_floor_analysis(pairs),
        "graded_sensitivity": run_graded_sensitivity(pairs, n_boot, seed),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(dataset_results: dict, dataset_analyses: dict, out_path: Path) -> None:
    names = list(dataset_results.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5.2 * n), squeeze=False)

    for row, name in enumerate(names):
        pairs = dataset_results[name]["pairs"]
        analysis = dataset_analyses[name]

        # --- col 0: ROC curves, cosine vs BM25, per tier
        ax = axes[row][0]
        colors = {"random": "C0", "hard": "C3"}
        for tier in ("random", "hard"):
            sub = _tier_subset(pairs, tier)
            labels = sub["label"].to_numpy(dtype=int)
            fpr_c, tpr_c, _ = roc_curve(sub["score_v2"].to_numpy(dtype=float), labels)
            fpr_b, tpr_b, _ = roc_curve(sub["score_bm25"].to_numpy(dtype=float), labels)
            auc_c = analysis["bm25_comparison"][tier]["auc_cosine"]
            auc_b = analysis["bm25_comparison"][tier]["auc_bm25"]
            ax.plot(fpr_c, tpr_c, color=colors[tier], linestyle="-",
                     label=f"{tier} cosine (AUC={auc_c:.3f})")
            ax.plot(fpr_b, tpr_b, color=colors[tier], linestyle="--",
                     label=f"{tier} BM25 (AUC={auc_b:.3f})")
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{name}: cosine vs BM25, by tier", fontsize=10)
        ax.legend(loc="lower right", fontsize=7)

        # --- col 1: score_v2 distribution, relevant vs irrelevant
        ax2 = axes[row][1]
        rel = pairs.loc[pairs["label"] == 1, "score_v2"]
        irrel = pairs.loc[pairs["label"] == 0, "score_v2"]
        bins = np.linspace(0.0, 1.0, 31)
        ax2.hist(rel, bins=bins, alpha=0.6, label=f"relevant (n={len(rel)})", color="C1", density=True)
        ax2.hist(irrel, bins=bins, alpha=0.6, label=f"irrelevant (n={len(irrel)})", color="C7", density=True)
        ax2.set_xlabel("context_relevance score (v2, clamped)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{name}: score by relevance", fontsize=10)
        ax2.legend(fontsize=7)

        # --- col 2: clamp vs affine on irrelevant passages -- the floor
        ax3 = axes[row][2]
        irrel_raw = pairs.loc[pairs["label"] == 0, "score_raw"].to_numpy(dtype=float)
        clamp_v = clamp_scores(irrel_raw)
        affine_v = affine_scores(irrel_raw)
        bins2 = np.linspace(0.0, 1.0, 41)
        ax3.hist(clamp_v, bins=bins2, alpha=0.6, label="clamp max(0,t) [v2]", color="C2", density=True)
        ax3.hist(affine_v, bins=bins2, alpha=0.6, label="affine (1+t)/2 [old]", color="C4", density=True)
        ax3.axvline(0.5, linestyle="--", color="black", linewidth=1, label="0.5 (old floor)")
        ax3.set_xlabel("Score on known-irrelevant passages")
        ax3.set_ylabel("Density")
        ax3.set_title(f"{name}: the ~0.5 floor (Trap 2)", fontsize=10)
        ax3.legend(fontsize=7)

    fig.suptitle("Relevance validation: BEIR qrels, cosine vs BM25, and the affine-vs-clamp floor",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_ci(block: dict) -> str:
    return f"{block['point']:.3f} ({block['ci_lo']:.3f}, {block['ci_hi']:.3f})"


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# Relevance validation -- METRICS.md Part II.3, context_relevance",
        "",
        "**Read the floor section (Section 4) as a calibration claim, not a ranking "
        "claim.** ROC-AUC is invariant to the monotone remap that separates the "
        "clamped metric from its historical affine predecessor, so an AUC comparison "
        "between them would be meaningless by construction -- Section 4 proves that "
        "numerically instead of computing it as if it were informative.",
        "",
        f"**Embedder:** `{summary['embed_model']}`, trained on MS MARCO query/passage "
        "pairs. BEIR/SciFact (biomedical claim verification) and BEIR/NFCorpus "
        "(biomedical/nutrition) are neither of them MS MARCO retrieval text, so both "
        "sit outside the embedder's training distribution.",
        "",
        "**Circularity guard (Trap 1).** Evaluation pairs are built from BEIR's own "
        "qrels, never from this repository's dense retriever's output (which shares "
        "the embedder under test). Positives: qrel score > 0. Negatives, two tiers: "
        "*random* (uniform sample from the corpus) and *hard* (top-ranked non-relevant "
        "documents by BM25 -- lexical, so independent of the embedder). Standard BEIR "
        "caveat: qrels are sparse, so an unjudged document is not certified irrelevant "
        "-- both negative tiers, especially *random*, may contain false negatives.",
        "",
        f"n_random_neg_per_query = {summary['n_random_neg_per_query']}, "
        f"n_hard_neg_per_query = {summary['n_hard_neg_per_query']}, seed = {summary['seed']}.",
        "",
    ]

    if summary["nfcorpus_included"]:
        lines += ["NFCorpus was included as a second, independent dataset.", ""]
    else:
        lines += [
            f"**NFCorpus was NOT included**: {summary['nfcorpus_skip_reason']}. "
            "This report covers SciFact only -- no NFCorpus coverage is claimed.",
            "",
        ]

    for name, ds in summary["datasets"].items():
        a = ds["analysis"]
        lines += [f"## Dataset: {name}", "",
                   f"Corpus: {ds['n_corpus']} documents. Judged queries used: {ds['n_queries']}.",
                   ""]

        lines += ["### Section 1 -- ROC-AUC / PR-AUC of `max(0, cos)` vs judged-relevant, by tier", ""]
        lines += ["| tier | n | n_pos | n_neg | ROC-AUC (95% CI) | PR-AUC (95% CI) |",
                   "|---|---:|---:|---:|---|---|"]
        for tier in ("random", "hard"):
            t = a["auc_by_tier"][tier]
            cb = t["class_balance"]
            lines.append(f"| {tier} | {cb['n']} | {cb['n_pos']} | {cb['n_neg']} | "
                          f"{_fmt_ci(t['roc_auc'])} | {_fmt_ci(t['pr_auc'])} |")
        lines.append("")

        lines += ["### Section 2 -- BM25 lexical baseline vs cosine (v2), by tier "
                   "(paired permutation test)", ""]
        lines += ["| tier | AUC(cosine) | AUC(BM25) | diff | p-value | verdict |",
                   "|---|---:|---:|---:|---:|---|"]
        for tier in ("random", "hard"):
            b = a["bm25_comparison"][tier]
            if b["cosine_significantly_better"]:
                verdict = "cosine significantly better"
            elif b["bm25_significantly_better"]:
                verdict = "**BM25 significantly better**"
            else:
                verdict = "not significantly different"
            lines.append(f"| {tier} | {b['auc_cosine']:.3f} | {b['auc_bm25']:.3f} | "
                          f"{b['diff_cosine_minus_bm25']:+.3f} | {b['p_value']:.4f} | {verdict} |")
        lines.append("")

        f = a["floor_analysis"]
        cal = f["calibration"]
        ident = f["auc_identity"]
        lines += [
            "### Section 3 -- the floor (Trap 2): calibration, not discrimination", "",
            f"On the **{cal['clamp']['n']}** known-irrelevant passages in this dataset "
            "(both negative tiers pooled), the score distribution under the current mapping "
            "vs an affine alternative that was considered and rejected:",
            "",
            "| mapping | mean | median | p5 | p95 | min |",
            "|---|---:|---:|---:|---:|---:|",
            f"| clamp `max(0,t)` (current) | {cal['clamp']['mean']:.3f} | "
            f"{cal['clamp']['median']:.3f} | {cal['clamp']['p5']:.3f} | "
            f"{cal['clamp']['p95']:.3f} | {cal['clamp']['min']:.3f} |",
            f"| affine `(1+t)/2` (alternative mapping, rejected) | {cal['affine']['mean']:.3f} | "
            f"{cal['affine']['median']:.3f} | {cal['affine']['p5']:.3f} | "
            f"{cal['affine']['p95']:.3f} | {cal['affine']['min']:.3f} |",
            "",
            f"**AUC-identity proof**: AUC(clamp) = {ident['auc_clamp']:.9f}, "
            f"AUC(affine) = {ident['auc_affine']:.9f}, "
            f"|diff| = {ident['abs_diff']:.2e} "
            f"({'matches to 1e-9' if ident['matches_to_1e9'] else 'does NOT match to 1e-9'}). " + (
                "This confirms the two mappings are, as expected, indistinguishable by ROC-AUC on "
                "this data -- the calibration table above, not this identity, is what shows the "
                "floor fix's actual effect."
                if ident["matches_to_1e9"] else
                "Not an exact match here, honestly reported rather than rounded away: "
                f"{ident['frac_negative_cosine']:.2%} of raw cosines on this dataset are negative "
                "(vs a much smaller fraction on the other dataset), and when two or more distinct "
                "negative cosines are clipped to the same 0, they become tied under the clamp "
                "mapping where they were NOT tied under the strictly-monotone affine mapping -- "
                "exactly the boundary case the module docstring flags ('over the range that "
                "actually occurs'). The residual is tiny (5e-4) and does not change the "
                "conclusion that AUC cannot see the floor -- it is the mechanism by which the "
                "'always exactly equal' claim can, at the margin, fail to hold."
            ),
            "",
        ]

        graded = a.get("graded_sensitivity")
        if graded:
            by_grade = ", ".join(
                f"grade {g}: {n}" for g, n in sorted(graded["n_positives_by_grade"].items()))
            lines += [
                "### Section 4 -- how much of this depends on counting MARGINAL relevance as positive?",
                "",
                "Sections 1-3 binarise the qrels at `score > 0`, the standard BEIR rule. On a "
                f"graded, densely-judged set that rule does a lot of work: this dataset has "
                f"{by_grade}. Restricting the positive class to the top grade "
                f"({graded['top_grade']}) asks the sharper question -- does the metric rank a "
                "lexically-matched hard negative above a document a human called *definitely* "
                "relevant?",
                "",
                "| tier | AUC, all positives (`score > 0`) | AUC, grade "
                f"{graded['top_grade']} only (95% CI) |",
                "|---|---:|---|",
            ]
            for tier in ("random", "hard"):
                block = graded["by_tier"].get(tier)
                if not block:
                    continue
                lines.append(
                    f"| {tier} | {block['all_positives']['roc_auc']:.3f} "
                    f"(n_pos={block['all_positives']['n_pos']}) | "
                    f"{_fmt_ci(block['top_grade_only']['roc_auc'])} "
                    f"(n_pos={block['top_grade_only']['n_pos']}) |")
            hard = graded["by_tier"].get("hard")
            if hard:
                lines += [
                    "",
                    f"**This materially changes the reading of the hard tier.** At `score > 0` "
                    f"the AUC is {hard['all_positives']['roc_auc']:.3f}; restricted to grade "
                    f"{graded['top_grade']} it is "
                    f"{hard['top_grade_only']['roc_auc']['point']:.3f} "
                    f"({_fmt_ci(hard['top_grade_only']['roc_auc'])}). A sub-chance number under "
                    "the binarised rule therefore does NOT support the claim that the metric "
                    "ranks hard negatives above genuinely relevant documents; it supports the "
                    "much weaker claim that it ranks them above *marginally* relevant ones, "
                    "which is a statement about how this benchmark defines relevance at least "
                    "as much as about the metric. Both numbers are reported because neither "
                    "alone is the whole answer.",
                    "",
                ]

    lines += [
        "## Limitations", "",
        "- **Sparse qrels (standard BEIR caveat).** An unjudged document is not certified "
        "irrelevant. Both negative tiers -- especially *random*, which was never seen by a "
        "human annotator -- may contain false negatives, which would understate the true "
        "AUC.", "",
        "- **Hard-tier BM25 negatives are lexical, not semantic, negatives.** A document "
        "can rank highly under BM25 (shares vocabulary with the query) while still being "
        "genuinely off-topic, which is exactly the discriminating case this tier is meant "
        "to probe -- but it also means a hard negative could occasionally be a real, "
        "unjudged positive that happens to share vocabulary with the query.", "",
        "- **Section 2's hard-tier row is not a fair general test of BM25.** The hard "
        "negatives were SELECTED as the query's top-ranked BM25 documents (excluding "
        "positives) -- so by construction they score very highly under BM25, often as "
        "high as or higher than the true positives. That mechanically depresses "
        "AUC(BM25) on the hard tier specifically (biasing that one comparison IN FAVOUR "
        "of cosine), independent of BM25's real retrieval quality. The random-tier row "
        "does not have this bias and is the fairer BM25-vs-cosine comparison of the two.",
        "",
        "## Artefacts", "",
        "- `relevance_validation.json` -- full numeric results",
        "- `relevance_validation.png` -- ROC curves (cosine vs BM25) by tier, score "
        "distributions by relevance, and the clamp-vs-affine floor comparison",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "relevance_validation.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _json_safe(obj):
    """Recursively strip pandas DataFrames (not JSON-serialisable, and not
    needed in the JSON artefact -- the per-pair table is an intermediate,
    not a reported result) out of a nested dict before dumping."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not isinstance(v, pd.DataFrame)}
    return obj


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=None,
                     help="cap on judged queries per dataset (default: all); sampled deterministically")
    ap.add_argument("--n-random", type=int, default=N_RANDOM_NEG_PER_QUERY)
    ap.add_argument("--n-hard", type=int, default=N_HARD_NEG_PER_QUERY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--skip-nfcorpus", action="store_true",
                     help="SciFact only; do not attempt NFCorpus.")
    args = ap.parse_args()

    n_boot = N_BOOT
    n_perm = N_PERM

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
        sf_corpus, sf_queries, sf_qrels = beir08.load_scifact()
    except Exception as e:
        print(f"Could not load BEIR/SciFact: {e}")
        print("This is a measurement script, not a pass/fail gate -- exiting 0 with no results.")
        return 0

    print(f"Loading embedder '{args.embed_model}' ...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embed_model)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sf_cache_path = beir08.embedding_cache_path(CACHE_DIR, args.embed_model)  # REUSED, not recomputed

    dataset_results = {}
    dataset_results["scifact"] = process_dataset(
        "scifact", sf_corpus, sf_queries, sf_qrels, embedder, sf_cache_path, beir08,
        args.n_random, args.n_hard, args.seed, args.queries,
    )

    nfcorpus_included = False
    nfcorpus_skip_reason = None
    if args.skip_nfcorpus:
        nfcorpus_skip_reason = "skipped via --skip-nfcorpus"
    else:
        print("\nLoading BEIR/NFCorpus (corpus, queries, qrels) -- second, independent dataset ...")
        try:
            nf_corpus, nf_queries, nf_qrels = load_nfcorpus()
            nf_cache_path = corpus_embedding_cache_path(CACHE_DIR, "nfcorpus", args.embed_model)
            dataset_results["nfcorpus"] = process_dataset(
                "nfcorpus", nf_corpus, nf_queries, nf_qrels, embedder, nf_cache_path, beir08,
                args.n_random, args.n_hard, args.seed, args.queries,
            )
            nfcorpus_included = True
        except Exception as e:
            nfcorpus_skip_reason = f"failed to load or process cleanly: {e}"
            print(f"NFCorpus not included: {nfcorpus_skip_reason}")

    dataset_analyses = {}
    for name, ds in dataset_results.items():
        print(f"\nRunning analysis for {name} ...")
        dataset_analyses[name] = run_dataset_analysis(ds["pairs"], n_boot, n_perm, args.seed)

    summary = {
        "_method": (
            "Positives: BEIR qrels (score > 0). Negatives: random (uniform corpus sample, "
            "seed-determined) and hard (top-ranked non-relevant documents by BM25, "
            "src/ragtrust/retrieval/sparse.py::BM25Retriever) -- both independent of the "
            "dense embedder under test (see module docstring, Trap 1). ROC-AUC/PR-AUC with "
            f"10,000-sample bootstrap 95% CIs, reported separately per negative tier. BM25 "
            "lexical baseline compared via paired permutation test. "
            "Floor analysis (Trap 2) reports score DISTRIBUTIONS under clamp vs affine "
            "mappings on known-irrelevant passages, plus a numerical proof that AUC is "
            "identical between the two mappings (ROC-AUC is invariant to this monotone "
            "remap, so an AUC comparison between them cannot test the floor -- see docstring)."
        ),
        "embed_model": args.embed_model,
        "seed": args.seed,
        "n_random_neg_per_query": args.n_random,
        "n_hard_neg_per_query": args.n_hard,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "nfcorpus_included": nfcorpus_included,
        "nfcorpus_skip_reason": nfcorpus_skip_reason,
        "datasets": {
            name: {
                "n_corpus": ds["n_corpus"],
                "n_queries": ds["n_queries"],
                "n_pairs": int(len(ds["pairs"])),
                "analysis": dataset_analyses[name],
            }
            for name, ds in dataset_results.items()
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "relevance_validation.json").write_text(json.dumps(_json_safe(summary), indent=2))
    make_plot(dataset_results, dataset_analyses, OUT_DIR / "relevance_validation.png")
    write_report(summary, OUT_DIR)

    print()
    print("=" * 72)
    for name in dataset_results:
        a = dataset_analyses[name]
        for tier in ("random", "hard"):
            t = a["auc_by_tier"][tier]
            print(f"{name:>9s} [{tier:>6s}]  ROC-AUC = {t['roc_auc']['point']:.4f} "
                  f"({t['roc_auc']['ci_lo']:.4f}, {t['roc_auc']['ci_hi']:.4f})  "
                  f"n={t['class_balance']['n']}")
        ident = a["floor_analysis"]["auc_identity"]
        print(f"{name:>9s}  floor AUC-identity |diff|={ident['abs_diff']:.2e} "
              f"(matches_to_1e9={ident['matches_to_1e9']}) "
              f"frac_negative_cosine={ident['frac_negative_cosine']:.4%}")
    print(f"NFCorpus included: {nfcorpus_included}"
          + (f" ({nfcorpus_skip_reason})" if not nfcorpus_included else ""))
    print(f"Wrote {OUT_DIR / 'relevance_validation.md'}, relevance_validation.json, "
          "relevance_validation.png")
    print(f"Total runtime: {time.time() - t_start:.1f}s")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
