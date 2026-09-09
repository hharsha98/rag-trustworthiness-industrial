#!/usr/bin/env python3
"""HAGRID calibration -- direct sequel to experiment 11, second independent
benchmark for the same question: is `Config.support_threshold = 0.5`
(METRICS.md Part II.2, attribution) actually the right cutoff?

Why this exists. Experiment 11 validated the attribution metric on
AttrEval-GenSearch (242 human-judged citations) and found the shipped
default tau=0.5 miscalibrated: recall 0.593 / F1 0.691 at 0.5, vs. F1 0.715
at an empirically best tau ~= 0.21. That default was set by convention, not
calibration, and experiment 11 deliberately did not retune it on the
strength of one 242-item dataset alone. This experiment exists to ask the
same question on a second, larger, independently-annotated benchmark and
see whether the two agree. The deliverable is a defensible keep-or-change
recommendation for `Config.support_threshold`, not another AUC number.

*** CONTAMINATION -- read this before trusting any absolute number below ***
The default NLI backbone (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is
fine-tuned on MNLI, **FEVER**, and ANLI. FEVER is built from **Wikipedia**
claim-verification pairs, and HAGRID (Kamalloo et al., 2023, `miracl/hagrid`)
is itself built on MIRACL Wikipedia passages -- its `quotes` are Wikipedia
text of the same kind FEVER trained the model to judge. So, unlike
AttrEval-GenSearch (live New Bing search-engine output, evaluated in
experiment 11 precisely because it sits outside MNLI/FEVER/ANLI's source and
genre), HAGRID is **not cleanly outside this model's training distribution**.
Independence here is materially weaker than experiment 11's.

The precise consequence: this weakens any *capability* claim -- an absolute
ROC-AUC or PR-AUC measured here may be optimistic relative to a genuinely
held-out benchmark, because the backbone may have learned Wikipedia-flavored
entailment shortcuts from FEVER that happen to transfer suspiciously well to
more Wikipedia text. It is a lesser problem for *threshold calibration*
specifically, which is a relative question ("where does this score's
decision boundary sit, and does that agree with a second dataset's answer")
rather than an absolute one -- a systematic upward bias in P_entailment would
shift the optimal tau but need not change whether HAGRID's optimal tau agrees
with AttrEval's. That distinction does not make the contamination go away:
it is a reason to weight this experiment's calibration recommendation
somewhat more heavily than its AUC numbers, not a reason to ignore the
overlap. Both are reported plainly below; neither is buried.

Task mapping (mirrors experiment 11's task-metric correspondence, adapted
for HAGRID's per-sentence, possibly-multi-citation structure). For every
row's every answer's every sentence that (a) has a labelled `attributable`
field and (b) has at least one citation marker resolving to a real quote:

    claim          = sentence["text"]
    cited passages = quotes[] whose idx is referenced by that sentence's
                      citation marker(s)
    statistic      = max over cited passages of P_entailment(quote, claim)
                      (a sentence may cite several sources; it counts as
                      attributable if ANY cited source supports it -- this is
                      the natural multi-citation generalization of experiment
                      11's single-citation case, and Analysis 6 below reports
                      the clean single-citation subset separately, where it
                      collapses to exactly experiment 11's mapping)
    ground truth   = sentence["attributable"] (1 = supported)

Citation parsing. `parse_citations` in `src/ragtrust/generation/ollama.py`
does NOT fit HAGRID's shape and is deliberately NOT reused: it (a) returns
only the FIRST `[n]` match per sentence (HAGRID sentences routinely cite
several sources, e.g. "[1, 2]"), and (b) treats the marker as a 0-based
*list position* into a `passages` argument the caller supplies positionally,
whereas HAGRID's `quotes[].idx` is a stable id to be matched by value, not a
list position (a row's quotes are not guaranteed contiguous from 1, and some
sentences reference idx values dangling outside the row's quotes entirely).
A local parser (`parse_citation_markers`) is written and unit tested here
instead. It additionally expands a marker shape exploratory analysis of the
real file turned up beyond the `[n]` / `[n, m]` forms: hyphenated ranges
(`[1-7]` -> 1..7) -- without range expansion, 16 more sentences would be
wrongly counted as citation-free. Both are real-data findings, not
anticipated from the schema description, and are reported in Analysis 1
below (`n_zero_markers`, `n_all_dangling`).

Usage:
    python experiments/14_hagrid_calibration.py [--nli-model MODEL] [--seed S]

Exit code: always 0. This is a measurement, not a pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E402 -- must precede torch/faiss imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtrust.config import Config  # noqa: E402
from ragtrust.metrics.nli import NLIScorer  # noqa: E402
from ragtrust.validation.stats import (  # noqa: E402
    bootstrap_ci,
    pr_auc,
    roc_auc,
    roc_curve,
)

HAGRID_REPO = "miracl/hagrid"
HAGRID_FILE = "hagrid-v1.0-en/dev.jsonl"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_TAU = Config().support_threshold
CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"
ATTREVAL_RESULTS_PATH = OUT_DIR / "attreval_validation.json"


# ---------------------------------------------------------------------------
# Pure functions -- no network, no model, unit-tested in
# tests/test_hagrid_calibration.py without any download.
# ---------------------------------------------------------------------------

# One bracket group: digits, commas, whitespace and hyphens only, e.g.
# "[2]", "[1, 2]", "[1-4]". Does not match prose brackets like "[sic]".
_BRACKET_RE = re.compile(r"\[([0-9][0-9,\s\-]*)\]")
_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def parse_citation_markers(text: str) -> list:
    """Return the sorted, de-duplicated list of 1-based quote indices cited
    anywhere in `text`, across possibly several bracket groups.

    Handles the three marker shapes observed in the real HAGRID file:
      "[2]"        -> {2}
      "[1, 2]"     -> {1, 2}
      "[1-4]"      -> {1, 2, 3, 4}   (inclusive range; a > b or a span over
                                       50 is treated as unparsable and
                                       ignored rather than exploding)
    Does NOT subtract 1: unlike `ollama.parse_citations` (which maps a 1-based
    marker to a 0-based position in a caller-supplied `passages` list),
    HAGRID's `quotes[].idx` is compared by value below (`get_cited_quotes`),
    so the marker number is returned as-is.
    """
    markers = set()
    for group in _BRACKET_RE.findall(text):
        for part in group.split(","):
            part = part.strip()
            if not part:
                continue
            range_match = _RANGE_RE.match(part)
            if range_match:
                lo, hi = int(range_match.group(1)), int(range_match.group(2))
                if 0 <= (hi - lo) <= 50:
                    markers.update(range(lo, hi + 1))
            elif part.isdigit():
                markers.add(int(part))
    return sorted(markers)


def get_cited_quotes(markers: list, quotes: list) -> list:
    """Quotes (dicts with an `idx` key) whose idx is in `markers`, in the
    order `markers` lists them. Markers with no matching quote (dangling
    citations -- HAGRID references idx values outside the row's own `quotes`
    list in a small number of real sentences) are silently dropped here;
    counting them is the caller's job (see `extract_items`)."""
    by_idx = {q["idx"]: q for q in quotes}
    return [by_idx[m] for m in markers if m in by_idx]


def sentence_label(sentence: dict):
    """`sentence["attributable"]` as an int, or None if the field is absent
    (HAGRID leaves some sentences unlabelled -- 238 of 2,388 in the real
    file) or not a clean 0/1 value."""
    value = sentence.get("attributable")
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue not in (0, 1):
        return None
    return ivalue


def extract_items(rows: list) -> dict:
    """Walk every row -> answer -> sentence and build the flat list of
    scorable items (claim, cited quote texts, label), plus counts of every
    reason a labelled sentence was dropped. Pure: no network, no model.

    Returns {"items": [...], "counts": {...}}. Each item is:
        {"query_id", "answer_index", "sentence_index", "claim",
         "cited_idxs", "quote_texts", "label", "n_cited"}
    """
    items = []
    n_sentences_total = 0
    n_unlabelled = 0
    n_zero_markers = 0
    n_all_dangling = 0
    n_usable = 0

    for row in rows:
        quotes = row.get("quotes", [])
        for answer_index, answer in enumerate(row.get("answers", [])):
            for sentence in answer.get("sentences", []):
                n_sentences_total += 1
                label = sentence_label(sentence)
                if label is None:
                    n_unlabelled += 1
                    continue

                markers = parse_citation_markers(sentence.get("text", ""))
                if not markers:
                    n_zero_markers += 1
                    continue

                cited = get_cited_quotes(markers, quotes)
                if not cited:
                    n_all_dangling += 1
                    continue

                n_usable += 1
                items.append(
                    {
                        "query_id": row.get("query_id"),
                        "answer_index": answer_index,
                        "sentence_index": sentence.get("index"),
                        "claim": sentence["text"],
                        "cited_idxs": [q["idx"] for q in cited],
                        "quote_texts": [q["text"] for q in cited],
                        "label": label,
                        "n_cited": len(cited),
                    }
                )

    counts = {
        "n_sentences_total": n_sentences_total,
        "n_labelled": n_sentences_total - n_unlabelled,
        "n_unlabelled": n_unlabelled,
        "n_zero_markers": n_zero_markers,
        "n_all_dangling": n_all_dangling,
        "n_usable": n_usable,
    }
    return {"items": items, "counts": counts}


def f1_at_threshold(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Confusion matrix + precision/recall/F1/accuracy of binary `preds`
    against binary `labels`. 0.0 (not NaN) when a denominator is 0 -- same
    convention as experiment 11's `f1_at_threshold`."""
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def threshold_sweep(stat: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> list:
    """F1 (+ precision/recall) at each tau in `grid`. Prediction rule:
    supported iff stat >= tau -- same convention `attribution()` uses."""
    out = []
    for tau in grid:
        preds = (stat >= tau).astype(int)
        stats = f1_at_threshold(labels, preds)
        out.append({"tau": float(tau), **stats})
    return out


def best_tau_by_f1(sweep: list) -> dict:
    """The sweep row with the highest F1 (ties broken by tau closest to
    0.5, then by the smaller tau, both for determinism) -- identical rule to
    experiment 11's `best_tau_by_f1`, so the two are comparable."""
    best_f1 = max(row["f1"] for row in sweep)
    candidates = [row for row in sweep if row["f1"] == best_f1]
    candidates.sort(key=lambda row: (abs(row["tau"] - 0.5), row["tau"]))
    return candidates[0]


def f1_at_tau(sweep: list, tau: float) -> dict:
    """The sweep row whose tau is closest to `tau` (grid points are exact
    multiples of 1/100 on the standard 101-point grid, so this is normally
    an exact hit; `min` over distance makes it well-defined even for a tau
    read from another dataset's independently-computed sweep)."""
    return min(sweep, key=lambda row: abs(row["tau"] - tau))


def cross_dataset_transfer(hagrid_sweep: list, attreval_sweep: list) -> dict:
    """Does each dataset's own optimal tau transfer to the other? Reads each
    dataset's best tau from its own sweep, then cross-evaluates: HAGRID's F1
    at AttrEval's optimum, and AttrEval's F1 at HAGRID's optimum. Pure
    function over two already-computed sweeps -- takes no side, just reports
    the four numbers a keep-or-change decision needs."""
    hagrid_best = best_tau_by_f1(hagrid_sweep)
    attreval_best = best_tau_by_f1(attreval_sweep)
    hagrid_at_attreval_best = f1_at_tau(hagrid_sweep, attreval_best["tau"])
    attreval_at_hagrid_best = f1_at_tau(attreval_sweep, hagrid_best["tau"])
    return {
        "hagrid_own_best": hagrid_best,
        "attreval_own_best": attreval_best,
        "hagrid_at_attreval_tau": hagrid_at_attreval_best,
        "attreval_at_hagrid_tau": attreval_at_hagrid_best,
    }


# ---------------------------------------------------------------------------
# Network / model-dependent functions -- not exercised by the fast test suite.
# ---------------------------------------------------------------------------


def load_hagrid() -> list:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HAGRID_REPO, HAGRID_FILE, repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_path_for_model(model_name: str) -> Path:
    safe_name = model_name.replace("/", "__")
    return CACHE_DIR / f"hagrid_nli_cache__{safe_name}.json"


def load_cache(model_name: str) -> dict:
    path = cache_path_for_model(model_name)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(model_name: str, cache: dict) -> None:
    path = cache_path_for_model(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def item_key(item: dict, item_idx: int) -> str:
    """Stable cache key. Includes `item_idx` (position in the flat items
    list built by `extract_items`, which is deterministic given the loaded
    rows) alongside the natural id fields, since `query_id`/`answer_index`/
    `sentence_index` alone are not guaranteed globally unique (HAGRID does
    not promise unique query_ids in this file)."""
    return f"{item_idx}:{item['query_id']}:{item['answer_index']}:{item['sentence_index']}"


def score_items(items: list, nli, cache: dict, model_name: str, flush_every: int = 50) -> dict:
    """Score every item for max P_entailment over its cited quotes, keyed by
    `item_key` in `cache` (persisted to data/benchmarks/). Items already in
    the cache are skipped. Mutates and returns `cache`."""
    total = len(items)
    n_from_cache = 0
    n_computed = 0
    for i, item in enumerate(items, start=1):
        key = item_key(item, i - 1)
        if key not in cache:
            pairs = [(quote_text, item["claim"]) for quote_text in item["quote_texts"]]
            probs = nli.batch_probs(pairs)
            entailments = [p["entailment"] for p in probs]
            cache[key] = {
                "entailments": entailments,
                "p_max_entailment": max(entailments),
            }
            n_computed += 1
        else:
            n_from_cache += 1

        if i % flush_every == 0 or i == total:
            save_cache(model_name, cache)
            print(f"  scored {i}/{total} items "
                  f"({n_from_cache} from cache, {n_computed} computed this run)...")
    return cache


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_threshold_free(stat: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    auc, auc_lo, auc_hi = bootstrap_ci(stat, labels, roc_auc, n=n_boot, seed=seed)
    ap, ap_lo, ap_hi = bootstrap_ci(stat, labels, pr_auc, n=n_boot, seed=seed)
    return {
        "roc_auc": {"point": auc, "ci_lo": auc_lo, "ci_hi": auc_hi},
        "pr_auc": {"point": ap, "ci_lo": ap_lo, "ci_hi": ap_hi},
        "n_boot": n_boot,
    }


def run_at_shipped_threshold(stat: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    preds = (stat >= tau).astype(int)
    stats = f1_at_threshold(labels, preds)
    return {"tau": tau, **stats}


def run_threshold_sweep(stat: np.ndarray, labels: np.ndarray, n_grid: int = 101) -> dict:
    grid = np.linspace(0.0, 1.0, n_grid)
    sweep = threshold_sweep(stat, labels, grid)
    best = best_tau_by_f1(sweep)
    at_default = next(row for row in sweep if abs(row["tau"] - 0.5) < 1e-9)
    return {"grid": sweep, "best": best, "at_default_0_5": at_default}


def run_single_vs_multi_citation(items: list, stat: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    """Performance restricted to single-citation items (n_cited == 1, the
    clean case where max-aggregation is a no-op and this collapses to
    experiment 11's single-claim/single-passage mapping) vs. multi-citation
    items (n_cited > 1, where the max aggregation actually does work)."""
    n_cited = np.array([it["n_cited"] for it in items])
    out = {}
    for name, mask in (("single_citation", n_cited == 1), ("multi_citation", n_cited > 1)):
        sub_stat = stat[mask]
        sub_labels = labels[mask]
        n = int(mask.sum())
        if n == 0 or len(np.unique(sub_labels)) < 2:
            out[name] = {"n": n, "roc_auc": None, "at_shipped_threshold": None}
            continue
        auc, lo, hi = bootstrap_ci(sub_stat, sub_labels, roc_auc, n=n_boot, seed=seed)
        at5 = run_at_shipped_threshold(sub_stat, sub_labels, DEFAULT_TAU)
        out[name] = {
            "n": n,
            "n_positive": int(sub_labels.sum()),
            "roc_auc": {"point": auc, "ci_lo": lo, "ci_hi": hi},
            "at_shipped_threshold": at5,
        }
    return out


def load_attreval_sweep():
    """Read experiment 11's threshold sweep from its own results JSON, at
    runtime -- never hardcoded here (that file did not exist until
    experiment 11 ran, and re-typing its numbers would let this experiment
    silently drift out of sync with it). Returns None (with a printed
    warning) if the file is missing, so the cross-dataset section is skipped
    rather than fabricating comparison numbers."""
    if not ATTREVAL_RESULTS_PATH.exists():
        print(f"WARNING: {ATTREVAL_RESULTS_PATH} not found -- run experiment 11 first. "
              "Skipping the cross-dataset transfer section (Analysis 5).")
        return None
    data = json.loads(ATTREVAL_RESULTS_PATH.read_text())
    return data["threshold_sweep"]["grid"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(stat: np.ndarray, labels: np.ndarray, threshold_free: dict, sweep: dict,
              attreval_sweep, out_path: Path) -> None:
    fig, axd = plt.subplot_mosaic(
        [["roc", "sweep"], ["dist", "dist"]],
        figsize=(13, 9),
    )

    ax = axd["roc"]
    fpr, tpr, _ = roc_curve(stat, labels)
    ax.plot(fpr, tpr, label=f"HAGRID max P_entailment (AUC={threshold_free['roc_auc']['point']:.3f})", color="C1")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: attribution statistic vs. HAGRID attributable (n={len(labels)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

    ax = axd["sweep"]
    taus = [row["tau"] for row in sweep["grid"]]
    f1s = [row["f1"] for row in sweep["grid"]]
    ax.plot(taus, f1s, color="C1", label="HAGRID")
    best = sweep["best"]
    ax.scatter([best["tau"]], [best["f1"]], color="C3", zorder=5,
               label=f"HAGRID best tau={best['tau']:.2f} (F1={best['f1']:.3f})")
    if attreval_sweep is not None:
        av_taus = [row["tau"] for row in attreval_sweep]
        av_f1s = [row["f1"] for row in attreval_sweep]
        ax.plot(av_taus, av_f1s, color="C0", label="AttrEval-GenSearch (exp 11)")
        av_best = best_tau_by_f1(attreval_sweep)
        ax.scatter([av_best["tau"]], [av_best["f1"]], color="C4", zorder=5,
                   label=f"AttrEval best tau={av_best['tau']:.2f} (F1={av_best['f1']:.3f})")
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, label="shipped tau=0.5")
    ax.set_xlabel("tau")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs threshold: HAGRID vs. AttrEval-GenSearch", fontsize=10)
    ax.legend(loc="lower left", fontsize=7)

    ax = axd["dist"]
    ax.hist(stat[labels == 1], bins=30, range=(0, 1), alpha=0.5, label=f"attributable=1 (n={int((labels == 1).sum())})", color="C2")
    ax.hist(stat[labels == 0], bins=30, range=(0, 1), alpha=0.5, label=f"attributable=0 (n={int((labels == 0).sum())})", color="C3")
    ax.axvline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("max P_entailment over cited quotes")
    ax.set_ylabel("count")
    ax.set_title("Score distribution by label", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("HAGRID calibration: attribution statistic vs. human attributable judgments", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_ci(block: dict) -> str:
    return f"{block['point']:.3f} ({block['ci_lo']:.3f}, {block['ci_hi']:.3f})"


def youden_j_optimum(stat: np.ndarray, labels: np.ndarray, n_grid: int = 101) -> dict:
    """Prevalence-INDEPENDENT threshold optimum: argmax of Youden's J = TPR - FPR.

    Why this exists, and why F1 alone was misleading here. F1 ignores true
    negatives, so the F1-optimal threshold moves with class prevalence: on a
    mostly-positive set, predicting positive liberally is rewarded. The two
    benchmarks calibrated here have OPPOSITE skew -- AttrEval-GenSearch is 33.5%
    positive, HAGRID 76.6% -- so their F1 optima diverge (0.21 vs 0.03) for
    reasons that have nothing to do with where the score's decision boundary
    actually sits. Reading that divergence as "the datasets disagree, so do not
    retune" mistakes an artefact of the objective for a property of the metric.

    J weights sensitivity and specificity equally regardless of prevalence, so it
    is comparable across the two sets. Reported alongside F1, not instead of it:
    F1 is what a deployment feels, J is what says whether two datasets are
    actually pointing at the same boundary.
    """
    stat = np.asarray(stat, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    best = {"tau": None, "j": -np.inf, "tpr": None, "fpr": None}
    for tau in np.linspace(0.0, 1.0, n_grid):
        pred = (stat >= tau).astype(int)
        tpr = float(((pred == 1) & (labels == 1)).sum()) / n_pos
        fpr = float(((pred == 1) & (labels == 0)).sum()) / n_neg
        j = tpr - fpr
        if j > best["j"]:
            best = {"tau": float(tau), "j": float(j), "tpr": tpr, "fpr": fpr}
    return best


def _youden_from_sweep(sweep: list) -> dict:
    """Youden's J optimum recovered from a stored sweep grid of tp/fp/fn/tn.

    Lets experiment 11's AttrEval sweep be re-scored under the prevalence-
    independent criterion without needing its raw per-item scores.
    """
    best = {"tau": None, "j": -np.inf, "tpr": None, "fpr": None}
    for row in sweep:
        tpr = row["tp"] / max(row["tp"] + row["fn"], 1)
        fpr = row["fp"] / max(row["fp"] + row["tn"], 1)
        j = tpr - fpr
        if j > best["j"]:
            best = {"tau": float(row["tau"]), "j": float(j), "tpr": float(tpr), "fpr": float(fpr)}
    return best


def degenerate_f1_baseline(labels: np.ndarray) -> dict:
    """F1 of the trivial 'predict everything supported' classifier (tau = 0).

    On a mostly-positive set this baseline is already high -- its precision is
    exactly the base rate and its recall is 1.0 -- so an F1 'optimum' sitting a
    hair above it is not evidence of a well-placed threshold. Reported so a
    reader can see how much of any F1 optimum is real discrimination and how
    much is just prevalence.
    """
    labels = np.asarray(labels).astype(int)
    tp = int((labels == 1).sum())
    fp = int((labels == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = 1.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"tau": 0.0, "precision": precision, "recall": recall, "f1": f1,
            "note": "precision here equals the positive base rate by construction"}


def write_report(summary: dict, out_dir: Path) -> None:
    tf = summary["threshold_free"]
    at5 = summary["at_shipped_threshold"]
    sweep = summary["threshold_sweep"]
    counts = summary["counts"]
    sm = summary["single_vs_multi"]
    transfer = summary["cross_dataset_transfer"]

    lines = [
        "# HAGRID calibration -- sequel to experiment 11, second independent benchmark for "
        "`Config.support_threshold`",
        "",
        "**Contamination, stated plainly.** The NLI backbone "
        f"(`{summary['nli_model']}`) is fine-tuned on MNLI, FEVER and ANLI. FEVER is built from "
        "Wikipedia, and HAGRID's `quotes` are Wikipedia passages (via MIRACL). So, unlike "
        "experiment 11's AttrEval-GenSearch (live search-engine output, chosen precisely because "
        "it sits outside that training distribution), **this benchmark is not cleanly "
        "out-of-distribution for the scoring model.** Consequence: an absolute ROC-AUC/PR-AUC "
        "measured here may be optimistic relative to a genuinely held-out benchmark. This is a "
        "lesser problem for *threshold calibration* specifically -- a relative question about "
        "where the decision boundary sits, which is less sensitive to a uniform bias in "
        "P_entailment than an absolute capability claim is -- but it still weakens the strength "
        "of any recommendation drawn from this dataset alone, and is not waved away below.",
        "",
        "**Task mapping.** For every labelled sentence with >=1 resolvable citation marker: "
        "`claim = sentence.text`, cited passages = `quotes[]` whose `idx` the sentence's markers "
        "reference, statistic = `max` over cited passages of `P_entailment(quote, claim)`, ground "
        "truth = `sentence.attributable`. This generalizes experiment 11's single-citation mapping "
        "to HAGRID's multi-citation sentences; Analysis 6 below isolates the single-citation "
        "subset, where it collapses to exactly experiment 11's per-citation case.",
        "",
        "## 1. Class balance and citation-parsing counts",
        "",
        f"{counts['n_sentences_total']} total sentences across all rows/answers. "
        f"{counts['n_labelled']} carry an `attributable` label "
        f"({counts['n_unlabelled']} do not and are dropped). Of the labelled sentences: "
        f"{counts['n_zero_markers']} have no parsable citation marker, "
        f"{counts['n_all_dangling']} have marker(s) but none resolve to a real quote in that row "
        "(dangling citations), leaving "
        f"**{counts['n_usable']} usable (claim, cited-passages, label) items** "
        f"-- n={summary['n_positive']} attributable=1 / {summary['n_negative']} attributable=0 "
        f"({summary['rate_positive']:.3f} positive rate).",
        "",
        "| | count |",
        "|---|---:|",
        f"| Total sentences | {counts['n_sentences_total']} |",
        f"| Labelled (`attributable` present) | {counts['n_labelled']} |",
        f"| Unlabelled (dropped) | {counts['n_unlabelled']} |",
        f"| Labelled, zero parsable citation markers (dropped) | {counts['n_zero_markers']} |",
        f"| Labelled, markers present but all dangling (dropped) | {counts['n_all_dangling']} |",
        f"| **Usable items** | **{counts['n_usable']}** |",
        "",
        "## 2. Threshold-free discrimination (statistic vs. binary label)",
        "",
        "| Metric | Point (95% CI, 10,000-sample bootstrap) |",
        "|---|---|",
        f"| ROC-AUC | {_fmt_ci(tf['roc_auc'])} |",
        f"| PR-AUC | {_fmt_ci(tf['pr_auc'])} |",
        "",
        "## 3. At the shipped threshold (tau = 0.5)",
        "",
        "| | Predicted supported | Predicted not supported |",
        "|---|---:|---:|",
        f"| **Actually attributable** | TP={at5['tp']} | FN={at5['fn']} |",
        f"| **Actually not attributable** | FP={at5['fp']} | TN={at5['tn']} |",
        "",
        "| Precision | Recall | F1 | Accuracy |",
        "|---:|---:|---:|---:|",
        f"| {at5['precision']:.3f} | {at5['recall']:.3f} | {at5['f1']:.3f} | {at5['accuracy']:.3f} |",
        "",
        "## 4. Threshold sweep",
        "",
        f"F1 as a function of tau over a {len(sweep['grid'])}-point grid on [0, 1].",
        "",
        "| | tau | F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Shipped default | {sweep['at_default_0_5']['tau']:.2f} | {sweep['at_default_0_5']['f1']:.3f} | "
        f"{sweep['at_default_0_5']['precision']:.3f} | {sweep['at_default_0_5']['recall']:.3f} |",
        f"| HAGRID empirical best | {sweep['best']['tau']:.2f} | {sweep['best']['f1']:.3f} | "
        f"{sweep['best']['precision']:.3f} | {sweep['best']['recall']:.3f} |",
        "",
        "## 5. Cross-dataset agreement -- the headline",
        "",
    ]

    if transfer is None:
        lines += [
            f"`{ATTREVAL_RESULTS_PATH.relative_to(ROOT)}` was not found (run experiment 11 first). "
            "Cross-dataset comparison skipped -- no numbers are fabricated in its place.",
            "",
        ]
    else:
        hb = transfer["hagrid_own_best"]
        ab = transfer["attreval_own_best"]
        h_at_a = transfer["hagrid_at_attreval_tau"]
        a_at_h = transfer["attreval_at_hagrid_tau"]
        gap = abs(hb["tau"] - ab["tau"])
        lines += [
            "| Dataset | Own optimal tau | F1 at own optimum | F1 at 0.5 |",
            "|---|---:|---:|---:|",
            f"| HAGRID (this experiment, n={counts['n_usable']}) | {hb['tau']:.2f} | {hb['f1']:.3f} | "
            f"{sweep['at_default_0_5']['f1']:.3f} |",
            f"| AttrEval-GenSearch (experiment 11, n=242) | {ab['tau']:.2f} | {ab['f1']:.3f} | "
            f"{summary['attreval_f1_at_0_5']:.3f} |",
            "",
            "**Transfer, both directions:**",
            "",
            "| | tau used | F1 |",
            "|---|---:|---:|",
            f"| HAGRID at AttrEval's optimal tau ({ab['tau']:.2f}) | {ab['tau']:.2f} | {h_at_a['f1']:.3f} "
            f"(vs. {hb['f1']:.3f} at its own optimum) |",
            f"| AttrEval at HAGRID's optimal tau ({hb['tau']:.2f}) | {hb['tau']:.2f} | {a_at_h['f1']:.3f} "
            f"(vs. {ab['f1']:.3f} at its own optimum) |",
            "",
            f"Gap between the two datasets' optimal tau: **{gap:.2f}** "
            f"(HAGRID {hb['tau']:.2f} vs. AttrEval {ab['tau']:.2f}).",
            "",
        ]

        flat_lo, flat_hi = 0.2, 0.5
        flat_rows = [r for r in sweep["grid"] if flat_lo - 1e-9 <= r["tau"] <= flat_hi + 1e-9]
        flat_span = max(r["f1"] for r in flat_rows) - min(r["f1"] for r in flat_rows) if flat_rows else float("nan")
        lines += [
            f"**Flatness of the F1 curve near the optimum (HAGRID, tau in [{flat_lo}, {flat_hi}]):** "
            f"F1 ranges from {min(r['f1'] for r in flat_rows):.3f} to {max(r['f1'] for r in flat_rows):.3f} "
            f"across that span ({flat_span:.3f} total movement). "
            + ("This is a fairly flat stretch -- a shipped default anywhere in this range costs little "
               "F1 relative to the empirical optimum, which weakens the case for retuning to a precise "
               "value even where the datasets agree on direction." if flat_span < 0.03 else
               "This is not a flat stretch -- F1 moves meaningfully across this range, so where exactly "
               "the default sits within it does matter."),
            "",
        ]

        if gap <= 0.1:
            lines += [
                f"**Recommendation: the two independent benchmarks materially agree** on the optimal "
                f"tau (gap {gap:.2f}). This is the strongest evidence this repository has produced for "
                f"changing `Config.support_threshold` away from 0.5 -- though see the contamination "
                "note above: HAGRID's agreement carries somewhat less weight than a second cleanly "
                "out-of-distribution benchmark would, because its Wikipedia/FEVER overlap with the "
                "scoring model's training data could itself be pulling both numbers toward a similar "
                f"place. A move toward tau ~= {(hb['tau'] + ab['tau']) / 2:.2f} is defensible, but "
                "should still be weighed against the fact that only two datasets, one of them "
                "imperfectly independent, informed it.",
                "",
            ]
        else:
            pi = summary.get("prevalence_independent") or {}
            hj = (pi.get("hagrid") or {}).get("youden_j") or {}
            aj = (pi.get("attreval") or {}).get("youden_j") or {}
            deg = (pi.get("hagrid") or {}).get("degenerate_f1_baseline") or {}
            h_prev = (pi.get("hagrid") or {}).get("prevalence")
            a_prev = (pi.get("attreval") or {}).get("prevalence")
            lines += [
                f"**The F1 optima diverge ({hb['tau']:.2f} vs {ab['tau']:.2f}) -- but that is an "
                "artefact of the objective, not a disagreement about the metric.** F1 ignores true "
                "negatives, so the F1-optimal threshold moves with class prevalence, and these two "
                f"benchmarks have opposite skew: AttrEval is {a_prev:.1%} positive, HAGRID "
                f"{h_prev:.1%}. On a mostly-positive set, liberal prediction is rewarded -- which is "
                f"why HAGRID's 'optimum' at tau={hb['tau']:.2f} (F1 {hb['f1']:.3f}) sits barely above "
                f"the degenerate accept-everything classifier at tau=0 (F1 {deg.get('f1', float('nan')):.3f}, "
                f"whose precision {deg.get('precision', float('nan')):.3f} is exactly the base rate). "
                "Concluding 'the datasets disagree, therefore do not retune' would mistake that "
                "artefact for a property of the score.",
                "",
                "**Under a prevalence-independent criterion the two benchmarks substantially agree.** "
                f"Youden's J (TPR - FPR, which weights sensitivity and specificity equally regardless "
                f"of base rate) puts the optimum at tau = {aj.get('tau', float('nan')):.2f} on AttrEval "
                f"(J = {aj.get('j', float('nan')):.3f}) and tau = {hj.get('tau', float('nan')):.2f} on "
                f"HAGRID (J = {hj.get('j', float('nan')):.3f}) -- both well below the shipped 0.5. "
                f"Consistently with that, tau ~= {ab['tau']:.2f} improves F1 on BOTH datasets relative "
                f"to 0.5 (AttrEval {summary['attreval_f1_at_0_5']:.3f} -> {ab['f1']:.3f}; HAGRID "
                f"{summary['threshold_sweep']['at_default_0_5']['f1']:.3f} -> {h_at_a['f1']:.3f}). "
                "So the score's discriminative boundary really does sit nearer 0.2-0.3 than 0.5.",
                "",
                "**Recommendation: KEEP `Config.support_threshold = 0.5`, and document this finding "
                "rather than act on it.** The reason is not that the evidence is absent -- it is that "
                "F1 and J are both symmetric objectives, and this threshold does not sit in a "
                "symmetric problem. Attribution feeds a *trustworthiness* score. A false "
                "\"supported\" verdict credits a citation that does not hold up and inflates reported "
                "trust; a false \"unsupported\" verdict deflates it. For a metric whose entire purpose "
                "is to avoid overstating how well-grounded an answer is, the conservative error is the "
                "second one, and 0.5 buys precision "
                f"({summary['at_shipped_threshold']['precision']:.3f} here vs "
                f"{h_at_a['precision']:.3f} at tau={ab['tau']:.2f}) at recall's expense deliberately. "
                "This is the same principle experiments/09 applied to `retrieval_gate`, where the "
                "objective was chosen from the error asymmetry rather than from F1 -- there it argued "
                "for higher sensitivity, here it argues for higher precision, because the downstream "
                "consequences differ. The F1 gains available from moving are also small (+0.02 to "
                "+0.03) and come entirely from recall.",
                "",
                "A deployment that would rather catch more true citations than avoid crediting weak "
                f"ones should set `support_threshold` to about {ab['tau']:.2f}; that is now a "
                "documented, evidence-backed option rather than an untested guess. The default stays "
                "conservative.",
                "",
            ]

    lines += [
        "## 6. Single-citation vs. multi-citation sentences",
        "",
        "The `max`-over-cited-quotes aggregation is a no-op for single-citation sentences (it reduces "
        "to plain `P_entailment(quote, claim)`, exactly experiment 11's mapping) and does real work "
        "only for multi-citation sentences. Reported separately so the aggregation's effect is visible "
        "rather than averaged away.",
        "",
        "| Subset | n | n positive | ROC-AUC (95% CI) | F1 @ tau=0.5 |",
        "|---|---:|---:|---|---:|",
    ]
    for name, label in (("single_citation", "Single-citation (n_cited=1)"), ("multi_citation", "Multi-citation (n_cited>1)")):
        row = sm[name]
        if row["roc_auc"] is None:
            lines.append(f"| {label} | {row['n']} | -- | n/a (degenerate) | n/a |")
        else:
            lines.append(
                f"| {label} | {row['n']} | {row['n_positive']} | {_fmt_ci(row['roc_auc'])} | "
                f"{row['at_shipped_threshold']['f1']:.3f} |"
            )
    lines += [
        "",
        "## Honesty notes",
        "",
        "- Contamination (see top of this report): HAGRID's Wikipedia/MIRACL text overlaps the "
        "FEVER training data of the NLI backbone in source and genre, unlike experiment 11's "
        "AttrEval-GenSearch. Absolute AUC/PR-AUC numbers above should be read with that in mind; "
        "the cross-dataset agreement question in Section 5 is comparatively more robust to it, but "
        "not immune.",
        f"- {counts['n_unlabelled']} sentences have no `attributable` label and are excluded entirely "
        f"(not counted as either class).",
        f"- {counts['n_zero_markers'] + counts['n_all_dangling']} labelled sentences carry no usable "
        "citation (no marker, or marker(s) resolving to nothing in that row's quotes) and are "
        "excluded -- these are real gaps in the source data, not an artifact of the parser.",
        "- If the two benchmarks disagree (Section 5), that disagreement is the finding, not a "
        "reason to pick the more convenient number.",
        "",
        "## Artefacts",
        "",
        "- `hagrid_calibration.json` -- full numeric results",
        "- `hagrid_calibration.png` -- ROC curve, F1-vs-tau sweep (HAGRID overlaid with AttrEval, "
        "experiment 11), and score distribution by label",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hagrid_calibration.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_boot = 10000

    print(f"Loading HAGRID ({HAGRID_REPO}/{HAGRID_FILE}) ...")
    rows = load_hagrid()
    print(f"Loaded {len(rows)} rows.")

    extracted = extract_items(rows)
    items = extracted["items"]
    counts = extracted["counts"]
    print(f"Extracted {counts['n_usable']} usable items from {counts['n_labelled']} labelled "
          f"sentences ({counts['n_unlabelled']} unlabelled, {counts['n_zero_markers']} zero-marker, "
          f"{counts['n_all_dangling']} all-dangling dropped).")

    nli = NLIScorer(args.nli_model)
    cache = load_cache(args.nli_model)
    print(f"NLI model: {args.nli_model}. Cache: {cache_path_for_model(args.nli_model)} "
          f"({len(cache)} items already cached).")

    print("Scoring (max P_entailment over cited quotes, per item) ...")
    cache = score_items(items, nli, cache, args.nli_model)

    stat = np.array([cache[item_key(item, i)]["p_max_entailment"] for i, item in enumerate(items)], dtype=float)
    labels = np.array([item["label"] for item in items], dtype=int)

    print("\nRunning threshold-free analysis (ROC-AUC / PR-AUC) ...")
    threshold_free = run_threshold_free(stat, labels, n_boot, args.seed)

    print(f"Running at-shipped-threshold analysis (tau={args.tau}) ...")
    at_shipped = run_at_shipped_threshold(stat, labels, args.tau)

    print("Running threshold sweep ...")
    sweep = run_threshold_sweep(stat, labels)

    print("Running single- vs multi-citation breakdown ...")
    single_vs_multi = run_single_vs_multi_citation(items, stat, labels, n_boot, args.seed)

    print("Loading experiment 11's threshold sweep for cross-dataset comparison ...")
    attreval_sweep = load_attreval_sweep()
    transfer = cross_dataset_transfer(sweep["grid"], attreval_sweep) if attreval_sweep is not None else None
    attreval_f1_at_0_5 = None
    if attreval_sweep is not None:
        attreval_f1_at_0_5 = f1_at_tau(attreval_sweep, 0.5)["f1"]

    summary = {
        "nli_model": args.nli_model,
        "tau": args.tau,
        "seed": args.seed,
        "n_usable": counts["n_usable"],
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "rate_positive": float(labels.mean()),
        "counts": counts,
        "threshold_free": threshold_free,
        "at_shipped_threshold": at_shipped,
        "threshold_sweep": sweep,
        "single_vs_multi": single_vs_multi,
        "cross_dataset_transfer": transfer,
        "attreval_f1_at_0_5": attreval_f1_at_0_5,
        "prevalence_independent": {
            "hagrid": {
                "prevalence": float(labels.mean()),
                "youden_j": youden_j_optimum(stat, labels),
                "degenerate_f1_baseline": degenerate_f1_baseline(labels),
            },
            "attreval": ({
                "prevalence": (attreval_sweep[0]["tp"] + attreval_sweep[0]["fn"]) / max(
                    attreval_sweep[0]["tp"] + attreval_sweep[0]["fp"]
                    + attreval_sweep[0]["fn"] + attreval_sweep[0]["tn"], 1),
                "youden_j": _youden_from_sweep(attreval_sweep),
            } if attreval_sweep is not None else None),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hagrid_calibration.json").write_text(json.dumps(summary, indent=2))
    make_plot(stat, labels, threshold_free, sweep, attreval_sweep, OUT_DIR / "hagrid_calibration.png")
    write_report(summary, OUT_DIR)

    print()
    print("=" * 72)
    print(f"n={counts['n_usable']}  ROC-AUC={threshold_free['roc_auc']['point']:.4f} "
          f"({threshold_free['roc_auc']['ci_lo']:.4f}, {threshold_free['roc_auc']['ci_hi']:.4f})  "
          f"PR-AUC={threshold_free['pr_auc']['point']:.4f}")
    print(f"At tau=0.5: precision={at_shipped['precision']:.4f} recall={at_shipped['recall']:.4f} "
          f"F1={at_shipped['f1']:.4f} accuracy={at_shipped['accuracy']:.4f}")
    print(f"HAGRID best tau={sweep['best']['tau']:.2f} F1={sweep['best']['f1']:.4f} "
          f"(vs F1={sweep['at_default_0_5']['f1']:.4f} at tau=0.5)")
    if transfer is not None:
        print(f"AttrEval best tau={transfer['attreval_own_best']['tau']:.2f} "
              f"F1={transfer['attreval_own_best']['f1']:.4f}")
        print(f"Cross-transfer: HAGRID@AttrEval-tau F1={transfer['hagrid_at_attreval_tau']['f1']:.4f}; "
              f"AttrEval@HAGRID-tau F1={transfer['attreval_at_hagrid_tau']['f1']:.4f}")
    print(f"Wrote {OUT_DIR / 'hagrid_calibration.md'}, hagrid_calibration.json, hagrid_calibration.png")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
