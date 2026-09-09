#!/usr/bin/env python3
"""AttrEval-GenSearch validation -- METRICS.md Part II.2 (attribution), against
an independent, human-annotated, third-party benchmark.

Why AttrEval-GenSearch. The default NLI backbone here
(`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is fine-tuned on MNLI, FEVER
and ANLI. Evaluating an attribution metric against a benchmark built from any
of those corpora would be circular -- the NLI model would already have seen
that exact style of premise/hypothesis pair as a training signal.
AttrEval-GenSearch (Yue et al., 2023 -- the AttrScore paper) is built from
live outputs of a generative search engine (New Bing), annotated in 2023
across everyday-web domains such as "Pet and Animal" and "Economics and
Finance" -- none of which are MNLI, FEVER or ANLI documents, and none of
which existed at the NLI checkpoint's training time in the MNLI/FEVER/ANLI
sense (those are curated NLP benchmark corpora, not live search-engine
output). So it sits outside the NLI model's training distribution. This is
the same argument experiment 10 makes for RAGTruth on the faithfulness
metric; here it is applied to the attribution metric specifically, using
citation-support judgments rather than sentence-level hallucination labels.
This claim should not be overstated: "outside the training distribution" is
a claim about the *source* and *genre* of the text, not a formal guarantee
of no overlap -- MNLI/FEVER/ANLI themselves draw from varied web/Wikipedia
text, so some topical overlap with "Pet and Animal" or "Economics and
Finance" content is possible in the general sense that all English NLI
corpora that broad. The benchmark's value is that it is independent,
third-party, real citation-support annotation of live generative-search
output -- not a further curated NLI-style dataset -- which is a different
and complementary kind of evidence from experiment 10's RAGTruth check.

The task-metric correspondence. `attribution()` in
`src/ragtrust/metrics/attribution.py` judges a citation "correct" iff
`P_entailment(cited_passage, claim) >= tau` (default `tau = Config().
support_threshold = 0.5`). Every AttrEval-GenSearch row is already exactly
one (statement, cited-passage) pair with a human "is this actually
supported" judgment, so it maps onto `attribution()`'s single-claim,
single-citation case with no aggregation needed: `claims=[row.answer]`,
`passages=[row.reference]`, `citations={0: 0}`. With one claim and one
citation, `AttributionResult.precision` is exactly 1.0 when
`P_entailment >= tau` and exactly 0.0 otherwise -- i.e. `precision` IS the
thresholded decision this experiment validates. `verify_correspondence()`
below asserts this equivalence on real rows rather than assuming it from
reading the source.

Ground truth. `label` in {Attributable, Extrapolatory, Contradictory} maps
to a binary "citation genuinely supports the claim" target:

    Attributable                -> 1  (genuinely supported)
    Extrapolatory, Contradictory -> 0  (not supported -- unsupported vs.
                                         refuted are both "not supported"
                                         under attribution(), which is
                                         exactly what Analysis 5 below tests)

Usage:
    python experiments/11_attreval_validation.py [--nli-model MODEL] [--seed S]

Exit code: always 0. This is a measurement, not a pass/fail gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E402 -- must precede torch/faiss imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtrust.config import Config  # noqa: E402
from ragtrust.metrics.attribution import attribution  # noqa: E402
from ragtrust.metrics.nli import NLIScorer  # noqa: E402
from ragtrust.validation.stats import (  # noqa: E402
    bootstrap_ci,
    pr_auc,
    roc_auc,
    roc_curve,
)

ATTREVAL_REPO = "osunlp/AttrScore"
ATTREVAL_FILE = "AttrEval-GenSearch.csv"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_TAU = Config().support_threshold
LABELS = ("Attributable", "Extrapolatory", "Contradictory")
CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Pure functions -- no network, no model, unit-tested in
# tests/test_attreval_validation.py without any download.
# ---------------------------------------------------------------------------


def label_to_binary(label: str) -> int:
    """Attributable -> 1 (genuinely supported); Extrapolatory and
    Contradictory -> 0 (not supported -- attribution() does not
    distinguish "refuted" from "unsupported", it only asks whether the
    cited passage entails the claim)."""
    if label == "Attributable":
        return 1
    if label in ("Extrapolatory", "Contradictory"):
        return 0
    raise ValueError(f"Unexpected AttrEval label: {label!r}")


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a `supported` int column from `label`."""
    df = df.copy()
    df["supported"] = df["label"].apply(label_to_binary)
    return df


def row_to_attribution_inputs(row) -> dict:
    """Map one AttrEval-GenSearch row onto attribution()'s call signature:
    one claim (the generative-search answer), one passage (the cited
    reference), and a single citation pointing claim 0 at passage 0."""
    return {
        "claims": [row["answer"]],
        "citations": {0: 0},
        "passages": [row["reference"]],
    }


def f1_at_threshold(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Confusion matrix + precision/recall/F1/accuracy of binary `preds`
    against binary `labels`. Returns 0.0 for precision/recall/f1 when the
    corresponding denominator is 0 (no positive predictions / no positive
    labels), rather than raising -- this is a diagnostic over a fixed
    threshold, not attribution()'s own vacuous-truth convention (which
    applies to a different situation: zero *claims* or zero *citations*
    emitted by a system, not zero positive predictions in a threshold
    sweep over many independent rows)."""
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


def threshold_sweep(p_entail: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> list:
    """F1 at each tau in `grid`. Prediction rule matches attribution()'s
    own: supported iff P_entailment >= tau."""
    out = []
    for tau in grid:
        preds = (p_entail >= tau).astype(int)
        stats = f1_at_threshold(labels, preds)
        out.append({"tau": float(tau), **stats})
    return out


def best_tau_by_f1(sweep: list) -> dict:
    """The sweep row with the highest F1 (ties broken by tau closest to
    0.5, then by the smaller tau, both for determinism)."""
    best_f1 = max(row["f1"] for row in sweep)
    candidates = [row for row in sweep if row["f1"] == best_f1]
    candidates.sort(key=lambda row: (abs(row["tau"] - 0.5), row["tau"]))
    return candidates[0]


# ---------------------------------------------------------------------------
# Network / model-dependent functions -- not exercised by the fast test suite.
# ---------------------------------------------------------------------------


def load_attreval() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ATTREVAL_REPO, ATTREVAL_FILE, repo_type="dataset")
    return pd.read_csv(path)


def cache_path_for_model(model_name: str) -> Path:
    safe_name = model_name.replace("/", "__")
    return CACHE_DIR / f"attreval_nli_cache__{safe_name}.json"


def load_cache(model_name: str) -> dict:
    path = cache_path_for_model(model_name)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(model_name: str, cache: dict) -> None:
    path = cache_path_for_model(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def verify_correspondence(df: pd.DataFrame, nli: NLIScorer, tau: float, n_check: int = 5) -> None:
    """Assert, on a handful of real rows, that attribution()'s precision
    under the single-claim/single-citation mapping equals the thresholded
    entailment decision it is defined to be -- rather than assuming the
    correspondence claimed in the module docstring."""
    for _, row in df.head(n_check).iterrows():
        inputs = row_to_attribution_inputs(row)
        result = attribution(inputs["claims"], inputs["citations"], inputs["passages"], nli, tau=tau)
        p_ent = nli.probs(row["reference"], row["answer"])["entailment"]
        expected = 1.0 if p_ent >= tau else 0.0
        assert result.precision == expected, (
            f"attribution().precision ({result.precision}) does not match the thresholded "
            f"entailment decision ({expected}) for row {row.name}; the claimed "
            f"claims=[answer]/passages=[reference]/citations={{0:0}} correspondence does not hold."
        )


def score_items(df: pd.DataFrame, nli: NLIScorer, cache: dict, model_name: str, flush_every: int = 20) -> dict:
    """Score every row of `df` for P_entailment and P_contradiction (kappa),
    keyed by row index in `cache` (persisted to data/benchmarks/). Rows
    already in the cache are skipped. Mutates and returns `cache`."""
    total = len(df)
    n_from_cache = 0
    n_computed = 0
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        key = str(idx)
        if key in cache:
            n_from_cache += 1
        else:
            probs = nli.probs(row["reference"], row["answer"])
            cache[key] = {
                "p_entailment": probs["entailment"],
                "p_contradiction": probs["contradiction"],
                "p_neutral": probs["neutral"],
            }
            n_computed += 1

        if i % flush_every == 0 or i == total:
            save_cache(model_name, cache)
            print(f"  scored {i}/{total} rows "
                  f"({n_from_cache} from cache, {n_computed} computed this run)...")
    return cache


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_threshold_free(p_entail: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    auc, auc_lo, auc_hi = bootstrap_ci(p_entail, labels, roc_auc, n=n_boot, seed=seed)
    ap, ap_lo, ap_hi = bootstrap_ci(p_entail, labels, pr_auc, n=n_boot, seed=seed)
    return {
        "roc_auc": {"point": auc, "ci_lo": auc_lo, "ci_hi": auc_hi},
        "pr_auc": {"point": ap, "ci_lo": ap_lo, "ci_hi": ap_hi},
        "n_boot": n_boot,
    }


def run_at_shipped_threshold(p_entail: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    preds = (p_entail >= tau).astype(int)
    stats = f1_at_threshold(labels, preds)
    return {"tau": tau, **stats}


def run_threshold_sweep(p_entail: np.ndarray, labels: np.ndarray, n_grid: int = 101) -> dict:
    grid = np.linspace(0.0, 1.0, n_grid)
    sweep = threshold_sweep(p_entail, labels, grid)
    best = best_tau_by_f1(sweep)
    at_default = next(row for row in sweep if abs(row["tau"] - 0.5) < 1e-9)
    return {"grid": sweep, "best": best, "at_default_0_5": at_default}


def run_three_way_separation(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    by_label = {}
    for lbl in LABELS:
        sub = df[df["label"] == lbl]
        by_label[lbl] = {
            "n": int(len(sub)),
            "p_entailment": sub["p_entailment"].tolist(),
            "p_contradiction": sub["p_contradiction"].tolist(),
        }

    # Contradictory vs Extrapolatory on kappa (contradiction probability),
    # excluding Attributable. Report each comparison ONCE: scoring kappa
    # against "is Contradictory" on this two-class subset and reporting
    # "is Extrapolatory" too would be the same fact printed twice, since
    # AUC(s, ~y) == 1 - AUC(s, y) identically on a two-class subset.
    ce = df[df["label"].isin(("Contradictory", "Extrapolatory"))].copy()
    is_contradictory = (ce["label"] == "Contradictory").to_numpy().astype(int)
    kappa = ce["p_contradiction"].to_numpy(dtype=float)
    kappa_auc, kappa_lo, kappa_hi = bootstrap_ci(kappa, is_contradictory, roc_auc, n=n_boot, seed=seed)

    return {
        "by_label": by_label,
        "contradictory_vs_extrapolatory": {
            "n_contradictory": int(is_contradictory.sum()),
            "n_extrapolatory": int((1 - is_contradictory).sum()),
            "statistic": "p_contradiction (kappa)",
            "target": "is_Contradictory (within Contradictory/Extrapolatory subset)",
            "roc_auc": {"point": kappa_auc, "ci_lo": kappa_lo, "ci_hi": kappa_hi},
            "clears_chance": bool(kappa_lo > 0.5),
            "note": (
                "AUC(kappa, is_Contradictory) and AUC(kappa, is_Extrapolatory) on this same "
                "two-class subset are related by AUC(s, ~y) == 1 - AUC(s, y) exactly, so only "
                "one is reported."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(scored: pd.DataFrame, threshold_free: dict, sweep: dict, out_path: Path) -> None:
    fig, axd = plt.subplot_mosaic(
        [["roc", "sweep"], ["dist_ent", "dist_kappa"]],
        figsize=(13, 10),
    )

    labels = scored["supported"].to_numpy().astype(int)
    p_entail = scored["p_entailment"].to_numpy(dtype=float)

    ax = axd["roc"]
    fpr, tpr, _ = roc_curve(p_entail, labels)
    ax.plot(fpr, tpr, label=f"P_entailment (AUC={threshold_free['roc_auc']['point']:.3f})", color="C1")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: attribution() (n={len(scored)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

    ax = axd["sweep"]
    taus = [row["tau"] for row in sweep["grid"]]
    f1s = [row["f1"] for row in sweep["grid"]]
    ax.plot(taus, f1s, color="C1")
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, label="shipped tau=0.5")
    best = sweep["best"]
    ax.scatter([best["tau"]], [best["f1"]], color="C3", zorder=5,
               label=f"best tau={best['tau']:.2f} (F1={best['f1']:.3f})")
    ax.set_xlabel("tau")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs threshold", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    colors = {"Attributable": "C2", "Extrapolatory": "C1", "Contradictory": "C3"}
    ax = axd["dist_ent"]
    for lbl in LABELS:
        vals = scored.loc[scored["label"] == lbl, "p_entailment"]
        ax.hist(vals, bins=20, range=(0, 1), alpha=0.5, label=f"{lbl} (n={len(vals)})", color=colors[lbl])
    ax.axvline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("P_entailment")
    ax.set_ylabel("count")
    ax.set_title("P_entailment by label", fontsize=10)
    ax.legend(fontsize=7)

    ax = axd["dist_kappa"]
    for lbl in LABELS:
        vals = scored.loc[scored["label"] == lbl, "p_contradiction"]
        ax.hist(vals, bins=20, range=(0, 1), alpha=0.5, label=f"{lbl} (n={len(vals)})", color=colors[lbl])
    ax.set_xlabel("P_contradiction (kappa)")
    ax.set_ylabel("count")
    ax.set_title("Contradiction probability by label", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("AttrEval-GenSearch validation: attribution() vs. human citation-support judgments", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_ci(block: dict) -> str:
    return f"{block['point']:.3f} ({block['ci_lo']:.3f}, {block['ci_hi']:.3f})"


def write_report(summary: dict, out_dir: Path) -> None:
    tf = summary["threshold_free"]
    at5 = summary["at_shipped_threshold"]
    sweep = summary["threshold_sweep"]
    sep = summary["three_way_separation"]
    cb = summary["class_balance"]

    lines = [
        "# AttrEval-GenSearch validation -- METRICS.md Part II.2 (attribution), independent benchmark",
        "",
        "Ground truth here is independent, third-party human annotation of live "
        "generative-search-engine output (New Bing), not built by construction: "
        "**AttrEval-GenSearch** (Yue et al., 2023 / `osunlp/AttrScore`), 242 "
        "(statement, cited-passage) pairs judged Attributable / Extrapolatory / Contradictory.",
        "",
        "**Why this benchmark.** The default NLI backbone "
        f"(`{summary['nli_model']}`) is fine-tuned on MNLI, FEVER and ANLI. AttrEval-GenSearch "
        "is built from live search-engine answers annotated in 2023 across everyday-web domains "
        "(e.g. \"Pet and Animal\", \"Economics and Finance\") -- not curated NLI benchmark text -- "
        "so it sits outside that training distribution in source and genre. This is not a formal "
        "guarantee of zero overlap: MNLI/FEVER/ANLI are themselves drawn from varied web/Wikipedia "
        "text, so some topical overlap is possible in the loose sense that any two broad English "
        "corpora can overlap. What this benchmark adds is independent, third-party citation-support "
        "judgment on real generative-search output, which experiment 10's RAGTruth check does not "
        "cover (that one judges hallucination in RAG *summaries/QA/data2txt*, not explicit "
        "citation-to-claim support).",
        "",
        f"**Task-metric correspondence.** Each row maps onto "
        "`attribution(claims=[answer], citations={0: 0}, passages=[reference], nli, tau)` exactly: "
        "one claim, one passage, one citation. With a single claim and citation, "
        "`AttributionResult.precision` is 1.0 iff `P_entailment(reference, answer) >= tau` and 0.0 "
        f"otherwise -- confirmed by direct assertion on {summary['n_correspondence_checked']} real "
        "rows before scoring (`verify_correspondence`), not assumed.",
        "",
        f"n = **{summary['n_items']}** (small -- confidence intervals below are correspondingly "
        "wide; treat point estimates cautiously). Class balance: "
        f"**{cb['n_attributable']} Attributable (supported=1)** vs. "
        f"**{cb['n_not_attributable']} not-supported (supported=0)** "
        f"({cb['n_extrapolatory']} Extrapolatory + {cb['n_contradictory']} Contradictory) -- "
        f"a {cb['rate_supported']:.3f} positive rate.",
        "",
        "## 1. Threshold-free discrimination (P_entailment vs. binary label)",
        "",
        "| Metric | Point (95% CI, 10,000-sample bootstrap) |",
        "|---|---|",
        f"| ROC-AUC | {_fmt_ci(tf['roc_auc'])} |",
        f"| PR-AUC | {_fmt_ci(tf['pr_auc'])} |",
        "",
        "## 2. At the shipped threshold (tau = 0.5) -- the deployment-relevant number",
        "",
        "This is what `Config().support_threshold` actually does to citation judgments today; "
        "the AUC above says the score *could* discriminate, this says whether the *default* "
        "cutoff realizes that.",
        "",
        "| | Predicted supported | Predicted not supported |",
        "|---|---:|---:|",
        f"| **Actually Attributable** | TP={at5['tp']} | FN={at5['fn']} |",
        f"| **Actually not supported** | FP={at5['fp']} | TN={at5['tn']} |",
        "",
        "| Precision | Recall | F1 | Accuracy |",
        "|---:|---:|---:|---:|",
        f"| {at5['precision']:.3f} | {at5['recall']:.3f} | {at5['f1']:.3f} | {at5['accuracy']:.3f} |",
        "",
        "## 3. Threshold sweep",
        "",
        f"F1 as a function of tau over a {len(sweep['grid'])}-point grid on [0, 1].",
        "",
        "| | tau | F1 | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
        f"| Shipped default | {sweep['at_default_0_5']['tau']:.2f} | {sweep['at_default_0_5']['f1']:.3f} | "
        f"{sweep['at_default_0_5']['precision']:.3f} | {sweep['at_default_0_5']['recall']:.3f} |",
        f"| Empirical best | {sweep['best']['tau']:.2f} | {sweep['best']['f1']:.3f} | "
        f"{sweep['best']['precision']:.3f} | {sweep['best']['recall']:.3f} |",
        "",
    ]

    tau_gap = abs(sweep["best"]["tau"] - 0.5)
    if tau_gap >= 0.1:
        lines += [
            f"**The optimal tau ({sweep['best']['tau']:.2f}) is far from the shipped default (0.5)**, "
            f"a gap of {tau_gap:.2f}. On this benchmark, `Config.support_threshold` would need to "
            f"move from 0.5 to approximately **{sweep['best']['tau']:.2f}** to reach the F1 achievable "
            f"here ({sweep['best']['f1']:.3f} vs. {sweep['at_default_0_5']['f1']:.3f} at 0.5). This is "
            "an actionable finding about the default, not a footnote -- though note n=242 (and the "
            "class split within it) makes a single-dataset optimum a fragile target for a global "
            "default; treat it as evidence to weigh against other benchmarks (e.g. experiment 10's "
            "RAGTruth-derived thresholds for faithfulness), not a mandate to retune blind.",
            "",
        ]
    else:
        lines += [
            f"The optimal tau ({sweep['best']['tau']:.2f}) is close to the shipped default (0.5, gap "
            f"{tau_gap:.2f}); this benchmark does not argue for changing `Config.support_threshold`.",
            "",
        ]

    lines += [
        "## 4. Three-way separation: does kappa distinguish Contradictory from Extrapolatory?",
        "",
        "`attribution()` collapses Extrapolatory and Contradictory into the same \"not supported\" "
        "outcome (Analyses 1-4 above). METRICS.md separately reports faithfulness's contradiction "
        "rate (`kappa`) on the theory that refuted and unsupported are different failure modes; "
        "AttrEval-GenSearch's own three-way label lets that claim be tested independently, on "
        "attribution's own NLI backbone, using P_contradiction as kappa.",
        "",
        "**One comparison, reported once.** Scoring kappa against `is_Contradictory` on the "
        "Contradictory/Extrapolatory subset (Attributable excluded) and also against "
        "`is_Extrapolatory` on the same subset would be the same fact twice: "
        "`AUC(kappa, is_Extrapolatory) == 1 - AUC(kappa, is_Contradictory)` identically on a "
        "two-class subset. Only `is_Contradictory` is reported.",
        "",
        "| Statistic | Target | n (Contradictory / Extrapolatory) | ROC-AUC (95% CI) | Clears chance? |",
        "|---|---|---|---|---|",
    ]
    cve = sep["contradictory_vs_extrapolatory"]
    clears = "yes" if cve["clears_chance"] else "no -- CI includes 0.5"
    lines.append(
        f"| kappa (P_contradiction) | is_Contradictory | {cve['n_contradictory']} / "
        f"{cve['n_extrapolatory']} | {_fmt_ci(cve['roc_auc'])} | {clears} |"
    )
    lines += ["", ""]
    if cve["clears_chance"]:
        lines.append(
            "**kappa does separate Contradictory from Extrapolatory** on this benchmark: its CI "
            "clears chance (0.5), supporting METRICS.md's claim that refuted and unsupported are "
            f"distinguishable failure modes (AUC={cve['roc_auc']['point']:.3f})."
        )
    else:
        lines.append(
            "**kappa does not clearly separate Contradictory from Extrapolatory here**: its CI "
            f"includes chance (AUC={cve['roc_auc']['point']:.3f}, CI "
            f"[{cve['roc_auc']['ci_lo']:.3f}, {cve['roc_auc']['ci_hi']:.3f}]). With only "
            f"{cve['n_contradictory']} Contradictory rows this is a low-powered test, not proof "
            "the distinction is decorative -- but on this evidence it is not established either."
        )
    lines += [
        "",
        f"Label distribution: Attributable={cb['n_attributable']}, "
        f"Extrapolatory={cb['n_extrapolatory']}, Contradictory={cb['n_contradictory']} "
        f"(n={summary['n_items']}).",
        "",
        "## Honesty notes",
        "",
        f"- n={summary['n_items']} is small; every CI above should be read at its full width, not "
        "just its point estimate.",
        f"- Class imbalance: {cb['n_attributable']}/{summary['n_items']} Attributable vs. "
        f"{cb['n_not_attributable']}/{summary['n_items']} not-supported "
        f"({cb['n_extrapolatory']} Extrapolatory + {cb['n_contradictory']} Contradictory).",
        "- If the metric performs poorly on this benchmark, that is the headline finding, not a "
        "footnote: a citation-support metric that fails on third-party, human-annotated citation "
        "judgments is exactly the kind of gap this repository exists to surface.",
        "",
        "## Artefacts",
        "",
        "- `attreval_validation.json` -- full numeric results",
        "- `attreval_validation.png` -- ROC curve (P_entailment), F1-vs-tau sweep, and "
        "P_entailment/P_contradiction distributions by label",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attreval_validation.md").write_text("\n".join(lines))


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

    print(f"Loading AttrEval-GenSearch ({ATTREVAL_REPO}/{ATTREVAL_FILE}) ...")
    df = load_attreval()
    df = add_binary_label(df)
    n_total = len(df)
    print(f"Loaded {n_total} rows. Label counts: "
          f"{df['label'].value_counts().to_dict()}")

    nli = NLIScorer(args.nli_model)
    cache = load_cache(args.nli_model)
    print(f"NLI model: {args.nli_model}. Cache: {cache_path_for_model(args.nli_model)} "
          f"({len(cache)} rows already cached).")

    print("Verifying attribution() <-> thresholded-entailment correspondence on 5 real rows ...")
    verify_correspondence(df, nli, args.tau, n_check=5)
    print("  OK: attribution().precision matches the thresholded entailment decision exactly.")

    print("Scoring (NLIScorer.probs on every reference/answer pair) ...")
    cache = score_items(df, nli, cache, args.nli_model)

    records = [cache[str(idx)] for idx in df.index]
    scored = df.reset_index(drop=True).copy()
    scored["p_entailment"] = [r["p_entailment"] for r in records]
    scored["p_contradiction"] = [r["p_contradiction"] for r in records]
    scored["p_neutral"] = [r["p_neutral"] for r in records]

    labels = scored["supported"].to_numpy().astype(int)
    p_entail = scored["p_entailment"].to_numpy(dtype=float)

    print("\nRunning threshold-free analysis (ROC-AUC / PR-AUC) ...")
    threshold_free = run_threshold_free(p_entail, labels, n_boot, args.seed)

    print(f"Running at-shipped-threshold analysis (tau={args.tau}) ...")
    at_shipped = run_at_shipped_threshold(p_entail, labels, args.tau)

    print("Running threshold sweep ...")
    sweep = run_threshold_sweep(p_entail, labels)

    print("Running three-way separation (kappa: Contradictory vs Extrapolatory) ...")
    three_way = run_three_way_separation(scored, n_boot, args.seed)

    cb = {
        "n_attributable": int((scored["label"] == "Attributable").sum()),
        "n_extrapolatory": int((scored["label"] == "Extrapolatory").sum()),
        "n_contradictory": int((scored["label"] == "Contradictory").sum()),
        "n_not_attributable": int((scored["label"] != "Attributable").sum()),
        "rate_supported": float(labels.mean()),
    }

    summary = {
        "nli_model": args.nli_model,
        "tau": args.tau,
        "seed": args.seed,
        "n_items": n_total,
        "n_correspondence_checked": 5,
        "class_balance": cb,
        "threshold_free": threshold_free,
        "at_shipped_threshold": at_shipped,
        "threshold_sweep": sweep,
        "three_way_separation": three_way,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "attreval_validation.json").write_text(json.dumps(summary, indent=2))
    make_plot(scored, threshold_free, sweep, OUT_DIR / "attreval_validation.png")
    write_report(summary, OUT_DIR)

    print()
    print("=" * 72)
    print(f"n={n_total}  ROC-AUC={threshold_free['roc_auc']['point']:.4f} "
          f"({threshold_free['roc_auc']['ci_lo']:.4f}, {threshold_free['roc_auc']['ci_hi']:.4f})  "
          f"PR-AUC={threshold_free['pr_auc']['point']:.4f}")
    print(f"At tau=0.5: precision={at_shipped['precision']:.4f} recall={at_shipped['recall']:.4f} "
          f"F1={at_shipped['f1']:.4f} accuracy={at_shipped['accuracy']:.4f}")
    print(f"Best tau={sweep['best']['tau']:.2f} F1={sweep['best']['f1']:.4f} "
          f"(vs F1={sweep['at_default_0_5']['f1']:.4f} at tau=0.5)")
    cve = three_way["contradictory_vs_extrapolatory"]
    print(f"kappa vs Contradictory/Extrapolatory: AUC={cve['roc_auc']['point']:.4f} "
          f"({cve['roc_auc']['ci_lo']:.4f}, {cve['roc_auc']['ci_hi']:.4f}) "
          f"clears_chance={cve['clears_chance']}")
    print(f"Wrote {OUT_DIR / 'attreval_validation.md'}, attreval_validation.json, attreval_validation.png")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
