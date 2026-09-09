#!/usr/bin/env python3
"""RAGTruth validation -- METRICS.md Part III, independent benchmark.

Why RAGTruth. The default NLI backbone here (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
is fine-tuned on MNLI, FEVER and ANLI. Evaluating a faithfulness metric against a benchmark
built from any of those corpora would be circular -- the NLI model has already seen that exact
kind of premise/hypothesis pair as a training signal. RAGTruth (Wu et al., 2024) is a
word-for-word-annotated RAG hallucination benchmark built from Yelp reviews (Data2txt),
CNN/DailyMail articles (Summary) and MARCO passages (QA) -- none of which are MNLI, FEVER or
ANLI -- so it sits outside the NLI model's training data. That is the entire reason a result
measured here is informative rather than circular.

Ground truth here is independent, third-party, human annotation:
`hallucination_labels_processed` is a dict ``{"evident_conflict": int, "baseless_info": int}``
per response, from which:

    hallucinated  = evident_conflict > 0 or baseless_info > 0
    has_conflict  = evident_conflict > 0     (response contradicts the source -- "refuted")
    has_baseless  = baseless_info > 0        (response adds ungrounded content -- "unsupported")

Experiment A asks how well faithfulness (mean max entailment) detects real hallucinations,
broken down by `task_type` (Summary / Data2txt / QA are different tasks and pooling them
would hide that).

Experiment B asks the sharper question: METRICS.md reports faithfulness `F` (mean max
entailment) and contradiction rate `kappa` (mean max contradiction) as two *separate* numbers
because it claims "refuted" and "unsupported" are different failure modes. RAGTruth's two label
types map directly onto that claim (evident_conflict ~ refuted, baseless_info ~ unsupported).
On items with exactly one label type present (so the two signals are not confounded), if the
design argument holds, kappa should detect conflicts better than (1-F) does, and (1-F) should
detect baseless info better than kappa does. If the two statistics are interchangeable instead,
that means the split reported in METRICS.md is decorative.

Usage:
    python experiments/10_ragtruth_validation.py [--items N] [--nli-model MODEL] [--seed S] [--full]

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

from ragtrust.ingest.loader import chunk_passages  # noqa: E402
from ragtrust.metrics.claims import split_claims  # noqa: E402
from ragtrust.metrics.faithfulness import faithfulness  # noqa: E402
from ragtrust.metrics.nli import NLIScorer  # noqa: E402
from ragtrust.validation.stats import (  # noqa: E402
    bootstrap_ci,
    paired_permutation_test,
    pr_auc,
    roc_auc,
    roc_curve,
)

RAGTRUTH_REPO = "wandb/RAGTruth-processed"
RAGTRUTH_FILE = "data/test-00000-of-00001.parquet"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TASK_TYPES = ("Summary", "Data2txt", "QA")
CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Pure functions -- no network, no model, unit-tested in
# tests/test_ragtruth_validation.py without any download.
# ---------------------------------------------------------------------------


def derive_labels(hallucination_labels_processed: dict) -> dict:
    """Map RAGTruth's raw per-item label-count dict to the three booleans this
    experiment needs. Robust to missing keys or None counts."""
    evident_conflict = int(hallucination_labels_processed.get("evident_conflict", 0) or 0)
    baseless_info = int(hallucination_labels_processed.get("baseless_info", 0) or 0)
    has_conflict = evident_conflict > 0
    has_baseless = baseless_info > 0
    return {
        "has_conflict": has_conflict,
        "has_baseless": has_baseless,
        "hallucinated": has_conflict or has_baseless,
    }


def add_derived_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with has_conflict / has_baseless / hallucinated columns."""
    df = df.copy()
    derived = df["hallucination_labels_processed"].apply(derive_labels)
    df["has_conflict"] = derived.apply(lambda d: d["has_conflict"])
    df["has_baseless"] = derived.apply(lambda d: d["has_baseless"])
    df["hallucinated"] = derived.apply(lambda d: d["hallucinated"])
    return df


def stratified_sample(df: pd.DataFrame, n_items: int, seed: int, task_col: str = "task_type") -> pd.DataFrame:
    """Sample `n_items` rows from `df`, split as evenly as possible across the
    distinct values of `task_col`, deterministically for a given seed.

    If `n_items` is not divisible by the number of groups, the first groups
    (in sorted order) each take one extra item. If `n_items` covers the whole
    frame, the whole frame is returned (shuffled deterministically)."""
    if n_items >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    groups = sorted(df[task_col].unique())
    n_groups = len(groups)
    base = n_items // n_groups
    remainder = n_items % n_groups

    parts = []
    for i, group in enumerate(groups):
        take = base + (1 if i < remainder else 0)
        pool = df[df[task_col] == group]
        take = min(take, len(pool))
        parts.append(pool.sample(n=take, random_state=seed))

    sampled = pd.concat(parts, ignore_index=False)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def filter_single_label_type(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where exactly one of has_conflict / has_baseless is True -- the
    subset Experiment B uses so the two failure-mode signals are not
    confounded with each other. Requires add_derived_labels to have run."""
    mask = df["has_conflict"] != df["has_baseless"]
    return df[mask].reset_index(drop=True)


def get_passages(context: str) -> list:
    """Chunk one RAGTruth item's context into passages via the same
    `chunk_passages` used for the bundled corpus, treating the whole context
    as a single "page" (per the experiment spec). Falls back to the raw
    context as a single passage if chunking drops everything (e.g. a very
    short context below `min_chars`)."""
    chunks = chunk_passages([context or ""])
    texts = [c["text"] for c in chunks]
    if not texts and context and context.strip():
        texts = [context.strip()]
    return texts


def get_claims(output: str) -> list:
    """Split one RAGTruth response into claims, falling back to the whole
    response as a single claim if segmentation finds no sentences."""
    claims = split_claims(output or "")
    if not claims and output and output.strip():
        claims = [output.strip()]
    return claims


# ---------------------------------------------------------------------------
# Network / model-dependent functions -- not exercised by the fast test suite.
# ---------------------------------------------------------------------------


def load_ragtruth() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(RAGTRUTH_REPO, RAGTRUTH_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def cache_path_for_model(model_name: str) -> Path:
    safe_name = model_name.replace("/", "__")
    return CACHE_DIR / f"ragtruth_nli_cache__{safe_name}.json"


def load_cache(model_name: str) -> dict:
    path = cache_path_for_model(model_name)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(model_name: str, cache: dict) -> None:
    path = cache_path_for_model(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def score_items(df: pd.DataFrame, nli: NLIScorer, cache: dict, model_name: str,
                 flush_every: int = 20) -> dict:
    """Score every row of `df` with faithfulness, keyed by item id in `cache`
    (persisted to data/benchmarks/, keyed by model+item id). Rows already in
    the cache are skipped. Mutates and returns `cache`."""
    total = len(df)
    n_from_cache = 0
    n_computed = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        item_id = str(row["id"])
        if item_id in cache:
            entry = cache[item_id]
            if "score" not in entry and "v2_score" in entry:
                # Migrate cache entries written by an older schema (which also
                # scored a since-removed baseline) to the current schema, in
                # place, so the warm cache is reused instead of recomputed.
                cache[item_id] = {
                    "score": entry["v2_score"],
                    "contradiction_rate": entry["v2_contradiction_rate"],
                    "n_claims": entry["n_claims"],
                    "n_passages": entry["n_passages"],
                }
            n_from_cache += 1
        else:
            passages = get_passages(row["context"])
            claims = get_claims(row["output"])
            result = faithfulness(claims, passages, nli)
            cache[item_id] = {
                "score": result.score,
                "contradiction_rate": result.contradiction_rate,
                "n_claims": len(claims),
                "n_passages": len(passages),
            }
            n_computed += 1

        if i % flush_every == 0 or i == total:
            save_cache(model_name, cache)
            print(f"  scored {i}/{total} items "
                  f"({n_from_cache} from cache, {n_computed} computed this run)...")
    return cache


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _auc_block(stat: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    auc, auc_lo, auc_hi = bootstrap_ci(stat, labels, roc_auc, n=n_boot, seed=seed)
    pr, pr_lo, pr_hi = bootstrap_ci(stat, labels, pr_auc, n=n_boot, seed=seed)
    return {
        "n": int(len(labels)),
        "n_hallucinated": int(np.sum(labels == 1)),
        "n_faithful": int(np.sum(labels == 0)),
        "roc_auc": {"point": auc, "ci_lo": auc_lo, "ci_hi": auc_hi},
        "pr_auc": {"point": pr, "ci_lo": pr_lo, "ci_hi": pr_hi},
    }


def run_experiment_a(scored: pd.DataFrame, n_boot: int, seed: int) -> dict:
    labels = scored["hallucinated"].to_numpy().astype(int)
    stat = -scored["score"].to_numpy(dtype=float)

    result = {"pooled": _auc_block(stat, labels, n_boot, seed)}
    per_task = {}
    for tt in TASK_TYPES:
        mask = (scored["task_type"] == tt).to_numpy()
        if mask.sum() == 0:
            continue
        sub_labels = labels[mask]
        if len(np.unique(sub_labels)) < 2:
            warnings.warn(f"Experiment A: task_type={tt} has a single class; skipping.")
            continue
        per_task[tt] = _auc_block(stat[mask], sub_labels, n_boot, seed)
    result["per_task_type"] = per_task
    return result


def run_experiment_b(scored: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> dict:
    """Test whether kappa and (1-F) are specialised to different failure modes.

    *** Only ONE target is scored here, deliberately. ***

    `filter_single_label_type` keeps rows where exactly one of has_conflict /
    has_baseless is True, so within this subset `has_baseless` IS `not
    has_conflict`. ROC-AUC against a complemented label is exactly `1 - AUC`,
    so scoring both targets yields four numbers carrying only two independent
    facts, and makes "kappa wins on conflict" and "(1-F) wins on baseless" the
    same statement written twice:

        AUC(1-F, B) > AUC(kappa, B)
      = 1 - AUC(1-F, C) > 1 - AUC(kappa, C)
      = AUC(kappa, C) > AUC(1-F, C)

    An earlier version of this experiment tabulated all four and reported those
    two implications as independent corroboration of the design claim. They are
    not independent; the conjunction was `A and A`. The complement is now
    recorded once as `auc_identity_note` and the verdict rests on a single
    paired test.

    Note also what this subset can and cannot show. Every item in it is
    hallucinated, so neither statistic is being asked to separate hallucinated
    from faithful output -- only conflict-type from baseless-type hallucination.
    A result here says the two statistics respond to different *kinds* of
    failure; it says nothing about either one's absolute detection ability,
    which is Experiment A's job.
    """
    single = filter_single_label_type(scored)
    kappa = single["contradiction_rate"].to_numpy(dtype=float)
    one_minus_f = 1.0 - single["score"].to_numpy(dtype=float)
    has_conflict = single["has_conflict"].to_numpy().astype(int)

    def _ci(stat, lbl):
        point, lo, hi = bootstrap_ci(stat, lbl, roc_auc, n=n_boot, seed=seed)
        return {"point": point, "ci_lo": lo, "ci_hi": hi}

    aucs = {
        "kappa_vs_has_conflict": _ci(kappa, has_conflict),
        "one_minus_F_vs_has_conflict": _ci(one_minus_f, has_conflict),
    }

    stat_k, stat_f, diff, p_value = paired_permutation_test(
        kappa, one_minus_f, has_conflict, roc_auc, n=n_perm, seed=seed)

    # kappa carries conflict-specific signal only if its own CI clears chance;
    # it is *better than* (1-F) at this only if the paired gap is significant.
    kappa_beats_chance = bool(aucs["kappa_vs_has_conflict"]["ci_lo"] > 0.5)
    f_beats_chance = bool(aucs["one_minus_F_vs_has_conflict"]["ci_lo"] > 0.5)
    gap_significant = bool(p_value < 0.05)

    if kappa_beats_chance and gap_significant:
        verdict = "SUPPORTED"
    elif kappa_beats_chance:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    return {
        "n_single_label_items": int(len(single)),
        "n_has_conflict": int(has_conflict.sum()),
        "n_has_baseless": int((1 - has_conflict).sum()),
        "aucs": aucs,
        "auc_identity_note": (
            "has_baseless == not has_conflict on this subset, so "
            "AUC(stat, has_baseless) == 1 - AUC(stat, has_conflict) exactly; "
            "scoring the complement adds no information and is not reported."
        ),
        "paired_permutation_test": {
            "statistic": "roc_auc",
            "target": "has_conflict",
            "stat_kappa": stat_k,
            "stat_one_minus_F": stat_f,
            "diff_kappa_minus_one_minus_F": diff,
            "p_value": p_value,
            "n_permutations": n_perm,
        },
        "kappa_beats_chance": kappa_beats_chance,
        "one_minus_F_beats_chance": f_beats_chance,
        "gap_significant": gap_significant,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(scored: pd.DataFrame, exp_a: dict, exp_b: dict, out_path: Path) -> None:
    fig, axd = plt.subplot_mosaic(
        [["pooled", "summary", "data2txt", "qa"], ["bar", "bar", "bar", "bar"]],
        figsize=(18, 9),
    )

    labels_all = scored["hallucinated"].to_numpy().astype(int)
    stat_all = -scored["score"].to_numpy(dtype=float)

    panels = [("pooled", "Pooled", None)]
    panels += [
        ("summary", "Summary", "Summary"),
        ("data2txt", "Data2txt", "Data2txt"),
        ("qa", "QA", "QA"),
    ]
    for key, title, task_type in panels:
        ax = axd[key]
        if task_type is None:
            mask = np.ones(len(scored), dtype=bool)
            block = exp_a["pooled"]
        else:
            mask = (scored["task_type"] == task_type).to_numpy()
            block = exp_a["per_task_type"].get(task_type)
        if block is None or mask.sum() == 0:
            ax.set_title(f"{title} (n/a)")
            continue
        fpr, tpr, _ = roc_curve(stat_all[mask], labels_all[mask])
        ax.plot(fpr, tpr, label=f"AUC={block['roc_auc']['point']:.3f}", color="C1")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{title} (n={block['n']})", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

    ax_bar = axd["bar"]
    # Only the has_conflict target is plotted. The has_baseless bars an earlier
    # version drew were exactly 1 - these, since has_baseless == not has_conflict
    # on this subset -- decoration that looked like extra evidence.
    names = ["kappa vs\nhas_conflict", "1-F vs\nhas_conflict"]
    keys = ["kappa_vs_has_conflict", "one_minus_F_vs_has_conflict"]
    points = [exp_b["aucs"][k]["point"] for k in keys]
    los = [exp_b["aucs"][k]["ci_lo"] for k in keys]
    his = [exp_b["aucs"][k]["ci_hi"] for k in keys]
    err_lo = [p - lo for p, lo in zip(points, los)]
    err_hi = [hi - p for p, hi in zip(points, his)]
    colors = ["C0", "C1"]
    x = np.arange(len(names))
    ax_bar.bar(x, points, color=colors, alpha=0.8, width=0.25)
    ax_bar.errorbar(x, points, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4)
    ax_bar.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    # This axis spans the full figure width, so without an explicit xlim two bars
    # stretch into slabs. Keep them readable next to the four ROC panels above.
    ax_bar.set_xlim(-0.6, len(names) - 0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names)
    ax_bar.set_ylabel("ROC-AUC")
    ax_bar.set_ylim(0, 1)
    verdict = exp_b["verdict"]
    perm_p = exp_b["paired_permutation_test"]["p_value"]
    ax_bar.set_title(
        f"Experiment B: is the conflict/unsupported split real? -- {verdict} "
        f"(n={exp_b['n_single_label_items']} single-label-type items, "
        f"paired p={perm_p:.3f})",
        fontsize=10,
    )

    fig.suptitle("RAGTruth validation: faithfulness detection, and the conflict/unsupported split",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_auc(block: dict, metric: str) -> str:
    d = block[metric]
    return f"{d['point']:.3f} ({d['ci_lo']:.3f}, {d['ci_hi']:.3f})"


def write_report(summary: dict, out_dir: Path) -> None:
    exp_a = summary["experiment_a"]
    exp_b = summary["experiment_b"]
    pooled = exp_a["pooled"]

    lines = [
        "# RAGTruth validation -- METRICS.md Part III, independent benchmark",
        "",
        "**2,700 independently, human-annotated RAG outputs from RAGTruth** (Wu et al., "
        "2024): Yelp reviews (Data2txt), CNN/DailyMail (Summary) and MARCO passages (QA).",
        "",
        "**Why this benchmark.** The default NLI backbone "
        f"(`{summary['nli_model']}`) is fine-tuned on MNLI, FEVER and ANLI. Evaluating a "
        "faithfulness metric against a benchmark built from any of those corpora would be "
        "circular -- the model would already have seen that exact style of premise/hypothesis "
        "pair as a training signal. RAGTruth is built from Yelp/CNN-DailyMail/MARCO text, none "
        "of which are MNLI, FEVER or ANLI, so it sits outside the NLI model's training "
        "distribution. That is the entire reason a result measured here is informative rather "
        "than circular.",
        "",
        f"Sampled **{summary['n_items']} of {summary['n_total_available']}** items "
        f"(seed={summary['seed']}), stratified evenly across `task_type`.",
        "",
        "## Observed class balance",
        "",
        "| Scope | n | hallucinated | rate |",
        "|---|---:|---:|---:|",
    ]
    cb = summary["class_balance"]
    lines.append(f"| Sampled (pooled) | {cb['sampled']['n']} | {cb['sampled']['n_hallucinated']} | "
                 f"{cb['sampled']['rate']:.3f} |")
    for tt in TASK_TYPES:
        row = cb["sampled_per_task_type"].get(tt)
        if row:
            lines.append(f"| Sampled -- {tt} | {row['n']} | {row['n_hallucinated']} | {row['rate']:.3f} |")
    lines.append(f"| Full RAGTruth test split (pooled) | {cb['full']['n']} | {cb['full']['n_hallucinated']} | "
                 f"{cb['full']['rate']:.3f} |")
    for tt in TASK_TYPES:
        row = cb["full_per_task_type"].get(tt)
        if row:
            lines.append(f"| Full -- {tt} | {row['n']} | {row['n_hallucinated']} | {row['rate']:.3f} |")

    lines += [
        "",
        "## Experiment A -- how well does faithfulness detect real hallucinations?",
        "",
        "Decision statistic is *negated* faithfulness (lower faithfulness -> more likely "
        "hallucinated), via "
        "`faithfulness.faithfulness(split_claims(output), chunk_passages([context]), nli).score`.",
        "",
        "| Scope | n | ROC-AUC (95% CI) | PR-AUC (95% CI) |",
        "|---|---:|---|---|",
        f"| Pooled | {pooled['n']} | {_fmt_auc(pooled, 'roc_auc')} | {_fmt_auc(pooled, 'pr_auc')} |",
    ]
    for tt in TASK_TYPES:
        block = exp_a["per_task_type"].get(tt)
        if block is None:
            lines.append(f"| {tt} | -- | n/a (single class) | |")
            continue
        lines.append(
            f"| {tt} | {block['n']} | {_fmt_auc(block, 'roc_auc')} | {_fmt_auc(block, 'pr_auc')} |"
        )
    lines.append("")

    lines += [
        "## Experiment B -- is the conflict/unsupported split real, or decorative?",
        "",
        "METRICS.md reports faithfulness `F` (mean max entailment) and contradiction rate "
        "`kappa` (mean max contradiction) separately, arguing refuted and unsupported claims "
        "are different failure modes. RAGTruth's two label types map directly: "
        "`evident_conflict` ~ refuted, `baseless_info` ~ unsupported. Tested here on the "
        f"**{exp_b['n_single_label_items']}** sampled items with exactly one label type present "
        "(so the two signals are not confounded): "
        f"{exp_b['n_has_conflict']} conflict-only, {exp_b['n_has_baseless']} baseless-only.",
        "",
        "**One target, not two.** On this subset `has_baseless` is by construction "
        "`not has_conflict`, and ROC-AUC against a complemented label is exactly `1 - AUC`. "
        "Scoring both targets therefore produces four numbers containing two independent "
        "facts, and turns \"kappa wins on conflict\" and \"(1-F) wins on baseless\" into the "
        "same statement written twice. An earlier version of this experiment did exactly "
        "that and reported the two as independent corroboration; they were not. Only the "
        "`has_conflict` target is reported below.",
        "",
        "**What this subset cannot show.** Every item here is hallucinated, so neither "
        "statistic is being asked to separate hallucinated from faithful output -- only "
        "conflict-type from baseless-type hallucination. Absolute detection ability is "
        "Experiment A's question, not this one's.",
        "",
        "| Statistic | Target | ROC-AUC (95% CI) | Clears chance? |",
        "|---|---|---|---|",
    ]
    for label, key, flag in [
        ("kappa (contradiction rate)", "kappa_vs_has_conflict", "kappa_beats_chance"),
        ("1 - F (negated faithfulness)", "one_minus_F_vs_has_conflict", "one_minus_F_beats_chance"),
    ]:
        d = exp_b["aucs"][key]
        clears = "yes" if exp_b[flag] else "no -- CI includes 0.5"
        lines.append(
            f"| {label} | has_conflict | {d['point']:.3f} ({d['ci_lo']:.3f}, {d['ci_hi']:.3f}) "
            f"| {clears} |")

    perm = exp_b["paired_permutation_test"]
    lines += [
        "",
        f"Paired permutation test (n={perm['n_permutations']}), kappa vs (1-F) on "
        f"`has_conflict`: gap = **{perm['diff_kappa_minus_one_minus_F']:+.4f}**, "
        f"**p = {perm['p_value']:.4f}**.",
    ]

    verdict_map = {
        "SUPPORTED": (
            "**The design claim is SUPPORTED**: kappa carries conflict-specific signal its "
            "own CI separates from chance, and it beats (1-F) at that by a margin the paired "
            "permutation test distinguishes from zero. Reporting the two separately in "
            "METRICS.md is substantive."
        ),
        "PARTIALLY SUPPORTED": (
            "**The design claim is PARTIALLY SUPPORTED, and the qualification matters.** "
            "kappa does carry conflict-specific signal -- its CI clears chance, while (1-F)'s "
            "does not, so the two statistics are not interchangeable. But kappa being "
            f"*better* than (1-F) at this does not reach significance (p = {perm['p_value']:.4f} "
            "against a 0.05 threshold), so the strong form of the claim -- that kappa is "
            "demonstrably the right instrument for contradictions -- is not established by "
            "this evidence. Reporting kappa separately is defensible; claiming it is proven "
            "superior is not."
        ),
        "NOT SUPPORTED": (
            "**The design claim is NOT SUPPORTED**: kappa's CI includes chance on the target "
            "it is supposed to be specialised for. On this evidence, reporting it as a "
            "separate number in METRICS.md is decorative rather than substantive, and that "
            "should be stated plainly rather than downplayed."
        ),
    }
    verdict_line = verdict_map[exp_b["verdict"]]
    lines += ["", verdict_line, "", "## Artefacts", "",
              "- `ragtruth_validation.json` -- full numeric results",
              "- `ragtruth_validation.png` -- ROC curves (pooled + per task_type) and Experiment B's two AUCs with CI bars",
              ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ragtruth_validation.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=int, default=900, help="total items to sample, stratified across task_type")
    ap.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="use all 2,700 items instead of --items")
    args = ap.parse_args()

    n_boot = 10000
    n_perm = 10000

    print(f"Loading RAGTruth ({RAGTRUTH_REPO}) ...")
    df = load_ragtruth()
    df = add_derived_labels(df)
    n_total = len(df)
    print(f"Loaded {n_total} items. Observed class balance:")
    print(f"  pooled: {int(df['hallucinated'].sum())}/{n_total} hallucinated "
          f"({df['hallucinated'].mean():.3f})")
    per_task_full = {}
    for tt in TASK_TYPES:
        sub = df[df["task_type"] == tt]
        rate = sub["hallucinated"].mean() if len(sub) else float("nan")
        per_task_full[tt] = {"n": int(len(sub)), "n_hallucinated": int(sub["hallucinated"].sum()), "rate": float(rate)}
        print(f"  {tt}: {per_task_full[tt]['n_hallucinated']}/{per_task_full[tt]['n']} hallucinated ({rate:.3f})")

    n_items = n_total if args.full else min(args.items, n_total)
    sample_df = stratified_sample(df, n_items, args.seed)
    print(f"\nSampled {len(sample_df)} items (seed={args.seed}, stratified by task_type).")

    nli = NLIScorer(args.nli_model)
    cache = load_cache(args.nli_model)
    print(f"NLI model: {args.nli_model}. Cache: {cache_path_for_model(args.nli_model)} "
          f"({len(cache)} items already cached).")
    print("Scoring (faithfulness.faithfulness) ...")
    cache = score_items(sample_df, nli, cache, args.nli_model)

    records = [cache[str(row["id"])] for _, row in sample_df.iterrows()]
    scored = sample_df.reset_index(drop=True).copy()
    scored["score"] = [r["score"] for r in records]
    scored["contradiction_rate"] = [r["contradiction_rate"] for r in records]
    scored["n_claims"] = [r["n_claims"] for r in records]
    scored["n_passages"] = [r["n_passages"] for r in records]

    valid = ~scored["score"].isna()
    n_dropped = int((~valid).sum())
    if n_dropped:
        warnings.warn(f"Dropping {n_dropped} item(s) with NaN scores.")
        scored = scored[valid].reset_index(drop=True)

    print(f"\nAvg claims/item: {scored['n_claims'].mean():.2f}, avg passages/item: {scored['n_passages'].mean():.2f}")

    print("\nRunning Experiment A (detection performance, pooled + per task_type) ...")
    exp_a = run_experiment_a(scored, n_boot, args.seed)

    print("Running Experiment B (conflict/unsupported split: real or decorative?) ...")
    exp_b = run_experiment_b(scored, n_boot, n_perm, args.seed)

    summary = {
        "nli_model": args.nli_model,
        "seed": args.seed,
        "n_items": len(scored),
        "n_total_available": n_total,
        "n_dropped_nan": n_dropped,
        "class_balance": {
            "sampled": {
                "n": len(scored),
                "n_hallucinated": int(scored["hallucinated"].sum()),
                "rate": float(scored["hallucinated"].mean()),
            },
            "sampled_per_task_type": {
                tt: {
                    "n": int((scored["task_type"] == tt).sum()),
                    "n_hallucinated": int(scored.loc[scored["task_type"] == tt, "hallucinated"].sum()),
                    "rate": float(scored.loc[scored["task_type"] == tt, "hallucinated"].mean())
                    if (scored["task_type"] == tt).any() else float("nan"),
                }
                for tt in TASK_TYPES
            },
            "full": {
                "n": n_total,
                "n_hallucinated": int(df["hallucinated"].sum()),
                "rate": float(df["hallucinated"].mean()),
            },
            "full_per_task_type": per_task_full,
        },
        "experiment_a": exp_a,
        "experiment_b": exp_b,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ragtruth_validation.json").write_text(json.dumps(summary, indent=2))
    make_plot(scored, exp_a, exp_b, OUT_DIR / "ragtruth_validation.png")
    write_report(summary, OUT_DIR)

    print()
    print("=" * 72)
    pooled = exp_a["pooled"]
    print(f"POOLED  ROC-AUC={pooled['roc_auc']['point']:.4f} "
          f"({pooled['roc_auc']['ci_lo']:.4f}, {pooled['roc_auc']['ci_hi']:.4f})")
    for tt in TASK_TYPES:
        block = exp_a["per_task_type"].get(tt)
        if block:
            print(f"{tt:9s} ROC-AUC={block['roc_auc']['point']:.4f} "
                  f"({block['roc_auc']['ci_lo']:.4f}, {block['roc_auc']['ci_hi']:.4f})")
    print(f"Experiment B verdict: {exp_b['verdict']} "
          f"(kappa vs 1-F on has_conflict, paired p="
          f"{exp_b['paired_permutation_test']['p_value']:.4f})")
    print(f"Wrote {OUT_DIR / 'ragtruth_validation.md'}, ragtruth_validation.json, ragtruth_validation.png")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
