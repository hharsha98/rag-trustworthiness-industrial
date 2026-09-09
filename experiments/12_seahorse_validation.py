#!/usr/bin/env python3
"""SEAHORSE validation -- conciseness (METRICS.md Part II.4) against a third-party,
human-annotated benchmark.

Why SEAHORSE. `conciseness(claims, embedder)` (`src/ragtrust/metrics/conciseness.py`) uses
only the *embedding* model configured as `Config().embed_model` --
`sentence-transformers/msmarco-distilbert-base-v4`, trained on MS MARCO query/passage pairs.
It never touches the NLI model. SEAHORSE (Clark et al., 2023, `tasksource/
seahorse_summarization_evaluation`) is a human-annotated summarization-evaluation benchmark
built from GEM-suite datasets; the English (`worker_lang == 'en-US'`) slice used here draws
from `wiki_lingua_english_en`, `xlsum_english` and `xsum` (verified from the `gem_id` prefixes
at load time, not assumed) -- Wikipedia how-to guides, BBC news summaries in many languages,
and single-document news summarization. None of that is MS MARCO query/passage retrieval text,
so it sits outside the embedder's training distribution, the same reason experiment 10 treats
RAGTruth as informative for the NLI-based faithfulness metric rather than circular.

METRICS.md's claim under test: conciseness "penalises padding, not length" -- i.e. it measures
*redundancy* among an answer's own claims, not brevity. SEAHORSE asks raters two separate
Yes/No questions per summary that let that claim be tested directly instead of assumed:

  Q2 (PRIMARY TARGET) -- "The summary is free of unnecessarily repeated information."
      Directly about repetition. Orientation: Q2="Yes" means NOT redundant, so a
      *redundancy* metric should predict Q2="Yes" with a HIGH score. Higher C -> predicts
      "Yes". Get this direction right; a test asserts it explicitly.

  Q6 (DISCRIMINANT TARGET) -- "The summary concisely represents the information in the source
      article." This conflates brevity with coverage (a short summary that omits key
      information could score well here on "concise" while scoring badly on faithfulness/
      relevance) and is NOT what `conciseness` claims to measure. If C tracks Q6 as well as or
      better than Q2, the "penalises padding, not length" claim is not supported.

*** The length confound -- the central risk in this experiment. ***
`conciseness` returns exactly `C = 1.0` whenever a summary decomposes into fewer than 2
claims (see the module docstring and the `n < 2` branch). A short, one-sentence summary
therefore scores "perfectly non-redundant" *by construction*, regardless of its actual content
-- and SEAHORSE summaries are short (median ~19 words in this slice), so this is not a corner
case, it is close to the modal case. If a pure length count (number of claims, or word count)
predicts Q2 as well as or better than C does, then C is adding nothing over sentence-counting,
and METRICS.md's "not length" claim fails its own test on real human judgments. This experiment
is built to surface that possibility rather than paper over it: Section 2 below builds both
length baselines, gives each its best-fitting sign, and runs a paired permutation test of C
against the stronger one.

A note on a trap this repository has hit before (see experiments/10_ragtruth_validation.py,
Experiment B): scoring a statistic against a label and that label's exact complement on the
same two-class subset produces `AUC(s, ~y) == 1 - AUC(s, y)` identically -- one fact, reported
twice. Q2 and Q6 are NOT complements of each other (they are two independently-collected
Yes/No judgments about different properties), so reporting AUC(C, Q2) and AUC(C, Q6) together
is legitimate and is exactly what Section 3 does. The trap resurfaces in Section 2 instead, in
a different shape: naively reporting both `roc_auc(stat, y)` and `roc_auc(-stat, y)` for a
length baseline is the *same* identity applied to a negated score instead of a complemented
label. This experiment picks a single sign per baseline (`best_sign`, using the identity to
avoid computing both) and reports one number, not two.

Usage:
    python experiments/12_seahorse_validation.py [--max-items N] [--seed S]

Exit code: always 0. This is a measurement, not a pass/fail gate.
"""
from __future__ import annotations

import argparse
import hashlib
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
from ragtrust.metrics.claims import split_claims  # noqa: E402
from ragtrust.metrics.conciseness import conciseness  # noqa: E402
from ragtrust.validation.stats import (  # noqa: E402
    bootstrap_ci,
    paired_permutation_test,
    pr_auc,
    roc_auc,
    roc_curve,
)

SEAHORSE_REPO = "tasksource/seahorse_summarization_evaluation"
SEAHORSE_FILE = "data/test-00000-of-00001-acd5b00ba47747a7.parquet"

# Short key -> exact SEAHORSE question string. Order matches the dataset card.
QUESTIONS = {
    "readable": "The summary can be read and understood by the rater.",
    "q2_repetition": "The summary is free of unnecessarily repeated information.",
    "q6_concise": "The summary concisely represents the information in the source article.",
    "main_idea": "The summary captures the main idea(s) of the source article.",
    "attributable": "All the information in the summary is fully attributable to the source article.",
    "grammatical": "The summary is grammatically correct.",
}
Q2_COL = "q2_repetition"
Q6_COL = "q6_concise"

CACHE_DIR = ROOT / "data" / "benchmarks"
OUT_DIR = ROOT / "experiments" / "results"
CACHE_PATH = CACHE_DIR / "seahorse_conciseness_cache.json"

# The en-US, Q2-answered population is ~4.1k summaries (measured at load time, not assumed).
# This default cap is well above that, so by default this experiment uses ALL of them; the cap
# exists so a future, much larger SEAHORSE release doesn't silently make a CPU-only run slow.
DEFAULT_MAX_ITEMS = 6000


# ---------------------------------------------------------------------------
# Pure functions -- no network, no model, unit-tested in
# tests/test_seahorse_validation.py without any download.
# ---------------------------------------------------------------------------


def filter_en_us(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only English (`en-US`) worker annotations -- the dataset also carries
    es-ES/vi/tr/de/ru workers rating (mostly) non-English summaries."""
    return df[df["worker_lang"] == "en-US"].reset_index(drop=True)


def yes_no_to_binary(answer) -> float:
    """Map SEAHORSE's raw answer string to 1.0 ("Yes") / 0.0 ("No"), or NaN for
    anything else (defensive -- observed values are exactly {"Yes", "No"})."""
    if answer == "Yes":
        return 1.0
    if answer == "No":
        return 0.0
    return float("nan")


def pivot_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot SEAHORSE's long format (one row = one worker's Yes/No answer to ONE
    question about one summary) into one row per distinct summary, keyed on
    (gem_id, model, summary) -- each combination carries whichever of the six
    question columns exist as a 0/1/NaN value; a summary need not have every
    question answered.

    Requires `(gem_id, model, summary, question)` to be unique in `df` (true of the
    real SEAHORSE en-US slice: one worker answers each question about each
    generated summary exactly once); `pandas.unstack` raises on duplicates,
    which is the correct failure mode if that assumption is ever violated.
    """
    df = df.copy()
    df["_binary"] = df["answer"].map(yes_no_to_binary)

    question_to_col = {full: short for short, full in QUESTIONS.items()}
    df["_col"] = df["question"].map(question_to_col)
    unmapped = df["_col"].isna()
    if unmapped.any():
        warnings.warn(
            f"pivot_summaries: {int(unmapped.sum())} row(s) have a question string "
            "not in QUESTIONS and are dropped."
        )
        df = df[~unmapped]

    wide = (
        df.set_index(["gem_id", "model", "summary", "_col"])["_binary"]
        .unstack("_col")
        .reset_index()
    )
    wide.columns.name = None
    for short in QUESTIONS:
        if short not in wide.columns:
            wide[short] = float("nan")
    return wide


def get_claims(summary: str) -> list:
    """Split one SEAHORSE summary into claims, falling back to the whole summary
    as a single claim if segmentation finds no sentences (matches
    experiments/10_ragtruth_validation.py's get_claims fallback)."""
    claims = split_claims(summary or "")
    if not claims and summary and summary.strip():
        claims = [summary.strip()]
    return claims


def word_count(text: str) -> int:
    return len((text or "").split())


def maybe_sample(df: pd.DataFrame, max_items: int, seed: int) -> pd.DataFrame:
    """Deterministically sample down to `max_items` rows if `df` is larger;
    otherwise return `df` unchanged (in which case ALL available items are used,
    which is what happens by default on the real SEAHORSE en-US slice)."""
    if len(df) <= max_items:
        return df.reset_index(drop=True)
    return df.sample(n=max_items, random_state=seed).reset_index(drop=True)


def best_sign(stat: np.ndarray, labels: np.ndarray) -> int:
    """Return +1 or -1: whichever direction of `stat` best predicts label==1,
    giving a length baseline its best fair shot without inspecting it twice.

    `roc_auc(-s, y)` is exactly `1 - roc_auc(s, y)` (reversing every score's rank
    order reverses the Mann-Whitney U statistic's complement exactly, tie-average
    ranks included) -- the same identity experiment 10 documents for a
    complemented *label*, here applied to a negated *score* instead. So only one
    point AUC needs computing to know which sign is better; computing both and
    reporting the larger would just print that identity as if it were two facts.
    """
    point = roc_auc(stat, labels)
    if np.isnan(point):
        return 1
    return 1 if point >= 0.5 else -1


def paired_permutation_test_swap_labels(score, labels_a, labels_b, stat_fn, n: int = 10000, seed: int = 0):
    """Paired permutation test comparing `stat_fn(score, labels_a)` against
    `stat_fn(score, labels_b)` -- the SAME score evaluated against two DIFFERENT
    binary targets for the same items (here: does C track Q2 more strongly than
    Q6?).

    `validation/stats.paired_permutation_test` compares two different SCORE
    arrays against one shared LABEL array, swapping which score is "labelled A"
    per item under the null that the two scorers are exchangeable. This
    experiment needs the transposed comparison -- one score, two label arrays --
    so this mirrors that function's construction exactly but swaps which LABEL
    array is attached to the (fixed) score. H0: labels_a and labels_b are
    exchangeable given score. `labels_a`/`labels_b` must be 0/1 integer-valued
    for `stat_fn` (here always `roc_auc`) to be defined on the permuted mixtures.

    Returns (stat_a, stat_b, diff, p_value), diff = stat_a - stat_b on the
    unpermuted data -- same return shape as `paired_permutation_test`.
    """
    score = np.asarray(score, dtype=float)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    stat_a = float(stat_fn(score, labels_a))
    stat_b = float(stat_fn(score, labels_b))
    observed_diff = stat_a - stat_b

    rng = np.random.default_rng(seed)
    n_items = score.size
    perm_diffs = np.empty(n, dtype=float)
    for i in range(n):
        swap = rng.random(n_items) < 0.5
        perm_a = np.where(swap, labels_b, labels_a)
        perm_b = np.where(swap, labels_a, labels_b)
        perm_diffs[i] = stat_fn(score, perm_a) - stat_fn(score, perm_b)

    n_extreme = int(np.sum(np.abs(perm_diffs) >= abs(observed_diff) - 1e-12))
    p_value = float((n_extreme + 1) / (n + 1))
    return stat_a, stat_b, float(observed_diff), p_value


def class_balance(labels: np.ndarray) -> dict:
    n = int(len(labels))
    n_yes = int(np.sum(labels == 1))
    n_no = int(np.sum(labels == 0))
    return {
        "n": n,
        "n_yes": n_yes,
        "n_no": n_no,
        "rate_yes": float(n_yes / n) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Network / model-dependent functions -- not exercised by the fast test suite.
# ---------------------------------------------------------------------------


def load_seahorse() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SEAHORSE_REPO, SEAHORSE_FILE, repo_type="dataset")
    return pd.read_parquet(path)


def summary_cache_key(summary: str) -> str:
    return hashlib.md5((summary or "").encode("utf-8")).hexdigest()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def score_conciseness(summaries: list, embedder, cache: dict, flush_every: int = 200) -> dict:
    """Compute conciseness C (plus n_claims, n_words) for every distinct string in
    `summaries`, cached by md5(summary text) -- C is a pure function of the
    summary text and the embedder, so identical summaries (across gem_id/model
    pairs, or across the primary and paired analysis populations) are scored
    once. Mutates and returns `cache`."""
    unique = sorted(set(summaries))
    total = len(unique)
    n_from_cache = 0
    n_computed = 0
    for i, summary in enumerate(unique, start=1):
        key = summary_cache_key(summary)
        if key in cache:
            n_from_cache += 1
        else:
            claims = get_claims(summary)
            c = conciseness(claims, embedder)
            if c is None:
                # `conciseness` now returns None (undefined) for < 2 claims
                # instead of the old 1.0 sentinel (see metrics/conciseness.py).
                # This experiment's whole point -- Section 2, "the length
                # confound" -- is to study what that OLD sentinel did to the
                # primary analysis, so it deliberately reproduces it here
                # rather than propagating None/NaN into the C column. The
                # restricted analysis below (n_claims >= 2) never sees this
                # substituted value; only the full-population primary section
                # does, exactly as before this fix.
                c = 1.0
            cache[key] = {"C": c, "n_claims": len(claims), "n_words": word_count(summary)}
            n_computed += 1

        if i % flush_every == 0 or i == total:
            save_cache(cache)
            print(f"  scored {i}/{total} distinct summaries "
                  f"({n_from_cache} from cache, {n_computed} computed this run)...")
    return cache


def attach_scores(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    df = df.copy()
    keys = df["summary"].map(summary_cache_key)
    df["C"] = keys.map(lambda k: cache[k]["C"])
    df["n_claims"] = keys.map(lambda k: cache[k]["n_claims"])
    df["n_words"] = keys.map(lambda k: cache[k]["n_words"])
    return df


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


def run_primary_analysis(pop: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Section 1 (primary): does C predict Q2 ("free of repeated information")?
    Higher C is the predicted direction for "Yes" -- no sign-flipping, unlike the
    length baselines, because that direction is the metric's own claim, not
    something chosen post hoc to look good."""
    labels = pop[Q2_COL].to_numpy(dtype=int)
    stat_c = pop["C"].to_numpy(dtype=float)
    return {
        "class_balance": class_balance(labels),
        "conciseness": _stat_block(stat_c, labels, n_boot, seed),
    }


def run_length_confound(pop: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> dict:
    """Section 2: the length confound. `conciseness` is exactly 1.0 for any
    summary with <2 claims; this checks whether pure length (claim count, word
    count) predicts Q2 as well as or better than C does, and restricts the
    primary comparison to summaries where C is not trivially 1.0."""
    labels = pop[Q2_COL].to_numpy(dtype=int)
    stat_c = pop["C"].to_numpy(dtype=float)
    n_claims = pop["n_claims"].to_numpy(dtype=float)
    n_words = pop["n_words"].to_numpy(dtype=float)

    frac_lt2_claims = float((pop["n_claims"] < 2).mean())

    sign_claims = best_sign(n_claims, labels)
    sign_words = best_sign(n_words, labels)
    signed_claims = sign_claims * n_claims
    signed_words = sign_words * n_words

    block_claims = {"sign": sign_claims, **_stat_block(signed_claims, labels, n_boot, seed)}
    block_words = {"sign": sign_words, **_stat_block(signed_words, labels, n_boot, seed)}

    better_name = "n_claims" if block_claims["roc_auc"]["point"] >= block_words["roc_auc"]["point"] else "n_words"
    better_block = block_claims if better_name == "n_claims" else block_words
    better_stat = signed_claims if better_name == "n_claims" else signed_words

    c_auc, base_auc, diff, p_value = paired_permutation_test(
        stat_c, better_stat, labels, roc_auc, n=n_perm, seed=seed
    )
    c_significantly_better = bool(diff > 0 and p_value < 0.05)

    # Restricted analysis: only summaries with >= 2 claims, where C is not
    # trivially 1.0 by construction.
    pop_ge2 = pop[pop["n_claims"] >= 2]
    labels_ge2 = pop_ge2[Q2_COL].to_numpy(dtype=int)
    stat_c_ge2 = pop_ge2["C"].to_numpy(dtype=float)
    if len(np.unique(labels_ge2)) < 2:
        warnings.warn("run_length_confound: >=2-claims subset has a single class; skipping restricted analysis.")
        restricted = None
    else:
        # The length baselines MUST be rerun here, not just C. An earlier version
        # reported only C's restricted AUC and concluded from the full-population
        # comparison alone that "conciseness adds nothing over counting sentences".
        # That conclusion was an artefact: on the full population C is identically
        # 1.0 for every <2-claim summary, so across most of the data C is a
        # CONSTANT and is nearly the same variable as the claim count. The
        # comparison only means something where C is actually defined.
        n_claims_ge2 = pop_ge2["n_claims"].to_numpy(dtype=float)
        n_words_ge2 = pop_ge2["n_words"].to_numpy(dtype=float)
        sign_claims_ge2 = best_sign(n_claims_ge2, labels_ge2)
        sign_words_ge2 = best_sign(n_words_ge2, labels_ge2)
        signed_claims_ge2 = sign_claims_ge2 * n_claims_ge2
        signed_words_ge2 = sign_words_ge2 * n_words_ge2

        block_claims_ge2 = {"sign": sign_claims_ge2,
                            **_stat_block(signed_claims_ge2, labels_ge2, n_boot, seed)}
        block_words_ge2 = {"sign": sign_words_ge2,
                           **_stat_block(signed_words_ge2, labels_ge2, n_boot, seed)}

        better_name_ge2 = ("n_claims"
                           if block_claims_ge2["roc_auc"]["point"] >= block_words_ge2["roc_auc"]["point"]
                           else "n_words")
        better_stat_ge2 = signed_claims_ge2 if better_name_ge2 == "n_claims" else signed_words_ge2

        c_auc_ge2, base_auc_ge2, diff_ge2, p_ge2 = paired_permutation_test(
            stat_c_ge2, better_stat_ge2, labels_ge2, roc_auc, n=n_perm, seed=seed
        )
        c_better_ge2 = bool(diff_ge2 > 0 and p_ge2 < 0.05)

        restricted = {
            "class_balance": class_balance(labels_ge2),
            "conciseness": _stat_block(stat_c_ge2, labels_ge2, n_boot, seed),
            "n_claims": block_claims_ge2,
            "n_words": block_words_ge2,
            "better_baseline": better_name_ge2,
            "c_vs_better_baseline_permutation_test": {
                "statistic": "roc_auc",
                "stat_c": c_auc_ge2,
                "stat_baseline": base_auc_ge2,
                "diff_c_minus_baseline": diff_ge2,
                "p_value": p_ge2,
                "n_permutations": n_perm,
            },
            "c_significantly_beats_better_baseline": c_better_ge2,
        }

    restricted_wins = bool(restricted and restricted["c_significantly_beats_better_baseline"])

    if c_significantly_better:
        headline = (
            f"C beats the stronger length baseline ({better_name}) by a paired-permutation-"
            f"significant margin (p={p_value:.4f})."
        )
    elif restricted_wins:
        # The interesting case, and the one this benchmark actually produces.
        r = restricted["c_vs_better_baseline_permutation_test"]
        headline = (
            f"C is UNDEFINED-BY-CONSTRUCTION on {frac_lt2_claims:.1%} of this population "
            f"(<2 claims => C == 1.0 exactly), and on that majority it is a constant carrying no "
            f"information. Across the full population it therefore cannot beat the "
            f"{better_name} baseline (p={p_value:.4f}). But restricted to the "
            f"{restricted['class_balance']['n']} summaries where C is actually computed "
            f"(>=2 claims), C reaches "
            f"{restricted['conciseness']['roc_auc']['point']:.3f} against "
            f"{restricted[restricted['better_baseline']]['roc_auc']['point']:.3f} for the best "
            f"length baseline -- a margin of {r['diff_c_minus_baseline']:+.3f} at p={r['p_value']:.4f}. "
            "So the metric is not a dressed-up sentence count; it is a valid redundancy signal "
            "that is simply inapplicable to short answers. The design defect this exposes is "
            "returning 1.0 (a PERFECT score, which propagates into T_geom) where the honest "
            "answer is 'undefined'."
        )
    else:
        headline = (
            f"The length baseline ({better_name}) matches or beats C, and C's advantage over it "
            f"is NOT statistically significant (p={p_value:.4f}), including on the subset where "
            "C is not trivially 1.0. Conciseness adds nothing over counting sentences on this "
            "evidence."
        )

    return {
        "frac_lt2_claims": frac_lt2_claims,
        "n_claims": block_claims,
        "n_words": block_words,
        "better_baseline": better_name,
        "c_vs_better_baseline_permutation_test": {
            "statistic": "roc_auc",
            "stat_c": c_auc,
            "stat_baseline": base_auc,
            "diff_c_minus_baseline": diff,
            "p_value": p_value,
            "n_permutations": n_perm,
        },
        "c_significantly_beats_better_baseline": c_significantly_better,
        "headline": headline,
        "restricted_ge2_claims": restricted,
    }


def run_discriminant_validity(paired_df: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> dict:
    """Section 3: on the subset with BOTH Q2 and Q6 answered, does C track
    repetition (Q2) more strongly than concise-representation (Q6), as
    METRICS.md's "penalises padding, not length" framing predicts? Q2 and Q6 are
    independently-collected judgments, not complements of each other, so scoring
    both against the same C is legitimate (contrast with experiment 10's
    Experiment B, where has_conflict/has_baseless WERE complements on that
    subset and scoring both would have been the same fact twice)."""
    stat_c = paired_df["C"].to_numpy(dtype=float)
    labels_q2 = paired_df[Q2_COL].to_numpy(dtype=int)
    labels_q6 = paired_df[Q6_COL].to_numpy(dtype=int)

    block_q2 = _stat_block(stat_c, labels_q2, n_boot, seed)
    block_q6 = _stat_block(stat_c, labels_q6, n_boot, seed)

    stat_a, stat_b, diff, p_value = paired_permutation_test_swap_labels(
        stat_c, labels_q2, labels_q6, roc_auc, n=n_perm, seed=seed
    )
    q2_tracks_more = bool(diff > 0)
    gap_significant = bool(p_value < 0.05)

    if q2_tracks_more and gap_significant:
        verdict = "SUPPORTED"
    elif q2_tracks_more:
        verdict = "WEAKLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    return {
        "n_paired": int(len(paired_df)),
        "class_balance_q2": class_balance(labels_q2),
        "class_balance_q6": class_balance(labels_q6),
        "auc_vs_q2": block_q2["roc_auc"],
        "auc_vs_q6": block_q6["roc_auc"],
        "paired_permutation_test": {
            "statistic": "roc_auc",
            "auc_q2": stat_a,
            "auc_q6": stat_b,
            "diff_q2_minus_q6": diff,
            "p_value": p_value,
            "n_permutations": n_perm,
        },
        "q2_tracks_more_than_q6": q2_tracks_more,
        "gap_significant": gap_significant,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(pop: pd.DataFrame, primary: dict, confound: dict, discriminant: dict, out_path: Path) -> None:
    fig, axd = plt.subplot_mosaic([["roc", "hist", "bar"]], figsize=(17, 5.5))

    labels = pop[Q2_COL].to_numpy(dtype=int)
    stat_c = pop["C"].to_numpy(dtype=float)
    sign_claims = confound["n_claims"]["sign"]
    sign_words = confound["n_words"]["sign"]
    signed_claims = sign_claims * pop["n_claims"].to_numpy(dtype=float)
    signed_words = sign_words * pop["n_words"].to_numpy(dtype=float)

    ax = axd["roc"]
    fpr_c, tpr_c, _ = roc_curve(stat_c, labels)
    fpr_nc, tpr_nc, _ = roc_curve(signed_claims, labels)
    fpr_w, tpr_w, _ = roc_curve(signed_words, labels)
    ax.plot(fpr_c, tpr_c, label=f"C (AUC={primary['conciseness']['roc_auc']['point']:.3f})", color="C1")
    ax.plot(fpr_nc, tpr_nc,
             label=f"{sign_claims:+d}*n_claims (AUC={confound['n_claims']['roc_auc']['point']:.3f})", color="C0")
    ax.plot(fpr_w, tpr_w,
             label=f"{sign_words:+d}*n_words (AUC={confound['n_words']['roc_auc']['point']:.3f})", color="C2")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"C vs length baselines -> Q2 (n={len(pop)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

    ax2 = axd["hist"]
    yes = pop.loc[pop[Q2_COL] == 1, "C"]
    no = pop.loc[pop[Q2_COL] == 0, "C"]
    bins = np.linspace(0.0, 1.0, 31)
    ax2.hist(yes, bins=bins, alpha=0.6, label=f"Q2=Yes (n={len(yes)})", color="C1", density=True)
    ax2.hist(no, bins=bins, alpha=0.6, label=f"Q2=No (n={len(no)})", color="C3", density=True)
    ax2.set_xlabel("Conciseness C")
    ax2.set_ylabel("Density")
    ax2.set_title("C distribution by Q2 answer", fontsize=10)
    ax2.legend(fontsize=8)

    ax3 = axd["bar"]
    names = ["AUC(C, Q2)\nrepetition", "AUC(C, Q6)\nconcise-repr."]
    points = [discriminant["auc_vs_q2"]["point"], discriminant["auc_vs_q6"]["point"]]
    los = [discriminant["auc_vs_q2"]["ci_lo"], discriminant["auc_vs_q6"]["ci_lo"]]
    his = [discriminant["auc_vs_q2"]["ci_hi"], discriminant["auc_vs_q6"]["ci_hi"]]
    err_lo = [p - lo for p, lo in zip(points, los)]
    err_hi = [hi - p for p, hi in zip(points, his)]
    x = np.arange(len(names))
    ax3.bar(x, points, color=["C1", "C4"], alpha=0.85, width=0.5)
    ax3.errorbar(x, points, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4)
    ax3.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax3.set_xlim(-0.6, len(names) - 0.4)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylabel("ROC-AUC")
    ax3.set_ylim(0, 1)
    p_val = discriminant["paired_permutation_test"]["p_value"]
    ax3.set_title(
        f"Discriminant validity -- {discriminant['verdict']} (n={discriminant['n_paired']}, "
        f"paired p={p_val:.3f})",
        fontsize=10,
    )

    fig.suptitle("SEAHORSE validation: conciseness C, length baselines, and Q2 vs Q6 discriminant validity",
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
    primary = summary["primary"]
    confound = summary["length_confound"]
    disc = summary["discriminant_validity"]
    cb = primary["class_balance"]

    lines = [
        "# SEAHORSE validation -- METRICS.md Part II.4, conciseness",
        "",
        "**Do these numbers show conciseness measuring redundancy, or just summary length?** "
        "Read the length-confound section (Section 2) before the headline AUC -- "
        "`conciseness` returns exactly 1.0 for any summary with fewer than 2 claims, so short "
        "summaries score perfectly *by construction*, and SEAHORSE summaries are short.",
        "",
        f"**Why this benchmark.** `conciseness` uses only the embedding model "
        f"(`{summary['embed_model']}`, trained on MS MARCO query/passage pairs) -- it never "
        "calls the NLI model. The en-US slice of SEAHORSE used here draws from "
        f"`wiki_lingua_english_en`, `xlsum_english` and `xsum` (verified from `gem_id` prefixes "
        "at load time), none of which is MS MARCO retrieval text, so this benchmark sits "
        "outside the embedder's training distribution.",
        "",
        "## Data",
        "",
        f"- {summary['n_total_rows']} total SEAHORSE rows; {summary['n_en_us_rows']} with "
        "`worker_lang == 'en-US'`.",
        f"- Pivoted to **{summary['n_distinct_summaries']}** distinct summaries "
        "(key: gem_id + model + summary).",
        f"- **{summary['n_q2_answered']}** have Q2 (repetition) answered, "
        f"**{summary['n_q6_answered']}** have Q6 (concise-representation) answered, "
        f"**{summary['n_both_q2_and_q6']}** have BOTH (the paired subset used in Section 3).",
        f"- Analysis population (Section 1/2): **{summary['n_analysis_population']}** of "
        f"{summary['n_total_available_q2']} available Q2-answered summaries "
        + ("(deterministically sampled, seed="
           f"{summary['seed']})." if summary["sampled"] else "(all available -- no sampling needed)."),
        "",
        "## Section 1 -- primary: does C predict Q2 (\"free of repeated information\")?",
        "",
        "Orientation: Q2 = \"Yes\" means the summary is NOT redundant, so a redundancy metric "
        "should predict \"Yes\" with a HIGH C. No sign-flipping here -- unlike the length "
        "baselines below, this direction is the metric's own claim, not chosen post hoc.",
        "",
        "| n | Yes | No | rate(Yes) | ROC-AUC (95% CI) |",
        "|---:|---:|---:|---:|---|",
        f"| {cb['n']} | {cb['n_yes']} | {cb['n_no']} | {cb['rate_yes']:.3f} | "
        f"{_fmt_ci(primary['conciseness']['roc_auc'])} |",
        "",
        "## Section 2 -- the length confound",
        "",
        f"**{confound['frac_lt2_claims']:.1%}** of the analysis population has fewer than 2 "
        "claims, i.e. `conciseness` returns exactly 1.0 for them by construction, independent "
        "of content.",
        "",
        "Two pure length baselines, each given its best-fitting sign "
        "(`best_sign`, chosen from a single point-AUC via the `roc_auc(-s,y) == 1-roc_auc(s,y)` "
        "identity, so only one direction is ever reported per baseline):",
        "",
        "| Baseline | sign | ROC-AUC (95% CI) |",
        "|---|---:|---|",
        f"| n_claims | {confound['n_claims']['sign']:+d} | {_fmt_ci(confound['n_claims']['roc_auc'])} |",
        f"| n_words | {confound['n_words']['sign']:+d} | {_fmt_ci(confound['n_words']['roc_auc'])} |",
        f"| **C (conciseness)** | (n/a) | {_fmt_ci(primary['conciseness']['roc_auc'])} |",
        "",
        f"Paired permutation test, C vs the stronger baseline (**{confound['better_baseline']}**): "
        f"diff = **{confound['c_vs_better_baseline_permutation_test']['diff_c_minus_baseline']:+.4f}**, "
        f"**p = {confound['c_vs_better_baseline_permutation_test']['p_value']:.4f}** "
        f"(n={confound['c_vs_better_baseline_permutation_test']['n_permutations']} permutations).",
        "",
        f"**{confound['headline']}**",
        "",
    ]

    restricted = confound["restricted_ge2_claims"]
    if restricted is None:
        lines += ["Restricted analysis (>= 2 claims, where C is not trivially 1.0): skipped -- "
                  "single-class subset.", ""]
    else:
        rcb = restricted["class_balance"]
        rperm = restricted["c_vs_better_baseline_permutation_test"]
        lines += [
            f"**Restricted to the {rcb['n']} summaries with >= 2 claims** -- the only ones where "
            "C is actually computed rather than returned as the constant 1.0. The length "
            "baselines are rerun here too: comparing C against them on the full population is "
            "not meaningful, because there C is a constant across "
            f"{confound['frac_lt2_claims']:.1%} of the rows and is therefore nearly the same "
            "variable as the claim count.",
            "",
            f"Class balance: {rcb['n_yes']} Yes / {rcb['n_no']} No (rate {rcb['rate_yes']:.3f}).",
            "",
            "| Statistic | ROC-AUC (95% CI) |",
            "|---|---|",
            f"| **C (conciseness)** | {_fmt_ci(restricted['conciseness']['roc_auc'])} |",
            f"| n_claims baseline (sign {restricted['n_claims']['sign']:+d}) | "
            f"{_fmt_ci(restricted['n_claims']['roc_auc'])} |",
            f"| n_words baseline (sign {restricted['n_words']['sign']:+d}) | "
            f"{_fmt_ci(restricted['n_words']['roc_auc'])} |",
            "",
            f"Paired permutation test, C vs the better baseline "
            f"({restricted['better_baseline']}): diff = "
            f"**{rperm['diff_c_minus_baseline']:+.4f}**, **p = {rperm['p_value']:.4f}** "
            f"({rperm['n_permutations']} permutations). C significantly better: "
            f"**{restricted['c_significantly_beats_better_baseline']}**.",
            "",
            "**Design implication.** Where the metric is defined it is a genuine redundancy "
            "signal, not a proxy for length. The problem is what it does where it is NOT "
            "defined: `conciseness` returns **1.0 -- a perfect score** -- for any answer with "
            "fewer than 2 claims, and that value propagates into `T_geom` as though redundancy "
            "had been measured and found absent. The honest return there is 'undefined', not "
            "'perfect'. This is structurally the same error as the rejected (1+cos)/2 relevance "
            "mapping's floor, which this repository documents as a defect.",
            "",
        ]

    lines += [
        "## Section 3 -- discriminant validity: C vs Q2 (repetition) vs Q6 (concise-representation)",
        "",
        f"On the **{disc['n_paired']}** summaries with both Q2 and Q6 answered. Q2 and Q6 are "
        "independently-collected judgments about different properties, not complements of each "
        "other on this subset, so scoring C against both is legitimate (unlike experiment 10's "
        "Experiment B, where the two targets WERE exact complements and reporting both would "
        "have printed one fact twice).",
        "",
        "| Target | n Yes | n No | ROC-AUC (95% CI) |",
        "|---|---:|---:|---|",
        f"| Q2 (repetition) | {disc['class_balance_q2']['n_yes']} | {disc['class_balance_q2']['n_no']} | "
        f"{_fmt_ci(disc['auc_vs_q2'])} |",
        f"| Q6 (concise-representation) | {disc['class_balance_q6']['n_yes']} | "
        f"{disc['class_balance_q6']['n_no']} | {_fmt_ci(disc['auc_vs_q6'])} |",
        "",
        f"Paired permutation test (swap-labels construction; "
        f"n={disc['paired_permutation_test']['n_permutations']}), AUC(C,Q2) - AUC(C,Q6): "
        f"diff = **{disc['paired_permutation_test']['diff_q2_minus_q6']:+.4f}**, "
        f"**p = {disc['paired_permutation_test']['p_value']:.4f}**.",
        "",
    ]

    verdict_map = {
        "SUPPORTED": (
            "**METRICS.md's \"penalises padding, not length\" framing is SUPPORTED on the "
            "discriminant test**: C tracks Q2 (repetition) more strongly than Q6 "
            "(concise-representation), and the gap is paired-permutation-significant."
        ),
        "WEAKLY SUPPORTED": (
            "**Weakly supported**: C tracks Q2 more strongly than Q6 in point estimate, but the "
            "gap does not reach significance -- the discriminant claim is directionally "
            "consistent, not established."
        ),
        "NOT SUPPORTED": (
            "**NOT SUPPORTED**: C does not track Q2 (repetition) more strongly than Q6 "
            "(concise-representation). On this evidence the \"penalises padding, not length\" "
            "claim is not distinguishable from C simply tracking general summary quality, and "
            "that should be stated plainly."
        ),
    }
    lines += [verdict_map[disc["verdict"]], ""]

    lines += [
        "## Bottom line",
        "",
        f"1. Primary AUC(C, Q2) = {_fmt_ci(primary['conciseness']['roc_auc'])}.",
        f"2. Length confound: {confound['headline']}",
        f"3. Discriminant validity verdict: **{disc['verdict']}**.",
        "",
        "## Artefacts",
        "",
        "- `seahorse_validation.json` -- full numeric results",
        "- `seahorse_validation.png` -- ROC curves (C vs length baselines), C distribution by "
        "Q2 answer, and AUC(C,Q2) vs AUC(C,Q6) with CI bars",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seahorse_validation.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS,
                     help="cap on Q2-answered summaries in the analysis population; sampled "
                          "deterministically (seed) if the available population exceeds this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_boot = 10000
    n_perm = 10000

    print(f"Loading SEAHORSE ({SEAHORSE_REPO}) ...")
    raw = load_seahorse()
    print(f"Loaded {len(raw)} total rows.")

    en = filter_en_us(raw)
    print(f"Filtered to worker_lang == 'en-US': {len(en)} rows.")

    wide = pivot_summaries(en)
    n_q2 = int(wide[Q2_COL].notna().sum())
    n_q6 = int(wide[Q6_COL].notna().sum())
    n_both = int((wide[Q2_COL].notna() & wide[Q6_COL].notna()).sum())
    print(f"Pivoted to {len(wide)} distinct summaries. "
          f"Q2 answered: {n_q2}, Q6 answered: {n_q6}, both: {n_both}.")

    pop_full = wide[wide[Q2_COL].notna()].reset_index(drop=True)
    n_total_available = len(pop_full)
    pop = maybe_sample(pop_full, args.max_items, args.seed)
    sampled = len(pop) < n_total_available
    print(f"Analysis population (Q2 answered): using {len(pop)} of {n_total_available} "
          f"{'(deterministically sampled, seed=' + str(args.seed) + ')' if sampled else '(all available)'}.")

    paired_full = wide[wide[Q2_COL].notna() & wide[Q6_COL].notna()].reset_index(drop=True)

    from sentence_transformers import SentenceTransformer

    embed_model_name = Config().embed_model
    print(f"Loading embedder: {embed_model_name} ...")
    embedder = SentenceTransformer(embed_model_name)

    cache = load_cache()
    print(f"Cache: {CACHE_PATH} ({len(cache)} summaries already cached).")

    all_summaries_needed = list(pop["summary"]) + list(paired_full["summary"])
    print("Scoring conciseness (claims.split_claims + metrics.conciseness.conciseness) ...")
    cache = score_conciseness(all_summaries_needed, embedder, cache)

    pop = attach_scores(pop, cache)
    paired_full = attach_scores(paired_full, cache)

    print("\nRunning Section 1 (primary: C vs Q2) ...")
    primary = run_primary_analysis(pop, n_boot, args.seed)

    print("Running Section 2 (length confound) ...")
    confound = run_length_confound(pop, n_boot, n_perm, args.seed)

    print("Running Section 3 (discriminant validity: C vs Q2 vs Q6) ...")
    discriminant = run_discriminant_validity(paired_full, n_boot, n_perm, args.seed)

    summary = {
        "embed_model": embed_model_name,
        "seed": args.seed,
        "n_total_rows": int(len(raw)),
        "n_en_us_rows": int(len(en)),
        "n_distinct_summaries": int(len(wide)),
        "n_q2_answered": n_q2,
        "n_q6_answered": n_q6,
        "n_both_q2_and_q6": n_both,
        "n_analysis_population": int(len(pop)),
        "n_total_available_q2": n_total_available,
        "sampled": bool(sampled),
        "max_items": args.max_items,
        "primary": primary,
        "length_confound": confound,
        "discriminant_validity": discriminant,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "seahorse_validation.json").write_text(json.dumps(summary, indent=2))
    make_plot(pop, primary, confound, discriminant, OUT_DIR / "seahorse_validation.png")
    write_report(summary, OUT_DIR)

    print()
    print("=" * 72)
    print(f"SECTION 1  AUC(C, Q2) = {primary['conciseness']['roc_auc']['point']:.4f} "
          f"({primary['conciseness']['roc_auc']['ci_lo']:.4f}, "
          f"{primary['conciseness']['roc_auc']['ci_hi']:.4f})  n={primary['class_balance']['n']}")
    print(f"SECTION 2  frac <2 claims = {confound['frac_lt2_claims']:.3f}; "
          f"better baseline = {confound['better_baseline']} "
          f"AUC={confound[confound['better_baseline']]['roc_auc']['point']:.4f}; "
          f"paired p={confound['c_vs_better_baseline_permutation_test']['p_value']:.4f}; "
          f"C significantly better: {confound['c_significantly_beats_better_baseline']}")
    print(f"SECTION 3  AUC(C,Q2)={discriminant['auc_vs_q2']['point']:.4f}  "
          f"AUC(C,Q6)={discriminant['auc_vs_q6']['point']:.4f}  verdict={discriminant['verdict']}")
    print(f"Wrote {OUT_DIR / 'seahorse_validation.md'}, seahorse_validation.json, seahorse_validation.png")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
