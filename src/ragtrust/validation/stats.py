"""Discrimination statistics for metric validation -- METRICS.md Part III.

Pure numpy/scipy. `scikit-learn` is only used in `tests/test_validation.py` to
cross-check `roc_auc` against `sklearn.metrics.roc_auc_score`, never here.

Every function here returns ``float("nan")`` (or an all-nan tuple, for the
curve/test functions that return more than one value) together with a
``warnings.warn`` when handed degenerate single-class input, rather than
raising -- a validation run over a small, real dataset should degrade
gracefully, not crash, when a particular perturbation slice happens to
contain only one label.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata


def _as_arrays(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    return scores, labels


def _is_degenerate(labels) -> bool:
    return len(np.unique(labels)) < 2


def roc_auc(scores, labels) -> float:
    """ROC-AUC via the Mann-Whitney U / rank-sum formulation:

        AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    where R_pos is the sum of the *average* ranks (ties split evenly) of the
    positive-class scores. This handles tied scores correctly, unlike a
    naive threshold sweep, and is equivalent to sklearn's
    ``roc_auc_score`` for the binary case.
    """
    scores, labels = _as_arrays(scores, labels)
    if _is_degenerate(labels):
        warnings.warn("roc_auc: only one class present in labels; returning nan")
        return float("nan")

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    ranks = rankdata(scores)  # average rank on ties
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve(scores, labels):
    """(fpr, tpr, thresholds), grouping tied scores onto a single threshold
    step (as sklearn.metrics.roc_curve does), so the curve is well-defined
    even with heavy ties."""
    scores, labels = _as_arrays(scores, labels)
    if _is_degenerate(labels):
        warnings.warn("roc_curve: only one class present in labels; returning nan")
        nan = np.array([float("nan")])
        return nan, nan, nan

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    distinct_idx = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, scores_sorted.size - 1]

    tps = np.cumsum(labels_sorted == 1)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tpr = tps / n_pos
    fpr = fps / n_neg
    thresholds = scores_sorted[threshold_idxs]

    # Prepend the (0, 0) point at an implicit threshold above the max score.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]

    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def pr_auc(scores, labels) -> float:
    """Average precision (area under the precision-recall step function),
    using the same grouped-threshold construction as `roc_curve` so tied
    scores are handled consistently."""
    scores, labels = _as_arrays(scores, labels)
    n_pos = int(np.sum(labels == 1))
    n_total = labels.size
    if n_pos == 0 or n_pos == n_total:
        warnings.warn("pr_auc: only one class present in labels; returning nan")
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    distinct_idx = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, scores_sorted.size - 1]

    tps = np.cumsum(labels_sorted == 1)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    precision = tps / (tps + fps)
    recall = tps / n_pos

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def bootstrap_ci(scores, labels, stat_fn, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap confidence interval for `stat_fn(scores, labels)`, stratified
    by label: each resample draws (with replacement) `n_pos` indices from the
    positive items and `n_neg` from the negative items independently, so a
    resample can never lose a class -- unlike a plain resample of the whole
    dataset, which occasionally would for a small or imbalanced set.

    Returns (point, lo, hi).
    """
    scores, labels = _as_arrays(scores, labels)
    if _is_degenerate(labels):
        warnings.warn("bootstrap_ci: only one class present in labels; returning nan")
        return float("nan"), float("nan"), float("nan")

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = pos_idx.size
    n_neg = neg_idx.size

    point = float(stat_fn(scores, labels))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n, dtype=float)
    for i in range(n):
        bp = rng.choice(pos_idx, size=n_pos, replace=True)
        bn = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([bp, bn])
        boot_stats[i] = stat_fn(scores[idx], labels[idx])

    lo = float(np.nanpercentile(boot_stats, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_permutation_test(scores_a, scores_b, labels, stat_fn, n: int = 10000, seed: int = 0):
    """Paired permutation test comparing two scorers evaluated on the *same*
    items. H0: the two scorers are exchangeable given the labels -- under H0,
    swapping which of scores_a/scores_b is "labelled A" for a given item does
    not change the distribution of stat_fn(A, labels) - stat_fn(B, labels).

    Each of the `n` permutations independently swaps scores_a[i] <-> scores_b[i]
    for each item i with probability 0.5, recomputes the statistic on each
    side, and takes the difference. The two-sided p-value is the (add-one
    smoothed) fraction of permuted |difference| at least as large as the
    observed one.

    Returns (stat_a, stat_b, diff, p_value) where diff = stat_a - stat_b on
    the *unpermuted* data.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels = np.asarray(labels)

    if _is_degenerate(labels):
        warnings.warn("paired_permutation_test: only one class present in labels; returning nan")
        nan = float("nan")
        return nan, nan, nan, nan

    stat_a = float(stat_fn(scores_a, labels))
    stat_b = float(stat_fn(scores_b, labels))
    observed_diff = stat_a - stat_b

    rng = np.random.default_rng(seed)
    n_items = labels.size
    perm_diffs = np.empty(n, dtype=float)
    for i in range(n):
        swap = rng.random(n_items) < 0.5
        perm_a = np.where(swap, scores_b, scores_a)
        perm_b = np.where(swap, scores_a, scores_b)
        perm_diffs[i] = stat_fn(perm_a, labels) - stat_fn(perm_b, labels)

    # Add-one smoothed two-sided p-value: avoids a p-value of exactly 0 and
    # gives p == 1.0 only when every permutation is at least as extreme,
    # which is what happens when scores_a and scores_b are identical (every
    # permuted difference is exactly 0, same as the observed difference).
    n_extreme = int(np.sum(np.abs(perm_diffs) >= abs(observed_diff) - 1e-12))
    p_value = float((n_extreme + 1) / (n + 1))

    return stat_a, stat_b, float(observed_diff), p_value
