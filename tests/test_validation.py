"""Tests for the metric-validation harness (perturbation operators + stats).

Fast: no model downloads, no network. `perturbations.paraphrase` normally
tries an Ollama call; every test here monkeypatches `_llm_paraphrase` to
force the deterministic rule-based fallback, so nothing touches the network
or the on-disk paraphrase cache.
"""
import math
import re

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ragtrust.validation import perturbations as pert_mod
from ragtrust.validation.perturbations import (
    apply_all,
    duplication,
    entity_swap,
    negation,
    number_corruption,
    off_topic_padding,
    paraphrase,
    unsupported_addition,
)
from ragtrust.validation.stats import (
    bootstrap_ci,
    paired_permutation_test,
    pr_auc,
    roc_auc,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Perturbation operators
# ---------------------------------------------------------------------------


def test_entity_swap_flips_label_and_changes_text():
    answer = "Deep Learning improves robotics performance using Convolutional Neural Networks."
    passages = [
        "Reinforcement Learning is another technique used in Robotics.",
        "Markov Decision Process models are common in this field.",
    ]
    result = entity_swap(answer, passages, np.random.default_rng(0))
    assert result is not None
    assert result.operator == "entity_swap"
    assert result.hallucinated is True
    assert result.text != answer


def test_number_corruption_flips_label_and_changes_number():
    answer = "The system achieves 95 percent accuracy on the benchmark."
    result = number_corruption(answer, [], np.random.default_rng(0))
    assert result is not None
    assert result.operator == "number_corruption"
    assert result.hallucinated is True
    assert result.text != answer
    assert not re.search(r"\b95\b", result.text)


def test_negation_flips_label_and_inserts_negation():
    answer = "This method improves accuracy significantly."
    result = negation(answer, [], np.random.default_rng(0))
    assert result is not None
    assert result.operator == "negation"
    assert result.hallucinated is True
    assert "does not improve" in result.text
    assert result.text != answer


def test_unsupported_addition_flips_label_and_appends_text():
    answer = "The result was positive."
    result = unsupported_addition(answer, [], np.random.default_rng(0))
    assert result is not None
    assert result.operator == "unsupported_addition"
    assert result.hallucinated is True
    assert result.text.startswith("The result was positive.")
    assert len(result.text) > len(answer)


def test_duplication_preserves_label_and_repeats_a_sentence():
    answer = "First sentence here. Second sentence here."
    result = duplication(answer, [], np.random.default_rng(0))
    assert result is not None
    assert result.operator == "duplication"
    assert result.hallucinated is False
    assert result.text != answer
    assert result.text.count("sentence here") >= 3


def test_off_topic_padding_preserves_label_and_appends_verbatim_passage_text():
    answer = "The result was positive."
    passages = ["Robotics is a broad field with many applications. It includes many subfields."]
    result = off_topic_padding(answer, passages, np.random.default_rng(0))
    assert result is not None
    assert result.operator == "off_topic_padding"
    assert result.hallucinated is False
    assert "Robotics is a broad field with many applications." in result.text


def test_paraphrase_preserves_label_and_meaning_bearing_tokens(monkeypatch):
    # Force the deterministic rule-based fallback -- no network, no cache writes.
    monkeypatch.setattr(pert_mod, "_llm_paraphrase", lambda answer: None)
    answer = "The Sensor Fusion method uses 42 measurements to reduce variance."
    result = paraphrase(answer, ["dummy passage"], np.random.default_rng(0))
    assert result is not None
    assert result.operator == "paraphrase"
    assert result.hallucinated is False
    assert result.text != answer
    # Meaning-bearing tokens (the number, the named entity) must survive a
    # meaning-preserving rewrite even though surrounding wording changes.
    assert "42" in result.text
    assert "Sensor Fusion" in result.text


def test_apply_all_includes_unmodified_original_first():
    answer = "The method improves performance using 10 samples."
    passages = ["Some passage text with Robotics content."]
    perturbations = apply_all(answer, passages, seeds=(0,))
    assert perturbations[0].operator == "original"
    assert perturbations[0].hallucinated is False
    assert perturbations[0].text == answer
    assert len(perturbations) > 1


# ---------------------------------------------------------------------------
# stats.roc_auc
# ---------------------------------------------------------------------------


def _balanced_labels(rng, size):
    labels = rng.integers(0, 2, size=size)
    if labels.sum() == 0 or labels.sum() == size:
        labels[0], labels[1] = 0, 1
    return labels


def test_roc_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(42)
    labels = _balanced_labels(rng, 200)
    scores = rng.normal(size=200) + labels * 0.5
    expected = roc_auc_score(labels, scores)
    assert roc_auc(scores, labels) == pytest.approx(expected, abs=1e-9)


def test_roc_auc_matches_sklearn_on_tie_heavy_data():
    rng = np.random.default_rng(7)
    labels = _balanced_labels(rng, 100)
    scores = rng.integers(0, 5, size=100).astype(float)  # only 5 distinct values -> heavy ties
    expected = roc_auc_score(labels, scores)
    assert roc_auc(scores, labels) == pytest.approx(expected, abs=1e-9)


def test_roc_auc_perfect_separator_is_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_constant_scorer_is_half():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_single_class_returns_nan_without_raising():
    with pytest.warns(UserWarning):
        result = roc_auc([0.1, 0.2, 0.3], [0, 0, 0])
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# stats.roc_curve / pr_auc
# ---------------------------------------------------------------------------


def test_roc_curve_endpoints_and_monotonicity():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(scores, labels)
    assert fpr[0] == pytest.approx(0.0)
    assert tpr[0] == pytest.approx(0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= -1e-9)
    assert np.all(np.diff(tpr) >= -1e-9)


def test_roc_curve_single_class_returns_nan_without_raising():
    with pytest.warns(UserWarning):
        fpr, tpr, thr = roc_curve([0.1, 0.2], [1, 1])
    assert math.isnan(fpr[0])
    assert math.isnan(tpr[0])
    assert math.isnan(thr[0])


def test_pr_auc_perfect_separator_is_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    assert pr_auc(scores, labels) == pytest.approx(1.0)


def test_pr_auc_single_class_returns_nan_without_raising():
    with pytest.warns(UserWarning):
        result = pr_auc([0.1, 0.2, 0.3], [1, 1, 1])
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# stats.bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    labels = _balanced_labels(rng, 60)
    scores = rng.normal(size=60) + labels * 0.7
    point, lo, hi = bootstrap_ci(scores, labels, roc_auc, n=300, seed=0)
    assert lo <= point <= hi


def test_bootstrap_ci_single_class_returns_nan_without_raising():
    with pytest.warns(UserWarning):
        point, lo, hi = bootstrap_ci([0.1, 0.2], [1, 1], roc_auc, n=50, seed=0)
    assert math.isnan(point)
    assert math.isnan(lo)
    assert math.isnan(hi)


# ---------------------------------------------------------------------------
# stats.paired_permutation_test
# ---------------------------------------------------------------------------


def test_paired_permutation_test_identical_inputs_p_approx_one():
    rng = np.random.default_rng(2)
    labels = _balanced_labels(rng, 40)
    scores = rng.normal(size=40) + labels * 0.5
    stat_a, stat_b, diff, p = paired_permutation_test(scores, scores, labels, roc_auc, n=300, seed=0)
    assert diff == pytest.approx(0.0)
    assert p > 0.9


def test_paired_permutation_test_strong_difference_gives_small_p():
    rng = np.random.default_rng(3)
    labels = np.array([0] * 30 + [1] * 30)
    scores_good = np.concatenate([rng.normal(0, 0.1, 30), rng.normal(3, 0.1, 30)])  # near-perfect
    scores_bad = rng.normal(size=60)  # near chance
    stat_a, stat_b, diff, p = paired_permutation_test(
        scores_good, scores_bad, labels, roc_auc, n=500, seed=0
    )
    assert stat_a > stat_b
    assert p < 0.05


def test_paired_permutation_test_single_class_returns_nan_without_raising():
    with pytest.warns(UserWarning):
        result = paired_permutation_test([0.1, 0.2], [0.3, 0.4], [1, 1], roc_auc, n=50, seed=0)
    assert all(math.isnan(x) for x in result)
