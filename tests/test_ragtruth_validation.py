"""Tests for experiments/10_ragtruth_validation.py -- pure functions only.

Fast: no network, no model downloads. Only the label-derivation, stratified
sampling and single-label-type-filter logic is exercised here; everything
that needs the RAGTruth download or the NLI model lives behind `main()` /
`load_ragtruth()` / `score_items()`, none of which these tests call.

The module under test is a standalone script (like experiments/03, 08, etc.),
not a package, so it is loaded by file path rather than imported by name.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "10_ragtruth_validation.py"
_SPEC = importlib.util.spec_from_file_location("ragtruth_validation_module", _MODULE_PATH)
rv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rv
_SPEC.loader.exec_module(rv)


# ---------------------------------------------------------------------------
# derive_labels
# ---------------------------------------------------------------------------


def test_derive_labels_neither_type_present():
    out = rv.derive_labels({"evident_conflict": 0, "baseless_info": 0})
    assert out == {"has_conflict": False, "has_baseless": False, "hallucinated": False}


def test_derive_labels_conflict_only():
    out = rv.derive_labels({"evident_conflict": 2, "baseless_info": 0})
    assert out == {"has_conflict": True, "has_baseless": False, "hallucinated": True}


def test_derive_labels_baseless_only():
    out = rv.derive_labels({"evident_conflict": 0, "baseless_info": 3})
    assert out == {"has_conflict": False, "has_baseless": True, "hallucinated": True}


def test_derive_labels_both_types_present():
    out = rv.derive_labels({"evident_conflict": 1, "baseless_info": 1})
    assert out == {"has_conflict": True, "has_baseless": True, "hallucinated": True}


def test_derive_labels_missing_keys_default_to_zero():
    out = rv.derive_labels({})
    assert out == {"has_conflict": False, "has_baseless": False, "hallucinated": False}


# ---------------------------------------------------------------------------
# add_derived_labels
# ---------------------------------------------------------------------------


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "task_type": ["Summary", "Summary", "QA", "QA"],
            "hallucination_labels_processed": [
                {"evident_conflict": 0, "baseless_info": 0},
                {"evident_conflict": 1, "baseless_info": 0},
                {"evident_conflict": 0, "baseless_info": 2},
                {"evident_conflict": 1, "baseless_info": 1},
            ],
        }
    )


def test_add_derived_labels_adds_expected_columns():
    df = rv.add_derived_labels(_toy_df())
    assert df["hallucinated"].tolist() == [False, True, True, True]
    assert df["has_conflict"].tolist() == [False, True, False, True]
    assert df["has_baseless"].tolist() == [False, False, True, True]


def test_add_derived_labels_does_not_mutate_input():
    original = _toy_df()
    rv.add_derived_labels(original)
    assert "hallucinated" not in original.columns


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


def _bigger_df(n_per_group: int = 50) -> pd.DataFrame:
    rows = []
    i = 0
    for tt in ["Summary", "Data2txt", "QA"]:
        for _ in range(n_per_group):
            rows.append(
                {
                    "id": str(i),
                    "task_type": tt,
                    "hallucination_labels_processed": {
                        "evident_conflict": i % 2,
                        "baseless_info": (i + 1) % 2,
                    },
                }
            )
            i += 1
    return rv.add_derived_labels(pd.DataFrame(rows))


def test_stratified_sample_returns_exact_count_and_balanced_groups():
    df = _bigger_df()
    sample = rv.stratified_sample(df, n_items=30, seed=0)
    assert len(sample) == 30
    counts = sample["task_type"].value_counts()
    assert set(counts.index) == {"Summary", "Data2txt", "QA"}
    assert all(c == 10 for c in counts)


def test_stratified_sample_deterministic_for_fixed_seed():
    df = _bigger_df()
    a = rv.stratified_sample(df, n_items=30, seed=42)
    b = rv.stratified_sample(df, n_items=30, seed=42)
    assert sorted(a["id"].tolist()) == sorted(b["id"].tolist())


def test_stratified_sample_different_seeds_give_different_samples():
    df = _bigger_df()
    a = rv.stratified_sample(df, n_items=30, seed=1)
    b = rv.stratified_sample(df, n_items=30, seed=2)
    assert sorted(a["id"].tolist()) != sorted(b["id"].tolist())


def test_stratified_sample_handles_remainder_not_divisible_by_group_count():
    df = _bigger_df()
    sample = rv.stratified_sample(df, n_items=31, seed=0)
    assert len(sample) == 31


def test_stratified_sample_full_size_returns_whole_frame():
    df = _bigger_df(n_per_group=5)
    sample = rv.stratified_sample(df, n_items=len(df), seed=0)
    assert len(sample) == len(df)
    assert sorted(sample["id"].tolist()) == sorted(df["id"].tolist())


# ---------------------------------------------------------------------------
# filter_single_label_type
# ---------------------------------------------------------------------------


def test_filter_single_label_type_keeps_only_exactly_one_type_present():
    df = rv.add_derived_labels(_toy_df())
    filtered = rv.filter_single_label_type(df)
    assert filtered["id"].tolist() == ["2", "3"]


def test_filter_single_label_type_excludes_neither_and_both():
    df = rv.add_derived_labels(_toy_df())
    filtered = rv.filter_single_label_type(df)
    assert "1" not in filtered["id"].tolist()  # neither type
    assert "4" not in filtered["id"].tolist()  # both types


def test_filter_single_label_type_labels_are_mutually_exclusive_on_result():
    df = rv.add_derived_labels(_bigger_df())
    filtered = rv.filter_single_label_type(df)
    assert (filtered["has_conflict"] != filtered["has_baseless"]).all()


# ---------------------------------------------------------------------------
# get_claims / get_passages fallbacks (pure string logic, no model calls)
# ---------------------------------------------------------------------------


def test_get_claims_falls_back_to_whole_output_when_no_sentences_found():
    claims = rv.get_claims("   ")
    assert claims == []


def test_get_claims_splits_normal_text_into_sentences():
    claims = rv.get_claims("First sentence here. Second sentence here.")
    assert len(claims) >= 1


def test_get_passages_falls_back_to_empty_for_empty_context():
    assert rv.get_passages("") == []


def test_get_passages_returns_at_least_one_passage_for_short_context():
    passages = rv.get_passages("A short context string that is long enough to survive chunking thresholds.")
    assert len(passages) >= 1


# ---------------------------------------------------------------------------
# Experiment B: the complementary-label identity
# ---------------------------------------------------------------------------


def test_auc_against_complemented_label_is_exactly_one_minus_auc():
    """Regression test for a real defect in this experiment's first version.

    On the single-label-type subset every item is hallucinated and exactly one
    of has_conflict / has_baseless is True, so has_baseless IS not-has_conflict.
    ROC-AUC against a complemented label is exactly `1 - AUC`, which made the
    original four-row results table two independent numbers dressed up as four,
    and made its verdict -- "kappa wins on conflict AND (1-F) wins on baseless"
    -- the conjunction `A and A`.

    If this identity ever fails to hold, the complement is no longer redundant
    and Experiment B's single-target design should be revisited.
    """
    import numpy as np

    from ragtrust.validation.stats import roc_auc

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        n = int(rng.integers(10, 60))
        scores = rng.random(n)
        has_conflict = rng.integers(0, 2, size=n)
        if len(np.unique(has_conflict)) < 2:
            continue
        has_baseless = 1 - has_conflict
        assert roc_auc(scores, has_baseless) == pytest.approx(
            1.0 - roc_auc(scores, has_conflict), abs=1e-12)
        checked += 1
    assert checked > 0, "no non-degenerate label vector was generated"


def test_experiment_b_reports_only_the_has_conflict_target():
    """The complement carries no information, so it must not be tabulated.

    Guards against someone re-adding the has_baseless rows and re-presenting
    one fact as two.
    """
    import inspect

    source = inspect.getsource(rv.run_experiment_b)
    assert "kappa_vs_has_conflict" in source
    assert "one_minus_F_vs_has_conflict" in source
    assert "kappa_vs_has_baseless" not in source
    assert "one_minus_F_vs_has_baseless" not in source


def test_experiment_b_verdict_requires_significance_for_the_strong_claim():
    """PARTIALLY SUPPORTED exists precisely so a non-significant gap cannot be
    reported as SUPPORTED. The first version had no such distinction: it
    declared support from point estimates alone, with no paired test at all."""
    import inspect

    source = inspect.getsource(rv.run_experiment_b)
    assert "paired_permutation_test" in source
    assert "PARTIALLY SUPPORTED" in source
    # The strong verdict must depend on BOTH conditions, never on one.
    assert "kappa_beats_chance and gap_significant" in source
