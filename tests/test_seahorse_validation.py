"""Tests for experiments/12_seahorse_validation.py -- pure functions only.

Fast: no network, no model downloads. Only pivot/grouping, Yes/No->binary mapping,
the direction convention (higher C predicts Q2="Yes"), the <2-claims->C==1.0
boundary, and the length-baseline / routing helpers are exercised here. Anything
needing the SEAHORSE download or the SentenceTransformer embedder lives behind
`main()` / `load_seahorse()` / `score_conciseness()`, none of which these tests call.

The module under test is a standalone script (like experiments/10, 11), not a
package, so it is loaded by file path rather than imported by name -- same
pattern as tests/test_ragtruth_validation.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FakeEmbedder

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "12_seahorse_validation.py"
_SPEC = importlib.util.spec_from_file_location("seahorse_validation_module", _MODULE_PATH)
sv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sv
_SPEC.loader.exec_module(sv)

sys.path.insert(0, str(_ROOT / "src"))
from ragtrust.metrics.conciseness import conciseness  # noqa: E402


# ---------------------------------------------------------------------------
# yes_no_to_binary
# ---------------------------------------------------------------------------


def test_yes_no_to_binary_yes_is_one():
    assert sv.yes_no_to_binary("Yes") == 1.0


def test_yes_no_to_binary_no_is_zero():
    assert sv.yes_no_to_binary("No") == 0.0


def test_yes_no_to_binary_unexpected_value_is_nan():
    assert np.isnan(sv.yes_no_to_binary("Maybe"))
    assert np.isnan(sv.yes_no_to_binary(None))


# ---------------------------------------------------------------------------
# filter_en_us
# ---------------------------------------------------------------------------


def test_filter_en_us_keeps_only_en_us_rows():
    df = pd.DataFrame({
        "worker_lang": ["en-US", "de", "en-US", "ru"],
        "value": [1, 2, 3, 4],
    })
    out = sv.filter_en_us(df)
    assert out["value"].tolist() == [1, 3]


# ---------------------------------------------------------------------------
# pivot_summaries -- the long-to-wide pivot/grouping logic
# ---------------------------------------------------------------------------


def _toy_long_df() -> pd.DataFrame:
    """Two distinct summaries (by gem_id+model+summary). Summary 1 has both Q2
    and Q6 answered; summary 2 has only Q2 answered -- mirrors the real dataset,
    where not every summary has every question answered."""
    q2 = sv.QUESTIONS["q2_repetition"]
    q6 = sv.QUESTIONS["q6_concise"]
    return pd.DataFrame({
        "gem_id": ["g1", "g1", "g2"],
        "model": ["m1", "m1", "m1"],
        "summary": ["Summary one.", "Summary one.", "Summary two."],
        "question": [q2, q6, q2],
        "answer": ["Yes", "No", "No"],
    })


def test_pivot_summaries_produces_one_row_per_distinct_summary():
    wide = sv.pivot_summaries(_toy_long_df())
    assert len(wide) == 2
    assert set(wide["gem_id"]) == {"g1", "g2"}


def test_pivot_summaries_places_answers_in_correct_question_columns():
    wide = sv.pivot_summaries(_toy_long_df())
    row_g1 = wide[wide["gem_id"] == "g1"].iloc[0]
    assert row_g1["q2_repetition"] == 1.0
    assert row_g1["q6_concise"] == 0.0


def test_pivot_summaries_missing_question_is_nan_not_dropped():
    wide = sv.pivot_summaries(_toy_long_df())
    row_g2 = wide[wide["gem_id"] == "g2"].iloc[0]
    assert row_g2["q2_repetition"] == 0.0
    assert pd.isna(row_g2["q6_concise"])


def test_pivot_summaries_adds_all_question_columns_even_if_unanswered():
    wide = sv.pivot_summaries(_toy_long_df())
    for short in sv.QUESTIONS:
        assert short in wide.columns


def test_pivot_summaries_drops_and_warns_on_unrecognised_question():
    df = _toy_long_df()
    df = pd.concat([df, pd.DataFrame({
        "gem_id": ["g3"], "model": ["m1"], "summary": ["Summary three."],
        "question": ["Some other question entirely."], "answer": ["Yes"],
    })], ignore_index=True)
    with pytest.warns(UserWarning, match="not in QUESTIONS"):
        wide = sv.pivot_summaries(df)
    assert "g3" not in set(wide["gem_id"])


# ---------------------------------------------------------------------------
# get_claims / word_count
# ---------------------------------------------------------------------------


def test_get_claims_falls_back_to_empty_for_blank_summary():
    assert sv.get_claims("   ") == []


def test_get_claims_splits_normal_text_into_sentences():
    claims = sv.get_claims("First sentence here. Second sentence here.")
    assert len(claims) >= 1


def test_word_count_counts_whitespace_separated_tokens():
    assert sv.word_count("one two three") == 3
    assert sv.word_count("") == 0
    assert sv.word_count(None) == 0


# ---------------------------------------------------------------------------
# The <2-claims -> C is undefined (None) boundary (direct on the real
# conciseness metric). Pairwise self-similarity has no meaning with fewer
# than 2 claims, so conciseness returns None rather than the old 1.0
# sentinel -- see metrics/conciseness.py module docstring for why that
# sentinel was a defect (it handed a perfect score to 68.5% of SEAHORSE
# summaries). `score_conciseness` in experiments/12_seahorse_validation.py
# substitutes 1.0 back in locally -- deliberately, to reproduce the old
# sentinel for the length-confound study -- which is why the full-population
# headline numbers in this file's other tests are unaffected by this change.
# ---------------------------------------------------------------------------


def test_conciseness_is_none_for_zero_claims():
    embedder = FakeEmbedder()
    assert conciseness([], embedder) is None


def test_conciseness_is_none_for_a_single_claim():
    embedder = FakeEmbedder()
    assert conciseness(["Only one claim here."], embedder) is None


def test_conciseness_is_not_trivially_one_for_two_identical_claims():
    """Two *redundant* (identical) claims must NOT score 1.0 -- this is the
    metric detecting the padding the <2-claims short-circuit cannot see."""
    embedder = FakeEmbedder()
    c = conciseness(["The exact same sentence.", "The exact same sentence."], embedder)
    assert c == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The direction convention: higher C must predict Q2 == "Yes" (not "No")
# ---------------------------------------------------------------------------


def test_run_primary_analysis_direction_convention_higher_c_predicts_yes():
    """Regression guard for the metric's orientation: Q2='Yes' means NOT
    redundant, so a redundancy metric must be scored UN-negated (higher C ->
    "Yes"). If run_primary_analysis ever negated C before scoring, this would
    flip to AUC == 0.0 instead of 1.0."""
    pop = pd.DataFrame({
        "q2_repetition": [1, 1, 0, 0],
        "C": [0.9, 0.8, 0.2, 0.1],
    })
    result = sv.run_primary_analysis(pop, n_boot=20, seed=0)
    assert result["conciseness"]["roc_auc"]["point"] == pytest.approx(1.0)


def test_run_primary_analysis_class_balance_reported_correctly():
    pop = pd.DataFrame({
        "q2_repetition": [1, 1, 1, 0],
        "C": [0.9, 0.8, 0.7, 0.5],
    })
    result = sv.run_primary_analysis(pop, n_boot=20, seed=0)
    cb = result["class_balance"]
    assert cb == {"n": 4, "n_yes": 3, "n_no": 1, "rate_yes": pytest.approx(0.75)}


# ---------------------------------------------------------------------------
# best_sign -- the length-baseline direction helper
# ---------------------------------------------------------------------------


def test_best_sign_picks_positive_when_higher_stat_predicts_positive_label():
    labels = np.array([0, 0, 0, 1, 1, 1])
    stat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # perfectly increasing with label
    assert sv.best_sign(stat, labels) == 1


def test_best_sign_picks_negative_when_lower_stat_predicts_positive_label():
    labels = np.array([0, 0, 0, 1, 1, 1])
    stat = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # perfectly decreasing with label
    assert sv.best_sign(stat, labels) == -1


def test_best_sign_flips_the_auc_via_the_stated_identity():
    """Sanity check of the identity best_sign relies on: roc_auc(-s,y) == 1-roc_auc(s,y)."""
    from ragtrust.validation.stats import roc_auc

    rng = np.random.default_rng(0)
    stat = rng.random(30)
    labels = (rng.random(30) < 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        pytest.skip("degenerate label draw")
    assert roc_auc(-stat, labels) == pytest.approx(1.0 - roc_auc(stat, labels), abs=1e-9)


# ---------------------------------------------------------------------------
# maybe_sample
# ---------------------------------------------------------------------------


def test_maybe_sample_returns_whole_frame_when_under_cap():
    df = pd.DataFrame({"x": range(10)})
    out = sv.maybe_sample(df, max_items=20, seed=0)
    assert len(out) == 10


def test_maybe_sample_samples_down_deterministically_when_over_cap():
    df = pd.DataFrame({"x": range(100)})
    a = sv.maybe_sample(df, max_items=10, seed=42)
    b = sv.maybe_sample(df, max_items=10, seed=42)
    assert len(a) == 10
    assert sorted(a["x"].tolist()) == sorted(b["x"].tolist())


# ---------------------------------------------------------------------------
# class_balance
# ---------------------------------------------------------------------------


def test_class_balance_counts_and_rate():
    labels = np.array([1, 1, 0, 0, 0])
    assert sv.class_balance(labels) == {"n": 5, "n_yes": 2, "n_no": 3, "rate_yes": pytest.approx(0.4)}


# ---------------------------------------------------------------------------
# paired_permutation_test_swap_labels -- the discriminant-validity test builder
# ---------------------------------------------------------------------------


def test_paired_permutation_test_swap_labels_identical_labels_give_zero_diff_and_p_one():
    from ragtrust.validation.stats import roc_auc

    rng = np.random.default_rng(0)
    score = rng.random(20)
    labels = (rng.random(20) < 0.5).astype(int)
    stat_a, stat_b, diff, p = sv.paired_permutation_test_swap_labels(
        score, labels, labels, roc_auc, n=200, seed=0
    )
    assert stat_a == pytest.approx(stat_b)
    assert diff == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_paired_permutation_test_swap_labels_detects_a_real_gap():
    """score tracks labels_a perfectly and labels_b not at all -> large positive
    diff and a small p-value."""
    from ragtrust.validation.stats import roc_auc

    n = 40
    labels_a = np.array([0] * (n // 2) + [1] * (n // 2))
    score = labels_a.astype(float)  # perfectly separates labels_a
    rng = np.random.default_rng(1)
    labels_b = rng.integers(0, 2, size=n)
    while len(np.unique(labels_b)) < 2:
        labels_b = rng.integers(0, 2, size=n)

    stat_a, stat_b, diff, p = sv.paired_permutation_test_swap_labels(
        score, labels_a, labels_b, roc_auc, n=500, seed=0
    )
    assert stat_a == pytest.approx(1.0)
    assert diff > 0
    assert p < 0.05


# ---------------------------------------------------------------------------
# run_length_confound -- the length-baseline routing and headline logic
# ---------------------------------------------------------------------------


def _confound_pop(n_per_class: int, c_values, n_claims_values, n_words_values) -> pd.DataFrame:
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({
        "q2_repetition": labels,
        "C": c_values,
        "n_claims": n_claims_values,
        "n_words": n_words_values,
    })


def test_run_length_confound_reports_correct_fraction_below_two_claims():
    n = 10
    pop = _confound_pop(
        n_per_class=n // 2,
        c_values=list(np.linspace(0.1, 0.9, n)),
        n_claims_values=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2],  # 6 of 10 below 2
        n_words_values=list(range(5, 15)),
    )
    result = sv.run_length_confound(pop, n_boot=20, n_perm=20, seed=0)
    assert result["frac_lt2_claims"] == pytest.approx(0.6)


def test_run_length_confound_headline_when_baseline_beats_c():
    """C is uninformative (constant); n_claims perfectly predicts Q2. The
    headline must say the baseline matches/beats C, and the significance flag
    must be False -- this is the 'conciseness adds nothing' case the experiment
    exists to be able to report honestly."""
    n = 40
    labels = [0] * (n // 2) + [1] * (n // 2)
    pop = pd.DataFrame({
        "q2_repetition": labels,
        "C": [0.5] * n,               # uninformative
        "n_claims": [float(x) for x in labels],  # perfect predictor
        "n_words": [3.0] * n,          # also uninformative
    })
    result = sv.run_length_confound(pop, n_boot=200, n_perm=500, seed=0)
    assert result["better_baseline"] == "n_claims"
    assert result["n_claims"]["roc_auc"]["point"] == pytest.approx(1.0)
    assert result["c_significantly_beats_better_baseline"] is False
    assert "matches or beats" in result["headline"] or "adds nothing" in result["headline"]


def test_run_length_confound_headline_when_c_beats_baseline():
    """C perfectly predicts Q2; both length baselines are uninformative. The
    headline must credit C, and the significance flag must be True."""
    n = 40
    labels = [0] * (n // 2) + [1] * (n // 2)
    pop = pd.DataFrame({
        "q2_repetition": labels,
        "C": [float(x) for x in labels],  # perfect predictor
        "n_claims": [3.0] * n,             # uninformative (constant)
        "n_words": [10.0] * n,             # uninformative (constant)
    })
    result = sv.run_length_confound(pop, n_boot=200, n_perm=500, seed=0)
    assert result["c_significantly_beats_better_baseline"] is True
    assert "beats the stronger length baseline" in result["headline"]


def test_run_length_confound_restricted_subset_excludes_short_summaries():
    n = 10
    # labels: first 5 items label 0, last 5 label 1. n_claims mixes >=2 into BOTH
    # classes (indices 2-4 of label 0, indices 7-9 of label 1) so the restricted
    # subset keeps both classes and isn't degenerate.
    pop = _confound_pop(
        n_per_class=n // 2,
        c_values=list(np.linspace(0.1, 0.9, n)),
        n_claims_values=[1, 1, 2, 2, 2, 1, 1, 2, 2, 2],  # 4 of 10 excluded (indices 0,1,5,6)
        n_words_values=list(range(5, 15)),
    )
    result = sv.run_length_confound(pop, n_boot=20, n_perm=20, seed=0)
    restricted = result["restricted_ge2_claims"]
    assert restricted is not None
    assert restricted["class_balance"]["n"] == 6


def test_run_length_confound_restricted_subset_none_when_single_class_remains():
    n = 10
    pop = _confound_pop(
        n_per_class=n // 2,
        c_values=list(np.linspace(0.1, 0.9, n)),
        n_claims_values=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        n_words_values=list(range(5, 15)),
    )
    # Only items with n_claims >= 2 (the last 5 rows) survive the restriction; force
    # them all to share one label so the restricted subset is single-class.
    pop["q2_repetition"] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    with pytest.warns(UserWarning, match="single class"):
        result = sv.run_length_confound(pop, n_boot=20, n_perm=20, seed=0)
    assert result["restricted_ge2_claims"] is None


# ---------------------------------------------------------------------------
# run_discriminant_validity -- Q2 vs Q6 verdict logic
# ---------------------------------------------------------------------------


def _discriminant_pop(c_values, q2_values, q6_values) -> pd.DataFrame:
    return pd.DataFrame({
        "C": c_values,
        "q2_repetition": q2_values,
        "q6_concise": q6_values,
    })


def test_run_discriminant_validity_supported_when_c_tracks_q2_more_than_q6():
    n = 40
    q2 = [0] * (n // 2) + [1] * (n // 2)  # C perfectly predicts q2
    rng = np.random.default_rng(2)
    q6 = list(rng.integers(0, 2, size=n))  # unrelated to C
    while len(set(q6)) < 2:
        q6 = list(rng.integers(0, 2, size=n))
    pop = _discriminant_pop(c_values=[float(x) for x in q2], q2_values=q2, q6_values=q6)
    result = sv.run_discriminant_validity(pop, n_boot=200, n_perm=500, seed=0)
    assert result["auc_vs_q2"]["point"] == pytest.approx(1.0)
    assert result["q2_tracks_more_than_q6"] is True
    assert result["verdict"] in ("SUPPORTED", "WEAKLY SUPPORTED")


def test_run_discriminant_validity_not_supported_when_c_tracks_q6_more_than_q2():
    n = 40
    q6 = [0] * (n // 2) + [1] * (n // 2)  # C perfectly predicts q6 instead
    rng = np.random.default_rng(3)
    q2 = list(rng.integers(0, 2, size=n))  # unrelated to C
    while len(set(q2)) < 2:
        q2 = list(rng.integers(0, 2, size=n))
    pop = _discriminant_pop(c_values=[float(x) for x in q6], q2_values=q2, q6_values=q6)
    result = sv.run_discriminant_validity(pop, n_boot=200, n_perm=500, seed=0)
    assert result["auc_vs_q6"]["point"] == pytest.approx(1.0)
    assert result["q2_tracks_more_than_q6"] is False
    assert result["verdict"] == "NOT SUPPORTED"


def test_run_discriminant_validity_scores_q2_and_q6_independently():
    """Guards against re-introducing the bug experiment 10 already hit once
    (Experiment B): Q2 and Q6 must each be scored directly against C, not one
    derived from the other as a complement -- they are independent judgments,
    not complementary labels, so both being present in the source is expected
    and correct here (contrast with experiment 10, where only ONE target was
    allowed to appear)."""
    import inspect

    source = inspect.getsource(sv.run_discriminant_validity)
    assert "labels_q2 = paired_df[Q2_COL]" in source
    assert "labels_q6 = paired_df[Q6_COL]" in source
