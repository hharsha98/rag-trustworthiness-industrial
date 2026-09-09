"""Tests for experiments/13_relevance_validation.py -- pure functions only.

Fast: no network, no model downloads. Covers the qrel->binary mapping, the
negative-sampling helpers (determinism under seed, and that sampled negatives
never include a judged-relevant document for that query), the floor
computation on both mappings, and the monotone-invariance fact ROC-AUC relies
on (so nobody later "fixes" the floor analysis into a meaningless AUC
comparison -- see the module's Trap 2 docstring). Anything needing the real
BEIR download or the real embedder's sentence-transformers plumbing lives
behind `main()` / `load_nfcorpus()` / `process_dataset()`, none of which
these tests call.

The module under test is a standalone script (like experiments/08-12), not a
package, so it is loaded by file path rather than imported by name -- same
pattern as tests/test_beir_loader.py and tests/test_gate_calibration.py.
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pytest

from conftest import FakeEmbedder

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "13_relevance_validation.py"
_SPEC = importlib.util.spec_from_file_location("relevance_validation_module", _MODULE_PATH)
rv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rv
_SPEC.loader.exec_module(rv)

sys.path.insert(0, str(_ROOT / "src"))
from ragtrust.metrics.relevance import context_relevance  # noqa: E402
from ragtrust.validation.stats import roc_auc  # noqa: E402


# ---------------------------------------------------------------------------
# positives_from_qrels -- the qrel -> binary-relevance mapping
# ---------------------------------------------------------------------------


def test_positives_from_qrels_groups_by_query_and_stringifies_ids():
    df = pd.DataFrame({
        "query-id": [1, 1, 3],
        "corpus-id": [100, 200, 300],
        "score": [1, 1, 1],
    })
    lookup = rv.positives_from_qrels(df)
    assert lookup == {"1": ["100", "200"], "3": ["300"]}


def test_positives_from_qrels_drops_zero_and_negative_scores():
    """qrel score > 0 is the relevance rule -- a score of 0 (or negative,
    were one ever present) must NOT count as a positive."""
    df = pd.DataFrame({
        "query-id": [1, 1, 2],
        "corpus-id": [10, 20, 30],
        "score": [1, 0, -1],
    })
    lookup = rv.positives_from_qrels(df)
    assert lookup == {"1": ["10"]}
    assert "2" not in lookup


def test_positives_from_qrels_treats_graded_scores_as_relevant():
    """NFCorpus qrels are graded {1, 2} -- both grades count as relevant
    under the score > 0 rule, not just score == 1."""
    df = pd.DataFrame({
        "query-id": ["q1", "q1"],
        "corpus-id": ["a", "b"],
        "score": [1, 2],
    })
    lookup = rv.positives_from_qrels(df)
    assert sorted(lookup["q1"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# ids_to_indices
# ---------------------------------------------------------------------------


def test_ids_to_indices_maps_string_ids_to_corpus_positions():
    id_to_index = {"a": 0, "b": 1, "c": 2}
    lookup = {"q1": ["b", "c"], "q2": ["a"]}
    out = rv.ids_to_indices(lookup, id_to_index)
    assert out == {"q1": [1, 2], "q2": [0]}


def test_ids_to_indices_drops_unknown_ids():
    id_to_index = {"a": 0}
    lookup = {"q1": ["a", "does-not-exist"]}
    out = rv.ids_to_indices(lookup, id_to_index)
    assert out == {"q1": [0]}


def test_ids_to_indices_drops_query_with_no_surviving_positives():
    id_to_index = {"a": 0}
    lookup = {"q1": ["missing-only"]}
    out = rv.ids_to_indices(lookup, id_to_index)
    assert "q1" not in out


# ---------------------------------------------------------------------------
# query_seed -- deterministic, query-specific seeding
# ---------------------------------------------------------------------------


def test_query_seed_is_deterministic():
    a = rv.query_seed(0, "q1", "random")
    b = rv.query_seed(0, "q1", "random")
    assert a == b


def test_query_seed_differs_by_query_id():
    a = rv.query_seed(0, "q1", "random")
    b = rv.query_seed(0, "q2", "random")
    assert a != b


def test_query_seed_differs_by_salt():
    a = rv.query_seed(0, "q1", "random")
    b = rv.query_seed(0, "q1", "hard")
    assert a != b


# ---------------------------------------------------------------------------
# sample_random_negatives -- determinism + exclusion (Trap 1, "easy" tier)
# ---------------------------------------------------------------------------


def test_sample_random_negatives_is_deterministic_under_seed():
    a = rv.sample_random_negatives(1000, exclude_idx=set(), n=20, seed=42)
    b = rv.sample_random_negatives(1000, exclude_idx=set(), n=20, seed=42)
    assert a == b


def test_sample_random_negatives_differs_across_seeds_generally():
    a = rv.sample_random_negatives(1000, exclude_idx=set(), n=20, seed=1)
    b = rv.sample_random_negatives(1000, exclude_idx=set(), n=20, seed=2)
    assert a != b


def test_sample_random_negatives_never_includes_excluded_ids():
    exclude = set(range(0, 500))  # exclude the first half of the corpus
    sample = rv.sample_random_negatives(1000, exclude_idx=exclude, n=100, seed=0)
    assert len(sample) == 100
    assert not (set(sample) & exclude)


def test_sample_random_negatives_caps_at_available_pool_size():
    exclude = set(range(0, 8))
    sample = rv.sample_random_negatives(10, exclude_idx=exclude, n=50, seed=0)
    assert sorted(sample) == [8, 9]


# ---------------------------------------------------------------------------
# select_hard_negatives -- rank-order preserved + exclusion (Trap 1, "hard" tier)
# ---------------------------------------------------------------------------


def test_select_hard_negatives_takes_top_n_excluding_relevant():
    ranked = [5, 2, 7, 1, 9, 3]
    out = rv.select_hard_negatives(ranked, exclude_idx={2, 9}, n=3)
    assert out == [5, 7, 1]  # 2 skipped (excluded); 9 would be 4th but n=3 stops first


def test_select_hard_negatives_never_includes_excluded_ids():
    ranked = list(range(50))
    exclude = set(range(0, 25))
    out = rv.select_hard_negatives(ranked, exclude_idx=exclude, n=10)
    assert len(out) == 10
    assert not (set(out) & exclude)
    assert out == list(range(25, 35))  # rank order preserved


def test_select_hard_negatives_returns_fewer_if_pool_exhausted():
    ranked = [1, 2, 3]
    out = rv.select_hard_negatives(ranked, exclude_idx={1, 2, 3}, n=5)
    assert out == []


# ---------------------------------------------------------------------------
# assemble_pairs_for_query -- composition of positive/random/hard rows
# ---------------------------------------------------------------------------


def test_assemble_pairs_for_query_has_expected_tiers_and_labels():
    rows = rv.assemble_pairs_for_query(
        query_id="q1", positive_idx=[3, 4], n_corpus=100,
        bm25_ranked_idx=[10, 11, 12, 13, 14, 15], n_random=2, n_hard=2, seed=0,
    )
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)

    assert len(by_tier["positive"]) == 2
    assert all(r["label"] == 1 for r in by_tier["positive"])
    assert {r["passage_idx"] for r in by_tier["positive"]} == {3, 4}

    assert len(by_tier["random"]) == 2
    assert all(r["label"] == 0 for r in by_tier["random"])
    assert {r["passage_idx"] for r in by_tier["random"]} & {3, 4} == set()

    assert len(by_tier["hard"]) == 2
    assert all(r["label"] == 0 for r in by_tier["hard"])
    assert {r["passage_idx"] for r in by_tier["hard"]} == {10, 11}  # first two ranked, none excluded


def test_assemble_pairs_for_query_hard_tier_excludes_positives_from_bm25_ranking():
    rows = rv.assemble_pairs_for_query(
        query_id="q1", positive_idx=[10], n_corpus=100,
        bm25_ranked_idx=[10, 11, 12], n_random=0, n_hard=2, seed=0,
    )
    hard_idxs = {r["passage_idx"] for r in rows if r["tier"] == "hard"}
    assert 10 not in hard_idxs
    assert hard_idxs == {11, 12}


def test_assemble_pairs_for_query_is_deterministic_across_calls():
    kwargs = dict(query_id="q7", positive_idx=[1], n_corpus=500,
                  bm25_ranked_idx=list(range(50, 60)), n_random=5, n_hard=3, seed=0)
    a = rv.assemble_pairs_for_query(**kwargs)
    b = rv.assemble_pairs_for_query(**kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# clamp_scores / affine_scores / score_distribution / floor_analysis (Trap 2)
# ---------------------------------------------------------------------------


def test_clamp_scores_zeroes_negatives_and_preserves_nonnegatives():
    raw = np.array([-0.5, -0.1, 0.0, 0.2, 0.9])
    out = rv.clamp_scores(raw)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.2, 0.9])


def test_affine_scores_maps_minus_one_to_zero_and_one_to_one():
    raw = np.array([-1.0, 0.0, 1.0])
    out = rv.affine_scores(raw)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_affine_scores_never_below_half_for_typical_nonnegative_cosine():
    """The floor claim itself: for any t >= 0 (the realistic regime), (1+t)/2 >= 0.5."""
    raw = np.array([0.0, 0.01, 0.3, 0.99])
    out = rv.affine_scores(raw)
    assert np.all(out >= 0.5)


def test_score_distribution_hand_computed():
    values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    d = rv.score_distribution(values)
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(0.2)
    assert d["median"] == pytest.approx(0.2)
    assert d["min"] == pytest.approx(0.0)
    assert d["max"] == pytest.approx(0.4)


def test_floor_analysis_shows_clamp_near_zero_and_affine_near_half_on_irrelevant():
    """The central calibration claim under test: on irrelevant passages with
    small-magnitude raw cosine (the realistic case for real encoders), the
    clamp mapping sits near 0 while the affine mapping sits near 0.5 -- the
    floor `relevance.py`'s docstring documents as the defect that was fixed."""
    raw_irrelevant = np.array([-0.02, 0.01, -0.05, 0.03, 0.00, -0.01])
    result = rv.floor_analysis(raw_irrelevant)
    assert result["clamp"]["mean"] < 0.05
    assert result["affine"]["mean"] == pytest.approx(0.5, abs=0.05)
    assert result["clamp"]["min"] == pytest.approx(0.0)  # clamp never goes negative
    assert result["affine"]["min"] < 0.5  # affine still reflects the (small) negative tail


def test_frac_negative_cosine_hand_computed():
    raw = np.array([0.1, -0.1, 0.2, -0.3, 0.0])
    assert rv.frac_negative_cosine(raw) == pytest.approx(0.4)  # 2 of 5 negative


def test_frac_negative_cosine_is_zero_when_all_nonnegative():
    raw = np.array([0.0, 0.1, 0.5, 0.9])
    assert rv.frac_negative_cosine(raw) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The monotone-invariance fact (Trap 2's core guard): ROC-AUC is IDENTICAL
# for max(0,t) and (1+t)/2 on the same scores, over the range that actually
# occurs (t >= 0, i.e. no clipping-induced ties -- see module docstring).
# This is the fact that makes an AUC comparison between the two mappings
# meaningless; this test exists so nobody "re-validates" the floor fix by
# reintroducing exactly that meaningless comparison later.
# ---------------------------------------------------------------------------


def test_clamp_and_affine_give_identical_auc_on_nonnegative_scores():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1.0, size=200)  # realistic regime: cosine >= 0 throughout
    labels = (rng.random(200) < 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        pytest.skip("degenerate label draw")

    auc_clamp = roc_auc(rv.clamp_scores(raw), labels)
    auc_affine = roc_auc(rv.affine_scores(raw), labels)
    assert auc_clamp == pytest.approx(auc_affine, abs=1e-9)


def test_clamp_and_affine_give_identical_auc_with_a_single_negative_score():
    """A lone negative cosine (no tie induced by clipping it to 0, since no
    other score is also <= 0) still preserves the identity exactly."""
    raw = np.array([-0.03, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8])
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    auc_clamp = roc_auc(rv.clamp_scores(raw), labels)
    auc_affine = roc_auc(rv.affine_scores(raw), labels)
    assert auc_clamp == pytest.approx(auc_affine, abs=1e-9)


def test_run_floor_analysis_reports_the_same_auc_identity():
    """Integration of the identity check into the module's own
    `run_floor_analysis` -- the function the real experiment calls."""
    rng = np.random.default_rng(1)
    n = 60
    pairs = pd.DataFrame({
        "label": ([1] * (n // 2)) + ([0] * (n // 2)),
        "score_raw": rng.uniform(0.0, 1.0, size=n),
    })
    result = rv.run_floor_analysis(pairs)
    assert result["auc_identity"]["matches_to_1e9"] is True
    assert result["auc_identity"]["abs_diff"] < 1e-9


# ---------------------------------------------------------------------------
# class_balance
# ---------------------------------------------------------------------------


def test_class_balance_counts_and_rate():
    labels = np.array([1, 1, 0, 0, 0])
    assert rv.class_balance(labels) == {"n": 5, "n_pos": 2, "n_neg": 3, "rate_pos": pytest.approx(0.4)}


# ---------------------------------------------------------------------------
# corpus_embedding_cache_path -- naming/collision avoidance vs experiment 08
# ---------------------------------------------------------------------------


def test_corpus_embedding_cache_path_is_namespaced_by_dataset():
    p_sf = rv.corpus_embedding_cache_path(Path("/tmp/cache"), "scifact", "org/model-name")
    p_nf = rv.corpus_embedding_cache_path(Path("/tmp/cache"), "nfcorpus", "org/model-name")
    assert p_sf.name == "scifact_corpus_embeddings__org__model-name.npy"
    assert p_nf.name == "nfcorpus_corpus_embeddings__org__model-name.npy"
    assert p_sf != p_nf
    assert "/" not in p_sf.name


# ---------------------------------------------------------------------------
# PrecomputedEmbedder -- the adapter that lets the REAL context_relevance run
# against cached vectors without re-encoding text. Exercised end-to-end here
# with small hand-built vectors (no model download) so the wiring itself is
# tested, not just the lookup dict.
# ---------------------------------------------------------------------------


def test_precomputed_embedder_matches_context_relevance_clamp_and_raw():
    # Query and passage vectors chosen so raw cosine has a known, hand-computed value.
    text_to_vec = {
        "the query": np.array([1.0, 0.0]),
        "an orthogonal passage": np.array([0.0, 1.0]),   # cos = 0.0
        "an opposite passage": np.array([-1.0, 0.0]),    # cos = -1.0
        "a matching passage": np.array([1.0, 0.0]),       # cos = 1.0
    }
    embedder = rv.PrecomputedEmbedder(text_to_vec)

    raw_orth = context_relevance("the query", ["an orthogonal passage"], embedder, scaled=False)
    v2_orth = context_relevance("the query", ["an orthogonal passage"], embedder, scaled=True)
    assert raw_orth == pytest.approx(0.0)
    assert v2_orth == pytest.approx(0.0)

    raw_opp = context_relevance("the query", ["an opposite passage"], embedder, scaled=False)
    v2_opp = context_relevance("the query", ["an opposite passage"], embedder, scaled=True)
    assert raw_opp == pytest.approx(-1.0)
    assert v2_opp == pytest.approx(0.0)  # clamped

    raw_match = context_relevance("the query", ["a matching passage"], embedder, scaled=False)
    assert raw_match == pytest.approx(1.0)


def test_precomputed_embedder_batches_multiple_texts():
    embedder = rv.PrecomputedEmbedder({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    out = embedder.encode(["a", "b"])
    assert out.shape == (2, 2)
    assert out[0].tolist() == [1.0, 0.0]
    assert out[1].tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# build_text_to_vec -- corpus vectors reused verbatim, queries encoded fresh
# ---------------------------------------------------------------------------


def test_build_text_to_vec_maps_corpus_texts_and_encodes_queries():
    corpus_texts = ["doc one", "doc two"]
    corpus_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    queries_by_id = {"q1": "some query text"}
    embedder = FakeEmbedder(dim=4)

    text_to_vec = rv.build_text_to_vec(corpus_texts, corpus_embs, queries_by_id, ["q1"], embedder)

    assert np.array_equal(text_to_vec["doc one"], corpus_embs[0])
    assert np.array_equal(text_to_vec["doc two"], corpus_embs[1])
    assert "some query text" in text_to_vec
    assert text_to_vec["some query text"].shape == (4,)


# ---------------------------------------------------------------------------
# _tier_subset -- tier separation (Required Analysis 1: never pool tiers)
# ---------------------------------------------------------------------------


def test_tier_subset_includes_positive_and_only_the_named_tier():
    pairs = pd.DataFrame({
        "tier": ["positive", "positive", "random", "random", "hard", "hard"],
        "label": [1, 1, 0, 0, 0, 0],
    })
    sub_random = rv._tier_subset(pairs, "random")
    sub_hard = rv._tier_subset(pairs, "hard")
    assert set(sub_random["tier"]) == {"positive", "random"}
    assert set(sub_hard["tier"]) == {"positive", "hard"}
    assert len(sub_random) == 4
    assert len(sub_hard) == 4
