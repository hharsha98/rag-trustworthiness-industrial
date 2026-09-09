"""Property tests asserting every Proposition in METRICS.md."""
import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtrust.metrics.aggregate import aggregate_arithmetic, aggregate_geometric
from ragtrust.metrics.conciseness import conciseness
from ragtrust.metrics.faithfulness import faithfulness
from ragtrust.metrics.nli import FakeNLI

# ---------------------------------------------------------------------------
# Proposition 1: F in [0,1]; permutation invariant; monotone on append
# ---------------------------------------------------------------------------


@given(
    st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=4),
    st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=4),
)
def test_prop1_faithfulness_bounds_and_permutation_invariance(claims, passages):
    nli = FakeNLI()
    result = faithfulness(claims, passages, nli)
    assert 0.0 <= result.score <= 1.0

    shuffled = passages[:]
    random.shuffle(shuffled)
    result_shuffled = faithfulness(claims, shuffled, nli)
    assert result.score == pytest.approx(result_shuffled.score, abs=1e-9)


@given(
    st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=4),
    st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=4),
    st.text(min_size=1, max_size=15),
)
def test_prop1_faithfulness_monotone_non_decreasing_on_append(claims, passages, extra_passage):
    nli = FakeNLI()
    base = faithfulness(claims, passages, nli)
    extended = faithfulness(claims, passages + [extra_passage], nli)
    assert extended.score >= base.score - 1e-9


def test_faithfulness_empty_inputs_score_zero():
    nli = FakeNLI()
    assert faithfulness([], ["p"], nli).score == 0.0
    assert faithfulness(["c"], [], nli).score == 0.0


# ---------------------------------------------------------------------------
# Proposition 2: C in [0,1]; C=1 for <2 claims; C~0 for n identical claims;
# C<1 for duplicated claims among otherwise-distinct ones
# ---------------------------------------------------------------------------


def test_prop2_conciseness_edge_cases(fake_embedder):
    # Pairwise self-similarity is undefined with fewer than 2 claims -- there
    # is no pair to compare -- so conciseness returns None, not the old 1.0
    # sentinel. See metrics/conciseness.py module docstring.
    assert conciseness([], fake_embedder) is None
    assert conciseness(["a single claim"], fake_embedder) is None

    identical = ["Robots use sensors to perceive the environment."] * 4
    assert conciseness(identical, fake_embedder) == pytest.approx(0.0, abs=1e-6)

    duplicated = [
        "Robots use sensors to perceive the environment.",
        "Robots use sensors to perceive the environment.",
        "Jazz music has nothing to do with robots at all.",
    ]
    score = conciseness(duplicated, fake_embedder)
    assert 0.0 <= score < 1.0


def test_conciseness_unchanged_for_two_or_more_claims(fake_embedder):
    """The None fix only touches the n < 2 branch; n >= 2 still returns a
    plain float in [0,1], same as before."""
    claims = [
        "Robots use sensors to perceive the environment.",
        "Jazz music has nothing to do with robots at all.",
        "The weather today is sunny with a light breeze.",
    ]
    score = conciseness(claims, fake_embedder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Proposition 3: T_geom in [0,1]; 0 if any metric is 0; strictly increasing;
# T_geom <= T_arith (weighted AM-GM)
# ---------------------------------------------------------------------------

_WEIGHTS = {"faithfulness": 0.4, "attribution": 0.2, "relevance": 0.2, "conciseness": 0.2}


@given(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)
def test_prop3_geometric_bounds_and_am_gm_inequality(a, b, c, d):
    metrics = {"faithfulness": a, "attribution": b, "relevance": c, "conciseness": d}
    geometric = aggregate_geometric(metrics, _WEIGHTS)
    arithmetic = aggregate_arithmetic(metrics, _WEIGHTS)
    assert 0.0 <= geometric <= 1.0 + 1e-9
    assert geometric <= arithmetic + 1e-9


def test_prop3_geometric_is_zero_if_any_metric_is_zero():
    metrics = {"faithfulness": 0.0, "attribution": 0.9, "relevance": 0.9, "conciseness": 0.9}
    assert aggregate_geometric(metrics, _WEIGHTS) == 0.0


def test_prop3_geometric_strictly_increasing_in_each_metric():
    weights = {"a": 0.5, "b": 0.5}
    base = aggregate_geometric({"a": 0.5, "b": 0.5}, weights)
    higher_a = aggregate_geometric({"a": 0.6, "b": 0.5}, weights)
    higher_b = aggregate_geometric({"a": 0.5, "b": 0.6}, weights)
    assert higher_a > base
    assert higher_b > base


def test_aggregate_geometric_raises_on_nonpositive_weight():
    with pytest.raises(ValueError):
        aggregate_geometric({"a": 0.5, "b": 0.5}, {"a": 0.0, "b": 1.0})
    with pytest.raises(ValueError):
        aggregate_geometric({"a": 0.5, "b": 0.5}, {"a": -0.1, "b": 1.1})


def test_prop3_equality_holds_when_all_metrics_equal():
    metrics = {"faithfulness": 0.7, "attribution": 0.7, "relevance": 0.7, "conciseness": 0.7}
    geometric = aggregate_geometric(metrics, _WEIGHTS)
    arithmetic = aggregate_arithmetic(metrics, _WEIGHTS)
    assert geometric == pytest.approx(arithmetic, abs=1e-9)
    assert geometric == pytest.approx(0.7, abs=1e-9)


# ---------------------------------------------------------------------------
# Regression test for the CRUX bug: aggregate_geometric already renormalises
# its weights over the metrics actually present (`total_weight`,
# `normalized_weights`), but aggregate_arithmetic used to sum raw weights
# over present keys without renormalising. Dropping conciseness (weight 0.2,
# e.g. because it is undefined for < 2 claims -- see metrics/conciseness.py)
# left T_arith capped at 0.8 of the weight mass while T_geom renormalised to
# 1.0. With faithfulness=attribution=relevance=1.0 that gave
# T_arith=0.8 < T_geom=1.0, violating Proposition 3 (T_geom <= T_arith,
# proved via weighted AM-GM for weights summing to 1). This is exactly the
# scenario a pipeline hits when it answers with a single claim.
# ---------------------------------------------------------------------------


def test_aggregate_arithmetic_renormalises_over_present_keys():
    """Before the fix, an all-1.0 metric set with conciseness dropped (weight
    0.2 missing) scored 0.8 -- the un-renormalised sum of the remaining
    weights -- instead of 1.0."""
    metrics = {"faithfulness": 1.0, "attribution": 1.0, "relevance": 1.0}
    assert aggregate_arithmetic(metrics, _WEIGHTS) == pytest.approx(1.0, abs=1e-9)


def test_prop3_holds_when_conciseness_is_dropped_all_ones_case():
    """The all-ones case is exactly where the un-renormalised
    aggregate_arithmetic used to fail Proposition 3: T_arith=0.8 < T_geom=1.0.
    Guards the CRUX fix directly."""
    metrics = {"faithfulness": 1.0, "attribution": 1.0, "relevance": 1.0}
    geometric = aggregate_geometric(metrics, _WEIGHTS)
    arithmetic = aggregate_arithmetic(metrics, _WEIGHTS)
    assert arithmetic == pytest.approx(1.0, abs=1e-9)
    assert geometric == pytest.approx(1.0, abs=1e-9)
    assert geometric <= arithmetic + 1e-9


@given(
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
    st.floats(min_value=0.01, max_value=1.0),
)
def test_prop3_holds_when_conciseness_is_dropped(a, b, c):
    """General form of the CRUX regression: with conciseness missing from
    `metrics` entirely (not just weighted 0 -- dropped, per aggregate.py's
    docstring), Proposition 3 (T_geom <= T_arith) must still hold once both
    aggregates renormalise over the same present-key weight basis."""
    metrics = {"faithfulness": a, "attribution": b, "relevance": c}
    geometric = aggregate_geometric(metrics, _WEIGHTS)
    arithmetic = aggregate_arithmetic(metrics, _WEIGHTS)
    assert 0.0 <= geometric <= 1.0 + 1e-9
    assert geometric <= arithmetic + 1e-9


def test_pipeline_single_claim_answer_exposes_none_and_keeps_trust_sane():
    """End-to-end regression: a real single-claim pipeline answer must expose
    metrics['conciseness'] as None (undefined, not the old 1.0 sentinel) while
    still producing finite, in-range trust aggregates -- i.e. the pipeline
    wiring (pipeline.py's `scored` dict) actually drops it rather than
    crashing or silently defaulting it."""
    import math

    from test_pipeline import StubGenerator, _pipeline

    generator = StubGenerator(text="Photosynthesis converts sunlight into chemical energy.")
    pipeline = _pipeline(retrieval_gate=-2.0, abstain_threshold=0.1)
    pipeline._generator = generator
    pipeline.index_texts(["Photosynthesis converts sunlight into chemical energy in plants."])

    result = pipeline.answer("How does photosynthesis work?")

    assert result.abstained is False
    assert len(result.claims) == 1, "this test needs a single-claim answer to exercise conciseness=None"
    assert result.metrics["conciseness"] is None

    for key in ("arithmetic", "geometric"):
        value = result.trust[key]
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
