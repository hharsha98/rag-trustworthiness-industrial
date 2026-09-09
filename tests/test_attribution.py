import pytest

from ragtrust.metrics.attribution import attribution
from ragtrust.metrics.nli import FakeNLI


def test_precision_vacuous_truth_when_no_citations_emitted():
    claims = ["The sky is blue today."]
    passages = ["The sky is blue today."]
    result = attribution(claims, {}, passages, FakeNLI(), tau=0.5)
    assert result.precision == 1.0


def test_recall_vacuous_truth_when_no_claim_is_supported():
    claims = ["Bananas are purple and made of solid steel."]
    passages = ["The sky is blue today."]
    result = attribution(claims, {}, passages, FakeNLI(), tau=0.9)
    assert result.recall == 1.0


def test_precision_and_recall_perfect_with_correct_citation():
    claims = ["The sky is blue today."]
    passages = ["The sky is blue today."]
    citations = {0: 0}
    result = attribution(claims, citations, passages, FakeNLI(), tau=0.5)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(1.0)


def test_precision_penalized_by_citation_pointing_at_wrong_passage():
    claims = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    passages = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    # claim 0 is cited against an unrelated passage index -> low entailment
    citations = {0: 1}
    result = attribution(claims, citations, passages, FakeNLI(), tau=0.5)
    assert result.precision == 0.0


def test_recall_penalized_when_supported_claim_has_no_citation():
    claims = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    passages = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    # both claims are well supported by an identical passage (tau small),
    # but only claim 0 carries a citation.
    citations = {0: 0}
    result = attribution(claims, citations, passages, FakeNLI(), tau=0.1)
    assert result.recall == pytest.approx(0.5)


def test_f1_zero_when_precision_and_recall_both_zero():
    claims = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    passages = [
        "The sky is blue today.",
        "Water boils at one hundred degrees celsius.",
    ]
    # Both claims are supported at this low tau, but the only citation
    # emitted targets an out-of-range claim index, so it contributes to
    # neither claim's "carries a citation" recall count (recall = 0) and is
    # not a valid/correct citation either (precision = 0).
    citations = {5: 0}
    result = attribution(claims, citations, passages, FakeNLI(), tau=0.1)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_empty_claims_is_vacuous_on_both_sides():
    result = attribution([], {}, ["some passage"], FakeNLI(), tau=0.5)
    assert result.precision == 1.0
    assert result.recall == 1.0
