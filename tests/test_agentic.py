"""Tests for trust-gated agentic retrieval (agentic.py::answer_iterative): the
loop that reformulates and re-retrieves while the MEASURED trust of the best
answer so far stays below Config.abstain_threshold, instead of asking an LLM
whether it is satisfied. No model downloads, no network -- uses the fake
embedder/NLI from conftest.py and scripted generator stubs defined here.
"""
import json

import pytest

from ragtrust.agentic import answer_iterative
from ragtrust.config import Config
from ragtrust.generation.base import GeneratedAnswer
from ragtrust.pipeline import RAGTrustPipeline

PASSAGE_A = "Photosynthesis converts sunlight into chemical energy inside plant leaves."
PASSAGE_C = "The mitochondria is the powerhouse of the animal cell structure entirely."
QUESTION = "How does photosynthesis work?"


class ScriptedGenerator:
    """Generator whose `generate()` text/citations and `complete()` (reformulation)
    output are scripted per-call-count, so a test can control exactly what each
    round of the loop sees without depending on a real model. Records call
    counts and the passages it was last handed, mirroring test_pipeline.py's
    `StubGenerator`.
    """

    def __init__(self, answers: list, citations_list: list = None, reformulations: list = None):
        self.answers = answers
        self.citations_list = citations_list or [{}] * len(answers)
        self.reformulations = reformulations or []
        self.generate_calls = 0
        self.complete_calls = 0
        self.last_passages = None

    def generate(self, query, passages):
        idx = min(self.generate_calls, len(self.answers) - 1)
        self.generate_calls += 1
        self.last_passages = passages
        return GeneratedAnswer(text=self.answers[idx], citations=self.citations_list[idx])

    def complete(self, prompt):
        idx = min(self.complete_calls, len(self.reformulations) - 1)
        self.complete_calls += 1
        return self.reformulations[idx]


class NoCompleteGenerator:
    """A generator with no `complete` method at all -- e.g. a backend that
    never implemented raw completion. Used to test that reformulation degrades
    rather than raising (contextualisation-style graceful degradation)."""

    def __init__(self, text: str):
        self.text = text
        self.generate_calls = 0

    def generate(self, query, passages):
        self.generate_calls += 1
        return GeneratedAnswer(text=self.text, citations={})


class RecordingFakeNLI:
    """Wraps FakeNLI and records every premise it is asked to score -- copied
    from test_pipeline.py's contextual-invariant test so the same check can be
    made through `answer_iterative` rather than only through `answer()`."""

    def __init__(self):
        from ragtrust.metrics.nli import FakeNLI

        self._inner = FakeNLI()
        self.premises_seen: list = []

    def probs(self, premise, hypothesis):
        self.premises_seen.append(premise)
        return self._inner.probs(premise, hypothesis)

    def batch_probs(self, pairs):
        self.premises_seen.extend(p for p, _ in pairs)
        return self._inner.batch_probs(pairs)


def _pipeline(**config_kwargs) -> RAGTrustPipeline:
    from ragtrust.metrics.nli import FakeNLI
    from conftest import FakeEmbedder

    cfg = Config(**config_kwargs)
    return RAGTrustPipeline(cfg, nli=FakeNLI(), embedder=FakeEmbedder())


# ------------------------------------------------------------- single round


def test_single_round_clearing_threshold_stops_immediately():
    # retrieval_gate=-2.0 means Gate 1 never fires; the default abstain_threshold
    # (0.5) is comfortably cleared by an answer that closely echoes its one
    # source passage with a correct citation -- verified empirically to reach
    # geometric trust ~0.64.
    pipeline = _pipeline(retrieval_gate=-2.0, k=1)
    pipeline.index_texts([PASSAGE_A])
    generator = ScriptedGenerator(
        [PASSAGE_A], citations_list=[{0: 0}],
        reformulations=["should never be requested"],
    )
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=3)

    assert len(result.rounds) == 1
    assert result.rounds[0].stop_reason == "trust_threshold_met"
    assert result.llm_calls == 1, "only the generation call -- reformulation must never be attempted"
    assert generator.complete_calls == 0
    assert result.result.abstained is False


# --------------------------------------------------------- multi-round loop


def test_low_trust_first_round_triggers_reformulation_and_second_round():
    # abstain_threshold=0.9 is set unreachably high on purpose so round 1
    # (trust ~0.64) does not clear it and the loop must reformulate and try
    # a second round.
    pipeline = _pipeline(retrieval_gate=-2.0, k=1, abstain_threshold=0.9)
    pipeline.index_texts([PASSAGE_A, PASSAGE_C])
    reformulated_query = "mitochondria powerhouse animal cell structure"
    generator = ScriptedGenerator(
        [PASSAGE_A, PASSAGE_A],
        citations_list=[{0: 0}, {0: 0}],
        reformulations=[reformulated_query],
    )
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=2)

    assert len(result.rounds) == 2
    assert result.rounds[0].query == QUESTION
    assert result.rounds[1].query == reformulated_query
    assert generator.complete_calls == 1


def test_highest_trust_round_is_returned_not_the_last_round():
    """Round 2 (after reformulation) is deliberately WORSE than round 1: its
    scripted answer text matches the newly-pooled passage C rather than A, but
    the citation still (wrongly) points at pooled-position 0, which is A --
    an unsupported citation, driving attribution precision, and therefore the
    whole geometric aggregate, to exactly 0.0. Verified empirically: round 1
    trust ~0.64, round 2 trust 0.0. abstain_threshold=0.7 is set so that
    NEITHER round clears it, so the loop runs to max_rounds=2 rather than
    stopping early at round 1 -- which is required to prove anything about
    "not simply the last round": if round 1 had cleared the threshold, the
    loop would stop there and round 2 would never run at all.

    Because neither round cleared the threshold, the honesty rule in
    `answer_iterative` (see its final comment block) forces the returned
    result to an abstention -- with more than one round run, iterating must
    never be able to present an unproven answer as final just because it was
    the best of a bad lot. What this test actually verifies is that the data
    behind that forced abstention -- the trust dict -- is ROUND 1's (~0.64),
    not round 2's (0.0): the selection picked the highest-trust round, not
    the round that happened to run last.
    """
    pipeline = _pipeline(retrieval_gate=-2.0, k=1, abstain_threshold=0.7)
    pipeline.index_texts([PASSAGE_A, PASSAGE_C])
    generator = ScriptedGenerator(
        [PASSAGE_A, PASSAGE_C],
        citations_list=[{0: 0}, {0: 0}],  # round 2's citation is wrong on purpose -- see docstring
        reformulations=["mitochondria powerhouse animal cell structure"],
    )
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=2)

    assert len(result.rounds) == 2
    round1_trust, round2_trust = result.rounds[0].trust, result.rounds[1].trust
    assert round1_trust > round2_trust, "test setup assumption: round 2 must be worse than round 1"
    assert round2_trust == pytest.approx(0.0)

    # Neither round cleared abstain_threshold=0.7, so per the honesty rule the
    # final result must be an abstention (see answer_iterative's comment).
    assert result.result.abstained is True
    # But the trust/metrics DATA carried forward is round 1's, not round 2's --
    # this is the assertion that actually distinguishes "highest trust" from
    # "last round" selection.
    assert result.result.trust["geometric"] == pytest.approx(round1_trust)
    assert result.result.trust["geometric"] != pytest.approx(round2_trust)


def test_all_rounds_fail_returns_abstention_never_a_low_trust_answer():
    # An answer text unrelated to either indexed passage fails Gate 2 (no
    # claim supported) on every round, so every round's own AnswerResult is
    # already abstained=True, trust 0.0 -- the loop must never manufacture a
    # "best" answer out of a set of legitimate abstentions.
    pipeline = _pipeline(retrieval_gate=-2.0, k=1)
    pipeline.index_texts([PASSAGE_A, PASSAGE_C])
    unrelated = "Something entirely different about quantum flux capacitors and warp drives here."
    generator = ScriptedGenerator(
        [unrelated, unrelated], citations_list=[{}, {}],
        reformulations=["quantum flux capacitor warp drive stabilizer"],
    )
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=2)

    assert all(r.abstained for r in result.rounds)
    assert result.result.abstained is True
    assert result.result.answer == "Not answerable from this corpus."
    assert result.result.trust["geometric"] == 0.0


def test_no_new_passages_stops_loop_without_spending_a_generation_call():
    # Only ONE passage is indexed, so no matter how the query is reformulated,
    # round 2's retrieval can only ever return the passage round 1 already
    # pooled -- the loop must detect that and stop BEFORE calling answer_with
    # again, rather than burning a second generation call to re-score an
    # identical pool. abstain_threshold=0.99 is unreachable on purpose so
    # round 1 (trust ~0.64) does not clear it and stop the loop before this
    # scenario -- no_new_passages, not the threshold -- gets to fire.
    pipeline = _pipeline(retrieval_gate=-2.0, k=1, abstain_threshold=0.99)
    pipeline.index_texts([PASSAGE_A])
    generator = ScriptedGenerator(
        [PASSAGE_A], citations_list=[{0: 0}],
        reformulations=["totally different wording about chlorophyll"],
    )
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=3)

    assert len(result.rounds) == 2
    assert result.rounds[1].stop_reason == "no_new_passages"
    assert generator.generate_calls == 1, "round 2 must not call generate() again"
    assert generator.complete_calls == 1, "the reformulation call itself still happened"
    assert result.llm_calls == 2  # 1 generation + 1 reformulation, no second generation


def test_generator_without_complete_degrades_to_one_round():
    # abstain_threshold set unreachably high; the point is that with no
    # `complete` method available at all, reformulation must degrade to
    # stopping the loop rather than raising AttributeError.
    pipeline = _pipeline(retrieval_gate=-2.0, k=1, abstain_threshold=0.99)
    pipeline.index_texts([PASSAGE_A])
    generator = NoCompleteGenerator(PASSAGE_A)
    pipeline._generator = generator

    result = answer_iterative(pipeline, QUESTION, max_rounds=3)  # must not raise

    assert len(result.rounds) == 1
    assert result.rounds[0].stop_reason == "reformulation_unavailable"


def test_max_rounds_1_runs_one_round_and_returns_answer_when_trust_clears():
    """max_rounds=1 does exactly one round and one generation call, and passes
    the answer through when the measured trust clears the threshold."""
    cfg_kwargs = dict(retrieval_gate=-2.0, k=1, abstain_threshold=0.0)

    direct = _pipeline(**cfg_kwargs)
    direct.index_texts([PASSAGE_A])
    direct._generator = NoCompleteGenerator(PASSAGE_A)
    direct_result = direct.answer(QUESTION)

    iterative_pipeline = _pipeline(**cfg_kwargs)
    iterative_pipeline.index_texts([PASSAGE_A])
    gen = NoCompleteGenerator(PASSAGE_A)
    iterative_pipeline._generator = gen
    iterative_result = answer_iterative(iterative_pipeline, QUESTION, max_rounds=1)

    assert iterative_result.result.answer == direct_result.answer
    assert iterative_result.result.abstained is False
    assert len(iterative_result.rounds) == 1
    assert iterative_result.llm_calls == 1


def test_single_round_below_threshold_abstains_unlike_plain_answer():
    """`answer_iterative` is deliberately STRICTER than `pipeline.answer()`,
    including at max_rounds=1.

    `answer()` gates on Gate 2's per-claim check and will return an answer whose
    whole-answer geometric aggregate is below `abstain_threshold`, leaving the
    caller to consult `is_trustworthy`. `answer_iterative` promises something
    narrower -- it retrieves until measured trust clears the threshold and says
    so when it never did -- so it declines here. Applying that rule only once a
    second round has run would make an identical answer report abstained=False
    after one round and abstained=True after two, i.e. make the outcome depend
    on how many rounds ran beside it rather than on the answer itself.
    """
    cfg_kwargs = dict(retrieval_gate=-2.0, k=1, abstain_threshold=0.1)

    direct = _pipeline(**cfg_kwargs)
    direct.index_texts([PASSAGE_A])
    direct._generator = NoCompleteGenerator(PASSAGE_A)
    direct_result = direct.answer(QUESTION)
    # Precondition: plain answer() does NOT abstain here, or this proves nothing.
    assert direct_result.abstained is False
    assert direct_result.trust["geometric"] < 0.1

    iterative_pipeline = _pipeline(**cfg_kwargs)
    iterative_pipeline.index_texts([PASSAGE_A])
    iterative_pipeline._generator = NoCompleteGenerator(PASSAGE_A)
    iterative_result = answer_iterative(iterative_pipeline, QUESTION, max_rounds=1)

    assert iterative_result.result.abstained is True
    assert "trust threshold" in (iterative_result.result.abstain_reason or "")
    assert len(iterative_result.rounds) == 1


# ------------------------------------------------------------------ guards


def test_answer_iterative_raises_without_generator():
    pipeline = _pipeline()
    pipeline.index_texts([PASSAGE_A])
    with pytest.raises(ValueError):
        answer_iterative(pipeline, QUESTION)


def test_answer_iterative_raises_when_nothing_indexed():
    pipeline = _pipeline()
    pipeline._generator = NoCompleteGenerator(PASSAGE_A)
    with pytest.raises(ValueError):
        answer_iterative(pipeline, QUESTION)


# ------------------------------------------------------------------- dicts


def test_to_dict_is_json_serialisable():
    pipeline = _pipeline(retrieval_gate=-2.0, k=1)
    pipeline.index_texts([PASSAGE_A])
    pipeline._generator = ScriptedGenerator([PASSAGE_A], citations_list=[{0: 0}])

    result = answer_iterative(pipeline, QUESTION, max_rounds=1)

    dumped = json.dumps(result.to_dict())
    assert "rounds" in json.loads(dumped)


# ----------------------------------------------------- contextual invariant


def test_contextual_blurb_never_reaches_nli_generator_or_output_via_answer_iterative():
    """The SAME invariant test_pipeline.py's `test_contextual_blurb_never_
    reaches_nli_generator_or_output` runs against `answer()`, run here against
    `answer_iterative` instead -- proving the invariant now holds for a caller
    that never goes through `answer()` at all, because `pipeline.py::answer_
    with` moved the source-text swap to its own top (see that method's
    comment)."""
    sentinel = "ZZSENTINEL topic overview."
    source = "Photosynthesis converts sunlight into chemical energy in plants."
    contextualized_text = f"{sentinel} {source}"

    nli = RecordingFakeNLI()
    generator = ScriptedGenerator(
        ["Photosynthesis converts sunlight into chemical energy."], citations_list=[{}],
    )
    pipeline = _pipeline(retrieval_gate=-2.0, k=1)
    pipeline._nli = nli
    pipeline._generator = generator
    pipeline._install(
        [contextualized_text], [{"source": "doc.md", "page": 1}], [source]
    )

    result = answer_iterative(pipeline, QUESTION, max_rounds=2)

    for premise in nli.premises_seen:
        assert sentinel not in premise, "sentinel blurb reached an NLI premise"
    assert generator.last_passages is not None
    for p in generator.last_passages:
        assert sentinel not in getattr(p, "text", p), "sentinel blurb reached the generator"
    for p in result.result.passages:
        assert sentinel not in getattr(p, "text", p), "sentinel blurb reached result.passages"
    for r in result.rounds:
        assert sentinel not in r.query, "sentinel blurb leaked into a round's retrieval query"

    dumped = json.dumps(result.to_dict())
    assert sentinel not in dumped, "sentinel blurb reached to_dict() output"
