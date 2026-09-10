"""Tests for data/golden_set.json's structure -- NOT for
experiments/17_golden_regression.py's behaviour (that script needs real
embed/NLI model weights, per its own module docstring, and is exercised
manually/in CI as a separate gate, not via pytest). This file only checks
the invariants that keep the golden set usable without a live LLM:

  - it parses as JSON and has all three classes;
  - every entry has the required fields with sane types;
  - every `in_corpus` question is a key in `data/cached_answers.json` --
    `CachedGenerator` (src/ragtrust/generation/cached.py) can only answer a
    question it has a recorded entry for, so this is the exact invariant
    that keeps experiments/17 (and any future CI job running it) free of a
    live LLM dependency for the "answer" class.

No model downloads, no network.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = ROOT / "data" / "golden_set.json"
CACHED_ANSWERS_PATH = ROOT / "data" / "cached_answers.json"

REQUIRED_FIELDS = {"question", "expect", "class", "note"}
VALID_EXPECT = {"answer", "abstain"}
VALID_CLASSES = {"in_corpus", "out_of_corpus", "adversarial"}


def _load_golden_set() -> list:
    return json.loads(GOLDEN_SET_PATH.read_text())


def test_golden_set_parses_as_a_list_of_objects():
    golden = _load_golden_set()
    assert isinstance(golden, list)
    assert len(golden) > 0
    assert all(isinstance(entry, dict) for entry in golden)


def test_golden_set_has_roughly_thirty_cases():
    # "~30" per the spec this file was written against -- not pinned to
    # exactly 30 so a future small addition/removal doesn't need this test
    # touched, but far enough from either extreme to catch "someone deleted
    # most of the file" or "someone pasted it in twice".
    golden = _load_golden_set()
    assert 24 <= len(golden) <= 40


def test_golden_set_has_all_three_classes():
    golden = _load_golden_set()
    classes_present = {entry.get("class") for entry in golden}
    assert VALID_CLASSES <= classes_present


def test_every_entry_has_the_required_fields_with_sane_types():
    golden = _load_golden_set()
    for entry in golden:
        assert REQUIRED_FIELDS <= set(entry.keys()), f"missing field(s) in {entry!r}"
        assert isinstance(entry["question"], str) and entry["question"].strip()
        assert isinstance(entry["note"], str) and entry["note"].strip()
        assert entry["expect"] in VALID_EXPECT, f"bad expect in {entry!r}"
        assert entry["class"] in VALID_CLASSES, f"bad class in {entry!r}"


def test_in_corpus_expects_answer_and_out_of_corpus_and_adversarial_expect_abstain():
    # Not strictly required by the schema alone, but is the whole point of
    # having three classes: this is what experiments/17 actually checks per
    # case, so a golden set where these are already mismatched would make
    # every future regression check meaningless.
    golden = _load_golden_set()
    for entry in golden:
        if entry["class"] == "in_corpus":
            assert entry["expect"] == "answer", entry
        else:
            assert entry["expect"] == "abstain", entry


def test_every_in_corpus_question_is_a_key_in_cached_answers():
    # The invariant that keeps CI LLM-free (see module docstring): CachedGenerator
    # (src/ragtrust/generation/cached.py) raises GenerationError for any question
    # it has no recorded entry for, so an "answer"-class golden case whose question
    # is not a cached_answers.json key would crash experiments/17 instead of
    # measuring anything.
    golden = _load_golden_set()
    cached_questions = set(json.loads(CACHED_ANSWERS_PATH.read_text()).keys())
    in_corpus_questions = [e["question"] for e in golden if e["class"] == "in_corpus"]
    assert in_corpus_questions, "expected at least one in_corpus case"
    missing = [q for q in in_corpus_questions if q not in cached_questions]
    assert not missing, f"in_corpus questions missing from cached_answers.json: {missing}"


def test_no_duplicate_questions_within_the_golden_set():
    golden = _load_golden_set()
    questions = [entry["question"] for entry in golden]
    assert len(questions) == len(set(questions))
