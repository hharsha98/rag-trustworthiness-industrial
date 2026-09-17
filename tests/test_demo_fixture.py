"""Guards on the recorded runs the dashboard replays.

The dashboard plays `dashboard/demo-run.json` and `dashboard/demo-abstain.json`
on first arrival so a visitor sees a real measurement immediately rather than
waiting ~60s for a live one. Because those files are committed and served
publicly, two things have to stay true, and neither is obvious enough to trust
to memory.
"""
import json
import re
from pathlib import Path

import pytest

from ragtrust.service import AnswerResponse

REPO = Path(__file__).resolve().parents[1]
FIXTURES = [REPO / "dashboard" / "demo-run.json",
            REPO / "dashboard" / "demo-abstain.json"]
DEMO_CORPUS = REPO / "data" / "demo_corpus.md"


def _squash(text: str) -> str:
    """Collapse all whitespace, so a chunk that joined source lines with spaces
    still matches the file it came from."""
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="module")
def corpus_text() -> str:
    return _squash(DEMO_CORPUS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_passages_come_only_from_the_bundled_corpus(path, corpus_text):
    """The licensing guard, and the highest-severity test in this file.

    A recorded response embeds `passages[].text` verbatim. The evaluation corpus
    used elsewhere in this project is third-party lecture material that
    .gitignore deliberately excludes because the repository has no
    redistribution rights to it (see data/README.md). A fixture captured with a
    `corpus_id` pointing at an uploaded document -- or at that PDF -- would
    commit the exact text the .gitignore exists to keep out, to a public repo,
    where deleting it later does not un-publish it.

    Capture with `corpus_id` omitted so the run goes against data/demo_corpus.md,
    which is original text written for this repository.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    passages = payload["response"]["passages"]
    assert passages, f"{path.name} recorded no passages, so this guard would pass vacuously"

    for p in passages:
        source = (p.get("source") or {}).get("source")
        assert source == "demo_corpus.md", (
            f"{path.name} passage #{p.get('rank')} cites {source!r}. Fixtures must be "
            f"captured against the bundled demo corpus only."
        )
        assert _squash(p["text"]) in corpus_text, (
            f"{path.name} passage #{p.get('rank')} contains text that is not in "
            f"data/demo_corpus.md. Either it was captured against a different corpus, "
            f"or the text was edited by hand -- both are disallowed."
        )


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_still_matches_the_answer_schema(path):
    """Shape only, never values.

    The fixture is a transcript: if a recorded run has an unflattering number,
    that is the run, and asserting on values here would create pressure to
    re-record until the numbers look good. What this catches is AnswerResponse
    drifting away from what the dashboard replays, which would otherwise surface
    as a silently broken demo rather than a failing build.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    AnswerResponse(**payload["response"])


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_carries_its_own_provenance(path):
    """The page tells the visitor this is a recording and offers the raw file.

    That claim is only worth anything if the file says when it was taken, what
    it was taken against, and at which commit.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("recorded_utc", "recorded_against", "retrieval_mode", "source_commit", "total_ms"):
        assert payload.get(key), f"{path.name} is missing provenance key {key!r}"
    assert payload["recorded_against"] == "demo_corpus.md"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["recorded_utc"]), (
        "recorded_utc must be ISO-8601 UTC with a trailing Z, so the page can print it "
        "without guessing a timezone"
    )


def test_the_two_fixtures_cover_both_outcomes():
    """An abstention is not a degenerate answer, it is the other half of the product.

    It is also where the timeline's 'not run' state is visible, since the gate
    returns before generate/decompose/entail/score ever happen.
    """
    run, abstain = (json.loads(p.read_text(encoding="utf-8"))["response"] for p in FIXTURES)

    assert run["abstained"] is False
    assert abstain["abstained"] is True

    answered_stages = [s["stage"] for s in run["stage_timings"]]
    declined_stages = [s["stage"] for s in abstain["stage_timings"]]
    assert answered_stages == ["retrieve", "gate", "generate", "decompose", "entail", "score"]
    assert declined_stages == ["retrieve", "gate"], (
        "the abstention fixture must stop at the gate -- that is what makes the four "
        "'not run' segments real rather than a rendering trick"
    )
