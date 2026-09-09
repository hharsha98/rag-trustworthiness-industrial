"""Tests for the ragtrust CLI. Invokes `main([...])` directly -- no shelling
out. `RAGTrustPipeline.embedder`/`.nli` are monkeypatched to the fake,
dependency-free implementations from conftest.py so these tests never
download a model or touch the network; `--generator cached` similarly keeps
generation itself offline.
"""
import json

import pytest

from ragtrust.cli import main
from ragtrust.metrics.nli import FakeNLI
from ragtrust.pipeline import RAGTrustPipeline
# `tests/` has no __init__.py, so it is not a package and `tests.conftest` does not
# resolve. Under pytest's default prepend import mode the test directory itself is on
# sys.path, so conftest is importable by its bare module name.
from conftest import FakeEmbedder


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """Every CLI subcommand builds its own RAGTrustPipeline internally, so
    the only patch point available from outside is the class properties
    themselves -- this keeps `ragtrust index`/`ask`/`eval` fast and offline
    without touching cli.py."""
    monkeypatch.setattr(RAGTrustPipeline, "embedder", property(lambda self: FakeEmbedder()))
    monkeypatch.setattr(RAGTrustPipeline, "nli", property(lambda self: FakeNLI()))


@pytest.fixture
def demo_corpus_path():
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "data" / "demo_corpus.md")


@pytest.fixture
def cache_path(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "What is the purpose of max-pooling layers in CNNs?": {
            "text": "Max-pooling reduces spatial resolution and enlarges the "
                     "receptive field of subsequent layers [1].",
            "citations": {"0": 0},
        },
        "What is the capital of France?": "NOT_IN_CORPUS",
    }))
    return str(path)


def test_index_then_ask_json_round_trip(tmp_path, demo_corpus_path, cache_path, capsys):
    index_dir = str(tmp_path / "idx")

    rc = main(["index", demo_corpus_path, "--out", index_dir])
    assert rc == 0
    index_out = capsys.readouterr().out
    assert "Indexed" in index_out
    assert "passages" in index_out

    rc = main([
        "ask", "What is the purpose of max-pooling layers in CNNs?",
        "--index", index_dir,
        "--generator", "cached", "--cache-path", cache_path,
        "--retrieval-gate", "-2.0", "--abstain-threshold", "0.0",
        "--json",
    ])
    assert rc == 0
    ask_out = capsys.readouterr().out
    result = json.loads(ask_out)
    assert "answer" in result
    assert "abstained" in result
    assert "trust" in result
    assert "passages" in result


def test_ask_abstains_clearly_for_out_of_corpus_question(tmp_path, demo_corpus_path, cache_path, capsys):
    index_dir = str(tmp_path / "idx")
    main(["index", demo_corpus_path, "--out", index_dir])
    capsys.readouterr()

    rc = main([
        "ask", "What is the capital of France?",
        "--index", index_dir,
        "--generator", "cached", "--cache-path", cache_path,
        # Forced above the max possible cosine similarity so the retrieval
        # gate always fires here, regardless of FakeEmbedder's hashed vectors
        # for this particular query/corpus pairing.
        "--retrieval-gate", "1.5",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ABSTAINED" in out


def test_ask_with_missing_index_exits_nonzero_with_message_not_traceback(tmp_path, capsys):
    rc = main([
        "ask", "Anything?",
        "--index", str(tmp_path / "does_not_exist"),
        "--generator", "cached",
    ])
    assert rc != 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
    assert "not found" in captured.err.lower() or "not found" in captured.out.lower()


def test_index_with_missing_corpus_exits_nonzero_with_message(tmp_path, capsys):
    rc = main(["index", str(tmp_path / "nope.md"), "--out", str(tmp_path / "idx")])
    assert rc != 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_eval_reports_aggregate_stats(tmp_path, demo_corpus_path, cache_path, capsys):
    index_dir = str(tmp_path / "idx")
    main(["index", demo_corpus_path, "--out", index_dir])
    capsys.readouterr()

    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([
        "What is the purpose of max-pooling layers in CNNs?",
        "What is the capital of France?",
    ]))
    out_path = tmp_path / "results.json"

    rc = main([
        "eval", str(questions_path),
        "--index", index_dir,
        "--generator", "cached", "--cache-path", cache_path,
        "--retrieval-gate", "-2.0", "--abstain-threshold", "0.0",
        "--out", str(out_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Questions: 2" in out
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["answered"] + payload["summary"]["abstained"] == 2
