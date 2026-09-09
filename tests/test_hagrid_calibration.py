"""Tests for experiments/14_hagrid_calibration.py -- pure functions only.

Fast: no network, no model downloads. `parse_citation_markers`,
`get_cited_quotes`, `sentence_label`, `extract_items`, `f1_at_threshold`,
`threshold_sweep`, `best_tau_by_f1`, `f1_at_tau`, `cross_dataset_transfer`
and the single/multi-citation split are exercised here on synthetic
HAGRID-shaped rows. Everything that needs the HAGRID download or the real
NLI model lives behind `main()` / `load_hagrid()` / `score_items()` (never
called here).

The module under test is a standalone script (like experiments/03, 08, 10,
11, etc.), not a package, so it is loaded by file path rather than imported
by name -- same pattern as tests/test_attreval_validation.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "14_hagrid_calibration.py"
_SPEC = importlib.util.spec_from_file_location("hagrid_calibration_module", _MODULE_PATH)
hc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hc
_SPEC.loader.exec_module(hc)


# ---------------------------------------------------------------------------
# parse_citation_markers
# ---------------------------------------------------------------------------


def test_parse_citation_markers_single():
    assert hc.parse_citation_markers("Some claim about Russell [2].") == [2]


def test_parse_citation_markers_comma_list_in_one_bracket():
    assert hc.parse_citation_markers("A claim with two sources [1, 2].") == [1, 2]


def test_parse_citation_markers_two_separate_brackets():
    text = "First part [1]. Second part continues here [3]."
    assert hc.parse_citation_markers(text) == [1, 3]


def test_parse_citation_markers_hyphenated_range():
    assert hc.parse_citation_markers("Casualties varied widely [1-4].") == [1, 2, 3, 4]


def test_parse_citation_markers_range_and_discrete_combined():
    assert hc.parse_citation_markers("See sources [1-3, 7].") == [1, 2, 3, 7]


def test_parse_citation_markers_no_markers():
    assert hc.parse_citation_markers("A sentence with no citation at all.") == []


def test_parse_citation_markers_deduplicates_and_sorts():
    assert hc.parse_citation_markers("Cited twice [3] and again [3], plus [1].") == [1, 3]


def test_parse_citation_markers_ignores_non_numeric_brackets():
    assert hc.parse_citation_markers("A footnote-style bracket [sic] here.") == []


def test_parse_citation_markers_ignores_reversed_range():
    # a > b: not a valid ascending range, ignored rather than raising.
    assert hc.parse_citation_markers("Malformed range [5-2].") == []


def test_parse_citation_markers_does_not_subtract_one():
    """Unlike ragtrust.generation.ollama.parse_citations (0-based list
    position), this parser returns the marker value as-is, since HAGRID's
    quotes[].idx is matched by value, not position."""
    assert hc.parse_citation_markers("Claim [1].") == [1]


# ---------------------------------------------------------------------------
# get_cited_quotes
# ---------------------------------------------------------------------------


def _quotes():
    return [
        {"idx": 1, "docid": "d1", "text": "quote one text"},
        {"idx": 2, "docid": "d2", "text": "quote two text"},
        {"idx": 3, "docid": "d3", "text": "quote three text"},
    ]


def test_get_cited_quotes_returns_matching_quotes():
    result = hc.get_cited_quotes([1, 3], _quotes())
    assert [q["idx"] for q in result] == [1, 3]
    assert [q["text"] for q in result] == ["quote one text", "quote three text"]


def test_get_cited_quotes_drops_dangling_markers():
    """A marker referencing an idx not present in this row's quotes (a
    dangling citation, observed in the real HAGRID file) is silently
    dropped rather than raising."""
    result = hc.get_cited_quotes([2, 99], _quotes())
    assert [q["idx"] for q in result] == [2]


def test_get_cited_quotes_all_dangling_returns_empty():
    assert hc.get_cited_quotes([42, 99], _quotes()) == []


def test_get_cited_quotes_empty_markers_returns_empty():
    assert hc.get_cited_quotes([], _quotes()) == []


# ---------------------------------------------------------------------------
# sentence_label
# ---------------------------------------------------------------------------


def test_sentence_label_present_positive():
    assert hc.sentence_label({"text": "x", "attributable": 1}) == 1


def test_sentence_label_present_negative():
    assert hc.sentence_label({"text": "x", "attributable": 0}) == 0


def test_sentence_label_missing_field_is_none():
    assert hc.sentence_label({"text": "x"}) is None


def test_sentence_label_none_value_is_none():
    assert hc.sentence_label({"text": "x", "attributable": None}) is None


def test_sentence_label_non_binary_value_is_none():
    assert hc.sentence_label({"text": "x", "attributable": 2}) is None


# ---------------------------------------------------------------------------
# extract_items -- the sentence -> (claim, cited passages, label) mapping
# ---------------------------------------------------------------------------


def _toy_row(query_id=1, quotes=None, sentences=None):
    return {
        "query_id": query_id,
        "query": "a query",
        "quotes": quotes if quotes is not None else _quotes(),
        "answers": [
            {
                "answer": "irrelevant full-answer text",
                "answer_type": "long",
                "informative": 1,
                "attributable": 1,
                "sentences": sentences,
            }
        ],
    }


def test_extract_items_single_citation_labelled_sentence():
    row = _toy_row(sentences=[
        {"text": "A supported claim [1].", "index": 0, "attributable": 1, "informative": 1},
    ])
    result = hc.extract_items([row])
    assert result["counts"]["n_usable"] == 1
    item = result["items"][0]
    assert item["claim"] == "A supported claim [1]."
    assert item["cited_idxs"] == [1]
    assert item["quote_texts"] == ["quote one text"]
    assert item["label"] == 1
    assert item["n_cited"] == 1


def test_extract_items_multi_citation_sentence():
    row = _toy_row(sentences=[
        {"text": "Multiple sources support this [1, 2].", "index": 0, "attributable": 0, "informative": 1},
    ])
    result = hc.extract_items([row])
    item = result["items"][0]
    assert item["cited_idxs"] == [1, 2]
    assert item["quote_texts"] == ["quote one text", "quote two text"]
    assert item["n_cited"] == 2
    assert item["label"] == 0


def test_extract_items_drops_unlabelled_sentence():
    row = _toy_row(sentences=[
        {"text": "No label here [1].", "index": 0, "informative": 1},
    ])
    result = hc.extract_items([row])
    assert result["counts"]["n_usable"] == 0
    assert result["counts"]["n_unlabelled"] == 1
    assert result["items"] == []


def test_extract_items_drops_sentence_with_no_citation_marker():
    row = _toy_row(sentences=[
        {"text": "No brackets in this sentence at all.", "index": 0, "attributable": 1, "informative": 1},
    ])
    result = hc.extract_items([row])
    assert result["counts"]["n_usable"] == 0
    assert result["counts"]["n_zero_markers"] == 1


def test_extract_items_drops_sentence_with_only_dangling_citation():
    row = _toy_row(sentences=[
        {"text": "Cites something absent [99].", "index": 0, "attributable": 1, "informative": 1},
    ])
    result = hc.extract_items([row])
    assert result["counts"]["n_usable"] == 0
    assert result["counts"]["n_all_dangling"] == 1


def test_extract_items_counts_add_up_across_multiple_rows():
    rows = [
        _toy_row(query_id=1, sentences=[
            {"text": "Usable [1].", "index": 0, "attributable": 1, "informative": 1},
            {"text": "Unlabelled [1].", "index": 1, "informative": 1},
        ]),
        _toy_row(query_id=2, sentences=[
            {"text": "No marker at all.", "index": 0, "attributable": 0, "informative": 1},
            {"text": "Dangling [123].", "index": 1, "attributable": 1, "informative": 1},
        ]),
    ]
    result = hc.extract_items(rows)
    counts = result["counts"]
    assert counts["n_sentences_total"] == 4
    assert counts["n_unlabelled"] == 1
    assert counts["n_zero_markers"] == 1
    assert counts["n_all_dangling"] == 1
    assert counts["n_usable"] == 1
    assert counts["n_labelled"] == 3


# ---------------------------------------------------------------------------
# f1_at_threshold
# ---------------------------------------------------------------------------


def test_f1_at_threshold_perfect_predictions():
    labels = np.array([1, 1, 0, 0])
    preds = np.array([1, 1, 0, 0])
    stats = hc.f1_at_threshold(labels, preds)
    assert stats == {
        "tp": 2, "fp": 0, "fn": 0, "tn": 2,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
    }


def test_f1_at_threshold_no_positive_predictions_gives_zero_not_nan():
    labels = np.array([1, 0, 0])
    preds = np.array([0, 0, 0])
    stats = hc.f1_at_threshold(labels, preds)
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["f1"] == 0.0
    assert not any(np.isnan(v) for v in stats.values() if isinstance(v, float))


def test_f1_at_threshold_no_positive_labels_gives_zero_recall_not_nan():
    labels = np.array([0, 0, 0])
    preds = np.array([1, 0, 0])
    stats = hc.f1_at_threshold(labels, preds)
    assert stats["recall"] == 0.0
    assert stats["precision"] == 0.0


# ---------------------------------------------------------------------------
# threshold_sweep / best_tau_by_f1 / f1_at_tau
# ---------------------------------------------------------------------------


def test_threshold_sweep_produces_one_row_per_grid_point():
    stat = np.array([0.9, 0.7, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0])
    grid = np.array([0.0, 0.5, 1.0])
    sweep = hc.threshold_sweep(stat, labels, grid)
    assert len(sweep) == 3
    assert [row["tau"] for row in sweep] == [0.0, 0.5, 1.0]


def test_threshold_sweep_perfect_separation_has_f1_one_at_correct_tau():
    stat = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    grid = np.linspace(0.0, 1.0, 11)
    sweep = hc.threshold_sweep(stat, labels, grid)
    best = hc.best_tau_by_f1(sweep)
    assert best["f1"] == pytest.approx(1.0)


def test_best_tau_by_f1_breaks_ties_toward_0_5():
    sweep = [
        {"tau": 0.1, "f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.8},
        {"tau": 0.6, "f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.8},
        {"tau": 0.9, "f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.5},
    ]
    best = hc.best_tau_by_f1(sweep)
    assert best["tau"] == 0.6


def test_f1_at_tau_finds_exact_grid_point():
    sweep = hc.threshold_sweep(np.array([0.9, 0.1]), np.array([1, 0]), np.linspace(0, 1, 101))
    row = hc.f1_at_tau(sweep, 0.21)
    assert row["tau"] == pytest.approx(0.21)


def test_f1_at_tau_finds_nearest_when_no_exact_match():
    sweep = [
        {"tau": 0.2, "f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.5},
        {"tau": 0.4, "f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.8},
    ]
    row = hc.f1_at_tau(sweep, 0.39)
    assert row["tau"] == 0.4


# ---------------------------------------------------------------------------
# cross_dataset_transfer -- synthetic sweeps standing in for HAGRID/AttrEval
# ---------------------------------------------------------------------------


def _synthetic_sweep(best_tau: float, best_f1: float, f1_at_0_5: float):
    """A minimal fake sweep: peak F1 at `best_tau`, a specific value at 0.5,
    and 0 elsewhere, sufficient to drive best_tau_by_f1/f1_at_tau."""
    grid = sorted({0.0, 0.5, best_tau, 1.0})
    rows = []
    for tau in grid:
        if abs(tau - best_tau) < 1e-9:
            f1 = best_f1
        elif abs(tau - 0.5) < 1e-9:
            f1 = f1_at_0_5
        else:
            f1 = 0.0
        rows.append({"tau": tau, "f1": f1, "precision": f1, "recall": f1, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": f1})
    return rows


def test_cross_dataset_transfer_reports_both_directions():
    hagrid_sweep = _synthetic_sweep(best_tau=0.4, best_f1=0.9, f1_at_0_5=0.7)
    attreval_sweep = _synthetic_sweep(best_tau=0.21, best_f1=0.8, f1_at_0_5=0.6)
    # Both sweeps must share the queried tau points for f1_at_tau to find them;
    # extend each grid with the other's optimum at a distinct F1, enough to
    # exercise the cross-lookup wiring.
    hagrid_sweep.append({"tau": 0.21, "f1": 0.3, "precision": 0.3, "recall": 0.3, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.3})
    attreval_sweep.append({"tau": 0.4, "f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.5})

    result = hc.cross_dataset_transfer(hagrid_sweep, attreval_sweep)
    assert result["hagrid_own_best"]["tau"] == pytest.approx(0.4)
    assert result["hagrid_own_best"]["f1"] == pytest.approx(0.9)
    assert result["attreval_own_best"]["tau"] == pytest.approx(0.21)
    assert result["attreval_own_best"]["f1"] == pytest.approx(0.8)
    # HAGRID evaluated at AttrEval's optimal tau (0.21) -> the row we injected at 0.3.
    assert result["hagrid_at_attreval_tau"]["f1"] == pytest.approx(0.3)
    # AttrEval evaluated at HAGRID's optimal tau (0.4) -> the row we injected at 0.5.
    assert result["attreval_at_hagrid_tau"]["f1"] == pytest.approx(0.5)


def test_cross_dataset_transfer_identical_sweeps_agree_perfectly():
    """When both datasets are literally the same sweep, cross-evaluating at
    the other's optimum must reproduce that dataset's own best -- a sanity
    check that the transfer computation does not introduce spurious loss."""
    sweep = _synthetic_sweep(best_tau=0.3, best_f1=0.85, f1_at_0_5=0.6)
    result = hc.cross_dataset_transfer(sweep, sweep)
    assert result["hagrid_at_attreval_tau"]["f1"] == pytest.approx(result["hagrid_own_best"]["f1"])
    assert result["attreval_at_hagrid_tau"]["f1"] == pytest.approx(result["attreval_own_best"]["f1"])


# ---------------------------------------------------------------------------
# run_at_shipped_threshold / run_threshold_sweep wiring (numeric, offline)
# ---------------------------------------------------------------------------


def test_run_at_shipped_threshold_uses_given_tau():
    stat = np.array([0.6, 0.4])
    labels = np.array([1, 0])
    result = hc.run_at_shipped_threshold(stat, labels, tau=0.5)
    assert result["tau"] == 0.5
    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["f1"] == pytest.approx(1.0)


def test_run_threshold_sweep_includes_default_row():
    stat = np.array([0.9, 0.1, 0.6, 0.4])
    labels = np.array([1, 0, 1, 0])
    result = hc.run_threshold_sweep(stat, labels, n_grid=11)
    assert result["at_default_0_5"]["tau"] == pytest.approx(0.5)
    assert "best" in result
    assert len(result["grid"]) == 11


# ---------------------------------------------------------------------------
# run_single_vs_multi_citation
# ---------------------------------------------------------------------------


def test_run_single_vs_multi_citation_splits_correctly():
    items = [
        {"n_cited": 1}, {"n_cited": 1}, {"n_cited": 2}, {"n_cited": 3},
    ]
    stat = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    result = hc.run_single_vs_multi_citation(items, stat, labels, n_boot=50, seed=0)
    assert result["single_citation"]["n"] == 2
    assert result["multi_citation"]["n"] == 2
    assert result["single_citation"]["roc_auc"] is not None
    assert result["multi_citation"]["roc_auc"] is not None


def test_run_single_vs_multi_citation_handles_degenerate_subset():
    """A subset with only one class present (e.g. no multi-citation items,
    or all of one label) must not raise -- roc_auc is reported as None."""
    items = [{"n_cited": 1}, {"n_cited": 1}]
    stat = np.array([0.9, 0.8])
    labels = np.array([1, 1])  # single class -> degenerate
    result = hc.run_single_vs_multi_citation(items, stat, labels, n_boot=50, seed=0)
    assert result["single_citation"]["roc_auc"] is None
    assert result["multi_citation"]["n"] == 0
    assert result["multi_citation"]["roc_auc"] is None


# ---------------------------------------------------------------------------
# load_attreval_sweep -- missing-file handling (no network involved)
# ---------------------------------------------------------------------------


def test_load_attreval_sweep_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "ATTREVAL_RESULTS_PATH", tmp_path / "does_not_exist.json")
    assert hc.load_attreval_sweep() is None
