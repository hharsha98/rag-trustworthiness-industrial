"""Tests for experiments/11_attreval_validation.py -- pure functions only.

Fast: no network, no model downloads. `label_to_binary`, `add_binary_label`,
`row_to_attribution_inputs`, `f1_at_threshold`, `threshold_sweep` and
`best_tau_by_f1` are exercised here, plus the row-to-attribution()
correspondence with a fake NLI stub. Everything that needs the
AttrEval-GenSearch download or the real NLI model lives behind `main()` /
`load_attreval()` / `score_items()` (which are never called here);
`verify_correspondence()` IS exercised here, but only against `FakeNLI` --
never the real model.

The module under test is a standalone script (like experiments/03, 08, 10,
etc.), not a package, so it is loaded by file path rather than imported by
name -- same pattern as tests/test_ragtruth_validation.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtrust.metrics.attribution import attribution
from ragtrust.metrics.nli import FakeNLI

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "experiments" / "11_attreval_validation.py"
_SPEC = importlib.util.spec_from_file_location("attreval_validation_module", _MODULE_PATH)
av = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = av
_SPEC.loader.exec_module(av)


# ---------------------------------------------------------------------------
# label_to_binary
# ---------------------------------------------------------------------------


def test_label_to_binary_attributable_is_one():
    assert av.label_to_binary("Attributable") == 1


def test_label_to_binary_extrapolatory_is_zero():
    assert av.label_to_binary("Extrapolatory") == 0


def test_label_to_binary_contradictory_is_zero():
    assert av.label_to_binary("Contradictory") == 0


def test_label_to_binary_extrapolatory_and_contradictory_collapse_to_same_value():
    """The spec's central mapping: attribution() only distinguishes
    supported/not-supported, so both non-Attributable labels must collapse
    to the identical binary value -- this is the Extrapolatory/Contradictory
    -> 0 collapse the honesty analysis (Analysis 5) exists to probe."""
    assert av.label_to_binary("Extrapolatory") == av.label_to_binary("Contradictory")


def test_label_to_binary_rejects_unknown_label():
    with pytest.raises(ValueError):
        av.label_to_binary("SomethingElse")


# ---------------------------------------------------------------------------
# add_binary_label
# ---------------------------------------------------------------------------


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query": ["q1", "q2", "q3", "q4"],
            "answer": ["a1", "a2", "a3", "a4"],
            "reference": ["r1", "r2", "r3", "r4"],
            "label": ["Attributable", "Extrapolatory", "Contradictory", "Attributable"],
        }
    )


def test_add_binary_label_all_three_classes():
    df = av.add_binary_label(_toy_df())
    assert df["supported"].tolist() == [1, 0, 0, 1]


def test_add_binary_label_does_not_mutate_input():
    original = _toy_df()
    av.add_binary_label(original)
    assert "supported" not in original.columns


# ---------------------------------------------------------------------------
# row_to_attribution_inputs
# ---------------------------------------------------------------------------


def test_row_to_attribution_inputs_maps_answer_reference_correctly():
    row = pd.Series({"query": "q", "answer": "The sky is blue.", "reference": "Skies are blue.", "label": "Attributable"})
    inputs = av.row_to_attribution_inputs(row)
    assert inputs["claims"] == ["The sky is blue."]
    assert inputs["passages"] == ["Skies are blue."]
    assert inputs["citations"] == {0: 0}


# ---------------------------------------------------------------------------
# Row -> attribution() correspondence (the core claim under test), using
# FakeNLI so this stays offline and model-free.
# ---------------------------------------------------------------------------


def test_attribution_precision_matches_thresholded_entailment_when_above_tau():
    nli = FakeNLI()
    row = pd.Series({"answer": "the sky is blue today", "reference": "the sky is blue today", "label": "Attributable"})
    inputs = av.row_to_attribution_inputs(row)
    tau = 0.5
    p_ent = nli.probs(row["reference"], row["answer"])["entailment"]
    assert p_ent >= tau  # identical strings -> Jaccard overlap 1.0
    result = attribution(inputs["claims"], inputs["citations"], inputs["passages"], nli, tau=tau)
    assert result.precision == 1.0
    assert result.precision == (1.0 if p_ent >= tau else 0.0)


def test_attribution_precision_matches_thresholded_entailment_when_below_tau():
    nli = FakeNLI()
    row = pd.Series({"answer": "completely unrelated content about spacecraft", "reference": "bananas are yellow fruit", "label": "Contradictory"})
    inputs = av.row_to_attribution_inputs(row)
    tau = 0.5
    p_ent = nli.probs(row["reference"], row["answer"])["entailment"]
    assert p_ent < tau  # near-zero token overlap
    result = attribution(inputs["claims"], inputs["citations"], inputs["passages"], nli, tau=tau)
    assert result.precision == 0.0
    assert result.precision == (1.0 if p_ent >= tau else 0.0)


def test_verify_correspondence_passes_on_fake_nli_rows():
    """verify_correspondence() must not raise when the correspondence holds
    -- exercised here against FakeNLI (offline) rather than the real model
    the script uses at runtime."""
    df = pd.DataFrame(
        {
            "answer": [
                "the sky is blue today",
                "completely unrelated content about spacecraft",
                "water boils at one hundred degrees",
            ],
            "reference": [
                "the sky is blue today",
                "bananas are yellow fruit",
                "water boils at one hundred degrees",
            ],
            "label": ["Attributable", "Contradictory", "Attributable"],
        }
    )
    av.verify_correspondence(df, FakeNLI(), tau=0.5, n_check=3)  # should not raise


def test_verify_correspondence_raises_when_correspondence_broken():
    """A stub NLI whose .probs() disagrees with what attribution() computes
    internally (e.g. non-symmetric premise/hypothesis handling) must trip
    the assertion, proving the check is not a no-op."""

    class _InconsistentNLI:
        """Returns a fixed entailment probability from .probs() but a
        different one from batch_probs(), so attribution()'s internal
        decision and verify_correspondence()'s external check disagree."""

        def probs(self, premise, hypothesis):
            return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}

        def batch_probs(self, pairs):
            return [{"entailment": 0.1, "neutral": 0.45, "contradiction": 0.45} for _ in pairs]

    df = pd.DataFrame({
        "answer": ["x"],
        "reference": ["y"],
        "label": ["Attributable"],
    })
    with pytest.raises(AssertionError):
        av.verify_correspondence(df, _InconsistentNLI(), tau=0.5, n_check=1)


# ---------------------------------------------------------------------------
# f1_at_threshold
# ---------------------------------------------------------------------------


def test_f1_at_threshold_perfect_predictions():
    labels = np.array([1, 1, 0, 0])
    preds = np.array([1, 1, 0, 0])
    stats = av.f1_at_threshold(labels, preds)
    assert stats == {
        "tp": 2, "fp": 0, "fn": 0, "tn": 2,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
    }


def test_f1_at_threshold_all_wrong():
    labels = np.array([1, 1, 0, 0])
    preds = np.array([0, 0, 1, 1])
    stats = av.f1_at_threshold(labels, preds)
    assert stats["tp"] == 0
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["f1"] == 0.0
    assert stats["accuracy"] == 0.0


def test_f1_at_threshold_no_positive_predictions_gives_zero_not_nan():
    labels = np.array([1, 0, 0])
    preds = np.array([0, 0, 0])
    stats = av.f1_at_threshold(labels, preds)
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["f1"] == 0.0
    assert not any(np.isnan(v) for v in stats.values() if isinstance(v, float))


def test_f1_at_threshold_no_positive_labels_gives_zero_recall_not_nan():
    labels = np.array([0, 0, 0])
    preds = np.array([1, 0, 0])
    stats = av.f1_at_threshold(labels, preds)
    assert stats["recall"] == 0.0
    assert stats["precision"] == 0.0  # the one positive prediction is a false positive


# ---------------------------------------------------------------------------
# threshold_sweep / best_tau_by_f1
# ---------------------------------------------------------------------------


def test_threshold_sweep_produces_one_row_per_grid_point():
    p_entail = np.array([0.9, 0.7, 0.3, 0.1])
    labels = np.array([1, 1, 0, 0])
    grid = np.array([0.0, 0.5, 1.0])
    sweep = av.threshold_sweep(p_entail, labels, grid)
    assert len(sweep) == 3
    assert [row["tau"] for row in sweep] == [0.0, 0.5, 1.0]


def test_threshold_sweep_perfect_separation_has_f1_one_at_correct_tau():
    p_entail = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    grid = np.linspace(0.0, 1.0, 11)
    sweep = av.threshold_sweep(p_entail, labels, grid)
    best = av.best_tau_by_f1(sweep)
    assert best["f1"] == pytest.approx(1.0)


def test_best_tau_by_f1_breaks_ties_toward_0_5():
    """When multiple tau achieve the same best F1, the row closest to 0.5
    wins -- deterministic and consistent with the shipped default being the
    natural tie-break anchor."""
    sweep = [
        {"tau": 0.1, "f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.8},
        {"tau": 0.6, "f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.8},
        {"tau": 0.9, "f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.5},
    ]
    best = av.best_tau_by_f1(sweep)
    assert best["tau"] == 0.6  # |0.6 - 0.5| = 0.1 < |0.1 - 0.5| = 0.4


def test_best_tau_by_f1_single_row():
    sweep = [{"tau": 0.5, "f1": 0.7, "precision": 0.7, "recall": 0.7, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "accuracy": 0.7}]
    assert av.best_tau_by_f1(sweep)["tau"] == 0.5


# ---------------------------------------------------------------------------
# run_at_shipped_threshold / run_threshold_sweep wiring (numeric, offline)
# ---------------------------------------------------------------------------


def test_run_at_shipped_threshold_uses_given_tau():
    p_entail = np.array([0.6, 0.4])
    labels = np.array([1, 0])
    result = av.run_at_shipped_threshold(p_entail, labels, tau=0.5)
    assert result["tau"] == 0.5
    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["f1"] == pytest.approx(1.0)


def test_run_threshold_sweep_includes_default_row():
    p_entail = np.array([0.9, 0.1, 0.6, 0.4])
    labels = np.array([1, 0, 1, 0])
    result = av.run_threshold_sweep(p_entail, labels, n_grid=11)
    assert result["at_default_0_5"]["tau"] == pytest.approx(0.5)
    assert "best" in result
    assert len(result["grid"]) == 11
