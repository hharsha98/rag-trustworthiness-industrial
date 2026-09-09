"""Tests for experiments/09_gate_calibration.py's pure functions. Fast, no network,
no model downloads -- anything that needs the real BEIR download or real model
weights is marked @pytest.mark.slow (skipped by default; see pyproject.toml's
`addopts = "-m \"not slow\""`).

`experiments/09_gate_calibration.py` is not an importable package module (its
filename starts with a digit), so it is loaded here via importlib -- the same
approach tests/test_beir_loader.py uses for experiment 08.
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "09_gate_calibration.py"
_spec = importlib.util.spec_from_file_location("gate_calibration", _MODULE_PATH)
gate_calibration = importlib.util.module_from_spec(_spec)
sys.modules["gate_calibration"] = gate_calibration
_spec.loader.exec_module(gate_calibration)


# --------------------------------------------------------------------------- build_threshold_grid


def test_build_threshold_grid_covers_range_inclusive():
    grid = gate_calibration.build_threshold_grid(0.0, 0.6, 0.01)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(0.6)
    assert len(grid) == 61


# --------------------------------------------------------------------------- retention_rate


def test_retention_rate_is_one_at_threshold_zero():
    pos_scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert gate_calibration.retention_rate(pos_scores, 0.0) == pytest.approx(1.0)


def test_retention_rate_is_zero_above_max_observed_similarity():
    pos_scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert gate_calibration.retention_rate(pos_scores, 0.9 + 1e-6) == 0.0


def test_retention_rate_hand_computed():
    pos_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # threshold 0.3 keeps 0.3, 0.4, 0.5 -> 3/5
    assert gate_calibration.retention_rate(pos_scores, 0.3) == pytest.approx(0.6)


# --------------------------------------------------------------------------- false_acceptance_rate


def test_false_acceptance_rate_hand_computed():
    neg_scores = np.array([0.0, 0.1, 0.2, 0.3])
    # threshold 0.2 admits 0.2, 0.3 -> 2/4
    assert gate_calibration.false_acceptance_rate(neg_scores, 0.2) == pytest.approx(0.5)


def test_false_acceptance_rate_zero_when_all_below_threshold():
    neg_scores = np.array([0.0, 0.05, 0.1])
    assert gate_calibration.false_acceptance_rate(neg_scores, 0.5) == 0.0


# --------------------------------------------------------------------------- sweep_thresholds monotonicity


def test_sweep_thresholds_retention_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    pos_scores = rng.uniform(0.0, 1.0, size=200)
    thresholds = gate_calibration.build_threshold_grid()
    table = gate_calibration.sweep_thresholds(pos_scores, {"tier": pos_scores}, thresholds)
    retentions = [row["retention"] for row in table]
    diffs = np.diff(retentions)
    assert np.all(diffs <= 1e-12), "retention must never increase as the threshold rises"


def test_sweep_thresholds_far_is_also_monotonically_non_increasing():
    rng = np.random.default_rng(1)
    pos_scores = rng.uniform(0.0, 1.0, size=50)
    neg_scores = rng.uniform(0.0, 1.0, size=200)
    thresholds = gate_calibration.build_threshold_grid()
    table = gate_calibration.sweep_thresholds(pos_scores, {"easy": neg_scores}, thresholds)
    fars = [row["far_easy"] for row in table]
    assert np.all(np.diff(fars) <= 1e-12)


def test_sweep_thresholds_retention_one_at_zero_and_zero_above_max():
    pos_scores = np.array([0.2, 0.4, 0.6, 0.8])
    thresholds = gate_calibration.build_threshold_grid(0.0, 1.0, 0.1)
    table = gate_calibration.sweep_thresholds(pos_scores, {"tier": pos_scores}, thresholds)
    assert table[0]["threshold"] == pytest.approx(0.0)
    assert table[0]["retention"] == pytest.approx(1.0)
    assert table[-1]["threshold"] == pytest.approx(1.0)
    assert table[-1]["retention"] == 0.0  # 1.0 > max observed similarity 0.8


# --------------------------------------------------------------------------- recommended_threshold


def test_recommended_threshold_hand_built_synthetic_distribution():
    # Positives uniformly spaced 0.0..0.99 in steps of 0.01 (100 points).
    pos_scores = np.round(np.arange(0, 1.0, 0.01), 2)
    thresholds = gate_calibration.build_threshold_grid(0.0, 0.99, 0.01)
    # Retention >= 0.99 requires keeping at least 99 of the 100 points, i.e.
    # threshold <= the 2nd-smallest value (0.01), since only 1 point (0.0) may
    # be dropped at threshold 0.01, and retention is exactly 0.99 there.
    t = gate_calibration.recommended_threshold(pos_scores, thresholds, 0.99)
    assert t == pytest.approx(0.01)
    assert gate_calibration.retention_rate(pos_scores, t) >= 0.99


def test_recommended_threshold_is_the_largest_meeting_the_floor():
    pos_scores = np.array([0.1] * 90 + [0.5] * 10)  # 90% at 0.1, 10% at 0.5
    thresholds = gate_calibration.build_threshold_grid(0.0, 0.6, 0.1)
    # A 90% retention floor should allow raising the threshold up to (but not
    # above) 0.1, since anything above 0.1 drops the 90% mass at 0.1.
    t = gate_calibration.recommended_threshold(pos_scores, thresholds, 0.90)
    assert t == pytest.approx(0.1)


def test_recommended_threshold_falls_back_to_minimum_when_floor_unreachable():
    # One positive (-0.1) sits below every threshold in the grid, so no
    # threshold on this grid can ever retain 100% of positives -- the floor
    # is unreachable, and the function must fall back to the grid's minimum
    # rather than raising.
    pos_scores = np.array([-0.1, 0.2, 0.3])
    thresholds = gate_calibration.build_threshold_grid(0.0, 0.6, 0.1)
    t = gate_calibration.recommended_threshold(pos_scores, thresholds, 1.0)
    assert t == pytest.approx(thresholds.min())


# --------------------------------------------------------------------------- complete overlap (no separation)


def test_identical_positive_and_negative_distributions_behaves_sanely():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.0, 1.0, size=300)
    pos_scores = scores
    neg_scores = scores.copy()  # complete overlap: no separation possible

    thresholds = gate_calibration.build_threshold_grid()
    t = gate_calibration.recommended_threshold(pos_scores, thresholds, 0.99)
    assert np.isfinite(t)
    assert thresholds.min() <= t <= thresholds.max()

    retention = gate_calibration.retention_rate(pos_scores, t)
    far = gate_calibration.false_acceptance_rate(neg_scores, t)
    # Identical distributions -> at any given threshold, retention and FAR must
    # be numerically identical (same underlying values, same comparison).
    assert retention == pytest.approx(far)

    # ROC-AUC on totally overlapping distributions should sit at chance (~0.5),
    # not crash and not report strong separation.
    from ragtrust.validation.stats import roc_auc

    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    all_scores = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc(all_scores, labels)
    assert 0.4 <= auc <= 0.6


# --------------------------------------------------------------------------- max_similarity_to_corpus


def test_max_similarity_to_corpus_picks_highest_cosine_and_clamps_to_zero_one():
    # Two normalized query vectors, three normalized corpus vectors.
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="float32")
    queries = np.array([[1.0, 0.0], [0.0, -1.0]], dtype="float32")
    sims = gate_calibration.max_similarity_to_corpus(queries, corpus)
    # query 0 matches corpus[0] exactly (cosine 1.0)
    assert sims[0] == pytest.approx(1.0)
    # query 1 is closest (least negative) to corpus[2] at cosine 0.0 (orthogonal),
    # since its best cosine against every corpus vector is <= 0 -- clamped to 0.
    assert sims[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- slow / network integration check


@pytest.mark.slow
def test_full_run_smoke_produces_all_artefacts(tmp_path, monkeypatch):
    """End-to-end smoke test with a tiny sample size -- needs network and real
    model weights, so it is marked slow and skipped by default."""
    monkeypatch.setattr(gate_calibration, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["09_gate_calibration.py", "--negatives", "5"])
    rc = gate_calibration.main()
    assert rc == 0
    assert (tmp_path / "gate_calibration.md").exists()
    assert (tmp_path / "gate_calibration.json").exists()
    assert (tmp_path / "gate_calibration.png").exists()
