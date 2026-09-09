"""Metric-validation harness -- METRICS.md Part III.

This package turns "the metric is defined this way" into "we measured how well
it works": label-preserving/label-flipping perturbation operators
(`perturbations.py`) and statistics for discrimination quality (`stats.py`),
consumed by the benchmark-validation experiments (e.g.
`experiments/10_ragtruth_validation.py`) and `experiments/04_weight_sensitivity.py`.
"""
