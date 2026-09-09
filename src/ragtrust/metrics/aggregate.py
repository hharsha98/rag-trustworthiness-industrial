"""Aggregation -- METRICS.md Part II.5.

T_arith = sum_i w_i * m_i                              (compensatory)
T_geom  = prod_i m_i ** w_i,  sum_i w_i = 1, w_i > 0   (non-compensatory)

Weights for the geometric aggregate must be strictly positive so 0**0 never
arises; a metric to be ignored is dropped from the product entirely, not
given weight 0. Computed in log-space for numerical stability, with an
explicit short-circuit for m_i == 0 so log(0) is never evaluated.
"""
import math

import numpy as np
import pandas as pd


def aggregate_arithmetic(metrics: dict, weights: dict) -> float:
    keys = [k for k in weights if k in metrics]
    # Renormalise over the keys actually present, exactly like aggregate_geometric
    # does below. Without this, a dropped metric leaves T_arith holding less than
    # the full weight mass (e.g. dropping conciseness at weight 0.2 caps T_arith at
    # 0.8) while T_geom renormalises to 1.0, which can make T_geom > T_arith and
    # violate Proposition 3 (T_geom <= T_arith, proved via weighted AM-GM for
    # weights that sum to 1). Sharing this renormalised weight basis between both
    # aggregates keeps weighted AM-GM -- and therefore Proposition 3 -- valid
    # whenever a metric is dropped from aggregation (e.g. conciseness undefined
    # for < 2 claims).
    total_weight = sum(weights[k] for k in keys)
    normalized_weights = {k: weights[k] / total_weight for k in keys}
    return float(sum(normalized_weights[k] * metrics[k] for k in keys))


def aggregate_geometric(metrics: dict, weights: dict) -> float:
    if any(w <= 0 for w in weights.values()):
        raise ValueError("all weights must be > 0 for geometric aggregation")

    keys = [k for k in weights if k in metrics]
    total_weight = sum(weights[k] for k in keys)
    normalized_weights = {k: weights[k] / total_weight for k in keys}

    for k in keys:
        if metrics[k] < 0:
            raise ValueError(f"metric {k!r} must be >= 0 for geometric aggregation")
        if metrics[k] == 0:
            return 0.0

    log_sum = sum(normalized_weights[k] * math.log(metrics[k]) for k in keys)
    return float(math.exp(log_sum))


def weight_sensitivity(metric_rows: list, n_samples: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Sample w ~ Dirichlet(1,...,1) over the weight simplex and report the
    resulting arithmetic and geometric aggregates for each row of metrics."""
    if not metric_rows:
        return pd.DataFrame(columns=["row", "arithmetic", "geometric"])

    rng = np.random.default_rng(seed)
    keys = list(metric_rows[0].keys())
    weight_samples = rng.dirichlet(np.ones(len(keys)), size=n_samples)  # (n_samples, n_keys)

    records = []
    for row_idx, row in enumerate(metric_rows):
        m = np.array([row[key] for key in keys], dtype=float)
        arithmetic = weight_samples @ m

        any_nonpositive = bool(np.any(m <= 0))
        if any_nonpositive:
            geometric = np.zeros(n_samples)
        else:
            log_m = np.log(m)
            geometric = np.exp(weight_samples @ log_m)

        for s in range(n_samples):
            record = {
                "row": row_idx,
                "arithmetic": float(arithmetic[s]),
                "geometric": float(geometric[s]),
            }
            for i, key in enumerate(keys):
                record[f"w_{key}"] = float(weight_samples[s, i])
            records.append(record)

    return pd.DataFrame(records)
