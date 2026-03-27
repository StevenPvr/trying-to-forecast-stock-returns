from __future__ import annotations

import math

import numpy as np


def aggregate_fold_rmse(
    fold_rmse: list[float],
    fold_weights: list[float],
    stability_penalty_alpha: float,
) -> dict[str, float]:
    if len(fold_rmse) != len(fold_weights):
        raise ValueError("fold_rmse and fold_weights must have the same length.")
    if not fold_rmse:
        raise ValueError("At least one fold RMSE is required.")

    rmse = np.asarray(fold_rmse, dtype=np.float64)
    weights = np.asarray(fold_weights, dtype=np.float64)
    if np.any(weights <= 0.0):
        raise ValueError("All fold weights must be strictly positive.")

    # Keep the argument for compatibility, but count every fold equally.
    equal_mean = float(rmse.mean())
    centered = rmse - equal_mean
    equal_variance = float(np.mean(np.square(centered)))
    equal_std = float(math.sqrt(equal_variance))
    objective_score = equal_mean + stability_penalty_alpha * equal_std
    return {
        "mean_rmse": equal_mean,
        "std_rmse": equal_std,
        "objective_score": objective_score,
    }
