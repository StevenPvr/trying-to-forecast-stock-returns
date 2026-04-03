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


def aggregate_fold_rank_ic(
    fold_rank_ic: list[float],
    fold_window_std: list[float],
    *,
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
    complexity_penalty: float,
    objective_standard_error: float = 0.0,
) -> dict[str, float]:
    if not fold_rank_ic:
        raise ValueError("At least one fold rank IC is required.")
    if len(fold_rank_ic) != len(fold_window_std):
        raise ValueError("fold_rank_ic and fold_window_std must have the same length.")

    rank_ic = np.asarray(fold_rank_ic, dtype=np.float64)
    window_std = np.asarray(fold_window_std, dtype=np.float64)
    mean_rank_ic = float(rank_ic.mean())
    centered = rank_ic - mean_rank_ic
    rank_ic_std = float(math.sqrt(float(np.mean(np.square(centered)))))
    window_std_mean = float(window_std.mean())
    objective_base_score = (
        -mean_rank_ic
        + stability_penalty_alpha * rank_ic_std
        + train_window_stability_alpha * window_std_mean
    )
    return {
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": rank_ic_std,
        "window_std_mean": window_std_mean,
        "objective_base_score": float(objective_base_score),
        "objective_standard_error": float(objective_standard_error),
        "objective_score": float(objective_base_score + complexity_penalty),
    }


def bootstrap_rank_ic_objective_standard_error(
    fold_rank_ic: list[float],
    fold_window_std: list[float],
    *,
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
    bootstrap_samples: int,
    random_seed: int,
) -> float:
    if bootstrap_samples <= 1 or len(fold_rank_ic) <= 1:
        return 0.0

    rank_ic = np.asarray(fold_rank_ic, dtype=np.float64)
    window_std = np.asarray(fold_window_std, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    indices = np.arange(rank_ic.size, dtype=np.int64)
    bootstrap_scores = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        sampled_indices = rng.choice(indices, size=rank_ic.size, replace=True)
        aggregate = aggregate_fold_rank_ic(
            rank_ic[sampled_indices].tolist(),
            window_std[sampled_indices].tolist(),
            stability_penalty_alpha=stability_penalty_alpha,
            train_window_stability_alpha=train_window_stability_alpha,
            complexity_penalty=0.0,
            objective_standard_error=0.0,
        )
        bootstrap_scores[sample_index] = aggregate["objective_base_score"]
    return float(bootstrap_scores.std(ddof=1))
