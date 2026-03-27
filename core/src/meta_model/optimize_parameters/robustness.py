from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.src.meta_model.optimize_parameters.config import DATE_COLUMN


@dataclass(frozen=True)
class TrainWindow:
    label: str
    train_indices: np.ndarray
    coverage_fraction: float


def build_train_windows(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    random_seed: int,
    recent_tail_fraction: float,
    random_window_count: int,
    random_window_min_fraction: float,
) -> list[TrainWindow]:
    if train_indices.size == 0:
        raise ValueError("train_indices must not be empty.")

    train_frame = data.iloc[train_indices]
    unique_dates = pd.Index(pd.to_datetime(train_frame[DATE_COLUMN]).drop_duplicates().sort_values())
    if unique_dates.empty:
        raise ValueError("No train dates available to build robustness windows.")

    windows: list[TrainWindow] = [
        TrainWindow(
            label="full",
            train_indices=np.asarray(train_indices, dtype=np.int64),
            coverage_fraction=1.0,
        ),
    ]

    tail_date_count = max(1, int(np.ceil(len(unique_dates) * recent_tail_fraction)))
    tail_dates = set(unique_dates[-tail_date_count:].tolist())
    tail_indices = train_frame.index[train_frame[DATE_COLUMN].isin(tail_dates)].to_numpy(dtype=np.int64)
    if 0 < tail_indices.size < train_indices.size:
        windows.append(
            TrainWindow(
                label="recent_tail",
                train_indices=tail_indices,
                coverage_fraction=float(tail_indices.size / train_indices.size),
            ),
        )

    rng = np.random.default_rng(random_seed)
    date_values = unique_dates.tolist()
    for window_index in range(random_window_count):
        sampled_fraction = float(rng.uniform(random_window_min_fraction, 0.95))
        sampled_date_count = max(1, int(np.ceil(len(date_values) * sampled_fraction)))
        max_start = max(0, len(date_values) - sampled_date_count)
        start_index = int(rng.integers(0, max_start + 1))
        sampled_dates = set(date_values[start_index: start_index + sampled_date_count])
        sampled_indices = train_frame.index[
            train_frame[DATE_COLUMN].isin(sampled_dates)
        ].to_numpy(dtype=np.int64)
        if 0 < sampled_indices.size < train_indices.size:
            windows.append(
                TrainWindow(
                    label=f"random_window_{window_index + 1}",
                    train_indices=sampled_indices,
                    coverage_fraction=float(sampled_indices.size / train_indices.size),
                ),
            )

    deduplicated_windows: list[TrainWindow] = []
    seen_signatures: set[tuple[int, ...]] = set()
    for window in windows:
        signature = tuple(window.train_indices.tolist())
        if signature in seen_signatures:
            continue
        deduplicated_windows.append(window)
        seen_signatures.add(signature)
    return deduplicated_windows


def compute_complexity_penalty(
    *,
    max_depth: int,
    average_best_iteration: float,
    boost_rounds: int,
    penalty_alpha: float,
) -> float:
    if boost_rounds <= 0:
        raise ValueError("boost_rounds must be strictly positive.")
    normalized_depth = max_depth / 10.0
    normalized_iteration = min(max(average_best_iteration / boost_rounds, 0.0), 1.0)
    return penalty_alpha * (0.5 * normalized_depth + 0.5 * normalized_iteration)


def _compute_equal_weight_objective_base(
    rmse: np.ndarray,
    weights: np.ndarray,
    window_std: np.ndarray,
    *,
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
) -> tuple[float, float, float, float]:
    del weights
    mean_rmse = float(rmse.mean())
    std_rmse = float(np.sqrt(np.mean(np.square(rmse - mean_rmse))))
    window_std_mean = float(window_std.mean())
    objective_base_score = (
        mean_rmse
        + stability_penalty_alpha * std_rmse
        + train_window_stability_alpha * window_std_mean
    )
    return (
        mean_rmse,
        std_rmse,
        window_std_mean,
        float(objective_base_score),
    )


def _bootstrap_objective_standard_error(
    *,
    rmse: np.ndarray,
    weights: np.ndarray,
    window_std: np.ndarray,
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
    bootstrap_samples: int,
    random_seed: int,
) -> float:
    if rmse.size <= 1 or bootstrap_samples <= 1:
        return 0.0

    rng = np.random.default_rng(random_seed)
    fold_indices = np.arange(rmse.size, dtype=np.int64)
    bootstrap_scores = np.empty(bootstrap_samples, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        sampled_indices = rng.choice(fold_indices, size=rmse.size, replace=True)
        _, _, _, bootstrap_score = _compute_equal_weight_objective_base(
            rmse[sampled_indices],
            weights[sampled_indices],
            window_std[sampled_indices],
            stability_penalty_alpha=stability_penalty_alpha,
            train_window_stability_alpha=train_window_stability_alpha,
        )
        bootstrap_scores[sample_index] = bootstrap_score
    return float(bootstrap_scores.std(ddof=1))


def aggregate_robust_objective(
    *,
    fold_rmse: list[float],
    fold_weights: list[float],
    fold_window_std: list[float],
    stability_penalty_alpha: float,
    train_window_stability_alpha: float,
    complexity_penalty: float,
    bootstrap_samples: int = 1024,
    random_seed: int = 7,
) -> dict[str, float]:
    rmse = np.asarray(fold_rmse, dtype=np.float64)
    weights = np.asarray(fold_weights, dtype=np.float64)
    window_std = np.asarray(fold_window_std, dtype=np.float64)
    if rmse.size == 0:
        raise ValueError("At least one fold metric is required to aggregate the robust objective.")
    if rmse.size != weights.size or rmse.size != window_std.size:
        raise ValueError("fold_rmse, fold_weights, and fold_window_std must have the same length.")
    if np.any(weights <= 0.0):
        raise ValueError("All fold weights must be strictly positive.")
    (
        mean_rmse,
        std_rmse,
        window_std_mean,
        objective_base_score,
    ) = _compute_equal_weight_objective_base(
        rmse,
        weights,
        window_std,
        stability_penalty_alpha=stability_penalty_alpha,
        train_window_stability_alpha=train_window_stability_alpha,
    )
    objective_standard_error = _bootstrap_objective_standard_error(
        rmse=rmse,
        weights=weights,
        window_std=window_std,
        stability_penalty_alpha=stability_penalty_alpha,
        train_window_stability_alpha=train_window_stability_alpha,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    objective_score = objective_base_score + complexity_penalty
    return {
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "window_std_mean": window_std_mean,
        "objective_base_score": float(objective_base_score),
        "objective_standard_error": objective_standard_error,
        "objective_score": float(objective_score),
        "normalized_weight_sum": 1.0,
    }
