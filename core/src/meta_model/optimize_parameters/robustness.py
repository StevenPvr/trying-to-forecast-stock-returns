from __future__ import annotations

"""Multi-window robustness: complexity penalty and aggregation across train windows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.src.meta_model.optimize_parameters.config import (
    COMPLEXITY_MAX_DEPTH_NORMALIZER,
    DATE_COLUMN,
)


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
    max_depth_normalizer: int = COMPLEXITY_MAX_DEPTH_NORMALIZER,
) -> float:
    if boost_rounds <= 0:
        raise ValueError("boost_rounds must be strictly positive.")
    normalized_depth = max_depth / float(max_depth_normalizer)
    normalized_iteration = min(max(average_best_iteration / boost_rounds, 0.0), 1.0)
    return penalty_alpha * (0.5 * normalized_depth + 0.5 * normalized_iteration)
