from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from core.src.meta_model.feature_corr_pca.config import (
    CORRELATION_BATCH_ROWS,
    DEFAULT_CORRELATION_THRESHOLD,
    MAX_IN_FLIGHT_FUTURES_MULTIPLIER,
    MIN_SHARED_ROWS_FOR_CORRELATION,
    TRAIN_END_DATE,
    resolve_max_workers,
)
from core.src.meta_model.feature_corr_pca.dataset_utils import (
    iter_dataset_batches,
    record_batch_to_frame,
)

LOGGER: logging.Logger = logging.getLogger(__name__)
CORRELATION_PROGRESS_EVERY_BATCHES: int = 25


def find_correlated_feature_groups(
    data: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> list[list[str]]:
    if not feature_columns:
        return []
    corr = data.loc[:, feature_columns].corr().abs()
    return find_correlated_feature_groups_from_matrix(
        feature_columns=feature_columns,
        corr_matrix=corr.to_numpy(dtype=np.float64),
        threshold=threshold,
    )


def find_correlated_feature_groups_from_matrix(
    feature_columns: list[str],
    corr_matrix: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {column: set() for column in feature_columns}
    for left_index, left_column in enumerate(feature_columns):
        row = corr_matrix[left_index]
        for right_index in range(left_index + 1, len(feature_columns)):
            if np.isfinite(row[right_index]) and float(abs(row[right_index])) >= threshold:
                right_column = feature_columns[right_index]
                adjacency[left_column].add(right_column)
                adjacency[right_column].add(left_column)

    groups: list[list[str]] = []
    visited: set[str] = set()
    for column in feature_columns:
        if column in visited or not adjacency[column]:
            continue
        stack = [column]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited))
        if len(component) >= 2:
            groups.append(sorted(component))
    return sorted(groups, key=lambda group: (group[0], len(group)))


def _empty_correlation_stats(feature_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    shape = (feature_count, feature_count)
    return (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        0,
    )


def _compute_batch_correlation_stats_from_values(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    valid = np.isfinite(values)
    if not bool(valid.any()):
        return _empty_correlation_stats(values.shape[1])
    filled = np.where(valid, values, 0.0)
    valid_float = valid.astype(np.float64, copy=False)
    squared = filled * filled
    return (
        valid_float.T @ valid_float,
        filled.T @ valid_float,
        squared.T @ valid_float,
        filled.T @ filled,
        int(values.shape[0]),
    )


def _accumulate_correlation_stats(
    target: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
    partial: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    count, sum_pair, sumsq_pair, cross, processed_rows = target
    partial_count, partial_sum_pair, partial_sumsq_pair, partial_cross, partial_rows = partial
    count += partial_count
    sum_pair += partial_sum_pair
    sumsq_pair += partial_sumsq_pair
    cross += partial_cross
    return count, sum_pair, sumsq_pair, cross, processed_rows + partial_rows


def _compute_corr_matrix_from_stats(
    count: np.ndarray,
    sum_pair: np.ndarray,
    sumsq_pair: np.ndarray,
    cross: np.ndarray,
    min_shared_rows: int,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = np.divide(sum_pair, count, out=np.zeros_like(sum_pair), where=count > 0.0)
        mean_y = mean_x.T
        var_x = np.divide(sumsq_pair, count, out=np.zeros_like(sumsq_pair), where=count > 0.0) - (
            mean_x * mean_x
        )
        var_y = var_x.T
        cov = np.divide(cross, count, out=np.zeros_like(cross), where=count > 0.0) - (
            mean_x * mean_y
        )
        denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
        corr = np.divide(
            cov,
            denom,
            out=np.zeros_like(cov),
            where=(count >= float(min_shared_rows)) & (denom > 0.0),
        )
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_pairwise_correlation_matrix_from_parquet(
    feature_parquet_path: Path,
    feature_columns: list[str],
    batch_size: int = CORRELATION_BATCH_ROWS,
    min_shared_rows: int = MIN_SHARED_ROWS_FOR_CORRELATION,
    train_end_date: date = TRAIN_END_DATE,
    start_date: date | pd.Timestamp | None = None,
    max_workers: int | None = None,
) -> np.ndarray:
    feature_count = len(feature_columns)
    if feature_count == 0:
        return np.zeros((0, 0), dtype=np.float64)

    worker_count = resolve_max_workers(max_workers)
    count, sum_pair, sumsq_pair, cross, processed_rows = _empty_correlation_stats(feature_count)
    stage_started_at = time.perf_counter()
    processed_batches = 0
    LOGGER.info(
        "Starting train-only pairwise correlation stage: %d features | batch_rows=%d | workers=%d.",
        feature_count,
        batch_size,
        worker_count,
    )

    def log_progress() -> None:
        LOGGER.info(
            "Correlation stage progress: %d batches processed | %d train rows accumulated | elapsed=%.2fs",
            processed_batches,
            processed_rows,
            time.perf_counter() - stage_started_at,
        )

    if worker_count == 1:
        for batch in iter_dataset_batches(
            feature_parquet_path,
            columns=feature_columns,
            batch_size=batch_size,
            train_only=True,
            train_end_date=train_end_date,
            start_date=start_date,
            use_threads=False,
        ):
            frame = record_batch_to_frame(batch)
            values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
            count, sum_pair, sumsq_pair, cross, processed_rows = _accumulate_correlation_stats(
                (count, sum_pair, sumsq_pair, cross, processed_rows),
                _compute_batch_correlation_stats_from_values(values),
            )
            processed_batches += 1
            if processed_batches == 1 or processed_batches % CORRELATION_PROGRESS_EVERY_BATCHES == 0:
                log_progress()
    else:
        max_in_flight = max(worker_count, worker_count * MAX_IN_FLIGHT_FUTURES_MULTIPLIER)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending: set[Future[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]] = set()

            def drain_completed(wait_for_all: bool) -> None:
                nonlocal count, sum_pair, sumsq_pair, cross, processed_rows, pending, processed_batches
                if not pending:
                    return
                if wait_for_all:
                    done, not_done = wait(pending)
                else:
                    done, not_done = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(not_done)
                for future in done:
                    count, sum_pair, sumsq_pair, cross, processed_rows = _accumulate_correlation_stats(
                        (count, sum_pair, sumsq_pair, cross, processed_rows),
                        future.result(),
                    )
                    processed_batches += 1
                    if (
                        processed_batches == 1
                        or processed_batches % CORRELATION_PROGRESS_EVERY_BATCHES == 0
                    ):
                        log_progress()

            for batch in iter_dataset_batches(
                feature_parquet_path,
                columns=feature_columns,
                batch_size=batch_size,
                train_only=True,
                train_end_date=train_end_date,
                start_date=start_date,
                use_threads=False,
            ):
                frame = record_batch_to_frame(batch)
                values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
                pending.add(executor.submit(_compute_batch_correlation_stats_from_values, values))
                if len(pending) >= max_in_flight:
                    drain_completed(wait_for_all=False)

            drain_completed(wait_for_all=True)

    LOGGER.info(
        "Correlation stage completed: %d train rows across %d features | batches=%d | workers=%d | elapsed=%.2fs.",
        processed_rows,
        feature_count,
        processed_batches,
        worker_count,
        time.perf_counter() - stage_started_at,
    )
    return _compute_corr_matrix_from_stats(
        count=count,
        sum_pair=sum_pair,
        sumsq_pair=sumsq_pair,
        cross=cross,
        min_shared_rows=min_shared_rows,
    )


def find_correlated_feature_groups_from_parquet(
    feature_parquet_path: Path,
    feature_columns: list[str],
    threshold: float,
    train_end_date: date = TRAIN_END_DATE,
    start_date: date | pd.Timestamp | None = None,
    max_workers: int | None = None,
) -> list[list[str]]:
    corr_matrix = compute_pairwise_correlation_matrix_from_parquet(
        feature_parquet_path,
        feature_columns=feature_columns,
        train_end_date=train_end_date,
        start_date=start_date,
        max_workers=max_workers,
    )
    groups = find_correlated_feature_groups_from_matrix(
        feature_columns=feature_columns,
        corr_matrix=corr_matrix,
        threshold=threshold,
    )
    LOGGER.info(
        "Detected %d train-only correlated feature groups at |corr| >= %.2f.",
        len(groups),
        threshold,
    )
    return groups
