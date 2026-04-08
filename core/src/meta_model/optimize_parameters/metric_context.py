from __future__ import annotations

"""Daily rank-IC context builder for Optuna objective evaluation."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class DailyRankIcGroup:
    start: int
    stop: int
    valid_positions: np.ndarray
    target_ranks_valid: np.ndarray
    target_centered: np.ndarray
    target_sum_squares: float


@dataclass(frozen=True)
class DailyRankIcContext:
    sort_index: np.ndarray | None
    groups: tuple[DailyRankIcGroup, ...]


def _build_group_boundaries(date_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if date_values.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    boundaries = np.flatnonzero(date_values[1:] != date_values[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    stops = np.concatenate((boundaries, np.asarray([date_values.size], dtype=np.int64)))
    return starts, stops


def _target_group_from_slice(target_slice: np.ndarray, start: int, stop: int) -> DailyRankIcGroup | None:
    if stop - start < 2:
        return None
    ranked_targets = rankdata(target_slice, method="average", nan_policy="omit")
    valid_positions = np.flatnonzero(~np.isnan(ranked_targets))
    if valid_positions.size < 2:
        return None
    target_ranks_valid = ranked_targets[valid_positions]
    if math.isclose(float(np.ptp(target_ranks_valid)), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    target_centered = target_ranks_valid - target_ranks_valid.mean()
    target_sum_squares = float(np.dot(target_centered, target_centered))
    if math.isclose(target_sum_squares, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    return DailyRankIcGroup(
        start=start,
        stop=stop,
        valid_positions=valid_positions,
        target_ranks_valid=target_ranks_valid,
        target_centered=target_centered,
        target_sum_squares=target_sum_squares,
    )


def build_daily_rank_ic_context(targets: np.ndarray, dates: np.ndarray) -> DailyRankIcContext:
    target_values = np.asarray(targets, dtype=np.float64)
    date_values = np.asarray(pd.to_datetime(dates), dtype="datetime64[ns]")
    sort_index: np.ndarray | None = None
    if date_values.size > 1 and np.any(date_values[1:] < date_values[:-1]):
        sort_index = np.argsort(date_values, kind="mergesort")
        date_values = date_values[sort_index]
        target_values = target_values[sort_index]
    starts, stops = _build_group_boundaries(date_values)
    groups = tuple(
        group
        for start, stop in zip(starts, stops, strict=False)
        if (group := _target_group_from_slice(target_values[start:stop], int(start), int(stop))) is not None
    )
    return DailyRankIcContext(sort_index=sort_index, groups=groups)


def _safe_rank_ic(prediction_ranks: np.ndarray, group: DailyRankIcGroup) -> float:
    prediction_valid = prediction_ranks[group.valid_positions]
    valid_mask = ~np.isnan(prediction_valid)
    if not np.all(valid_mask):
        if int(valid_mask.sum()) < 2:
            return float("nan")
        prediction_valid = prediction_valid[valid_mask]
        target_ranks_valid = group.target_ranks_valid[valid_mask]
        target_centered = target_ranks_valid - target_ranks_valid.mean()
        target_sum_squares = float(np.dot(target_centered, target_centered))
    else:
        target_centered = group.target_centered
        target_sum_squares = group.target_sum_squares
    if math.isclose(float(np.ptp(prediction_valid)), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return float("nan")
    prediction_centered = prediction_valid - prediction_valid.mean()
    prediction_sum_squares = float(np.dot(prediction_centered, prediction_centered))
    if (
        math.isclose(prediction_sum_squares, 0.0, rel_tol=0.0, abs_tol=1e-12)
        or math.isclose(target_sum_squares, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        return float("nan")
    return float(
        np.dot(prediction_centered, target_centered)
        / math.sqrt(prediction_sum_squares * target_sum_squares),
    )


def compute_mean_daily_rank_ic_from_context(
    predictions: np.ndarray,
    context: DailyRankIcContext,
) -> float:
    if not context.groups:
        return 0.0
    prediction_values = np.asarray(predictions, dtype=np.float64)
    if context.sort_index is not None:
        prediction_values = prediction_values[context.sort_index]
    total_rank_ic = 0.0
    valid_group_count = 0
    has_missing_predictions = bool(np.isnan(prediction_values).any())
    for group in context.groups:
        prediction_slice = prediction_values[group.start:group.stop]
        if has_missing_predictions:
            prediction_ranks = rankdata(prediction_slice, method="average", nan_policy="omit")
        else:
            prediction_ranks = rankdata(prediction_slice, method="average")
        rank_ic = _safe_rank_ic(prediction_ranks, group)
        if not math.isnan(rank_ic):
            total_rank_ic += rank_ic
            valid_group_count += 1
    if valid_group_count == 0:
        return 0.0
    return float(total_rank_ic / valid_group_count)
