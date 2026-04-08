from __future__ import annotations

"""Research-grade signal diagnostics: daily Rank IC, Linear IC, top/bottom spread."""

import math
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    PREDICTION_COLUMN,
    REALIZED_RETURN_COLUMN,
)


def _column_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    return cast(pd.Series, frame.loc[:, column_name])


def _to_timestamp(value: str | pd.Timestamp | np.datetime64) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Encountered NaT while building signal diagnostics.")
    validated_timestamp: pd.Timestamp = cast(pd.Timestamp, timestamp)
    return validated_timestamp


def compute_mean_daily_spearman_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
) -> float:
    """Compute the mean daily Spearman rank IC across cross-sections."""
    frame = pd.DataFrame({
        DATE_COLUMN: pd.to_datetime(dates),
        PREDICTION_COLUMN: predictions,
        "target": targets,
    })
    daily_rank_ic: list[float] = []
    for _, group in frame.groupby(DATE_COLUMN, sort=False):
        if len(group) < 2:
            continue
        prediction_ranks = _column_series(group, PREDICTION_COLUMN).rank(method="average")
        target_ranks = _column_series(group, "target").rank(method="average")
        if prediction_ranks.nunique(dropna=False) < 2 or target_ranks.nunique(dropna=False) < 2:
            continue
        rank_ic = prediction_ranks.corr(target_ranks)
        if pd.notna(rank_ic):
            daily_rank_ic.append(float(rank_ic))
    if not daily_rank_ic:
        return 0.0
    return float(np.mean(np.asarray(daily_rank_ic, dtype=np.float64)))


def build_daily_signal_diagnostics(
    predictions: pd.DataFrame,
    *,
    target_column: str,
    realized_return_column: str = REALIZED_RETURN_COLUMN,
    top_fraction: float = 0.01,
) -> pd.DataFrame:
    """Build a per-day DataFrame of rank IC, linear IC, and long/short spread."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    ordered = predictions.sort_values([DATE_COLUMN, PREDICTION_COLUMN]).reset_index(drop=True)
    for current_date, group in ordered.groupby(DATE_COLUMN, sort=False):
        if len(group) < 2:
            continue
        prediction_series = _column_series(group, PREDICTION_COLUMN)
        target_series = _column_series(group, target_column)
        prediction_ranks = prediction_series.rank(method="average")
        target_ranks = target_series.rank(method="average")
        rank_ic = (
            prediction_ranks.corr(target_ranks)
            if prediction_ranks.nunique(dropna=False) >= 2 and target_ranks.nunique(dropna=False) >= 2
            else np.nan
        )
        linear_ic = (
            prediction_series.corr(target_series)
            if prediction_series.nunique(dropna=False) >= 2 and target_series.nunique(dropna=False) >= 2
            else np.nan
        )
        selection_count = max(1, int(math.ceil(len(group) * top_fraction)))
        long_mean = float(
            _column_series(group.nlargest(selection_count, PREDICTION_COLUMN), realized_return_column).mean(),
        )
        short_mean = float(
            _column_series(group.nsmallest(selection_count, PREDICTION_COLUMN), realized_return_column).mean(),
        )
        rows.append({
            DATE_COLUMN: _to_timestamp(cast(str | pd.Timestamp | np.datetime64, current_date)),
            "rank_ic": 0.0 if pd.isna(rank_ic) else float(rank_ic),
            "linear_ic": 0.0 if pd.isna(linear_ic) else float(linear_ic),
            "top_bottom_spread": long_mean - short_mean,
            "cross_section_count": float(len(group)),
        })
    return pd.DataFrame(rows).sort_values(DATE_COLUMN).reset_index(drop=True)


def summarize_daily_signal_diagnostics(daily_diagnostics: pd.DataFrame) -> dict[str, float]:
    """Aggregate daily diagnostics into a summary dict (mean IC, IR, spread)."""
    if daily_diagnostics.empty:
        return {
            "daily_rank_ic_mean": 0.0,
            "daily_rank_ic_std": 0.0,
            "daily_rank_ic_ir": 0.0,
            "daily_linear_ic_mean": 0.0,
            "daily_top_bottom_spread_mean": 0.0,
            "daily_top_bottom_spread_std": 0.0,
        }

    rank_ic = _column_series(daily_diagnostics, "rank_ic").to_numpy(dtype=np.float64)
    spread = _column_series(daily_diagnostics, "top_bottom_spread").to_numpy(dtype=np.float64)
    rank_ic_std = float(rank_ic.std(ddof=0))
    return {
        "daily_rank_ic_mean": float(rank_ic.mean()),
        "daily_rank_ic_std": rank_ic_std,
        "daily_rank_ic_ir": (
            0.0
            if math.isclose(rank_ic_std, 0.0, rel_tol=0.0, abs_tol=1e-12)
            else float(rank_ic.mean() / rank_ic_std)
        ),
        "daily_linear_ic_mean": float(_column_series(daily_diagnostics, "linear_ic").mean()),
        "daily_top_bottom_spread_mean": float(spread.mean()),
        "daily_top_bottom_spread_std": float(spread.std(ddof=0)),
    }
