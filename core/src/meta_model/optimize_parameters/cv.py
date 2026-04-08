from __future__ import annotations

"""Walk-forward fold construction with configurable train/validation ratios."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.src.meta_model.optimize_parameters.config import (
    DATE_COLUMN,
    SPLIT_COLUMN,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
    TARGET_HORIZON_DAYS,
)


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    weight: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp
    train_row_count: int
    validation_row_count: int


def build_walk_forward_folds(
    data: pd.DataFrame,
    fold_count: int,
    *,
    target_horizon_days: int = TARGET_HORIZON_DAYS,
) -> list[WalkForwardFold]:
    if fold_count <= 0:
        raise ValueError("fold_count must be strictly positive.")
    if target_horizon_days < 0:
        raise ValueError("target_horizon_days must be non-negative.")

    ordered = data.sort_values([DATE_COLUMN, "ticker"]).reset_index(drop=True)
    train_mask = ordered[SPLIT_COLUMN] == TRAIN_SPLIT_NAME
    validation_mask = ordered[SPLIT_COLUMN] == VAL_SPLIT_NAME
    if not train_mask.any():
        raise ValueError("Optimization dataset does not contain any train rows.")
    if not validation_mask.any():
        raise ValueError("Optimization dataset does not contain any validation rows.")

    validation_dates = pd.Index(pd.to_datetime(ordered.loc[validation_mask, DATE_COLUMN]).drop_duplicates())
    validation_date_chunks = [
        list(chunk.tolist())
        for chunk in np.array_split(validation_dates.to_numpy(), fold_count)
        if len(chunk) > 0
    ]
    if len(validation_date_chunks) != fold_count:
        raise ValueError(
            f"Unable to create {fold_count} non-empty validation folds from {len(validation_dates)} dates.",
        )

    folds: list[WalkForwardFold] = []
    for fold_index, validation_dates_chunk in enumerate(validation_date_chunks, start=1):
        validation_start = pd.Timestamp(validation_dates_chunk[0])
        validation_end = pd.Timestamp(validation_dates_chunk[-1])
        expanding_train_mask = train_mask | (
            validation_mask & (ordered[DATE_COLUMN] < validation_start)
        )
        if target_horizon_days > 0:
            train_candidate_dates = pd.Index(
                pd.to_datetime(ordered.loc[expanding_train_mask, DATE_COLUMN]).drop_duplicates().sort_values(),
            )
            embargoed_dates = train_candidate_dates[-target_horizon_days:].tolist()
            expanding_train_mask = expanding_train_mask & ~ordered[DATE_COLUMN].isin(embargoed_dates)
        fold_validation_mask = validation_mask & ordered[DATE_COLUMN].isin(validation_dates_chunk)
        train_indices = np.flatnonzero(expanding_train_mask.to_numpy())
        validation_indices = np.flatnonzero(fold_validation_mask.to_numpy())
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError(f"Fold {fold_index} produced an empty train or validation block.")
        train_end_date = pd.Timestamp(ordered.loc[train_indices, DATE_COLUMN].max())
        folds.append(
            WalkForwardFold(
                index=fold_index,
                weight=float(fold_index),
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_end_date=train_end_date,
                validation_start_date=validation_start,
                validation_end_date=validation_end,
                train_row_count=int(train_indices.size),
                validation_row_count=int(validation_indices.size),
            ),
        )
    return folds
