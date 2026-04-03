from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from core.src.meta_model.feature_selection.io import FeatureSelectionMetadata
from core.src.meta_model.model_contract import DATE_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionFold:
    index: int
    weight: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def build_train_only_selection_folds(
    metadata: FeatureSelectionMetadata,
    fold_count: int,
    *,
    label_embargo_days: int,
) -> list[SelectionFold]:
    if fold_count <= 0:
        raise ValueError("fold_count must be strictly positive.")
    train_frame = pd.DataFrame(
        metadata.frame.take(metadata.train_row_indices).reset_index(drop=True),
    )
    unique_train_dates = pd.Index(pd.to_datetime(train_frame[DATE_COLUMN]).drop_duplicates().sort_values())
    date_chunks = [list(chunk.tolist()) for chunk in np.array_split(unique_train_dates.to_numpy(), fold_count + 1) if len(chunk) > 0]
    if len(date_chunks) != fold_count + 1:
        raise ValueError("Unable to build train-only feature-selection folds.")
    folds: list[SelectionFold] = []
    for fold_index, validation_dates in enumerate(date_chunks[1:], start=1):
        validation_start = pd.Timestamp(validation_dates[0])
        validation_end = pd.Timestamp(validation_dates[-1])
        train_candidate_dates = unique_train_dates[unique_train_dates < validation_start]
        if len(train_candidate_dates) <= label_embargo_days:
            raise ValueError("Not enough train dates to respect the selection embargo.")
        allowed_train_dates = train_candidate_dates[:-label_embargo_days] if label_embargo_days > 0 else train_candidate_dates
        train_mask = train_frame[DATE_COLUMN].isin(allowed_train_dates)
        validation_mask = train_frame[DATE_COLUMN].isin(validation_dates)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(validation_mask.to_numpy())
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError(f"Feature-selection fold {fold_index} is empty.")
        folds.append(
            SelectionFold(
                index=fold_index,
                weight=float(fold_index),
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_end_date=pd.Timestamp(allowed_train_dates[-1]),
                validation_start_date=validation_start,
                validation_end_date=validation_end,
            ),
        )
        LOGGER.info(
            "Feature selection fold %d/%d | train_rows=%d | validation_rows=%d | train_end=%s | validation_start=%s | validation_end=%s",
            fold_index,
            fold_count,
            train_indices.size,
            validation_indices.size,
            pd.Timestamp(allowed_train_dates[-1]).date(),
            validation_start.date(),
            validation_end.date(),
        )
    return folds


__all__ = ["SelectionFold", "build_train_only_selection_folds"]
