from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.src.meta_model.data.paths import PREPROCESSED_OUTPUT_PARQUET
from core.src.meta_model.optimize_parameters.config import (
    DATE_COLUMN,
    EXCLUDED_FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TRAIN_SPLIT_NAME,
    TARGET_COLUMN,
    TICKER_COLUMN,
    VAL_SPLIT_NAME,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationDatasetBundle:
    metadata: pd.DataFrame
    feature_columns: list[str]
    feature_matrix: np.ndarray
    target_array: np.ndarray


def load_preprocessed_dataset(
    path: Path = PREPROCESSED_OUTPUT_PARQUET,
    *,
    allowed_splits: tuple[str, ...] | None = (TRAIN_SPLIT_NAME, VAL_SPLIT_NAME),
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")
    parquet_filters: list[tuple[str, str, list[str]]] | None = None
    if allowed_splits:
        parquet_filters = [(SPLIT_COLUMN, "in", list(allowed_splits))]
    try:
        data = pd.read_parquet(path, filters=parquet_filters)
    except (TypeError, ValueError, NotImplementedError):
        data = pd.read_parquet(path)
    required_columns = {DATE_COLUMN, TICKER_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in optimization dataset: {missing}")
    prepared = data.copy()
    if allowed_splits:
        prepared = pd.DataFrame(
            prepared.loc[prepared[SPLIT_COLUMN].astype(str).isin(allowed_splits)].copy(),
        )
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    prepared[TICKER_COLUMN] = prepared[TICKER_COLUMN].astype("category")
    prepared[SPLIT_COLUMN] = prepared[SPLIT_COLUMN].astype("category")
    ordered = prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    LOGGER.info(
        "Loaded optimization dataset: %d rows x %d cols from %s%s",
        len(ordered),
        len(ordered.columns),
        path,
        "" if not allowed_splits else f" | splits={','.join(allowed_splits)}",
    )
    return ordered


def build_feature_columns(data: pd.DataFrame) -> list[str]:
    return sorted(
        column_name
        for column_name in data.columns
        if column_name not in EXCLUDED_FEATURE_COLUMNS
    )


def build_optimization_dataset_bundle(data: pd.DataFrame) -> OptimizationDatasetBundle:
    feature_columns = build_feature_columns(data)
    feature_matrix = np.ascontiguousarray(
        data.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False),
    )
    target_array = np.ascontiguousarray(
        data[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False),
    )
    metadata = data.loc[:, [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN]].copy()
    LOGGER.info(
        "Prepared optimization dataset bundle: rows=%d | features=%d | matrix_dtype=%s | feature_matrix=%.2f MB | target=%.2f MB",
        len(metadata),
        len(feature_columns),
        feature_matrix.dtype,
        feature_matrix.nbytes / (1024.0 * 1024.0),
        target_array.nbytes / (1024.0 * 1024.0),
    )
    return OptimizationDatasetBundle(
        metadata=metadata,
        feature_columns=feature_columns,
        feature_matrix=feature_matrix,
        target_array=target_array,
    )
