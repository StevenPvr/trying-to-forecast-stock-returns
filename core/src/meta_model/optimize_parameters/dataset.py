from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.data.paths import (
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    FEATURE_SELECTION_SCHEMA_MANIFEST_JSON,
)
from core.src.meta_model.data.registry import compute_feature_schema_hash, load_feature_schema_manifest
from core.src.meta_model.model_contract import is_excluded_feature_column
from core.src.meta_model.optimize_parameters.config import (
    DATE_COLUMN,
    SPLIT_COLUMN,
    TRAIN_SPLIT_NAME,
    TARGET_COLUMN,
    TICKER_COLUMN,
    VAL_SPLIT_NAME,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _normalize_manifest_feature_names(raw_feature_names: object) -> list[str]:
    if not isinstance(raw_feature_names, list):
        raise ValueError("Optimization dataset feature schema manifest is malformed.")
    typed_feature_names = cast(list[object], raw_feature_names)
    normalized_feature_names: list[str] = []
    for raw_name in typed_feature_names:
        normalized_feature_names.append(str(raw_name))
    return normalized_feature_names


def resolve_feature_schema_manifest_path(dataset_path: Path) -> Path | None:
    if dataset_path == FEATURE_SELECTION_FILTERED_DATASET_PARQUET:
        return FEATURE_SELECTION_SCHEMA_MANIFEST_JSON
    adjacent_manifest = dataset_path.with_name("feature_schema_manifest.json")
    return adjacent_manifest if adjacent_manifest.exists() else None


@dataclass(frozen=True)
class OptimizationDatasetBundle:
    metadata: pd.DataFrame
    feature_columns: list[str]
    feature_matrix: np.ndarray
    target_array: np.ndarray


def load_preprocessed_dataset(
    path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
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
    return [
        column_name
        for column_name in data.columns
        if not is_excluded_feature_column(column_name)
    ]


def validate_feature_schema_manifest(
    feature_columns: list[str],
    dataset_path: Path,
) -> None:
    manifest_path = resolve_feature_schema_manifest_path(dataset_path)
    if manifest_path is None or not manifest_path.exists():
        return
    manifest = load_feature_schema_manifest(manifest_path)
    manifest_feature_names = _normalize_manifest_feature_names(
        manifest.get("feature_names", manifest.get("feature_columns")),
    )
    expected_hash = str(
        manifest.get(
            "feature_schema_hash",
            compute_feature_schema_hash(manifest_feature_names),
        ),
    )
    actual_hash = compute_feature_schema_hash(feature_columns)
    if actual_hash != expected_hash:
        raise ValueError(
            "Optimization dataset feature schema does not match the feature-selection manifest.",
        )


def build_optimization_dataset_bundle(
    data: pd.DataFrame,
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> OptimizationDatasetBundle:
    feature_columns = build_feature_columns(data)
    validate_feature_schema_manifest(feature_columns, dataset_path)
    feature_frame = pd.DataFrame(data.loc[:, feature_columns].copy())
    feature_matrix = np.ascontiguousarray(
        feature_frame.to_numpy(dtype=np.float32, copy=False),
    )
    invalid_mask = ~np.isfinite(feature_matrix)
    if invalid_mask.any():
        invalid_counts = invalid_mask.sum(axis=0, dtype=np.int64)
        affected_columns = [
            f"{feature_columns[column_index]}={int(invalid_counts[column_index])}"
            for column_index in range(len(feature_columns))
            if int(invalid_counts[column_index]) > 0
        ]
        LOGGER.warning(
            "Optimization dataset contained non-finite feature values; coercing them to NaN before XGBoost | affected_columns=%s",
            ",".join(affected_columns[:20]),
        )
        feature_matrix[invalid_mask] = np.nan
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
