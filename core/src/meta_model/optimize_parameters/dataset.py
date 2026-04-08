from __future__ import annotations

"""Dataset loading, feature-column resolution, and schema manifest for optimisation."""

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
from core.src.meta_model.model_contract import (
    is_excluded_feature_column,
    is_structural_categorical_feature_column,
    is_temporarily_disabled_alpha_feature_column,
)
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


def optimization_feature_payload_bytes(bundle: "OptimizationDatasetBundle") -> int:
    return int(bundle.feature_frame.memory_usage(deep=True).sum() + bundle.target_array.nbytes)


def bundle_has_native_xgboost_categoricals(bundle: "OptimizationDatasetBundle") -> bool:
    return any(
        isinstance(bundle.feature_frame[column_name].dtype, pd.CategoricalDtype)
        for column_name in bundle.feature_columns
        if column_name in bundle.feature_frame.columns
    )


@dataclass(frozen=True)
class OptimizationDatasetBundle:
    metadata: pd.DataFrame
    feature_columns: list[str]
    feature_frame: pd.DataFrame
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


def _column_eligible_as_feature(column_name: str, series: pd.Series) -> bool:
    if is_excluded_feature_column(column_name):
        return False
    if is_temporarily_disabled_alpha_feature_column(column_name):
        return False
    if is_structural_categorical_feature_column(column_name):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    return False


def build_feature_columns(data: pd.DataFrame) -> list[str]:
    feature_columns: list[str] = []
    skipped_non_numeric_columns: list[str] = []
    for column_name in data.columns:
        column_series = data[column_name]
        if not _column_eligible_as_feature(column_name, column_series):
            continue
        if is_structural_categorical_feature_column(column_name):
            feature_columns.append(column_name)
            continue
        if pd.api.types.is_numeric_dtype(column_series) or pd.api.types.is_bool_dtype(column_series):
            feature_columns.append(column_name)
            continue
        skipped_non_numeric_columns.append(column_name)
    if skipped_non_numeric_columns:
        LOGGER.warning(
            "Optimization dataset excludes non-numeric feature columns: count=%d | preview=%s",
            len(skipped_non_numeric_columns),
            ",".join(skipped_non_numeric_columns[:20]),
        )
    return feature_columns


def _prepare_structural_series(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype("category")
    string_series = series.astype("string")
    filled = string_series.fillna("__missing__")
    return filled.astype("category")


def validate_feature_schema_manifest(
    feature_columns: list[str],
    dataset_path: Path,
) -> None:
    manifest_path = resolve_feature_schema_manifest_path(dataset_path)
    if manifest_path is None or not manifest_path.exists():
        return
    manifest = load_feature_schema_manifest(manifest_path)
    manifest_feature_names = [
        feature_name
        for feature_name in _normalize_manifest_feature_names(
            manifest.get("feature_names", manifest.get("feature_columns")),
        )
        if not is_temporarily_disabled_alpha_feature_column(feature_name)
    ]
    expected_hash = compute_feature_schema_hash(manifest_feature_names)
    actual_hash = compute_feature_schema_hash(
        [
            feature_name
            for feature_name in feature_columns
            if not is_temporarily_disabled_alpha_feature_column(feature_name)
        ],
    )
    if actual_hash != expected_hash:
        raise ValueError(
            "Optimization dataset feature schema does not match the feature-selection manifest.",
        )


def build_optimization_dataset_bundle(
    data: pd.DataFrame,
    dataset_path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> OptimizationDatasetBundle:
    feature_columns = build_feature_columns(data)
    if not feature_columns:
        raise ValueError(
            "Optimization dataset contains no feature columns after exclusions.",
        )
    validate_feature_schema_manifest(feature_columns, dataset_path)
    raw_frame = pd.DataFrame(data.loc[:, feature_columns].copy())
    feature_parts: list[pd.Series] = []
    for column_name in feature_columns:
        column_series = raw_frame[column_name]
        if is_structural_categorical_feature_column(column_name):
            feature_parts.append(_prepare_structural_series(column_series))
            continue
        numeric = pd.to_numeric(column_series, errors="coerce").astype(np.float32)
        feature_parts.append(numeric)
    feature_frame = pd.concat(feature_parts, axis=1)
    feature_frame.columns = feature_columns
    invalid_mask = np.zeros(feature_frame.shape, dtype=bool)
    for column_index, column_name in enumerate(feature_columns):
        if is_structural_categorical_feature_column(column_name):
            continue
        column_values = feature_frame.iloc[:, column_index].to_numpy(dtype=np.float64, copy=False)
        invalid_mask[:, column_index] = ~np.isfinite(column_values)
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
        for column_index, column_name in enumerate(feature_columns):
            if is_structural_categorical_feature_column(column_name):
                continue
            mask = invalid_mask[:, column_index]
            if bool(mask.any()):
                feature_frame.iloc[mask, column_index] = np.nan
    target_array = np.ascontiguousarray(
        data[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False),
    )
    metadata = data.loc[:, [DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN]].copy()
    payload_mb = (
        float(feature_frame.memory_usage(deep=True).sum()) + float(target_array.nbytes)
    ) / (1024.0 * 1024.0)
    LOGGER.info(
        "Prepared optimization dataset bundle: rows=%d | features=%d | feature_payload=%.2f MB | target=%.2f MB",
        len(metadata),
        len(feature_columns),
        payload_mb,
        target_array.nbytes / (1024.0 * 1024.0),
    )
    return OptimizationDatasetBundle(
        metadata=metadata,
        feature_columns=feature_columns,
        feature_frame=feature_frame,
        target_array=target_array,
    )
