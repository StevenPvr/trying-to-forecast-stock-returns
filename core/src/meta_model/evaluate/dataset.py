from __future__ import annotations

"""Dataset loading, feature-column resolution, and schema validation for the evaluate pipeline."""

from pathlib import Path
from typing import cast

import pandas as pd

from core.src.meta_model.data.paths import (
    FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    FEATURE_SELECTION_SCHEMA_MANIFEST_JSON,
)
from core.src.meta_model.data.registry import compute_feature_schema_hash, load_feature_schema_manifest
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT_NAME,
    TICKER_COLUMN,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
    is_temporarily_disabled_alpha_feature_column,
)
from core.src.meta_model.optimize_parameters.dataset import build_feature_columns


def _normalize_manifest_feature_names(raw_feature_names: object) -> list[str]:
    if not isinstance(raw_feature_names, list):
        raise ValueError("Evaluation dataset feature schema manifest is malformed.")
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


def load_manifest_feature_columns(
    dataset_path: Path,
) -> list[str]:
    manifest_path = resolve_feature_schema_manifest_path(dataset_path)
    if manifest_path is None or not manifest_path.exists():
        return []
    manifest = load_feature_schema_manifest(manifest_path)
    return [
        feature_name
        for feature_name in _normalize_manifest_feature_names(
            manifest.get("feature_names", manifest.get("feature_columns")),
        )
        if not is_temporarily_disabled_alpha_feature_column(feature_name)
    ]


def _resolve_available_parquet_columns(path: Path) -> set[str] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        schema = pq.read_schema(path)
    except Exception:
        return None
    return {str(name) for name in schema.names}


def load_preprocessed_evaluation_dataset(
    path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
    *,
    allowed_splits: tuple[str, ...] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load and prepare the preprocessed evaluation dataset from Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    projected_columns: list[str] | None = None
    if columns is not None:
        requested = [str(column_name) for column_name in columns]
        requested.extend([DATE_COLUMN, TICKER_COLUMN, SPLIT_COLUMN])
        requested_columns = list(dict.fromkeys(requested))
        available_columns = _resolve_available_parquet_columns(path)
        if available_columns is None:
            projected_columns = requested_columns
        else:
            projected_columns = [
                column_name for column_name in requested_columns if column_name in available_columns
            ]
    parquet_filters: list[tuple[str, str, list[str]]] | None = None
    if allowed_splits:
        parquet_filters = [(SPLIT_COLUMN, "in", list(allowed_splits))]
    try:
        data = pd.read_parquet(path, columns=projected_columns, filters=parquet_filters)
    except (TypeError, ValueError, NotImplementedError):
        data = pd.read_parquet(path, columns=projected_columns)
    prepared = data.copy()
    if allowed_splits:
        prepared = pd.DataFrame(
            prepared.loc[prepared[SPLIT_COLUMN].astype(str).isin(allowed_splits)].copy(),
        )
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def validate_feature_schema_manifest(
    feature_columns: list[str],
    dataset_path: Path,
) -> None:
    """Verify that *feature_columns* match the saved schema manifest hash."""
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
            "Evaluation dataset feature schema does not match the feature-selection manifest.",
        )


def split_training_and_test_frames(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train+val, test) frames using the dataset_split column."""
    train_mask = data[SPLIT_COLUMN].isin([TRAIN_SPLIT_NAME, VAL_SPLIT_NAME])
    test_mask = data[SPLIT_COLUMN] == TEST_SPLIT_NAME
    training_frame = pd.DataFrame(data.loc[train_mask].copy())
    test_frame = pd.DataFrame(data.loc[test_mask].copy())
    if training_frame.empty:
        raise ValueError("No training rows found for evaluate pipeline.")
    if test_frame.empty:
        raise ValueError("No test rows found for evaluate pipeline.")
    return (
        training_frame.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True),
        test_frame.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True),
    )
