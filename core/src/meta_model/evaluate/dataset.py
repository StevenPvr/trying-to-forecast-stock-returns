from __future__ import annotations

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
)
from core.src.meta_model.model_contract import is_excluded_feature_column


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


def load_preprocessed_evaluation_dataset(
    path: Path = FEATURE_SELECTION_FILTERED_DATASET_PARQUET,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    data = pd.read_parquet(path)
    prepared = data.copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


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
            "Evaluation dataset feature schema does not match the feature-selection manifest.",
        )


def split_training_and_test_frames(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
