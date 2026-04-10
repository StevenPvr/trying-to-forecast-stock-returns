from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.registry import compute_feature_schema_hash
from core.src.meta_model.evaluate.dataset import (
    build_feature_columns,
    load_manifest_feature_columns,
    load_preprocessed_evaluation_dataset,
    validate_feature_schema_manifest,
)
from core.src.meta_model.evaluate.config import MODEL_TARGET_COLUMN
from core.src.meta_model.model_contract import (
    DATE_COLUMN,
    SPLIT_COLUMN,
    TEST_SPLIT_NAME,
    TICKER_COLUMN,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
)


def test_validate_feature_schema_manifest_accepts_feature_names_key(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "feature_schema_manifest.json"
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "dataset_split": ["test"],
        MODEL_TARGET_COLUMN: [0.1],
        "feature_a": [1.0],
    }).to_parquet(dataset_path, index=False)
    manifest_path.write_text(
        json.dumps({
            "feature_names": ["feature_a"],
            "feature_schema_hash": compute_feature_schema_hash(["feature_a"]),
        }),
        encoding="utf-8",
    )

    validate_feature_schema_manifest(["feature_a"], dataset_path)


def test_build_feature_columns_preserves_dataset_order() -> None:
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "dataset_split": ["test"],
        MODEL_TARGET_COLUMN: [0.1],
        "earnings_days_to_next": [2.0],
        "feature_b": [2.0],
        "feature_a": [1.0],
    })

    assert build_feature_columns(data) == ["ticker", "feature_b", "feature_a"]


def test_load_preprocessed_evaluation_dataset_supports_split_and_column_pushdown(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame({
        DATE_COLUMN: pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        TICKER_COLUMN: ["AAA", "AAA", "AAA"],
        SPLIT_COLUMN: [TRAIN_SPLIT_NAME, VAL_SPLIT_NAME, TEST_SPLIT_NAME],
        MODEL_TARGET_COLUMN: [0.1, 0.2, 0.3],
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [10.0, 20.0, 30.0],
    }).to_parquet(dataset_path, index=False)
    loaded = load_preprocessed_evaluation_dataset(
        dataset_path,
        allowed_splits=(TRAIN_SPLIT_NAME, VAL_SPLIT_NAME),
        columns=["feature_a", MODEL_TARGET_COLUMN],
    )
    assert set(loaded[SPLIT_COLUMN].astype(str).unique()) == {TRAIN_SPLIT_NAME, VAL_SPLIT_NAME}
    assert "feature_a" in loaded.columns
    assert "feature_b" not in loaded.columns


def test_load_preprocessed_evaluation_dataset_ignores_missing_projected_columns(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame({
        DATE_COLUMN: pd.to_datetime(["2024-01-02"]),
        TICKER_COLUMN: ["AAA"],
        SPLIT_COLUMN: [TRAIN_SPLIT_NAME],
        MODEL_TARGET_COLUMN: [0.1],
        "feature_a": [1.0],
    }).to_parquet(dataset_path, index=False)
    loaded = load_preprocessed_evaluation_dataset(
        dataset_path,
        allowed_splits=(TRAIN_SPLIT_NAME,),
        columns=["feature_a", "missing_feature", "missing_price"],
    )
    assert "feature_a" in loaded.columns
    assert "missing_feature" not in loaded.columns
    assert "missing_price" not in loaded.columns


def test_load_manifest_feature_columns_filters_temporarily_disabled_features(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "feature_schema_manifest.json"
    pd.DataFrame({
        DATE_COLUMN: pd.to_datetime(["2024-01-02"]),
        TICKER_COLUMN: ["AAA"],
        SPLIT_COLUMN: [TEST_SPLIT_NAME],
        MODEL_TARGET_COLUMN: [0.1],
    }).to_parquet(dataset_path, index=False)
    manifest_path.write_text(
        json.dumps({
            "feature_names": ["ticker", "feature_a", "earnings_days_to_next"],
            "feature_schema_hash": compute_feature_schema_hash(["ticker", "feature_a", "earnings_days_to_next"]),
        }),
        encoding="utf-8",
    )
    assert load_manifest_feature_columns(dataset_path) == ["ticker", "feature_a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
