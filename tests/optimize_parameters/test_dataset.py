from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.registry import compute_feature_schema_hash
from core.src.meta_model.optimize_parameters.dataset import (
    build_optimization_dataset_bundle,
    build_feature_columns,
    validate_feature_schema_manifest,
)
from core.src.meta_model.optimize_parameters.config import TARGET_COLUMN


def test_validate_feature_schema_manifest_accepts_feature_names_key(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "feature_schema_manifest.json"
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "ticker": ["AAA"],
        "dataset_split": ["train"],
        TARGET_COLUMN: [0.1],
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
        "dataset_split": ["train"],
        TARGET_COLUMN: [0.1],
        "feature_b": [2.0],
        "feature_text": ["foo"],
        "feature_a": [1.0],
    })

    assert build_feature_columns(data) == ["feature_b", "feature_a"]


def test_build_feature_columns_excludes_non_numeric_types() -> None:
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "ticker": ["AAA", "AAA"],
        "dataset_split": ["train", "val"],
        TARGET_COLUMN: [0.1, 0.2],
        "feature_float": [1.0, 2.0],
        "feature_bool": [True, False],
        "feature_name": ["Agilent Technologies", "Apple"],
    })

    assert build_feature_columns(data) == ["feature_float", "feature_bool"]


def test_build_optimization_dataset_bundle_replaces_inf_with_nan(tmp_path: Path) -> None:
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "ticker": ["AAA", "AAA"],
        "dataset_split": ["train", "val"],
        TARGET_COLUMN: [0.1, 0.2],
        "feature_ok": [1.0, 2.0],
        "feature_bad": [np.inf, 3.0],
    })
    dataset_path = tmp_path / "optimization.parquet"
    data.to_parquet(dataset_path, index=False)

    bundle = build_optimization_dataset_bundle(data, dataset_path)

    assert bundle.feature_columns == ["feature_ok", "feature_bad"]
    assert not np.isinf(bundle.feature_matrix).any()
    assert np.isnan(bundle.feature_matrix[0, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
