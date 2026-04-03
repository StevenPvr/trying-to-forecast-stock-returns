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
    validate_feature_schema_manifest,
)
from core.src.meta_model.evaluate.config import MODEL_TARGET_COLUMN


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
        "feature_b": [2.0],
        "feature_a": [1.0],
    })

    assert build_feature_columns(data) == ["feature_b", "feature_a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
