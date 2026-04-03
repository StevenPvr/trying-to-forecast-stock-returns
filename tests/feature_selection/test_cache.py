from __future__ import annotations

import sys
from pathlib import Path
import warnings

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.cache import FeatureSelectionRuntimeCache
from core.src.meta_model.feature_selection.io import load_feature_selection_metadata
from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN


def _make_cache_dataset(feature_count: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows: list[dict[str, object]] = []
    for date_index, current_date in enumerate(dates):
        for ticker_index, ticker in enumerate(("AAA", "BBB")):
            row: dict[str, object] = {
                "date": current_date,
                "ticker": ticker,
                "dataset_split": "train",
                MODEL_TARGET_COLUMN: float(ticker_index),
                "company_sector": "Technology",
                "company_beta": 1.1,
                "stock_open_price": 100.0 + date_index + ticker_index,
                "stock_trading_volume": 1_000_000.0 + date_index,
            }
            for feature_index in range(feature_count):
                row[f"feature_{feature_index:03d}"] = float(date_index + ticker_index + feature_index)
            rows.append(row)
    return pd.DataFrame(rows)


class TestFeatureSelectionCache:
    def test_build_feature_frame_avoids_fragmentation_warning(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_cache_dataset(feature_count=160)
        dataset.to_parquet(dataset_path, index=False)
        metadata = load_feature_selection_metadata(dataset_path)
        cache = FeatureSelectionRuntimeCache(
            dataset_path,
            metadata,
            random_seed=7,
            max_cache_gib=0.05,
        )
        feature_names = [f"feature_{feature_index:03d}" for feature_index in range(160)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            feature_frame = cache.build_feature_frame(feature_names)

        performance_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]
        assert not performance_warnings
        assert feature_frame.columns.tolist()[-160:] == feature_names

    def test_build_feature_frame_preserves_requested_columns_when_cache_eviction_occurs(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_cache_dataset(feature_count=48)
        dataset.to_parquet(dataset_path, index=False)
        metadata = load_feature_selection_metadata(dataset_path)
        cache = FeatureSelectionRuntimeCache(
            dataset_path,
            metadata,
            random_seed=7,
            max_cache_gib=0.000001,
        )
        feature_names = [f"feature_{feature_index:03d}" for feature_index in range(48)]

        feature_frame = cache.build_feature_frame(feature_names)

        assert feature_frame.columns.tolist()[-48:] == feature_names
        assert feature_frame.shape[1] >= 48

    def test_build_feature_frame_deduplicates_context_feature_overlaps(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_cache_dataset(feature_count=2)
        dataset.to_parquet(dataset_path, index=False)
        metadata = load_feature_selection_metadata(dataset_path)
        cache = FeatureSelectionRuntimeCache(
            dataset_path,
            metadata,
            random_seed=7,
            max_cache_gib=0.05,
        )

        feature_frame = cache.build_feature_frame(["stock_open_price", "company_beta", "feature_000"])

        assert feature_frame.columns.tolist().count("stock_open_price") == 1
        assert feature_frame.columns.tolist().count("company_beta") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
