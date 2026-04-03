from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.feature_selection.io import (
    build_feature_selection_input_inventory,
    build_selected_feature_dataset,
    discover_selection_feature_columns,
    load_feature_selection_metadata,
    subsample_train_feature_selection_metadata,
)
from core.src.meta_model.model_contract import MODEL_TARGET_COLUMN, is_excluded_feature_column


def _make_preprocessed_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
        "ticker": ["BBB", "BBB", "AAA"],
        "dataset_split": ["train", "val", "train"],
        MODEL_TARGET_COLUMN: [0.2, 0.3, 0.1],
        "feature_float": [1.0, 2.0, 3.0],
        "feature_int": [1, 2, 3],
        "feature_text": ["x", "y", "z"],
    })


def _make_larger_preprocessed_dataset() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    rows: list[dict[str, object]] = []
    for current_date in dates:
        for ticker_index, ticker in enumerate(("AAA", "BBB")):
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "dataset_split": "train",
                    MODEL_TARGET_COLUMN: float(ticker_index),
                    "feature_float": float(ticker_index + 1),
                    "feature_int": ticker_index + 1,
                    "feature_text": "x",
                },
            )
    return pd.DataFrame(rows)


class TestFeatureSelectionIo:
    def test_discover_selection_feature_columns_uses_numeric_schema_only(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)

        feature_columns = discover_selection_feature_columns(dataset_path)

        assert "feature_float" in feature_columns
        assert "feature_int" in feature_columns
        assert "feature_text" not in feature_columns
        assert MODEL_TARGET_COLUMN not in feature_columns

    def test_build_feature_selection_input_inventory_counts_columns(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)

        inventory = build_feature_selection_input_inventory(dataset_path)

        assert inventory.total_columns == 7
        assert inventory.excluded_columns == 4
        assert inventory.numeric_feature_columns == 2
        assert inventory.non_numeric_non_excluded_columns == 1

    def test_load_feature_selection_metadata_sorts_canonically(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)

        metadata = load_feature_selection_metadata(dataset_path)

        ordered_pairs = list(
            zip(
                metadata.frame["date"].dt.strftime("%Y-%m-%d"),
                metadata.frame["ticker"].astype(str),
                strict=True,
            ),
        )
        assert ordered_pairs == [
            ("2024-01-02", "AAA"),
            ("2024-01-02", "BBB"),
            ("2024-01-03", "BBB"),
        ]
        assert metadata.train_row_indices.tolist() == [0, 2]

    def test_subsample_train_feature_selection_metadata_keeps_first_twenty_percent_of_train_dates(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_larger_preprocessed_dataset().to_parquet(dataset_path, index=False)

        metadata = load_feature_selection_metadata(dataset_path)
        sampled = subsample_train_feature_selection_metadata(
            metadata,
            train_sampling_fraction=0.20,
            minimum_unique_dates=1,
        )

        sampled_dates = pd.Index(
            pd.to_datetime(
                sampled.frame.take(sampled.train_row_indices).reset_index(drop=True)["date"],
            ).drop_duplicates().sort_values(),
        )
        assert len(sampled_dates) == 2
        assert sampled.train_row_indices.size == 4
        assert sampled_dates.tolist() == [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        ]

    def test_subsample_train_feature_selection_metadata_respects_minimum_unique_dates(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_larger_preprocessed_dataset().to_parquet(dataset_path, index=False)

        metadata = load_feature_selection_metadata(dataset_path)
        sampled = subsample_train_feature_selection_metadata(
            metadata,
            train_sampling_fraction=0.20,
            minimum_unique_dates=6,
        )

        sampled_dates = pd.Index(
            pd.to_datetime(
                sampled.frame.take(sampled.train_row_indices).reset_index(drop=True)["date"],
            ).drop_duplicates().sort_values(),
        )
        assert len(sampled_dates) == 6
        assert sampled.train_row_indices.size == 12

    def test_build_selected_feature_dataset_projects_protected_columns(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        _make_preprocessed_dataset().to_parquet(dataset_path, index=False)

        filtered_dataset = build_selected_feature_dataset(dataset_path, ["feature_float"])

        assert filtered_dataset.columns.tolist() == [
            "date",
            "ticker",
            "dataset_split",
            MODEL_TARGET_COLUMN,
            "feature_float",
        ]

    def test_build_selected_feature_dataset_retains_context_columns_with_excluded_prefix(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_preprocessed_dataset().assign(
            company_sector=["Tech", "Tech", "Tech"],
            stock_open_price=[100.0, 101.0, 102.0],
            stock_high_price=[101.0, 102.0, 103.0],
            stock_low_price=[99.0, 100.0, 101.0],
            stock_close_price=[100.5, 101.5, 102.5],
            stock_trading_volume=[1_000.0, 1_100.0, 1_200.0],
        )
        dataset.to_parquet(dataset_path, index=False)

        filtered_dataset = build_selected_feature_dataset(
            dataset_path,
            ["feature_float"],
            retained_context_columns={
                "company_sector": "hl_context_company_sector",
                "stock_open_price": "hl_context_stock_open_price",
            },
        )

        assert "hl_context_company_sector" in filtered_dataset.columns
        assert "hl_context_stock_open_price" in filtered_dataset.columns
        assert "company_sector" not in filtered_dataset.columns
        assert "stock_open_price" not in filtered_dataset.columns

    def test_build_selected_feature_dataset_preserves_selected_context_feature(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_preprocessed_dataset().assign(
            stock_open_price=[100.0, 101.0, 102.0],
        )
        dataset.to_parquet(dataset_path, index=False)

        filtered_dataset = build_selected_feature_dataset(
            dataset_path,
            ["stock_open_price", "feature_float"],
            retained_context_columns={
                "stock_open_price": "hl_context_stock_open_price",
            },
        )

        assert "stock_open_price" in filtered_dataset.columns
        assert "hl_context_stock_open_price" in filtered_dataset.columns
        assert filtered_dataset["stock_open_price"].tolist() == [102.0, 101.0, 100.0]
        assert filtered_dataset["hl_context_stock_open_price"].tolist() == [102.0, 101.0, 100.0]

    def test_build_selected_feature_dataset_preserves_selected_feature_order_with_context_overlap(
        self,
        tmp_path: Path,
    ) -> None:
        dataset_path = tmp_path / "preprocessed.parquet"
        dataset = _make_preprocessed_dataset().assign(
            stock_open_price=[100.0, 101.0, 102.0],
        )
        dataset.to_parquet(dataset_path, index=False)

        filtered_dataset = build_selected_feature_dataset(
            dataset_path,
            ["feature_float", "stock_open_price"],
            retained_context_columns={
                "stock_open_price": "hl_context_stock_open_price",
            },
        )

        actual_feature_names = [
            column_name
            for column_name in filtered_dataset.columns
            if column_name in {"feature_float", "stock_open_price"}
        ]
        assert actual_feature_names == ["feature_float", "stock_open_price"]

    def test_high_level_context_columns_are_excluded_from_feature_candidates(self) -> None:
        assert is_excluded_feature_column("hl_context_company_sector")
        assert is_excluded_feature_column("hl_context_stock_open_price")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
