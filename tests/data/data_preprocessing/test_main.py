from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_preprocessing.main import (
    _sample_train_subset,
    assign_dataset_splits,
    create_target_main,
    drop_columns_with_missing_values,
    exclude_covid_period,
    filter_from_start_date,
    forward_fill_features_by_ticker,
    load_feature_dataset,
    main,
    remove_rows_with_missing_values,
    prune_correlated_features,
    save_preprocessed_dataset,
    validate_no_missing_values,
)


def _make_feature_df() -> pd.DataFrame:
    dates: pd.DatetimeIndex = pd.date_range("2008-12-01", "2025-03-31", freq="BMS")
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAPL", "MSFT"), start=1):
        close_prices = 100.0 * ticker_index + np.arange(len(dates), dtype=float)
        prev_close = np.concatenate(([close_prices[0]], close_prices[:-1]))
        close_log_return = np.log(close_prices / prev_close)
        base_signal = np.linspace(0.0, 1.0, len(dates), dtype=float) + ticker_index
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "stock_close_price": close_prices[i],
                    "stock_close_log_return": close_log_return[i],
                    "feature_keep": base_signal[i],
                    "feature_drop": base_signal[i] + 0.001,
                    "feature_other": np.sin(i / 5.0),
                    "feature_with_nan": np.nan if i == 3 else base_signal[i] * 2.0,
                },
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


class TestLoadFeatureDataset:
    def test_loads_parquet(self, tmp_path: Path) -> None:
        path: Path = tmp_path / "features.parquet"
        df: pd.DataFrame = _make_feature_df()
        df.to_parquet(path, index=False)

        result: pd.DataFrame = load_feature_dataset(path)

        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Input parquet not found"):
            load_feature_dataset(tmp_path / "missing.parquet")


class TestPreprocessingSteps:
    def test_filters_from_2009_01_01(self) -> None:
        result: pd.DataFrame = filter_from_start_date(_make_feature_df())

        assert result["date"].min() >= pd.Timestamp("2009-01-01")

    def test_creates_target_main_as_forward_weekly_close_log_return(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="B"),
                "ticker": ["AAPL"] * 8,
                "stock_close_price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                "stock_close_log_return": [0.0] * 8,
            },
        )

        result = create_target_main(df)

        assert result.loc[0, "target_main"] == pytest.approx(np.log(105.0 / 100.0))
        assert result.loc[1, "target_main"] == pytest.approx(np.log(106.0 / 101.0))
        assert pd.isna(result.loc[4, "target_main"])

    def test_excludes_covid_period_and_pre_covid_target_bridge_dates(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-20", "2020-02-10", freq="B"),
                "ticker": ["AAPL"] * 16,
                "stock_close_price": np.arange(16, dtype=float) + 100.0,
                "stock_close_log_return": np.arange(16, dtype=float) / 100.0,
                "target_main": np.arange(16, dtype=float) / 10.0,
            },
        )

        result = exclude_covid_period(df)

        remaining_dates = result["date"].tolist()
        assert max(remaining_dates) == pd.Timestamp("2020-01-24")
        assert pd.Timestamp("2020-01-27") not in remaining_dates
        assert pd.Timestamp("2020-02-03") not in remaining_dates

    def test_assigns_train_val_test_and_drops_purge_embargo_windows(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2018-11-30",
                        "2018-12-15",
                        "2019-01-15",
                        "2019-02-01",
                        "2021-11-30",
                        "2021-12-15",
                        "2022-01-15",
                        "2022-02-01",
                    ],
                ),
                "ticker": ["AAPL"] * 8,
                "stock_close_price": np.arange(8, dtype=float) + 100.0,
                "stock_close_log_return": np.arange(8, dtype=float) / 100.0,
                "target_main": np.arange(8, dtype=float) / 10.0,
            },
        )

        result = assign_dataset_splits(df)

        assert result["dataset_split"].tolist() == ["train", "val", "val", "test"]
        assert result["date"].tolist() == [
            pd.Timestamp("2018-11-30"),
            pd.Timestamp("2019-02-01"),
            pd.Timestamp("2021-11-30"),
            pd.Timestamp("2022-02-01"),
        ]

    def test_prunes_correlated_features_using_target_distance_correlation(self) -> None:
        dates = pd.date_range("2010-01-01", periods=40, freq="B")
        x = np.linspace(0.0, 1.0, len(dates))
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * len(dates),
                "dataset_split": ["train"] * len(dates),
                "target_main": x + 0.001,
                "feature_keep": x + 0.001,
                "feature_drop": x + 0.02,
                "feature_other": np.sin(np.linspace(0.0, 3.0, len(dates))),
            },
        )

        result = prune_correlated_features(
            df,
            target_column="target_main",
            split_column="dataset_split",
            feature_sample_frac=0.5,
            prescreener_threshold=0.9,
            distance_threshold=0.95,
        )

        assert "feature_keep" in result.columns
        assert "feature_other" in result.columns
        assert "feature_drop" not in result.columns

    def test_removes_rows_with_missing_values(self) -> None:
        df = create_target_main(filter_from_start_date(_make_feature_df()))

        result = remove_rows_with_missing_values(df, required_columns=["target_main"])

        assert result["target_main"].isna().sum() == 0
        assert len(result) > 0

    def test_forward_fills_feature_values_within_each_ticker(self) -> None:
        df = create_target_main(filter_from_start_date(_make_feature_df()))
        split_ready = assign_dataset_splits(df)

        result = forward_fill_features_by_ticker(
            split_ready,
            protected_columns=["date", "ticker", "target_main", "dataset_split"],
        )

        filled_row = result.loc[
            (result["ticker"] == "AAPL") & (result["date"] == pd.Timestamp("2009-04-01")),
            "feature_with_nan",
        ]
        assert len(filled_row) == 1
        assert not pd.isna(filled_row.iloc[0])

    def test_drops_columns_with_missing_values_and_keeps_target(self) -> None:
        df = create_target_main(filter_from_start_date(_make_feature_df()))
        df = remove_rows_with_missing_values(df, required_columns=["target_main"])

        result = drop_columns_with_missing_values(df, protected_columns=["date", "ticker", "target_main"])

        assert "feature_with_nan" not in result.columns
        assert "target_main" in result.columns
        assert result.isna().sum().sum() == 0

    def test_validate_no_missing_values_raises_when_any_missing_remains(self) -> None:
        df = create_target_main(filter_from_start_date(_make_feature_df()))

        with pytest.raises(ValueError, match="Missing values remain"):
            validate_no_missing_values(df)

    def test_caps_pruning_sample_size_before_distance_correlation(self) -> None:
        train_data = pd.DataFrame({
            "date": pd.date_range("2010-01-01", periods=5000, freq="B"),
            "ticker": ["AAPL"] * 5000,
            "dataset_split": ["train"] * 5000,
            "target_main": np.linspace(0.0, 1.0, 5000),
            "feature_keep": np.linspace(0.0, 1.0, 5000),
            "feature_drop": np.linspace(0.0, 1.0, 5000) + 0.001,
        })

        sampled = _sample_train_subset(train_data, sample_frac=1.0)

        assert len(sampled) == 2000


class TestSavePreprocessedDataset:
    def test_saves_parquet_and_sample(self, tmp_path: Path) -> None:
        df: pd.DataFrame = assign_dataset_splits(
            drop_columns_with_missing_values(
                remove_rows_with_missing_values(
                    create_target_main(filter_from_start_date(_make_feature_df())),
                    required_columns=["target_main"],
                ),
                protected_columns=["date", "ticker", "target_main", "dataset_split"],
            ),
        )
        parquet_path: Path = tmp_path / "preprocessed.parquet"
        sample_path: Path = tmp_path / "preprocessed_sample.csv"

        paths = save_preprocessed_dataset(df, parquet_path, sample_path)

        assert parquet_path.exists()
        assert sample_path.exists()
        assert paths["parquet"] == parquet_path
        assert paths["sample_csv"] == sample_path


class TestMain:
    def test_full_flow_creates_clean_split_outputs(self, tmp_path: Path) -> None:
        input_path: Path = tmp_path / "features.parquet"
        output_path: Path = tmp_path / "preprocessed.parquet"
        sample_path: Path = tmp_path / "preprocessed_sample.csv"
        train_path: Path = tmp_path / "train.parquet"
        train_sample_path: Path = tmp_path / "train_sample.csv"
        val_path: Path = tmp_path / "val.parquet"
        val_sample_path: Path = tmp_path / "val_sample.csv"
        test_path: Path = tmp_path / "test.parquet"
        test_sample_path: Path = tmp_path / "test_sample.csv"
        _make_feature_df().to_parquet(input_path, index=False)

        with (
            patch("core.src.meta_model.data.data_preprocessing.main.GREEDY_FORWARD_SELECTION_FILTERED_FEATURES_PARQUET", input_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_OUTPUT_PARQUET", output_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_OUTPUT_SAMPLE_CSV", sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TRAIN_PARQUET", train_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TRAIN_SAMPLE_CSV", train_sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_VAL_PARQUET", val_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_VAL_SAMPLE_CSV", val_sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TEST_PARQUET", test_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TEST_SAMPLE_CSV", test_sample_path),
        ):
            main()

        result: pd.DataFrame = pd.read_parquet(output_path)
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)

        assert "target_main" in result.columns
        assert "dataset_split" in result.columns
        assert result["date"].min() >= pd.Timestamp("2009-01-01")
        assert not result["date"].between("2020-02-01", "2021-12-31").any()
        assert result.isna().sum().sum() == 0
        assert len(result) > 0
        assert "feature_drop" not in result.columns
        assert set(result["dataset_split"].unique()) == {"train", "val", "test"}
        assert set(train["dataset_split"].unique()) == {"train"}
        assert set(val["dataset_split"].unique()) == {"val"}
        assert set(test["dataset_split"].unique()) == {"test"}
        assert sample_path.exists()
        assert train_sample_path.exists()
        assert val_sample_path.exists()
        assert test_sample_path.exists()
