from __future__ import annotations

import sys
from datetime import date
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
from core.src.meta_model.data.data_preprocessing.streaming import (
    iter_ticker_groups_from_parquet,
    PreprocessingColumnStats,
    resolve_final_columns,
)
from core.src.meta_model.model_contract import (
    INTRADAY_CS_ZSCORE_TARGET_COLUMN,
    INTRADAY_GROSS_RETURN_COLUMN,
    MODEL_TARGET_COLUMN,
    WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN,
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
                    "stock_open_price": close_prices[i] - 0.5,
                    "stock_close_price": close_prices[i],
                    "stock_close_log_return": close_log_return[i],
                    "company_sector": "Technology" if ticker == "AAPL" else "Software",
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

    def test_iterates_ticker_groups_from_single_row_group_parquet(self, tmp_path: Path) -> None:
        path: Path = tmp_path / "features.parquet"
        df: pd.DataFrame = _make_feature_df()
        df.to_parquet(path, index=False)

        groups = list(iter_ticker_groups_from_parquet(path))

        assert [str(group["ticker"].iloc[0]) for group in groups] == ["AAPL", "MSFT"]
        assert sum(len(group) for group in groups) == len(df)


class TestPreprocessingSteps:
    def test_filters_from_2009_01_01(self) -> None:
        result: pd.DataFrame = filter_from_start_date(_make_feature_df())

        assert result["date"].min() >= pd.Timestamp("2009-01-01")

    def test_creates_target_main_week_hold_geometry(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="B"),
                "ticker": ["AAPL"] * 8,
                "stock_open_price": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0],
                "stock_close_price": [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0, 115.0],
                "stock_close_log_return": [0.0] * 8,
                "company_sector": ["Technology"] * 8,
            },
        )

        result = create_target_main(df)

        assert result.loc[0, "target_main"] == pytest.approx(np.log(113.0 / 102.0))
        assert result.loc[1, "target_main"] == pytest.approx(np.log(115.0 / 104.0))
        assert pd.isna(result.loc[2, "target_main"])
        assert pd.isna(result.loc[7, "target_main"])
        assert MODEL_TARGET_COLUMN == WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN
        assert result.loc[0, MODEL_TARGET_COLUMN] == pytest.approx(result.loc[0, WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN])
        assert INTRADAY_GROSS_RETURN_COLUMN in result.columns

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

        result = exclude_covid_period(df, target_horizon_days=5)

        remaining_dates = result["date"].tolist()
        assert max(remaining_dates) == pd.Timestamp("2020-01-24")
        assert pd.Timestamp("2020-01-27") not in remaining_dates
        assert pd.Timestamp("2020-02-03") not in remaining_dates

    def test_assigns_train_val_test_and_drops_purge_embargo_windows(self) -> None:
        # Fixed test_start and zero embargo so a short calendar is deterministic.
        trading_days = pd.bdate_range("2022-01-03", periods=20, freq="C")
        df = pd.DataFrame(
            {
                "date": trading_days,
                "ticker": ["AAPL"] * len(trading_days),
                "stock_close_price": np.arange(len(trading_days), dtype=float) + 100.0,
                "stock_close_log_return": np.arange(len(trading_days), dtype=float) / 100.0,
                "target_main": np.arange(len(trading_days), dtype=float) / 10.0,
            },
        )
        extra_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-03-15", "2022-03-16"]),
                "ticker": ["AAPL", "AAPL"],
                "stock_close_price": [200.0, 201.0],
                "stock_close_log_return": [0.01, 0.02],
                "target_main": [0.5, 0.6],
            },
        )
        df = pd.concat([df, extra_test], ignore_index=True)

        with (
            patch(
                "core.src.meta_model.data.data_preprocessing.main.DATASET_SPLIT_TEST_START_DATE",
                date(2022, 3, 1),
            ),
            patch(
                "core.src.meta_model.data.data_preprocessing.main.LABEL_EMBARGO_DAYS",
                0,
            ),
        ):
            result = assign_dataset_splits(df)

        assert set(result.loc[result["dataset_split"] == "test", "date"].tolist()) == {
            pd.Timestamp("2022-03-15"),
            pd.Timestamp("2022-03-16"),
        }
        n_pre_test = 20
        n_val_expected = max(1, int(round(0.30 * n_pre_test)))
        n_val_rows = int((result["dataset_split"] == "val").sum())
        assert n_val_rows == n_val_expected
        assert int((result["dataset_split"] == "train").sum()) == n_pre_test - n_val_expected
        assert len(result) == n_pre_test + 2

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

    def test_resolve_final_columns_keeps_partially_missing_features(self) -> None:
        stats = PreprocessingColumnStats(
            columns=["date", "ticker", "target_main", "feature_partial", "feature_empty"],
            row_count=10,
            missing_count_by_column={
                "date": 0,
                "ticker": 0,
                "target_main": 0,
                "feature_partial": 3,
                "feature_empty": 10,
            },
            non_null_count_by_column={
                "date": 10,
                "ticker": 10,
                "target_main": 10,
                "feature_partial": 7,
                "feature_empty": 0,
            },
        )

        final_columns = resolve_final_columns(
            stats,
            protected_columns=["date", "ticker", "target_main"],
        )

        assert "feature_partial" in final_columns
        assert "feature_empty" not in final_columns


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
        label_panel_path: Path = tmp_path / "research_label_panel.parquet"
        registry_parquet_path: Path = tmp_path / "feature_registry.parquet"
        registry_json_path: Path = tmp_path / "feature_registry.json"
        manifest_path: Path = tmp_path / "feature_schema_manifest.json"
        train_path: Path = tmp_path / "train.parquet"
        train_sample_path: Path = tmp_path / "train_sample.csv"
        val_path: Path = tmp_path / "val.parquet"
        val_sample_path: Path = tmp_path / "val_sample.csv"
        test_path: Path = tmp_path / "test.parquet"
        test_sample_path: Path = tmp_path / "test_sample.csv"
        _make_feature_df().to_parquet(input_path, index=False)

        with (
            patch("core.src.meta_model.data.data_preprocessing.main.FEATURES_OUTPUT_PARQUET", input_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_OUTPUT_PARQUET", output_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_OUTPUT_SAMPLE_CSV", sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_RESEARCH_LABEL_PANEL_PARQUET", label_panel_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_FEATURE_REGISTRY_PARQUET", registry_parquet_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_FEATURE_REGISTRY_JSON", registry_json_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_FEATURE_SCHEMA_MANIFEST_JSON", manifest_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TRAIN_PARQUET", train_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TRAIN_SAMPLE_CSV", train_sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_VAL_PARQUET", val_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_VAL_SAMPLE_CSV", val_sample_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TEST_PARQUET", test_path),
            patch("core.src.meta_model.data.data_preprocessing.main.PREPROCESSED_TEST_SAMPLE_CSV", test_sample_path),
            # Monthly fixture rarely retains rows >= 2025-01-01 after label horizons; use an
            # earlier cutoff so this integration test still exercises train/val/test outputs.
            patch(
                "core.src.meta_model.data.data_preprocessing.main.DATASET_SPLIT_TEST_START_DATE",
                date(2015, 1, 1),
            ),
        ):
            main()

        result: pd.DataFrame = pd.read_parquet(output_path)
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)

        assert "target_main" in result.columns
        assert INTRADAY_CS_ZSCORE_TARGET_COLUMN in result.columns
        assert WEEK_HOLD_CS_ZSCORE_TARGET_COLUMN in result.columns
        assert "dataset_split" in result.columns
        assert result["date"].min() >= pd.Timestamp("2009-01-01")
        assert result["target_main"].notna().sum() > 0
        assert result["dataset_split"].isna().sum() == 0
        assert len(result) > 0
        assert "feature_drop" in result.columns
        assert "feature_with_nan" in result.columns
        assert set(result["dataset_split"].unique()) == {"train", "val", "test"}
        assert set(train["dataset_split"].unique()) == {"train"}
        assert set(val["dataset_split"].unique()) == {"val"}
        assert set(test["dataset_split"].unique()) == {"test"}
        assert sample_path.exists()
        assert label_panel_path.exists()
        assert registry_parquet_path.exists()
        assert registry_json_path.exists()
        assert manifest_path.exists()
        assert train_sample_path.exists()
        assert val_sample_path.exists()
        assert test_sample_path.exists()
