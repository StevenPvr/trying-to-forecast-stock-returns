from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_cleaning.main import (
    finalize_modeling_dataset,
    load_raw_dataset,
    log_nan_report,
    log_nan_report_by_ticker,
    main,
    save_cleaned,
)


def _make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
        "ticker": ["MSFT", "AAPL", "MSFT", "AAPL"],
        "adj_close": [100.0, 200.0, 101.0, 201.0],
    })


class TestLoadRawDataset:
    def test_loads_parquet(self, tmp_path: Path) -> None:
        path: Path = tmp_path / "data.parquet"
        df: pd.DataFrame = _make_sample_df()
        df.to_parquet(path, index=False)
        result: pd.DataFrame = load_raw_dataset(path)
        assert len(result) == 4
        assert list(result.columns) == ["date", "ticker", "adj_close"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Input parquet not found"):
            load_raw_dataset(tmp_path / "nope.parquet")


class TestLogNanReport:
    def test_no_nan(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = _make_sample_df()
        with caplog.at_level(logging.INFO):
            log_nan_report(df, "test")
        assert "Total NaN: 0" in caplog.text

    def test_with_nan_by_column(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "MSFT"],
            "adj_close": [100.0, np.nan],
            "volume": [np.nan, np.nan],
        })
        with caplog.at_level(logging.INFO):
            log_nan_report(df, "before")
        assert "Total NaN: 3" in caplog.text
        assert "adj_close: 1 NaN (50.0%)" in caplog.text
        assert "volume: 2 NaN (100.0%)" in caplog.text

    def test_no_ticker_column(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = pd.DataFrame({"val": [1.0, np.nan]})
        with caplog.at_level(logging.INFO):
            log_nan_report(df, "test")
        assert "Total NaN: 1" in caplog.text


class TestLogNanReportByTicker:
    def test_reports_per_ticker(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["AAPL", "MSFT"],
            "adj_close": [100.0, np.nan],
            "volume": [np.nan, np.nan],
        })
        with caplog.at_level(logging.INFO):
            log_nan_report_by_ticker(df, "test")
        assert "AAPL" in caplog.text
        assert "MSFT" in caplog.text

    def test_no_ticker_column(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = pd.DataFrame({"val": [1.0, np.nan]})
        with caplog.at_level(logging.INFO):
            log_nan_report_by_ticker(df, "test")
        assert "skipping per-ticker" in caplog.text

    def test_no_nan(self, caplog: pytest.LogCaptureFixture) -> None:
        df: pd.DataFrame = _make_sample_df()
        with caplog.at_level(logging.INFO):
            log_nan_report_by_ticker(df, "test")
        assert "tickers with missing data" in caplog.text


class TestSaveCleaned:
    def test_creates_files(self, tmp_path: Path) -> None:
        df: pd.DataFrame = _make_sample_df()
        pq: Path = tmp_path / "out.parquet"
        csv: Path = tmp_path / "out.csv"
        paths: dict[str, Path] = save_cleaned(df, pq, csv)
        assert pq.exists()
        assert csv.exists()
        assert paths["parquet"] == pq
        assert paths["sample_csv"] == csv

    def test_parquet_roundtrip(self, tmp_path: Path) -> None:
        df: pd.DataFrame = _make_sample_df()
        pq: Path = tmp_path / "out.parquet"
        csv: Path = tmp_path / "out.csv"
        save_cleaned(df, pq, csv)
        loaded: pd.DataFrame = pd.read_parquet(pq)
        assert len(loaded) == 4
        assert list(loaded.columns) == ["date", "ticker", "adj_close"]

    def test_sample_random_fraction(self, tmp_path: Path) -> None:
        dates: list[pd.Timestamp] = list(pd.bdate_range("2024-01-01", periods=100))
        rows: list[dict[str, object]] = []
        for ticker in ["AAPL", "MSFT"]:
            for d in dates:
                rows.append({"date": d, "ticker": ticker, "adj_close": 100.0})
        df: pd.DataFrame = pd.DataFrame(rows)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        pq: Path = tmp_path / "out.parquet"
        csv: Path = tmp_path / "out.csv"
        save_cleaned(df, pq, csv)
        sample: pd.DataFrame = pd.read_csv(csv)
        assert len(sample) == 10
        assert "ticker" in sample.columns
        assert sample["ticker"].nunique() == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        pq: Path = tmp_path / "sub" / "dir" / "out.parquet"
        csv: Path = tmp_path / "sub" / "dir" / "out.csv"
        save_cleaned(_make_sample_df(), pq, csv)
        assert pq.exists()


class TestFinalizeModelingDataset:
    def test_filters_data_errors_and_keeps_binary_outlier_features(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "stock_open_log_return": [0.01, 0.02, 0.03],
            "data_error_flag": [False, False, True],
            "ticker_return_extreme_flag": [True, False, True],
            "cross_section_return_extreme_flag": [False, True, False],
            "is_outlier_flag": [True, True, True],
            "outlier_severity": ["elevated", "elevated", "data_error"],
            "outlier_reason": [
                "TICKER_RETURN_EXTREME",
                "CROSS_SECTION_RETURN_EXTREME",
                "NEGATIVE_VOLUME",
            ],
        })

        result: pd.DataFrame = finalize_modeling_dataset(df)

        assert len(result) == 2
        assert list(result["ticker"]) == ["AAPL", "MSFT"]
        assert list(result.columns) == [
            "date",
            "ticker",
            "stock_open_log_return",
            "ticker_return_extreme_flag",
            "cross_section_return_extreme_flag",
        ]
        assert result["ticker_return_extreme_flag"].dtype == np.int8
        assert result["cross_section_return_extreme_flag"].dtype == np.int8
        assert int(result.loc[0, "ticker_return_extreme_flag"]) == 1
        assert int(result.loc[1, "cross_section_return_extreme_flag"]) == 1

    def test_drops_constant_non_identifier_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "MSFT"],
            "stock_open_log_return": [0.01, 0.02],
            "constant_float_feature": [1.5, 1.5],
            "constant_flag_feature": [0, 0],
            "all_nan_feature": [np.nan, np.nan],
        })

        result: pd.DataFrame = finalize_modeling_dataset(df)

        assert "date" in result.columns
        assert "ticker" in result.columns
        assert "stock_open_log_return" in result.columns
        assert "constant_float_feature" not in result.columns
        assert "constant_flag_feature" not in result.columns
        assert "all_nan_feature" not in result.columns

    def test_drops_quasi_constant_non_binary_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=1000, freq="D"),
            "ticker": [f"T{idx:04d}" for idx in range(1000)],
            "signal_feature": np.linspace(0.0, 1.0, 1000),
            "quasi_constant_numeric": [7.5] * 995 + [8.5] * 5,
        })

        result: pd.DataFrame = finalize_modeling_dataset(df)

        assert "signal_feature" in result.columns
        assert "quasi_constant_numeric" not in result.columns

    def test_keeps_quasi_constant_binary_indicator_features(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=1000, freq="D"),
            "ticker": [f"T{idx:04d}" for idx in range(1000)],
            "signal_feature": np.linspace(0.0, 1.0, 1000),
            "rare_binary_flag": [0] * 999 + [1],
        })

        result: pd.DataFrame = finalize_modeling_dataset(df)

        assert "signal_feature" in result.columns
        assert "rare_binary_flag" in result.columns


class TestMain:
    def test_full_flow(self, tmp_path: Path) -> None:
        df: pd.DataFrame = _make_sample_df()
        input_pq: Path = tmp_path / "input.parquet"
        df.to_parquet(input_pq, index=False)
        output_pq: Path = tmp_path / "cleaned.parquet"
        output_csv: Path = tmp_path / "cleaned.csv"
        with (
            patch(
                "core.src.meta_model.data.data_cleaning.main.MERGED_OUTPUT_PARQUET",
                input_pq,
            ),
            patch(
                "core.src.meta_model.data.data_cleaning.main.CLEANED_OUTPUT_PARQUET",
                output_pq,
            ),
            patch(
                "core.src.meta_model.data.data_cleaning.main.CLEANED_OUTPUT_SAMPLE_CSV",
                output_csv,
            ),
        ):
            main()
        result: pd.DataFrame = pd.read_parquet(output_pq)
        assert list(result["ticker"]) == ["MSFT", "AAPL", "MSFT", "AAPL"]
        assert list(result.columns) == ["date", "ticker", "adj_close"]
        assert "data_error_flag" not in result.columns
        assert "is_outlier_flag" not in result.columns
        assert "outlier_severity" not in result.columns
        assert "outlier_reason" not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
