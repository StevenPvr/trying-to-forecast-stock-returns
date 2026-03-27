from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_cleaning.outlier_pipeline import apply_outlier_flags


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


class TestApplyOutlierFlags:
    def test_adds_expected_columns(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
            ]),
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "stock_open_log_return": [np.nan, np.nan, 0.01, 0.01],
            "stock_trading_volume": [100, 100, 110, 110],
        })
        result: pd.DataFrame = apply_outlier_flags(df)
        assert "data_error_flag" in result.columns
        assert "ticker_return_extreme_flag" in result.columns
        assert "cross_section_return_extreme_flag" in result.columns
        assert "is_outlier_flag" in result.columns
        assert "outlier_severity" in result.columns
        assert "outlier_reason" in result.columns

    def test_negative_volume_flagged_as_data_error(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "AAPL"],
            "stock_open_log_return": [0.0, 0.01],
            "stock_trading_volume": [100, -1],
        })
        result: pd.DataFrame = apply_outlier_flags(df)
        assert bool(result.loc[1, "data_error_flag"])
        assert bool(result.loc[1, "is_outlier_flag"])
        assert result.loc[1, "outlier_severity"] == "data_error"
        assert "NEGATIVE_VOLUME" in str(result.loc[1, "outlier_reason"])

    def test_cross_section_extreme_flag_detects_market_outlier(self) -> None:
        dates: list[pd.Timestamp] = [_ts("2024-01-02"), _ts("2024-01-03")]
        rows: list[dict[str, object]] = []
        for ticker, day2_return in [
            ("A", 0.01),
            ("B", 0.01),
            ("C", 0.01),
            ("D", 0.50),
        ]:
            rows.append({
                "date": dates[0],
                "ticker": ticker,
                "stock_open_log_return": 0.0,
                "stock_trading_volume": 100,
            })
            rows.append({
                "date": dates[1],
                "ticker": ticker,
                "stock_open_log_return": day2_return,
                "stock_trading_volume": 100,
            })
        df: pd.DataFrame = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

        result: pd.DataFrame = apply_outlier_flags(df)
        outlier_row = result[(result["date"] == dates[1]) & (result["ticker"] == "D")].iloc[0]
        assert bool(outlier_row["cross_section_return_extreme_flag"])
        assert bool(outlier_row["is_outlier_flag"])
        assert outlier_row["outlier_severity"] == "extreme"
        assert "CROSS_SECTION_RETURN_EXTREME" in str(outlier_row["outlier_reason"])

    def test_graceful_when_return_column_missing(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAPL"],
            "stock_trading_volume": [100],
        })
        result: pd.DataFrame = apply_outlier_flags(df)
        assert not bool(result.loc[0, "is_outlier_flag"])
        assert result.loc[0, "outlier_severity"] == "normal"


if __name__ == "__main__":
    pytest_module = __import__("pytest")
    pytest_module.main([__file__, "-v"])
