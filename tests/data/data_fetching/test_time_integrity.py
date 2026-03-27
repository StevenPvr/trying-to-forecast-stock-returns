from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_fetching.main import _build_date_features
from core.src.meta_model.data.data_fetching.macro_pipeline import _build_macro_dataframe
from core.src.meta_model.data.data_fetching.sentiment_pipeline import _build_sentiment_dataframe
from core.src.meta_model.data.data_fetching.sp500_pipeline import PipelineConfig, build_dataset
from core.src.meta_model.data.trading_calendar import get_nyse_sessions


def _make_price_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 100.0 + len(index) - 1, len(index)),
            "high": np.linspace(101.0, 101.0 + len(index) - 1, len(index)),
            "low": np.linspace(99.0, 99.0 + len(index) - 1, len(index)),
            "close": np.linspace(100.5, 100.5 + len(index) - 1, len(index)),
            "adj_close": np.linspace(100.5, 100.5 + len(index) - 1, len(index)),
            "volume": np.linspace(1_000.0, 1_000.0 + len(index) - 1, len(index)),
        },
        index=index,
    )


class TestNyseTradingCalendar:
    def test_excludes_holidays_and_special_closures(self) -> None:
        sessions: pd.DatetimeIndex = get_nyse_sessions("2012-10-26", "2012-10-31")

        assert pd.Timestamp("2012-10-29") not in sessions
        assert pd.Timestamp("2012-10-30") not in sessions
        assert pd.Timestamp("2012-10-31") in sessions

        july_sessions: pd.DatetimeIndex = get_nyse_sessions("2024-07-03", "2024-07-05")
        assert pd.Timestamp("2024-07-04") not in july_sessions
        assert list(july_sessions) == [
            pd.Timestamp("2024-07-03"),
            pd.Timestamp("2024-07-05"),
        ]


class TestAvailabilityLags:
    def test_macro_daily_series_available_next_session_only(self) -> None:
        series: pd.Series = pd.Series(  # type: ignore[type-arg]
            [1.0, 2.0],
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

        result: pd.DataFrame = _build_macro_dataframe(
            {"DGS10": series},
            "2020-01-02",
            "2020-01-07",
        )

        assert pd.isna(result["dgs10"].iloc[0])
        assert result["dgs10"].iloc[1:].tolist() == [1.0, 2.0, 2.0]

    def test_sentiment_weekly_series_not_available_immediately(self) -> None:
        aaii: pd.DataFrame = pd.DataFrame(
            {"aaii_bullish": [0.3, 0.4]},
            index=pd.to_datetime(["2020-01-02", "2020-01-09"]),
        )
        aaii.index.name = "date"

        result: pd.DataFrame = _build_sentiment_dataframe(
            aaii,
            pd.DataFrame(),
            "2020-01-01",
            "2020-01-10",
        )

        jan2_value = result.loc[result["date"] == pd.Timestamp("2020-01-02"), "aaii_bullish"].iloc[0]
        jan9_value = result.loc[result["date"] == pd.Timestamp("2020-01-09"), "aaii_bullish"].iloc[0]
        assert pd.isna(jan2_value)
        assert jan9_value == pytest.approx(0.3)


class TestPointInTimeUniverseAndFundamentals:
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline.load_constituents")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_prices")
    def test_build_dataset_uses_snapshot_universe_by_default(
        self,
        mock_fetch_prices: MagicMock,
        mock_load_constituents: MagicMock,
    ) -> None:
        mock_load_constituents.return_value = {"AAPL"}
        sessions: pd.DatetimeIndex = pd.to_datetime(["2024-07-03", "2024-07-05"])
        mock_fetch_prices.return_value = {"AAPL": _make_price_df(sessions)}

        config: PipelineConfig = PipelineConfig(
            start_date="2024-07-03",
            end_date="2024-07-05",
        )
        result: pd.DataFrame = build_dataset(config)

        assert list(result["date"]) == list(sessions)
        assert result["ticker"].tolist() == ["AAPL", "AAPL"]

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_all_fundamentals")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_prices")
    def test_build_dataset_filters_membership_history_and_skips_holiday_rows(
        self,
        mock_fetch_prices: MagicMock,
        mock_live_fundamentals: MagicMock,
        tmp_path: Path,
    ) -> None:
        membership_path: Path = tmp_path / "membership.csv"
        membership_path.write_text(
            "ticker,start_date,end_date\nAAPL,2024-07-03,2024-07-05\n"
        )
        sessions: pd.DatetimeIndex = pd.to_datetime(["2024-07-03", "2024-07-05"])
        mock_fetch_prices.return_value = {"AAPL": _make_price_df(sessions)}
        mock_live_fundamentals.side_effect = AssertionError(
            "live fundamentals must not be used by default",
        )

        config: PipelineConfig = PipelineConfig(
            start_date="2024-07-03",
            end_date="2024-07-05",
            membership_history_csv=membership_path,
        )
        result: pd.DataFrame = build_dataset(config)

        assert list(result["date"]) == list(sessions)
        assert "sector" not in result.columns
        assert "industry" not in result.columns

    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_all_fundamentals")
    @patch("core.src.meta_model.data.data_fetching.sp500_pipeline._fetch_prices")
    def test_build_dataset_merges_point_in_time_fundamentals_history(
        self,
        mock_fetch_prices: MagicMock,
        mock_live_fundamentals: MagicMock,
        tmp_path: Path,
    ) -> None:
        membership_path: Path = tmp_path / "membership.csv"
        membership_path.write_text(
            "ticker,start_date,end_date\nAAPL,2024-07-03,2024-07-05\n"
        )
        fundamentals_path: Path = tmp_path / "fundamentals.csv"
        fundamentals_path.write_text(
            "date,ticker,company_market_cap_usd,company_trailing_pe_ratio\n"
            "2024-07-02,AAPL,100,10\n"
            "2024-07-05,AAPL,200,20\n"
        )

        sessions: pd.DatetimeIndex = pd.to_datetime(["2024-07-03", "2024-07-05"])
        mock_fetch_prices.return_value = {"AAPL": _make_price_df(sessions)}
        mock_live_fundamentals.side_effect = AssertionError(
            "live fundamentals must not be used when PIT history is supplied",
        )

        config: PipelineConfig = PipelineConfig(
            start_date="2024-07-03",
            end_date="2024-07-05",
            membership_history_csv=membership_path,
            fundamentals_history_csv=fundamentals_path,
        )
        result: pd.DataFrame = build_dataset(config)

        assert result["company_market_cap_usd"].tolist() == [100, 200]
        assert result["company_trailing_pe_ratio"].tolist() == [10, 20]


class TestStrictDateFeatureBuild:
    @patch("core.src.meta_model.data.data_fetching.main.build_cross_asset_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_sentiment_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_calendar_dataset")
    @patch("core.src.meta_model.data.data_fetching.main.build_macro_dataset")
    def test_raises_on_partial_failure_by_default(
        self,
        mock_macro: MagicMock,
        mock_calendar: MagicMock,
        mock_sentiment: MagicMock,
        mock_cross: MagicMock,
    ) -> None:
        mock_macro.side_effect = RuntimeError("FRED failed")
        mock_calendar.return_value = pd.DataFrame(
            {"date": [pd.Timestamp("2020-01-06")], "is_fomc_day": [0]},
        )
        mock_sentiment.return_value = pd.DataFrame(
            {"date": [pd.Timestamp("2020-01-06")], "aaii_bullish": [0.3]},
        )
        mock_cross.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-06"]),
                "ticker": ["XLK"],
                "adj_close": [200.0],
            },
        )

        with pytest.raises(RuntimeError, match="macro"):
            _build_date_features()
