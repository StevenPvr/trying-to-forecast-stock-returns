# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.data.data_fetching.calendar_pipeline import (
    CalendarConfig,
    _compute_fomc_features,
    _compute_quad_witching_features,
    _get_fomc_dates,
    _get_quad_witching_dates,
    build_calendar_dataset,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import get_nyse_sessions


def _ts(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


# ---------------------------------------------------------------------------
# CalendarConfig
# ---------------------------------------------------------------------------


class TestCalendarConfig:
    def test_default_values(self) -> None:
        config: CalendarConfig = CalendarConfig()
        assert config.start_date == "2004-01-01"
        assert config.end_date == "2025-12-31"
        assert config.output_dir == DATA_FETCHING_DIR
        assert config.sample_frac == 0.05
        assert config.random_seed == 7

    def test_frozen(self) -> None:
        config: CalendarConfig = CalendarConfig()
        with pytest.raises(AttributeError):
            config.start_date = "2010-01-01"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config: CalendarConfig = CalendarConfig(
            start_date="2020-01-01", end_date="2020-12-31",
        )
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2020-12-31"


# ---------------------------------------------------------------------------
# _get_fomc_dates
# ---------------------------------------------------------------------------


class TestGetFomcDates:
    def test_returns_dates_in_range(self) -> None:
        dates: list[pd.Timestamp] = _get_fomc_dates("2020-01-01", "2020-12-31")
        assert len(dates) > 0
        for d in dates:
            assert pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2020-12-31")

    def test_empty_when_no_meetings(self) -> None:
        dates: list[pd.Timestamp] = _get_fomc_dates("1990-01-01", "1990-12-31")
        assert len(dates) == 0

    def test_known_count_2020(self) -> None:
        """2020 had 9 FOMC meetings including emergency ones."""
        dates: list[pd.Timestamp] = _get_fomc_dates("2020-01-01", "2020-12-31")
        assert len(dates) == 9

    def test_sorted_output(self) -> None:
        dates: list[pd.Timestamp] = _get_fomc_dates("2018-01-01", "2022-12-31")
        for i in range(len(dates) - 1):
            assert dates[i] < dates[i + 1]


# ---------------------------------------------------------------------------
# _get_quad_witching_dates
# ---------------------------------------------------------------------------


class TestGetQuadWitchingDates:
    def test_always_third_friday(self) -> None:
        dates: list[pd.Timestamp] = _get_quad_witching_dates(
            "2020-01-01", "2020-12-31",
        )
        for d in dates:
            assert d.weekday() == 4  # Friday
            assert 15 <= d.day <= 21  # 3rd week

    def test_exactly_four_per_year(self) -> None:
        dates: list[pd.Timestamp] = _get_quad_witching_dates(
            "2020-01-01", "2020-12-31",
        )
        assert len(dates) == 4

    def test_correct_months(self) -> None:
        dates: list[pd.Timestamp] = _get_quad_witching_dates(
            "2020-01-01", "2020-12-31",
        )
        months: list[int] = [d.month for d in dates]
        assert months == [3, 6, 9, 12]

    def test_known_date_2020_march(self) -> None:
        """3rd Friday of March 2020 is March 20."""
        dates: list[pd.Timestamp] = _get_quad_witching_dates(
            "2020-03-01", "2020-03-31",
        )
        assert len(dates) == 1
        assert dates[0] == pd.Timestamp("2020-03-20")


# ---------------------------------------------------------------------------
# _compute_fomc_features
# ---------------------------------------------------------------------------


class TestComputeFomcFeatures:
    def test_is_fomc_day_on_known_date(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-01-27", "2020-01-31")
        fomc: list[pd.Timestamp] = [_ts("2020-01-29")]
        df: pd.DataFrame = _compute_fomc_features(bdays, fomc)
        assert df.loc[pd.Timestamp("2020-01-29"), "is_fomc_day"] == 1
        assert df.loc[pd.Timestamp("2020-01-27"), "is_fomc_day"] == 0

    def test_days_to_next_fomc_zero_on_day(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-01-27", "2020-01-31")
        fomc: list[pd.Timestamp] = [_ts("2020-01-29")]
        df: pd.DataFrame = _compute_fomc_features(bdays, fomc)
        assert df.loc[pd.Timestamp("2020-01-29"), "days_to_next_fomc"] == 0

    def test_is_fomc_week_correct(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-01-20", "2020-02-07")
        fomc: list[pd.Timestamp] = [_ts("2020-01-29")]
        df: pd.DataFrame = _compute_fomc_features(bdays, fomc)
        # 2020-01-27 (Mon) to 2020-01-31 (Fri) is same week as Jan 29
        assert df.loc[pd.Timestamp("2020-01-27"), "is_fomc_week"] == 1
        assert df.loc[pd.Timestamp("2020-01-31"), "is_fomc_week"] == 1
        # 2020-01-20 is the week before
        assert df.loc[pd.Timestamp("2020-01-20"), "is_fomc_week"] == 0

    def test_days_since_last_fomc_increases(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-01-29", "2020-02-05")
        fomc: list[pd.Timestamp] = [_ts("2020-01-29")]
        df: pd.DataFrame = _compute_fomc_features(bdays, fomc)
        vals: list[int] = df["days_since_last_fomc"].tolist()
        # Should be 0 on Jan 29, then increasing each calendar day
        assert vals[0] == 0
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1]


# ---------------------------------------------------------------------------
# _compute_quad_witching_features
# ---------------------------------------------------------------------------


class TestComputeQuadWitchingFeatures:
    def test_is_quad_witching_correct(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-03-16", "2020-03-20")
        qw: list[pd.Timestamp] = [_ts("2020-03-20")]
        df: pd.DataFrame = _compute_quad_witching_features(bdays, qw)
        assert df.loc[pd.Timestamp("2020-03-20"), "is_quad_witching"] == 1
        assert df.loc[pd.Timestamp("2020-03-16"), "is_quad_witching"] == 0

    def test_days_to_next_counts_down(self) -> None:
        bdays: pd.DatetimeIndex = pd.bdate_range("2020-03-16", "2020-03-20")
        qw: list[pd.Timestamp] = [_ts("2020-03-20")]
        df: pd.DataFrame = _compute_quad_witching_features(bdays, qw)
        vals: list[int] = df["days_to_next_quad_witching"].tolist()
        # Should count down to 0
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1]
        assert vals[-1] == 0


# ---------------------------------------------------------------------------
# build_calendar_dataset
# ---------------------------------------------------------------------------


class TestBuildCalendarDataset:
    def test_correct_columns(self) -> None:
        config: CalendarConfig = CalendarConfig(
            start_date="2020-01-01", end_date="2020-03-31",
        )
        df: pd.DataFrame = build_calendar_dataset(config)
        expected_cols: list[str] = [
            "date", "is_fomc_day", "days_to_next_fomc",
            "days_since_last_fomc", "is_fomc_week",
            "is_quad_witching", "days_to_next_quad_witching",
        ]
        assert list(df.columns) == expected_cols

    def test_date_range_matches_nyse_sessions(self) -> None:
        config: CalendarConfig = CalendarConfig(
            start_date="2020-01-01", end_date="2020-06-30",
        )
        df: pd.DataFrame = build_calendar_dataset(config)
        sessions: pd.DatetimeIndex = get_nyse_sessions("2020-01-01", "2020-06-30")
        assert len(df) == len(sessions)

    def test_no_nan_in_output(self) -> None:
        config: CalendarConfig = CalendarConfig(
            start_date="2010-01-01", end_date="2010-12-31",
        )
        df: pd.DataFrame = build_calendar_dataset(config)
        assert df.isna().sum().sum() == 0

    def test_date_dtype(self) -> None:
        config: CalendarConfig = CalendarConfig(
            start_date="2020-01-01", end_date="2020-01-31",
        )
        df: pd.DataFrame = build_calendar_dataset(config)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])



# --- Execution directe (clic dans l'IDE) ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
