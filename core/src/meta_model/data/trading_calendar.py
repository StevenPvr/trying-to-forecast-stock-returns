from __future__ import annotations

"""NYSE trading calendar utilities: session lookup, date normalisation."""

from datetime import date, datetime
from typing import Sequence, cast

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USThanksgivingDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

_TRADING_DAY = CustomBusinessDay()


def _normalize_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return timestamp.floor("D")


def _normalize_datetime_index(
    values: pd.Index | Sequence[object] | np.ndarray,
) -> pd.DatetimeIndex:
    datetime_values = pd.to_datetime(values if isinstance(values, (pd.Index, np.ndarray)) else list(values))
    timestamps = [_normalize_timestamp(value) for value in datetime_values.tolist()]
    return pd.DatetimeIndex(timestamps)


class _NyseHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "JuneteenthNationalIndependenceDay",
            month=6,
            day=19,
            observance=nearest_workday,
            start_date="2022-06-19",
        ),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_SPECIAL_CLOSURES: frozenset[pd.Timestamp] = frozenset({
    _normalize_timestamp("2004-06-11"),  # Reagan funeral
    _normalize_timestamp("2007-01-02"),  # Ford funeral
    _normalize_timestamp("2012-10-29"),  # Hurricane Sandy
    _normalize_timestamp("2012-10-30"),  # Hurricane Sandy
    _normalize_timestamp("2018-12-05"),  # George H. W. Bush funeral
    _normalize_timestamp("2025-01-09"),  # Jimmy Carter funeral
})


def get_nyse_sessions(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> pd.DatetimeIndex:
    start: pd.Timestamp = _normalize_timestamp(start_date)
    end: pd.Timestamp = _normalize_timestamp(end_date)
    calendar = _NyseHolidayCalendar()
    day_offset = CustomBusinessDay(calendar=calendar)
    sessions: pd.DatetimeIndex = pd.date_range(start=start, end=end, freq=day_offset)
    special_closures: pd.DatetimeIndex = pd.DatetimeIndex(
        sorted(
            closure
            for closure in _NYSE_SPECIAL_CLOSURES
            if start <= closure <= end
        ),
    )
    if len(special_closures) == 0:
        return sessions
    return cast(pd.DatetimeIndex, sessions.difference(special_closures))


def shift_series_to_session_availability(
    series: pd.Series,  # type: ignore[type-arg]
    sessions: pd.DatetimeIndex,
    lag_sessions: int,
) -> pd.Series:  # type: ignore[type-arg]
    """Shift observations to the first session where they are safely available.

    The lag is expressed in trading sessions strictly after the observation date.
    """
    if lag_sessions < 1:
        raise ValueError("lag_sessions must be >= 1")
    if series.empty:
        return series.copy()

    clean_series = cast(pd.Series, series.dropna().copy())
    if clean_series.empty:
        return clean_series

    observation_index: pd.DatetimeIndex = _normalize_datetime_index(clean_series.index)
    session_index: pd.DatetimeIndex = _normalize_datetime_index(sessions)

    insertion_points: np.ndarray = session_index.searchsorted(
        observation_index,
        side="right",
    )
    target_positions: np.ndarray = insertion_points + (lag_sessions - 1)
    valid_mask: np.ndarray = target_positions < len(session_index)

    if not valid_mask.any():
        return pd.Series(dtype=clean_series.dtype, name=clean_series.name)

    availability_index: pd.DatetimeIndex = session_index.take(target_positions[valid_mask])
    shifted: pd.Series = pd.Series(
        clean_series.to_numpy()[valid_mask],
        index=availability_index,
        name=clean_series.name,
    )
    return cast(pd.Series, shifted.groupby(level=0).last().sort_index())
