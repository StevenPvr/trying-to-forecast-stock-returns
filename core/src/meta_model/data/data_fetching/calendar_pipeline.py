from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.data.constants import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    RANDOM_SEED,
    SAMPLE_FRAC,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import get_nyse_sessions

LOGGER: logging.Logger = logging.getLogger(__name__)


def _as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return timestamp

# ---------------------------------------------------------------------------
# FOMC statement release dates 2004-2025 (historical facts)
# ---------------------------------------------------------------------------

FOMC_MEETING_DATES: frozenset[str] = frozenset({
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30",
    "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-08",
    "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CalendarConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    output_dir: Path = DATA_FETCHING_DIR
    sample_frac: float = SAMPLE_FRAC
    random_seed: int = RANDOM_SEED


# ---------------------------------------------------------------------------
# FOMC helpers
# ---------------------------------------------------------------------------


def _get_fomc_dates(start_date: str, end_date: str) -> list[pd.Timestamp]:
    """Filter FOMC_MEETING_DATES to the [start, end] range, sorted."""
    start: pd.Timestamp = _as_timestamp(start_date)
    end: pd.Timestamp = _as_timestamp(end_date)
    filtered: list[pd.Timestamp] = sorted(
        _as_timestamp(d) for d in FOMC_MEETING_DATES
        if start <= _as_timestamp(d) <= end
    )
    LOGGER.debug("FOMC dates in range: %d", len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Quad witching helpers
# ---------------------------------------------------------------------------


def _get_quad_witching_dates(
    start_date: str, end_date: str,
) -> list[pd.Timestamp]:
    """Quad witching = 3rd Friday of March, June, September, December."""
    start: pd.Timestamp = _as_timestamp(start_date)
    end: pd.Timestamp = _as_timestamp(end_date)
    qw_months: tuple[int, ...] = (3, 6, 9, 12)
    results: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in qw_months:
            third_friday: pd.Timestamp = _third_friday(year, month)
            if start <= third_friday <= end:
                results.append(third_friday)
    LOGGER.debug("Quad witching dates in range: %d", len(results))
    return sorted(results)


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the 3rd Friday of the given year/month."""
    first_day: pd.Timestamp = _as_timestamp(pd.Timestamp(year=year, month=month, day=1))
    weekday: int = first_day.weekday()  # Monday=0 ... Friday=4
    first_friday_offset: int = (4 - weekday) % 7
    third_friday_day: int = 1 + first_friday_offset + 14
    return _as_timestamp(pd.Timestamp(year=year, month=month, day=third_friday_day))


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _compute_fomc_features(
    bdays: pd.DatetimeIndex, fomc_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute FOMC-related features for each business day."""
    fomc_arr: np.ndarray = np.array(fomc_dates, dtype="datetime64[ns]")
    bdays_arr: np.ndarray = bdays.values

    is_fomc: np.ndarray = np.isin(bdays_arr, fomc_arr)

    days_to_next: np.ndarray = _days_to_next_event(bdays_arr, fomc_arr)
    days_since_last: np.ndarray = _days_since_last_event(bdays_arr, fomc_arr)

    is_fomc_week: np.ndarray = _is_same_week(bdays_arr, fomc_arr)

    return pd.DataFrame({
        "is_fomc_day": is_fomc.astype(int),
        "days_to_next_fomc": days_to_next,
        "days_since_last_fomc": days_since_last,
        "is_fomc_week": is_fomc_week.astype(int),
    }, index=bdays)


def _compute_quad_witching_features(
    bdays: pd.DatetimeIndex, qw_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute quad-witching-related features for each business day."""
    qw_arr: np.ndarray = np.array(qw_dates, dtype="datetime64[ns]")
    bdays_arr: np.ndarray = bdays.values

    is_qw: np.ndarray = np.isin(bdays_arr, qw_arr)
    days_to_next: np.ndarray = _days_to_next_event(bdays_arr, qw_arr)

    return pd.DataFrame({
        "is_quad_witching": is_qw.astype(int),
        "days_to_next_quad_witching": days_to_next,
    }, index=bdays)


# ---------------------------------------------------------------------------
# Vectorised calendar-day helpers (np.searchsorted)
# ---------------------------------------------------------------------------


def _days_to_next_event(
    bdays: np.ndarray, events: np.ndarray,
) -> np.ndarray:
    """Calendar days until the next event (0 on event day itself)."""
    idx: np.ndarray = np.searchsorted(events, bdays, side="left")
    result: np.ndarray = np.full(len(bdays), -1, dtype=int)
    valid: np.ndarray = idx < len(events)
    diff: np.ndarray = (
        events[idx[valid]] - bdays[valid]
    ).astype("timedelta64[D]").astype(int)
    result[valid] = diff
    return result


def _days_since_last_event(
    bdays: np.ndarray, events: np.ndarray,
) -> np.ndarray:
    """Calendar days since the last event (0 on event day itself)."""
    idx: np.ndarray = np.searchsorted(events, bdays, side="right") - 1
    result: np.ndarray = np.full(len(bdays), -1, dtype=int)
    valid: np.ndarray = idx >= 0
    diff: np.ndarray = (
        bdays[valid] - events[idx[valid]]
    ).astype("timedelta64[D]").astype(int)
    result[valid] = diff
    return result


def _is_same_week(
    bdays: np.ndarray, events: np.ndarray,
) -> np.ndarray:
    """True if any event falls in the same ISO week as the business day."""
    bday_weeks = np.asarray(
        pd.DatetimeIndex(bdays).isocalendar()[["year", "week"]].values,
    )
    event_weeks_df = cast(
        pd.DataFrame,
        pd.DatetimeIndex(events).isocalendar()[["year", "week"]],
    )
    event_week_set: set[tuple[int, int]] = {
        (int(row[0]), int(row[1])) for row in event_weeks_df.values
    }
    return np.array(
        [(int(r[0]), int(r[1])) in event_week_set for r in bday_weeks],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------


def build_calendar_dataset(config: CalendarConfig) -> pd.DataFrame:
    """Build the full calendar features DataFrame."""
    bdays: pd.DatetimeIndex = get_nyse_sessions(
        config.start_date, config.end_date,
    )
    fomc_dates: list[pd.Timestamp] = _get_fomc_dates(
        config.start_date, config.end_date,
    )
    qw_dates: list[pd.Timestamp] = _get_quad_witching_dates(
        config.start_date, config.end_date,
    )
    fomc_df: pd.DataFrame = _compute_fomc_features(bdays, fomc_dates)
    qw_df: pd.DataFrame = _compute_quad_witching_features(bdays, qw_dates)

    result: pd.DataFrame = fomc_df.join(qw_df)
    result = result.reset_index().rename(columns={"index": "date"})
    result["date"] = pd.to_datetime(result["date"])
    LOGGER.info(
        "Built calendar dataset: %d rows x %d columns",
        len(result), len(result.columns),
    )
    return result
