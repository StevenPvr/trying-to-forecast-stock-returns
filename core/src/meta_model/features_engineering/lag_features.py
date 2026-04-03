from __future__ import annotations

import logging

import pandas as pd

from core.src.meta_model.features_engineering.config import (
    BROKER_FEATURE_PREFIX,
    CALENDAR_FEATURE_PREFIX,
    CALENDAR_SINCE_LAG_WINDOWS,
    COMPANY_FEATURE_LAG_WINDOWS,
    COMPANY_FEATURE_PREFIX,
    CROSS_ASSET_FEATURE_PREFIX,
    DEEP_FEATURE_PREFIX,
    EARNINGS_FEATURE_PREFIX,
    FEATURE_LAG_WINDOWS,
    MACRO_FEATURE_PREFIX,
    NON_LAGGABLE_QUANT_FEATURES,
    NON_LAGGABLE_TA_PREFIXES,
    OPEN_FEATURE_PREFIX,
    QUANT_FEATURE_PREFIX,
    SECTOR_FEATURE_PREFIX,
    SIGNAL_FEATURE_PREFIX,
    SENTIMENT_FEATURE_PREFIX,
    SLOW_FEATURE_LAG_WINDOWS,
    STOCK_LOG_RETURN_COLUMNS,
    TA_FEATURE_PREFIX,
)

LOGGER: logging.Logger = logging.getLogger(__name__)


def _is_laggable_ta_feature(column: str) -> bool:
    return column.startswith(TA_FEATURE_PREFIX) and not column.startswith(
        NON_LAGGABLE_TA_PREFIXES,
    )


def _is_calendar_since_feature(column: str) -> bool:
    return column.startswith(CALENDAR_FEATURE_PREFIX) and "days_since_" in column


def get_lag_windows_for_feature(column: str) -> tuple[int, ...]:
    if column in {"date", "ticker"} or "_lag_" in column:
        return ()
    if column in STOCK_LOG_RETURN_COLUMNS:
        return FEATURE_LAG_WINDOWS
    if _is_laggable_ta_feature(column):
        return FEATURE_LAG_WINDOWS
    if column.startswith(QUANT_FEATURE_PREFIX) and column not in NON_LAGGABLE_QUANT_FEATURES:
        return FEATURE_LAG_WINDOWS
    if column.startswith(DEEP_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(BROKER_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(SECTOR_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(OPEN_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(EARNINGS_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(SIGNAL_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(CROSS_ASSET_FEATURE_PREFIX):
        return FEATURE_LAG_WINDOWS
    if column.startswith(COMPANY_FEATURE_PREFIX):
        return COMPANY_FEATURE_LAG_WINDOWS
    if column.startswith(SENTIMENT_FEATURE_PREFIX):
        return SLOW_FEATURE_LAG_WINDOWS
    if column.startswith(MACRO_FEATURE_PREFIX):
        return SLOW_FEATURE_LAG_WINDOWS
    if column.startswith(CALENDAR_FEATURE_PREFIX):
        if _is_calendar_since_feature(column):
            return CALENDAR_SINCE_LAG_WINDOWS
        return ()
    return ()


def _is_numeric_feature_column(data: pd.DataFrame, column: str) -> bool:
    if column not in data.columns:
        return False
    return bool(pd.api.types.is_numeric_dtype(data[column]))


def get_laggable_feature_columns(
    columns: list[str] | pd.Index,
    data: pd.DataFrame | None = None,
) -> list[str]:
    laggable_columns = [column for column in columns if get_lag_windows_for_feature(column)]
    if data is None:
        return laggable_columns
    return [
        column
        for column in laggable_columns
        if _is_numeric_feature_column(data, column)
    ]


def downcast_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    optimized: pd.DataFrame = data.copy()
    for column in optimized.columns:
        series = optimized[column]
        if pd.api.types.is_float_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="integer")
    return optimized


def build_lagged_feature_group(
    data: pd.DataFrame,
    laggable_columns: list[str] | None = None,
) -> pd.DataFrame:
    lagged_data: pd.DataFrame = data.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    selected_laggable_columns: list[str] = (
        laggable_columns
        if laggable_columns is not None
        else get_laggable_feature_columns(list(lagged_data.columns), lagged_data)
    )
    selected_laggable_columns = [
        column
        for column in selected_laggable_columns
        if _is_numeric_feature_column(lagged_data, column)
    ]

    if not selected_laggable_columns:
        return lagged_data.sort_values(["date", "ticker"]).reset_index(drop=True)

    lagged_frames: list[pd.DataFrame] = []
    columns_by_window: dict[tuple[int, ...], list[str]] = {}
    for column in selected_laggable_columns:
        windows = get_lag_windows_for_feature(column)
        if not windows:
            continue
        columns_by_window.setdefault(windows, []).append(column)

    for lag_windows, grouped_columns in columns_by_window.items():
        for lag in lag_windows:
            shifted = pd.DataFrame(lagged_data[grouped_columns].shift(lag))
            shifted = shifted.astype("float32")
            shifted.columns = [f"{column}_lag_{lag}d" for column in grouped_columns]
            lagged_frames.append(shifted)

    lagged_feature_block: pd.DataFrame = pd.concat(lagged_frames, axis=1)
    lagged_group: pd.DataFrame = pd.concat([lagged_data, lagged_feature_block], axis=1)
    return lagged_group.sort_values(["date", "ticker"]).reset_index(drop=True)


def add_feature_lags(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return build_lagged_feature_group(data)

    lagged_groups: list[pd.DataFrame] = []
    laggable_columns: list[str] = get_laggable_feature_columns(list(data.columns), data)
    for _, group in data.groupby("ticker", sort=False):
        lagged_groups.append(
            build_lagged_feature_group(group, laggable_columns=laggable_columns),
        )
    lagged_data: pd.DataFrame = pd.concat(lagged_groups, ignore_index=True)
    LOGGER.info(
        "Added %d lagged feature columns across adaptive lag windows.",
        sum(len(get_lag_windows_for_feature(column)) for column in laggable_columns),
    )
    return lagged_data.sort_values(["date", "ticker"]).reset_index(drop=True)
