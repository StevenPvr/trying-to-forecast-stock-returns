from __future__ import annotations

import numpy as np
import pandas as pd

from core.src.meta_model.features_engineering.config import (
    CROSS_SECTIONAL_BASE_FEATURES,
    INTERNAL_FEATURE_PREFIX,
    MARKET_WINDOWS,
    QUANT_FEATURE_PREFIX,
    TRADING_DAYS_PER_YEAR,
)
from core.src.meta_model.features_engineering.utils import safe_divide

MARKET_RETURN_SUM_COLUMN: str = f"{INTERNAL_FEATURE_PREFIX}market_return_sum"
MARKET_RETURN_COUNT_COLUMN: str = f"{INTERNAL_FEATURE_PREFIX}market_return_count"
MARKET_POSITIVE_COUNT_COLUMN: str = f"{INTERNAL_FEATURE_PREFIX}market_positive_count"
MARKET_DOLLAR_VOLUME_SUM_COLUMN: str = f"{INTERNAL_FEATURE_PREFIX}market_dollar_volume_sum"
MARKET_RETURN_STD_COLUMN: str = f"{INTERNAL_FEATURE_PREFIX}market_return_std"


def _as_series(value: object, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.Series(value.to_numpy(), index=value.index)
    return pd.Series(value, index=index)


def _add_cyclical_time_feature(
    data: pd.DataFrame,
    feature_name: str,
    values: pd.Series,
    period: float,
) -> None:
    angle: pd.Series = 2.0 * np.pi * values / period
    data[f"{QUANT_FEATURE_PREFIX}{feature_name}_sin"] = np.sin(angle)
    data[f"{QUANT_FEATURE_PREFIX}{feature_name}_cos"] = np.cos(angle)


def build_daily_market_aggregates(data: pd.DataFrame) -> pd.DataFrame:
    return_1d_col: str = f"{INTERNAL_FEATURE_PREFIX}return_1d"
    dollar_volume_col: str = f"{QUANT_FEATURE_PREFIX}dollar_volume"

    market_source: pd.DataFrame = pd.DataFrame({
        "date": pd.to_datetime(data["date"]),
        return_1d_col: pd.to_numeric(data[return_1d_col], errors="coerce"),
        dollar_volume_col: pd.to_numeric(data[dollar_volume_col], errors="coerce"),
    })
    market_source[f"{INTERNAL_FEATURE_PREFIX}return_sq"] = market_source[return_1d_col].pow(2)
    market_source[f"{INTERNAL_FEATURE_PREFIX}positive_return"] = market_source[return_1d_col].gt(0).astype(
        float,
    )

    aggregates: pd.DataFrame = market_source.groupby("date", sort=False).agg(
        **{
            MARKET_RETURN_SUM_COLUMN: (return_1d_col, "sum"),
            MARKET_RETURN_COUNT_COLUMN: (return_1d_col, "count"),
            MARKET_POSITIVE_COUNT_COLUMN: (f"{INTERNAL_FEATURE_PREFIX}positive_return", "sum"),
            MARKET_DOLLAR_VOLUME_SUM_COLUMN: (dollar_volume_col, "sum"),
            f"{INTERNAL_FEATURE_PREFIX}market_return_sum_sq": (f"{INTERNAL_FEATURE_PREFIX}return_sq", "sum"),
        },
    ).reset_index()
    market_return_count = pd.Series(
        pd.to_numeric(aggregates[MARKET_RETURN_COUNT_COLUMN], errors="coerce"),
        index=aggregates.index,
    )
    market_return_sum = pd.Series(
        pd.to_numeric(aggregates[MARKET_RETURN_SUM_COLUMN], errors="coerce"),
        index=aggregates.index,
    )
    market_return_sum_sq = pd.Series(
        pd.to_numeric(
            aggregates[f"{INTERNAL_FEATURE_PREFIX}market_return_sum_sq"],
            errors="coerce",
        ),
        index=aggregates.index,
    )
    variance_numerator = (
        market_return_sum_sq
        - ((market_return_sum * market_return_sum) / market_return_count)
    )
    sample_variance = pd.Series(
        safe_divide(variance_numerator, market_return_count - 1.0),
        index=aggregates.index,
    )
    aggregates[MARKET_RETURN_STD_COLUMN] = pd.Series(
        np.sqrt(sample_variance.clip(lower=0.0).to_numpy()),
        index=aggregates.index,
    )
    aggregates.loc[market_return_count <= 1.0, MARKET_RETURN_STD_COLUMN] = np.nan

    return aggregates.drop(
        columns=[f"{INTERNAL_FEATURE_PREFIX}market_return_sum_sq"],
        errors="ignore",
    )


def add_universe_market_features_for_ticker(
    data: pd.DataFrame,
    daily_market_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    enriched: pd.DataFrame = data.sort_values("date").reset_index(drop=True).copy()
    enriched = enriched.merge(
        daily_market_aggregates,
        on="date",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    data_index: pd.Index = enriched.index

    return_1d_col: str = f"{INTERNAL_FEATURE_PREFIX}return_1d"
    log_return_1d_col: str = f"{INTERNAL_FEATURE_PREFIX}log_return_1d"
    dollar_volume_col: str = f"{QUANT_FEATURE_PREFIX}dollar_volume"

    stock_return_all: pd.Series = _as_series(enriched[return_1d_col], data_index)
    dollar_volume_all: pd.Series = _as_series(enriched[dollar_volume_col], data_index)
    daily_return_sum: pd.Series = _as_series(enriched[MARKET_RETURN_SUM_COLUMN], data_index)
    daily_return_count: pd.Series = _as_series(enriched[MARKET_RETURN_COUNT_COLUMN], data_index)
    daily_positive_count: pd.Series = _as_series(enriched[MARKET_POSITIVE_COUNT_COLUMN], data_index)
    daily_dollar_volume_sum: pd.Series = _as_series(enriched[MARKET_DOLLAR_VOLUME_SUM_COLUMN], data_index)

    universe_return_ex_self: pd.Series = _as_series(
        safe_divide(daily_return_sum - stock_return_all, daily_return_count - 1),
        data_index,
    )
    enriched[f"{QUANT_FEATURE_PREFIX}universe_return_1d_ex_self"] = universe_return_ex_self
    enriched[f"{QUANT_FEATURE_PREFIX}universe_positive_share_1d_ex_self"] = _as_series(
        safe_divide(
            daily_positive_count - stock_return_all.gt(0).astype(float),
            daily_return_count - 1,
        ),
        data_index,
    )
    enriched[f"{QUANT_FEATURE_PREFIX}universe_dispersion_1d"] = _as_series(enriched[MARKET_RETURN_STD_COLUMN], data_index)
    enriched[f"{QUANT_FEATURE_PREFIX}volume_share_of_universe"] = _as_series(
        safe_divide(dollar_volume_all, daily_dollar_volume_sum),
        data_index,
    )
    enriched[f"{QUANT_FEATURE_PREFIX}excess_return_1d"] = (
        stock_return_all - universe_return_ex_self
    )

    for ticker, group in enriched.groupby("ticker", sort=False):
        idx = group.index
        market_return: pd.Series = _as_series(
            group[f"{QUANT_FEATURE_PREFIX}universe_return_1d_ex_self"],
            idx,
        )
        stock_return: pd.Series = _as_series(group[return_1d_col], idx)
        log_return: pd.Series = _as_series(group[log_return_1d_col], idx)

        for window in MARKET_WINDOWS:
            rolling_covariance: pd.Series = _as_series(
                stock_return.rolling(window).cov(market_return),
                idx,
            )
            market_variance: pd.Series = _as_series(
                market_return.rolling(window).var(),
                idx,
            )
            beta: pd.Series = _as_series(
                safe_divide(rolling_covariance, market_variance),
                idx,
            )
            correlation: pd.Series = _as_series(stock_return.rolling(window).corr(market_return), idx)
            idio_return: pd.Series = _as_series(stock_return - (beta * market_return), idx)
            market_log_return: pd.Series = _as_series(np.log1p(market_return.to_numpy()), idx)
            relative_strength: pd.Series = _as_series(
                log_return.rolling(window).sum() - market_log_return.rolling(window).sum(),
                idx,
            )

            enriched.loc[idx, f"{QUANT_FEATURE_PREFIX}beta_{window}d"] = beta.to_numpy()
            enriched.loc[idx, f"{QUANT_FEATURE_PREFIX}correlation_to_universe_{window}d"] = (
                correlation.to_numpy()
            )
            enriched.loc[idx, f"{QUANT_FEATURE_PREFIX}idiosyncratic_vol_{window}d"] = (
                idio_return.rolling(window).std().to_numpy() * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
            enriched.loc[idx, f"{QUANT_FEATURE_PREFIX}relative_strength_{window}d"] = (
                relative_strength.to_numpy()
            )

    return enriched.drop(
        columns=[
            MARKET_RETURN_SUM_COLUMN,
            MARKET_RETURN_COUNT_COLUMN,
            MARKET_POSITIVE_COUNT_COLUMN,
            MARKET_DOLLAR_VOLUME_SUM_COLUMN,
            MARKET_RETURN_STD_COLUMN,
        ],
        errors="ignore",
    )


def add_universe_market_features(data: pd.DataFrame) -> pd.DataFrame:
    daily_market_aggregates: pd.DataFrame = build_daily_market_aggregates(data)
    enriched_groups: list[pd.DataFrame] = []
    for _, group in data.groupby("ticker", sort=False):
        enriched_groups.append(
            add_universe_market_features_for_ticker(group, daily_market_aggregates),
        )
    return pd.concat(enriched_groups, ignore_index=True)


def add_cross_sectional_features(data: pd.DataFrame) -> pd.DataFrame:
    enriched: pd.DataFrame = data.copy()

    for feature_name in CROSS_SECTIONAL_BASE_FEATURES:
        if feature_name not in enriched.columns:
            continue

        feature_slug: str = feature_name.removeprefix(QUANT_FEATURE_PREFIX)
        date_group = enriched.groupby("date")[feature_name]
        feature_values: pd.Series = _as_series(enriched[feature_name], enriched.index)
        mean_by_date: pd.Series = _as_series(date_group.transform("mean"), enriched.index)
        std_by_date: pd.Series = _as_series(date_group.transform("std"), enriched.index)

        enriched[f"{QUANT_FEATURE_PREFIX}cs_rank_{feature_slug}"] = _as_series(
            date_group.rank(
                pct=True,
                method="average",
            ),
            enriched.index,
        )
        enriched[f"{QUANT_FEATURE_PREFIX}cs_zscore_{feature_slug}"] = _as_series(
            safe_divide(feature_values - mean_by_date, std_by_date),
            enriched.index,
        )

    return enriched


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    enriched: pd.DataFrame = data.copy()
    dates: pd.Series = pd.to_datetime(enriched["date"])

    day_of_week: pd.Series = dates.dt.dayofweek.astype(float)
    month_of_year: pd.Series = dates.dt.month.astype(float)
    day_of_month: pd.Series = dates.dt.day.astype(float)
    day_of_year: pd.Series = dates.dt.dayofyear.astype(float)

    _add_cyclical_time_feature(enriched, "day_of_week", day_of_week, 7.0)
    _add_cyclical_time_feature(enriched, "month_of_year", month_of_year, 12.0)
    _add_cyclical_time_feature(enriched, "day_of_month", day_of_month, 31.0)
    _add_cyclical_time_feature(enriched, "day_of_year", day_of_year, 366.0)
    enriched[f"{QUANT_FEATURE_PREFIX}is_month_start"] = dates.dt.is_month_start.astype(int)
    enriched[f"{QUANT_FEATURE_PREFIX}is_month_end"] = dates.dt.is_month_end.astype(int)
    enriched[f"{QUANT_FEATURE_PREFIX}is_quarter_start"] = dates.dt.is_quarter_start.astype(int)
    enriched[f"{QUANT_FEATURE_PREFIX}is_quarter_end"] = dates.dt.is_quarter_end.astype(int)

    return enriched


def drop_internal_columns(data: pd.DataFrame) -> pd.DataFrame:
    internal_columns: list[str] = [
        column for column in data.columns if column.startswith(INTERNAL_FEATURE_PREFIX)
    ]
    return data.drop(columns=internal_columns, errors="ignore")
