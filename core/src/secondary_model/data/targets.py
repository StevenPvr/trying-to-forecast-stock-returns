from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.src.meta_model.data.data_preprocessing.main import TARGET_COLUMN

LOGGER: logging.Logger = logging.getLogger(__name__)

TARGET_HORIZON_DAYS: int = 5
VOLUME_BASELINE_WINDOW_DAYS: int = 20
MISSING_STOCK_CLOSE_PRICE_ERROR: str = "Missing required column for target creation: stock_close_price"
MISSING_STOCK_TRADING_VOLUME_ERROR: str = "Missing required column for target creation: stock_trading_volume"


@dataclass(frozen=True)
class SecondaryTargetSpec:
    name: str
    build_target: Callable[[pd.DataFrame], pd.DataFrame]
    required_metadata_columns: tuple[str, ...]


def _future_shift_matrix(series: pd.Series, horizon_days: int) -> pd.DataFrame:
    return pd.concat(
        [series.shift(-offset) for offset in range(1, horizon_days + 1)],
        axis=1,
    )


def _create_grouped_target(
    data: pd.DataFrame,
    target_builder: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    ordered = pd.DataFrame(data.sort_values(["ticker", "date"]).reset_index(drop=True))
    target_parts: list[pd.DataFrame] = []
    for _, group in ordered.groupby("ticker", sort=False):
        ticker_group = group.copy()
        ticker_group[TARGET_COLUMN] = target_builder(ticker_group)
        target_parts.append(ticker_group)
    return pd.concat(target_parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def create_future_trend_target(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    if "stock_close_price" not in data.columns:
        raise ValueError(MISSING_STOCK_CLOSE_PRICE_ERROR)

    def _build_target(group: pd.DataFrame) -> pd.Series:
        close = group["stock_close_price"]
        return pd.Series(np.log(close.shift(-horizon_days) / close), index=group.index)

    LOGGER.info("Creating secondary target future_trend_5d.")
    return _create_grouped_target(data, _build_target)


def create_future_realized_vol_target(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    if "stock_close_price" not in data.columns:
        raise ValueError(MISSING_STOCK_CLOSE_PRICE_ERROR)

    def _build_target(group: pd.DataFrame) -> pd.Series:
        close = group["stock_close_price"]
        daily_log_returns = pd.Series(np.log(close / close.shift(1)), index=group.index)
        future_returns = _future_shift_matrix(daily_log_returns, horizon_days)
        target = future_returns.std(axis=1, ddof=0)
        target[future_returns.notna().sum(axis=1) < horizon_days] = np.nan
        return target

    LOGGER.info("Creating secondary target future_realized_vol_5d.")
    return _create_grouped_target(data, _build_target)


def create_future_volume_regime_target(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
    baseline_window_days: int = VOLUME_BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    if "stock_trading_volume" not in data.columns:
        raise ValueError(MISSING_STOCK_TRADING_VOLUME_ERROR)

    def _build_target(group: pd.DataFrame) -> pd.Series:
        volume = pd.Series(group["stock_trading_volume"], index=group.index, dtype=float)
        future_matrix = _future_shift_matrix(volume, horizon_days)
        future_mean = future_matrix.mean(axis=1)
        future_mean[future_matrix.notna().sum(axis=1) < horizon_days] = np.nan
        trailing_mean = volume.rolling(window=baseline_window_days, min_periods=baseline_window_days).mean()
        return pd.Series(np.log(future_mean / trailing_mean), index=group.index)

    LOGGER.info("Creating secondary target future_volume_regime_5d.")
    return _create_grouped_target(data, _build_target)


def create_future_drawdown_target(
    data: pd.DataFrame,
    horizon_days: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    if "stock_close_price" not in data.columns:
        raise ValueError(MISSING_STOCK_CLOSE_PRICE_ERROR)

    def _build_target(group: pd.DataFrame) -> pd.Series:
        close = pd.Series(group["stock_close_price"], index=group.index, dtype=float)
        future_returns = pd.concat(
            [pd.Series(np.log(close.shift(-offset) / close), index=group.index) for offset in range(1, horizon_days + 1)],
            axis=1,
        )
        target = future_returns.min(axis=1)
        target[future_returns.notna().sum(axis=1) < horizon_days] = np.nan
        return target

    LOGGER.info("Creating secondary target future_drawdown_5d.")
    return _create_grouped_target(data, _build_target)


SECONDARY_TARGET_SPECS: tuple[SecondaryTargetSpec, ...] = (
    SecondaryTargetSpec(
        name="future_trend_5d",
        build_target=create_future_trend_target,
        required_metadata_columns=("stock_close_price",),
    ),
    SecondaryTargetSpec(
        name="future_realized_vol_5d",
        build_target=create_future_realized_vol_target,
        required_metadata_columns=("stock_close_price",),
    ),
    SecondaryTargetSpec(
        name="future_volume_regime_5d",
        build_target=create_future_volume_regime_target,
        required_metadata_columns=("stock_close_price", "stock_trading_volume"),
    ),
    SecondaryTargetSpec(
        name="future_drawdown_5d",
        build_target=create_future_drawdown_target,
        required_metadata_columns=("stock_close_price",),
    ),
)

