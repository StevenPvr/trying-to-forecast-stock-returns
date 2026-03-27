from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from core.src.meta_model.features_engineering.config import TRADING_DAYS_PER_YEAR


def safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
) -> pd.Series | float:
    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        denominator = denominator.where(denominator != 0, np.nan)
    elif denominator == 0:
        denominator = np.nan
    return cast(pd.Series | float, numerator / denominator)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = cast(pd.Series, series.rolling(window).mean())
    rolling_std = cast(pd.Series, series.rolling(window).std())
    return cast(pd.Series, safe_divide(series - rolling_mean, rolling_std))


def annualized_trend_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x: np.ndarray = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(np.expm1(slope * TRADING_DAYS_PER_YEAR))


def trend_r2(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x: np.ndarray = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted: np.ndarray = intercept + slope * x
    ss_tot: float = float(np.square(values - values.mean()).sum())
    if ss_tot == 0:
        return np.nan
    ss_res: float = float(np.square(values - fitted).sum())
    return float(max(0.0, 1.0 - (ss_res / ss_tot)))


def rolling_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    directional_move = cast(pd.Series, close.diff(window).abs())
    path_length = cast(pd.Series, close.diff().abs().rolling(window).sum())
    return cast(pd.Series, safe_divide(directional_move, path_length))
