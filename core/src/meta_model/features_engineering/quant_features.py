from __future__ import annotations

import numpy as np
import pandas as pd

from core.src.meta_model.features_engineering.config import (
    DRAWDOWN_WINDOWS,
    INTERNAL_FEATURE_PREFIX,
    LIQUIDITY_WINDOWS,
    MARKET_WINDOWS,
    QUANT_FEATURE_PREFIX,
    RETURN_WINDOWS,
    TREND_WINDOWS,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_WINDOWS,
)
from core.src.meta_model.features_engineering.utils import (
    annualized_trend_slope,
    rolling_efficiency_ratio,
    rolling_zscore,
    safe_divide,
    trend_r2,
)


def _as_series(value: object, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.Series(value.to_numpy(), index=value.index)
    return pd.Series(value, index=index)


def _log_series(series: pd.Series) -> pd.Series:
    return pd.Series(np.log(series.to_numpy()), index=series.index)


def _compute_range_based_volatility_features(
    group: pd.DataFrame,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
) -> pd.DataFrame:
    index = group.index
    open_series = _as_series(group[open_col], index)
    high_series = _as_series(group[high_col], index)
    low_series = _as_series(group[low_col], index)
    close_series = _as_series(group[close_col], index)

    log_hl = _log_series(_as_series(safe_divide(high_series, low_series), index))
    log_co = _log_series(_as_series(safe_divide(close_series, open_series), index))
    prev_close = _as_series(close_series.shift(1), index)
    log_oc_prev = _log_series(_as_series(safe_divide(open_series, prev_close), index))
    log_ho = _log_series(_as_series(safe_divide(high_series, open_series), index))
    log_lo = _log_series(_as_series(safe_divide(low_series, open_series), index))
    log_hc = _log_series(_as_series(safe_divide(high_series, close_series), index))
    log_lc = _log_series(_as_series(safe_divide(low_series, close_series), index))

    parkinson_daily_var: pd.Series = (log_hl.pow(2)) / (4.0 * np.log(2.0))
    garman_klass_daily_var: pd.Series = (
        0.5 * log_hl.pow(2) - ((2.0 * np.log(2.0)) - 1.0) * log_co.pow(2)
    )
    rogers_satchell_daily_var: pd.Series = (log_hc * log_ho) + (log_lc * log_lo)
    yz_open_var_proxy: pd.Series = log_oc_prev
    yz_close_var_proxy: pd.Series = log_co

    for window in MARKET_WINDOWS:
        k: float = 0.34 / (1.34 + ((window + 1.0) / max(window - 1.0, 1.0)))
        parkinson = _as_series(
            np.sqrt(
                parkinson_daily_var.rolling(window).mean().clip(0.0)
                * TRADING_DAYS_PER_YEAR,
            ),
            index,
        )
        garman_klass = _as_series(
            np.sqrt(
                garman_klass_daily_var.rolling(window).mean().clip(0.0)
                * TRADING_DAYS_PER_YEAR,
            ),
            index,
        )
        rogers_satchell = _as_series(
            np.sqrt(
                rogers_satchell_daily_var.rolling(window).mean().clip(0.0)
                * TRADING_DAYS_PER_YEAR,
            ),
            index,
        )
        yang_zhang_var = _as_series(
            yz_open_var_proxy.rolling(window).var()
            + (k * yz_close_var_proxy.rolling(window).var())
            + ((1.0 - k) * rogers_satchell_daily_var.rolling(window).mean()),
            index,
        )
        yang_zhang = _as_series(
            np.sqrt(yang_zhang_var.clip(0.0) * TRADING_DAYS_PER_YEAR),
            index,
        )

        group[f"{QUANT_FEATURE_PREFIX}parkinson_vol_{window}d"] = parkinson
        group[f"{QUANT_FEATURE_PREFIX}garman_klass_vol_{window}d"] = garman_klass
        group[f"{QUANT_FEATURE_PREFIX}rogers_satchell_vol_{window}d"] = rogers_satchell
        group[f"{QUANT_FEATURE_PREFIX}yang_zhang_vol_{window}d"] = yang_zhang

    return group


def add_quant_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True).copy()
    index = group.index

    open_price = _as_series(group["stock_open_price"], index)
    high_price = _as_series(group["stock_high_price"], index)
    low_price = _as_series(group["stock_low_price"], index)
    close_price = _as_series(group["stock_close_price"], index)
    volume = _as_series(group["stock_trading_volume"], index)

    prev_close = _as_series(close_price.shift(1), index)
    daily_range = _as_series(
        (high_price - low_price).where((high_price - low_price) != 0, np.nan),
        index,
    )
    close_diff = _as_series(close_price.diff(), index)
    returns_1d = _as_series(close_price.pct_change(), index)
    log_returns_1d = _as_series(_log_series(close_price).diff(), index)
    dollar_volume = _as_series(close_price * volume, index)
    signed_dollar_volume = _as_series(
        close_diff.apply(np.sign).fillna(0.0) * dollar_volume,
        index,
    )

    true_range_components: pd.DataFrame = pd.concat(
        [
            (high_price - low_price).abs(),
            (high_price - prev_close).abs(),
            (low_price - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = _as_series(true_range_components.max(axis=1), index)

    group[f"{INTERNAL_FEATURE_PREFIX}return_1d"] = returns_1d
    group[f"{INTERNAL_FEATURE_PREFIX}log_return_1d"] = log_returns_1d
    group[f"{INTERNAL_FEATURE_PREFIX}signed_dollar_volume"] = signed_dollar_volume

    date_series = _as_series(group["date"], index)
    group[f"{QUANT_FEATURE_PREFIX}calendar_gap_days"] = date_series.diff().dt.days
    group[f"{QUANT_FEATURE_PREFIX}days_in_sample"] = np.arange(1, len(group) + 1)

    group[f"{QUANT_FEATURE_PREFIX}gap_return"] = safe_divide(open_price, prev_close) - 1.0
    group[f"{QUANT_FEATURE_PREFIX}intraday_return"] = safe_divide(close_price, open_price) - 1.0
    group[f"{QUANT_FEATURE_PREFIX}high_low_range_pct"] = safe_divide(high_price - low_price, close_price)
    group[f"{QUANT_FEATURE_PREFIX}true_range_pct"] = safe_divide(true_range, prev_close)
    group[f"{QUANT_FEATURE_PREFIX}close_location_value"] = safe_divide(
        (2.0 * close_price) - high_price - low_price,
        daily_range,
    )
    group[f"{QUANT_FEATURE_PREFIX}body_to_range"] = safe_divide(
        (close_price - open_price).abs(),
        daily_range,
    )
    group[f"{QUANT_FEATURE_PREFIX}upper_shadow_to_range"] = safe_divide(
        high_price - pd.concat([open_price, close_price], axis=1).max(axis=1),
        daily_range,
    )
    group[f"{QUANT_FEATURE_PREFIX}lower_shadow_to_range"] = safe_divide(
        pd.concat([open_price, close_price], axis=1).min(axis=1) - low_price,
        daily_range,
    )

    for window in RETURN_WINDOWS:
        group[f"{QUANT_FEATURE_PREFIX}momentum_{window}d"] = close_price.pct_change(window)

    for window in VOLATILITY_WINDOWS:
        realized_vol = _as_series(
            returns_1d.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            index,
        )
        group[f"{QUANT_FEATURE_PREFIX}realized_vol_{window}d"] = realized_vol
        group[f"{QUANT_FEATURE_PREFIX}return_skew_{window}d"] = returns_1d.rolling(window).skew()
        group[f"{QUANT_FEATURE_PREFIX}return_kurt_{window}d"] = returns_1d.rolling(window).kurt()
        group[f"{QUANT_FEATURE_PREFIX}return_autocorr_lag1_{window}d"] = returns_1d.rolling(window).corr(
            returns_1d.shift(1),
        )

    for window in TREND_WINDOWS:
        trend_slope = _log_series(close_price).rolling(window).apply(
            annualized_trend_slope,
            raw=True,
        )
        trend_r2_value = _log_series(close_price).rolling(window).apply(
            trend_r2,
            raw=True,
        )
        group[f"{QUANT_FEATURE_PREFIX}trend_slope_{window}d"] = trend_slope
        group[f"{QUANT_FEATURE_PREFIX}trend_r2_{window}d"] = trend_r2_value
        group[f"{QUANT_FEATURE_PREFIX}trend_strength_{window}d"] = trend_slope * trend_r2_value
        group[f"{QUANT_FEATURE_PREFIX}efficiency_ratio_{window}d"] = rolling_efficiency_ratio(
            close_price,
            window,
        )
        group[f"{QUANT_FEATURE_PREFIX}positive_day_ratio_{window}d"] = (
            returns_1d.gt(0).rolling(window).mean()
        )

    for window in (21, 63):
        negative_return_square_mean = returns_1d.clip(upper=0.0).pow(2).rolling(window).mean()
        positive_return_square_mean = returns_1d.clip(lower=0.0).pow(2).rolling(window).mean()
        group[f"{QUANT_FEATURE_PREFIX}downside_vol_{window}d"] = np.sqrt(
            negative_return_square_mean * TRADING_DAYS_PER_YEAR,
        )
        group[f"{QUANT_FEATURE_PREFIX}upside_vol_{window}d"] = np.sqrt(
            positive_return_square_mean * TRADING_DAYS_PER_YEAR,
        )

    for window in DRAWDOWN_WINDOWS:
        rolling_max = _as_series(close_price.rolling(window).max(), index)
        rolling_min = _as_series(close_price.rolling(window).min(), index)
        group[f"{QUANT_FEATURE_PREFIX}drawdown_{window}d"] = safe_divide(close_price, rolling_max) - 1.0
        group[f"{QUANT_FEATURE_PREFIX}rebound_from_low_{window}d"] = safe_divide(close_price, rolling_min) - 1.0

    group = _compute_range_based_volatility_features(
        group,
        open_col="stock_open_price",
        high_col="stock_high_price",
        low_col="stock_low_price",
        close_col="stock_close_price",
    )

    group[f"{QUANT_FEATURE_PREFIX}dollar_volume"] = dollar_volume
    group[f"{QUANT_FEATURE_PREFIX}log_dollar_volume"] = _as_series(
        np.log1p(dollar_volume.to_numpy()),
        index,
    )

    for window in LIQUIDITY_WINDOWS:
        avg_volume = _as_series(volume.rolling(window).mean(), index)
        avg_dollar_volume = _as_series(dollar_volume.rolling(window).mean(), index)
        volume_ratio = _as_series(safe_divide(volume, avg_volume) - 1.0, index)
        dollar_volume_ratio = _as_series(
            safe_divide(dollar_volume, avg_dollar_volume) - 1.0,
            index,
        )
        signed_dollar_volume_mean = _as_series(
            signed_dollar_volume.rolling(window).mean(),
            index,
        )
        signed_dollar_volume_ratio = _as_series(
            safe_divide(
                signed_dollar_volume_mean,
                avg_dollar_volume,
            ),
            index,
        )
        group[f"{QUANT_FEATURE_PREFIX}adv_{window}d"] = avg_dollar_volume
        group[f"{QUANT_FEATURE_PREFIX}volume_ratio_{window}d"] = volume_ratio
        group[f"{QUANT_FEATURE_PREFIX}dollar_volume_ratio_{window}d"] = dollar_volume_ratio
        group[f"{QUANT_FEATURE_PREFIX}volume_zscore_{window}d"] = rolling_zscore(volume, window)
        group[f"{QUANT_FEATURE_PREFIX}dollar_volume_zscore_{window}d"] = rolling_zscore(
            dollar_volume,
            window,
        )
        group[f"{QUANT_FEATURE_PREFIX}volume_cv_{window}d"] = safe_divide(
            _as_series(volume.rolling(window).std(), index),
            avg_volume,
        )
        amihud_daily = _as_series(
            safe_divide(returns_1d.abs(), dollar_volume),
            index,
        )
        group[f"{QUANT_FEATURE_PREFIX}amihud_illiquidity_{window}d"] = (
            amihud_daily.rolling(window).mean()
        )
        group[f"{QUANT_FEATURE_PREFIX}signed_dollar_volume_ratio_{window}d"] = (
            signed_dollar_volume_ratio
        )

    return group
