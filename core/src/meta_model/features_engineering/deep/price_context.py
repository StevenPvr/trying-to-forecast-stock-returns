from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.src.meta_model.features_engineering.deep.helpers import as_series
from core.src.meta_model.features_engineering.utils import safe_divide


@dataclass(frozen=True)
class PriceContext:
    index: pd.Index
    open_price: pd.Series
    high_price: pd.Series
    low_price: pd.Series
    close_price: pd.Series
    volume: pd.Series
    prev_close: pd.Series
    prev_high: pd.Series
    prev_low: pd.Series
    returns_1d: pd.Series
    intraday_return: pd.Series
    gap_return: pd.Series
    true_range_pct: pd.Series
    close_location: pd.Series
    avg_volume_21d: pd.Series
    return_std_21d: pd.Series
    gap_std_63d: pd.Series
    prior_high_63d: pd.Series
    prior_low_63d: pd.Series
    true_range_median_63d: pd.Series


def build_price_context(group: pd.DataFrame) -> PriceContext:
    index = group.index
    open_price = as_series(pd.to_numeric(group["stock_open_price"], errors="coerce"), index)
    high_price = as_series(pd.to_numeric(group["stock_high_price"], errors="coerce"), index)
    low_price = as_series(pd.to_numeric(group["stock_low_price"], errors="coerce"), index)
    close_price = as_series(pd.to_numeric(group["stock_close_price"], errors="coerce"), index)
    volume = as_series(pd.to_numeric(group["stock_trading_volume"], errors="coerce"), index)
    prev_close = as_series(close_price.shift(1), index)
    prev_high = as_series(high_price.shift(1), index)
    prev_low = as_series(low_price.shift(1), index)
    returns_1d = as_series(close_price.pct_change(), index)
    intraday_return = as_series(safe_divide(close_price, open_price) - 1.0, index)
    gap_return = as_series(safe_divide(open_price, prev_close) - 1.0, index)
    true_range_pct = _build_true_range_pct(high_price, low_price, prev_close, index)
    close_location = as_series(
        safe_divide((2.0 * close_price) - high_price - low_price, high_price - low_price),
        index,
    )
    return PriceContext(
        index=index,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        prev_close=prev_close,
        prev_high=prev_high,
        prev_low=prev_low,
        returns_1d=returns_1d,
        intraday_return=intraday_return,
        gap_return=gap_return,
        true_range_pct=true_range_pct,
        close_location=close_location,
        avg_volume_21d=as_series(volume.rolling(21).mean(), index),
        return_std_21d=as_series(returns_1d.rolling(21).std(), index),
        gap_std_63d=as_series(gap_return.rolling(63).std(), index),
        prior_high_63d=as_series(high_price.shift(1).rolling(63).max(), index),
        prior_low_63d=as_series(low_price.shift(1).rolling(63).min(), index),
        true_range_median_63d=as_series(true_range_pct.shift(1).rolling(63).median(), index),
    )


def _build_true_range_pct(
    high_price: pd.Series,
    low_price: pd.Series,
    prev_close: pd.Series,
    index: pd.Index,
) -> pd.Series:
    true_range = as_series(
        pd.concat(
            [
                (high_price - low_price).abs(),
                (high_price - prev_close).abs(),
                (low_price - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1),
        index,
    )
    return as_series(safe_divide(true_range, prev_close), index)


__all__ = ["PriceContext", "build_price_context"]
