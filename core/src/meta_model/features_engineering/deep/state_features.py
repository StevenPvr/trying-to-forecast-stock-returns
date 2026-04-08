from __future__ import annotations

import pandas as pd

from core.src.meta_model.features_engineering.deep.helpers import as_series
from core.src.meta_model.features_engineering.deep.price_context import PriceContext
from core.src.meta_model.features_engineering.utils import safe_divide


def add_state_features(data: pd.DataFrame, context: PriceContext) -> None:
    breakout_up = context.high_price > context.prior_high_63d
    breakout_failure_up = breakout_up & (context.close_price <= context.prev_high)
    gap_fill = _build_gap_fill_flag(context)
    up_day = context.returns_1d > 0.0
    down_day = context.returns_1d < 0.0

    data["deep_state_range_expansion_rate_21d"] = (
        context.true_range_pct.gt(context.true_range_median_63d).astype(float).rolling(21, min_periods=21).mean()
    )
    data["deep_state_gap_fill_rate_21d"] = gap_fill.astype(float).rolling(21, min_periods=21).mean()
    data["deep_state_breakout_up_rate_63d"] = breakout_up.astype(float).rolling(63, min_periods=63).mean()
    data["deep_state_breakout_failure_up_rate_63d"] = breakout_failure_up.astype(float).rolling(63, min_periods=63).mean()
    overnight_dominance = as_series(
        safe_divide(context.gap_return.abs(), context.intraday_return.abs()),
        context.index,
    )
    intraday_dominance = as_series(
        safe_divide(context.intraday_return.abs(), context.gap_return.abs()),
        context.index,
    )
    data["deep_state_overnight_dominance_21d"] = overnight_dominance.rolling(21, min_periods=21).mean()
    data["deep_state_intraday_dominance_21d"] = intraday_dominance.rolling(21, min_periods=21).mean()
    data["deep_state_volume_on_up_days_ratio_21d"] = _conditional_volume_ratio(context, up_day)
    data["deep_state_volume_on_down_days_ratio_21d"] = _conditional_volume_ratio(context, down_day)


def _build_gap_fill_flag(context: PriceContext) -> pd.Series:
    gap_up_fill = (context.gap_return > 0.0) & (context.low_price <= context.prev_close)
    gap_down_fill = (context.gap_return < 0.0) & (context.high_price >= context.prev_close)
    return gap_up_fill | gap_down_fill


def _conditional_volume_ratio(context: PriceContext, mask: pd.Series) -> pd.Series:
    conditional_volume = as_series(context.volume.where(mask).rolling(21, min_periods=21).mean(), context.index)
    return as_series(safe_divide(conditional_volume, context.avg_volume_21d), context.index)


__all__ = ["add_state_features"]
