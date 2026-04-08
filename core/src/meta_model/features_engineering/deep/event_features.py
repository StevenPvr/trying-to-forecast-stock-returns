from __future__ import annotations

"""Event-based features: gap events, range expansions, volume spikes, breakout recency."""

import pandas as pd

from core.src.meta_model.features_engineering.deep.helpers import days_since_last_true
from core.src.meta_model.features_engineering.deep.price_context import PriceContext


def add_event_features(data: pd.DataFrame, context: PriceContext) -> None:
    inside_day = (context.high_price <= context.prev_high) & (context.low_price >= context.prev_low)
    outside_day = (context.high_price >= context.prev_high) & (context.low_price <= context.prev_low)
    gap_fill = _build_gap_fill_flag(context)
    breakout_up = context.high_price > context.prior_high_63d
    breakout_down = context.low_price < context.prior_low_63d
    breakout_failure_up = breakout_up & (context.close_price <= context.prev_high)
    breakout_failure_down = breakout_down & (context.close_price >= context.prev_low)
    large_gap_up = context.gap_return > context.gap_std_63d
    large_gap_down = context.gap_return < -context.gap_std_63d

    data["deep_event_inside_day_flag"] = inside_day.astype(float)
    data["deep_event_outside_day_flag"] = outside_day.astype(float)
    data["deep_event_gap_fill_flag"] = gap_fill.astype(float)
    data["deep_event_breakout_up_63d_flag"] = breakout_up.astype(float)
    data["deep_event_breakout_down_63d_flag"] = breakout_down.astype(float)
    data["deep_event_breakout_failure_up_63d_flag"] = breakout_failure_up.astype(float)
    data["deep_event_breakout_failure_down_63d_flag"] = breakout_failure_down.astype(float)
    data["deep_event_days_since_breakout_up_63d"] = days_since_last_true(breakout_up)
    data["deep_event_days_since_breakout_down_63d"] = days_since_last_true(breakout_down)
    data["deep_event_days_since_large_gap_up_63d"] = days_since_last_true(large_gap_up)
    data["deep_event_days_since_large_gap_down_63d"] = days_since_last_true(large_gap_down)


def _build_gap_fill_flag(context: PriceContext) -> pd.Series:
    gap_up = context.gap_return > 0.0
    gap_down = context.gap_return < 0.0
    gap_up_fill = gap_up & (context.low_price <= context.prev_close)
    gap_down_fill = gap_down & (context.high_price >= context.prev_close)
    return gap_up_fill | gap_down_fill


__all__ = ["add_event_features"]
