from __future__ import annotations

from dataclasses import dataclass

from core.src.meta_model.broker_xtb.specs import XtbInstrumentSpec


@dataclass(frozen=True)
class BrokerCostEstimate:
    entry_cost_rate: float
    exit_cost_rate: float
    financing_cost_rate: float
    total_cost_rate: float


def estimate_trade_cost(
    spec: XtbInstrumentSpec,
    *,
    side: str,
    expected_holding_days: int,
    fx_conversion_bps: float = 0.0,
) -> BrokerCostEstimate:
    spread_component = spec.spread_bps / 20_000.0
    slippage_component = spec.slippage_bps / 10_000.0
    fx_component = fx_conversion_bps / 10_000.0
    entry_cost_rate = spread_component + slippage_component + fx_component
    exit_cost_rate = spread_component + slippage_component + fx_component
    swap_bps = spec.long_swap_bps_daily if side == "long" else spec.short_swap_bps_daily
    financing_cost_rate = max(expected_holding_days, 0) * (swap_bps / 10_000.0)
    total_cost_rate = entry_cost_rate + exit_cost_rate + financing_cost_rate
    return BrokerCostEstimate(
        entry_cost_rate=entry_cost_rate,
        exit_cost_rate=exit_cost_rate,
        financing_cost_rate=financing_cost_rate,
        total_cost_rate=total_cost_rate,
    )
