from __future__ import annotations

"""XTB trade cost estimation.

Computes commission, FX conversion, and total cost for a single order based on
the instrument spec and the month-to-date turnover allowance.
"""

from dataclasses import dataclass

from core.src.meta_model.broker_xtb.specs import XtbInstrumentSpec


@dataclass(frozen=True)
class BrokerCostEstimate:
    """Immutable breakdown of the estimated cost for a single trade."""

    order_value_eur: float
    billable_turnover_eur: float
    commission_amount_eur: float
    fx_conversion_amount_eur: float
    total_cost_amount_eur: float
    total_cost_rate: float
    month_to_date_turnover_eur_after_order: float


def estimate_trade_cost(
    spec: XtbInstrumentSpec,
    *,
    order_value_eur: float,
    month_to_date_turnover_eur: float,
    account_currency: str = "EUR",
) -> BrokerCostEstimate:
    """Estimate commission + FX cost for a single order.

    Args:
        spec: Instrument trading specification.
        order_value_eur: Notional order value in EUR.
        month_to_date_turnover_eur: Cumulative turnover this month.
        account_currency: Account base currency for FX check.

    Returns:
        Detailed cost breakdown.
    """
    normalized_order_value = max(float(order_value_eur), 0.0)
    free_turnover_remaining = max(
        spec.monthly_commission_free_turnover_eur - max(month_to_date_turnover_eur, 0.0),
        0.0,
    )
    billable_turnover_eur = max(normalized_order_value - free_turnover_remaining, 0.0)
    free_tier_applies = (
        spec.instrument_group == "stock_cash"
        and billable_turnover_eur <= 0.0
    )
    commission_amount_eur = 0.0
    if billable_turnover_eur > 0.0:
        commission_amount_eur = max(
            spec.minimum_commission_eur,
            billable_turnover_eur * spec.commission_rate,
        )
    fx_conversion_amount_eur = 0.0
    if not free_tier_applies and spec.currency.upper() != account_currency.upper():
        fx_conversion_amount_eur = billable_turnover_eur * (spec.fx_conversion_bps / 10_000.0)
    total_cost_amount_eur = commission_amount_eur + fx_conversion_amount_eur
    total_cost_rate = 0.0
    if normalized_order_value > 0.0:
        total_cost_rate = total_cost_amount_eur / normalized_order_value
    return BrokerCostEstimate(
        order_value_eur=normalized_order_value,
        billable_turnover_eur=billable_turnover_eur,
        commission_amount_eur=commission_amount_eur,
        fx_conversion_amount_eur=fx_conversion_amount_eur,
        total_cost_amount_eur=total_cost_amount_eur,
        total_cost_rate=total_cost_rate,
        month_to_date_turnover_eur_after_order=(
            max(month_to_date_turnover_eur, 0.0) + normalized_order_value
        ),
    )
