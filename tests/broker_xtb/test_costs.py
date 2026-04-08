from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.costs import estimate_trade_cost
from core.src.meta_model.broker_xtb.specs import XtbInstrumentSpec


def _build_spec(*, currency: str = "USD") -> XtbInstrumentSpec:
    return XtbInstrumentSpec(
        symbol="AAPL",
        instrument_group="stock_cash",
        currency=currency,
        spread_bps=0.0,
        slippage_bps=0.0,
        long_swap_bps_daily=0.0,
        short_swap_bps_daily=0.0,
        margin_requirement=1.0,
        max_adv_participation=0.05,
        effective_from="2000-01-01",
    )


def test_estimate_trade_cost_uses_remaining_free_monthly_quota() -> None:
    estimate = estimate_trade_cost(
        _build_spec(currency="EUR"),
        order_value_eur=50_000.0,
        month_to_date_turnover_eur=80_000.0,
        account_currency="EUR",
    )

    assert estimate.billable_turnover_eur == pytest.approx(30_000.0)
    assert estimate.commission_amount_eur == pytest.approx(60.0)
    assert estimate.fx_conversion_amount_eur == pytest.approx(0.0)
    assert estimate.month_to_date_turnover_eur_after_order == pytest.approx(130_000.0)


def test_estimate_trade_cost_applies_fx_only_when_currency_differs() -> None:
    foreign_estimate = estimate_trade_cost(
        _build_spec(currency="USD"),
        order_value_eur=10_000.0,
        month_to_date_turnover_eur=0.0,
        account_currency="EUR",
    )
    local_estimate = estimate_trade_cost(
        _build_spec(currency="EUR"),
        order_value_eur=10_000.0,
        month_to_date_turnover_eur=0.0,
        account_currency="EUR",
    )

    assert foreign_estimate.fx_conversion_amount_eur == pytest.approx(50.0)
    assert local_estimate.fx_conversion_amount_eur == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
