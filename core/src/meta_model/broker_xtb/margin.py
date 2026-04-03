from __future__ import annotations

from dataclasses import dataclass

from core.src.meta_model.broker_xtb.specs import XtbInstrumentSpec


@dataclass(frozen=True)
class BrokerMarginEstimate:
    required_margin: float
    leverage: float
    headroom_after_trade: float


def estimate_margin(
    spec: XtbInstrumentSpec,
    *,
    notional: float,
    available_equity: float,
) -> BrokerMarginEstimate:
    required_margin = max(0.0, notional * spec.margin_requirement)
    leverage = 0.0 if required_margin <= 0.0 else notional / required_margin
    headroom_after_trade = available_equity - required_margin
    return BrokerMarginEstimate(
        required_margin=required_margin,
        leverage=leverage,
        headroom_after_trade=headroom_after_trade,
    )
