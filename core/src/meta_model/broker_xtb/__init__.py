from __future__ import annotations

"""XTB broker integration layer.

Provides instrument specification management, cost estimation, tradable universe
construction, and manual execution bundle generation for XTB cash-equity CFDs.
"""

from core.src.meta_model.broker_xtb.bridge import ManualOrderTicket, build_manual_execution_bundle
from core.src.meta_model.broker_xtb.costs import BrokerCostEstimate, estimate_trade_cost
from core.src.meta_model.broker_xtb.reference_snapshot import (
    build_xtb_reference_snapshot_from_pdf,
    build_xtb_reference_snapshot_payload,
    download_xtb_equity_pdf,
    extract_us_stock_symbols_from_pdf_text,
    save_xtb_reference_snapshot,
)
from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    XtbInstrumentSpec,
    build_default_spec_provider,
)
from core.src.meta_model.broker_xtb.universe import TradableInstrument, build_tradable_universe

__all__ = [
    "BrokerCostEstimate",
    "BrokerSpecProvider",
    "ManualOrderTicket",
    "TradableInstrument",
    "XtbInstrumentSpec",
    "build_xtb_reference_snapshot_from_pdf",
    "build_xtb_reference_snapshot_payload",
    "build_default_spec_provider",
    "build_manual_execution_bundle",
    "build_tradable_universe",
    "download_xtb_equity_pdf",
    "estimate_trade_cost",
    "extract_us_stock_symbols_from_pdf_text",
    "save_xtb_reference_snapshot",
]
