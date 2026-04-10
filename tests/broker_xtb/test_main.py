from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.bridge import build_manual_execution_bundle
from core.src.meta_model.broker_xtb.specs import (
    BrokerSpecProvider,
    XtbInstrumentSpec,
    build_default_spec_provider,
)
from core.src.meta_model.broker_xtb.universe import build_tradable_universe
from core.src.meta_model.evaluate.backtest import ActiveTrade


def test_default_spec_provider_requires_explicit_snapshot(tmp_path: Path) -> None:
    missing_specs_path = tmp_path / "missing_specs.json"

    with pytest.raises(FileNotFoundError, match="Missing XTB instrument specification snapshot"):
        build_default_spec_provider(path=missing_specs_path)


def test_tradable_universe_keeps_cash_equity_snapshot_symbols(tmp_path: Path) -> None:
    specs_path = tmp_path / "xtb_specs.json"
    specs_path.write_text(
        json.dumps(
            [
                {
                    "symbol": "AAPL",
                    "instrument_group": "stock_cash",
                    "currency": "USD",
                    "spread_bps": 0.0,
                    "slippage_bps": 0.0,
                    "long_swap_bps_daily": 0.0,
                    "short_swap_bps_daily": 0.0,
                    "margin_requirement": 1.0,
                    "max_adv_participation": 0.05,
                    "effective_from": "2000-01-01",
                    "effective_to": None,
                },
            ],
        ),
        encoding="utf-8",
    )
    provider = build_default_spec_provider(
        path=specs_path,
        allow_defaults_if_missing=False,
        require_explicit_symbols=True,
    )
    frame = pd.DataFrame({"ticker": ["AAPL"]})

    universe = build_tradable_universe(
        frame,
        provider,
        trade_date=pd.Timestamp("2024-01-02"),
        max_spread_bps=5.0,
    )

    assert universe["ticker"].tolist() == ["AAPL"]
    assert set(universe["instrument_group"]) == {"stock_cash"}


def test_tradable_universe_skips_symbols_missing_from_explicit_snapshot(
    tmp_path: Path,
) -> None:
    specs_path = tmp_path / "xtb_instrument_specs.json"
    specs_path.write_text(
        json.dumps([
            {
                "symbol": "AAPL",
                "instrument_group": "stock_cash",
                "currency": "USD",
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
                "long_swap_bps_daily": 0.0,
                "short_swap_bps_daily": 0.0,
                "margin_requirement": 1.0,
                "max_adv_participation": 0.05,
                "effective_from": "2000-01-01",
                "effective_to": None,
            },
        ]),
        encoding="utf-8",
    )
    provider = build_default_spec_provider(
        path=specs_path,
        allow_defaults_if_missing=False,
        require_explicit_symbols=True,
    )
    frame = pd.DataFrame({"ticker": ["AAPL"]})

    universe = build_tradable_universe(
        frame,
        provider,
        trade_date=pd.Timestamp("2024-01-02"),
        max_spread_bps=40.0,
    )

    assert universe["ticker"].tolist() == ["AAPL"]


def test_manual_execution_bundle_contains_cash_equity_outputs() -> None:
    provider = BrokerSpecProvider(
        specs=(
            XtbInstrumentSpec(
                symbol="AAPL",
                instrument_group="stock_cash",
                currency="USD",
                spread_bps=0.0,
                slippage_bps=0.0,
                long_swap_bps_daily=0.0,
                short_swap_bps_daily=0.0,
                margin_requirement=1.0,
                max_adv_participation=0.05,
                effective_from="2000-01-01",
                fx_conversion_bps=0.0,
            ),
        ),
        fallback_to_defaults=False,
    )
    spec = provider.resolve("AAPL", pd.Timestamp("2024-01-02"))
    trade = ActiveTrade(
        ticker="AAPL",
        side="long",
        entry_date=pd.Timestamp("2024-01-02"),
        exit_date=pd.Timestamp("2024-01-03"),
        notional=25_000.0,
        predicted_return=0.02,
        realized_log_return=0.01,
        signal_rank=1,
        spec=spec,
        entry_transaction_cost_amount=125.0,
        entry_commission_amount=75.0,
        entry_fx_conversion_amount=50.0,
        expected_entry_cost_rate=0.005,
    )

    orders, watchlist, checklist = build_manual_execution_bundle([trade])

    assert list(orders["ticker"]) == ["AAPL"]
    assert list(watchlist["ticker"]) == ["AAPL"]
    assert "FIFO" in " ".join(checklist["steps"])


def test_manual_execution_bundle_handles_empty_trade_list() -> None:
    orders, watchlist, checklist = build_manual_execution_bundle([])

    assert list(orders.columns) == [
        "ticker",
        "side",
        "share_count",
        "reference_price_eur",
        "order_value_eur",
        "signal_rank",
        "predicted_return",
        "expected_entry_cost_eur",
        "expected_entry_cost_rate",
        "instrument_currency",
        "minimum_order_value_eur",
    ]
    assert list(watchlist.columns) == ["ticker", "side", "signal_rank", "predicted_return"]
    assert orders.empty
    assert watchlist.empty
    assert checklist["steps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
