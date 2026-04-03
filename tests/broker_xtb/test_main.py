from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.src.meta_model.broker_xtb.bridge import build_manual_execution_bundle
from core.src.meta_model.broker_xtb.costs import estimate_trade_cost
from core.src.meta_model.broker_xtb.margin import estimate_margin
from core.src.meta_model.broker_xtb.specs import build_default_spec_provider
from core.src.meta_model.broker_xtb.universe import build_tradable_universe
from core.src.meta_model.evaluate.backtest import ActiveTrade


def test_default_spec_provider_resolves_stock_and_index_defaults() -> None:
    provider = build_default_spec_provider()

    stock_spec = provider.resolve("AAPL", pd.Timestamp("2024-01-02"))
    index_spec = provider.resolve("US500", pd.Timestamp("2024-01-02"))

    assert stock_spec.instrument_group == "stock_cfd"
    assert index_spec.instrument_group == "index_cfd"
    assert stock_spec.spread_bps > index_spec.spread_bps


def test_trade_cost_and_margin_are_broker_aware() -> None:
    provider = build_default_spec_provider()
    spec = provider.resolve("AAPL", pd.Timestamp("2024-01-02"))

    intraday_cost = estimate_trade_cost(spec, side="long", expected_holding_days=0)
    swing_cost = estimate_trade_cost(spec, side="long", expected_holding_days=2)
    margin = estimate_margin(spec, notional=50_000.0, available_equity=20_000.0)

    assert swing_cost.total_cost_rate > intraday_cost.total_cost_rate
    assert margin.required_margin > 0.0
    assert margin.leverage > 0.0


def test_tradable_universe_filters_wide_spreads() -> None:
    provider = build_default_spec_provider()
    frame = pd.DataFrame({"ticker": ["AAPL", "US500"]})

    universe = build_tradable_universe(
        frame,
        provider,
        trade_date=pd.Timestamp("2024-01-02"),
        max_spread_bps=10.0,
    )

    assert universe["ticker"].tolist() == ["US500"]


def test_tradable_universe_skips_symbols_missing_from_explicit_snapshot(
    tmp_path: Path,
) -> None:
    specs_path = tmp_path / "xtb_instrument_specs.json"
    specs_path.write_text(
        json.dumps([
            {
                "symbol": "AAPL",
                "instrument_group": "stock_cfd",
                "currency": "USD",
                "spread_bps": 25.0,
                "slippage_bps": 5.0,
                "long_swap_bps_daily": 2.0,
                "short_swap_bps_daily": 1.0,
                "margin_requirement": 0.20,
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
    frame = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})

    universe = build_tradable_universe(
        frame,
        provider,
        trade_date=pd.Timestamp("2024-01-02"),
        max_spread_bps=40.0,
    )

    assert universe["ticker"].tolist() == ["AAPL"]


def test_manual_execution_bundle_contains_operational_outputs() -> None:
    trade = ActiveTrade(
        ticker="AAPL",
        side="long",
        entry_date=pd.Timestamp("2024-01-02"),
        exit_date=pd.Timestamp("2024-01-02"),
        notional=25_000.0,
        predicted_return=0.02,
        realized_log_return=0.01,
        signal_rank=1,
        expected_total_cost_rate=0.004,
        margin_requirement=0.20,
        required_margin=5_000.0,
    )

    orders, watchlist, checklist = build_manual_execution_bundle([trade])

    assert list(orders["ticker"]) == ["AAPL"]
    assert list(watchlist["ticker"]) == ["AAPL"]
    assert checklist["steps"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
