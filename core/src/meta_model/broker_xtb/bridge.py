from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from core.src.meta_model.data.paths import (
    EVALUATE_EXECUTION_CHECKLIST_JSON,
    EVALUATE_MANUAL_ORDERS_CSV,
    EVALUATE_MANUAL_WATCHLIST_CSV,
    EVALUATE_POST_TRADE_RECONCILIATION_PARQUET,
)
if TYPE_CHECKING:
    from core.src.meta_model.evaluate.backtest import ActiveTrade


@dataclass(frozen=True)
class ManualOrderTicket:
    ticker: str
    side: str
    notional: float
    signal_rank: int
    predicted_return: float
    expected_cost_rate: float
    margin_requirement: float
    required_margin: float


def build_manual_execution_bundle(
    trades: list["ActiveTrade"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    tickets = [
        asdict(ManualOrderTicket(
            ticker=trade.ticker,
            side=trade.side,
            notional=trade.notional,
            signal_rank=trade.signal_rank,
            predicted_return=trade.predicted_return,
            expected_cost_rate=trade.expected_total_cost_rate,
            margin_requirement=trade.margin_requirement,
            required_margin=trade.required_margin,
        ))
        for trade in trades
    ]
    orders: pd.DataFrame = pd.DataFrame(tickets)
    watchlist = cast(
        pd.DataFrame,
        orders.loc[:, ["ticker", "side", "signal_rank", "predicted_return"]].copy(),
    )
    checklist: dict[str, list[str]] = {
        "steps": [
            "Verify instrument availability in xStation.",
            "Check spread and swap against latest XTB tables before execution.",
            "Validate required margin and account headroom.",
            "Confirm no corporate action or market-halt event invalidates the signal.",
            "Execute manually and reconcile fills after market close.",
        ],
    }
    return orders, watchlist, checklist


def save_manual_execution_bundle(
    orders: pd.DataFrame,
    watchlist: pd.DataFrame,
    checklist: dict[str, list[str]],
    *,
    orders_path: Path = EVALUATE_MANUAL_ORDERS_CSV,
    watchlist_path: Path = EVALUATE_MANUAL_WATCHLIST_CSV,
    checklist_path: Path = EVALUATE_EXECUTION_CHECKLIST_JSON,
    reconciliation_path: Path = EVALUATE_POST_TRADE_RECONCILIATION_PARQUET,
) -> None:
    for output_path in (orders_path, watchlist_path, checklist_path, reconciliation_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(orders_path, index=False)
    watchlist.to_csv(watchlist_path, index=False)
    checklist_path.write_text(json.dumps(checklist, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(columns=["ticker", "side", "filled_notional", "fill_price", "notes"]).to_parquet(
        reconciliation_path,
        index=False,
    )
