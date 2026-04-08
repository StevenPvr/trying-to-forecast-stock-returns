from __future__ import annotations

"""Manual execution bridge for XTB cash-equity orders.

Generates order tickets, watchlists, and pre-trade checklists that can be
executed manually through the xStation platform.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

LOGGER: logging.Logger = logging.getLogger(__name__)

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
    """Immutable representation of a single manual order to be placed in xStation."""

    ticker: str
    side: str
    order_value_eur: float
    signal_rank: int
    predicted_return: float
    expected_entry_cost_eur: float
    expected_entry_cost_rate: float
    instrument_currency: str
    minimum_order_value_eur: float


def build_manual_execution_bundle(
    trades: list["ActiveTrade"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Build orders DataFrame, watchlist DataFrame, and pre-trade checklist.

    Args:
        trades: Active trades from the backtest engine.

    Returns:
        Tuple of (orders, watchlist, checklist).
    """
    order_columns = [
        "ticker",
        "side",
        "order_value_eur",
        "signal_rank",
        "predicted_return",
        "expected_entry_cost_eur",
        "expected_entry_cost_rate",
        "instrument_currency",
        "minimum_order_value_eur",
    ]
    watchlist_columns = ["ticker", "side", "signal_rank", "predicted_return"]
    tickets = [
        asdict(ManualOrderTicket(
            ticker=trade.ticker,
            side=trade.side,
            order_value_eur=trade.notional,
            signal_rank=trade.signal_rank,
            predicted_return=trade.predicted_return,
            expected_entry_cost_eur=trade.entry_transaction_cost_amount,
            expected_entry_cost_rate=trade.expected_entry_cost_rate,
            instrument_currency=trade.spec.currency,
            minimum_order_value_eur=trade.spec.minimum_order_value_eur,
        ))
        for trade in trades
    ]
    orders = pd.DataFrame(tickets, columns=order_columns)
    watchlist = cast(pd.DataFrame, orders.loc[:, watchlist_columns].copy())
    checklist: dict[str, list[str]] = {
        "steps": [
            "Verify the share is available in xStation cash equities.",
            "Confirm the order value is at least 10 EUR.",
            "Check the remaining monthly commission-free turnover before sending the order.",
            "Check whether 0.5% FX conversion applies for the instrument currency.",
            "Remember that XTB cash equities follow FIFO when reducing positions.",
            "Reconcile executed order values and fees after market close.",
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
    """Persist execution bundle artifacts to disk."""
    for output_path in (orders_path, watchlist_path, checklist_path, reconciliation_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(orders_path, index=False)
    watchlist.to_csv(watchlist_path, index=False)
    checklist_path.write_text(
        json.dumps(checklist, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        columns=["ticker", "filled_order_value_eur", "fill_price", "fees_eur", "notes"],
    ).to_parquet(
        reconciliation_path,
        index=False,
    )
