from __future__ import annotations

"""Tradable universe construction from XTB instrument specs.

Resolves which tickers in the price dataset are actually tradable through XTB
and persists the universe to Parquet and CSV.
"""

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider
from core.src.meta_model.data.paths import XTB_TRADABLE_UNIVERSE_CSV, XTB_TRADABLE_UNIVERSE_PARQUET

LOGGER: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradableInstrument:
    """Immutable record for one tradable instrument in the XTB universe."""

    ticker: str
    xtb_symbol: str
    instrument_group: str
    currency: str
    minimum_order_value_eur: float
    max_adv_participation: float


def build_tradable_universe(
    frame: pd.DataFrame,
    spec_provider: BrokerSpecProvider,
    *,
    trade_date: pd.Timestamp,
    max_spread_bps: float,
) -> pd.DataFrame:
    """Build a DataFrame of instruments tradable at *trade_date* via the spec provider."""
    del max_spread_bps
    rows: list[dict[str, object]] = []
    for ticker in sorted({str(value) for value in frame["ticker"].dropna().tolist()}):
        try:
            spec = spec_provider.resolve(ticker, trade_date)
        except KeyError:
            continue
        rows.append(asdict(TradableInstrument(
            ticker=ticker,
            xtb_symbol=spec.symbol if not spec.symbol.startswith("__default") else ticker,
            instrument_group=spec.instrument_group,
            currency=spec.currency,
            minimum_order_value_eur=spec.minimum_order_value_eur,
            max_adv_participation=spec.max_adv_participation,
        )))
    return pd.DataFrame(rows)


def save_tradable_universe(universe: pd.DataFrame) -> None:
    """Persist the tradable universe to Parquet and CSV."""
    XTB_TRADABLE_UNIVERSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(XTB_TRADABLE_UNIVERSE_PARQUET, index=False)
    universe.to_csv(XTB_TRADABLE_UNIVERSE_CSV, index=False)
