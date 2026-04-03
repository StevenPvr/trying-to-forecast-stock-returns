from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider
from core.src.meta_model.data.paths import XTB_TRADABLE_UNIVERSE_CSV, XTB_TRADABLE_UNIVERSE_PARQUET


@dataclass(frozen=True)
class TradableInstrument:
    ticker: str
    xtb_symbol: str
    instrument_group: str
    spread_bps: float
    margin_requirement: float
    max_adv_participation: float


def build_tradable_universe(
    frame: pd.DataFrame,
    spec_provider: BrokerSpecProvider,
    *,
    trade_date: pd.Timestamp,
    max_spread_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in sorted({str(value) for value in frame["ticker"].dropna().tolist()}):
        try:
            spec = spec_provider.resolve(ticker, trade_date)
        except KeyError:
            continue
        if spec.spread_bps > max_spread_bps:
            continue
        rows.append(asdict(TradableInstrument(
            ticker=ticker,
            xtb_symbol=spec.symbol if not spec.symbol.startswith("__default") else ticker,
            instrument_group=spec.instrument_group,
            spread_bps=spec.spread_bps,
            margin_requirement=spec.margin_requirement,
            max_adv_participation=spec.max_adv_participation,
        )))
    return pd.DataFrame(rows)


def save_tradable_universe(universe: pd.DataFrame) -> None:
    XTB_TRADABLE_UNIVERSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(XTB_TRADABLE_UNIVERSE_PARQUET, index=False)
    universe.to_csv(XTB_TRADABLE_UNIVERSE_CSV, index=False)
