from __future__ import annotations

import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import pandas as pd

from core.src.meta_model.data.constants import (
    CHUNK_SIZE,
    CROSS_ASSET_INDICES,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    MAX_RETRIES,
    RANDOM_SEED,
    RETRY_SLEEP,
    RISK_APPETITE_SYMBOLS,
    SAMPLE_FRAC,
    SECTOR_ETFS,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import get_nyse_sessions
from core.src.meta_model.runtime_parallelism import resolve_executor_worker_count

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

LOGGER: logging.Logger = logging.getLogger(__name__)

ALL_CROSS_ASSET_SYMBOLS: tuple[str, ...] = (
    *CROSS_ASSET_INDICES,
    *SECTOR_ETFS,
    *RISK_APPETITE_SYMBOLS,
)

CROSS_ASSET_COLUMNS: list[str] = ["adj_close", "volume"]


@dataclasses.dataclass(frozen=True)
class CrossAssetConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    output_dir: Path = DATA_FETCHING_DIR
    sample_frac: float = SAMPLE_FRAC
    random_seed: int = RANDOM_SEED
    chunk_size: int = CHUNK_SIZE
    max_retries: int = MAX_RETRIES
    retry_sleep: float = RETRY_SLEEP


def _extract_symbol_df(
    raw: pd.DataFrame, symbol: str, is_multi: bool,
) -> pd.DataFrame | None:
    if is_multi:
        try:
            ticker_df: pd.DataFrame = cast(
                pd.DataFrame, raw.xs(symbol, level=1, axis=1),
            )
        except KeyError:
            return None
    else:
        ticker_df = raw.copy()

    col_map: dict[str, str] = {"Adj Close": "adj_close", "Volume": "volume"}
    ticker_df = ticker_df.rename(columns=col_map)
    present: list[str] = [c for c in CROSS_ASSET_COLUMNS if c in ticker_df.columns]
    if not present:
        return None
    filtered_ticker_df = cast(pd.DataFrame, ticker_df[present].dropna(how="all"))
    return filtered_ticker_df if not filtered_ticker_df.empty else None


def _fetch_cross_asset_batch(
    symbols: list[str], start_date: str, end_date: str,
) -> dict[str, pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is not installed")

    raw: pd.DataFrame | None = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=False,
    )

    if raw is None or raw.empty:
        return {}

    is_multi: bool = isinstance(raw.columns, pd.MultiIndex)
    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        ticker_df: pd.DataFrame | None = _extract_symbol_df(
            raw, symbol, is_multi,
        )
        if ticker_df is not None:
            result[symbol] = ticker_df

    return result


def _fetch_cross_asset_chunk_with_retry(
    batch: list[str], config: CrossAssetConfig,
) -> dict[str, pd.DataFrame]:
    """Fetch a single cross-asset chunk with retry logic."""
    attempts: int = 0
    while attempts < config.max_retries:
        try:
            return _fetch_cross_asset_batch(
                batch, config.start_date, config.end_date,
            )
        except Exception as exc:  # noqa: BLE001 - external API
            attempts += 1
            LOGGER.warning(
                "yfinance cross-asset batch failed (%s/%s): %s",
                attempts, config.max_retries, exc,
            )
            time.sleep(config.retry_sleep * attempts)
    return {}


def _fetch_all_cross_assets(
    symbols: tuple[str, ...], config: CrossAssetConfig,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    symbol_list: list[str] = list(symbols)
    chunks: list[list[str]] = [
        symbol_list[idx : idx + config.chunk_size]
        for idx in range(0, len(symbol_list), config.chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=len(chunks))) as executor:
        futures = {
            executor.submit(
                _fetch_cross_asset_chunk_with_retry, batch, config,
            ): batch
            for batch in chunks
        }
        for future in as_completed(futures):
            results.update(future.result())

    LOGGER.info(
        "Fetched cross-asset data for %d/%d symbols", len(results), len(symbols),
    )
    return results


def build_cross_asset_dataset(config: CrossAssetConfig) -> pd.DataFrame:
    price_map: dict[str, pd.DataFrame] = _fetch_all_cross_assets(
        ALL_CROSS_ASSET_SYMBOLS, config,
    )

    if not price_map:
        raise RuntimeError("No cross-asset data retrieved from yfinance")

    full_index: pd.DatetimeIndex = get_nyse_sessions(
        config.start_date, config.end_date,
    )
    rows: list[pd.DataFrame] = []

    for symbol in ALL_CROSS_ASSET_SYMBOLS:
        ticker_df: pd.DataFrame | None = price_map.get(symbol)
        if ticker_df is None or ticker_df.empty:
            continue
        ticker_df = ticker_df.reindex(full_index).ffill()
        ticker_df = ticker_df.copy()
        ticker_df["date"] = ticker_df.index
        ticker_df["ticker"] = symbol
        rows.append(ticker_df)

    if not rows:
        raise RuntimeError("No cross-asset data after reindexing")

    data: pd.DataFrame = pd.concat(rows, ignore_index=True)
    col_order: list[str] = ["date", "ticker"] + [
        c for c in CROSS_ASSET_COLUMNS if c in data.columns
    ]
    data = cast(pd.DataFrame, data[col_order])
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    LOGGER.info(
        "Built cross-asset dataset: %d rows x %d columns",
        len(data), len(data.columns),
    )
    return data
