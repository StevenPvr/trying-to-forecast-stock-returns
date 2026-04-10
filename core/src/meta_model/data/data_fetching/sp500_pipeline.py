from __future__ import annotations

"""S&P 500 price pipeline: constituents, PIT membership, multi-provider price download."""

import dataclasses
import datetime as dt
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pandas as pd
import requests

from core.src.meta_model.broker_xtb.specs import BrokerSpecProvider, build_default_spec_provider
from core.src.meta_model.data.constants import (
    CHUNK_SIZE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    FUNDAMENTAL_FIELDS,
    FUNDAMENTALS_SLEEP,
    MAX_RETRIES,
    RANDOM_SEED,
    RETRY_SLEEP,
    SAMPLE_FRAC,
    TIINGO_API_URL,
    WIKIPEDIA_SP500_URL,
    XTB_DEFAULT_MAX_SPREAD_BPS,
)
from core.src.meta_model.data.data_fetching.yfinance_download_lock import YFINANCE_DOWNLOAD_LOCK
from core.src.meta_model.data.paths import DATA_FETCHING_DIR, OUTPUT_PARQUET, OUTPUT_SAMPLE_CSV
from core.src.meta_model.data.trading_calendar import get_nyse_sessions
from core.src.meta_model.runtime_parallelism import resolve_executor_worker_count

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

STOOQ_URL: str = "https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"

LOGGER: logging.Logger = logging.getLogger(__name__)

OHLCV_COLUMNS: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

OHLCV_OUTPUT_COLS: list[str] = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    output_dir: Path = DATA_FETCHING_DIR
    sample_frac: float = SAMPLE_FRAC
    random_seed: int = RANDOM_SEED
    chunk_size: int = CHUNK_SIZE
    max_retries: int = MAX_RETRIES
    retry_sleep: float = RETRY_SLEEP
    use_stooq_fallback: bool = True
    use_tiingo_fallback: bool = True
    constituents_csv: Path | None = None
    ticker_aliases_csv: Path | None = None
    membership_history_csv: Path | None = None
    fundamentals_history_csv: Path | None = None
    xtb_only: bool = False
    xtb_instrument_specs_json: Path | None = None
    require_xtb_snapshot: bool = False
    xtb_max_spread_bps: float | None = None
    allow_current_constituents_snapshot: bool = False


def _parse_date(value: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _as_timestamp(value: str | dt.date | dt.datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return timestamp


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join(str(part) for part in col if part and part != "nan").strip()
            for col in df.columns
        ]
    return df



def _rename_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: pd.DataFrame = df.rename(columns=OHLCV_COLUMNS)
    present: list[str] = [col for col in OHLCV_OUTPUT_COLS if col in renamed.columns]
    renamed = cast(pd.DataFrame, renamed[present])
    renamed = cast(pd.DataFrame, renamed.dropna(how="all"))
    return renamed


def _camel_to_snake(name: str) -> str:
    result: str = re.sub(r"([A-Z])", r"_\1", name)
    return result.lower().lstrip("_")


def _fetch_ticker_info(
    symbol: str, max_retries: int, retry_sleep: float
) -> dict[str, object]:
    if yf is None:
        raise ImportError("yfinance is not installed")
    attempts: int = 0
    while attempts < max_retries:
        try:
            return yf.Ticker(symbol).info
        except Exception as exc:  # noqa: BLE001 - external API
            attempts += 1
            LOGGER.warning(
                "yf.Ticker(%s).info failed (%s/%s): %s",
                symbol, attempts, max_retries, exc,
            )
            time.sleep(retry_sleep * attempts)
    LOGGER.error("Failed to fetch info for %s after %s retries", symbol, max_retries)
    return {}


def _extract_fundamentals(
    info: dict[str, object], fields: tuple[str, ...]
) -> dict[str, object]:
    return {_camel_to_snake(field): info.get(field) for field in fields}


def _fetch_single_fundamental(
    symbol: str, aliases: dict[str, str], config: PipelineConfig,
) -> dict[str, object]:
    """Fetch fundamentals for one ticker, with rate-limit sleep."""
    info = _fetch_ticker_info(
        _yfinance_symbol(symbol, aliases), config.max_retries, config.retry_sleep,
    )
    row: dict[str, object] = {"ticker": symbol}
    row.update(_extract_fundamentals(info, FUNDAMENTAL_FIELDS))
    time.sleep(FUNDAMENTALS_SLEEP)
    return row


def _fetch_all_fundamentals(
    symbols: list[str], config: PipelineConfig,
) -> pd.DataFrame:
    aliases: dict[str, str] = _load_aliases(config.ticker_aliases_csv)
    with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=len(symbols))) as executor:
        futures = [
            executor.submit(_fetch_single_fundamental, s, aliases, config)
            for s in symbols
        ]
        rows = [f.result() for f in futures]
    LOGGER.info("Fetched fundamentals for %d tickers", len(rows))
    return pd.DataFrame(rows)


def _load_aliases(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        LOGGER.warning("Alias file not found: %s", path)
        return {}
    aliases: pd.DataFrame = pd.read_csv(path)
    if "symbol" not in aliases.columns or "provider_symbol" not in aliases.columns:
        raise ValueError("Alias file must contain 'symbol' and 'provider_symbol' columns")
    return dict(zip(aliases["symbol"].astype(str), aliases["provider_symbol"].astype(str)))


def _load_constituents_from_csv(path: Path) -> set[str]:
    df: pd.DataFrame = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("Constituents CSV must contain a 'symbol' column")
    return set(df["symbol"].astype(str).str.strip())


def _load_constituents_from_wikipedia() -> set[str]:
    table = load_constituents_table_from_wikipedia()
    LOGGER.info("Found %d current S&P 500 constituents", len(table))
    return set(table["ticker"].astype(str).str.strip())


def load_constituents_table_from_wikipedia() -> pd.DataFrame:
    response: requests.Response = requests.get(
        WIKIPEDIA_SP500_URL,
        headers={"User-Agent": "prevision-sp500/0.1"},
    )
    response.raise_for_status()
    tables: list[pd.DataFrame] = pd.read_html(io.StringIO(response.text))

    for table in tables:
        table = _flatten_columns(table)
        columns_lower: list[str] = [str(col).lower() for col in table.columns]
        if "symbol" in columns_lower:
            symbol_column = next(
                column for column in table.columns if str(column).lower() == "symbol"
            )
            security_column = next(
                (column for column in table.columns if str(column).lower() == "security"),
                None,
            )
            result = pd.DataFrame({
                "ticker": table[symbol_column].astype(str).str.strip(),
                "company_name": (
                    table[security_column].astype(str).str.strip()
                    if security_column is not None
                    else pd.Series("", index=table.index, dtype="object")
                ),
            })
            return result.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    raise RuntimeError("Could not locate S&P 500 constituent table on Wikipedia")


_load_constituents_table_from_wikipedia = load_constituents_table_from_wikipedia


def load_constituents(config: PipelineConfig) -> set[str]:
    if config.constituents_csv:
        return _load_constituents_from_csv(config.constituents_csv)

    table = load_constituents_table_from_wikipedia()
    LOGGER.info("Found %d current S&P 500 constituents", len(table))
    return set(table["ticker"].astype(str).str.strip())


def _load_membership_history_from_csv(
    path: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)
    rename_map: dict[str, str] = {"symbol": "ticker"}
    df = df.rename(columns=rename_map)
    if "ticker" not in df.columns:
        raise ValueError("Membership history CSV must contain 'ticker' or 'symbol'")
    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise ValueError(
            "Membership history CSV must contain 'start_date' and 'end_date' columns",
        )
    history: pd.DataFrame = cast(pd.DataFrame, df[["ticker", "start_date", "end_date"]].copy())
    history["ticker"] = history["ticker"].astype(str).str.strip()
    history["start_date"] = pd.to_datetime(history["start_date"])
    history["end_date"] = pd.to_datetime(history["end_date"])
    start_ts: pd.Timestamp = _as_timestamp(start_date)
    end_ts: pd.Timestamp = _as_timestamp(end_date)
    history = cast(
        pd.DataFrame,
        history.loc[
            (cast(pd.Series, history["end_date"]) >= start_ts)
            & (cast(pd.Series, history["start_date"]) <= end_ts),
        ].copy(),
    )
    history["start_date"] = history["start_date"].clip(lower=start_ts)
    history["end_date"] = history["end_date"].clip(upper=end_ts)
    return history.drop_duplicates().sort_values(
        ["ticker", "start_date", "end_date"],
    ).reset_index(drop=True)


def load_membership_history(config: PipelineConfig) -> pd.DataFrame:
    if config.membership_history_csv is not None:
        return _load_membership_history_from_csv(
            config.membership_history_csv,
            config.start_date,
            config.end_date,
        )

    if not config.allow_current_constituents_snapshot:
        raise ValueError(
            "build_dataset requires membership_history_csv for a point-in-time universe.",
        )

    start_ts: pd.Timestamp = _as_timestamp(config.start_date)
    end_ts: pd.Timestamp = _as_timestamp(config.end_date)
    snapshot_symbols: list[str] = sorted(load_constituents(config))
    LOGGER.warning(
        "No membership history CSV provided. Using the current S&P 500 "
        "constituents snapshot across %s to %s. This run is survivorship-biased.",
        config.start_date,
        config.end_date,
    )
    return pd.DataFrame({
        "ticker": snapshot_symbols,
        "start_date": [start_ts] * len(snapshot_symbols),
        "end_date": [end_ts] * len(snapshot_symbols),
    })


def _yfinance_symbol(symbol: str, aliases: dict[str, str]) -> str:
    return aliases.get(symbol, symbol.replace(".", "-"))


def _stooq_symbol(symbol: str, aliases: dict[str, str]) -> str:
    base: str = aliases.get(symbol, symbol)
    return f"{base.lower()}.us"


def _fetch_yfinance_batch(
    symbols: list[str], start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is not installed")

    with YFINANCE_DOWNLOAD_LOCK:
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

    data: pd.DataFrame = raw
    result: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level: int = 1
        for symbol in symbols:
            try:
                ticker_df: pd.DataFrame = cast(pd.DataFrame, data.xs(symbol, level=ticker_level, axis=1))
            except KeyError:
                continue
            ticker_df = _rename_ohlcv_columns(ticker_df)
            if not ticker_df.empty:
                result[symbol] = ticker_df
    else:
        ticker_df = _rename_ohlcv_columns(data)
        if not ticker_df.empty and symbols:
            result[symbols[0]] = ticker_df

    return result


def _fetch_stooq_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    url = STOOQ_URL.format(symbol=symbol, d1=start_date.replace("-", ""), d2=end_date.replace("-", ""))
    response = requests.get(url, headers={"User-Agent": "prevision-sp500/0.1"}, timeout=30)
    response.raise_for_status()
    payload: str = response.text.strip()
    lines: list[str] = payload.splitlines()
    if "No data" in payload or len(lines) < 2:
        return None
    header_columns: list[str] = [column.strip() for column in lines[0].split(",")]
    if "Date" not in header_columns:
        LOGGER.debug("Stooq returned a non-CSV payload for %s", symbol)
        return None
    data: pd.DataFrame = pd.read_csv(io.StringIO(payload), parse_dates=["Date"])
    if data.empty or "Close" not in data.columns:
        return None
    data = data.sort_values("Date").set_index("Date")
    data = _rename_ohlcv_columns(data)
    if "adj_close" not in data.columns and "close" in data.columns:
        data["adj_close"] = data["close"]
    data = data.dropna(how="all")
    return data if not data.empty else None


def _resolve_tiingo_api_key() -> str:
    key: str | None = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise ValueError("TIINGO_API_KEY environment variable is not set")
    return key


def _fetch_tiingo_symbol(
    symbol: str, start_date: str, end_date: str, api_key: str,
) -> pd.DataFrame | None:
    url: str = TIINGO_API_URL.format(symbol=symbol)
    headers: dict[str, str] = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    params: dict[str, str] = {
        "startDate": start_date,
        "endDate": end_date,
        "format": "csv",
    }
    response: requests.Response = requests.get(
        url, headers=headers, params=params, timeout=30,
    )
    response.raise_for_status()
    if not response.text.strip() or len(response.text.strip().splitlines()) < 2:
        return None
    data: pd.DataFrame = pd.read_csv(io.StringIO(response.text), parse_dates=["date"])
    if data.empty:
        return None
    raw_price_cols: set[str] = {"close", "high", "low", "open", "volume"}
    data = data.drop(columns=[c for c in raw_price_cols if c in data.columns])
    rename_map: dict[str, str] = {
        "adjClose": "adj_close",
        "adjOpen": "open",
        "adjHigh": "high",
        "adjLow": "low",
        "adjVolume": "volume",
    }
    data = data.rename(columns=rename_map)
    data = data.sort_values("date").set_index("date")
    present: list[str] = [c for c in OHLCV_OUTPUT_COLS if c in data.columns]
    data = cast(pd.DataFrame, data[present].dropna(how="all"))
    return data if not data.empty else None


def _fetch_chunk_with_retry(
    batch: list[str], start_date: str, end_date: str,
    symbol_lookup: dict[str, str], config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Fetch a single yfinance chunk with retry logic."""
    attempts: int = 0
    while attempts < config.max_retries:
        try:
            batch_results = _fetch_yfinance_batch(batch, start_date, end_date)
            resolved_pairs: list[tuple[str, pd.DataFrame]] = [
                (symbol_lookup.get(yf_sym, yf_sym), df) for yf_sym, df in list(batch_results.items())
            ]
            return dict(resolved_pairs)
        except Exception as exc:  # noqa: BLE001 - external API
            attempts += 1
            LOGGER.warning(
                "yfinance batch failed (%s/%s): %s",
                attempts, config.max_retries, exc,
            )
            time.sleep(config.retry_sleep * attempts)
    return {}


def _fetch_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    aliases: dict[str, str] = _load_aliases(config.ticker_aliases_csv)
    yfinance_symbols: list[str] = [_yfinance_symbol(symbol, aliases) for symbol in symbols]
    symbol_lookup: dict[str, str] = dict(zip(yfinance_symbols, symbols))

    results: dict[str, pd.DataFrame] = {}
    chunks: list[list[str]] = [
        yfinance_symbols[idx : idx + config.chunk_size]
        for idx in range(0, len(yfinance_symbols), config.chunk_size)
    ]

    if len(chunks) <= 1:
        for batch in chunks:
            results.update(
                _fetch_chunk_with_retry(batch, start_date, end_date, symbol_lookup, config),
            )
    else:
        with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=len(chunks))) as executor:
            futures = [
                executor.submit(
                    _fetch_chunk_with_retry,
                    batch,
                    start_date,
                    end_date,
                    symbol_lookup,
                    config,
                )
                for batch in chunks
            ]
            for future in futures:
                results.update(future.result())

    if not config.use_stooq_fallback:
        return results

    missing: list[str] = [symbol for symbol in symbols if symbol not in results]
    if not missing:
        return results

    LOGGER.info("yfinance got %d/%d tickers; trying stooq for %d missing", len(results), len(symbols), len(missing))
    stooq_aliases: dict[str, str] = _load_aliases(config.ticker_aliases_csv)
    stooq_recovered: int = 0

    for symbol in missing:
        stooq_sym: str = _stooq_symbol(symbol, stooq_aliases)
        attempts: int = 0
        while attempts < config.max_retries:
            try:
                ticker_df = _fetch_stooq_symbol(stooq_sym, start_date, end_date)
                if ticker_df is not None and not ticker_df.empty:
                    results[symbol] = ticker_df
                    stooq_recovered += 1
                break
            except Exception as exc:  # noqa: BLE001 - external API
                attempts += 1
                LOGGER.warning(
                    "stooq fetch failed for %s (%s/%s): %s",
                    symbol,
                    attempts,
                    config.max_retries,
                    exc,
                )
                time.sleep(config.retry_sleep * attempts)

    LOGGER.info("stooq recovered %d/%d missing tickers", stooq_recovered, len(missing))

    if not config.use_tiingo_fallback:
        return results

    missing_after_stooq: list[str] = [s for s in symbols if s not in results]
    if not missing_after_stooq:
        return results

    LOGGER.info(
        "stooq left %d missing tickers; trying tiingo", len(missing_after_stooq),
    )
    api_key: str = _resolve_tiingo_api_key()
    tiingo_recovered: int = 0

    for symbol in missing_after_stooq:
        attempts: int = 0
        while attempts < config.max_retries:
            try:
                ticker_df = _fetch_tiingo_symbol(
                    symbol, start_date, end_date, api_key,
                )
                if ticker_df is not None and not ticker_df.empty:
                    results[symbol] = ticker_df
                    tiingo_recovered += 1
                break
            except Exception as exc:  # noqa: BLE001 - external API
                attempts += 1
                LOGGER.warning(
                    "tiingo fetch failed for %s (%s/%s): %s",
                    symbol, attempts, config.max_retries, exc,
                )
                time.sleep(config.retry_sleep * attempts)

    LOGGER.info(
        "tiingo recovered %d/%d missing tickers", tiingo_recovered, len(missing_after_stooq),
    )
    return results


def _expand_membership_history_to_sessions(
    membership_history: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    tickers = membership_history["ticker"].tolist()
    start_dates = membership_history["start_date"].tolist()
    end_dates = membership_history["end_date"].tolist()
    for ticker_value, start_value, end_value in zip(tickers, start_dates, end_dates):
        start_date = _as_timestamp(cast(pd.Timestamp, start_value))
        end_date = _as_timestamp(cast(pd.Timestamp, end_value))
        ticker = str(ticker_value)
        active_sessions: pd.DatetimeIndex = sessions[
            (sessions >= start_date) & (sessions <= end_date)
        ]
        if len(active_sessions) == 0:
            continue
        rows.append(
            pd.DataFrame({
                "date": active_sessions,
                "ticker": ticker,
            }),
        )
    if not rows:
        return pd.DataFrame(columns=["date", "ticker"])
    expanded: pd.DataFrame = pd.concat(rows, ignore_index=True)
    return expanded.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)


def _load_fundamentals_history(path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path)
    df = df.rename(columns={"symbol": "ticker"})
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError(
            "Fundamentals history CSV must contain 'ticker'/'symbol' and 'date' columns",
        )
    history: pd.DataFrame = df.copy()
    history["ticker"] = history["ticker"].astype(str).str.strip()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["ticker", "date"]).reset_index(drop=True)
    return history


def _merge_point_in_time_fundamentals(
    data: pd.DataFrame,
    fundamentals_history_csv: Path,
) -> pd.DataFrame:
    fundamentals: pd.DataFrame = _load_fundamentals_history(fundamentals_history_csv)
    left = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    right = fundamentals.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged: pd.DataFrame = pd.merge_asof(
        left,
        right,
        on="date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


def _build_xtb_spec_provider(config: PipelineConfig) -> BrokerSpecProvider:
    xtb_specs_path = config.xtb_instrument_specs_json
    if xtb_specs_path is None:
        raise ValueError(
            "build_dataset requires xtb_instrument_specs_json when xtb_only=True.",
        )
    return build_default_spec_provider(
        path=xtb_specs_path,
        allow_defaults_if_missing=not config.require_xtb_snapshot,
        require_explicit_symbols=config.require_xtb_snapshot,
    )


def _filter_membership_history_to_xtb_universe(
    membership_history: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    if not config.xtb_only:
        return membership_history

    provider = _build_xtb_spec_provider(config)
    pipeline_end_date = _as_timestamp(config.end_date)
    filtered_rows: list[dict[str, object]] = []

    tickers = membership_history["ticker"].tolist()
    start_dates = membership_history["start_date"].tolist()
    end_dates = membership_history["end_date"].tolist()
    for ticker_value, start_value, end_value in zip(tickers, start_dates, end_dates):
        ticker = str(ticker_value)
        membership_start = _as_timestamp(cast(pd.Timestamp, start_value))
        membership_end = _as_timestamp(cast(pd.Timestamp, end_value))
        for spec in provider.find_explicit_specs(ticker, instrument_group="stock_cash"):
            if (
                config.xtb_max_spread_bps is not None
                and spec.spread_bps > config.xtb_max_spread_bps
            ):
                continue
            spec_start = _as_timestamp(spec.effective_from)
            spec_end = (
                pipeline_end_date
                if spec.effective_to is None
                else _as_timestamp(spec.effective_to)
            )
            clipped_start = max(membership_start, spec_start)
            clipped_end = min(membership_end, spec_end)
            if clipped_start > clipped_end:
                continue
            filtered_rows.append({
                "ticker": ticker,
                "start_date": clipped_start,
                "end_date": clipped_end,
            })

    if not filtered_rows:
        raise RuntimeError("No XTB-tradable constituents were found to fetch.")

    filtered_history = pd.DataFrame(filtered_rows).drop_duplicates().sort_values(
        ["ticker", "start_date", "end_date"],
    )
    LOGGER.info(
        "XTB tradable universe filter retained %d/%d tickers",
        filtered_history["ticker"].nunique(),
        membership_history["ticker"].nunique(),
    )
    return filtered_history.reset_index(drop=True)






def build_dataset(config: PipelineConfig) -> pd.DataFrame:
    start_date: dt.date = _parse_date(config.start_date)
    end_date: dt.date = _parse_date(config.end_date)

    if config.membership_history_csv is None and not config.allow_current_constituents_snapshot:
        raise ValueError(
            "build_dataset requires membership_history_csv for a point-in-time universe.",
        )
    if config.fundamentals_history_csv is None:
        raise ValueError(
            "build_dataset requires fundamentals_history_csv for point-in-time fundamentals.",
        )
    if config.xtb_only and config.xtb_instrument_specs_json is None:
        raise ValueError(
            "build_dataset requires xtb_instrument_specs_json when xtb_only=True.",
        )

    membership_history: pd.DataFrame = load_membership_history(config)
    membership_history = _filter_membership_history_to_xtb_universe(
        membership_history,
        dataclasses.replace(
            config,
            xtb_max_spread_bps=(
                config.xtb_max_spread_bps
                if config.xtb_max_spread_bps is not None
                else XTB_DEFAULT_MAX_SPREAD_BPS
            ),
        ),
    )
    symbols: list[str] = sorted(membership_history["ticker"].unique().tolist())
    if not symbols:
        raise RuntimeError("No constituents were found to fetch")

    LOGGER.info("Fetching %s tickers", len(symbols))

    price_map: dict[str, pd.DataFrame] = _fetch_prices(
        symbols, config.start_date, config.end_date, config
    )

    full_index: pd.DatetimeIndex = get_nyse_sessions(start_date, end_date)
    rows: list[pd.DataFrame] = []

    for symbol in symbols:
        ticker_df: pd.DataFrame | None = price_map.get(symbol)
        if ticker_df is None or ticker_df.empty:
            continue
        ticker_data: pd.DataFrame = ticker_df.reindex(full_index).copy()
        ticker_data["date"] = ticker_data.index
        ticker_data["ticker"] = symbol
        rows.append(ticker_data)

    if not rows:
        raise RuntimeError("No price data retrieved from providers")

    data: pd.DataFrame = pd.concat(rows, ignore_index=True)
    col_order: list[str] = ["date", "ticker"] + [
        c for c in OHLCV_OUTPUT_COLS if c in data.columns
    ]
    data = cast(pd.DataFrame, data[col_order])

    active_universe: pd.DataFrame = _expand_membership_history_to_sessions(
        membership_history,
        full_index,
    )
    data = active_universe.merge(data, on=["date", "ticker"], how="left")

    if config.fundamentals_history_csv is not None and config.fundamentals_history_csv.exists():
        data = _merge_point_in_time_fundamentals(
            data,
            config.fundamentals_history_csv,
        )
    elif config.fundamentals_history_csv is not None:
        LOGGER.warning(
            "Fundamentals history file not found at %s. Proceeding without fundamentals merge.",
            config.fundamentals_history_csv,
        )

    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    return data


def save_outputs(data: pd.DataFrame, config: PipelineConfig) -> dict[str, Path]:
    output_dir: Path = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path: Path = output_dir / OUTPUT_PARQUET.name
    csv_path: Path = output_dir / OUTPUT_SAMPLE_CSV.name
    data.to_parquet(parquet_path, index=False)
    sample = data.sample(frac=config.sample_frac, random_state=config.random_seed)
    sample.to_csv(csv_path, index=False)
    return {"parquet": parquet_path, "sample_csv": csv_path}


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    data: pd.DataFrame = build_dataset(config)
    return save_outputs(data, config)
