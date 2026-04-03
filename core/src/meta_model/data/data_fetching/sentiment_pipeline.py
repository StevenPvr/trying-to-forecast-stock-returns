from __future__ import annotations

import dataclasses
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pandas as pd
import requests

from core.src.meta_model.data.constants import (
    AAII_SENTIMENT_URL,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    GPR_MONTHLY_URL,
    MAX_RETRIES,
    RANDOM_SEED,
    RETRY_SLEEP,
    SAMPLE_FRAC,
)
from core.src.meta_model.data.paths import DATA_FETCHING_DIR
from core.src.meta_model.data.trading_calendar import (
    get_nyse_sessions,
    shift_series_to_session_availability,
)
from core.src.meta_model.runtime_parallelism import resolve_executor_worker_count

LOGGER: logging.Logger = logging.getLogger(__name__)
_AAII_LAG_SESSIONS: int = 5
_GPR_LAG_SESSIONS: int = 22

_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "application/vnd.ms-excel,"
        "application/octet-stream,*/*"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclasses.dataclass(frozen=True)
class SentimentConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    output_dir: Path = DATA_FETCHING_DIR
    sample_frac: float = SAMPLE_FRAC
    random_seed: int = RANDOM_SEED
    max_retries: int = MAX_RETRIES
    retry_sleep: float = RETRY_SLEEP


def _download_bytes(url: str, max_retries: int, retry_sleep: float) -> bytes:
    """Download raw bytes from *url* with exponential-backoff retry."""
    attempts: int = 0
    while attempts < max_retries:
        try:
            response: requests.Response = requests.get(url, headers=_HTTP_HEADERS, timeout=60)
            response.raise_for_status()
            LOGGER.debug("Downloaded %d bytes from %s", len(response.content), url)
            return response.content
        except Exception as exc:  # noqa: BLE001 - external HTTP
            attempts += 1
            LOGGER.warning(
                "Download %s failed (%s/%s): %s",
                url, attempts, max_retries, exc,
            )
            time.sleep(retry_sleep * attempts)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries")


def _fetch_aaii_sentiment(config: SentimentConfig) -> pd.DataFrame:
    """Download and parse the AAII Investor Sentiment Survey Excel file."""
    try:
        raw: bytes = _download_bytes(
            AAII_SENTIMENT_URL, config.max_retries, config.retry_sleep,
        )
        if raw[:15].lstrip().startswith(b"<"):
            LOGGER.warning("AAII returned HTML instead of Excel (bot protection)")
            return pd.DataFrame()
        # Auto-detect header row: scan first 10 rows for 'Date' + 'Bullish'
        probe: pd.DataFrame = pd.read_excel(io.BytesIO(raw), header=None, nrows=10)
        header_row: int = 0
        for idx in range(len(probe)):
            row_vals: list[str] = [str(v).strip() for v in probe.iloc[idx].tolist()]
            if "Date" in row_vals and "Bullish" in row_vals:
                header_row = idx
                break
        df: pd.DataFrame = pd.read_excel(io.BytesIO(raw), header=header_row)
        return _parse_aaii_dataframe(df, config.start_date, config.end_date)
    except Exception as exc:  # noqa: BLE001 - resilient pipeline
        LOGGER.warning("AAII sentiment fetch failed: %s", exc)
        return pd.DataFrame()


def _parse_aaii_dataframe(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Normalise raw AAII Excel DataFrame to standard schema."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower: str = str(col).strip().lower()
        if col_lower == "date":
            rename_map[col] = "date"
        elif col_lower == "bullish":
            rename_map[col] = "aaii_bullish"
        elif col_lower == "neutral":
            rename_map[col] = "aaii_neutral"
        elif col_lower == "bearish":
            rename_map[col] = "aaii_bearish"
    df = df.rename(columns=rename_map)
    required: list[str] = ["date", "aaii_bullish", "aaii_neutral", "aaii_bearish"]
    missing: list[str] = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AAII Excel missing columns: {missing}")
    df = cast(pd.DataFrame, df[required].copy())
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start_date:end_date]
    LOGGER.info("Parsed AAII sentiment: %d rows", len(df))
    return df


def _fetch_gpr_index(config: SentimentConfig) -> pd.DataFrame:
    """Download and parse the Geopolitical Risk Index Excel file."""
    try:
        raw: bytes = _download_bytes(
            GPR_MONTHLY_URL, config.max_retries, config.retry_sleep,
        )
        df: pd.DataFrame = pd.read_excel(io.BytesIO(raw))
        return _parse_gpr_dataframe(df, config.start_date, config.end_date)
    except Exception as exc:  # noqa: BLE001 - resilient pipeline
        LOGGER.warning("GPR index fetch failed: %s", exc)
        return pd.DataFrame()


def _parse_gpr_dataframe(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Normalise raw GPR Excel DataFrame to standard schema."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_lower: str = str(col).strip().lower()
        if col_lower in ("month", "date"):
            rename_map[col] = "date"
        elif col_lower in ("gpr", "gprd"):
            rename_map[col] = "gpr_index"
        elif col_lower == "gpr_act":
            rename_map[col] = "gpr_act"
        elif col_lower == "gpr_threat":
            rename_map[col] = "gpr_threat"
    df = df.rename(columns=rename_map)
    if "date" not in df.columns or "gpr_index" not in df.columns:
        raise ValueError("GPR Excel missing 'date' or 'gpr_index' column")
    keep: list[str] = ["date", "gpr_index"]
    for optional in ("gpr_act", "gpr_threat"):
        if optional in df.columns:
            keep.append(optional)
    df = cast(pd.DataFrame, df[keep].copy())
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start_date:end_date]
    LOGGER.info("Parsed GPR index: %d rows", len(df))
    return df


def _build_sentiment_dataframe(
    aaii_df: pd.DataFrame,
    gpr_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Align AAII and GPR data to NYSE sessions with conservative availability lags."""
    sessions: pd.DatetimeIndex = get_nyse_sessions(start_date, end_date)
    result: pd.DataFrame = pd.DataFrame(index=sessions)
    result.index.name = "date"
    if not aaii_df.empty:
        for col in aaii_df.columns:
            shifted: pd.Series = shift_series_to_session_availability(  # type: ignore[type-arg]
                cast(pd.Series, aaii_df[col]),
                sessions,
                _AAII_LAG_SESSIONS,
            )
            result[col] = shifted.reindex(sessions).ffill()
    if not gpr_df.empty:
        for col in gpr_df.columns:
            shifted = shift_series_to_session_availability(  # type: ignore[type-arg]
                cast(pd.Series, gpr_df[col]),
                sessions,
                _GPR_LAG_SESSIONS,
            )
            result[col] = shifted.reindex(sessions).ffill()
    result = result.reset_index()
    result["date"] = pd.to_datetime(result["date"])
    return result


def build_sentiment_dataset(config: SentimentConfig) -> pd.DataFrame:
    """Fetch AAII and GPR data in parallel, merge onto a business-day index."""
    with ThreadPoolExecutor(max_workers=resolve_executor_worker_count(task_count=2)) as executor:
        aaii_future = executor.submit(_fetch_aaii_sentiment, config)
        gpr_future = executor.submit(_fetch_gpr_index, config)
        aaii_df: pd.DataFrame = aaii_future.result()
        gpr_df: pd.DataFrame = gpr_future.result()

    if aaii_df.empty and gpr_df.empty:
        raise RuntimeError("Both sentiment sources returned empty data")
    data: pd.DataFrame = _build_sentiment_dataframe(
        aaii_df, gpr_df, config.start_date, config.end_date,
    )
    LOGGER.info(
        "Built sentiment dataset: %d rows x %d columns",
        len(data), len(data.columns),
    )
    return data
