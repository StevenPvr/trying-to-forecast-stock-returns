from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from core.src.meta_model.data.constants import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    FRED_DAILY_SERIES,
    FRED_MONTHLY_SERIES,
    FRED_QUARTERLY_SERIES,
    FRED_RATE_LIMIT_SLEEP,
    FRED_WEEKLY_SERIES,
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

ALL_FRED_SERIES: tuple[str, ...] = (
    *FRED_DAILY_SERIES,
    *FRED_WEEKLY_SERIES,
    *FRED_MONTHLY_SERIES,
    *FRED_QUARTERLY_SERIES,
)


@dataclasses.dataclass(frozen=True)
class MacroConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    output_dir: Path = DATA_FETCHING_DIR
    sample_frac: float = SAMPLE_FRAC
    random_seed: int = RANDOM_SEED
    max_retries: int = MAX_RETRIES
    retry_sleep: float = RETRY_SLEEP
    fred_api_key: str | None = None


def _resolve_api_key(config: MacroConfig) -> str:
    if config.fred_api_key:
        return config.fred_api_key.strip()
    key: str | None = os.environ.get("FRED_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "FRED API key not found. Set FRED_API_KEY env var or pass fred_api_key to MacroConfig."
        )
    key = key.strip()
    if len(key) != 32:
        LOGGER.warning("FRED API key has %d chars (expected 32) -- check your .env file", len(key))
    return key


def _create_fred_client(api_key: str) -> object:
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "fredapi is not installed. Run: uv pip install fredapi"
        ) from exc
    return Fred(api_key=api_key)


def _fetch_single_series(
    client: object,
    series_id: str,
    start_date: str,
    end_date: str,
    max_retries: int,
    retry_sleep: float,
) -> pd.Series:  # type: ignore[type-arg]
    from fredapi import Fred

    fred: Fred = client  # type: ignore[assignment]
    attempts: int = 0
    while attempts < max_retries:
        try:
            result: pd.Series = fred.get_series(  # type: ignore[type-arg]
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            LOGGER.debug("Fetched %s: %d observations", series_id, len(result))
            return result
        except Exception as exc:  # noqa: BLE001 - external API
            attempts += 1
            LOGGER.warning(
                "FRED fetch %s failed (%s/%s): %s",
                series_id, attempts, max_retries, exc,
            )
            time.sleep(retry_sleep * attempts)
    raise RuntimeError(f"Failed to fetch {series_id} after {max_retries} retries")


_DAILY_SERIES_LAG_SESSIONS: int = 1
_WEEKLY_SERIES_LAG_SESSIONS: int = 5
_MONTHLY_SERIES_LAG_SESSIONS: int = 22
_QUARTERLY_SERIES_LAG_SESSIONS: int = 66


def _fetch_series_with_semaphore(
    client: object,
    series_id: str,
    start_date: str,
    end_date: str,
    max_retries: int,
    retry_sleep: float,
    rate_limit_sleep: float,
    semaphore: threading.Semaphore,
) -> tuple[str, pd.Series | None]:  # type: ignore[type-arg]
    """Fetch one FRED series under semaphore rate limiting."""
    with semaphore:
        try:
            result: pd.Series = _fetch_single_series(  # type: ignore[type-arg]
                client, series_id, start_date, end_date,
                max_retries, retry_sleep,
            )
            return (series_id, result)
        except RuntimeError:
            LOGGER.error("Skipping %s after all retries failed", series_id)
            return (series_id, None)
        finally:
            time.sleep(rate_limit_sleep)


def _fetch_all_series(
    client: object,
    series_ids: tuple[str, ...],
    start_date: str,
    end_date: str,
    max_retries: int,
    retry_sleep: float,
    rate_limit_sleep: float,
) -> dict[str, pd.Series]:  # type: ignore[type-arg]
    results: dict[str, pd.Series] = {}  # type: ignore[type-arg]
    worker_count = resolve_executor_worker_count(task_count=len(series_ids))
    semaphore: threading.Semaphore = threading.Semaphore(worker_count)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_series_with_semaphore,
                client, series_id, start_date, end_date,
                max_retries, retry_sleep, rate_limit_sleep, semaphore,
            ): series_id
            for series_id in series_ids
        }
        for future in as_completed(futures):
            series_id, series = future.result()
            if series is not None:
                results[series_id] = series
    return results


def _build_macro_dataframe(
    series_map: dict[str, pd.Series],  # type: ignore[type-arg]
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    sessions: pd.DatetimeIndex = get_nyse_sessions(start_date, end_date)
    df: pd.DataFrame = pd.DataFrame(index=sessions)
    df.index.name = "date"
    for series_id, series in series_map.items():
        col_name: str = series_id.lower()
        aligned_source: pd.Series = shift_series_to_session_availability(  # type: ignore[type-arg]
            series,
            sessions,
            _series_lag_sessions(series_id),
        )
        aligned: pd.Series = aligned_source.reindex(sessions).ffill()  # type: ignore[type-arg]
        df[col_name] = aligned
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _series_lag_sessions(series_id: str) -> int:
    if series_id in FRED_DAILY_SERIES:
        return _DAILY_SERIES_LAG_SESSIONS
    if series_id in FRED_WEEKLY_SERIES:
        return _WEEKLY_SERIES_LAG_SESSIONS
    if series_id in FRED_MONTHLY_SERIES:
        return _MONTHLY_SERIES_LAG_SESSIONS
    if series_id in FRED_QUARTERLY_SERIES:
        return _QUARTERLY_SERIES_LAG_SESSIONS
    return _DAILY_SERIES_LAG_SESSIONS


def build_macro_dataset(config: MacroConfig) -> pd.DataFrame:
    api_key: str = _resolve_api_key(config)
    client: object = _create_fred_client(api_key)
    series_map: dict[str, pd.Series] = _fetch_all_series(  # type: ignore[type-arg]
        client,
        ALL_FRED_SERIES,
        config.start_date,
        config.end_date,
        config.max_retries,
        config.retry_sleep,
        FRED_RATE_LIMIT_SLEEP,
    )
    if not series_map:
        raise RuntimeError("No macro series were fetched from FRED")
    data: pd.DataFrame = _build_macro_dataframe(
        series_map, config.start_date, config.end_date,
    )
    LOGGER.info("Built macro dataset: %d rows x %d columns", len(data), len(data.columns))
    return data
